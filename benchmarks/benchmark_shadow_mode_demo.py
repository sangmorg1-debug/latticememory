"""Shadow-mode cache replay benchmark.

Replays a prompt stream through LatticeLLMProxy in shadow mode.
The router detects would-be Hamming-NN cache hits but does NOT serve them;
it logs what WOULD have happened.

Output: JSON report + HTML page showing:
  - Would-be hit rate (shadow hits)
  - Would-be wrong-route rate (estimated from calibration)
  - Latency overhead per query
  - Distribution of shadow Hamming distances
  - Estimated cost savings

Usage:
  python -m benchmarks.benchmark_shadow_mode_demo \
    --model benchmarks/results/snap_product_gate_hard_symmetric_8ep/best_snap_encoder \
    --prompts-responses benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json \
    --calibration benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json \
    --output benchmarks/results/shadow_mode_demo.json

To run without a trained checkpoint, pass --model synthetic to use a fast
synthetic encoder (results will not be meaningful, but the pipeline is validated).
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic encoder (for testing without a real model)
# ---------------------------------------------------------------------------

class _SyntheticEncoder:
    """Deterministic unit-sphere embedding based on text hash — test use only."""

    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def get_sentence_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, texts: list[str], normalize_embeddings: bool = True, batch_size: int = 64):
        import hashlib
        import numpy as np

        out = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.d_model).astype(np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                vec = vec / max(norm, 1e-8)
            out.append(vec)
        return np.stack(out)


# ---------------------------------------------------------------------------
# Core shadow-mode runner
# ---------------------------------------------------------------------------

def run_shadow_demo(
    *,
    model: str,
    prompts_responses: list[dict[str, str]],
    calibration_data: dict[str, Any],
    hamming_threshold: int = 70,
    cost_per_query_usd: float = 0.00025,
    output_path: str | Path,
    latency_probe_n: int = 200,
) -> dict[str, Any]:
    import numpy as np

    # Load encoder
    if model == "synthetic":
        encoder = _SyntheticEncoder(d_model=128)
        d_model = 128
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model)
        d_model = int(encoder.get_sentence_embedding_dimension())

    # Build E8 key table for calibration set (canonical prompts → cache entries)
    # We simulate: first time a prompt is seen, it is a cache miss and gets added.
    # Subsequent semantically-similar prompts trigger a shadow hit.
    from latticememory.hamming_router import HammingRouter

    router = HammingRouter(encoder=encoder, d_model=d_model, threshold=hamming_threshold)

    # Warm the router with calibration set canonical texts
    cal_paraphrases = calibration_data.get("paraphrases", [])
    canonical_texts: list[str] = []
    for pair in cal_paraphrases:
        if pair and pair[0] not in canonical_texts:
            canonical_texts.append(pair[0])

    response_by_canonical: dict[str, str] = {}
    for row in prompts_responses:
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        if prompt and response and prompt not in response_by_canonical:
            response_by_canonical[prompt] = response

    # Latency measurement: time the full encode + nearest-neighbour lookup
    all_prompts = [row["prompt"] for row in prompts_responses]

    # Pre-encode all prompts at once for efficiency
    all_texts = sorted(set(all_prompts) | set(canonical_texts))
    all_embs = encoder.encode(all_texts, normalize_embeddings=True, batch_size=16)
    emb_by_text = {t: e for t, e in zip(all_texts, all_embs)}

    def _text_to_key(text: str) -> bytes:
        """Return raw E8 key bytes for a text using pre-computed embedding."""
        import torch as _torch
        emb = emb_by_text[text]
        key_bytes = router._lattice._quantize_to_indices(
            _torch.tensor(emb, dtype=_torch.float32)
        )
        return bytes(np.frombuffer(key_bytes, dtype=np.uint8))

    # Warm the router with canonical texts first (simulates a pre-loaded cache)
    for text in canonical_texts:
        key = _text_to_key(text)
        router.add_from_key(key, value=text)

    # Latency probe on the lookup path only (key already computed)
    latency_samples: list[float] = []
    if all_prompts and canonical_texts:
        probe_text = all_prompts[0]
        if probe_text in emb_by_text:
            probe_key = _text_to_key(probe_text)
            for _ in range(latency_probe_n):
                t0 = time.perf_counter()
                router.lookup_key(probe_key)
                latency_samples.append(time.perf_counter() - t0)
    mean_latency_ms = (sum(latency_samples) / len(latency_samples)) * 1000 if latency_samples else 0.0
    p95_latency_ms = float(sorted(latency_samples)[int(len(latency_samples) * 0.95)]) * 1000 if latency_samples else 0.0

    # Stream replay — use pre-computed keys
    shadow_hits = 0
    true_misses = 0
    hamming_distances: list[int] = []
    shadow_examples: list[dict[str, str]] = []
    miss_examples: list[dict[str, str]] = []

    seen_keys: dict[bytes, str] = {}  # key bytes -> canonical text (exact match)

    for row in prompts_responses:
        prompt = row.get("prompt", "")
        if not prompt or prompt not in emb_by_text:
            continue

        key = _text_to_key(prompt)

        if key in seen_keys:
            # Exact E8 key match — exact cache hit
            shadow_hits += 1
            hamming_distances.append(0)
            if len(shadow_examples) < 8:
                shadow_examples.append({
                    "query": prompt,
                    "matched_canonical": seen_keys[key],
                    "hamming": 0,
                    "type": "exact_e8",
                })
        else:
            # Try Hamming-NN router
            match = router.lookup_key(key, threshold=hamming_threshold)
            if match is not None:
                shadow_hits += 1
                hamming_distances.append(match.hamming_distance)
                if len(shadow_examples) < 8:
                    shadow_examples.append({
                        "query": prompt,
                        "matched_canonical": str(match.value),
                        "hamming": match.hamming_distance,
                        "type": "hamming_nn",
                    })
            else:
                true_misses += 1
                seen_keys[key] = prompt
                router.add_from_key(key, value=prompt)
                if len(miss_examples) < 5:
                    miss_examples.append({"query": prompt, "type": "first_seen"})

    total = shadow_hits + true_misses
    shadow_hit_rate = shadow_hits / total if total else 0.0
    would_be_savings_usd = shadow_hits * cost_per_query_usd

    # Distance histogram (buckets of 10)
    dist_histogram: dict[str, int] = {}
    for d in hamming_distances:
        bucket = f"{(d // 10) * 10}-{(d // 10) * 10 + 9}"
        dist_histogram[bucket] = dist_histogram.get(bucket, 0) + 1

    # Storage footprint
    n_keys = len(seen_keys) + len(canonical_texts)
    key_bytes = n_keys * (d_model // 8)

    report = {
        "artifact_type": "latticememory_shadow_mode_demo",
        "artifact_version": 1,
        "model": model,
        "d_model": d_model,
        "hamming_threshold": hamming_threshold,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "stream": {
            "total_prompts": total,
            "shadow_hits": shadow_hits,
            "true_misses": true_misses,
            "shadow_hit_rate": round(shadow_hit_rate, 4),
            "cost_per_query_usd": cost_per_query_usd,
            "would_be_savings_usd": round(would_be_savings_usd, 6),
            "would_be_savings_pct": round(shadow_hit_rate * 100, 1),
        },
        "latency": {
            "mean_lookup_ms": round(mean_latency_ms, 4),
            "p95_lookup_ms": round(p95_latency_ms, 4),
            "probe_n": latency_probe_n,
        },
        "hamming_distribution": {
            "n": len(hamming_distances),
            "mean": round(sum(hamming_distances) / len(hamming_distances), 2) if hamming_distances else 0.0,
            "exact_hits": hamming_distances.count(0),
            "histogram": dist_histogram,
        },
        "index": {
            "n_keys": n_keys,
            "key_bytes_per_entry": d_model // 8,
            "total_key_bytes": key_bytes,
        },
        "examples": {
            "shadow_hits": shadow_examples,
            "misses": miss_examples,
        },
        "calibration": {
            "n_canonical_texts": len(canonical_texts),
            "n_calibration_pairs": len(cal_paraphrases),
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

def render_shadow_demo_html(report: dict[str, Any]) -> str:
    import html as _html

    stream = report.get("stream", {})
    latency = report.get("latency", {})
    index = report.get("index", {})
    dist = report.get("hamming_distribution", {})
    examples = report.get("examples", {})
    shadow_hits_ex = examples.get("shadow_hits", [])
    miss_ex = examples.get("misses", [])
    hit_rate = stream.get("shadow_hit_rate", 0.0)
    savings_usd = stream.get("would_be_savings_usd", 0.0)
    model = _html.escape(str(report.get("model", "unknown")))

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _bytes_str(v: int) -> str:
        if v < 1024:
            return f"{v} B"
        elif v < 1024 * 1024:
            return f"{v / 1024:.1f} KB"
        return f"{v / 1024 / 1024:.2f} MB"

    shadow_rows = "".join(
        f"<tr>"
        f"<td>{_html.escape(ex.get('query',''))}</td>"
        f"<td>{_html.escape(ex.get('matched_canonical',''))}</td>"
        f"<td>{ex.get('hamming', '-')}</td>"
        f"<td>{_html.escape(ex.get('type',''))}</td>"
        f"</tr>"
        for ex in shadow_hits_ex
    )
    miss_rows = "".join(
        f"<tr><td>{_html.escape(ex.get('query',''))}</td><td>{_html.escape(ex.get('type',''))}</td></tr>"
        for ex in miss_ex
    )

    hist = dist.get("histogram", {})
    hist_bars = ""
    if hist:
        max_count = max(hist.values()) if hist else 1
        for bucket, count in sorted(hist.items()):
            bar_pct = int(count / max_count * 100)
            hist_bars += (
                f"<div class='bar-row'>"
                f"<span class='bar-label'>{_html.escape(bucket)}</span>"
                f"<div class='bar-track'><div class='bar-fill' style='width:{bar_pct}%'></div></div>"
                f"<span class='bar-count'>{count}</span>"
                f"</div>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LatticeMemory Shadow-Mode Cache Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --ink: #e6edf3;
      --muted: #8b949e;
      --green: #3fb950;
      --blue: #58a6ff;
      --purple: #bc8cff;
      --yellow: #e3b341;
      --red: #f85149;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
      border-bottom: 1px solid var(--border);
      padding: 32px 40px 24px;
    }}
    header h1 {{
      margin: 0 0 4px;
      font-size: 28px;
      font-weight: 700;
      background: linear-gradient(90deg, var(--blue), var(--purple));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    header p {{ margin: 0; color: var(--muted); font-size: 14px; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 40px 80px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .kpi {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 18px 16px;
      position: relative;
      overflow: hidden;
    }}
    .kpi::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--blue), var(--purple));
    }}
    .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
    .kpi-value {{ font-size: 28px; font-weight: 700; color: var(--green); }}
    .kpi-value.blue {{ color: var(--blue); }}
    .kpi-value.yellow {{ color: var(--yellow); }}
    .kpi-value.purple {{ color: var(--purple); }}
    section {{ margin-bottom: 40px; }}
    h2 {{
      font-size: 18px; font-weight: 600;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px; margin: 0 0 16px;
      color: var(--blue);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{ background: rgba(88,166,255,0.07); color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: none; }}
    .bar-row {{
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 6px; font-size: 13px;
    }}
    .bar-label {{ width: 60px; color: var(--muted); flex-shrink: 0; }}
    .bar-track {{
      flex: 1; height: 14px;
      background: rgba(255,255,255,0.05);
      border-radius: 7px; overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--blue), var(--purple));
      border-radius: 7px;
    }}
    .bar-count {{ width: 32px; text-align: right; color: var(--muted); }}
    .verdict {{
      background: rgba(63,185,80,0.08);
      border: 1px solid rgba(63,185,80,0.3);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 32px;
    }}
    .verdict strong {{ color: var(--green); font-size: 16px; }}
    .verdict p {{ margin: 8px 0 0; color: var(--muted); font-size: 14px; }}
    code {{
      background: rgba(255,255,255,0.06);
      padding: 2px 6px; border-radius: 4px;
      font-size: 12px; color: var(--blue);
    }}
    .footnote {{ color: var(--muted); font-size: 12px; margin-top: 40px; }}
  </style>
</head>
<body>
  <header>
    <h1>LatticeMemory Shadow-Mode Cache Demo</h1>
    <p>Hamming-NN router in shadow mode — detects would-be hits without serving them</p>
  </header>
  <main>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Shadow Hit Rate</div>
        <div class="kpi-value">{_pct(hit_rate)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Total Queries</div>
        <div class="kpi-value blue">{stream.get('total_prompts', 0)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Would-be Savings</div>
        <div class="kpi-value yellow">${savings_usd:.4f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Mean Lookup Latency</div>
        <div class="kpi-value purple">{latency.get('mean_lookup_ms', 0):.3f} ms</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">P95 Latency</div>
        <div class="kpi-value purple">{latency.get('p95_lookup_ms', 0):.3f} ms</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Index Size</div>
        <div class="kpi-value blue">{_bytes_str(index.get('total_key_bytes', 0))}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Shadow Hits</div>
        <div class="kpi-value">{stream.get('shadow_hits', 0)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">True Misses</div>
        <div class="kpi-value blue">{stream.get('true_misses', 0)}</div>
      </div>
    </div>

    <div class="verdict">
      <strong>Shadow-mode read:</strong>
      <p>
        On this prompt stream, <strong>{_pct(hit_rate)}</strong> of queries would have been served from
        the Hamming router cache (threshold = {report.get('hamming_threshold')}, shadow mode — not served).
        Estimated savings: <strong>${savings_usd:.4f}</strong> at ${stream.get('cost_per_query_usd', 0):.5f}/query.
        Mean lookup overhead: <strong>{latency.get('mean_lookup_ms', 0):.3f} ms</strong>.
        <br><br>
        Model: <code>{model}</code> &nbsp;|&nbsp;
        Embedding dim: <code>{report.get('d_model', '?')}</code> &nbsp;|&nbsp;
        Key bytes/entry: <code>{index.get('key_bytes_per_entry', '?')}</code>
      </p>
    </div>

    <section>
      <h2>Shadow Hit Examples</h2>
      <table>
        <tr><th>Query</th><th>Would-match Canonical</th><th>Hamming Dist</th><th>Match Type</th></tr>
        {shadow_rows if shadow_rows else '<tr><td colspan="4" style="color:var(--muted)">No shadow hits recorded</td></tr>'}
      </table>
    </section>

    <section>
      <h2>Hamming Distance Distribution (Shadow Hits)</h2>
      {hist_bars if hist_bars else '<p style="color:var(--muted);font-size:13px">No Hamming distances recorded.</p>'}
    </section>

    <section>
      <h2>True Miss Examples (First-Seen Prompts)</h2>
      <table>
        <tr><th>Query</th><th>Type</th></tr>
        {miss_rows if miss_rows else '<tr><td colspan="2" style="color:var(--muted)">No misses recorded</td></tr>'}
      </table>
    </section>

    <section>
      <h2>Index & Latency Detail</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Hamming threshold</td><td>{report.get('hamming_threshold')}</td></tr>
        <tr><td>Cached keys</td><td>{index.get('n_keys', 0)}</td></tr>
        <tr><td>Bytes per key</td><td>{index.get('key_bytes_per_entry', 0)}</td></tr>
        <tr><td>Total index bytes</td><td>{_bytes_str(index.get('total_key_bytes', 0))}</td></tr>
        <tr><td>Mean lookup latency</td><td>{latency.get('mean_lookup_ms', 0):.4f} ms</td></tr>
        <tr><td>P95 lookup latency</td><td>{latency.get('p95_lookup_ms', 0):.4f} ms</td></tr>
        <tr><td>Latency probe n</td><td>{latency.get('probe_n', 0)}</td></tr>
        <tr><td>Exact E8 hits</td><td>{dist.get('exact_hits', 0)}</td></tr>
        <tr><td>Mean Hamming distance (hits)</td><td>{dist.get('mean', 0):.2f}</td></tr>
      </table>
    </section>

    <p class="footnote">
      Shadow mode: the router identifies would-be hits but does not serve them.
      Activate <code>hamming_router_mode="serve"</code> after calibration to enable live serving.
      Generated {_html.escape(str(report.get('created_at', '')))} from
      <code>latticememory_shadow_mode_demo</code> v{report.get('artifact_version', 1)}.
    </p>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LatticeLLMProxy shadow-mode cache replay benchmark"
    )
    parser.add_argument("--model", required=True, help="Encoder path or 'synthetic'")
    parser.add_argument(
        "--prompts-responses",
        default="benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json",
    )
    parser.add_argument(
        "--calibration",
        default="benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json",
    )
    parser.add_argument("--hamming-threshold", type=int, default=70)
    parser.add_argument("--cost-per-query-usd", type=float, default=0.00025)
    parser.add_argument("--output", default="benchmarks/results/shadow_mode_demo.json")
    args = parser.parse_args()

    prompts_responses = json.loads(Path(args.prompts_responses).read_text(encoding="utf-8"))
    calibration_data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    report = run_shadow_demo(
        model=args.model,
        prompts_responses=prompts_responses,
        calibration_data=calibration_data,
        hamming_threshold=args.hamming_threshold,
        cost_per_query_usd=args.cost_per_query_usd,
        output_path=args.output,
    )

    # Also write HTML
    html_path = Path(args.output).with_suffix(".html")
    html_path.write_text(render_shadow_demo_html(report), encoding="utf-8")
    print(f"JSON: {args.output}")
    print(f"HTML: {html_path}")
    print(json.dumps({
        "shadow_hit_rate": report["stream"]["shadow_hit_rate"],
        "would_be_savings_usd": report["stream"]["would_be_savings_usd"],
        "mean_lookup_ms": report["latency"]["mean_lookup_ms"],
        "total_key_bytes": report["index"]["total_key_bytes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
