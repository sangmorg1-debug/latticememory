"""Render a LatticeMemory proof benchmark JSON artifact as a premium static HTML page.

Supports:
- latticememory_intent_cache_proof_results (v1 and v2)
- latticememory_hamming_proof_results
- latticememory_multidomain_proof_results
- latticememory_shadow_mode_demo
"""
from __future__ import annotations

import argparse
import html as _html
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(value: float | int | None, *, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{decimals}f}%"


def _num(value: Any, *, decimals: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _bytes_str(value: Any) -> str:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "n/a"
    if v < 1024:
        return f"{v:,} B"
    elif v < 1024 * 1024:
        return f"{v / 1024:.1f} KB"
    return f"{v / 1024 / 1024:.2f} MB"


def _esc(v: Any) -> str:
    return _html.escape(str(v))


# ---------------------------------------------------------------------------
# CSS — shared premium dark-mode design system
# ---------------------------------------------------------------------------

_SHARED_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2230;
    --border: #30363d;
    --ink: #e6edf3;
    --muted: #8b949e;
    --green: #3fb950;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --yellow: #e3b341;
    --red: #f85149;
    --orange: #f0883e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.55;
  }
  header {
    background: linear-gradient(135deg, #151b2b 0%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 48px 28px;
  }
  header h1 {
    margin: 0 0 6px;
    font-size: 30px;
    font-weight: 700;
    background: linear-gradient(90deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  header .subtitle { margin: 0; color: var(--muted); font-size: 14px; max-width: 800px; }
  header .meta { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
  main { max-width: 1200px; margin: 0 auto; padding: 36px 48px 80px; }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 40px;
  }
  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 16px 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .kpi:hover { border-color: var(--blue); }
  .kpi::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--blue), var(--purple));
  }
  .kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: var(--green); }
  .kpi-value.blue { color: var(--blue); }
  .kpi-value.yellow { color: var(--yellow); }
  .kpi-value.purple { color: var(--purple); }
  .kpi-value.red { color: var(--red); }
  .kpi-value.muted { color: var(--muted); }
  section { margin-bottom: 44px; }
  h2 {
    font-size: 17px; font-weight: 600;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin: 0 0 18px;
    color: var(--blue);
  }
  h3 { font-size: 14px; color: var(--muted); margin: 0 0 10px; font-weight: 600; }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    font-size: 13px;
    margin-bottom: 16px;
  }
  th, td { padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left; }
  th { background: rgba(88,166,255,0.06); color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .verdict {
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 36px;
    border: 1px solid;
  }
  .verdict.pass { background: rgba(63,185,80,0.07); border-color: rgba(63,185,80,0.3); }
  .verdict.fail { background: rgba(248,81,73,0.07); border-color: rgba(248,81,73,0.3); }
  .verdict strong { font-size: 15px; }
  .verdict p { margin: 8px 0 0; color: var(--muted); font-size: 13px; line-height: 1.6; }
  code {
    background: rgba(255,255,255,0.06);
    padding: 2px 6px; border-radius: 4px;
    font-size: 12px; color: var(--blue); font-family: monospace;
  }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } main { padding: 20px; } header { padding: 24px; } }
  .example-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 13px;
  }
  .example-block .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .example-block .q { color: var(--ink); margin-bottom: 4px; }
  .example-block .expected { color: var(--green); }
  .example-block .wrong { color: var(--red); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; font-size: 13px; }
  .bar-label { width: 64px; color: var(--muted); flex-shrink: 0; font-size: 12px; }
  .bar-track { flex: 1; height: 12px; background: rgba(255,255,255,0.04); border-radius: 6px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--blue), var(--purple)); border-radius: 6px; }
  .bar-count { width: 36px; text-align: right; color: var(--muted); font-size: 12px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .tag.pass { background: rgba(63,185,80,0.15); color: var(--green); }
  .tag.fail { background: rgba(248,81,73,0.15); color: var(--red); }
  .footnote { color: var(--muted); font-size: 12px; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 16px; }
  .splits-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;
  }
  .splits-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px;
  }
  .splits-card .sc-label { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .splits-card .sc-mean { font-size: 22px; font-weight: 700; color: var(--blue); }
  .splits-card .sc-std { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .splits-card .sc-values { font-size: 11px; color: var(--muted); margin-top: 6px; font-family: monospace; }
  .threshold-chart { position: relative; height: 160px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .threshold-chart svg { width: 100%; height: 100%; }
  .domain-row td:first-child { font-weight: 600; color: var(--ink); }
"""

# ---------------------------------------------------------------------------
# SVG threshold curve chart
# ---------------------------------------------------------------------------

def _render_threshold_svg(curve: list[dict[str, Any]], width: int = 800, height: int = 160) -> str:
    """Render recall and FP rate curves as inline SVG."""
    if not curve:
        return ""
    n = len(curve)
    pad_l, pad_r, pad_t, pad_b = 40, 20, 12, 28
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    def _x(i: int) -> float:
        return pad_l + i / max(n - 1, 1) * w

    def _y(v: float) -> float:
        return pad_t + (1.0 - v) * h

    # Recall line (blue)
    recall_pts = " ".join(
        f"{_x(i):.1f},{_y(float(row.get('recall', 0))):.1f}" for i, row in enumerate(curve)
    )
    # FP rate line (red)
    fp_pts = " ".join(
        f"{_x(i):.1f},{_y(float(row.get('fp_rate', 0))):.1f}" for i, row in enumerate(curve)
    )

    # X-axis labels (first, mid, last)
    def _x_label(i: int, row: dict) -> str:
        x = _x(i)
        val = row.get("min_score_threshold", row.get("threshold", ""))
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        return f'<text x="{x:.0f}" y="{height - 4}" fill="#8b949e" font-size="9" text-anchor="middle">{val_str}</text>'

    label_indices = [0, n // 2, n - 1]
    x_labels = "\n".join(_x_label(i, curve[i]) for i in label_indices if i < n)

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="display:block">\n'
        f'  <!-- grid lines -->\n'
        f'  <line x1="{pad_l}" y1="{_y(0.0):.1f}" x2="{pad_l + w}" y2="{_y(0.0):.1f}" stroke="#30363d" stroke-width="1"/>\n'
        f'  <line x1="{pad_l}" y1="{_y(0.5):.1f}" x2="{pad_l + w}" y2="{_y(0.5):.1f}" stroke="#30363d" stroke-width="1" stroke-dasharray="3,3"/>\n'
        f'  <line x1="{pad_l}" y1="{_y(1.0):.1f}" x2="{pad_l + w}" y2="{_y(1.0):.1f}" stroke="#30363d" stroke-width="1"/>\n'
        f'  <text x="{pad_l - 4}" y="{_y(1.0):.0f}" fill="#8b949e" font-size="9" text-anchor="end">100%</text>\n'
        f'  <text x="{pad_l - 4}" y="{_y(0.5):.0f}" fill="#8b949e" font-size="9" text-anchor="end">50%</text>\n'
        f'  <text x="{pad_l - 4}" y="{_y(0.0):.0f}" fill="#8b949e" font-size="9" text-anchor="end">0%</text>\n'
        f'  <!-- recall line -->\n'
        f'  <polyline points="{recall_pts}" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linejoin="round"/>\n'
        f'  <!-- fp rate line -->\n'
        f'  <polyline points="{fp_pts}" fill="none" stroke="#f85149" stroke-width="2" stroke-linejoin="round" stroke-dasharray="6,3"/>\n'
        f'  <!-- x labels -->\n'
        f'{x_labels}\n'
        f'  <text x="{pad_l}" y="{_y(1.0) - 4:.0f}" fill="#58a6ff" font-size="9">Recall</text>\n'
        f'  <text x="{pad_l + 48}" y="{_y(1.0) - 4:.0f}" fill="#f85149" font-size="9">- - FP Rate</text>\n'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Intent cache report renderer (v1 + v2)
# ---------------------------------------------------------------------------

def _render_intent_cache(report: dict[str, Any], title: str) -> str:
    metrics = report.get("metrics", {})
    cache = report.get("cache_simulation", {})
    gate = report.get("product_gate", {})
    index = report.get("index", {})
    splits = report.get("splits_summary")
    domains = report.get("domains_summary")
    para_examples = report.get("paraphrase_examples", [])
    nm_examples = report.get("near_miss_examples", [])
    threshold_curve = report.get("threshold_curve", [])
    passed = bool(gate.get("passed"))
    verdict_cls = "pass" if passed else "fail"
    recall = metrics.get("held_out_recall", 0.0)
    fp_rate = metrics.get("held_out_fp_rate", 0.0)
    hit_rate = cache.get("hit_rate", 0.0)
    latency = metrics.get("mean_latency_ms", 0.0)

    kpis = [
        ("Product Gate", "PASS" if passed else "FAIL", "green" if passed else "red"),
        ("Held-out Recall", _pct(recall), "green" if recall >= 0.80 else "yellow"),
        ("FP Rate", _pct(fp_rate), "green" if fp_rate == 0.0 else "red"),
        ("Cache Sim Hit Rate", _pct(hit_rate), "blue"),
        ("Mean Latency", f"{latency:.3f} ms", "purple"),
        ("Index Size", _bytes_str(index.get("stored_key_bytes")), "blue"),
        ("Intents", str(index.get("n_intents", "?")), "muted"),
        ("Paraphrase n", str(metrics.get("total_held_out_paraphrases", "?")), "muted"),
    ]
    kpi_html = "\n".join(
        f'<div class="kpi"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value {cls}">{_esc(val)}</div></div>'
        for label, val, cls in kpis
    )

    # Multi-split section
    splits_html = ""
    if splits:
        rec = splits.get("recall", {})
        wr = splits.get("wrong_route_rate", {})
        hr = splits.get("cache_hit_rate", {})
        splits_html = f"""
<section>
  <h2>Cross-Validation Stability ({splits.get('n_splits', '?')} Splits)</h2>
  <div class="splits-grid">
    <div class="splits-card">
      <div class="sc-label">Recall (mean ± std)</div>
      <div class="sc-mean">{_pct(rec.get('mean'))}</div>
      <div class="sc-std">± {_pct(rec.get('std'))} &nbsp;|&nbsp; min {_pct(rec.get('min'))} &nbsp;max {_pct(rec.get('max'))}</div>
      <div class="sc-values">splits: {', '.join(_pct(v) for v in rec.get('values', []))}</div>
    </div>
    <div class="splits-card">
      <div class="sc-label">Wrong Route Rate (mean ± std)</div>
      <div class="sc-mean" style="color:var(--{'green' if float(wr.get('mean', 0)) == 0.0 else 'red'})">{_pct(wr.get('mean'))}</div>
      <div class="sc-std">± {_pct(wr.get('std'))}</div>
      <div class="sc-values">splits: {', '.join(_pct(v) for v in wr.get('values', []))}</div>
    </div>
    <div class="splits-card">
      <div class="sc-label">Cache Hit Rate (mean ± std)</div>
      <div class="sc-mean" style="color:var(--blue)">{_pct(hr.get('mean'))}</div>
      <div class="sc-std">± {_pct(hr.get('std'))}</div>
      <div class="sc-values">splits: {', '.join(_pct(v) for v in hr.get('values', []))}</div>
    </div>
  </div>
</section>"""

    # Threshold curve
    curve_html = ""
    if threshold_curve:
        svg = _render_threshold_svg(threshold_curve)
        _sample = threshold_curve[::max(1, len(threshold_curve) // 12)]
        curve_table_rows = "".join(
            "<tr><td>" + _esc(row.get("min_score_threshold", row.get("threshold", "?"))) + "</td>"
            + "<td>" + _pct(row.get("recall")) + "</td>"
            + "<td>" + _pct(row.get("fp_rate")) + "</td>"
            + "<td>" + _pct(row.get("coverage")) + "</td></tr>"
            for row in _sample
        )
        curve_html = f"""
<section>
  <h2>Routing Confidence vs Recall / FP Rate</h2>
  <div class="threshold-chart">{svg}</div>
  <p style="color:var(--muted);font-size:12px;margin-top:8px">
    X-axis: minimum centroid score threshold. Blue: paraphrase recall. Red (dashed): FP rate.
    The left edge (no threshold) gives maximum recall; the right edge gives maximum precision.
  </p>
  <table>
    <tr><th>Min Score</th><th>Recall</th><th>FP Rate</th><th>Coverage</th></tr>
    {curve_table_rows}
  </table>
</section>"""

    # Examples
    def _render_example(ex: dict, kind: str) -> str:
        result = ex.get("result", "")
        tag_cls = "pass" if result == "correct" else ("fail" if result == "wrong_route" else "")
        tag_text = "\u2713 correct" if result == "correct" else ("\u2717 wrong" if result == "wrong_route" else result)
        predicted_html = (
            '<div class="wrong">Predicted: ' + _esc(ex.get("predicted", "")) + "</div>"
            if result == "wrong_route" else ""
        )
        tag_html = (
            '<span class="tag ' + tag_cls + '">' + tag_text + "</span>"
            if tag_cls else ""
        )
        return (
            '<div class="example-block">'
            + '<div class="label">' + _esc(kind) + "</div>"
            + '<div class="q">Query: <strong>' + _esc(ex.get("query", "")) + "</strong></div>"
            + '<div class="expected">Expected intent: ' + _esc(ex.get("expected", ex.get("actual", ""))) + "</div>"
            + predicted_html
            + tag_html
            + "</div>"
        )

    examples_html = ""
    if para_examples or nm_examples:
        para_ex_html = "\n".join(_render_example(ex, "Paraphrase hit") for ex in para_examples[:5])
        nm_ex_html = "\n".join(_render_example(ex, "Near-miss FP") for ex in nm_examples[:5])
        examples_html = f"""
<section>
  <h2>Example Routing Results</h2>
  <div class="two-col">
    <div>
      <h3>Paraphrase Hits (correctly routed)</h3>
      {para_ex_html if para_ex_html else '<p style="color:var(--muted);font-size:13px">None recorded.</p>'}
    </div>
    <div>
      <h3>Near-Miss False Positives</h3>
      {nm_ex_html if nm_ex_html else '<p style="color:var(--green);font-size:13px">✓ Zero false positive routes recorded.</p>'}
    </div>
  </div>
</section>"""

    context_rows = [
        ("Model", str(report.get("model", "?"))),
        ("Embedding dimension", str(report.get("d_model", "?"))),
        ("Domain", str(report.get("domain", "?"))),
        ("Created at", str(report.get("created_at", "?"))),
        ("Product gate target", _pct(gate.get("recall_target"))),
        ("Exact snapping required", str(gate.get("exact_snap_required", False))),
        ("Fragmentation metric role", str(gate.get("fragmentation_metric_role", "research_exact_snap"))),
        ("Cache sim total prompts", str(cache.get("total_prompts", "?"))),
        ("Held-out paraphrases", str(metrics.get("total_held_out_paraphrases", "?"))),
        ("Held-out near-miss queries", str(metrics.get("total_near_miss_queries", "?"))),
        ("Stored centroid bytes", _bytes_str(index.get("stored_key_bytes"))),
    ]
    context_html = "\n".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>' for k, v in context_rows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <p class="subtitle">
      Closed-set semantic cache proof: route prompts to approved answer intents
      while tracking held-out recall, wrong-route rate, hit-rate simulation, and latency.
    </p>
    <p class="meta">
      Model: <code>{_esc(str(report.get('model', '?')))}</code> &nbsp;|&nbsp;
      Domain: <code>{_esc(str(report.get('domain', '?')))}</code> &nbsp;|&nbsp;
      {_esc(str(report.get('created_at', '')))}
    </p>
  </header>
  <main>
    <div class="kpi-grid">{kpi_html}</div>

    <div class="verdict {verdict_cls}">
      <strong>{"✓ PASS" if passed else "✗ FAIL"} — {_esc(str(gate.get("name", "intent_recall_at_zero_wrong_routes")))}</strong>
      <p>
        With nearest-centroid routing, this benchmark measured
        <strong>{_pct(recall)}</strong> held-out paraphrase recall and
        <strong>{_pct(fp_rate)}</strong> wrong-route rate on hard near-miss queries.
        Cache simulation: <strong>{_pct(hit_rate)}</strong> hit rate.
        Mean routing latency: <strong>{latency:.3f} ms</strong>.
        Exact same-cell snapping is <code>{_esc(str(gate.get("fragmentation_metric_role", "research_exact_snap")))}</code>
        and is not required for this gate.
      </p>
    </div>

    {splits_html}
    {curve_html}
    {examples_html}

    <section>
      <h2>Benchmark Context</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        {context_html}
      </table>
    </section>

    <p class="footnote">
      Generated from <code>{_esc(str(report.get('artifact_type', 'latticememory_intent_cache_proof_results')))}</code>
      artifact version <code>{_esc(str(report.get('artifact_version', 1)))}</code>.
      Exact snapping is a research stretch goal; calibrated Hamming routing is the product gate.
    </p>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Hamming proof report renderer
# ---------------------------------------------------------------------------

def _render_hamming_proof(report: dict[str, Any], title: str) -> str:
    metrics = report.get("metrics", {})
    cache = report.get("cache_simulation", {})
    gate = report.get("product_gate", {})
    index = report.get("index", {})
    para = report.get("distributions", {}).get("paraphrase", {})
    near = report.get("distributions", {}).get("near_miss", {})
    threshold_curve = report.get("threshold_curve", [])
    passed = bool(gate.get("passed"))
    verdict_cls = "pass" if passed else "fail"
    recall = metrics.get("held_out_recall", metrics.get("held_out_recall_at_fp_budget", 0.0)) or 0.0
    fp_rate = metrics.get("held_out_fp_rate", 0.0)

    kpis = [
        ("Product Gate", "PASS" if passed else "FAIL", "green" if passed else "red"),
        ("Held-out Recall", _pct(recall), "green" if float(recall) >= 0.80 else "yellow"),
        ("FP Rate @ Budget", _pct(fp_rate), "green" if float(fp_rate) == 0.0 else "red"),
        ("Threshold", str(report.get("calibrated_threshold", "?")), "blue"),
        ("Cache Hit Rate", _pct(cache.get("hit_rate")), "blue"),
        ("Mean Latency", f"{metrics.get('mean_latency_ms', 0):.3f} ms", "purple"),
        ("Key Compression", f"{index.get('compression_vs_float32_keys_only', 1.0):.1f}x", "yellow"),
        ("Stored Keys", _bytes_str(index.get("stored_key_bytes")), "blue"),
    ]
    kpi_html = "\n".join(
        f'<div class="kpi"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value {cls}">{_esc(val)}</div></div>'
        for label, val, cls in kpis
    )

    curve_html = ""
    if threshold_curve:
        svg = _render_threshold_svg(threshold_curve)
        curve_html = f"""
<section>
  <h2>Threshold Curve (Recall vs FP Rate)</h2>
  <div class="threshold-chart">{svg}</div>
  <table>
    <tr><th>Threshold</th><th>Recall</th><th>FP Rate</th></tr>
    {"".join(f'<tr><td>{_esc(row.get("threshold","?"))}</td><td>{_pct(row.get("recall"))}</td><td>{_pct(row.get("fp_rate"))}</td></tr>' for row in threshold_curve[::max(1, len(threshold_curve)//12)])}
  </table>
</section>"""

    dist_rows_para = "\n".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(_num(v))}</td></tr>' for k, v in para.items()
    )
    dist_rows_near = "\n".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(_num(v))}</td></tr>' for k, v in near.items()
    )

    context_rows = [
        ("Model", str(report.get("model", "?"))),
        ("Embedding dimension", str(report.get("d_model", "?"))),
        ("Created at", str(report.get("created_at", "?"))),
        ("Calibration data SHA256", str(report.get("calibration_data_sha256", "?"))[:16] + "…"),
        ("Product gate target", _pct(gate.get("recall_target"))),
        ("FP budget", _pct(report.get("fp_budget"))),
        ("Exact snapping required", str(gate.get("exact_snap_required", False))),
        ("Key bytes per entry", str(index.get("key_bytes_per_entry", "?"))),
        ("Float32 equivalent bytes", _bytes_str(index.get("float32_embedding_bytes_equivalent"))),
    ]
    context_html = "\n".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>' for k, v in context_rows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <p class="subtitle">
      Calibrated E8 Hamming routing: reuse answers only when the Hamming distance is inside a measured threshold,
      while tracking recall, false positives, hit-rate simulation, and latency.
    </p>
    <p class="meta">
      Model: <code>{_esc(str(report.get('model', '?')))}</code> &nbsp;|&nbsp;
      Threshold: <code>{_esc(str(report.get('calibrated_threshold', '?')))}</code> &nbsp;|&nbsp;
      {_esc(str(report.get('created_at', '')))}
    </p>
  </header>
  <main>
    <div class="kpi-grid">{kpi_html}</div>

    <div class="verdict {verdict_cls}">
      <strong>{"✓ PASS" if passed else "✗ FAIL"} — {_esc(str(gate.get("name", "recall_at_FP=0")))}</strong>
      <p>
        At threshold <strong>{_esc(str(report.get('calibrated_threshold', '?')))}</strong>,
        this benchmark measured <strong>{_pct(recall)}</strong> held-out recall and
        <strong>{_pct(fp_rate)}</strong> FP rate at budget.
        Cache simulation: <strong>{_pct(cache.get('hit_rate'))}</strong> hit rate.
      </p>
    </div>

    {curve_html}

    <section>
      <h2>Distance Distributions</h2>
      <div class="two-col">
        <table>
          <tr><th colspan="2">Paraphrase pairs</th></tr>
          {dist_rows_para}
        </table>
        <table>
          <tr><th colspan="2">Near-miss pairs</th></tr>
          {dist_rows_near}
        </table>
      </div>
    </section>

    <section>
      <h2>Benchmark Context</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        {context_html}
      </table>
    </section>

    <p class="footnote">
      Generated from <code>{_esc(str(report.get('artifact_type', '')))}</code>
      artifact version <code>{_esc(str(report.get('artifact_version', 1)))}</code>.
    </p>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Multi-domain report renderer
# ---------------------------------------------------------------------------

def _render_multidomain(report: dict[str, Any], title: str) -> str:
    agg = report.get("aggregate_metrics", {})
    gate = report.get("product_gate", {})
    domains = report.get("domains", [])
    passed = bool(gate.get("passed"))
    verdict_cls = "pass" if passed else "fail"

    kpis = [
        ("Overall Gate", "PASS" if passed else "FAIL", "green" if passed else "red"),
        ("Mean Recall", _pct(agg.get("mean_held_out_recall")), "green"),
        ("Max FP Rate", _pct(agg.get("max_held_out_fp_rate")), "green" if float(agg.get("max_held_out_fp_rate", 1)) == 0.0 else "red"),
        ("Domains Tested", str(agg.get("n_domains", "?")), "blue"),
        ("All Domains Pass", "Yes" if gate.get("all_domains_pass") else "No", "green" if gate.get("all_domains_pass") else "red"),
    ]
    kpi_html = "\n".join(
        f'<div class="kpi"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value {cls}">{_esc(val)}</div></div>'
        for label, val, cls in kpis
    )
    domain_rows = "\n".join(
        f'<tr>'
        f'<td>{_esc(d.get("domain", "?"))}</td>'
        f'<td>{_pct(d.get("held_out_recall"))}</td>'
        f'<td>{_pct(d.get("held_out_fp_rate"))}</td>'
        f'<td>{_pct(d.get("cache_hit_rate"))}</td>'
        f'<td>{str(d.get("n_intents", "?"))}</td>'
        f'<td>{_pct(d.get("splits_recall_mean"))} ± {_pct(d.get("splits_recall_std"))}</td>'
        f'<td><span class="tag {"pass" if d.get("passed") else "fail"}">{"PASS" if d.get("passed") else "FAIL"}</span></td>'
        f'</tr>'
        for d in domains
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <p class="subtitle">
      Multi-domain intent cache proof: same encoder checkpoint evaluated across multiple domain datasets.
    </p>
    <p class="meta">Model: <code>{_esc(str(report.get('model', '?')))}</code> &nbsp;|&nbsp; {_esc(str(report.get('created_at', '')))} </p>
  </header>
  <main>
    <div class="kpi-grid">{kpi_html}</div>

    <div class="verdict {verdict_cls}">
      <strong>{"✓ PASS" if passed else "✗ FAIL"} across all domains</strong>
      <p>
        Mean held-out recall: <strong>{_pct(agg.get('mean_held_out_recall'))}</strong>.
        Max FP rate: <strong>{_pct(agg.get('max_held_out_fp_rate'))}</strong>.
        All {agg.get('n_domains', '?')} domains {"pass" if gate.get("all_domains_pass") else "do not all pass"} the product gate.
      </p>
    </div>

    <section>
      <h2>Per-Domain Results</h2>
      <table class="domain-row">
        <tr>
          <th>Domain</th><th>Recall</th><th>FP Rate</th>
          <th>Cache Hit Rate</th><th>Intents</th>
          <th>CV Recall (mean ± std)</th><th>Gate</th>
        </tr>
        {domain_rows}
      </table>
    </section>

    <p class="footnote">
      Generated from <code>{_esc(str(report.get('artifact_type', '')))}</code>
      artifact version <code>{_esc(str(report.get('artifact_version', 1)))}</code>.
    </p>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def render_results_page(
    report: dict[str, Any],
    *,
    title: str = "LatticeMemory Product Proof",
) -> str:
    artifact_type = str(report.get("artifact_type", ""))

    if artifact_type == "latticememory_multidomain_proof_results":
        return _render_multidomain(report, title)

    route_type = str(report.get("route_type", ""))
    if route_type == "closed_set_intent_centroid_cache" or artifact_type == "latticememory_intent_cache_proof_results":
        return _render_intent_cache(report, title)

    # Default: hamming proof
    return _render_hamming_proof(report, title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render LatticeMemory benchmark results as a premium HTML page"
    )
    parser.add_argument("--input", required=True, help="Path to benchmark JSON output")
    parser.add_argument("--output", required=True, help="Path to write HTML results page")
    parser.add_argument("--title", default="LatticeMemory Product Proof")
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    html_text = render_results_page(report, title=args.title)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"Wrote results page to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
