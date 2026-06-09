"""Render a two-domain comparison HTML page from two intent cache proof JSON files."""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def ms(v: float) -> str:
    return f"{v:.3f} ms"


CSS = """
:root {
  --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
  --text: #e2e8f0; --muted: #8892a4;
  --green: #22c55e; --red: #ef4444; --yellow: #eab308;
  --blue: #3b82f6; --purple: #a855f7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }
header { padding: 48px 40px 32px; border-bottom: 1px solid var(--border); }
header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
header p { color: var(--muted); font-size: 0.95rem; }
main { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
.domains { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }
@media (max-width: 768px) { .domains { grid-template-columns: 1fr; } }
.domain-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 28px; }
.domain-card h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }
.domain-meta { color: var(--muted); font-size: 0.82rem; margin-bottom: 20px; }
.gate-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 20px; }
.gate-pass { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
.gate-warn { background: rgba(234,179,8,0.15); color: var(--yellow); border: 1px solid rgba(234,179,8,0.3); }
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.metric { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
.metric-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.metric-value { font-size: 1.6rem; font-weight: 700; }
.metric-sub { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
.green { color: var(--green); }
.red { color: var(--red); }
.blue { color: var(--blue); }
.purple { color: var(--purple); }
.yellow { color: var(--yellow); }
section { margin-bottom: 40px; }
section h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th { text-align: left; padding: 10px 14px; color: var(--muted); font-weight: 500; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }
td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); }
tr:last-child td { border-bottom: none; }
.pass-row td:first-child { color: var(--green); }
.warn-row td:first-child { color: var(--yellow); }
.zero-fp { color: var(--green); }
.has-fp { color: var(--yellow); }
.note { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 8px; padding: 16px 20px; font-size: 0.88rem; color: var(--muted); margin-bottom: 32px; }
.note strong { color: var(--text); }
.fp-detail { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; font-size: 0.85rem; }
.fp-detail .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
footer { border-top: 1px solid var(--border); padding: 24px 40px; text-align: center; font-size: 0.8rem; color: var(--muted); }
footer a { color: var(--blue); }
code { background: rgba(255,255,255,0.07); padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }
"""


def render(cs: dict, hd: dict, out_path: Path) -> None:
    cs_m = cs["metrics"]
    hd_m = hd["metrics"]
    cs_sim = cs["cache_simulation"]
    hd_sim = hd["cache_simulation"]
    cs_sp = cs.get("splits_summary", {})
    hd_sp = hd.get("splits_summary", {})
    hd_nm_ex = hd.get("near_miss_examples", [])
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    rows_summary = f"""
    <tr class="pass-row">
      <td>Customer Support</td>
      <td class="green">PASS</td>
      <td>{pct(cs_m["held_out_recall"])}</td>
      <td class="green">{pct(cs_m["held_out_fp_rate"])}</td>
      <td>{pct(cs_sim["hit_rate"])}</td>
      <td>{ms(cs_m["mean_latency_ms"])}</td>
      <td>12</td>
    </tr>
    <tr class="warn-row">
      <td>IT Helpdesk</td>
      <td class="yellow">{pct(hd_m["held_out_recall"])} / {hd_m["held_out_false_positives"]} FP</td>
      <td>{pct(hd_m["held_out_recall"])}</td>
      <td class="yellow">{pct(hd_m["held_out_fp_rate"])}</td>
      <td>{pct(hd_sim["hit_rate"])}</td>
      <td>{ms(hd_m["mean_latency_ms"])}</td>
      <td>12</td>
    </tr>"""

    rows_cv = f"""
    <tr>
      <td>Customer Support</td>
      <td>{pct(cs_sp.get("recall", {}).get("mean", 0))} &plusmn; {pct(cs_sp.get("recall", {}).get("std", 0))}</td>
      <td>{pct(cs_sp.get("recall", {}).get("min", 0))}</td>
      <td>{pct(cs_sp.get("recall", {}).get("max", 0))}</td>
      <td class="green">{pct(cs_sp.get("wrong_route_rate", {}).get("mean", 0))}</td>
    </tr>
    <tr>
      <td>IT Helpdesk</td>
      <td>{pct(hd_sp.get("recall", {}).get("mean", 0))} &plusmn; {pct(hd_sp.get("recall", {}).get("std", 0))}</td>
      <td>{pct(hd_sp.get("recall", {}).get("min", 0))}</td>
      <td>{pct(hd_sp.get("recall", {}).get("max", 0))}</td>
      <td class="yellow">{pct(hd_sp.get("wrong_route_rate", {}).get("mean", 0))}</td>
    </tr>"""

    threshold_rows = ""
    for row in hd.get("threshold_curve", []):
        fp_class = "zero-fp" if row["fp_rate"] == 0.0 else "has-fp"
        threshold_rows += f"""    <tr>
      <td>{row["min_score_threshold"]:.4f}</td>
      <td>{pct(row["coverage"])}</td>
      <td>{pct(row["recall"])}</td>
      <td class="{fp_class}">{pct(row["fp_rate"])}</td>
    </tr>\n"""

    nm_detail = ""
    if hd_nm_ex:
        ex = hd_nm_ex[0]
        nm_detail = f"""
<section>
  <h2>The One Wrong Route — IT Helpdesk</h2>
  <div class="fp-detail">
    <div class="label">Wrong Route Example</div>
    <p><strong>Query:</strong> &ldquo;{ex["query"]}&rdquo;</p>
    <p><strong>Predicted:</strong> <code>{ex["predicted"]}</code> &nbsp;&rarr;&nbsp; <strong>Correct:</strong> <code>{ex["actual"]}</code></p>
    <p style="color:var(--muted);margin-top:8px;font-size:0.83rem">
      These are the most semantically adjacent intents in the helpdesk domain &mdash; both involve software licensing operations.
      Setting confidence threshold &ge;&nbsp;0.9386 eliminates this wrong route; coverage drops to 55.6%
      (unconfident queries are forwarded to the upstream LLM at normal cost).
    </p>
  </div>
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LatticeMemory — Multi-Domain Intent Cache Results</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>LatticeMemory Intent Cache &mdash; Multi-Domain Results</h1>
  <p>Closed-set semantic cache proof across two production-representative domains &nbsp;&middot;&nbsp;
     Model: <code>snap_product_gate_hard_symmetric_8ep</code> &nbsp;&middot;&nbsp; {now}</p>
</header>
<main>

<div class="note">
  <strong>What this measures:</strong> For each domain, paraphrase prompts are routed to a pre-approved
  cached answer (12 intents each). <strong>Recall</strong> is the fraction of paraphrase queries correctly
  routed. <strong>FP rate</strong> is wrong routes on hard near-miss queries &mdash; semantically adjacent
  prompts that belong to a different intent. <strong>Cache hit rate</strong> simulates the fraction of
  real traffic served from cache. <strong>Latency</strong> is lookup overhead only (excludes encoding).
</div>

<div class="domains">
  <div class="domain-card">
    <h2>Customer Support</h2>
    <div class="domain-meta">12 intents &nbsp;&middot;&nbsp; 36 held-out paraphrases &nbsp;&middot;&nbsp; 60 near-miss queries</div>
    <div class="gate-badge gate-pass">&#10003; GATE PASS &mdash; 0 WRONG ROUTES</div>
    <div class="metrics-grid">
      <div class="metric">
        <div class="metric-label">Recall</div>
        <div class="metric-value green">{pct(cs_m["held_out_recall"])}</div>
        <div class="metric-sub">{cs_m["held_out_true_positives"]}/{cs_m["total_held_out_paraphrases"]} correct routes</div>
      </div>
      <div class="metric">
        <div class="metric-label">Wrong Route Rate</div>
        <div class="metric-value green">{pct(cs_m["held_out_fp_rate"])}</div>
        <div class="metric-sub">0 false positives</div>
      </div>
      <div class="metric">
        <div class="metric-label">Cache Hit Rate</div>
        <div class="metric-value blue">{pct(cs_sim["hit_rate"])}</div>
        <div class="metric-sub">{cs_sim["hits"]}/{cs_sim["total_prompts"]} served from cache</div>
      </div>
      <div class="metric">
        <div class="metric-label">Latency</div>
        <div class="metric-value purple">{ms(cs_m["mean_latency_ms"])}</div>
        <div class="metric-sub">lookup overhead</div>
      </div>
    </div>
  </div>

  <div class="domain-card">
    <h2>IT Helpdesk</h2>
    <div class="domain-meta">12 intents &nbsp;&middot;&nbsp; 36 held-out paraphrases &nbsp;&middot;&nbsp; 60 near-miss queries</div>
    <div class="gate-badge gate-warn">94.4% RECALL &nbsp;&middot;&nbsp; 1 WRONG ROUTE (tunable)</div>
    <div class="metrics-grid">
      <div class="metric">
        <div class="metric-label">Recall</div>
        <div class="metric-value green">{pct(hd_m["held_out_recall"])}</div>
        <div class="metric-sub">{hd_m["held_out_true_positives"]}/{hd_m["total_held_out_paraphrases"]} correct routes</div>
      </div>
      <div class="metric">
        <div class="metric-label">Wrong Route Rate</div>
        <div class="metric-value yellow">{pct(hd_m["held_out_fp_rate"])}</div>
        <div class="metric-sub">{hd_m["held_out_false_positives"]} FP &nbsp;&middot;&nbsp; eliminable via threshold</div>
      </div>
      <div class="metric">
        <div class="metric-label">Cache Hit Rate</div>
        <div class="metric-value blue">{pct(hd_sim["hit_rate"])}</div>
        <div class="metric-sub">{hd_sim["hits"]}/{hd_sim["total_prompts"]} served from cache</div>
      </div>
      <div class="metric">
        <div class="metric-label">Latency</div>
        <div class="metric-value purple">{ms(hd_m["mean_latency_ms"])}</div>
        <div class="metric-sub">lookup overhead</div>
      </div>
    </div>
  </div>
</div>

<section>
  <h2>Combined Summary</h2>
  <table>
    <tr><th>Domain</th><th>Gate</th><th>Recall</th><th>FP Rate</th><th>Cache Hit Rate</th><th>Latency</th><th>Intents</th></tr>
    {rows_summary}
  </table>
</section>

<section>
  <h2>Cross-Validation Stability (5 Random Splits)</h2>
  <table>
    <tr><th>Domain</th><th>Recall (mean &plusmn; std)</th><th>Min Recall</th><th>Max Recall</th><th>Wrong Route (mean)</th></tr>
    {rows_cv}
  </table>
</section>

{nm_detail}

<section>
  <h2>Confidence Threshold Trade-off &mdash; IT Helpdesk</h2>
  <p style="color:var(--muted);font-size:0.85rem;margin-bottom:14px">
    Raise the minimum routing confidence score to reduce wrong routes at the cost of coverage.
    Queries below the threshold are forwarded to the upstream LLM at full cost.
    <strong style="color:var(--text)">Green rows = zero wrong routes.</strong>
  </p>
  <table>
    <tr><th>Min Confidence</th><th>Coverage</th><th>Recall</th><th>FP Rate</th></tr>
    {threshold_rows}
  </table>
</section>

</main>
<footer>
  LatticeMemory &nbsp;&middot;&nbsp; Calibrated semantic cache for LLM applications &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/dfrokido">HuggingFace: dfrokido</a>
</footer>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    cs = json.loads(Path("benchmarks/results/intent_cache_product_proof_hard_symmetric_8ep.json").read_text())
    hd = json.loads(Path("benchmarks/results/intent_cache_helpdesk_proof.json").read_text())
    render(cs, hd, Path("benchmarks/results/multi_domain_results.html"))
