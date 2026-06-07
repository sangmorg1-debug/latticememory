"""Render a HammingRouter proof benchmark JSON artifact as a static HTML page."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_results_page(report: dict[str, Any], *, title: str = "LatticeMemory HammingRouter Demo") -> str:
    metrics = report.get("metrics", {})
    cache = report.get("cache_simulation", {})
    distributions = report.get("distributions", {})
    para = distributions.get("paraphrase", {})
    near = distributions.get("near_miss", {})
    threshold_curve = report.get("threshold_curve", [])
    sample_rows = threshold_curve[:: max(1, len(threshold_curve) // 12)] if threshold_curve else []
    if threshold_curve and threshold_curve[-1] not in sample_rows:
        sample_rows.append(threshold_curve[-1])

    cards = [
        ("Calibrated Threshold", report.get("calibrated_threshold")),
        ("Held-out Recall", _pct(metrics.get("held_out_recall"))),
        ("Held-out FP Rate", _pct(metrics.get("held_out_fp_rate"))),
        ("Cache Hit Simulation", _pct(cache.get("hit_rate"))),
        ("Mean Lookup Latency", f"{metrics.get('mean_latency_ms', 0):.2f} ms"),
        ("Calibration FP Budget", _pct(report.get("fp_budget"))),
    ]

    card_html = "\n".join(
        f"<section class='metric'><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></section>"
        for label, value in cards
    )
    curve_html = "\n".join(
        "<tr>"
        f"<td>{row.get('threshold')}</td>"
        f"<td>{_pct(row.get('recall'))}</td>"
        f"<td>{_pct(row.get('fp_rate'))}</td>"
        "</tr>"
        for row in sample_rows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --ink: #14171f;
      --muted: #5f6877;
      --panel: #ffffff;
      --line: #d9dee8;
      --green: #1f8a4c;
      --red: #b3261e;
      --blue: #2358a5;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 32px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0 0 24px;
      max-width: 760px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric strong {{
      font-size: 24px;
    }}
    .verdict {{
      border-left: 5px solid var(--blue);
      background: var(--panel);
      padding: 16px;
      margin-top: 20px;
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      background: #eef2f7;
      color: #303847;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      background: #eef2f7;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">
      Product proof for a calibrated semantic cache: reuse answers only when the E8 Hamming distance is inside
      a measured threshold, while tracking recall, false positives, hit-rate simulation, and latency.
    </p>

    <div class="metrics">{card_html}</div>

    <section class="verdict">
      <strong>Product read:</strong>
      At threshold <code>{html.escape(str(report.get("calibrated_threshold")))}</code>, this benchmark measured
      <strong>{html.escape(_pct(metrics.get("held_out_recall")))}</strong> held-out paraphrase recall and
      <strong>{html.escape(_pct(metrics.get("held_out_fp_rate")))}</strong> held-out false-positive rate.
    </section>

    <h2>Benchmark Context</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Model</td><td>{html.escape(str(report.get("model")))}</td></tr>
      <tr><td>Embedding dimension</td><td>{html.escape(str(report.get("d_model")))}</td></tr>
      <tr><td>Created at</td><td>{html.escape(str(report.get("created_at")))}</td></tr>
      <tr><td>Calibration data hash</td><td><code>{html.escape(str(report.get("calibration_data_sha256")))}</code></td></tr>
      <tr><td>Cache simulation prompts</td><td>{html.escape(str(cache.get("total_prompts", 0)))}</td></tr>
    </table>

    <h2>Distance Distributions</h2>
    <div class="grid">
      <table>
        <tr><th colspan="2">Paraphrase pairs</th></tr>
        {"".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_num(v))}</td></tr>" for k, v in para.items())}
      </table>
      <table>
        <tr><th colspan="2">Near-miss pairs</th></tr>
        {"".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_num(v))}</td></tr>" for k, v in near.items())}
      </table>
    </div>

    <h2>Threshold Curve Sample</h2>
    <table>
      <tr><th>Threshold</th><th>Recall</th><th>FP Rate</th></tr>
      {curve_html}
    </table>

    <p class="small">
      Generated from <code>{html.escape(str(report.get("artifact_type")))}</code> artifact version
      <code>{html.escape(str(report.get("artifact_version")))}</code>.
    </p>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render HammingRouter benchmark results as HTML")
    parser.add_argument("--input", required=True, help="Path to benchmark_hamming_router JSON output")
    parser.add_argument("--output", required=True, help="Path to write HTML results page")
    parser.add_argument("--title", default="LatticeMemory HammingRouter Product Demo")
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
