"""Replay a support workload against a running LatticeMemory proxy.

This is the live-HTTP companion to ``examples/proxy_replay_demo.py``. It targets
an already-running proxy, optionally seeds ``/v1/cache`` through the admin API,
replays evaluation/adversarial prompts through ``/v1/chat/completions``, fetches
``/v1/analytics``, and writes JSON/Markdown/HTML proof artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from latticememory.proof_pack import _response_body, load_support_dataset_jsonl


def run_live_replay_demo(
    *,
    dataset_path: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    model: str = "demo-model",
    admin_key: str | None = None,
    seed_cache: bool = True,
    limit: int | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    output_html: str | Path | None = None,
) -> dict[str, Any]:
    dataset = load_support_dataset_jsonl(dataset_path)
    base_url = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["X-Lattice-Admin-Key"] = admin_key

    if seed_cache:
        for row in dataset["cache_seed"]:
            _request_json(
                f"{base_url}/v1/cache",
                method="POST",
                headers=headers,
                payload={
                    "prompt": row["prompt"],
                    "value": _response_body(row["canonical_answer"], model="demo-seed"),
                    "metadata": {"intent_id": row["intent_id"], "seed_id": row["id"]},
                },
            )

    rows = _select_replay_rows(dataset, limit=limit)
    hits = misses = false_positives = adversarial_false_positives = 0
    latencies: list[float] = []
    started = time.perf_counter()
    for row in rows:
        row_start = time.perf_counter()
        response_body, response_headers = _request_json_with_headers(
            f"{base_url}/v1/chat/completions",
            method="POST",
            headers=headers,
            payload={"model": model, "messages": [{"role": "user", "content": row["prompt"]}]},
        )
        latencies.append((time.perf_counter() - row_start) * 1000.0)
        if response_headers.get("x-lattice-cache", "").upper() == "HIT":
            hits += 1
        else:
            misses += 1
        if _content_from_chat_body(response_body) != row["canonical_answer"]:
            false_positives += 1
            if row["is_adversarial"]:
                adversarial_false_positives += 1

    analytics = _request_json(f"{base_url}/v1/analytics", method="GET", headers=headers, payload=None)
    adversarial_total = sum(1 for row in rows if row["is_adversarial"])
    total = len(rows)
    summary = {
        "base_url": base_url,
        "dataset_path": str(dataset_path),
        "seed_cache": seed_cache,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
        "upstream_calls": misses,
        "hit_rate": hits / total if total else 0.0,
        "upstream_call_rate": misses / total if total else 0.0,
        "false_positive_rate": false_positives / total if total else 0.0,
        "adversarial_false_positive_rate": (
            adversarial_false_positives / adversarial_total if adversarial_total else 0.0
        ),
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "elapsed_s": time.perf_counter() - started,
        "analytics": analytics,
    }

    if output_json is not None:
        Path(output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if output_md is not None:
        Path(output_md).write_text(_render_markdown_report(summary), encoding="utf-8")
    if output_html is not None:
        Path(output_html).write_text(_render_html_report(summary), encoding="utf-8")
    return summary


def _select_replay_rows(dataset: dict[str, list[dict[str, Any]]], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = dataset["evaluation"] + dataset["adversarial"]
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def _content_from_chat_body(body: dict[str, Any]) -> str:
    try:
        return str(body["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    body, _ = _request_json_with_headers(url, method=method, headers=headers, payload=payload)
    return body


def _request_json_with_headers(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _render_markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LatticeMemory Live Proxy Replay Demo",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total requests | {summary['total_requests']} |",
            f"| Hit rate | {summary['hit_rate']:.4f} |",
            f"| Upstream call rate | {summary['upstream_call_rate']:.4f} |",
            f"| False-positive rate | {summary['false_positive_rate']:.4f} |",
            f"| Adversarial false-positive rate | {summary['adversarial_false_positive_rate']:.4f} |",
            f"| Avg latency ms | {summary['latency_ms_avg']:.3f} |",
            "",
            "## Analytics",
            "",
            "```json",
            json.dumps(summary.get("analytics", {}), indent=2, sort_keys=True),
            "```",
            "",
            "This live replay validates proxy wiring and analytics. It does not prove RedisVL superiority,",
            "general RAG superiority, or safety of raw PQ hits without validation.",
            "",
        ]
    )


def _render_html_report(summary: dict[str, Any]) -> str:
    analytics = json.dumps(summary.get("analytics", {}), indent=2, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LatticeMemory Live Proxy Replay Demo</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #172033; }}
    table {{ border-collapse: collapse; min-width: 520px; }}
    th, td {{ border-bottom: 1px solid #d8dde8; padding: 8px 10px; text-align: left; }}
    td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
    pre {{ background: #f6f8fb; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>LatticeMemory Live Proxy Replay Demo</h1>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total requests</td><td>{summary['total_requests']}</td></tr>
    <tr><td>Hit rate</td><td>{summary['hit_rate']:.4f}</td></tr>
    <tr><td>Upstream call rate</td><td>{summary['upstream_call_rate']:.4f}</td></tr>
    <tr><td>False-positive rate</td><td>{summary['false_positive_rate']:.4f}</td></tr>
    <tr><td>Adversarial false-positive rate</td><td>{summary['adversarial_false_positive_rate']:.4f}</td></tr>
    <tr><td>Avg latency ms</td><td>{summary['latency_ms_avg']:.3f}</td></tr>
  </table>
  <h2>Analytics</h2>
  <pre>{html.escape(analytics)}</pre>
  <p>This live replay validates proxy wiring and analytics. It does not prove RedisVL superiority,
  general RAG superiority, or safety of raw PQ hits without validation.</p>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="demo-model")
    parser.add_argument("--admin-key", default=None)
    parser.add_argument("--no-seed-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-html", default=None)
    args = parser.parse_args()
    summary = run_live_replay_demo(
        dataset_path=args.dataset_jsonl,
        base_url=args.base_url,
        model=args.model,
        admin_key=args.admin_key,
        seed_cache=not args.no_seed_cache,
        limit=args.limit,
        output_json=args.output_json,
        output_md=args.output_md,
        output_html=args.output_html,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
