"""Replay a proof-pack support workload through the LatticeMemory proxy.

The proof-pack benchmark isolates cache policies. This demo exercises the
OpenAI-compatible proxy path directly with Redis-backed cache storage and the
general cache cosine validation gate enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latticememory.proof_pack import (
    _build_pq_cache,
    _content_from_body,
    _response_body,
    load_support_dataset_jsonl,
)
from latticememory.proxy import LatticeLLMProxy


class DemoUpstream:
    def __init__(self, answers_by_prompt: dict[str, str]) -> None:
        self.answers_by_prompt = answers_by_prompt
        self.calls = 0

    def __call__(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        del headers
        self.calls += 1
        prompt = _prompt_from_payload(payload)
        answer = self.answers_by_prompt.get(prompt, "A support specialist must review this request.")
        return _response_body(answer, model=payload.get("model", "demo-model"))


def run_replay_demo(
    *,
    dataset_path: str | Path,
    redis_url: str | None,
    redis_namespace: str = "lattice-demo",
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    limit: int | None = None,
    cache_cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    dataset = load_support_dataset_jsonl(dataset_path)
    cache, _ = _build_pq_cache(
        dataset,
        use_redis=bool(redis_url),
        redis_url=redis_url,
        redis_namespace=redis_namespace,
        flush_redis=True,
    )
    for row in dataset["cache_seed"]:
        cache.put(
            row["prompt"],
            value=_response_body(row["canonical_answer"], model="demo-seed"),
            metadata={"intent_id": row["intent_id"], "seed_id": row["id"]},
        )

    answers_by_prompt = {
        row["prompt"]: row["canonical_answer"]
        for rows in dataset.values()
        for row in rows
    }
    upstream = DemoUpstream(answers_by_prompt)
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="demo-key",
        semantic_cache=cache,
        upstream_client=upstream,
        cache_cosine_gate=True,
        cache_cosine_threshold=cache_cosine_threshold,
    )
    client = TestClient(proxy.create_app())

    rows = dataset["evaluation"] + dataset["adversarial"]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    adversarial_total = sum(1 for row in rows if row["is_adversarial"])

    false_positives = 0
    adversarial_false_positives = 0
    hits = 0
    misses = 0
    rejected = 0
    latencies: list[float] = []
    started = time.perf_counter()
    for row in rows:
        row_start = time.perf_counter()
        response = client.post(
            "/v1/chat/completions",
            json={"model": "demo-model", "messages": [{"role": "user", "content": row["prompt"]}]},
        )
        latencies.append((time.perf_counter() - row_start) * 1000.0)
        served = response.headers.get("x-lattice-cache", "MISS")
        path = response.headers.get("x-lattice-retrieval-path", "")
        if served == "HIT":
            hits += 1
        else:
            misses += 1
        if response.headers.get("x-lattice-cache-cosine-gate-rejected") == "true":
            rejected += 1
        content = _content_from_body(response.json())
        if content != row["canonical_answer"]:
            false_positives += 1
            if row["is_adversarial"]:
                adversarial_false_positives += 1

    total = len(rows)
    summary = {
        "dataset_path": str(dataset_path),
        "redis_url": redis_url,
        "redis_namespace": redis_namespace,
        "cache_cosine_threshold": cache_cosine_threshold,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
        "rejected_candidates": rejected,
        "upstream_calls": upstream.calls,
        "hit_rate": hits / total if total else 0.0,
        "upstream_call_rate": upstream.calls / total if total else 0.0,
        "false_positive_rate": false_positives / total if total else 0.0,
        "adversarial_false_positive_rate": (
            adversarial_false_positives / adversarial_total
            if adversarial_total
            else 0.0
        ),
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "elapsed_s": time.perf_counter() - started,
    }

    if output_json is not None:
        Path(output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if output_md is not None:
        Path(output_md).write_text(_render_demo_report(summary), encoding="utf-8")
    return summary


def _render_demo_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LatticeMemory Proxy Replay Demo",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total requests | {summary['total_requests']} |",
            f"| Hit rate | {summary['hit_rate']:.4f} |",
            f"| Upstream call rate | {summary['upstream_call_rate']:.4f} |",
            f"| False-positive rate | {summary['false_positive_rate']:.4f} |",
            f"| Adversarial false-positive rate | {summary['adversarial_false_positive_rate']:.4f} |",
            f"| Rejected candidates | {summary['rejected_candidates']} |",
            f"| Avg latency ms | {summary['latency_ms_avg']:.3f} |",
            "",
            "This demo runs the proxy path with Redis-backed cache storage and cache cosine validation enabled.",
            "",
        ]
    )


def _prompt_from_payload(payload: dict[str, Any]) -> str:
    return "\n".join(str(message.get("content", "")) for message in payload.get("messages") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--redis-namespace", default="lattice-demo")
    parser.add_argument("--cache-cosine-threshold", type=float, default=0.999)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()
    summary = run_replay_demo(
        dataset_path=args.dataset_jsonl,
        redis_url=args.redis_url,
        redis_namespace=args.redis_namespace,
        output_json=args.output_json,
        output_md=args.output_md,
        limit=args.limit,
        cache_cosine_threshold=args.cache_cosine_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
