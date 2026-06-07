"""Block-level failure audit for hard near-miss HammingRouter experiments."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_hamming_router import load_pairs
from latticememory.hamming_router import HammingRouter


def _rank_counter(counter: Counter[int], *, n_blocks: int, limit: int = 20) -> list[dict[str, int | float]]:
    total = sum(counter.values())
    rows = []
    for block, count in counter.most_common(limit):
        rows.append({
            "block": int(block),
            "dim_start": int(block * 8),
            "dim_end": int(block * 8 + 7),
            "count": int(count),
            "share": round(count / total, 4) if total else 0.0,
        })
    return rows


def _least_counter(counter: Counter[int], *, n_blocks: int, limit: int = 20) -> list[dict[str, int | float]]:
    total = sum(counter.values())
    rows = []
    for block in range(n_blocks):
        rows.append((block, counter.get(block, 0)))
    rows.sort(key=lambda item: (item[1], item[0]))
    return [
        {
            "block": int(block),
            "dim_start": int(block * 8),
            "dim_end": int(block * 8 + 7),
            "count": int(count),
            "share": round(count / total, 4) if total else 0.0,
        }
        for block, count in rows[:limit]
    ]


def block_failure_summary(
    *,
    paraphrase_rows: list[dict[str, Any]],
    near_miss_rows: list[dict[str, Any]],
    threshold: int,
    n_blocks: int,
    closest_near_miss_count: int = 10,
) -> dict[str, Any]:
    false_negatives = [row for row in paraphrase_rows if row["hamming"] > threshold]
    near_miss_confusions = [row for row in near_miss_rows if row["hamming"] <= threshold]
    closest_near_misses = sorted(near_miss_rows, key=lambda row: row["hamming"])[:closest_near_miss_count]

    fn_counter: Counter[int] = Counter()
    confusion_diff_counter: Counter[int] = Counter()
    confusion_same_counter: Counter[int] = Counter()
    closest_diff_counter: Counter[int] = Counter()
    closest_same_counter: Counter[int] = Counter()

    for row in false_negatives:
        fn_counter.update(row["diff_blocks"])
    for row in near_miss_confusions:
        diff_blocks = set(row["diff_blocks"])
        confusion_diff_counter.update(diff_blocks)
        confusion_same_counter.update(block for block in range(n_blocks) if block not in diff_blocks)
    for row in closest_near_misses:
        diff_blocks = set(row["diff_blocks"])
        closest_diff_counter.update(diff_blocks)
        closest_same_counter.update(block for block in range(n_blocks) if block not in diff_blocks)

    return {
        "threshold": threshold,
        "n_blocks": n_blocks,
        "false_negative_count": len(false_negatives),
        "near_miss_confusion_count": len(near_miss_confusions),
        "closest_near_miss_count": len(closest_near_misses),
        "false_negative_blocks": _rank_counter(fn_counter, n_blocks=n_blocks),
        "near_miss_diff_blocks": _rank_counter(confusion_diff_counter, n_blocks=n_blocks),
        "stable_but_confusing_blocks": _least_counter(confusion_diff_counter, n_blocks=n_blocks),
        "near_miss_same_blocks": _rank_counter(confusion_same_counter, n_blocks=n_blocks),
        "closest_near_miss_diff_blocks": _rank_counter(closest_diff_counter, n_blocks=n_blocks),
        "closest_near_miss_same_blocks": _rank_counter(closest_same_counter, n_blocks=n_blocks),
    }


def _pair_rows(router: HammingRouter, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    texts = [text for pair in pairs for text in pair]
    embs = router._encoder.encode(texts, normalize_embeddings=True)
    keys = [router._emb_to_key_arr(emb) for emb in embs]
    rows = []
    for idx, (a, b) in enumerate(pairs):
        left = keys[2 * idx]
        right = keys[2 * idx + 1]
        diff = np.flatnonzero(left != right).astype(int).tolist()
        rows.append({
            "a": a,
            "b": b,
            "hamming": len(diff),
            "diff_blocks": diff,
        })
    return rows


def run_audit(
    *,
    model: str,
    paraphrases: str | Path,
    near_misses: str | Path,
    threshold: int,
    output_path: str | Path,
) -> dict[str, Any]:
    paraphrase_pairs = load_pairs(str(paraphrases), key="paraphrases")
    near_miss_pairs = load_pairs(str(near_misses), key="near_misses")
    router = HammingRouter.from_model(model)
    n_blocks = router._d_model // 8
    paraphrase_rows = _pair_rows(router, paraphrase_pairs)
    near_miss_rows = _pair_rows(router, near_miss_pairs)
    summary = block_failure_summary(
        paraphrase_rows=paraphrase_rows,
        near_miss_rows=near_miss_rows,
        threshold=threshold,
        n_blocks=n_blocks,
    )
    false_negatives = [row for row in paraphrase_rows if row["hamming"] > threshold]
    near_miss_confusions = [row for row in near_miss_rows if row["hamming"] <= threshold]
    closest_near_misses = sorted(near_miss_rows, key=lambda row: row["hamming"])[:10]
    report = {
        "artifact_type": "latticememory_block_failure_audit",
        "artifact_version": 1,
        "model": model,
        "d_model": router._d_model,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "threshold": threshold,
        "summary": summary,
        "false_negatives": sorted(false_negatives, key=lambda row: row["hamming"], reverse=True),
        "near_miss_confusions": sorted(near_miss_confusions, key=lambda row: row["hamming"]),
        "closest_near_misses": closest_near_misses,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit E8 blocks responsible for HammingRouter failures")
    parser.add_argument("--model", default="dfrokido/bge-large-e8-snap")
    parser.add_argument("--paraphrases", required=True)
    parser.add_argument("--near-misses", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--output", default="benchmarks/results/block_failure_audit.json")
    args = parser.parse_args()
    report = run_audit(
        model=args.model,
        paraphrases=args.paraphrases,
        near_misses=args.near_misses,
        threshold=args.threshold,
        output_path=args.output,
    )
    print(json.dumps({
        "output": args.output,
        "threshold": report["threshold"],
        "false_negative_count": report["summary"]["false_negative_count"],
        "near_miss_confusion_count": report["summary"]["near_miss_confusion_count"],
        "top_false_negative_blocks": report["summary"]["false_negative_blocks"][:5],
        "top_closest_near_miss_same_blocks": report["summary"]["closest_near_miss_same_blocks"][:5],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
