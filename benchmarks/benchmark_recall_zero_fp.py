"""Recall-at-zero-FP benchmark for HammingRouter safety claims."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_hamming_router import load_pairs
from latticememory.hamming_router import HammingRouter, compute_calibration_data_sha256, validate_calibration_data_schema


def distance_stats(dists: list[int]) -> dict[str, Any]:
    if not dists:
        return {"n": 0, "min": 0, "p5": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0}
    arr = np.array(dists, dtype=float)
    return {
        "n": len(dists),
        "min": int(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "mean": round(float(arr.mean()), 2),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def recall_at_fp_budgets(
    *,
    paraphrase_dists: list[int],
    near_miss_dists: list[int],
    fp_budgets: list[float],
    max_threshold: int,
) -> dict[str, dict[str, float | int]]:
    if not paraphrase_dists:
        raise ValueError("paraphrase_dists must not be empty")
    if not near_miss_dists:
        raise ValueError("near_miss_dists must not be empty")

    out: dict[str, dict[str, float | int]] = {}
    for budget in fp_budgets:
        best = {"threshold": -1, "recall": 0.0, "fp_rate": 0.0}
        for threshold in range(max_threshold + 1):
            tp = sum(1 for dist in paraphrase_dists if dist <= threshold)
            fp = sum(1 for dist in near_miss_dists if dist <= threshold)
            recall = tp / len(paraphrase_dists)
            fp_rate = fp / len(near_miss_dists)
            if fp_rate <= budget and recall >= best["recall"]:
                best = {
                    "threshold": threshold,
                    "recall": round(recall, 4),
                    "fp_rate": round(fp_rate, 4),
                }
        out[str(budget)] = best
    return out


def _examples_at_threshold(
    pairs: list[tuple[str, str]],
    dists: list[int],
    *,
    threshold: int,
    predicate: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for pair, dist in zip(pairs, dists, strict=True):
        if (predicate == "above" and dist > threshold) or (predicate == "below_or_equal" and dist <= threshold):
            rows.append({"a": pair[0], "b": pair[1], "hamming": dist})
        if len(rows) >= limit:
            break
    return rows


def run_benchmark(
    *,
    model: str,
    calibration_data: str | Path,
    paraphrases: str | Path,
    near_misses: str | Path,
    fp_budgets: list[float],
    output_path: str | Path,
) -> dict[str, Any]:
    calibration_path = Path(calibration_data)
    cal_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_calibration_data_schema(cal_raw)

    paraphrase_pairs = load_pairs(str(paraphrases), key="paraphrases")
    near_miss_pairs = load_pairs(str(near_misses), key="near_misses")
    router = HammingRouter.from_model(model)
    para_dists = router._batch_pair_hamming(paraphrase_pairs)
    near_dists = router._batch_pair_hamming(near_miss_pairs)
    n_blocks = router._d_model // 8
    budget_metrics = recall_at_fp_budgets(
        paraphrase_dists=para_dists,
        near_miss_dists=near_dists,
        fp_budgets=fp_budgets,
        max_threshold=n_blocks,
    )
    zero_fp_threshold = int(budget_metrics[str(fp_budgets[0])]["threshold"])
    report = {
        "artifact_type": "latticememory_recall_at_zero_fp",
        "artifact_version": 1,
        "model": model,
        "d_model": router._d_model,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "calibration_data_sha256": compute_calibration_data_sha256(cal_raw),
        "fp_budgets": fp_budgets,
        "budget_metrics": budget_metrics,
        "distributions": {
            "paraphrase": distance_stats(para_dists),
            "near_miss": distance_stats(near_dists),
        },
        "hamming_gap_p5_minus_p95": round(
            distance_stats(near_dists)["p5"] - distance_stats(para_dists)["p95"], 2
        ),
        "false_negatives_at_zero_fp": _examples_at_threshold(
            paraphrase_pairs,
            para_dists,
            threshold=zero_fp_threshold,
            predicate="above",
        ),
        "false_positives_at_zero_fp": _examples_at_threshold(
            near_miss_pairs,
            near_dists,
            threshold=zero_fp_threshold,
            predicate="below_or_equal",
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure HammingRouter recall at strict FP budgets")
    parser.add_argument("--model", default="dfrokido/bge-large-e8-snap")
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--paraphrases", required=True)
    parser.add_argument("--near-misses", required=True)
    parser.add_argument("--fp-budgets", default="0,0.001,0.01")
    parser.add_argument("--output", default="benchmarks/results/recall_zero_fp.json")
    args = parser.parse_args()
    budgets = [float(part.strip()) for part in args.fp_budgets.split(",") if part.strip()]
    report = run_benchmark(
        model=args.model,
        calibration_data=args.calibration_data,
        paraphrases=args.paraphrases,
        near_misses=args.near_misses,
        fp_budgets=budgets,
        output_path=args.output,
    )
    print(json.dumps({
        "output": args.output,
        "budget_metrics": report["budget_metrics"],
        "hamming_gap_p5_minus_p95": report["hamming_gap_p5_minus_p95"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
