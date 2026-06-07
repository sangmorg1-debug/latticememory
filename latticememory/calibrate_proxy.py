"""Offline E8 HammingRouter Calibration Utility.

Loads a model and a labeled dataset of paraphrase and near-miss pairs,
evaluates the Hamming distance distributions, prints a threshold table,
and recommends the optimal calibrated threshold for the proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from latticememory.hamming_router import HammingRouter, validate_calibration_data_schema


def main() -> None:
    p = argparse.ArgumentParser(description="Offline HammingRouter Calibration Utility")
    p.add_argument(
        "--model",
        default="dfrokido/bge-large-e8-snap",
        help="Encoder model name or path (default: dfrokido/bge-large-e8-snap)",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to the JSON calibration dataset",
    )
    p.add_argument(
        "--fp-budget",
        type=float,
        default=0.0,
        help="Maximum false-positive rate budget (default: 0.0)",
    )
    p.add_argument(
        "--output",
        help="Optional path to write full calibration results as a JSON artifact",
    )
    args = p.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Calibration file does not exist: {args.data}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            data = json.load(f)
        validate_calibration_data_schema(data)
    except Exception as exc:
        print(f"Schema Validation Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading encoder model: {args.model} ...")
    try:
        router = HammingRouter.from_model(args.model)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Computing Hamming distance statistics...")
    paraphrase_pairs = data["paraphrases"]
    near_miss_pairs = data["near_misses"]

    gap_results = router.gap_stats(paraphrase_pairs, near_miss_pairs)
    cal_results = router.calibrate_threshold(
        paraphrase_pairs, near_miss_pairs, fp_budget=args.fp_budget
    )

    print("\n=======================================================")
    print("                HAMMING DISTANCE STATISTICS            ")
    print("=======================================================")
    print(f"Paraphrase pairs: {gap_results['n_paraphrase_pairs']}")
    print(f"  Min Hamming: {gap_results['paraphrase']['min']}")
    print(f"  P5 Hamming:  {gap_results['paraphrase']['p5']}")
    print(f"  Mean Hamming: {gap_results['paraphrase']['mean']}")
    print(f"  P95 Hamming: {gap_results['paraphrase']['p95']}")
    print(f"  Max Hamming: {gap_results['paraphrase']['max']}")
    print()
    print(f"Near-miss pairs: {gap_results['n_near_miss_pairs']}")
    print(f"  Min Hamming: {gap_results['near_miss']['min']}")
    print(f"  P5 Hamming:  {gap_results['near_miss']['p5']}")
    print(f"  Mean Hamming: {gap_results['near_miss']['mean']}")
    print(f"  P95 Hamming: {gap_results['near_miss']['p95']}")
    print(f"  Max Hamming: {gap_results['near_miss']['max']}")
    print()
    print(f"Hamming Gap (near_miss_p5 - paraphrase_p95): {gap_results['gap']}")
    print("=======================================================")

    print("\nTHRESHOLD TABLE:")
    print("-------------------------------------------------------")
    print(f"{'Threshold':10} | {'Recall (TP Rate)':17} | {'FP Rate':10}")
    print("-------------------------------------------------------")
    for row in gap_results["threshold_table"]:
        print(f"{row['threshold']:<10} | {row['recall']:<17.3f} | {row['fp_rate']:<10.3f}")
    print("-------------------------------------------------------")

    print("\nRECOMMENDED CONFIGURATION:")
    print("-------------------------------------------------------")
    if cal_results["threshold"] == -1:
        print("WARNING: No valid threshold satisfies the requested FP budget.")
        print(f"FP Budget: {args.fp_budget}")
    else:
        print(f"Optimal Threshold: {cal_results['threshold']}")
        print(f"Expected Recall:   {cal_results['recall']:.2%}")
        print(f"Expected FP Rate:  {cal_results['fp_rate']:.2%}")
        print(f"FP Budget:         {cal_results['fp_budget']:.2%}")
        print(f"Reliable:          {'Yes' if len(paraphrase_pairs) >= 100 and len(near_miss_pairs) >= 100 else 'No (low pair count)'}")
    print("-------------------------------------------------------")

    if args.output:
        import datetime
        from latticememory.hamming_router import compute_calibration_data_sha256
        sha = compute_calibration_data_sha256(data)

        out_data = {
            "artifact_type": "latticememory_hamming_calibration",
            "artifact_version": 1,
            "model": args.model,
            "d_model": router._d_model,
            "calibration_data_sha256": sha,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "fp_budget": args.fp_budget,
            "calibration": {
                "threshold": cal_results["threshold"],
                "recall": cal_results["recall"],
                "fp_rate": cal_results["fp_rate"],
                "n_paraphrase_pairs": len(paraphrase_pairs),
                "n_near_miss_pairs": len(near_miss_pairs),
            },
            "gap_stats": gap_results,
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            print(f"\nWrote full calibration JSON artifact to: {args.output}")
        except Exception as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
