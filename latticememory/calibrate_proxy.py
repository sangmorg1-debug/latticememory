"""Offline E8 HammingRouter Calibration Utility.

Loads a model and a labeled dataset of paraphrase and near-miss pairs,
evaluates the Hamming distance distributions, prints a threshold table,
and recommends the optimal calibrated threshold for the proxy.

Usage:
    # Basic calibration (auto-selects threshold at zero FP)
    latticememory-calibrate-proxy --data calibration.json --output cal.json

    # Allow 1% false positives for higher recall
    latticememory-calibrate-proxy --data calibration.json --fp-budget 0.01 --output cal.json

    # Force a specific threshold (for experimentation / override)
    latticememory-calibrate-proxy --data calibration.json --threshold 65 --dry-run

    # Quiet mode: only write the JSON artifact, no stdout tables
    latticememory-calibrate-proxy --data calibration.json --output cal.json --quiet
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
        help="Maximum false-positive rate budget (default: 0.0 = zero FP)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Manually override the recommended threshold (skips auto-calibration; use with --dry-run to evaluate)",
    )
    p.add_argument(
        "--output",
        help="Optional path to write full calibration results as a JSON artifact",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and recommended config but do NOT write --output file",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable table output (useful when piping --output to scripts)",
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

    if not args.quiet:
        print(f"Loading encoder model: {args.model} ...")
    try:
        router = HammingRouter.from_model(args.model)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Computing Hamming distance statistics...")
    paraphrase_pairs = data["paraphrases"]
    near_miss_pairs = data["near_misses"]

    gap_results = router.gap_stats(paraphrase_pairs, near_miss_pairs)

    # Determine threshold to use
    if args.threshold is not None:
        # Manual override: compute recall and FP at the given threshold
        para_dists = router._batch_pair_hamming(paraphrase_pairs)
        nm_dists = router._batch_pair_hamming(near_miss_pairs)
        t = args.threshold
        tp = sum(1 for d in para_dists if d <= t)
        fp = sum(1 for d in nm_dists if d <= t)
        recall = round(tp / len(para_dists), 4) if para_dists else 0.0
        fp_rate = round(fp / len(nm_dists), 4) if nm_dists else 0.0
        cal_results = {
            "threshold": t,
            "recall": recall,
            "fp_rate": fp_rate,
            "fp_budget": args.fp_budget,
            "n_paraphrase_pairs": len(paraphrase_pairs),
            "n_near_miss_pairs": len(near_miss_pairs),
            "source": "manual_override",
        }
    else:
        cal_results = router.calibrate_threshold(
            paraphrase_pairs, near_miss_pairs, fp_budget=args.fp_budget
        )
        cal_results["source"] = "auto_calibrated"

    if not args.quiet:
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
        if gap_results["gap"] <= 0:
            print("  WARNING: Gap <= 0. No threshold gives FP=0 at non-zero recall.")
            print("  Consider training a snap encoder or raising --fp-budget.")
        print("=======================================================")

        print("\nTHRESHOLD TABLE:")
        print("-------------------------------------------------------")
        print(f"{'Threshold':10} | {'Recall (TP Rate)':17} | {'FP Rate':10}")
        print("-------------------------------------------------------")
        for row in gap_results["threshold_table"]:
            marker = " <-- selected" if row["threshold"] == cal_results["threshold"] else ""
            print(f"{row['threshold']:<10} | {row['recall']:<17.3f} | {row['fp_rate']:<10.3f}{marker}")
        print("-------------------------------------------------------")

        print("\nRECOMMENDED CONFIGURATION:")
        print("-------------------------------------------------------")
        src = cal_results.get("source", "")
        if args.threshold is not None:
            print(f"Mode:              Manual override (--threshold {args.threshold})")
        else:
            print(f"Mode:              Auto-calibrated (fp_budget={args.fp_budget:.3f})")
        if cal_results["threshold"] == -1:
            print("WARNING: No valid threshold satisfies the requested FP budget.")
            print(f"FP Budget: {args.fp_budget}")
        else:
            reliable = len(paraphrase_pairs) >= 100 and len(near_miss_pairs) >= 100
            print(f"Optimal Threshold: {cal_results['threshold']}")
            print(f"Expected Recall:   {cal_results['recall']:.2%}")
            print(f"Expected FP Rate:  {cal_results['fp_rate']:.2%}")
            print(f"FP Budget:         {cal_results['fp_budget']:.2%}")
            print(f"Reliable:          {'Yes' if reliable else 'No (< 100 pairs per type — not production-ready)'}")
        print("-------------------------------------------------------")

        if args.dry_run:
            print("\n[dry-run] No artifact written.")
            return

    if args.output and not args.dry_run:
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
                "source": cal_results.get("source", "auto_calibrated"),
            },
            "gap_stats": gap_results,
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            if not args.quiet:
                print(f"\nWrote full calibration JSON artifact to: {args.output}")
            else:
                # Quiet mode: emit the artifact path as a single JSON line for scripting
                print(json.dumps({"artifact": args.output, "threshold": cal_results["threshold"],
                                  "recall": cal_results["recall"], "fp_rate": cal_results["fp_rate"]}))
        except Exception as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
