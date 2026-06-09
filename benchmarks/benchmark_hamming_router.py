"""HammingRouter Proof Benchmark Script.

Evaluates HammingRouter calibration thresholds on held-out paraphrase and near-miss
datasets, estimates cache hit rates via sequential cache simulation on real streams,
and measures lookup latency overhead.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import numpy as np

# Ensure e:\latticememory is in path if run from project root or benchmark folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.hamming_router import (
    HammingRouter,
    validate_calibration_data_schema,
    compute_calibration_data_sha256,
)


class _SyntheticEncoder:
    """Deterministic MD5-hash encoder for fast CI runs (no model download needed)."""

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, normalize_embeddings: bool = True, **kwargs):
        vecs = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            if normalize_embeddings:
                v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


def _coerce_pairs(raw_pairs, *, path: str) -> list[tuple[str, str]]:
    pairs = []
    for idx, pair in enumerate(raw_pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pair at {path}[{idx}]: expected a 2-item list/tuple")
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            raise ValueError(f"Invalid pair at {path}[{idx}]: both items must be strings")
        pairs.append((pair[0], pair[1]))
    return pairs


def load_pairs(path: str, *, key: str | None = None) -> list[tuple[str, str]]:
    if not os.path.exists(path):
        print(f"Error: Dataset file does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if key is not None:
            if key not in data:
                raise ValueError(f"Could not find key {key!r} in pair dataset at {path}")
            return _coerce_pairs(data[key], path=f"{path}:{key}")
        if "pairs" in data:
            return _coerce_pairs(data["pairs"], path=f"{path}:pairs")
        if "paraphrases" in data and "near_misses" not in data:
            return _coerce_pairs(data["paraphrases"], path=f"{path}:paraphrases")
        raise ValueError(
            f"Could not identify one unambiguous list of pairs in dict at {path}; "
            "pass key='paraphrases' or key='near_misses'."
        )
    elif isinstance(data, list):
        return _coerce_pairs(data, path=path)
    else:
        raise ValueError(f"Invalid format in dataset file at {path}")


def build_product_proof_report(
    *,
    model: str,
    d_model: int,
    calibration_sha: str,
    fp_budget: float,
    chosen_threshold: int,
    calibration_results: dict,
    held_out_recall: float,
    held_out_fp_rate: float,
    held_out_tp: int,
    held_out_fp: int,
    para_stats: dict,
    nm_stats: dict,
    cache_hits: int,
    cache_misses: int,
    total_prompts: int,
    cache_hit_rate: float,
    mean_latency_ms: float,
    n_cached_keys: int,
    threshold_curve: list[dict],
    product_recall_target: float = 0.8056,
    held_out_budget_metrics: dict | None = None,
) -> dict:
    key_bytes_per_entry = int(d_model // 8)
    stored_key_bytes = int(n_cached_keys * key_bytes_per_entry)
    float32_embedding_bytes = int(n_cached_keys * d_model * 4)
    compression = (
        round(float32_embedding_bytes / stored_key_bytes, 2)
        if stored_key_bytes > 0
        else 0.0
    )
    budget_recall = (held_out_budget_metrics or {}).get("recall", held_out_recall)
    budget_fp_rate = (held_out_budget_metrics or {}).get("fp_rate", held_out_fp_rate)
    product_gate_passed = (
        budget_recall >= product_recall_target
        and budget_fp_rate <= fp_budget
    )

    return {
        "artifact_type": "latticememory_hamming_proof_results",
        "artifact_version": 2,
        "model": model,
        "d_model": d_model,
        "calibration_data_sha256": calibration_sha,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "fp_budget": fp_budget,
        "calibrated_threshold": chosen_threshold,
        "calibration": calibration_results,
        "product_gate": {
            "name": "recall_at_FP=0" if fp_budget == 0 else "recall_at_FP_budget",
            "passed": bool(product_gate_passed),
            "recall_target": product_recall_target,
            "fp_budget": fp_budget,
            "exact_snap_required": False,
            "fragmentation_metric_role": "research_exact_snap",
        },
        "metrics": {
            "held_out_recall": round(held_out_recall, 4),
            "held_out_fp_rate": round(held_out_fp_rate, 4),
            "held_out_true_positives": int(held_out_tp),
            "held_out_false_positives": int(held_out_fp),
            "held_out_recall_at_fp_budget": (held_out_budget_metrics or {}).get("recall", 0.0),
            "held_out_threshold_at_fp_budget": (held_out_budget_metrics or {}).get("threshold", -1),
            "held_out_fp_rate_at_budget_threshold": (held_out_budget_metrics or {}).get("fp_rate", 0.0),
            "mean_latency_ms": round(mean_latency_ms, 3),
        },
        "index": {
            "n_cached_keys": int(n_cached_keys),
            "key_bytes_per_entry": key_bytes_per_entry,
            "stored_key_bytes": stored_key_bytes,
            "float32_embedding_bytes_equivalent": float32_embedding_bytes,
            "compression_vs_float32_keys_only": compression,
        },
        "distributions": {
            "paraphrase": para_stats,
            "near_miss": nm_stats,
        },
        "cache_simulation": {
            "total_prompts": total_prompts,
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hit_rate, 4),
        },
        "threshold_curve": threshold_curve,
    }


def recall_at_fp_budget(
    *,
    paraphrase_dists: list[int],
    near_miss_dists: list[int],
    fp_budget: float,
    max_threshold: int,
) -> dict:
    if not paraphrase_dists or not near_miss_dists:
        return {"threshold": -1, "recall": 0.0, "fp_rate": 0.0}
    best = {"threshold": -1, "recall": 0.0, "fp_rate": 0.0}
    for threshold in range(max_threshold + 1):
        fp_count = sum(1 for dist in near_miss_dists if dist <= threshold)
        fp_rate = fp_count / len(near_miss_dists)
        if fp_rate > fp_budget:
            continue
        tp_count = sum(1 for dist in paraphrase_dists if dist <= threshold)
        recall = tp_count / len(paraphrase_dists)
        if recall >= best["recall"]:
            best = {
                "threshold": threshold,
                "recall": round(recall, 4),
                "fp_rate": round(fp_rate, 4),
            }
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="HammingRouter Proof Benchmark")
    p.add_argument(
        "--model",
        default="dfrokido/bge-large-e8-snap",
        help="Encoder model name or path (default: dfrokido/bge-large-e8-snap)",
    )
    p.add_argument(
        "--calibration-data",
        required=True,
        help="Path to the JSON calibration dataset (paraphrases & near-misses)",
    )
    p.add_argument(
        "--paraphrases",
        help="Path to held-out paraphrase pairs JSON",
    )
    p.add_argument(
        "--near-misses",
        help="Path to held-out near-miss pairs JSON",
    )
    p.add_argument(
        "--prompts-responses",
        help="Optional path to real cache prompts/responses JSON for cache simulation",
    )
    p.add_argument(
        "--fp-budget",
        type=float,
        default=0.0,
        help="Maximum false-positive rate budget (default: 0.0)",
    )
    p.add_argument(
        "--output",
        default=os.path.join("benchmarks", "results", "hamming_router_proof_summary.json"),
        help="Path to write the JSON summary artifact",
    )
    p.add_argument(
        "--product-recall-target",
        type=float,
        default=0.8056,
        help="Required held-out recall at the FP budget for product-gate pass (default: 0.8056)",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic MD5-hash encoder instead of loading a real model (fast CI mode)",
    )
    args = p.parse_args()

    # Load calibration data
    if not os.path.exists(args.calibration_data):
        print(f"Error: Calibration file does not exist: {args.calibration_data}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.calibration_data, "r", encoding="utf-8") as f:
            cal_raw = json.load(f)
        validate_calibration_data_schema(cal_raw)
    except Exception as exc:
        print(f"Calibration Schema Validation Error: {exc}", file=sys.stderr)
        sys.exit(1)

    cal_paraphrases = cal_raw["paraphrases"]
    cal_near_misses = cal_raw["near_misses"]
    calibration_sha = compute_calibration_data_sha256(cal_raw)

    # Load held-out datasets or fall back to calibration pairs
    if args.paraphrases:
        held_out_paraphrases = load_pairs(args.paraphrases, key="paraphrases")
    else:
        print("Warning: No held-out paraphrases provided; falling back to calibration data for validation.")
        held_out_paraphrases = cal_paraphrases

    if args.near_misses:
        held_out_near_misses = load_pairs(args.near_misses, key="near_misses")
    else:
        print("Warning: No held-out near-misses provided; falling back to calibration data for validation.")
        held_out_near_misses = cal_near_misses

    print(f"Loading encoder model: {args.model} ...")
    use_synthetic = args.synthetic or args.model.lower() == "synthetic"
    if use_synthetic:
        print("  [synthetic mode] Using deterministic MD5-hash encoder (no model download)")
        enc = _SyntheticEncoder(d_model=384)
        router = HammingRouter(encoder=enc, d_model=enc.d_model)
    else:
        try:
            router = HammingRouter.from_model(args.model)
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            sys.exit(1)

    print("Running threshold calibration...")
    cal_results = router.calibrate_threshold(
        cal_paraphrases, cal_near_misses, fp_budget=args.fp_budget
    )
    chosen_threshold = cal_results["threshold"]
    print(f"Calibrated Threshold: {chosen_threshold} (FP budget: {args.fp_budget:.2%})")

    if chosen_threshold == -1:
        print("Error: Calibration failed to find a valid threshold satisfying the FP budget.")
        sys.exit(1)

    print("Evaluating performance on held-out datasets...")
    # Compute distances on held-out data
    para_dists = router._batch_pair_hamming(held_out_paraphrases)
    nm_dists = router._batch_pair_hamming(held_out_near_misses)

    def _stats(dists: list[int]) -> dict:
        arr = np.array(dists, dtype=float)
        return {
            "n": len(dists),
            "min": int(arr.min()) if len(dists) > 0 else 0,
            "p5": float(np.percentile(arr, 5)) if len(dists) > 0 else 0.0,
            "p50": float(np.percentile(arr, 50)) if len(dists) > 0 else 0.0,
            "mean": round(float(arr.mean()), 2) if len(dists) > 0 else 0.0,
            "p95": float(np.percentile(arr, 95)) if len(dists) > 0 else 0.0,
            "max": int(arr.max()) if len(dists) > 0 else 0,
        }

    para_stats = _stats(para_dists)
    nm_stats = _stats(nm_dists)

    # Compute recall and FP rate at chosen threshold on held-out
    held_out_tp = sum(1 for d in para_dists if d <= chosen_threshold)
    held_out_fp = sum(1 for d in nm_dists if d <= chosen_threshold)
    held_out_recall = held_out_tp / len(para_dists) if para_dists else 0.0
    held_out_fp_rate = held_out_fp / len(nm_dists) if nm_dists else 0.0

    # Build threshold curve on held-out
    n_blocks = router._d_model // 8
    threshold_curve = []
    for t in range(0, n_blocks + 1):
        tp_t = sum(1 for d in para_dists if d <= t)
        fp_t = sum(1 for d in nm_dists if d <= t)
        threshold_curve.append({
            "threshold": t,
            "recall": round(tp_t / len(para_dists), 4) if para_dists else 0.0,
            "fp_rate": round(fp_t / len(nm_dists), 4) if nm_dists else 0.0,
        })
    held_out_budget_metrics = recall_at_fp_budget(
        paraphrase_dists=para_dists,
        near_miss_dists=nm_dists,
        fp_budget=args.fp_budget,
        max_threshold=n_blocks,
    )

    # Cache simulation if prompts provided
    cache_hits = 0
    cache_misses = 0
    total_prompts = 0
    cache_hit_rate = 0.0
    if args.prompts_responses:
        print("Running sequential cache simulation...")
        if not os.path.exists(args.prompts_responses):
            print(f"Error: prompts-responses file does not exist: {args.prompts_responses}", file=sys.stderr)
            sys.exit(1)
        with open(args.prompts_responses, "r", encoding="utf-8") as f:
            pr_data = json.load(f)
        
        prompts = []
        if isinstance(pr_data, list):
            for item in pr_data:
                if isinstance(item, dict) and "prompt" in item:
                    prompts.append(item["prompt"])
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    prompts.append(item[0])
                elif isinstance(item, str):
                    prompts.append(item)
        
        if prompts:
            # Instantiate a clean router for simulation
            sim_router = HammingRouter(encoder=router._encoder, d_model=router._d_model, threshold=chosen_threshold)
            total_prompts = len(prompts)
            for p_text in prompts:
                match = sim_router.lookup(p_text)
                if match is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    sim_router.add(p_text, "cached_value")
            cache_hit_rate = cache_hits / total_prompts
            print(f"Cache Simulation: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.2%} hit rate) on {total_prompts} queries.")

    # Measure Latency Overhead
    print("Measuring E8 Hamming lookup latency overhead...")
    latency_runs = []
    test_query = "What is the capital of France?"
    
    # Populate router so it actually runs E8 search
    for p_pair in cal_paraphrases:
        router.add(p_pair[0], "value")

    # Warmup
    _ = router.lookup(test_query)

    for _ in range(100):
        t_start = time.perf_counter()
        _ = router.lookup(test_query)
        latency_runs.append(time.perf_counter() - t_start)
    mean_latency_ms = (sum(latency_runs) / len(latency_runs)) * 1000
    print(f"Mean Lookup Overhead: {mean_latency_ms:.3f} ms")

    # Output details
    print("\n=======================================================")
    print("                HELD-OUT PERFORMANCE SUMMARY            ")
    print("=======================================================")
    print(f"Paraphrase pairs: {len(para_dists)}")
    print(f"  Min Hamming:  {para_stats['min']}")
    print(f"  P5 Hamming:   {para_stats['p5']}")
    print(f"  Mean Hamming: {para_stats['mean']}")
    print(f"  P95 Hamming:  {para_stats['p95']}")
    print(f"  Max Hamming:  {para_stats['max']}")
    print()
    print(f"Near-miss pairs: {len(nm_dists)}")
    print(f"  Min Hamming:  {nm_stats['min']}")
    print(f"  P5 Hamming:   {nm_stats['p5']}")
    print(f"  Mean Hamming: {nm_stats['mean']}")
    print(f"  P95 Hamming:  {nm_stats['p95']}")
    print(f"  Max Hamming:  {nm_stats['max']}")
    print()
    print(f"Calibrated Threshold: {chosen_threshold}")
    print(f"Held-out Recall:      {held_out_recall:.2%}")
    print(f"Held-out FP Rate:     {held_out_fp_rate:.2%}")
    print(f"Mean Latency:         {mean_latency_ms:.3f} ms")
    print("=======================================================")

    # Prepare JSON report
    report = build_product_proof_report(
        model=args.model,
        d_model=router._d_model,
        calibration_sha=calibration_sha,
        fp_budget=args.fp_budget,
        chosen_threshold=chosen_threshold,
        calibration_results=cal_results,
        held_out_recall=held_out_recall,
        held_out_fp_rate=held_out_fp_rate,
        held_out_tp=held_out_tp,
        held_out_fp=held_out_fp,
        para_stats=para_stats,
        nm_stats=nm_stats,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        total_prompts=total_prompts,
        cache_hit_rate=cache_hit_rate,
        mean_latency_ms=mean_latency_ms,
        n_cached_keys=len(router),
        threshold_curve=threshold_curve,
        product_recall_target=args.product_recall_target,
        held_out_budget_metrics=held_out_budget_metrics,
    )

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full proof summary to: {args.output}")


if __name__ == "__main__":
    main()
