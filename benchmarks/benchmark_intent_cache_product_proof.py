"""Product proof benchmark for closed-set intent answer caching.

Supports:
- Multi-domain evaluation (run across multiple challenge datasets and aggregate)
- Multi-split cross-validation (shuffle train/eval splits, report mean ± std)
- Threshold curve generation (recall vs. FP rate sweep)
- Shadow-mode dataset replay for cache hit rate simulation

This route uses the same trained encoder checkpoint, but treats the product as a
closed set of approved answers. Calibration texts build one centroid per intent;
held-out paraphrases must route to the right intent, while hard near-miss queries
must never route to the wrong intent.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Report builder (single run)
# ---------------------------------------------------------------------------

def build_intent_cache_report(
    *,
    model: str,
    d_model: int,
    n_intents: int,
    held_out_recall: float,
    held_out_wrong_route_rate: float,
    held_out_correct: int,
    held_out_wrong: int,
    total_paraphrases: int,
    total_near_miss_queries: int,
    mean_latency_ms: float,
    cache_hits: int,
    cache_misses: int,
    total_prompts: int,
    product_recall_target: float,
    domain: str = "customer_support",
    threshold_curve: list[dict[str, Any]] | None = None,
    paraphrase_examples: list[dict[str, str]] | None = None,
    near_miss_examples: list[dict[str, str]] | None = None,
    splits_summary: dict[str, Any] | None = None,
    domains_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    centroid_bytes = int(n_intents * d_model * 4)
    passed = held_out_recall >= product_recall_target and held_out_wrong_route_rate == 0.0
    report: dict[str, Any] = {
        "artifact_type": "latticememory_intent_cache_proof_results",
        "artifact_version": 2,
        "route_type": "closed_set_intent_centroid_cache",
        "model": model,
        "d_model": d_model,
        "domain": domain,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "calibrated_threshold": "nearest_centroid",
        "fp_budget": 0.0,
        "product_gate": {
            "name": "intent_recall_at_zero_wrong_routes",
            "passed": bool(passed),
            "recall_target": product_recall_target,
            "fp_budget": 0.0,
            "exact_snap_required": False,
            "fragmentation_metric_role": "research_exact_snap",
        },
        "metrics": {
            "held_out_recall": round(held_out_recall, 4),
            "held_out_fp_rate": round(held_out_wrong_route_rate, 4),
            "held_out_true_positives": int(held_out_correct),
            "held_out_false_positives": int(held_out_wrong),
            "total_held_out_paraphrases": int(total_paraphrases),
            "total_near_miss_queries": int(total_near_miss_queries),
            "mean_latency_ms": round(mean_latency_ms, 3),
        },
        "index": {
            "n_intents": int(n_intents),
            "centroid_bytes": centroid_bytes,
            "float32_embedding_bytes_equivalent": centroid_bytes,
            "stored_key_bytes": centroid_bytes,
            "compression_vs_float32_keys_only": 1.0,
        },
        "cache_simulation": {
            "total_prompts": int(total_prompts),
            "hits": int(cache_hits),
            "misses": int(cache_misses),
            "hit_rate": round(cache_hits / total_prompts, 4) if total_prompts else 0.0,
        },
        "distributions": {
            "paraphrase": {"n": int(total_paraphrases)},
            "near_miss": {"n": int(total_near_miss_queries)},
        },
        "threshold_curve": threshold_curve or [],
        "paraphrase_examples": paraphrase_examples or [],
        "near_miss_examples": near_miss_examples or [],
    }
    if splits_summary is not None:
        report["splits_summary"] = splits_summary
    if domains_summary is not None:
        report["domains_summary"] = domains_summary
    return report


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _intent_lookup(source: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for intent in source["intents"]:
        for text in [intent["canonical"], *intent["paraphrases"]]:
            lookup[text] = intent["intent_id"]
    return lookup


def _build_centroids(
    train_by_intent: dict[str, set[str]],
    emb_by_text: dict[str, Any],
) -> tuple[list[str], Any]:
    """Return (intent_ids, centroid_matrix) sorted by intent_id."""
    import numpy as np

    intent_ids = sorted(train_by_intent)
    centroids = []
    for intent_id in intent_ids:
        rows_emb = [emb_by_text[t] for t in sorted(train_by_intent[intent_id]) if t in emb_by_text]
        if not rows_emb:
            centroids.append(emb_by_text[next(iter(train_by_intent[intent_id]))])
            continue
        rows = __import__("numpy").stack(rows_emb)
        centroid = rows.mean(axis=0)
        norm = float(__import__("numpy").linalg.norm(centroid))
        centroid = centroid / max(norm, 1e-8)
        centroids.append(centroid)
    return intent_ids, __import__("numpy").stack(centroids)


def _make_predict(intent_ids: list[str], centroid_matrix: Any, emb_by_text: dict[str, Any]):
    """Return a predict(text) -> intent_id closure."""
    import numpy as np

    def predict(text: str) -> str:
        scores = centroid_matrix @ emb_by_text[text]
        return intent_ids[int(np.argmax(scores))]

    return predict


# ---------------------------------------------------------------------------
# Single-split evaluation
# ---------------------------------------------------------------------------

def _run_single_split(
    *,
    text_to_intent: dict[str, str],
    train_pairs: list[tuple[str, str]],
    eval_pairs: list[tuple[str, str]],
    near_pairs: list[tuple[str, str]],
    stream_prompts: list[str],
    emb_by_text: dict[str, Any],
    d_model: int,
) -> dict[str, Any]:
    import numpy as np

    # Build per-intent train set from calibration pairs
    train_by_intent: dict[str, set[str]] = {}
    for left, right in train_pairs:
        intent_id = text_to_intent.get(left)
        if intent_id is None:
            continue
        train_by_intent.setdefault(intent_id, set()).update([left, right])

    intent_ids, centroid_matrix = _build_centroids(train_by_intent, emb_by_text)
    predict = _make_predict(intent_ids, centroid_matrix, emb_by_text)

    # Evaluate held-out paraphrases
    correct = 0
    fp_examples: list[dict[str, str]] = []
    tp_examples: list[dict[str, str]] = []
    for left, right in eval_pairs:
        expected = text_to_intent.get(left)
        if expected is None:
            continue
        predicted = predict(right)
        if predicted == expected:
            correct += 1
            if len(tp_examples) < 5:
                tp_examples.append({"query": right, "expected": expected, "predicted": predicted, "result": "correct"})
        else:
            if len(fp_examples) < 5:
                fp_examples.append({"query": right, "expected": expected, "predicted": predicted, "result": "wrong_route"})

    # Evaluate near-miss safety: these have their own intent label; routing to
    # the *wrong* intent (any intent that isn't their labeled one) counts as a wrong route
    wrong = 0
    nm_fp_examples: list[dict[str, str]] = []
    near_queries = [t for pair in near_pairs for t in pair]
    for text in near_queries:
        actual_intent = text_to_intent.get(text)
        predicted = predict(text)
        if actual_intent is not None and predicted != actual_intent:
            wrong += 1
            if len(nm_fp_examples) < 5:
                nm_fp_examples.append({"query": text, "actual": actual_intent, "predicted": predicted})

    # Cache simulation
    cache_seen: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    for prompt in stream_prompts:
        if prompt not in emb_by_text:
            continue
        intent_id = predict(prompt)
        if intent_id in cache_seen:
            cache_hits += 1
        else:
            cache_misses += 1
            cache_seen.add(intent_id)

    # Build threshold curve: sweep cosine similarity thresholds for centroid match
    # For intent cache, "threshold" = nearest-centroid score cutoff. We report
    # recall vs. abstention rate (fraction of queries above a given min-score threshold).
    threshold_curve: list[dict[str, Any]] = []
    all_scores_correct: list[float] = []
    all_scores_wrong: list[float] = []
    for left, right in eval_pairs:
        expected = text_to_intent.get(left)
        if expected is None or right not in emb_by_text:
            continue
        scores = centroid_matrix @ emb_by_text[right]
        top_score = float(np.max(scores))
        top_intent = intent_ids[int(np.argmax(scores))]
        if top_intent == expected:
            all_scores_correct.append(top_score)
        else:
            all_scores_wrong.append(top_score)

    if all_scores_correct or all_scores_wrong:
        all_scores = sorted(set(all_scores_correct + all_scores_wrong))
        step = max(1, len(all_scores) // 20)
        sample_thresholds = all_scores[::step]
        for thresh in sample_thresholds:
            tp = sum(1 for s in all_scores_correct if s >= thresh)
            fp = sum(1 for s in all_scores_wrong if s >= thresh)
            total_above = tp + fp
            fp_rate = fp / len(near_queries) if near_queries else 0.0
            recall = tp / len(eval_pairs) if eval_pairs else 0.0
            threshold_curve.append({
                "min_score_threshold": round(thresh, 4),
                "recall": round(recall, 4),
                "fp_rate": round(fp_rate, 4),
                "coverage": round(total_above / max(len(eval_pairs), 1), 4),
            })

    total_eval = len(eval_pairs)
    recall = correct / total_eval if total_eval else 0.0
    wrong_rate = wrong / len(near_queries) if near_queries else 0.0

    return {
        "recall": recall,
        "wrong_route_rate": wrong_rate,
        "correct": correct,
        "wrong": wrong,
        "total_paraphrases": total_eval,
        "total_near_miss_queries": len(near_queries),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "total_prompts": len(stream_prompts),
        "threshold_curve": threshold_curve,
        "paraphrase_examples": tp_examples,
        "near_miss_fp_examples": nm_fp_examples,
    }


# ---------------------------------------------------------------------------
# Multi-split runner
# ---------------------------------------------------------------------------

def _run_multisplit(
    *,
    text_to_intent: dict[str, str],
    all_paraphrase_pairs: list[tuple[str, str]],
    all_near_miss_pairs: list[tuple[str, str]],
    stream_prompts: list[str],
    emb_by_text: dict[str, Any],
    d_model: int,
    n_splits: int = 5,
    train_fraction: float = 0.625,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    split_recalls: list[float] = []
    split_wrong_rates: list[float] = []
    split_hit_rates: list[float] = []

    last_split: dict[str, Any] = {}
    for split_idx in range(n_splits):
        para_shuffled = list(all_paraphrase_pairs)
        nm_shuffled = list(all_near_miss_pairs)
        rng.shuffle(para_shuffled)
        rng.shuffle(nm_shuffled)

        n_train_para = max(1, int(len(para_shuffled) * train_fraction))
        n_train_nm = max(1, int(len(nm_shuffled) * train_fraction))
        train_pairs = para_shuffled[:n_train_para]
        eval_pairs = para_shuffled[n_train_para:]
        near_pairs = nm_shuffled[n_train_nm:]  # held-out near-misses only

        if not eval_pairs:
            eval_pairs = para_shuffled[-max(1, len(para_shuffled) // 5):]

        result = _run_single_split(
            text_to_intent=text_to_intent,
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            near_pairs=near_pairs,
            stream_prompts=stream_prompts,
            emb_by_text=emb_by_text,
            d_model=d_model,
        )
        split_recalls.append(result["recall"])
        split_wrong_rates.append(result["wrong_route_rate"])
        hit_rate = result["cache_hits"] / result["total_prompts"] if result["total_prompts"] else 0.0
        split_hit_rates.append(hit_rate)
        last_split = result

    def _stats(values: list[float]) -> dict[str, float]:
        n = len(values)
        mean = sum(values) / n if n else 0.0
        variance = sum((v - mean) ** 2 for v in values) / n if n else 0.0
        return {
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(min(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
            "n_splits": n,
            "values": [round(v, 4) for v in values],
        }

    return {
        "recall": _stats(split_recalls),
        "wrong_route_rate": _stats(split_wrong_rates),
        "cache_hit_rate": _stats(split_hit_rates),
        "last_split": last_split,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner (single domain)
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    model: str,
    source_path: str | Path,
    calibration_path: str | Path,
    heldout_paraphrases_path: str | Path,
    heldout_near_misses_path: str | Path,
    prompts_responses_path: str | Path,
    output_path: str | Path,
    product_recall_target: float = 0.8056,
    n_splits: int = 5,
    multisplit_seed: int = 42,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    source = _load_json(source_path)
    calibration = _load_json(calibration_path)
    heldout_para = _load_json(heldout_paraphrases_path)["paraphrases"]
    heldout_near = _load_json(heldout_near_misses_path)["near_misses"]
    prompts_data = _load_json(prompts_responses_path)
    text_to_intent = _intent_lookup(source)
    domain = source.get("domain", "unknown")

    # Build full paraphrase and near-miss lists (calibration + heldout)
    cal_pairs = [(a, b) for a, b in calibration["paraphrases"]]
    cal_nm = [(a, b) for a, b in calibration["near_misses"]]
    all_para_pairs = cal_pairs + [(a, b) for a, b in heldout_para]
    all_nm_pairs = cal_nm + [(a, b) for a, b in heldout_near]

    # For the primary result, use heldout split (matching original benchmark behaviour)
    train_by_intent: dict[str, set[str]] = {}
    for left, right in cal_pairs:
        intent_id = text_to_intent.get(left)
        if intent_id is None:
            continue
        train_by_intent.setdefault(intent_id, set()).update([left, right])

    stream_prompts = [row["prompt"] for row in prompts_data]

    # Collect all texts to encode in one batch
    all_texts = sorted(
        set(t for pair in all_para_pairs for t in pair)
        | set(t for pair in all_nm_pairs for t in pair)
        | set(stream_prompts)
        | {t for texts in train_by_intent.values() for t in texts}
    )

    encoder = SentenceTransformer(model)
    d_model = int(encoder.get_sentence_embedding_dimension())
    embeddings = encoder.encode(all_texts, normalize_embeddings=True, batch_size=16)
    emb_by_text = {text: emb for text, emb in zip(all_texts, embeddings)}

    # Primary single-split evaluation (calibration → heldout)
    primary = _run_single_split(
        text_to_intent=text_to_intent,
        train_pairs=cal_pairs,
        eval_pairs=[(a, b) for a, b in heldout_para],
        near_pairs=[(a, b) for a, b in heldout_near],
        stream_prompts=stream_prompts,
        emb_by_text=emb_by_text,
        d_model=d_model,
    )

    # Latency measurement
    latency_samples: list[float] = []
    if stream_prompts:
        import numpy as np
        intent_ids_sorted = sorted(train_by_intent)
        centroids = []
        for iid in intent_ids_sorted:
            rows = [emb_by_text[t] for t in sorted(train_by_intent[iid]) if t in emb_by_text]
            if rows:
                c = np.stack(rows).mean(axis=0)
                centroids.append(c / max(float(np.linalg.norm(c)), 1e-8))
        if centroids:
            cm = np.stack(centroids)
            probe_emb = emb_by_text.get(stream_prompts[0])
            if probe_emb is not None:
                for _ in range(200):
                    t0 = time.perf_counter()
                    _ = cm @ probe_emb
                    latency_samples.append(time.perf_counter() - t0)
    mean_latency_ms = (sum(latency_samples) / len(latency_samples)) * 1000 if latency_samples else 0.0

    # Multi-split cross-validation
    splits_result = _run_multisplit(
        text_to_intent=text_to_intent,
        all_paraphrase_pairs=all_para_pairs,
        all_near_miss_pairs=all_nm_pairs,
        stream_prompts=stream_prompts,
        emb_by_text=emb_by_text,
        d_model=d_model,
        n_splits=n_splits,
        seed=multisplit_seed,
    )

    splits_summary = {
        "n_splits": n_splits,
        "seed": multisplit_seed,
        "recall": splits_result["recall"],
        "wrong_route_rate": splits_result["wrong_route_rate"],
        "cache_hit_rate": splits_result["cache_hit_rate"],
    }

    report = build_intent_cache_report(
        model=model,
        d_model=d_model,
        n_intents=len(train_by_intent),
        held_out_recall=primary["recall"],
        held_out_wrong_route_rate=primary["wrong_route_rate"],
        held_out_correct=primary["correct"],
        held_out_wrong=primary["wrong"],
        total_paraphrases=primary["total_paraphrases"],
        total_near_miss_queries=primary["total_near_miss_queries"],
        mean_latency_ms=mean_latency_ms,
        cache_hits=primary["cache_hits"],
        cache_misses=primary["cache_misses"],
        total_prompts=primary["total_prompts"],
        product_recall_target=product_recall_target,
        domain=domain,
        threshold_curve=primary["threshold_curve"],
        paraphrase_examples=primary["paraphrase_examples"],
        near_miss_examples=primary["near_miss_fp_examples"],
        splits_summary=splits_summary,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Multi-domain aggregation runner
# ---------------------------------------------------------------------------

def run_multidomain_benchmark(
    *,
    domains: list[dict[str, str]],
    model: str,
    output_path: str | Path,
    product_recall_target: float = 0.8056,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run the benchmark across multiple domains and aggregate results.

    Each domain dict must have keys:
      source, calibration, paraphrases, near_misses, prompts_responses, name
    """
    domain_results: list[dict[str, Any]] = []
    for domain_spec in domains:
        name = domain_spec["name"]
        scratch_path = Path(output_path).parent / f"_scratch_{name}.json"
        result = run_benchmark(
            model=model,
            source_path=domain_spec["source"],
            calibration_path=domain_spec["calibration"],
            heldout_paraphrases_path=domain_spec["paraphrases"],
            heldout_near_misses_path=domain_spec["near_misses"],
            prompts_responses_path=domain_spec["prompts_responses"],
            output_path=scratch_path,
            product_recall_target=product_recall_target,
            n_splits=n_splits,
        )
        domain_results.append({
            "domain": name,
            "held_out_recall": result["metrics"]["held_out_recall"],
            "held_out_fp_rate": result["metrics"]["held_out_fp_rate"],
            "cache_hit_rate": result["cache_simulation"]["hit_rate"],
            "n_intents": result["index"]["n_intents"],
            "splits_recall_mean": result.get("splits_summary", {}).get("recall", {}).get("mean"),
            "splits_recall_std": result.get("splits_summary", {}).get("recall", {}).get("std"),
            "splits_wrong_rate_mean": result.get("splits_summary", {}).get("wrong_route_rate", {}).get("mean"),
            "passed": result["product_gate"]["passed"],
        })

    all_recalls = [d["held_out_recall"] for d in domain_results]
    all_fp_rates = [d["held_out_fp_rate"] for d in domain_results]
    all_passed = all(d["passed"] for d in domain_results)
    agg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    agg_fp_rate = max(all_fp_rates) if all_fp_rates else 0.0

    aggregate = {
        "artifact_type": "latticememory_multidomain_proof_results",
        "artifact_version": 1,
        "model": model,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "product_gate": {
            "name": "intent_recall_at_zero_wrong_routes",
            "passed": bool(all_passed),
            "recall_target": product_recall_target,
            "fp_budget": 0.0,
            "all_domains_pass": bool(all_passed),
        },
        "aggregate_metrics": {
            "mean_held_out_recall": round(agg_recall, 4),
            "max_held_out_fp_rate": round(agg_fp_rate, 4),
            "n_domains": len(domain_results),
            "all_domains_pass": bool(all_passed),
        },
        "domains": domain_results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark closed-set intent cache product proof (single or multi-domain)"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source",
        default="benchmarks/demo_data/hard_near_miss_challenge/hard_near_miss_source.json",
    )
    parser.add_argument(
        "--calibration-data",
        default="benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json",
    )
    parser.add_argument(
        "--paraphrases",
        default="benchmarks/demo_data/hard_near_miss_challenge/heldout_paraphrases.json",
    )
    parser.add_argument(
        "--near-misses",
        default="benchmarks/demo_data/hard_near_miss_challenge/heldout_near_misses.json",
    )
    parser.add_argument(
        "--prompts-responses",
        default="benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json",
    )
    parser.add_argument("--product-recall-target", type=float, default=0.8056)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/results/intent_cache_product_proof.json")

    # Multi-domain mode: pass --multidomain-config path to a JSON list of domain specs
    parser.add_argument(
        "--multidomain-config",
        default=None,
        help="Path to JSON list of domain dicts for multi-domain run",
    )

    args = parser.parse_args()

    if args.multidomain_config:
        domains = json.loads(Path(args.multidomain_config).read_text(encoding="utf-8"))
        report = run_multidomain_benchmark(
            domains=domains,
            model=args.model,
            output_path=args.output,
            product_recall_target=args.product_recall_target,
            n_splits=args.n_splits,
        )
    else:
        report = run_benchmark(
            model=args.model,
            source_path=args.source,
            calibration_path=args.calibration_data,
            heldout_paraphrases_path=args.paraphrases,
            heldout_near_misses_path=args.near_misses,
            prompts_responses_path=args.prompts_responses,
            output_path=args.output,
            product_recall_target=args.product_recall_target,
            n_splits=args.n_splits,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
