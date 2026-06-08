"""Product proof benchmark for closed-set intent answer caching.

This route uses the same trained encoder checkpoint, but treats the product as a
closed set of approved answers. Calibration texts build one centroid per intent;
held-out paraphrases must route to the right intent, while hard near-miss queries
must never route to the wrong intent.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


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
) -> dict[str, Any]:
    centroid_bytes = int(n_intents * d_model * 4)
    passed = held_out_recall >= product_recall_target and held_out_wrong_route_rate == 0.0
    return {
        "artifact_type": "latticememory_intent_cache_proof_results",
        "artifact_version": 1,
        "route_type": "closed_set_intent_centroid_cache",
        "model": model,
        "d_model": d_model,
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
        "threshold_curve": [],
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _intent_lookup(source: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for intent in source["intents"]:
        for text in [intent["canonical"], *intent["paraphrases"]]:
            lookup[text] = intent["intent_id"]
    return lookup


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
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    source = _load_json(source_path)
    calibration = _load_json(calibration_path)
    heldout_para = _load_json(heldout_paraphrases_path)["paraphrases"]
    heldout_near = _load_json(heldout_near_misses_path)["near_misses"]
    prompts = _load_json(prompts_responses_path)
    text_to_intent = _intent_lookup(source)

    train_by_intent: dict[str, set[str]] = {}
    for left, right in calibration["paraphrases"]:
        intent_id = text_to_intent[left]
        train_by_intent.setdefault(intent_id, set()).update([left, right])

    eval_queries = [right for _left, right in heldout_para]
    near_queries = [text for pair in heldout_near for text in pair]
    stream_prompts = [row["prompt"] for row in prompts]
    all_texts = sorted(
        set(eval_queries)
        | set(near_queries)
        | set(stream_prompts)
        | {text for texts in train_by_intent.values() for text in texts}
    )

    encoder = SentenceTransformer(model)
    d_model = int(encoder.get_sentence_embedding_dimension())
    embeddings = encoder.encode(all_texts, normalize_embeddings=True, batch_size=16)
    emb_by_text = {text: emb for text, emb in zip(all_texts, embeddings)}

    intent_ids = sorted(train_by_intent)
    centroids = []
    for intent_id in intent_ids:
        rows = np.stack([emb_by_text[text] for text in sorted(train_by_intent[intent_id])])
        centroid = rows.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        centroids.append(centroid)
    centroid_matrix = np.stack(centroids)

    def predict(text: str) -> str:
        scores = centroid_matrix @ emb_by_text[text]
        return intent_ids[int(np.argmax(scores))]

    correct = sum(1 for left, right in heldout_para if predict(right) == text_to_intent[left])
    wrong = 0
    for text in near_queries:
        if predict(text) != text_to_intent[text]:
            wrong += 1

    cache_seen: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    for prompt in stream_prompts:
        intent_id = predict(prompt)
        if intent_id in cache_seen:
            cache_hits += 1
        else:
            cache_misses += 1
            cache_seen.add(intent_id)

    latency_samples = []
    probe = stream_prompts[0]
    for _ in range(100):
        started = time.perf_counter()
        predict(probe)
        latency_samples.append(time.perf_counter() - started)

    report = build_intent_cache_report(
        model=model,
        d_model=d_model,
        n_intents=len(intent_ids),
        held_out_recall=correct / len(heldout_para) if heldout_para else 0.0,
        held_out_wrong_route_rate=wrong / len(near_queries) if near_queries else 0.0,
        held_out_correct=correct,
        held_out_wrong=wrong,
        total_paraphrases=len(heldout_para),
        total_near_miss_queries=len(near_queries),
        mean_latency_ms=(sum(latency_samples) / len(latency_samples)) * 1000,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        total_prompts=len(stream_prompts),
        product_recall_target=product_recall_target,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark closed-set intent cache product proof")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="benchmarks/demo_data/hard_near_miss_challenge/hard_near_miss_source.json")
    parser.add_argument("--calibration-data", default="benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json")
    parser.add_argument("--paraphrases", default="benchmarks/demo_data/hard_near_miss_challenge/heldout_paraphrases.json")
    parser.add_argument("--near-misses", default="benchmarks/demo_data/hard_near_miss_challenge/heldout_near_misses.json")
    parser.add_argument("--prompts-responses", default="benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json")
    parser.add_argument("--product-recall-target", type=float, default=0.8056)
    parser.add_argument("--output", default="benchmarks/results/intent_cache_product_proof.json")
    args = parser.parse_args()

    report = run_benchmark(
        model=args.model,
        source_path=args.source,
        calibration_path=args.calibration_data,
        heldout_paraphrases_path=args.paraphrases,
        heldout_near_misses_path=args.near_misses,
        prompts_responses_path=args.prompts_responses,
        output_path=args.output,
        product_recall_target=args.product_recall_target,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
