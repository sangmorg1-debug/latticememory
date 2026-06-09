"""Multi-domain intent cache product proof runner.

Aggregates ``benchmark_intent_cache_product_proof.run_benchmark`` results
across multiple domains defined in a JSON config file, producing a single
rolled-up report.  Designed to be run after training a snap encoder.

Usage (synthetic, no real model — fast CI validation):
    python -m benchmarks.benchmark_multidomain_runner \\
        --model synthetic \\
        --config benchmarks/multidomain_config.json \\
        --output benchmarks/results/multidomain_proof.json

Usage (trained checkpoint):
    python -m benchmarks.benchmark_multidomain_runner \\
        --model benchmarks/results/snap_product_gate_hard_symmetric_8ep/best_snap_encoder \\
        --config benchmarks/multidomain_config.json \\
        --output benchmarks/results/multidomain_proof.json

Config format (JSON list):
    [
        {
            "name": "domain_label",
            "source": "path/to/source.json",
            "calibration": "path/to/calibration_data.json",
            "paraphrases": "path/to/heldout_paraphrases.json",
            "near_misses": "path/to/heldout_near_misses.json",
            "prompts_responses": "path/to/prompts_responses.json"
        },
        ...
    ]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic encoder (mirrors benchmark_shadow_mode_demo — no real model needed)
# ---------------------------------------------------------------------------

class _SyntheticEncoder:
    """Deterministic unit-sphere embedding based on text hash — test use only."""

    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def get_sentence_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, texts: list[str], normalize_embeddings: bool = True, batch_size: int = 64):
        import hashlib
        import numpy as np

        out = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.d_model).astype("float32")
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                vec = vec / max(norm, 1e-8)
            out.append(vec)
        return np.stack(out)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_multidomain_report(
    *,
    model: str,
    domain_results: list[dict[str, Any]],
    product_recall_target: float,
) -> dict[str, Any]:
    """Build a rolled-up multi-domain proof report from per-domain results.

    Parameters
    ----------
    model:
        Model name / path used for all domains.
    domain_results:
        List of per-domain dicts, each with keys:
        ``domain``, ``held_out_recall``, ``held_out_fp_rate``,
        ``cache_hit_rate``, ``n_intents``, ``passed``,
        and optionally ``splits_recall_mean``, ``splits_recall_std``.
    product_recall_target:
        The recall threshold used as the product gate.
    """
    if not domain_results:
        raise ValueError("domain_results must not be empty")

    recalls = [d["held_out_recall"] for d in domain_results]
    fp_rates = [d["held_out_fp_rate"] for d in domain_results]
    hit_rates = [d.get("cache_hit_rate", 0.0) for d in domain_results]
    all_passed = all(d["passed"] for d in domain_results)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals: list[float]) -> float:
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if vals else 0.0

    return {
        "artifact_type": "latticememory_multidomain_proof_results",
        "artifact_version": 2,
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
            "n_domains": len(domain_results),
            "mean_held_out_recall": round(_mean(recalls), 4),
            "std_held_out_recall": round(_std(recalls), 4),
            "min_held_out_recall": round(min(recalls), 4),
            "max_held_out_fp_rate": round(max(fp_rates), 4),
            "mean_cache_hit_rate": round(_mean(hit_rates), 4),
            "all_domains_pass": bool(all_passed),
        },
        "domains": domain_results,
    }


# ---------------------------------------------------------------------------
# Per-domain runner (wraps intent cache proof benchmark)
# ---------------------------------------------------------------------------

def run_domain(
    *,
    name: str,
    model: str,
    source_path: str | Path,
    calibration_path: str | Path,
    heldout_paraphrases_path: str | Path,
    heldout_near_misses_path: str | Path,
    prompts_responses_path: str | Path,
    product_recall_target: float,
    n_splits: int,
    scratch_dir: Path,
    encoder=None,
) -> dict[str, Any]:
    """Run a single domain and return a summary dict suitable for domain_results."""
    from benchmarks.benchmark_intent_cache_product_proof import run_benchmark

    scratch_path = scratch_dir / f"_domain_{name}.json"

    if encoder is not None:
        # Inject pre-built encoder to avoid reloading the model per domain.
        _run_with_encoder(
            encoder=encoder,
            name=name,
            source_path=source_path,
            calibration_path=calibration_path,
            heldout_paraphrases_path=heldout_paraphrases_path,
            heldout_near_misses_path=heldout_near_misses_path,
            prompts_responses_path=prompts_responses_path,
            product_recall_target=product_recall_target,
            n_splits=n_splits,
            output_path=scratch_path,
        )
        result = json.loads(scratch_path.read_text(encoding="utf-8"))
    else:
        result = run_benchmark(
            model=model,
            source_path=source_path,
            calibration_path=calibration_path,
            heldout_paraphrases_path=heldout_paraphrases_path,
            heldout_near_misses_path=heldout_near_misses_path,
            prompts_responses_path=prompts_responses_path,
            output_path=scratch_path,
            product_recall_target=product_recall_target,
            n_splits=n_splits,
        )

    return {
        "domain": name,
        "held_out_recall": result["metrics"]["held_out_recall"],
        "held_out_fp_rate": result["metrics"]["held_out_fp_rate"],
        "cache_hit_rate": result["cache_simulation"]["hit_rate"],
        "n_intents": result["index"]["n_intents"],
        "splits_recall_mean": result.get("splits_summary", {}).get("recall", {}).get("mean"),
        "splits_recall_std": result.get("splits_summary", {}).get("recall", {}).get("std"),
        "splits_wrong_rate_mean": result.get("splits_summary", {}).get("wrong_route_rate", {}).get("mean"),
        "passed": result["product_gate"]["passed"],
    }


def _run_with_encoder(
    *,
    encoder,
    name: str,
    source_path: str | Path,
    calibration_path: str | Path,
    heldout_paraphrases_path: str | Path,
    heldout_near_misses_path: str | Path,
    prompts_responses_path: str | Path,
    product_recall_target: float,
    n_splits: int,
    output_path: Path,
) -> None:
    """Run intent cache proof for one domain using a pre-built encoder object.

    This avoids loading SentenceTransformer once per domain when we are
    running in multi-domain mode with a synthetic or pre-loaded encoder.
    """
    import time
    import numpy as np
    from benchmarks.benchmark_intent_cache_product_proof import (
        _load_json,
        _intent_lookup,
        _build_centroids,
        _make_predict,
        _run_single_split,
        _run_multisplit,
        build_intent_cache_report,
    )

    source = _load_json(source_path)
    calibration = _load_json(calibration_path)
    heldout_para = _load_json(heldout_paraphrases_path)["paraphrases"]
    heldout_near = _load_json(heldout_near_misses_path)["near_misses"]
    prompts_data = _load_json(prompts_responses_path)
    text_to_intent = _intent_lookup(source)
    domain = source.get("domain", name)

    d_model = int(encoder.get_sentence_embedding_dimension())

    cal_pairs = [(a, b) for a, b in calibration["paraphrases"]]
    cal_nm = [(a, b) for a, b in calibration["near_misses"]]
    all_para_pairs = cal_pairs + [(a, b) for a, b in heldout_para]
    all_nm_pairs = cal_nm + [(a, b) for a, b in heldout_near]

    train_by_intent: dict[str, set[str]] = {}
    for left, right in cal_pairs:
        intent_id = text_to_intent.get(left)
        if intent_id is None:
            continue
        train_by_intent.setdefault(intent_id, set()).update([left, right])

    stream_prompts = [row["prompt"] for row in prompts_data]

    all_texts = sorted(
        set(t for pair in all_para_pairs for t in pair)
        | set(t for pair in all_nm_pairs for t in pair)
        | set(stream_prompts)
        | {t for texts in train_by_intent.values() for t in texts}
    )

    embeddings = encoder.encode(all_texts, normalize_embeddings=True, batch_size=16)
    emb_by_text = {text: emb for text, emb in zip(all_texts, embeddings)}

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
                for _ in range(100):
                    t0 = time.perf_counter()
                    _ = cm @ probe_emb
                    latency_samples.append(time.perf_counter() - t0)
    mean_latency_ms = (sum(latency_samples) / len(latency_samples)) * 1000 if latency_samples else 0.0

    splits_result = _run_multisplit(
        text_to_intent=text_to_intent,
        all_paraphrase_pairs=all_para_pairs,
        all_near_miss_pairs=all_nm_pairs,
        stream_prompts=stream_prompts,
        emb_by_text=emb_by_text,
        d_model=d_model,
        n_splits=n_splits,
        seed=42,
    )
    splits_summary = {
        "n_splits": n_splits,
        "seed": 42,
        "recall": splits_result["recall"],
        "wrong_route_rate": splits_result["wrong_route_rate"],
        "cache_hit_rate": splits_result["cache_hit_rate"],
    }

    report = build_intent_cache_report(
        model=str(encoder.__class__.__name__),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level multi-domain runner
# ---------------------------------------------------------------------------

def run_multidomain_proof(
    *,
    model: str,
    config: list[dict[str, str]],
    output_path: str | Path,
    product_recall_target: float = 0.8056,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run product proof across all domains in config, write rolled-up report.

    If ``model == "synthetic"`` a fast hash-based encoder is used so the
    entire pipeline can be validated without downloading a real model.

    Returns the rolled-up report dict.
    """
    if not config:
        raise ValueError("config must contain at least one domain spec")

    required_keys = {"name", "source", "calibration", "paraphrases", "near_misses", "prompts_responses"}
    for idx, spec in enumerate(config):
        missing = required_keys - set(spec.keys())
        if missing:
            raise ValueError(f"Domain spec at index {idx} is missing keys: {missing}")

    output = Path(output_path)
    scratch_dir = output.parent / "_multidomain_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Build encoder once and share across all domains
    if model == "synthetic":
        encoder = _SyntheticEncoder(d_model=128)
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model)

    domain_results: list[dict[str, Any]] = []
    for spec in config:
        result = run_domain(
            name=spec["name"],
            model=model,
            source_path=spec["source"],
            calibration_path=spec["calibration"],
            heldout_paraphrases_path=spec["paraphrases"],
            heldout_near_misses_path=spec["near_misses"],
            prompts_responses_path=spec["prompts_responses"],
            product_recall_target=product_recall_target,
            n_splits=n_splits,
            scratch_dir=scratch_dir,
            encoder=encoder,
        )
        domain_results.append(result)

    report = build_multidomain_report(
        model=model,
        domain_results=domain_results,
        product_recall_target=product_recall_target,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-domain LatticeMemory intent cache product proof"
    )
    parser.add_argument("--model", required=True, help="Encoder path or 'synthetic'")
    parser.add_argument(
        "--config",
        default="benchmarks/multidomain_config.json",
        help="Path to JSON list of domain specs",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/multidomain_proof.json",
    )
    parser.add_argument("--product-recall-target", type=float, default=0.8056)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run_multidomain_proof(
        model=args.model,
        config=config,
        output_path=args.output,
        product_recall_target=args.product_recall_target,
        n_splits=args.n_splits,
    )

    print(json.dumps({
        "all_domains_pass": report["product_gate"]["all_domains_pass"],
        "mean_recall": report["aggregate_metrics"]["mean_held_out_recall"],
        "max_fp_rate": report["aggregate_metrics"]["max_held_out_fp_rate"],
        "n_domains": report["aggregate_metrics"]["n_domains"],
    }, indent=2))
    return 0 if report["product_gate"]["all_domains_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
