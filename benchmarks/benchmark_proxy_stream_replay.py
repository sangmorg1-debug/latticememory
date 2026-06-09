"""Proxy stream replay benchmark — measures end-to-end HammingRouter cache performance.

Replays a prompt stream through the HammingRouter (the same engine used by LatticeLLMProxy)
and records:
  - Cache hit rate (shadow-mode: would-be hits if the cache were live)
  - Latency p50/p95 for the key lookup path (no encode; pre-computed keys)
  - Encode latency p50/p95 (embed + key)
  - Index storage bytes
  - Hamming distance distribution of hits

This is a pure-Python benchmark with no FastAPI/HTTP overhead — it tests the
semantic-cache engine layer directly, which is what matters for latency SLAs.

Usage (synthetic encoder — fast, no model download):
    python -m benchmarks.benchmark_proxy_stream_replay \\
        --model synthetic \\
        --prompts benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json \\
        --calibration benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json \\
        --output benchmarks/results/proxy_stream_replay.json

Usage (trained checkpoint):
    python -m benchmarks.benchmark_proxy_stream_replay \\
        --model benchmarks/results/snap_product_gate_hard_symmetric_8ep/best_snap_encoder \\
        --prompts benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json \\
        --calibration benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json \\
        --output benchmarks/results/proxy_stream_replay.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic encoder
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
# Report builder (pure data, no I/O)
# ---------------------------------------------------------------------------

def build_replay_report(
    *,
    model: str,
    d_model: int,
    hamming_threshold: int,
    total_prompts: int,
    cache_hits: int,
    true_misses: int,
    hamming_distances: list[int],
    lookup_latencies_ms: list[float],
    encode_latencies_ms: list[float],
    n_keys_in_index: int,
    cost_per_query_usd: float = 0.00025,
) -> dict[str, Any]:
    """Build a structured replay report from raw metrics.

    All latency lists may be empty (e.g. when total_prompts is very small).
    """
    hit_rate = cache_hits / total_prompts if total_prompts else 0.0
    savings_usd = cache_hits * cost_per_query_usd

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        idx = min(int(len(vals) * p), len(vals) - 1)
        return round(sorted(vals)[idx], 4)

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _std(vals: list[float]) -> float:
        if not vals:
            return 0.0
        m = sum(vals) / len(vals)
        return round(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)), 4)

    hist: dict[str, int] = {}
    for d in hamming_distances:
        bucket = f"{(d // 10) * 10}-{(d // 10) * 10 + 9}"
        hist[bucket] = hist.get(bucket, 0) + 1

    key_bytes_per_entry = d_model // 8
    total_key_bytes = n_keys_in_index * key_bytes_per_entry
    float32_bytes_equivalent = n_keys_in_index * d_model * 4

    return {
        "artifact_type": "latticememory_proxy_stream_replay",
        "artifact_version": 1,
        "model": model,
        "d_model": d_model,
        "hamming_threshold": hamming_threshold,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "stream": {
            "total_prompts": total_prompts,
            "cache_hits": cache_hits,
            "true_misses": true_misses,
            "hit_rate": round(hit_rate, 4),
            "cost_per_query_usd": cost_per_query_usd,
            "would_be_savings_usd": round(savings_usd, 6),
            "would_be_savings_pct": round(hit_rate * 100, 1),
        },
        "lookup_latency_ms": {
            "n": len(lookup_latencies_ms),
            "mean": _mean(lookup_latencies_ms),
            "std": _std(lookup_latencies_ms),
            "p50": _pct(lookup_latencies_ms, 0.50),
            "p95": _pct(lookup_latencies_ms, 0.95),
            "p99": _pct(lookup_latencies_ms, 0.99),
        },
        "encode_latency_ms": {
            "n": len(encode_latencies_ms),
            "mean": _mean(encode_latencies_ms),
            "std": _std(encode_latencies_ms),
            "p50": _pct(encode_latencies_ms, 0.50),
            "p95": _pct(encode_latencies_ms, 0.95),
        },
        "hamming_distribution": {
            "n": len(hamming_distances),
            "mean": round(sum(hamming_distances) / len(hamming_distances), 2) if hamming_distances else 0.0,
            "exact_hits": hamming_distances.count(0),
            "histogram": hist,
        },
        "index": {
            "n_keys": n_keys_in_index,
            "key_bytes_per_entry": key_bytes_per_entry,
            "total_key_bytes": total_key_bytes,
            "float32_bytes_equivalent": float32_bytes_equivalent,
            "compression_vs_float32_keys_only": round(float32_bytes_equivalent / total_key_bytes, 2)
            if total_key_bytes > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_proxy_stream_replay(
    *,
    model: str,
    prompts_responses: list[dict[str, Any]],
    calibration_data: dict[str, Any],
    hamming_threshold: int = 70,
    cost_per_query_usd: float = 0.00025,
    output_path: str | Path,
    latency_probe_n: int = 200,
) -> dict[str, Any]:
    """Replay a prompt stream through HammingRouter and record cache metrics.

    Parameters
    ----------
    model:
        Path to a SentenceTransformer checkpoint, or ``"synthetic"`` for a
        hash-based encoder (fast, no real model download needed).
    prompts_responses:
        List of dicts with at least a ``"prompt"`` key.
    calibration_data:
        Dict with ``"paraphrases"`` key (list of [a, b] pairs).
        The canonical side of each pair is pre-loaded into the router.
    hamming_threshold:
        Hamming distance at or below which a lookup is considered a hit.
    cost_per_query_usd:
        Assumed LLM API cost per query (for savings estimates).
    output_path:
        Where to write the JSON report.
    latency_probe_n:
        Number of repeated lookups for latency measurement.
    """
    import numpy as np
    import torch

    if model == "synthetic":
        encoder = _SyntheticEncoder(d_model=128)
        d_model = 128
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model)
        d_model = int(encoder.get_sentence_embedding_dimension())

    from latticememory.hamming_router import HammingRouter
    from latticememory.rag.e8_retriever import E8LatticeDB

    router = HammingRouter(encoder=encoder, d_model=d_model, threshold=hamming_threshold)
    lattice = E8LatticeDB(d_model=d_model)

    # Pre-encode all unique texts in a single batch (fast)
    all_prompts = [row["prompt"] for row in prompts_responses if "prompt" in row]
    cal_pairs = calibration_data.get("paraphrases", [])
    canonical_texts = [pair[0] for pair in cal_pairs if pair]
    all_unique = sorted(set(all_prompts + canonical_texts))

    encode_start = time.perf_counter()
    all_embs = encoder.encode(all_unique, normalize_embeddings=True, batch_size=32)
    encode_total_ms = (time.perf_counter() - encode_start) * 1000
    emb_by_text = {t: e for t, e in zip(all_unique, all_embs)}

    def _to_key(text: str) -> bytes:
        emb = emb_by_text[text]
        indices = lattice._quantize_to_indices(torch.tensor(emb, dtype=torch.float32))
        return bytes(np.frombuffer(indices, dtype=np.uint8))

    # Pre-load canonical texts into router
    canonical_keys: dict[bytes, str] = {}
    for text in canonical_texts:
        if text in emb_by_text:
            key = _to_key(text)
            router.add_from_key(key, value=text)
            canonical_keys[key] = text

    # Latency probe on lookup-only path (no encode, key already computed)
    lookup_latencies_ms: list[float] = []
    if all_prompts and canonical_keys:
        probe_text = all_prompts[0]
        if probe_text in emb_by_text:
            probe_key = _to_key(probe_text)
            for _ in range(latency_probe_n):
                t0 = time.perf_counter()
                router.lookup_key(probe_key, threshold=hamming_threshold)
                lookup_latencies_ms.append((time.perf_counter() - t0) * 1000)

    # Per-query encode latency (sampling every 5th prompt to keep test fast)
    encode_latencies_ms: list[float] = []
    sample_texts = all_prompts[::5][:40]
    for text in sample_texts:
        t0 = time.perf_counter()
        encoder.encode([text], normalize_embeddings=True)
        encode_latencies_ms.append((time.perf_counter() - t0) * 1000)

    # Stream replay
    cache_hits = 0
    true_misses = 0
    hamming_distances: list[int] = []
    live_keys: dict[bytes, str] = dict(canonical_keys)

    for row in prompts_responses:
        prompt = row.get("prompt", "")
        if not prompt or prompt not in emb_by_text:
            continue

        key = _to_key(prompt)

        if key in live_keys:
            # Exact E8 key match
            cache_hits += 1
            hamming_distances.append(0)
        else:
            match = router.lookup_key(key, threshold=hamming_threshold)
            if match is not None:
                cache_hits += 1
                hamming_distances.append(match.hamming_distance)
            else:
                true_misses += 1
                live_keys[key] = prompt
                router.add_from_key(key, value=prompt)

    n_keys_in_index = len(live_keys)

    report = build_replay_report(
        model=model,
        d_model=d_model,
        hamming_threshold=hamming_threshold,
        total_prompts=cache_hits + true_misses,
        cache_hits=cache_hits,
        true_misses=true_misses,
        hamming_distances=hamming_distances,
        lookup_latencies_ms=lookup_latencies_ms,
        encode_latencies_ms=encode_latencies_ms,
        n_keys_in_index=n_keys_in_index,
        cost_per_query_usd=cost_per_query_usd,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Proxy stream replay benchmark — measures HammingRouter cache hit rate and latency"
    )
    parser.add_argument("--model", required=True, help="Encoder path or 'synthetic'")
    parser.add_argument(
        "--prompts",
        default="benchmarks/demo_data/hard_near_miss_challenge/prompts_responses.json",
    )
    parser.add_argument(
        "--calibration",
        default="benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json",
    )
    parser.add_argument("--hamming-threshold", type=int, default=70)
    parser.add_argument("--cost-per-query-usd", type=float, default=0.00025)
    parser.add_argument("--output", default="benchmarks/results/proxy_stream_replay.json")
    parser.add_argument("--latency-probe-n", type=int, default=200)
    args = parser.parse_args()

    prompts_responses = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    calibration_data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))

    report = run_proxy_stream_replay(
        model=args.model,
        prompts_responses=prompts_responses,
        calibration_data=calibration_data,
        hamming_threshold=args.hamming_threshold,
        cost_per_query_usd=args.cost_per_query_usd,
        output_path=args.output,
        latency_probe_n=args.latency_probe_n,
    )

    print(json.dumps({
        "hit_rate": report["stream"]["hit_rate"],
        "would_be_savings_usd": report["stream"]["would_be_savings_usd"],
        "lookup_p50_ms": report["lookup_latency_ms"]["p50"],
        "lookup_p95_ms": report["lookup_latency_ms"]["p95"],
        "total_key_bytes": report["index"]["total_key_bytes"],
        "compression_vs_float32": report["index"]["compression_vs_float32_keys_only"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
