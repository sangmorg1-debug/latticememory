from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from latticememory.dedup import LatticeDedup

from benchmarks.common import load_text_encoder, write_json_result


def build_corpus(n_docs: int, duplicate_rate: float) -> list[str]:
    duplicate_count = int(round(n_docs * duplicate_rate))
    unique_count = n_docs - duplicate_count
    if unique_count < 1:
        raise ValueError("duplicate_rate leaves no unique documents")
    corpus = [f"training example {idx:06d}: concept {idx % 997}" for idx in range(unique_count)]
    for idx in range(duplicate_count):
        source = idx % unique_count
        corpus.append(f"{corpus[source]} :: duplicate variant {idx}")
    return corpus


def run_benchmark(
    *,
    n_docs: int = 100_000,
    duplicate_rate: float = 0.15,
    d_model: int | None = 1024,
    model: str = "synthetic",
    brute_force_limit: int = 2_000,
    output_path: str | Path | None = None,
) -> dict:
    if n_docs < 1:
        raise ValueError("n_docs must be positive")
    if not 0 <= duplicate_rate < 1:
        raise ValueError("duplicate_rate must be in [0, 1)")

    corpus = build_corpus(n_docs, duplicate_rate)
    encoder, runtime_dim = load_text_encoder(model, d_model)
    deduper = LatticeDedup(d_model=runtime_dim)
    deduper.encoder = encoder

    start = time.perf_counter()
    dedup_result = deduper.deduplicate(corpus)
    elapsed_seconds = time.perf_counter() - start

    duplicate_count = int(dedup_result["duplicate_count"])
    unique_count = len(dedup_result["unique_documents"])
    brute_force = _measure_bruteforce_baseline(
        corpus=corpus,
        encoder=encoder,
        batch_size=64,
        limit=brute_force_limit,
    )
    # Matched-size E8 run so the speedup ratio is apples-to-apples.
    e8_subset = _measure_e8_subset(
        corpus=corpus,
        encoder=encoder,
        d_model=runtime_dim,
        limit=brute_force_limit,
    )
    speedup: float | None = None
    if (
        brute_force["elapsed_seconds"] is not None
        and e8_subset["elapsed_seconds"] is not None
        and e8_subset["elapsed_seconds"] > 0
    ):
        speedup = brute_force["elapsed_seconds"] / e8_subset["elapsed_seconds"]
    result = {
        "benchmark": "dedup",
        "model": model,
        "n_docs": int(n_docs),
        "d_model": int(runtime_dim),
        "duplicate_count": duplicate_count,
        "unique_count": unique_count,
        "dedup_rate": round(duplicate_count / n_docs, 6),
        "e8_elapsed_seconds": elapsed_seconds,
        "e8_docs_per_second": n_docs / max(elapsed_seconds, 1e-12),
        "bruteforce_pairwise_comparisons": int(n_docs * (n_docs - 1) // 2),
        "bruteforce_elapsed_seconds": brute_force["elapsed_seconds"],
        "bruteforce_docs_per_second": brute_force["docs_per_second"],
        "bruteforce_documents_measured": brute_force["documents_measured"],
        "bruteforce_duplicate_pairs": brute_force["duplicate_pairs"],
        "speedup_vs_bruteforce": speedup,
        "speedup_docs_measured": e8_subset["documents_measured"],
    }
    return write_json_result(result, output_path)


def _measure_bruteforce_baseline(*, corpus: list[str], encoder, batch_size: int, limit: int) -> dict:
    if limit <= 1:
        return {
            "elapsed_seconds": None,
            "docs_per_second": None,
            "documents_measured": 0,
            "duplicate_pairs": None,
        }
    measured = corpus[: min(len(corpus), limit)]
    start = time.perf_counter()
    embeddings = np.asarray(encoder.encode(measured, batch_size=batch_size), dtype=np.float32)
    duplicate_pairs = 0
    for left in range(len(measured)):
        left_vec = embeddings[left]
        for right in range(left + 1, len(measured)):
            if float(left_vec @ embeddings[right]) >= 0.999999:
                duplicate_pairs += 1
    elapsed = time.perf_counter() - start
    return {
        "elapsed_seconds": elapsed,
        "docs_per_second": len(measured) / max(elapsed, 1e-12),
        "documents_measured": len(measured),
        "duplicate_pairs": duplicate_pairs,
    }


def _measure_e8_subset(*, corpus: list[str], encoder, d_model: int, limit: int) -> dict:
    if limit <= 1:
        return {"elapsed_seconds": None, "documents_measured": 0}
    measured = corpus[: min(len(corpus), limit)]
    deduper = LatticeDedup(d_model=d_model)
    deduper.encoder = encoder
    start = time.perf_counter()
    deduper.deduplicate(measured)
    elapsed = time.perf_counter() - start
    return {"elapsed_seconds": elapsed, "documents_measured": len(measured)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LatticeMemory O(N) dedup throughput.")
    parser.add_argument("--n-docs", type=int, default=100_000)
    parser.add_argument("--duplicate-rate", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--model", default="synthetic")
    parser.add_argument("--brute-force-limit", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/dedup.json"))
    args = parser.parse_args()
    result = run_benchmark(
        n_docs=args.n_docs,
        duplicate_rate=args.duplicate_rate,
        d_model=args.d_model,
        model=args.model,
        brute_force_limit=args.brute_force_limit,
        output_path=args.output,
    )
    print(f"Wrote {result['benchmark']} benchmark to {args.output}")


if __name__ == "__main__":
    main()
