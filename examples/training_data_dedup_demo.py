"""Phase 5 demo: O(N) training-corpus deduplication with E8 addresses."""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.pipeline import LatticeDataPipeline


class CanonicalFakeEncoder:
    """Hashes duplicate variants to the same deterministic embedding."""

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        vectors = []
        for sentence in sentences:
            canonical = str(sentence).split(" :: duplicate variant")[0]
            seed = int(hashlib.md5(canonical.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.d_model).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        return np.stack(vectors)


def build_corpus(n_docs: int = 10_000, duplicate_rate: float = 0.15) -> list[str]:
    duplicate_count = int(n_docs * duplicate_rate)
    unique_count = n_docs - duplicate_count
    corpus = [f"training example {idx:05d}: concept {idx % 250}" for idx in range(unique_count)]
    for idx in range(duplicate_count):
        source = idx % unique_count
        corpus.append(f"{corpus[source]} :: duplicate variant {idx}")
    return corpus


def run_demo() -> None:
    d_model = 384
    corpus = build_corpus()
    pipeline = LatticeDataPipeline(text_encoder=CanonicalFakeEncoder(d_model), d_model=d_model)

    start = time.perf_counter()
    result = pipeline.deduplicate_text(corpus)
    elapsed_ms = (time.perf_counter() - start) * 1000

    n = len(corpus)
    pairwise_comparisons = n * (n - 1) // 2
    print("--- Phase 5: Training Data Dedup Demo ---")
    print(f"Documents: {n:,}")
    print(f"E8 O(N) elapsed: {elapsed_ms:.2f} ms")
    print(f"Simulated O(N^2) cosine comparisons avoided: {pairwise_comparisons:,}")
    print(f"Unique documents: {len(result['unique_documents']):,}")
    print(f"Duplicates removed: {result['duplicate_count']:,}")
    print(f"Compression ratio: {result['compression_ratio'] * 100:.1f}%")

    print("\nExample duplicate clusters:")
    for address, duplicates in list(result["duplicates"].items())[:3]:
        print(f"  E8_{address[:16]}... -> {duplicates[:2]}")


if __name__ == "__main__":
    run_demo()
