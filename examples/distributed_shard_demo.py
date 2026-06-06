"""Phase 5 demo: deterministic semantic sharding for distributed training."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.pipeline import LatticeDataPipeline


def _unit_vectors(count: int, d_model: int) -> np.ndarray:
    rng = np.random.default_rng(321)
    vectors = rng.standard_normal((count, d_model)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def run_demo() -> None:
    d_model = 384
    num_shards = 8
    embeddings = _unit_vectors(1_000, d_model)
    duplicate_pairs = [(25, 900), (100, 901), (250, 902), (500, 903)]
    for source, duplicate in duplicate_pairs:
        embeddings[duplicate] = embeddings[source]

    pipeline = LatticeDataPipeline(d_model=d_model)
    first = pipeline.assign_shards(embeddings, num_shards=num_shards)
    second = pipeline.assign_shards(embeddings, num_shards=num_shards)

    duplicate_shards_match = all(first[left] == first[right] for left, right in duplicate_pairs)
    shard_counts = np.bincount(first, minlength=num_shards)

    print("--- Phase 5: Distributed Shard Demo ---")
    print(f"Items: {len(embeddings):,}")
    print(f"Shards: {num_shards}")
    print(f"Identical sharding on rerun: {np.array_equal(first, second)}")
    print(f"Semantic duplicates stay on same shard: {duplicate_shards_match}")
    print(f"Shard counts: {shard_counts.tolist()}")


if __name__ == "__main__":
    run_demo()
