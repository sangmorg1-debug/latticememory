"""Phase 5 demo: cross-modal caption quality filtering with E8 keys."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.pipeline import LatticeDataPipeline


def _unit_vectors(count: int, d_model: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    vectors = rng.standard_normal((count, d_model)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def run_demo() -> None:
    d_model = 384
    pair_count = 100
    mismatch_count = 20
    text_embeddings = _unit_vectors(pair_count, d_model)
    image_embeddings = text_embeddings + 0.01

    captions = text_embeddings.copy()
    mismatch_indices = list(range(pair_count - mismatch_count, pair_count))
    for idx in mismatch_indices:
        captions[idx] = text_embeddings[(idx + 17) % pair_count]

    pipeline = LatticeDataPipeline(d_model=d_model)
    before = pipeline.filter_caption_quality(images=image_embeddings, captions=captions)

    pipeline.fit_cross_modal_adapter(
        image_embeddings=torch.as_tensor(image_embeddings[: pair_count - mismatch_count]),
        text_embeddings=torch.as_tensor(text_embeddings[: pair_count - mismatch_count]),
    )
    after = pipeline.filter_caption_quality(images=image_embeddings, captions=captions)

    true_mismatches = set(mismatch_indices)
    detected_mismatches = set(after["mismatch_pairs"])
    detection_rate = len(true_mismatches & detected_mismatches) / len(true_mismatches)

    print("--- Phase 5: Caption Quality Filter Demo ---")
    print(f"Pairs: {pair_count}")
    print(f"Injected mismatches: {mismatch_count}")
    print(f"Before adapter clean pairs: {len(before['clean_pairs'])}")
    print(f"After adapter detected mismatches: {len(after['mismatch_pairs'])}")
    print(f"Mismatch detection rate: {detection_rate * 100:.1f}%")
    print(f"Mismatch rate reported: {after['mismatch_rate'] * 100:.1f}%")


if __name__ == "__main__":
    run_demo()
