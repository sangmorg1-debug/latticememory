from __future__ import annotations

import numpy as np
import pytest
import torch

from latticememory.pipeline import LatticeDataPipeline
from tests.test_lattice_index import FakeEncoder


def _unit_vectors(count: int, d_model: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((count, d_model)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def test_deduplicate_text_wraps_lattice_dedup_with_injected_encoder():
    encoder = FakeEncoder(d_model=384)
    pipeline = LatticeDataPipeline(text_encoder=encoder, d_model=384)

    result = pipeline.deduplicate_text(
        [
            "same training example",
            "same training example",
            "different training example",
        ]
    )

    assert result["unique_documents"] == [
        "same training example",
        "different training example",
    ]
    assert result["duplicate_count"] == 1
    assert result["compression_ratio"] == pytest.approx(1 / 3)


def test_filter_caption_quality_flags_pairs_with_different_e8_keys():
    embeddings = _unit_vectors(4, 384)
    captions = embeddings.copy()
    captions[3] = embeddings[0]
    pipeline = LatticeDataPipeline(d_model=384)

    report = pipeline.filter_caption_quality(images=embeddings, captions=captions)

    assert report["clean_pairs"] == [0, 1, 2]
    assert report["mismatch_pairs"] == [3]
    assert report["mismatch_rate"] == pytest.approx(0.25)


def test_fit_cross_modal_adapter_maps_image_embeddings_before_quality_filter():
    text_embeddings = torch.as_tensor(_unit_vectors(5, 384))
    image_embeddings = text_embeddings + 0.01
    pipeline = LatticeDataPipeline(d_model=384)

    pipeline.fit_cross_modal_adapter(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )
    report = pipeline.filter_caption_quality(
        images=image_embeddings.numpy(),
        captions=text_embeddings.numpy(),
    )

    assert report["clean_pairs"] == [0, 1, 2, 3, 4]
    assert report["mismatch_pairs"] == []
    assert report["mismatch_rate"] == 0.0


def test_assign_shards_is_deterministic_and_keeps_duplicate_embeddings_together():
    embeddings = _unit_vectors(6, 384)
    embeddings[5] = embeddings[2]
    pipeline = LatticeDataPipeline(d_model=384)

    first = pipeline.assign_shards(embeddings, num_shards=8)
    second = pipeline.assign_shards(embeddings, num_shards=8)

    assert first.dtype.kind in {"i", "u"}
    assert first.tolist() == second.tolist()
    assert first[5] == first[2]
    assert first.min() >= 0
    assert first.max() < 8


def test_assign_shards_rejects_invalid_shard_count():
    pipeline = LatticeDataPipeline(d_model=384)

    with pytest.raises(ValueError, match="num_shards must be positive"):
        pipeline.assign_shards(_unit_vectors(1, 384), num_shards=0)
