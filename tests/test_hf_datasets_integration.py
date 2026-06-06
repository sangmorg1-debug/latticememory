from __future__ import annotations

import numpy as np

from latticememory.integrations.hf_datasets import LatticeDatasetFilter
from tests.test_lattice_index import FakeEncoder


def _unit_vectors(count: int, d_model: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((count, d_model)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def test_deduplicate_streaming_yields_first_row_per_lattice_key():
    rows = [
        {"id": 1, "caption": "duplicate caption"},
        {"id": 2, "caption": "duplicate caption"},
        {"id": 3, "caption": "unique caption"},
    ]
    dataset_filter = LatticeDatasetFilter(encoder=FakeEncoder(384), d_model=384)

    clean_rows = list(dataset_filter.deduplicate_streaming(rows, text_column="caption", batch_size=2))

    assert clean_rows == [
        {"id": 1, "caption": "duplicate caption"},
        {"id": 3, "caption": "unique caption"},
    ]


def test_filter_caption_quality_yields_only_rows_with_matching_embedding_keys():
    embeddings = _unit_vectors(3, 384)
    rows = [
        {"id": 1, "image_embedding": embeddings[0], "caption_embedding": embeddings[0]},
        {"id": 2, "image_embedding": embeddings[1], "caption_embedding": embeddings[2]},
        {"id": 3, "image_embedding": embeddings[2], "caption_embedding": embeddings[2]},
    ]
    dataset_filter = LatticeDatasetFilter(d_model=384)

    clean_rows = list(
        dataset_filter.filter_caption_quality(
            rows,
            image_column="image_embedding",
            caption_column="caption_embedding",
            batch_size=2,
        )
    )

    assert [row["id"] for row in clean_rows] == [1, 3]
