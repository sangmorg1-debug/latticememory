from __future__ import annotations

import numpy as np

from latticememory.shape_runtime import RFSnapShapeMemory, ShapeHammingRouter


def _shape_embedding(d_model: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(d_model).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-9
    return vector


def test_shape_memory_indexes_and_retrieves_precomputed_vectors() -> None:
    runtime = RFSnapShapeMemory(d_model=384)
    shape_embeddings = [_shape_embedding(384, seed) for seed in range(5)]
    doc_ids = [f"mesh-{index}" for index in range(5)]
    metadatas = [{"category": "chair" if index % 2 == 0 else "table"} for index in range(5)]

    result = runtime.add_shapes(
        shape_embeddings,
        doc_ids=doc_ids,
        metadatas=metadatas,
        dataset="synthetic-shapes",
        index_id="shape-smoke",
    )

    assert result.indexed == 5
    assert result.total_documents == 5
    assert result.doc_ids == doc_ids

    retrieved = runtime.retrieve_shape(shape_embeddings[2], top_k=2)
    assert retrieved.hits
    assert retrieved.hits[0].doc_id == "mesh-2"
    assert retrieved.hits[0].metadata["category"] == "chair"
    assert retrieved.hits[0].metadata["dataset"] == "synthetic-shapes"


def test_shape_hamming_router_matches_nearby_vector_and_rejects_far_vector() -> None:
    router = ShapeHammingRouter(d_model=384, threshold=15)
    anchor = _shape_embedding(384, 42)
    rng = np.random.default_rng(99)
    nearby = anchor + rng.normal(scale=0.001, size=384).astype(np.float32)
    nearby /= np.linalg.norm(nearby) + 1e-9
    far = _shape_embedding(384, 100)

    raw_key = router.add_vector(anchor, value="Chair_Model_3D_V1")
    assert len(raw_key) == 48

    exact = router.lookup_vector(anchor)
    assert exact is not None
    assert exact.value == "Chair_Model_3D_V1"
    assert exact.hamming_distance == 0

    approximate = router.lookup_vector(nearby)
    assert approximate is not None
    assert approximate.value == "Chair_Model_3D_V1"
    assert approximate.hamming_distance <= 15

    assert router.lookup_vector(far) is None
