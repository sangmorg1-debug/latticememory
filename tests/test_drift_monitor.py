from __future__ import annotations

import hashlib

import numpy as np

from latticememory.drift_monitor import compare_cache_snapshot, create_cache_snapshot
from latticememory.memory import RFSnapLatticeMemory
from latticememory.semantic_cache import RFSnapSemanticCache
from latticememory.text_runtime import RFSnapTextMemory


class PerturbedFakeEncoder:
    def __init__(
        self,
        d_model: int = 384,
        noise: float = 0.0,
        model_id: str = "fake-model",
    ) -> None:
        self.d_model = d_model
        self.noise = noise
        self.model_id = model_id

    def encode(self, sentences, **kwargs):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.d_model).astype(np.float32)
            if self.noise:
                vector += rng.normal(scale=self.noise, size=self.d_model).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        if single:
            return vectors[0]
        return np.stack(vectors)


def _create_cache(encoder: PerturbedFakeEncoder) -> RFSnapSemanticCache:
    memory = RFSnapLatticeMemory(d_model=encoder.d_model)
    runtime = RFSnapTextMemory(
        encoder=encoder,
        d_model=encoder.d_model,
        memory=memory,
        model_id=encoder.model_id,
    )
    return RFSnapSemanticCache(runtime=runtime)


def test_snapshot_creation_and_comparison_no_drift() -> None:
    cache = _create_cache(PerturbedFakeEncoder(noise=0.0))
    reference_texts = [
        "First stable reference sentence.",
        "Second semantic anchor statement.",
        "Third coordinate validation probe.",
    ]

    snapshot = create_cache_snapshot(cache, reference_texts)
    assert snapshot["model_id"] == "fake-model"
    assert snapshot["d_model"] == 384
    assert snapshot["reference_count"] == 3
    assert len(snapshot["snapshots"]) == 3

    result = compare_cache_snapshot(cache, snapshot)
    assert result["mean_hamming_distance"] == 0.0
    assert result["exact_match_ratio"] == 1.0
    assert result["drift_percentage"] == 0.0
    assert result["needs_reindexing"] is False


def test_snapshot_comparison_flags_model_change() -> None:
    cache_v1 = _create_cache(PerturbedFakeEncoder(model_id="fake-model-v1"))
    snapshot = create_cache_snapshot(cache_v1, ["Sentence to monitor model swap."])

    cache_v2 = _create_cache(PerturbedFakeEncoder(model_id="fake-model-v2"))
    result = compare_cache_snapshot(cache_v2, snapshot)

    assert result["needs_reindexing"] is True
    assert "Model changed" in result["details"]


def test_snapshot_comparison_flags_coordinate_drift() -> None:
    cache_stable = _create_cache(PerturbedFakeEncoder(noise=0.0))
    snapshot = create_cache_snapshot(
        cache_stable,
        ["Sentence one", "Sentence two", "Sentence three", "Sentence four", "Sentence five"],
    )

    cache_perturbed = _create_cache(PerturbedFakeEncoder(noise=0.5))
    result = compare_cache_snapshot(cache_perturbed, snapshot, drift_threshold=0.1)

    assert result["mean_hamming_distance"] > 0.0
    assert result["drift_percentage"] > 0.0
    assert result["needs_reindexing"] is True
