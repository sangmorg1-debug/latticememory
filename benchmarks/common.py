from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class SyntheticTextEncoder:
    """Deterministic offline encoder for benchmark smoke tests and demos."""

    def __init__(self, d_model: int = 384, *, canonical_suffix: str = " :: duplicate variant"):
        self.d_model = int(d_model)
        self.canonical_suffix = canonical_suffix

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        inputs = [sentences] if isinstance(sentences, str) else list(sentences)
        vectors = []
        for sentence in inputs:
            canonical = str(sentence).split(self.canonical_suffix)[0]
            seed = int(hashlib.md5(canonical.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.d_model).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        stacked = np.stack(vectors)
        return stacked[0] if isinstance(sentences, str) else stacked


def load_text_encoder(model: str, d_model: int | None, *, batch_size: int = 64, device: str | None = None):
    if model == "synthetic":
        if d_model is None:
            d_model = 384
        return SyntheticTextEncoder(d_model=d_model), int(d_model)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model, device=device)
    detected = int(encoder.get_embedding_dimension() or 0)
    if detected <= 0:
        probe = encoder.encode(["dimension probe"], batch_size=batch_size)
        detected = int(np.asarray(probe).shape[-1])
    runtime_dim = int(d_model or detected)
    if runtime_dim != detected:
        raise ValueError(f"requested d_model {runtime_dim} does not match model dimension {detected}")
    return encoder, runtime_dim


def generate_unit_embeddings(n_items: int, d_model: int, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_items, d_model)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_json_result(result: dict[str, Any], output_path: str | Path | None) -> dict[str, Any]:
    if output_path is None:
        return result
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
