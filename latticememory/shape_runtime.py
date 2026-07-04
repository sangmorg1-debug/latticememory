"""Experimental same-modality runtime for precomputed shape vectors.

This is not a raw mesh encoder. It provides a small adapter around
``RFSnapLatticeMemory`` for callers that already have fixed-size shape,
geometry, CAD, or product-feature embeddings and want lattice retrieval over
those vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .hamming_router import HammingMatch, HammingRouter
from .memory import MemoryDocument, MemoryQuery, MemoryResult, RFSnapLatticeMemory


@dataclass(frozen=True)
class ShapeIndexResult:
    indexed: int
    total_documents: int
    doc_ids: list[str]


class RFSnapShapeMemory:
    """Lattice retrieval for precomputed same-modality shape feature vectors."""

    def __init__(
        self,
        *,
        d_model: int,
        memory: RFSnapLatticeMemory | None = None,
        hamming_pool_multiplier: int = 10,
        rerank_mode: str = "cosine",
        beam_radius: int = 1,
    ) -> None:
        self.d_model = d_model
        self.memory = memory or RFSnapLatticeMemory(
            d_model=d_model,
            hamming_pool_multiplier=hamming_pool_multiplier,
            rerank_mode=rerank_mode,
            beam_radius=beam_radius,
        )
        if self.memory.d_model != d_model:
            raise ValueError(
                f"memory d_model {self.memory.d_model} does not match runtime d_model {d_model}"
            )

    def add_shapes(
        self,
        shapes: Sequence[np.ndarray | torch.Tensor | list[float]],
        *,
        doc_ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
    ) -> ShapeIndexResult:
        """Index a batch of precomputed shape feature vectors."""

        shape_list = list(shapes)
        if not shape_list:
            return ShapeIndexResult(
                indexed=0,
                total_documents=self.memory.num_documents,
                doc_ids=[],
            )

        ids = (
            list(doc_ids)
            if doc_ids is not None
            else [f"shape-{self.memory.num_documents + index}" for index in range(len(shape_list))]
        )
        if len(ids) != len(shape_list):
            raise ValueError("doc_ids length must match shapes length")

        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in shape_list]
        if len(metadata_list) != len(shape_list):
            raise ValueError("metadatas length must match shapes length")

        documents: list[MemoryDocument] = []
        for vector, doc_id, metadata in zip(shape_list, ids, metadata_list):
            tensor_vector = _normalized_vector(vector, self.d_model)
            enriched = dict(metadata)
            if dataset is not None:
                enriched.setdefault("dataset", dataset)
            if index_id is not None:
                enriched.setdefault("index_id", index_id)
            documents.append(
                MemoryDocument(
                    doc_id=doc_id,
                    text=doc_id,
                    embedding=tensor_vector,
                    metadata=enriched,
                )
            )

        self.memory.add_documents(documents)
        return ShapeIndexResult(
            indexed=len(documents),
            total_documents=self.memory.num_documents,
            doc_ids=ids,
        )

    def retrieve_shape(
        self,
        shape_vector: np.ndarray | torch.Tensor | list[float],
        *,
        top_k: int = 5,
        request_id: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
        quality_tags: list[str] | None = None,
    ) -> MemoryResult:
        """Retrieve nearest indexed shapes for a precomputed shape vector."""

        tensor_vector = _normalized_vector(shape_vector, self.d_model)
        return self.memory.retrieve(
            MemoryQuery(
                text="query_shape",
                embedding=tensor_vector,
                top_k=top_k,
                request_id=request_id,
                product=product,
                dataset=dataset,
                index_id=index_id,
                quality_tags=list(quality_tags or []),
            )
        )


class ShapeHammingRouter(HammingRouter):
    """E8 Hamming nearest-neighbor router for precomputed shape vectors."""

    def __init__(self, d_model: int, threshold: int = 70) -> None:
        super().__init__(encoder=None, d_model=d_model, threshold=threshold)

    def add_vector(self, key_vector: np.ndarray | torch.Tensor | list[float], value: Any) -> bytes:
        """Quantize and store a vector key with ``value``."""

        vector = _normalized_vector(key_vector, self._d_model)
        key_arr = self._emb_to_key_arr(vector.detach().cpu().numpy())
        raw_key = key_arr.tobytes()
        if raw_key not in self._key_set:
            self._key_set.add(raw_key)
            self._keys.append(key_arr)
            self._values.append(value)
            self._packed_matrix = None
        return raw_key

    def lookup_vector(
        self,
        query_vector: np.ndarray | torch.Tensor | list[float],
        threshold: int | None = None,
    ) -> HammingMatch | None:
        """Return the nearest stored vector if it is within threshold."""

        if not self._keys:
            return None
        vector = _normalized_vector(query_vector, self._d_model)
        query_arr = self._emb_to_key_arr(vector.detach().cpu().numpy())
        return self._nearest(query_arr, threshold=threshold)


def _normalized_vector(
    vector: np.ndarray | torch.Tensor | list[float],
    d_model: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(vector, dtype=torch.float32)
    if tensor.dim() != 1 or tensor.numel() != d_model:
        raise ValueError(f"shape vector must be 1-D with dim {d_model}, got shape {tuple(tensor.shape)}")
    norm = torch.linalg.norm(tensor)
    if norm > 0.0:
        tensor = tensor / norm
    return tensor
