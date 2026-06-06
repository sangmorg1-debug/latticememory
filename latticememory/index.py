"""LatticeIndex — public product API wrapping RFSnapTextMemory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from latticememory.text_runtime import RFSnapTextMemory

DEFAULT_MODEL = "dfrokido/bge-large-e8-snap"


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    address: str
    doc_id: str
    metadata: dict
    retrieval_path: str


@dataclass(frozen=True)
class LatticeStats:
    docs: int
    index_size_mb: float
    compression_vs_float32: float
    exact_hit_rate: float | None = None


class LatticeIndex:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 64, beam_radius: int = 1):
        from sentence_transformers import SentenceTransformer
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(model, device=device)
        d_model = int(encoder.get_embedding_dimension() or 0)
        if d_model <= 0:
            import numpy as np
            probe = encoder.encode(["dimension probe"])
            d_model = int(np.asarray(probe).shape[-1])
        self._init_with_encoder(encoder, d_model=d_model, batch_size=batch_size, beam_radius=beam_radius)

    def _init_with_encoder(self, encoder, *, d_model: int, batch_size: int = 64, beam_radius: int = 1) -> None:
        from latticememory.memory import DenseVectorFallback
        self._d_model = d_model
        fallback = DenseVectorFallback(d_model=d_model)
        self._runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, batch_size=batch_size, fallback=fallback, beam_radius=beam_radius)
        self._total_queries: int = 0
        self._exact_hits: int = 0

    def add(self, texts: Sequence[str], doc_ids: Sequence[str] | None = None, metadatas: Sequence[dict] | None = None) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []
        embs = self._runtime._encode_texts(text_list)
        addresses = [self._runtime.memory.lattice_key_for(embs[i]).hex() for i in range(len(text_list))]
        self._runtime.add_texts(text_list, doc_ids=doc_ids, metadatas=metadatas)
        return addresses

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if self._runtime.memory.num_documents == 0:
            return []
        result = self._runtime.retrieve_text(query, top_k=top_k)
        self._total_queries += 1
        if result.path == "lattice_exact":
            self._exact_hits += 1
        hits = []
        for h in result.hits:
            emb = self._runtime._encode_texts([h.text])[0]
            address = self._runtime.memory.lattice_key_for(emb).hex()
            hits.append(SearchResult(text=h.text, score=h.score, address=address, doc_id=h.doc_id, metadata=dict(h.metadata), retrieval_path=result.path))
        return hits

    def snap(self, text: str) -> str:
        emb = self._runtime._encode_texts([text])[0]
        return self._runtime.memory.lattice_key_for(emb).hex()

    def stats(self) -> LatticeStats:
        docs = self._runtime.memory.num_documents
        address_bytes = docs * (self._d_model // 8) * 3
        float32_bytes = docs * self._d_model * 4
        index_size_mb = address_bytes / (1024 * 1024)
        compression = (float32_bytes / address_bytes) if address_bytes > 0 else 0.0
        exact_hit_rate = (self._exact_hits / self._total_queries if self._total_queries > 0 else None)
        return LatticeStats(docs=docs, index_size_mb=round(index_size_mb, 4), compression_vs_float32=round(compression, 1), exact_hit_rate=exact_hit_rate)
