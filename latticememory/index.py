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
    e8_key_size_mb: float | None = None
    fallback_size_mb: float | None = None
    total_index_size_mb: float | None = None
    e8_key_bytes: int | None = None
    fallback_bytes: int | None = None
    total_index_bytes: int | None = None
    float32_embedding_bytes: int | None = None
    fallback_quantization: int | None = None
    compression_mode: str | None = None
    key_only_compression_vs_float32: float | None = None
    total_compression_vs_float32: float | None = None


class LatticeIndex:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 64,
        beam_radius: int = 1,
        fallback_quantization: int | None = None,
    ):
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
        self._init_with_encoder(
            encoder,
            d_model=d_model,
            batch_size=batch_size,
            beam_radius=beam_radius,
            fallback_quantization=fallback_quantization,
        )

    def _init_with_encoder(
        self,
        encoder,
        *,
        d_model: int,
        batch_size: int = 64,
        beam_radius: int = 1,
        fallback_quantization: int | None = None,
    ) -> None:
        from latticememory.memory import DenseVectorFallback
        self._d_model = d_model
        fallback = DenseVectorFallback(d_model=d_model, quantization_bits=fallback_quantization)
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
        e8_key_bytes = docs * (self._d_model // 8) * 3
        fallback_bytes = 0
        fallback_quantization = None
        if self._runtime.memory.fallback is not None:
            fallback_quantization = getattr(self._runtime.memory.fallback, "quantization_bits", None)
            if fallback_quantization is not None:
                fallback_bytes = getattr(self._runtime.memory.fallback, "get_index_size_bytes", lambda: 0)()
            
        total_bytes = e8_key_bytes + fallback_bytes
        float32_bytes = docs * self._d_model * 4
        
        e8_key_size_mb = e8_key_bytes / (1024 * 1024)
        fallback_size_mb = fallback_bytes / (1024 * 1024)
        total_index_size_mb = total_bytes / (1024 * 1024)
        
        key_only_compression = (float32_bytes / e8_key_bytes) if e8_key_bytes > 0 else 0.0
        total_compression = (float32_bytes / total_bytes) if total_bytes > 0 else 0.0
        compression_mode = "hybrid_quantized_fallback" if fallback_quantization is not None else "e8_key_only"
        compression = total_compression if fallback_quantization is not None else key_only_compression
        exact_hit_rate = (self._exact_hits / self._total_queries if self._total_queries > 0 else None)
        return LatticeStats(
            docs=docs,
            index_size_mb=round(total_index_size_mb, 4),
            compression_vs_float32=round(compression, 1),
            exact_hit_rate=exact_hit_rate,
            e8_key_size_mb=round(e8_key_size_mb, 4),
            fallback_size_mb=round(fallback_size_mb, 4),
            total_index_size_mb=round(total_index_size_mb, 4),
            e8_key_bytes=int(e8_key_bytes),
            fallback_bytes=int(fallback_bytes),
            total_index_bytes=int(total_bytes),
            float32_embedding_bytes=int(float32_bytes),
            fallback_quantization=fallback_quantization,
            compression_mode=compression_mode,
            key_only_compression_vs_float32=round(key_only_compression, 1),
            total_compression_vs_float32=round(total_compression, 1),
        )
