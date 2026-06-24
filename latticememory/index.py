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


_VALID_MODES = frozenset({"cache", "hybrid", "pq"})

# M=8 blocks, K=256 codewords/block: the validated sweet spot from real-model
# testing (PAWS, real encoder, genuine held-out split, 2026-06-24). Going
# coarser (M=2/M=4) raises raw recall further but explodes the candidate pool
# (100+ docs) - giving up the speed/compression point of addressing in the
# first place. M=8 holds up at ~7x corpus scale, not just on a small test set.
# See docs/manual-results/2026-06-24-open-vocab-semantic-addressing-redesign.md.
_DEFAULT_PQ_NUM_BLOCKS = 8
_DEFAULT_PQ_CODEBOOK_SIZE = 256


class LatticeIndex:
    """Semantic index with E8 lattice routing and optional dense fallback.

    mode="cache"  (default): optimised for repeat/paraphrase cache hits and
        deduplication.  E8 exact + Hamming-1 paths only.  Suitable for
        symmetric workloads where query text ≈ indexed text.

    mode="hybrid": E8 paths first, mandatory Int8 dense fallback for novel
        queries.  Required for asymmetric RAG/search where queries and
        documents are structurally different (e.g. question vs. passage).
        Defaults fallback_quantization=8 (Int8) unless overridden.

    mode="pq": data-calibrated Product Quantization addressing instead of
        the fixed E8 lattice - real-model validation found E8's 128-block
        exact/Hamming-1 mechanism has ~0% hit rate on open-vocabulary
        paraphrases (and even on a genuine closed-vocabulary held-out test);
        PQ (8 coarser, learned-codebook blocks by default) reaches 31%+
        Recall@1 on the same real PAWS benchmark, confirmed to hold up at
        ~7x corpus scale. Unlike E8's fixed mathematical lattice, PQ's
        codebooks must be calibrated on real data before they're useful:
        call `fit_pq(sample_texts)` with a representative sample BEFORE
        `add()` for best quality, or just call `add()` directly and the
        first batch added will be used to fit the codebooks automatically
        (works, but a dedicated calibration sample - ideally 1,000+ texts
        from your actual domain - generalizes better than whatever happens
        to be in the first `add()` call).
        See docs/manual-results/2026-06-24-open-vocab-semantic-addressing-redesign.md.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 64,
        beam_radius: int = 1,
        fallback_quantization: int | None = None,
        mode: str = "cache",
        pq_num_blocks: int = _DEFAULT_PQ_NUM_BLOCKS,
        pq_codebook_size: int = _DEFAULT_PQ_CODEBOOK_SIZE,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if mode == "hybrid" and fallback_quantization is None:
            fallback_quantization = 8  # Int8 — 4× smaller than float32, 95% recall parity
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
            mode=mode,
            device=device,
            pq_num_blocks=pq_num_blocks,
            pq_codebook_size=pq_codebook_size,
        )

    def _init_with_encoder(
        self,
        encoder,
        *,
        d_model: int,
        batch_size: int = 64,
        beam_radius: int = 1,
        fallback_quantization: int | None = None,
        mode: str = "cache",
        device: str = "cpu",
        pq_num_blocks: int = _DEFAULT_PQ_NUM_BLOCKS,
        pq_codebook_size: int = _DEFAULT_PQ_CODEBOOK_SIZE,
    ) -> None:
        # Self-sufficient regardless of how it's called: existing tests build
        # a LatticeIndex via __new__() + _init_with_encoder() directly,
        # bypassing __init__ (and its mode/device/_pq_fitted setup) entirely -
        # this method must not depend on __init__ having run first.
        self._mode = mode
        self._device = device
        self._pq_fitted = False
        from latticememory.memory import DenseVectorFallback, RFSnapLatticeMemory
        self._d_model = d_model
        fallback = DenseVectorFallback(d_model=d_model, quantization_bits=fallback_quantization)
        if self._mode == "pq":
            from latticememory.rag.pq_retriever import PQLatticeDB
            lattice = PQLatticeDB(
                d_model=d_model, num_blocks=pq_num_blocks, codebook_size=pq_codebook_size, device=self._device,
            )
            memory = RFSnapLatticeMemory(d_model=d_model, fallback=fallback, beam_radius=beam_radius, lattice=lattice)
            self._runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, batch_size=batch_size, memory=memory)
        else:
            self._runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, batch_size=batch_size, fallback=fallback, beam_radius=beam_radius)
        self._total_queries: int = 0
        self._exact_hits: int = 0

    def fit_pq(self, sample_texts: Sequence[str]) -> None:
        """Calibrate PQ codebooks on a representative text sample. Only valid
        for mode="pq". Call this BEFORE add() with a sample from your actual
        domain (1,000+ texts recommended) for the best quality - codebooks
        calibrated on real data noticeably outperform whatever happens to be
        in the first add() batch (see the zero-shot vs. in-domain comparison
        in docs/manual-results/2026-06-24-open-vocab-semantic-addressing-redesign.md)."""
        if self._mode != "pq":
            raise ValueError("fit_pq() is only valid when mode='pq'")
        text_list = list(sample_texts)
        if not text_list:
            raise ValueError("sample_texts must not be empty")
        embs = self._runtime._encode_texts(text_list)
        self._runtime.memory.lattice.fit(embs)
        self._pq_fitted = True

    def add(self, texts: Sequence[str], doc_ids: Sequence[str] | None = None, metadatas: Sequence[dict] | None = None) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []
        if self._mode == "pq" and not self._pq_fitted:
            # Auto-fit on first add() if the caller didn't call fit_pq() explicitly -
            # works, but see fit_pq()'s docstring for why a dedicated calibration
            # sample generalizes better than whatever's in this first batch.
            self.fit_pq(text_list)
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

    def observatory(self) -> "LatticeObservatory":
        """Return a LatticeObservatory bound to this index for block-level analysis."""
        from latticememory.observatory import LatticeObservatory
        return LatticeObservatory(self)

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
        compression_mode = (
            "hybrid_int8_fallback" if fallback_quantization == 8
            else "hybrid_quantized_fallback" if fallback_quantization is not None
            else "e8_key_only"
        )
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
