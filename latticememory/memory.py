from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F

from latticememory.rag.e8_retriever import E8LatticeDB, RetrievalHit
from .observability import LatticeObservability, RetrievalEvent


_RERANK_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_for_rerank(text: str) -> list[str]:
    return _RERANK_TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class MemoryDocument:
    doc_id: str
    text: str
    embedding: torch.Tensor | Iterable[float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    embedding: torch.Tensor | Iterable[float]
    top_k: int = 5
    request_id: str | None = None
    product: str | None = None
    dataset: str | None = None
    model_id: str | None = None
    index_id: str | None = None
    quality_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryHit:
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationContextPacket:
    query: str
    retrieval_path: str
    fallback_used: bool
    latency_ms: float
    context_chunks: list[MemoryHit]
    candidate_count: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "retrieval_path": self.retrieval_path,
            "fallback_used": self.fallback_used,
            "latency_ms": self.latency_ms,
            "candidate_count": self.candidate_count,
            "context_chunks": [
                {
                    "doc_id": hit.doc_id,
                    "text": hit.text,
                    "score": hit.score,
                    "metadata": hit.metadata,
                }
                for hit in self.context_chunks
            ],
        }


@dataclass(frozen=True)
class MemoryResult:
    query: str
    hits: list[MemoryHit]
    path: str
    fallback_used: bool
    latency_ms: float
    candidate_count: int

    def to_context_packet(self) -> dict:
        return GenerationContextPacket(
            query=self.query,
            retrieval_path=self.path,
            fallback_used=self.fallback_used,
            latency_ms=self.latency_ms,
            context_chunks=self.hits,
            candidate_count=self.candidate_count,
        ).to_dict()


class VectorFallback(Protocol):
    def add_documents(self, documents: Iterable[MemoryDocument]) -> None:
        ...

    def search(self, query: torch.Tensor | Iterable[float], top_k: int) -> list[RetrievalHit]:
        ...


class TextPairReranker(Protocol):
    def score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        ...


class DenseVectorFallback:
    """Standard dense cosine retrieval baseline and production fallback hook."""

    def __init__(self, d_model: int, quantization_bits: int | None = None):
        if quantization_bits is not None and quantization_bits not in (4, 8):
            raise ValueError("quantization_bits must be 4, 8, or None")
        if quantization_bits == 4 and d_model % 2 != 0:
            raise ValueError("d_model must be even for 4-bit quantization")
        self.d_model = d_model
        self.quantization_bits = quantization_bits
        self._doc_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._embeddings: torch.Tensor | None = None

    @property
    def num_documents(self) -> int:
        return len(self._doc_ids)

    def add_documents(self, documents: Iterable[MemoryDocument]) -> None:
        vectors = []
        for doc in documents:
            vector = torch.as_tensor(doc.embedding, dtype=torch.float32)
            if vector.dim() != 1 or vector.numel() != self.d_model:
                raise ValueError(f"document {doc.doc_id!r} embedding must have dim {self.d_model}")
            self._doc_ids.append(doc.doc_id)
            self._texts[doc.doc_id] = doc.text
            self._metadata[doc.doc_id] = dict(doc.metadata)
            normed = F.normalize(vector, p=2, dim=0)
            
            if self.quantization_bits == 8:
                quantized = (normed * 127.0).round().clamp(-127, 127).to(torch.int8)
                vectors.append(quantized)
            elif self.quantization_bits == 4:
                # Signed symmetric Int4: round/clamp to [-7, 7], shift by +8 to unsigned [1, 15]
                val_4 = (normed * 7.0).round().clamp(-7, 7).to(torch.int8)
                unsigned_val = (val_4 + 8).to(torch.uint8)
                # Pack two elements into one uint8 byte
                even_cols = unsigned_val[0::2]
                odd_cols = unsigned_val[1::2]
                packed = (even_cols << 4) | odd_cols
                vectors.append(packed)
            else:
                vectors.append(normed)
                
        if not vectors:
            return
        batch = torch.stack(vectors)
        self._embeddings = batch if self._embeddings is None else torch.cat([self._embeddings, batch], dim=0)

    def search(self, query: torch.Tensor | Iterable[float], top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._embeddings is None or not self._doc_ids:
            return []
        vector = torch.as_tensor(query, dtype=torch.float32)
        if vector.dim() != 1 or vector.numel() != self.d_model:
            raise ValueError(f"query embedding must have dim {self.d_model}")
            
        if self.quantization_bits == 8:
            dequantized = self._embeddings.float() / 127.0
            dequantized = F.normalize(dequantized, p=2, dim=1)
        elif self.quantization_bits == 4:
            packed = self._embeddings
            even_cols_unpacked = (packed >> 4) & 0x0F
            odd_cols_unpacked = packed & 0x0F
            
            N = packed.shape[0]
            unpacked = torch.zeros(N, self.d_model, dtype=torch.uint8, device=packed.device)
            unpacked[:, 0::2] = even_cols_unpacked
            unpacked[:, 1::2] = odd_cols_unpacked
            
            dequantized = (unpacked.to(torch.int8) - 8).float() / 7.0
            dequantized = F.normalize(dequantized, p=2, dim=1)
        else:
            dequantized = self._embeddings
            
        q = F.normalize(vector, p=2, dim=0)
        scores = dequantized @ q
        k = min(top_k, scores.numel())
        values, indices = torch.topk(scores, k=k)
        hits = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            doc_id = self._doc_ids[idx]
            hits.append(RetrievalHit(doc_id=doc_id, score=float(score), metadata=self._metadata.get(doc_id, {})))
        return hits

    def text_for(self, doc_id: str) -> str:
        return self._texts.get(doc_id, "")

    def get_index_size_bytes(self) -> int:
        if self._embeddings is None:
            return 0
        return self._embeddings.numel() * self._embeddings.element_size()

    def save(self, path: str | Path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "_version": 1,
                "d_model": self.d_model,
                "quantization_bits": self.quantization_bits,
                "doc_ids": list(self._doc_ids),
                "texts": dict(self._texts),
                "metadata": {doc_id: dict(meta) for doc_id, meta in self._metadata.items()},
                "embeddings": self._embeddings,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DenseVectorFallback":
        from pathlib import Path
        path = Path(path)
        data = torch.load(str(path), weights_only=False)
        if data.get("_version") != 1:
            raise ValueError(f"Unsupported DenseVectorFallback save version: {data.get('_version')}")
        fallback = cls(d_model=int(data["d_model"]), quantization_bits=data.get("quantization_bits"))
        fallback._doc_ids = list(data["doc_ids"])
        fallback._texts = dict(data["texts"])
        fallback._metadata = dict(data["metadata"])
        fallback._embeddings = data["embeddings"]
        return fallback


class RFSnapLatticeMemory:
    """Structured retrieval and cache acceleration layer for generation systems.

    The hot path is:
      1. RF-Snap/E8 exact lattice address cache.
      2. Radius-1 lattice neighborhood prefilter plus rerank.
      3. Dense/vector-DB fallback for misses.
      4. Generation context packet with path/candidate telemetry.
    """

    def __init__(
        self,
        d_model: int = 1024,
        fallback: VectorFallback | None = None,
        hamming_pool_multiplier: int = 10,
        rerank_mode: str = "cosine",
        lexical_weight: float = 0.0,
        neural_reranker: TextPairReranker | None = None,
        neural_rerank_top_n: int | None = None,
        observability: LatticeObservability | None = None,
        beam_radius: int = 1,
        beam_key_scan_max_documents: int = 4096,
        sqlite_path: str | Path | None = None,
    ):
        self.d_model = d_model
        self.lattice = E8LatticeDB(d_model=d_model)
        self.fallback = fallback
        self.hamming_pool_multiplier = max(int(hamming_pool_multiplier), 1)
        if rerank_mode not in {"cosine", "hybrid_text", "neural_text"}:
            raise ValueError("rerank_mode must be 'cosine', 'hybrid_text', or 'neural_text'")
        if rerank_mode == "neural_text" and neural_reranker is None:
            raise ValueError("neural_reranker is required when rerank_mode='neural_text'")
        self.rerank_mode = rerank_mode
        self.lexical_weight = max(float(lexical_weight), 0.0)
        self.neural_reranker = neural_reranker
        self.neural_rerank_top_n = max(int(neural_rerank_top_n or 0), 0)
        self.observability = observability or LatticeObservability()
        if not isinstance(beam_radius, int) or beam_radius < 1:
            raise ValueError(f"beam_radius must be a positive integer, got {beam_radius!r}")
        self.beam_radius = beam_radius
        self.beam_key_scan_max_documents = max(int(beam_key_scan_max_documents), 0)
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._doc_ids: list[str] = []
        self._doc_id_set: set[str] = set()
        self._doc_index: dict[str, int] = {}
        self._ann_embeddings: torch.Tensor | None = None
        self._last_query_vector: torch.Tensor | None = None

        self.sqlite_store = None
        if sqlite_path is not None:
            from latticememory.sqlite_store import LatticeSqliteStore
            self.sqlite_store = LatticeSqliteStore(sqlite_path)
            if self.sqlite_store.count() > 0:
                self._load_from_store()

    def _load_from_store(self) -> None:
        if self.sqlite_store is None:
            return
        import json
        import numpy as np
        rows = self.sqlite_store.get_all_rows()
        new_ann_embeddings: list[torch.Tensor] = []
        docs_to_fallback: list[MemoryDocument] = []
        for row in rows:
            doc_id = row["doc_id"]
            text = row["text"]
            lattice_key = row["lattice_key"]
            metadata = json.loads(row["metadata"])
            
            vector = torch.from_numpy(np.frombuffer(row["embedding"], dtype=np.float32).copy()).to(self.lattice.device)
            
            self.lattice.hash_store[lattice_key].append(doc_id)
            self.lattice._embeddings[doc_id] = F.normalize(vector, p=2, dim=0)
            self.lattice._metadata[doc_id] = metadata
            self.lattice._keys[doc_id] = lattice_key
            
            ann_vector = F.normalize(vector, p=2, dim=0).detach().cpu()
            self._texts[doc_id] = text
            self._metadata[doc_id] = metadata
            self._doc_index[doc_id] = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._doc_id_set.add(doc_id)
            new_ann_embeddings.append(ann_vector)
            
            if self.fallback is not None:
                docs_to_fallback.append(
                    MemoryDocument(
                        doc_id=doc_id,
                        text=text,
                        embedding=vector,
                        metadata=metadata,
                    )
                )
            
        if new_ann_embeddings:
            self._ann_embeddings = torch.stack(new_ann_embeddings)
        if self.fallback is not None and docs_to_fallback:
            self.fallback.add_documents(docs_to_fallback)

    def lattice_key_for(self, embedding: torch.Tensor | Iterable[float]) -> bytes:
        """Compute the quantized E8 lattice key for the given embedding."""
        vector = self.lattice._as_vector(embedding)
        return self.lattice._quantize_to_indices(vector)

    def get_document_ids_by_lattice_key(self, lattice_key: bytes) -> list[str]:
        """Retrieve the document IDs mapped to this E8 lattice key."""
        return list(self.lattice.hash_store.get(lattice_key, []))

    def get_document_text(self, doc_id: str) -> str | None:
        """Retrieve the text associated with a document ID."""
        return self._texts.get(doc_id)

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Retrieve the metadata associated with a document ID."""
        return self._metadata.get(doc_id)

    def get_document_lattice_key(self, doc_id: str) -> bytes | None:
        """Retrieve the E8 lattice key associated with a document ID."""
        return self.lattice._keys.get(doc_id)

    @property
    def num_documents(self) -> int:
        return self.lattice.num_documents

    def add_documents(self, documents: Iterable[MemoryDocument]) -> None:
        docs = list(documents)
        if not docs:
            return

        # Phase 1: validate all embedding dimensions before any mutation so that
        # a bad document never leaves the index in a partially-committed state.
        for doc in docs:
            vec = torch.as_tensor(doc.embedding, dtype=torch.float32)
            if vec.dim() != 1 or vec.numel() != self.d_model:
                raise ValueError(
                    f"document {doc.doc_id!r}: expected 1-D embedding with dim {self.d_model}, "
                    f"got shape {tuple(vec.shape)}"
                )

        # Phase 2: commit — all dimensions verified.
        new_ann_embeddings: list[torch.Tensor] = []
        for doc in docs:
            is_update = doc.doc_id in self._doc_id_set
            vector = self.lattice._as_vector(doc.embedding)
            self.lattice.add_document(doc.doc_id, vector, metadata=doc.metadata)
            ann_vector = F.normalize(vector, p=2, dim=0).detach().cpu()
            self._texts[doc.doc_id] = doc.text
            self._metadata[doc.doc_id] = dict(doc.metadata)
            if is_update:
                # Update the existing ann_embedding row in-place; don't grow _doc_ids.
                idx = self._doc_index[doc.doc_id]
                if self._ann_embeddings is not None:
                    self._ann_embeddings[idx] = ann_vector
            else:
                self._doc_index[doc.doc_id] = len(self._doc_ids)
                self._doc_ids.append(doc.doc_id)
                self._doc_id_set.add(doc.doc_id)
                new_ann_embeddings.append(ann_vector)

        if new_ann_embeddings:
            ann_batch = torch.stack(new_ann_embeddings)
            self._ann_embeddings = (
                ann_batch if self._ann_embeddings is None else torch.cat([self._ann_embeddings, ann_batch], dim=0)
            )
        if self.fallback is not None:
            self.fallback.add_documents(docs)
        if self.sqlite_store is not None:
            lattice_keys = {doc.doc_id: self.lattice._keys[doc.doc_id] for doc in docs}
            self.sqlite_store.add_documents(docs, lattice_keys)

    def retrieve(self, query: MemoryQuery) -> MemoryResult:
        if query.top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()
        stage_latency_ms: dict[str, float] = {}

        stage_start = time.perf_counter()
        vector = self.lattice._as_vector(query.embedding)
        stage_latency_ms["vector_prepare"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        self._last_query_vector = vector
        query_key = self.lattice._quantize_to_indices(vector)
        exact_ids = list(self.lattice.hash_store.get(query_key, []))
        stage_latency_ms["exact_lookup"] = (time.perf_counter() - stage_start) * 1000
        if exact_ids:
            stage_start = time.perf_counter()
            exact_indices = [self._doc_index[doc_id] for doc_id in exact_ids if doc_id in self._doc_index]
            hits = self._hits_from_ann_indices(vector, exact_indices, query.top_k, query_text=query.text)
            stage_latency_ms["rerank"] = (time.perf_counter() - stage_start) * 1000
            result = MemoryResult(
                query=query.text,
                hits=hits,
                path="lattice_exact",
                fallback_used=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                candidate_count=len(exact_ids),
            )
            self._record_event(
                result,
                query=query,
                query_key=query_key,
                stage_latency_ms=stage_latency_ms,
                neighborhood=[{"doc_id": doc_id, "hamming_distance": 0} for doc_id in exact_ids],
            )
            return result

        stage_start = time.perf_counter()
        ann_neighbors = self._ann_candidate_distances(query_key)
        ann_indices = [idx for idx, _distance in ann_neighbors]
        stage_latency_ms["ann_lookup"] = (time.perf_counter() - stage_start) * 1000
        if ann_indices:
            stage_start = time.perf_counter()
            hits = self._hits_from_ann_indices(vector, ann_indices, query.top_k, query_text=query.text)
            stage_latency_ms["rerank"] = (time.perf_counter() - stage_start) * 1000
            beam_path = "lattice_hamming1" if self.beam_radius == 1 else f"lattice_beam_R{self.beam_radius}"
            result = MemoryResult(
                query=query.text,
                hits=hits,
                path=beam_path,
                fallback_used=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                candidate_count=len(ann_indices),
            )
            self._record_event(
                result,
                query=query,
                query_key=query_key,
                stage_latency_ms=stage_latency_ms,
                neighborhood=[
                    {"doc_id": self._doc_ids[idx], "hamming_distance": int(distance)}
                    for idx, distance in ann_neighbors
                ],
            )
            return result

        stage_start = time.perf_counter()
        fallback_hits = self.fallback.search(vector.detach().cpu(), query.top_k) if self.fallback is not None else []
        stage_latency_ms["fallback"] = (time.perf_counter() - stage_start) * 1000
        result = MemoryResult(
            query=query.text,
            hits=self._hits_from_retrieval(fallback_hits),
            path="fallback" if fallback_hits else "miss",
            fallback_used=bool(fallback_hits),
            latency_ms=(time.perf_counter() - start) * 1000,
            candidate_count=0,
        )
        self._record_event(
            result,
            query=query,
            query_key=query_key,
            stage_latency_ms=stage_latency_ms,
            neighborhood=[],
        )
        return result

    def _hits_from_retrieval(self, hits: list[RetrievalHit]) -> list[MemoryHit]:
        return [
            MemoryHit(
                doc_id=hit.doc_id,
                text=self._text_for(hit.doc_id),
                score=hit.score,
                metadata=hit.metadata if hit.metadata is not None else self._metadata.get(hit.doc_id, {}),
            )
            for hit in hits
        ]

    def _text_for(self, doc_id: str) -> str:
        if doc_id in self._texts:
            return self._texts[doc_id]
        if hasattr(self.fallback, "text_for"):
            return self.fallback.text_for(doc_id)
        return ""

    def _ann_candidate_distances(self, query_key: bytes) -> list[tuple[int, int]]:
        """Return (row_index, hamming_distance) for docs within beam_radius.

        beam_radius=1 : exhaustive Hamming-1 walk (identical to prior behaviour).
        beam_radius>1 : extends with retrieve_within_radius top-K approximation.
        """
        if not self._doc_ids:
            return []
        result: list[tuple[int, int]] = []
        seen: set[str] = set()

        def _add(doc_id: str, dist: int) -> None:
            if doc_id in self._doc_index and doc_id not in seen:
                result.append((self._doc_index[doc_id], dist))
                seen.add(doc_id)

        # Distance-0: exact
        for doc_id in self.lattice.hash_store.get(query_key, []):
            _add(doc_id, 0)

        if self.beam_radius == 1:
            # Exhaustive Hamming-1 (original behaviour)
            probe = bytearray(query_key)
            for block_idx in range(self.lattice.num_blocks):
                original = probe[block_idx]
                for alt in range(240):
                    if alt == original:
                        continue
                    probe[block_idx] = alt
                    candidates = self.lattice.hash_store.get(bytes(probe), None)
                    if candidates:
                        for doc_id in candidates:
                            _add(doc_id, 1)
                probe[block_idx] = original
        else:
            if (
                self.beam_key_scan_max_documents > 0
                and self.lattice.num_documents <= self.beam_key_scan_max_documents
            ):
                for doc_id, doc_key in self.lattice._keys.items():
                    dist = sum(a != b for a, b in zip(query_key, doc_key))
                    if dist <= self.beam_radius:
                        _add(doc_id, dist)
            else:
                # Beam search via retrieve_within_radius
                if self._last_query_vector is None:
                    return result
                beam_ids = self.lattice.retrieve_within_radius(
                    self._last_query_vector,
                    radius=self.beam_radius,
                    top_k_alts=5,
                    max_candidates=500,
                )
                for doc_id in beam_ids:
                    doc_key = self.lattice._keys.get(doc_id)
                    if doc_key is not None:
                        dist = sum(a != b for a, b in zip(query_key, doc_key))
                    else:
                        dist = 1
                    _add(doc_id, dist)

        return result

    def _hits_from_ann_indices(
        self,
        query: torch.Tensor,
        candidate_indices: list[int],
        top_k: int,
        *,
        query_text: str | None = None,
    ) -> list[MemoryHit]:
        if not candidate_indices or self._ann_embeddings is None:
            return []
        q = F.normalize(query.detach().cpu(), p=2, dim=0)
        index_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        candidate_embeddings = self._ann_embeddings[index_tensor]
        cosine_scores = candidate_embeddings @ q
        scores = cosine_scores
        if self.rerank_mode == "hybrid_text" and self.lexical_weight > 0 and query_text:
            lexical_scores = torch.tensor(
                [self._lexical_overlap_score(query_text, self._text_for(self._doc_ids[idx])) for idx in candidate_indices],
                dtype=torch.float32,
            )
            scores = cosine_scores + (lexical_scores * self.lexical_weight)
        elif self.rerank_mode == "neural_text" and query_text:
            rerank_indices = candidate_indices
            if self.neural_rerank_top_n > 0 and self.neural_rerank_top_n < len(candidate_indices):
                n = max(top_k, self.neural_rerank_top_n)
                n = min(n, cosine_scores.numel())
                preselect = torch.topk(cosine_scores, k=n).indices.tolist()
                rerank_indices = [candidate_indices[pos] for pos in preselect]
            candidates = [(self._doc_ids[idx], self._text_for(self._doc_ids[idx])) for idx in rerank_indices]
            neural_scores = self.neural_reranker.score(query_text, candidates) if self.neural_reranker is not None else []
            if len(neural_scores) != len(candidates):
                raise ValueError("neural_reranker returned a score count that does not match candidates")
            candidate_indices = rerank_indices
            scores = torch.tensor(neural_scores, dtype=torch.float32)
        k = min(top_k, scores.numel())
        values, order = torch.topk(scores, k=k)
        hits = []
        for score, candidate_pos in zip(values.tolist(), order.tolist()):
            doc_idx = candidate_indices[candidate_pos]
            doc_id = self._doc_ids[doc_idx]
            hits.append(
                MemoryHit(
                    doc_id=doc_id,
                    text=self._text_for(doc_id),
                    score=float(score),
                    metadata=self._metadata.get(doc_id, {}),
                )
            )
        return hits

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        query_tokens = set(_tokenize_for_rerank(query))
        if not query_tokens:
            return 0.0
        text_tokens = set(_tokenize_for_rerank(text))
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _record_event(
        self,
        result: MemoryResult,
        *,
        query: MemoryQuery,
        query_key: bytes,
        stage_latency_ms: dict[str, float],
        neighborhood: list[dict],
    ) -> None:
        self.observability.record(
            RetrievalEvent(
                query=result.query,
                path=result.path,
                latency_ms=result.latency_ms,
                fallback_used=result.fallback_used,
                candidate_count=result.candidate_count,
                hit_count=len(result.hits),
                top_doc_id=result.hits[0].doc_id if result.hits else None,
                request_id=query.request_id,
                product=query.product,
                dataset=query.dataset,
                model_id=query.model_id,
                index_id=query.index_id,
                query_key=list(query_key),
                stage_latency_ms=dict(stage_latency_ms),
                neighborhood=neighborhood,
                quality_tags=list(query.quality_tags),
            )
        )

    def observability_summary(self) -> dict:
        return self.observability.summary()

    def lattice_key_for(self, embedding: torch.Tensor | Iterable[float]) -> bytes:
        return self.lattice._quantize_to_indices(embedding)

    def stats(self) -> dict:
        return {
            "documents": self.num_documents,
            "d_model": self.d_model,
            "blocks": self.lattice.num_blocks,
            "fallback_configured": self.fallback is not None,
            "hamming_candidates": bool(self._doc_ids),
            "candidate_backend": "hash_radius1",
            "rerank_mode": self.rerank_mode,
            "lexical_weight": self.lexical_weight,
            "neural_rerank_top_n": self.neural_rerank_top_n,
            "observability": self.observability.summary(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the lattice index and document texts/metadata to `path`.

        Saves the E8LatticeDB index via `E8LatticeDB.save()` and stores
        document text/metadata in an adjacent `.meta.pt` file.
        `path` should end in `.pt` (e.g. `my_index.pt`).
        The observability stream is intentionally not persisted — events
        are ephemeral.
        """
        import os
        self.lattice.save(path)
        meta_path = path + ".meta.pt"
        os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
        torch.save({
            "_version": 1,
            "texts": dict(self._texts),
            "metadata": {d: dict(m) for d, m in self._metadata.items()},
            "hamming_pool_multiplier": self.hamming_pool_multiplier,
            "rerank_mode": self.rerank_mode,
            "lexical_weight": self.lexical_weight,
            "neural_rerank_top_n": self.neural_rerank_top_n,
        }, meta_path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        fallback=None,
        hamming_pool_multiplier: int | None = None,
        rerank_mode: str | None = None,
        lexical_weight: float | None = None,
        neural_reranker: TextPairReranker | None = None,
        neural_rerank_top_n: int | None = None,
    ) -> "RFSnapLatticeMemory":
        """Reconstruct a `RFSnapLatticeMemory` from files saved with `save()`.

        Observability starts fresh; document fallback must be re-supplied.
        """
        from latticememory.rag.e8_retriever import E8LatticeDB
        lattice = E8LatticeDB.load(path)
        meta_path = path + ".meta.pt"
        meta = torch.load(meta_path, weights_only=False)
        memory = cls(
            d_model=lattice.d_model,
            fallback=fallback,
            hamming_pool_multiplier=(
                hamming_pool_multiplier
                if hamming_pool_multiplier is not None
                else int(meta.get("hamming_pool_multiplier", 10))
            ),
            rerank_mode=rerank_mode or str(meta.get("rerank_mode", "cosine")),
            lexical_weight=(
                lexical_weight
                if lexical_weight is not None
                else float(meta.get("lexical_weight", 0.0))
            ),
            neural_reranker=neural_reranker,
            neural_rerank_top_n=(
                neural_rerank_top_n
                if neural_rerank_top_n is not None
                else int(meta.get("neural_rerank_top_n", 0))
            ),
        )
        memory.lattice = lattice
        memory._texts = dict(meta["texts"])
        memory._metadata = dict(meta["metadata"])
        memory._doc_ids = list(lattice._embeddings.keys())
        memory._doc_id_set = set(memory._doc_ids)
        memory._doc_index = {doc_id: i for i, doc_id in enumerate(memory._doc_ids)}
        ann_embeddings = [lattice._embeddings[d].detach().cpu() for d in memory._doc_ids]
        memory._ann_embeddings = torch.stack(ann_embeddings) if ann_embeddings else None
        return memory

    def save_to_sqlite(self, db_path: str) -> None:
        """Persist the index and documents to an SQLite database at `db_path`."""
        from latticememory.sqlite_store import LatticeSqliteStore
        store = LatticeSqliteStore(db_path)
        docs = []
        lattice_keys = {}
        for doc_id in self._doc_ids:
            emb = self.lattice._embeddings[doc_id]
            text = self._texts[doc_id]
            meta = self._metadata[doc_id]
            docs.append(MemoryDocument(doc_id=doc_id, text=text, embedding=emb, metadata=meta))
            lattice_keys[doc_id] = self.lattice._keys[doc_id]
        store.add_documents(docs, lattice_keys)
        store.close()

    @classmethod
    def load_from_sqlite(
        cls,
        db_path: str,
        *,
        fallback=None,
        hamming_pool_multiplier: int | None = None,
        rerank_mode: str | None = None,
        lexical_weight: float | None = None,
        neural_reranker: TextPairReranker | None = None,
        neural_rerank_top_n: int | None = None,
        beam_radius: int = 1,
    ) -> "RFSnapLatticeMemory":
        """Load and reconstruct a `RFSnapLatticeMemory` from an SQLite database."""
        from latticememory.sqlite_store import LatticeSqliteStore
        store = LatticeSqliteStore(db_path)
        
        rows = store.get_all_rows()
        if not rows:
            raise ValueError(f"SQLite database at {db_path} contains no documents")
            
        import numpy as np
        first_emb = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        d_model = len(first_emb)
        
        memory = cls(
            d_model=d_model,
            fallback=fallback,
            hamming_pool_multiplier=hamming_pool_multiplier or 10,
            rerank_mode=rerank_mode or "cosine",
            lexical_weight=lexical_weight or 0.0,
            neural_reranker=neural_reranker,
            neural_rerank_top_n=neural_rerank_top_n,
            beam_radius=beam_radius,
        )
        memory.sqlite_store = store
        memory._load_from_store()
        return memory








