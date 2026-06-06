from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from .memory import MemoryQuery
from .text_runtime import RFSnapTextMemory


@dataclass(frozen=True)
class SemanticCacheEntry:
    cache_id: str
    prompt: str
    value: Any
    lattice_key: bytes
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SemanticCacheResult:
    hit: bool
    hit_type: str
    value: Any | None
    cache_id: str | None
    lattice_key: bytes
    source_prompt: str | None = None
    metadata: dict = field(default_factory=dict)
    retrieval_path: str = "miss"
    latency_ms: float = 0.0


class RFSnapSemanticCache:
    """Response cache keyed by RF-Snap/E8 lattice address.

    This is an E8-addressability product: repeated prompts do not need to be
    raw-string identical. If the RF-Snap encoder maps them to the same lattice
    address, they resolve to the same cache entry.
    """

    def __init__(
        self,
        *,
        runtime: RFSnapTextMemory,
        product: str | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
        allow_neighborhood: bool = False,
    ):
        self.runtime = runtime
        self.product = product
        self.dataset = dataset
        self.index_id = index_id
        self.allow_neighborhood = allow_neighborhood
        self._entries: dict[str, SemanticCacheEntry] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    def put(self, prompt: str, *, value: Any, metadata: dict | None = None) -> SemanticCacheEntry:
        embedding = self.runtime._encode_texts([prompt])[0]
        lattice_key = self.runtime.memory.lattice_key_for(embedding)
        cache_id = self._cache_id_for(lattice_key)
        now = time.time()
        existing = self._entries.get(cache_id)
        entry = SemanticCacheEntry(
            cache_id=cache_id,
            prompt=prompt,
            value=value,
            lattice_key=lattice_key,
            metadata=dict(metadata or {}),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        self._entries[cache_id] = entry
        self.runtime.memory.add_documents([
            self._entry_to_document(entry, embedding)
        ])
        return entry

    def get(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
    ) -> SemanticCacheResult:
        embedding = self.runtime._encode_texts([prompt])[0]
        lattice_key = self.runtime.memory.lattice_key_for(embedding)
        query = MemoryQuery(
            text=prompt,
            embedding=embedding,
            top_k=1,
            request_id=request_id,
            product=product or self.product,
            dataset=dataset or self.dataset,
            model_id=self.runtime.model_id,
            index_id=index_id or self.index_id,
        )
        result = self.runtime.memory.retrieve(query)
        if not result.hits:
            return SemanticCacheResult(
                hit=False,
                hit_type="miss",
                value=None,
                cache_id=None,
                lattice_key=lattice_key,
                retrieval_path=result.path,
                latency_ms=result.latency_ms,
            )
        if result.path in {"lattice_hamming", "lattice_hamming1"} and not self.allow_neighborhood:
            return SemanticCacheResult(
                hit=False,
                hit_type="miss",
                value=None,
                cache_id=None,
                lattice_key=lattice_key,
                retrieval_path=result.path,
                latency_ms=result.latency_ms,
            )
        hit = result.hits[0]
        entry = self._entries.get(hit.doc_id)
        if entry is None:
            return SemanticCacheResult(
                hit=False,
                hit_type="miss",
                value=None,
                cache_id=None,
                lattice_key=lattice_key,
                retrieval_path=result.path,
                latency_ms=result.latency_ms,
            )
        return SemanticCacheResult(
            hit=True,
            hit_type="exact" if result.path == "lattice_exact" else "neighborhood",
            value=entry.value,
            cache_id=entry.cache_id,
            lattice_key=lattice_key,
            source_prompt=entry.prompt,
            metadata=dict(entry.metadata),
            retrieval_path=result.path,
            latency_ms=result.latency_ms,
        )

    def stats(self) -> dict:
        return {
            "entries": self.size,
            "allow_neighborhood": self.allow_neighborhood,
            "product": self.product,
            "dataset": self.dataset,
            "index_id": self.index_id,
            "memory": self.runtime.memory.stats(),
        }

    def _entry_to_document(self, entry: SemanticCacheEntry, embedding):
        from .memory import MemoryDocument

        metadata = dict(entry.metadata)
        metadata.update(
            {
                "cache_id": entry.cache_id,
                "source_prompt": entry.prompt,
                "cache_created_at": entry.created_at,
                "cache_updated_at": entry.updated_at,
            }
        )
        return MemoryDocument(
            doc_id=entry.cache_id,
            text=entry.prompt,
            embedding=embedding,
            metadata=metadata,
        )

    @staticmethod
    def _cache_id_for(lattice_key: bytes) -> str:
        return "cache-" + hashlib.sha1(lattice_key).hexdigest()[:16]

