from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .memory import MemoryQuery
from .text_runtime import RFSnapTextMemory

if TYPE_CHECKING:
    from .hamming_router import HammingRouter


@dataclass(frozen=True)
class SemanticCacheEntry:
    cache_id: str
    prompt: str
    value: Any
    lattice_key: bytes
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl_seconds: float | None = None  # None = never expires

    def is_expired(self) -> bool:
        return self.ttl_seconds is not None and time.time() > self.created_at + self.ttl_seconds


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
    hamming_distance: int = -1  # -1 = not a Hamming-NN hit
    shadow_hit: bool = False
    shadow_cache_id: str | None = None
    shadow_hamming_distance: int = -1
    shadow_source_prompt: str | None = None
    shadow_value: Any | None = None


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
        hamming_router: "HammingRouter | None" = None,
        hamming_threshold: int = 70,
        hamming_router_mode: str = "off",
    ):
        self.runtime = runtime
        self.product = product
        self.dataset = dataset
        self.index_id = index_id
        self.allow_neighborhood = allow_neighborhood
        self._hamming_router = hamming_router
        self._hamming_threshold = hamming_threshold
        self.hamming_router_mode = hamming_router_mode
        self._entries: dict[str, SemanticCacheEntry] = {}

        # Restore existing cache entries from runtime memory on startup
        if hasattr(self.runtime, "memory") and hasattr(self.runtime.memory, "_doc_ids"):
            for doc_id in self.runtime.memory._doc_ids:
                meta = self.runtime.memory.get_document_metadata(doc_id)
                if meta and "source_prompt" in meta:
                    prompt = meta["source_prompt"]
                    created_at = meta.get("cache_created_at", time.time())
                    updated_at = meta.get("cache_updated_at", time.time())
                    value = meta.get("cache_value")
                    lattice_key = self.runtime.memory.get_document_lattice_key(doc_id)
                    user_metadata = {k: v for k, v in meta.items() if k not in {
                        "cache_id", "source_prompt", "cache_created_at", "cache_updated_at", "cache_value"
                    }}
                    entry = SemanticCacheEntry(
                        cache_id=doc_id,
                        prompt=prompt,
                        value=value,
                        lattice_key=lattice_key,
                        metadata=user_metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    self._entries[doc_id] = entry
                    if self._hamming_router is not None and lattice_key:
                        self._hamming_router.add_from_key(lattice_key, doc_id)

    @property
    def size(self) -> int:
        return len(self._entries)

    def put(
        self,
        prompt: str,
        *,
        value: Any,
        metadata: dict | None = None,
        ttl_seconds: float | None = None,
    ) -> SemanticCacheEntry:
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
            ttl_seconds=ttl_seconds,
        )
        self._entries[cache_id] = entry
        self.runtime.memory.add_documents([
            self._entry_to_document(entry, embedding)
        ])
        if self._hamming_router is not None:
            # add_from_key deduplicates internally — no separate tracking needed
            self._hamming_router.add_from_key(lattice_key, cache_id)
        return entry

    def delete(self, cache_id: str) -> bool:
        """Remove a cache entry by ID from all backing stores.

        Propagates to the Hamming router and the underlying runtime memory
        (including SQLite when configured).  Returns True if the entry existed.
        """
        entry = self._entries.pop(cache_id, None)
        if entry is None:
            return False
        if self._hamming_router is not None:
            self._hamming_router.remove_by_value(cache_id)
        self.runtime.memory.delete_documents([cache_id])
        return True

    def evict_expired(self) -> int:
        """Delete all TTL-expired entries. Returns the count of evicted entries."""
        expired = [cid for cid, e in list(self._entries.items()) if e.is_expired()]
        for cid in expired:
            self.delete(cid)
        return len(expired)

    def restore_entry(self, entry: "SemanticCacheEntry") -> None:
        """Restore a pre-built entry (e.g. from a JSONL export) without re-encoding.

        Writes to ``_entries``, the Hamming router, and SQLite.  The embedding
        stored in SQLite is a zero placeholder — sufficient for exact lattice key
        lookup on reload (``_load_from_store`` uses the stored key directly) but
        incorrect for ANN cosine reranking.  Acceptable for import use cases where
        exact key matching is the only retrieval path.
        """
        import numpy as np
        from .memory import MemoryDocument

        self._entries[entry.cache_id] = entry
        if self._hamming_router is not None and entry.lattice_key:
            self._hamming_router.add_from_key(entry.lattice_key, entry.cache_id)

        sqlite_store = getattr(self.runtime.memory, "sqlite_store", None)
        if sqlite_store is not None and entry.lattice_key:
            d = self.runtime.d_model or 1024
            zero_emb = np.zeros(d, dtype=np.float32)
            meta = dict(entry.metadata)
            meta.update({
                "cache_id": entry.cache_id,
                "source_prompt": entry.prompt,
                "cache_created_at": entry.created_at,
                "cache_updated_at": entry.updated_at,
                "cache_value": entry.value,
            })
            doc = MemoryDocument(
                doc_id=entry.cache_id,
                text=entry.prompt,
                embedding=zero_emb,
                metadata=meta,
            )
            sqlite_store.add_documents([doc], {entry.cache_id: entry.lattice_key})

    def get(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
    ) -> SemanticCacheResult:
        t0 = time.time()
        embedding = self.runtime._encode_texts([prompt])[0]
        lattice_key = self.runtime.memory.lattice_key_for(embedding)

        # Fast path: O(1) exact dict lookup (avoids memory.retrieve() round-trip)
        cache_id = self._cache_id_for(lattice_key)
        entry = self._entries.get(cache_id)
        if entry is not None:
            if entry.is_expired():
                self.delete(cache_id)
            else:
                return SemanticCacheResult(
                    hit=True,
                    hit_type="exact",
                    value=entry.value,
                    cache_id=entry.cache_id,
                    lattice_key=lattice_key,
                    source_prompt=entry.prompt,
                    metadata=dict(entry.metadata),
                    retrieval_path="lattice_exact",
                    latency_ms=(time.time() - t0) * 1000,
                )

        # Hamming-NN path: opt-in, off by default. WARNING: false-positive risk in dense caches
        # with adjacent-but-distinct prompts (see threshold calibration notes in hamming_router.py).
        if self._hamming_router is not None and self.hamming_router_mode in ("serve", "shadow"):
            router_result = self._hamming_router.lookup_key(
                lattice_key, threshold=self._hamming_threshold
            )
            if router_result is not None:
                nn_entry = self._entries.get(router_result.value)
                if nn_entry is not None:
                    if self.hamming_router_mode == "serve":
                        return SemanticCacheResult(
                            hit=True,
                            hit_type="hamming_nn",
                            value=nn_entry.value,
                            cache_id=nn_entry.cache_id,
                            lattice_key=lattice_key,
                            source_prompt=nn_entry.prompt,
                            metadata=dict(nn_entry.metadata),
                            retrieval_path="hamming_nn",
                            latency_ms=(time.time() - t0) * 1000,
                            hamming_distance=router_result.hamming_distance,
                        )
                    else:  # shadow mode
                        return SemanticCacheResult(
                            hit=False,
                            hit_type="miss",
                            value=None,
                            cache_id=None,
                            lattice_key=lattice_key,
                            retrieval_path="miss",
                            latency_ms=(time.time() - t0) * 1000,
                            hamming_distance=-1,
                            shadow_hit=True,
                            shadow_cache_id=nn_entry.cache_id,
                            shadow_hamming_distance=router_result.hamming_distance,
                            shadow_source_prompt=nn_entry.prompt,
                            shadow_value=nn_entry.value,
                        )

        # Legacy memory.retrieve() path: preserves allow_neighborhood=True behavior
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
        base: dict[str, Any] = {
            "entries": self.size,
            "allow_neighborhood": self.allow_neighborhood,
            "product": self.product,
            "dataset": self.dataset,
            "index_id": self.index_id,
            "memory": self.runtime.memory.stats(),
        }
        if self._hamming_router is not None:
            base["hamming_router"] = {
                "enabled": True,
                "mode": self.hamming_router_mode,
                "threshold": self._hamming_threshold,
                "entries": len(self._hamming_router),
                "is_reliable": len(self._hamming_router) >= 100,
            }
        else:
            base["hamming_router"] = {"enabled": False, "mode": self.hamming_router_mode}
        return base

    def _entry_to_document(self, entry: SemanticCacheEntry, embedding):
        from .memory import MemoryDocument

        metadata = dict(entry.metadata)
        metadata.update(
            {
                "cache_id": entry.cache_id,
                "source_prompt": entry.prompt,
                "cache_created_at": entry.created_at,
                "cache_updated_at": entry.updated_at,
                "cache_value": entry.value,
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

