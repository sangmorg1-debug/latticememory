from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import Any

from latticememory.text_runtime import RFSnapTextMemory
from latticememory.memory import MemoryDocument


class LatticeStreamDedup:
    """Real-time streaming semantic deduplication using E8 lattice addresses.

    Allows sliding-window semantic deduplication of high-frequency streams
    (e.g., news wires, social feeds) with O(1) E8 snapping, falling back to
    neighborhood Hamming searches for near-duplicates.
    """

    def __init__(
        self,
        *,
        runtime: RFSnapTextMemory,
        time_window_seconds: float = 3600.0,
        max_entries: int | None = None,
        allow_neighborhood: bool = True,
    ):
        self.runtime = runtime
        self.time_window_seconds = float(time_window_seconds)
        self.max_entries = max_entries
        self.allow_neighborhood = allow_neighborhood

        # History log: deque of (timestamp, lattice_key, doc_id)
        self._history: deque[tuple[float, bytes, str]] = deque()
        # Direct lookup from key to lists of active (timestamp, doc_id) entries
        self._key_to_active: dict[bytes, list[tuple[float, str]]] = {}
        # Keep tracking doc_id to text for callback or canonical resolution
        self._doc_id_to_text: dict[str, str] = {}

    @property
    def size(self) -> int:
        """Returns the number of active items in the sliding window."""
        return len(self._history)

    def process(self, text: str, *, timestamp: float | None = None) -> dict[str, Any]:
        """Process a streaming text item and check for duplication.

        Args:
            text: Raw input text.
            timestamp: Event timestamp (defaults to current time).

        Returns:
            dict containing:
                - is_duplicate (bool)
                - key (str): hex representation of the E8 key
                - doc_id (str): document ID assigned to this text
                - canonical_id (str | None): ID of the matching canonical text
                - match_path (str): 'exact', 'neighborhood', or 'miss'
        """
        now = float(timestamp if timestamp is not None else time.time())
        self._prune(now)

        # Snaps text to its E8 lattice key
        embedding = self.runtime._encode_texts([text])[0]
        lattice_key = self.runtime.memory.lattice_key_for(embedding)
        hex_key = lattice_key.hex()

        # 1. Check exact E8 address key collision
        active_list = self._key_to_active.get(lattice_key)
        if active_list:
            # Match! Return the earliest active doc_id for this key
            canonical_id = active_list[0][1]
            doc_id = "stream-" + hashlib.sha1(f"{hex_key}-{now}".encode()).hexdigest()[:16]
            return {
                "is_duplicate": True,
                "key": hex_key,
                "doc_id": doc_id,
                "canonical_id": canonical_id,
                "match_path": "exact",
            }

        # 2. Check E8 neighborhood (Hamming-1) collision if enabled
        if self.allow_neighborhood:
            # We can use memory's radius search to scan active keys
            from latticememory.memory import MemoryQuery
            query = MemoryQuery(text=text, embedding=embedding, top_k=1)
            res = self.runtime.memory.retrieve(query)
            if res.hits and res.path in {"lattice_exact", "lattice_hamming1"}:
                canonical_id = res.hits[0].doc_id
                # Verify that the matched canonical_id is still active in the sliding window
                if canonical_id in self._doc_id_to_text:
                    doc_id = "stream-" + hashlib.sha1(f"{hex_key}-{now}".encode()).hexdigest()[:16]
                    return {
                        "is_duplicate": True,
                        "key": hex_key,
                        "doc_id": doc_id,
                        "canonical_id": canonical_id,
                        "match_path": "neighborhood",
                    }

        # 3. Miss -> Add new unique concept to active window
        doc_id = "stream-" + hashlib.sha1(f"{hex_key}-{now}".encode()).hexdigest()[:16]
        self._key_to_active.setdefault(lattice_key, []).append((now, doc_id))
        self._history.append((now, lattice_key, doc_id))
        self._doc_id_to_text[doc_id] = text

        # Also add to the runtime memory index so we can retrieve/neighborhood-match it!
        doc = MemoryDocument(doc_id=doc_id, text=text, embedding=embedding, metadata={"stream_timestamp": now})
        self.runtime.memory.add_documents([doc])

        # Second limit: max active entries
        if self.max_entries is not None and len(self._history) > self.max_entries:
            self._evict_oldest()

        return {
            "is_duplicate": False,
            "key": hex_key,
            "doc_id": doc_id,
            "canonical_id": None,
            "match_path": "miss",
        }

    def _prune(self, current_time: float) -> None:
        cutoff = current_time - self.time_window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        if not self._history:
            return
        ts, key, doc_id = self._history.popleft()
        
        # Remove from key list
        if key in self._key_to_active:
            # Filter out this doc_id
            self._key_to_active[key] = [item for item in self._key_to_active[key] if item[1] != doc_id]
            if not self._key_to_active[key]:
                del self._key_to_active[key]
                
        # Remove from doc mapping
        if doc_id in self._doc_id_to_text:
            del self._doc_id_to_text[doc_id]
            
        # Delete from runtime memory too to keep index clean
        if hasattr(self.runtime.memory, "sqlite_store") and self.runtime.memory.sqlite_store is not None:
            self.runtime.memory.sqlite_store.delete_documents([doc_id])
        # Also clean up internal lattice mappings
        if doc_id in self.runtime.memory.lattice._keys:
            del self.runtime.memory.lattice._keys[doc_id]
        if doc_id in self.runtime.memory.lattice._embeddings:
            del self.runtime.memory.lattice._embeddings[doc_id]
        if doc_id in self.runtime.memory.lattice._metadata:
            del self.runtime.memory.lattice._metadata[doc_id]
        if doc_id in self.runtime.memory._texts:
            del self.runtime.memory._texts[doc_id]
        if doc_id in self.runtime.memory._metadata:
            del self.runtime.memory._metadata[doc_id]
        if doc_id in self.runtime.memory._doc_index:
            del self.runtime.memory._doc_index[doc_id]
        self.runtime.memory._doc_ids = [d for d in self.runtime.memory._doc_ids if d != doc_id]
        self.runtime.memory._doc_id_set.discard(doc_id)
        # Rebuild hash_store references
        for k, ids in list(self.runtime.memory.lattice.hash_store.items()):
            if doc_id in ids:
                updated_ids = [i for i in ids if i != doc_id]
                if updated_ids:
                    self.runtime.memory.lattice.hash_store[k] = updated_ids
                else:
                    del self.runtime.memory.lattice.hash_store[k]
