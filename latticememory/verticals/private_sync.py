"""LatticePrivateSync — privacy-preserving semantic knowledge sharing.

New version of: Private Set Intersection protocols / federated threat intelligence.

Current approaches:
  - Share nothing (privacy preserved, but duplicate effort, no shared learning)
  - Share raw text/reports (full content disclosed, privacy regulations violated)
  - Cryptographic PSI (correct but high overhead, no semantic equivalence)

E8 keys are semantic fingerprints: they reveal nothing about the content, but
two organizations that independently keyed the same concept produce the same
key. Sharing key manifests (lists of 128-byte keys) reveals only *that* shared
knowledge exists, not *what* it is.

Workflow:
  1. Each org calls export_key_manifest() → list of 256-char hex strings
  2. Orgs exchange manifests over any channel (public is fine — keys are opaque)
  3. find_overlap(remote_manifest) → keys both parties have already know
  4. find_gaps(remote_manifest) → keys remote has that we don't yet
  5. For gap keys the receiving org explicitly requests, export_documents_for_keys()
     returns the text+metadata for those specific keys — content only flows for
     explicitly requested, mutually agreed keys.

Use cases: security vendor threat sharing, hospital network rare-case sharing,
financial institution fraud pattern sharing — without violating data regulations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry


@dataclass
class SyncReport:
    overlap_count: int
    gap_count: int       # remote has, we don't
    surplus_count: int   # we have, remote doesn't
    overlap_keys: list[str] = field(default_factory=list)
    gap_keys: list[str] = field(default_factory=list)
    surplus_keys: list[str] = field(default_factory=list)


class LatticePrivateSync:
    def __init__(self, cache: "RFSnapSemanticCache") -> None:
        self._cache = cache

    def export_key_manifest(self) -> list[str]:
        """Return a sorted list of E8 key hex strings for all cached entries.

        Safe to share publicly: the hex strings are semantic fingerprints with
        no recoverable plaintext.
        """
        keys = []
        for entry in self._cache._entries.values():
            if entry.lattice_key:
                keys.append(entry.lattice_key.hex())
        return sorted(set(keys))

    def compare(self, remote_manifest: list[str]) -> SyncReport:
        """Compute overlap and gaps between local and remote key manifests."""
        local_set = set(self.export_key_manifest())
        remote_set = set(remote_manifest)
        overlap = local_set & remote_set
        gaps = remote_set - local_set      # remote has, we don't
        surplus = local_set - remote_set   # we have, remote doesn't
        return SyncReport(
            overlap_count=len(overlap),
            gap_count=len(gaps),
            surplus_count=len(surplus),
            overlap_keys=sorted(overlap),
            gap_keys=sorted(gaps),
            surplus_keys=sorted(surplus),
        )

    def export_documents_for_keys(self, key_hexes: list[str]) -> list[dict]:
        """Export full document content for the requested keys.

        Only called for keys that the receiving party has explicitly requested
        after reviewing the manifest overlap. Caller decides transmission security.
        """
        wanted = set(key_hexes)
        docs = []
        for entry in self._cache._entries.values():
            if entry.lattice_key and entry.lattice_key.hex() in wanted:
                docs.append({
                    "key_hex": entry.lattice_key.hex(),
                    "prompt": entry.prompt,
                    "value": entry.value,
                    "metadata": dict(entry.metadata),
                })
        return docs

    def import_documents(self, documents: list[dict]) -> int:
        """Import documents received from a remote peer into the local cache.

        Uses restore_entry() so SQLite persistence is respected.
        """
        from latticememory.semantic_cache import SemanticCacheEntry
        imported = 0
        for doc in documents:
            key_hex = doc.get("key_hex", "")
            if not key_hex:
                continue
            try:
                lattice_key = bytes.fromhex(key_hex)
            except ValueError:
                continue
            entry = SemanticCacheEntry(
                cache_id=self._cache._cache_id_for(lattice_key),
                prompt=doc.get("prompt", ""),
                value=doc.get("value"),
                lattice_key=lattice_key,
                metadata=doc.get("metadata", {}),
            )
            self._cache.restore_entry(entry)
            imported += 1
        return imported
