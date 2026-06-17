"""LatticeTicketAnalyzer — semantic help desk with automatic gap detection.

New version of: Zendesk / Freshdesk / ServiceNow knowledge base search.

Current tools: keyword search returns no results for paraphrase variants of
answered questions. "Top unanswered queries" is reported as a raw list, not
clustered by intent. Documentation teams have no signal for what to write next.

This inverts the architecture: the knowledge base is a semantic cache. Every
question that snaps to a known answer gets it instantly. Every miss is logged
and clustered — the flywheel outputs a prioritized list of documentation gaps
(cluster representative + count), not an unsorted query dump.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latticememory.flywheel import MissCluster
    from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry


@dataclass
class TicketResult:
    ticket_id: str
    suggested_answer: Any | None
    hit_type: str              # 'exact', 'approx', 'miss'
    cache_id: str | None
    is_documentation_gap: bool


class LatticeTicketAnalyzer:
    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        gap_log_path: str | None = None,
    ) -> None:
        self._cache = cache
        self._counter = 0

        self._flywheel = None
        if gap_log_path is not None:
            from latticememory.flywheel import LatticeFlywheel
            self._flywheel = LatticeFlywheel(gap_log_path)

    def ingest_ticket(self, ticket_text: str) -> TicketResult:
        self._counter += 1
        ticket_id = f"ticket-{self._counter}"

        result = self._cache.get(ticket_text)

        if result.hit:
            hit_type = "approx" if result.hamming_distance >= 0 else "exact"
            return TicketResult(
                ticket_id=ticket_id,
                suggested_answer=result.value,
                hit_type=hit_type,
                cache_id=result.cache_id,
                is_documentation_gap=False,
            )

        # Miss — log to flywheel
        if self._flywheel is not None:
            key_hex = result.lattice_key.hex() if result.lattice_key else ""
            self._flywheel.log_miss(ticket_text, e8_key_hex=key_hex)

        return TicketResult(
            ticket_id=ticket_id,
            suggested_answer=None,
            hit_type="miss",
            cache_id=None,
            is_documentation_gap=True,
        )

    def add_resolved_ticket(
        self,
        question: str,
        answer: Any,
        metadata: dict | None = None,
    ) -> "SemanticCacheEntry":
        """Add a resolved Q&A pair so future similar tickets are auto-answered."""
        return self._cache.put(question, value=answer, metadata=metadata)

    def top_documentation_gaps(self, n: int = 10) -> list["MissCluster"]:
        """Return the N largest clusters of unanswered questions."""
        if self._flywheel is None:
            return []
        return self._flywheel.top_gaps(n=n, min_cluster_size=1)

    def export_review_queue(self, path: str) -> int:
        """Write gap clusters to JSON for the documentation team."""
        if self._flywheel is None:
            return 0
        return self._flywheel.export_review_queue(path, min_cluster_size=1)

    def load_reviewed(self, path: str) -> int:
        """Ingest answered items from a reviewed queue back into the cache."""
        if self._flywheel is None:
            raise ValueError("gap_log_path must be set to use load_reviewed()")
        return self._flywheel.load_reviewed(path, self._cache)

    def stats(self) -> dict:
        s = {"tickets_ingested": self._counter, "answered_entries": self._cache.size}
        if self._flywheel is not None:
            s["flywheel"] = self._flywheel.stats()
        return s
