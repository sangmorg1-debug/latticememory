"""LatticeSOCMonitor — semantic alert deduplication and coverage-gap detection.

New version of: PagerDuty / OpsGenie / SIEM correlation rules.

Current tools deduplicate alerts with brittle regex rules written by engineers.
The same incident fired by five different monitoring tools pages the on-call
engineer five times. Novel attack patterns only surface in post-incident reviews.

This replaces manual correlation rules with two semantic layers:

  Layer 1 (dedup): alerts that snap to the same E8 cell are the same incident.
                   One page, one ticket, regardless of which tool fired.

  Layer 2 (gap detection): alerts that don't match any known pattern are logged
                           to a LatticeFlywheel. cluster_gaps() surfaces novel
                           attack techniques not covered by existing playbooks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latticememory.flywheel import MissCluster
    from latticememory.semantic_cache import RFSnapSemanticCache


@dataclass
class AlertResult:
    alert_id: str
    is_duplicate: bool
    canonical_alert_id: str | None
    is_known_pattern: bool
    decision: Any | None
    hamming_distance: int


class LatticeSOCMonitor:
    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        gap_log_path: str | None = None,
        incident_ttl_seconds: float = 3600.0,
    ) -> None:
        self._cache = cache
        self._ttl = incident_ttl_seconds
        self._counter = 0

        self._flywheel = None
        if gap_log_path is not None:
            from latticememory.flywheel import LatticeFlywheel
            self._flywheel = LatticeFlywheel(gap_log_path)

    def ingest_alert(
        self,
        alert_text: str,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> AlertResult:
        self._counter += 1
        alert_id = f"alert-{int(time.time())}-{self._counter}"

        result = self._cache.get(alert_text)

        if result.hit:
            entry = self._cache._entries.get(result.cache_id)
            is_playbook = bool(entry and entry.metadata.get("is_playbook"))
            return AlertResult(
                alert_id=alert_id,
                is_duplicate=True,
                canonical_alert_id=result.cache_id,
                is_known_pattern=is_playbook,
                decision=result.value,
                hamming_distance=result.hamming_distance,
            )

        key_hex = result.lattice_key.hex() if result.lattice_key else ""
        if self._flywheel is not None:
            self._flywheel.log_miss(
                alert_text,
                e8_key_hex=key_hex,
                metadata={"source": source, **(metadata or {})},
            )

        entry = self._cache.put(
            alert_text,
            value={"alert_id": alert_id, "source": source},
            metadata={"is_playbook": False, **(metadata or {})},
            ttl_seconds=self._ttl,
        )
        return AlertResult(
            alert_id=alert_id,
            is_duplicate=False,
            canonical_alert_id=None,
            is_known_pattern=False,
            decision=None,
            hamming_distance=-1,
        )

    def add_known_pattern(
        self,
        pattern_text: str,
        playbook: Any,
        ttl_seconds: float | None = None,
    ) -> str:
        """Register a known incident type + playbook. Permanent unless ttl_seconds is set."""
        entry = self._cache.put(
            pattern_text,
            value=playbook,
            metadata={"is_playbook": True},
            ttl_seconds=ttl_seconds,
        )
        return entry.cache_id

    def top_novel_patterns(self, n: int = 10) -> list["MissCluster"]:
        """Top N alert clusters not covered by any playbook."""
        if self._flywheel is None:
            return []
        return self._flywheel.top_gaps(n=n, min_cluster_size=1)

    def export_gap_review_queue(self, path: str) -> int:
        """Export novel alert clusters to JSON for analyst review."""
        if self._flywheel is None:
            return 0
        return self._flywheel.export_review_queue(path, min_cluster_size=1)

    def stats(self) -> dict:
        base = {"alerts_ingested": self._counter, "known_patterns": self._cache.size}
        if self._flywheel is not None:
            base["flywheel"] = self._flywheel.stats()
        return base
