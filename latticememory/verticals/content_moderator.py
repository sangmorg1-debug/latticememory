"""LatticeContentModerator — moderation decision reuse via semantic fingerprinting.

New version of: PhotoDNA / per-post ML content moderation pipelines.

Current tools:
  - Exact hash (PhotoDNA, MD5): misses rewrites and paraphrase variants
  - Per-post ML inference: correct but ~$0.001/post * 500M posts/day = $500K/day

E8 sits between them: key violating content once, then every semantic near-
equivalent inherits the decision via O(1) Hamming lookup. The audit log ties
every auto-applied decision to the specific precedent it was derived from.

Decisions are: 'approved', 'rejected', or 'pending_review'.
New content defaults to 'pending_review' unless it semantically matches a
previously decided entry.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latticememory.semantic_cache import RFSnapSemanticCache


@dataclass
class ModerationResult:
    content_id: str
    status: str                  # 'approved' | 'rejected' | 'pending_review'
    decision: Any | None
    precedent_content_id: str | None
    hamming_distance: int
    auto_applied: bool


class LatticeContentModerator:
    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        audit_log_path: str | None = None,
    ) -> None:
        self._cache = cache
        self._audit_path = Path(audit_log_path) if audit_log_path else None
        self._pending: dict[str, str] = {}  # content_id → content_text

    def submit(
        self,
        content: str,
        content_id: str | None = None,
    ) -> ModerationResult:
        """Submit content for moderation. Returns immediately if a precedent exists."""
        cid = content_id or str(uuid.uuid4())
        result = self._cache.get(content)

        if result.hit and result.value is not None:
            decision = result.value
            status = decision.get("status", "pending_review") if isinstance(decision, dict) else str(decision)
            self._append_audit(cid, status, result.cache_id, result.hamming_distance, auto=True)
            return ModerationResult(
                content_id=cid,
                status=status,
                decision=decision,
                precedent_content_id=result.cache_id,
                hamming_distance=result.hamming_distance,
                auto_applied=True,
            )

        # Novel content: queue for review
        self._pending[cid] = content
        self._append_audit(cid, "pending_review", None, -1, auto=False)
        return ModerationResult(
            content_id=cid,
            status="pending_review",
            decision=None,
            precedent_content_id=None,
            hamming_distance=-1,
            auto_applied=False,
        )

    def record_decision(
        self,
        content_text: str,
        decision: str,
        reviewer_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        """Record a human moderation decision so future near-duplicates inherit it."""
        entry = self._cache.put(
            content_text,
            value={"status": decision, "reviewer": reviewer_id, "decided_at": time.time()},
            metadata={"reviewer_id": reviewer_id or ""},
            ttl_seconds=ttl_seconds,
        )
        self._pending.pop(entry.cache_id, None)
        self._append_audit(entry.cache_id, decision, None, -1, auto=False, reviewer=reviewer_id)
        return entry.cache_id

    def audit_log(self, since: float | None = None) -> list[dict]:
        """Return all audit entries, optionally filtered by timestamp."""
        if self._audit_path is None or not self._audit_path.exists():
            return []
        entries = []
        with open(self._audit_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if since is None or e.get("timestamp", 0) >= since:
                        entries.append(e)
        return entries

    def pending_count(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------

    def _append_audit(
        self,
        content_id: str,
        status: str,
        precedent_id: str | None,
        hamming_distance: int,
        auto: bool,
        reviewer: str | None = None,
    ) -> None:
        if self._audit_path is None:
            return
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "content_id": content_id,
            "status": status,
            "precedent_content_id": precedent_id,
            "hamming_distance": hamming_distance,
            "auto_applied": auto,
            "reviewer": reviewer,
            "timestamp": time.time(),
        }
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
