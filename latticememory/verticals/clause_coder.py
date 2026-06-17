"""LatticeClauseCoder — legal clause review propagation.

New version of: Relativity / Luminance / Kira eDiscovery near-duplicate coding.

Current tools propagate review decisions via exact-string fingerprinting. The
same indemnification clause reworded in a new contract is reviewed from scratch.
Relativity's "near-duplicate coding" still requires the reviewer to manually
link the documents.

E8 keys extend propagation to semantic near-equivalents: same clause in
different words → same key → same review decision applied automatically. The
auto-coding rate (fraction of clauses that match a prior decision) is a direct
measure of how repetitive the document set is — typically 60-80% in M&A review.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry


@dataclass
class ClauseDecision:
    clause_text: str
    decision: str | None
    precedent_id: str | None
    auto_coded: bool
    hamming_distance: int


@dataclass
class DocumentCodingResult:
    doc_id: str
    auto_coded: list[ClauseDecision]
    needs_review: list[str]

    @property
    def auto_rate(self) -> float:
        total = len(self.auto_coded) + len(self.needs_review)
        return len(self.auto_coded) / total if total else 0.0


def _default_splitter(text: str) -> list[str]:
    """Split on blank lines or numbered/lettered section markers."""
    # Split on double newlines first
    raw = re.split(r"\n\s*\n", text)
    # Further split on common legal section starters: "1.", "a.", "(i)"
    clauses = []
    for block in raw:
        block = block.strip()
        if not block:
            continue
        sub = re.split(r"(?m)^(?:\d+\.|[a-z]\.|[ivxIVX]+\.|\([ivxIVX]+\))\s+", block)
        for s in sub:
            s = s.strip()
            if len(s) >= 20:
                clauses.append(s)
    return clauses or [text.strip()]


class LatticeClauseCoder:
    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        clause_splitter: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._cache = cache
        self._splitter = clause_splitter or _default_splitter
        self._review_queue: list[tuple[str, str]] = []  # (doc_id, clause_text)

    def code_document(
        self,
        document_text: str,
        doc_id: str | None = None,
    ) -> DocumentCodingResult:
        """Split document into clauses and apply prior decisions where available."""
        doc_id = doc_id or str(uuid.uuid4())
        clauses = self._splitter(document_text)
        auto_coded: list[ClauseDecision] = []
        needs_review: list[str] = []

        for clause in clauses:
            result = self._cache.get(clause)
            if result.hit and result.value is not None:
                decision = result.value
                auto_coded.append(ClauseDecision(
                    clause_text=clause,
                    decision=decision if isinstance(decision, str) else str(decision),
                    precedent_id=result.cache_id,
                    auto_coded=True,
                    hamming_distance=result.hamming_distance,
                ))
            else:
                needs_review.append(clause)
                self._review_queue.append((doc_id, clause))

        return DocumentCodingResult(
            doc_id=doc_id,
            auto_coded=auto_coded,
            needs_review=needs_review,
        )

    def record_decision(
        self,
        clause_text: str,
        decision: str,
        notes: str | None = None,
        ttl_seconds: float | None = None,
    ) -> "SemanticCacheEntry":
        """Record a review decision so future similar clauses inherit it."""
        return self._cache.put(
            clause_text,
            value=decision,
            metadata={"notes": notes or ""},
            ttl_seconds=ttl_seconds,
        )

    def export_review_queue(self, path: str) -> int:
        """Write pending clauses to a JSON file for reviewer assignment."""
        queue = [
            {"doc_id": doc_id, "clause_text": clause, "decision": None, "notes": None}
            for doc_id, clause in self._review_queue
        ]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
        return len(queue)

    def load_reviewed(self, path: str) -> int:
        """Ingest completed review decisions from the exported queue file."""
        items = json.loads(Path(path).read_text(encoding="utf-8"))
        added = 0
        for item in items:
            decision = item.get("decision")
            clause = item.get("clause_text", "")
            if decision and clause:
                self.record_decision(clause, decision, notes=item.get("notes"))
                added += 1
        return added

    def clear_queue(self) -> None:
        self._review_queue.clear()
