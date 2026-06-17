"""LatticeTrainingCleaner — semantic near-duplicate removal for LLM training datasets.

Problem:
  Large training corpora (web crawls, synthetic Q&A sets, scraped documents)
  contain near-duplicate examples. Near-duplicates:
    - Inflate compute cost (model trains on the same example repeatedly)
    - Bias the model toward repeated phrasings
    - Pollute evaluation splits with contamination from training

Existing tools:
  - MinHash LSH: good but requires tuning; does not capture meaning-level duplicates
  - Pairwise cosine: O(N²), infeasible at scale
  - Exact dedup (MD5): misses paraphrase variants

This cleaner:
  - O(N) dedup via E8 keys — each text is hashed to one cell; collision = duplicate
  - Configurable strictness: exact-cell only, or Hamming-neighborhood (with HammingRouter)
  - Returns kept texts plus a full report of what was removed and why
  - Streaming mode for large datasets that don't fit in memory
  - JSONL export: writes unique texts to disk as they arrive

Usage:
    cleaner = LatticeTrainingCleaner(runtime=my_runtime)

    result = cleaner.clean(["How do I reset my password?",
                            "I forgot my password, what do I do?",  # may be duplicate
                            "What is photosynthesis?"])

    print(result.kept_count, "unique", result.duplicate_count, "removed")
    for d in result.duplicates:
        print(f"  '{d.text[:50]}' duplicates '{d.canonical_text[:50]}'")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DuplicateRecord:
    text: str
    canonical_text: str   # the first-seen text with this E8 key
    canonical_id: str     # cache_id of the canonical entry
    hamming_distance: int # 0 = exact cell match, >0 = approximate neighbor


@dataclass
class CleaningResult:
    total: int
    kept_count: int
    duplicate_count: int
    kept: list[str]
    duplicates: list[DuplicateRecord]
    elapsed_seconds: float

    @property
    def dedup_rate(self) -> float:
        return self.duplicate_count / self.total if self.total else 0.0

    def summary(self) -> str:
        pct = self.dedup_rate * 100
        return (
            f"Total: {self.total}  Kept: {self.kept_count}  "
            f"Duplicates removed: {self.duplicate_count} ({pct:.1f}%)  "
            f"Time: {self.elapsed_seconds:.2f}s"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LatticeTrainingCleaner:
    """Semantic near-duplicate cleaner for training datasets.

    Parameters
    ----------
    cache:
        Pre-built RFSnapSemanticCache. Entries added during cleaning are
        the canonical representatives of each semantic cluster.
    clear_on_init:
        If True (default), evict all existing cache entries before cleaning
        so prior runs do not affect results. Set False to continue a
        previously interrupted cleaning run (deduplicates against already-seen).
    """

    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        clear_on_init: bool = True,
    ) -> None:
        self._cache = cache
        if clear_on_init:
            self._clear_cache()

    # ------------------------------------------------------------------
    # Batch clean
    # ------------------------------------------------------------------

    def clean(self, texts: list[str], *, metadata: list[dict] | None = None) -> CleaningResult:
        """Deduplicate a list of texts. Returns a CleaningResult.

        Parameters
        ----------
        texts:
            The corpus to deduplicate.
        metadata:
            Optional list of per-text metadata dicts (same length as texts).
            Stored alongside canonical entries.
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")

        t0 = time.perf_counter()
        kept: list[str] = []
        duplicates: list[DuplicateRecord] = []

        for i, text in enumerate(texts):
            meta = metadata[i] if metadata else {}
            result = self._cache.get(text)

            if result.hit:
                # Duplicate — find the canonical text
                canonical_text = result.source_prompt or text
                duplicates.append(DuplicateRecord(
                    text=text,
                    canonical_text=canonical_text,
                    canonical_id=result.cache_id or "",
                    hamming_distance=result.hamming_distance,
                ))
            else:
                # Unique — store as canonical
                self._cache.put(text, value={"is_canonical": True}, metadata=meta)
                kept.append(text)

        elapsed = time.perf_counter() - t0
        return CleaningResult(
            total=len(texts),
            kept_count=len(kept),
            duplicate_count=len(duplicates),
            kept=kept,
            duplicates=duplicates,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Streaming mode
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str, metadata: dict | None = None) -> bool:
        """Return True if this text is a duplicate of something already seen.

        Call this one-by-one for streaming / large-dataset use.
        The first occurrence is stored as canonical; subsequent matches return True.
        """
        result = self._cache.get(text)
        if result.hit:
            return True
        self._cache.put(text, value={"is_canonical": True}, metadata=metadata or {})
        return False

    def stream(self, texts: Iterator[str]) -> Iterator[str]:
        """Yield only unique texts from an iterator (streaming dedup)."""
        for text in texts:
            if not self.is_duplicate(text):
                yield text

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def clean_to_jsonl(
        self,
        texts: list[str],
        output_path: str | Path,
        *,
        text_key: str = "text",
        metadata: list[dict] | None = None,
    ) -> CleaningResult:
        """Deduplicate texts and write unique ones to a JSONL file.

        Each line in the output file is a JSON object with at minimum `text_key`.
        Extra fields from `metadata` (if provided) are merged in.
        """
        result = self.clean(texts, metadata=metadata)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for i, text in enumerate(result.kept):
                record: dict = {text_key: text}
                if metadata is not None:
                    # Find original index of this kept text
                    orig_idx = texts.index(text)
                    record.update(metadata[orig_idx])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return result

    def load_and_clean_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "text",
    ) -> CleaningResult:
        """Read a JSONL file, deduplicate by the text field, write unique lines out."""
        lines = Path(input_path).read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        texts = [r[text_key] for r in records]
        metadata = [{k: v for k, v in r.items() if k != text_key} for r in records]

        result = self.clean(texts, metadata=metadata)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for text in result.kept:
                orig_idx = texts.index(text)
                record = {text_key: text, **metadata[orig_idx]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def seen_count(self) -> int:
        """Number of canonical (unique) entries seen so far."""
        return self._cache.size

    def reset(self) -> None:
        """Clear all seen entries, starting fresh."""
        self._clear_cache()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear_cache(self) -> None:
        cache_ids = list(self._cache._entries.keys())
        for cache_id in cache_ids:
            self._cache.delete(cache_id)
