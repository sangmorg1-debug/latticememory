"""LatticeSemanticRateLimiter — rate limiting by semantic intent, not exact string.

Problem with string-based rate limiting:
  A user who is rate-limited on "What is the stock price of Apple?"
  can bypass it immediately with "Tell me Apple's current share value."
  These are the same intent — they should count against the same limit.

This rate limiter uses E8 lattice keys to group requests by semantic intent.
All prompts that hash to the same E8 cell are counted together.

Features:
  - Per-client limits: each client_id gets its own counter per intent
  - Global limits: enforce a total cap across all clients for a given intent
  - Sliding window: old requests expire and don't count against the limit
  - Configurable: limit (max requests) and window_seconds
  - Thread-safe: uses a simple in-process lock (for single-process deployments)

Usage:
    limiter = LatticeSemanticRateLimiter(cache, limit=5, window_seconds=60.0)

    result = limiter.check("What is the stock price of Apple?", client_id="user-42")
    if not result.allowed:
        return f"Rate limited. Retry after {result.retry_after:.0f}s"
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latticememory.semantic_cache import RFSnapSemanticCache


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RateLimitResult:
    allowed: bool
    count: int            # number of requests in this window for this (client, intent)
    limit: int
    window_seconds: float
    client_id: str
    intent_key: str       # hex of the E8 lattice key for this prompt
    retry_after: float    # seconds until the oldest request in the window expires (0.0 if allowed)
    remaining: int        # requests remaining in window (0 if blocked)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LatticeSemanticRateLimiter:
    """Semantic intent rate limiter.

    Parameters
    ----------
    cache:
        RFSnapSemanticCache used to compute E8 keys for prompts.
        The cache is read-only — no entries are written.
    limit:
        Maximum number of requests per (client_id, intent_key) per window.
    window_seconds:
        Length of the sliding window in seconds.
    global_limit:
        If set, also enforce this cap across ALL clients per intent_key.
        None (default) = no global cap.
    """

    def __init__(
        self,
        cache: "RFSnapSemanticCache",
        *,
        limit: int = 10,
        window_seconds: float = 60.0,
        global_limit: int | None = None,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        self._cache = cache
        self._limit = limit
        self._window = window_seconds
        self._global_limit = global_limit

        # {(client_id, intent_key): [timestamp, ...]}
        self._buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        # {intent_key: [timestamp, ...]}  — for global limiting
        self._global_buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Stats
        self._total_allowed = 0
        self._total_blocked = 0

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(self, prompt: str, client_id: str = "default") -> RateLimitResult:
        """Check whether a request from client_id for this prompt is allowed.

        This method is safe to call concurrently. It uses a sliding window:
        requests older than window_seconds do not count toward the limit.
        """
        intent_key = self._get_intent_key(prompt)
        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            bucket_key = (client_id, intent_key)
            bucket = self._buckets[bucket_key]

            # Evict expired timestamps
            while bucket and bucket[0] <= cutoff:
                bucket.pop(0)

            count = len(bucket)

            # Check per-client limit
            if count >= self._limit:
                oldest = bucket[0]
                retry_after = (oldest + self._window) - now
                self._total_blocked += 1
                return RateLimitResult(
                    allowed=False,
                    count=count,
                    limit=self._limit,
                    window_seconds=self._window,
                    client_id=client_id,
                    intent_key=intent_key,
                    retry_after=max(0.0, retry_after),
                    remaining=0,
                )

            # Check global limit (if configured)
            if self._global_limit is not None:
                g_bucket = self._global_buckets[intent_key]
                while g_bucket and g_bucket[0] <= cutoff:
                    g_bucket.pop(0)
                g_count = len(g_bucket)
                if g_count >= self._global_limit:
                    oldest_g = g_bucket[0]
                    retry_after = (oldest_g + self._window) - now
                    self._total_blocked += 1
                    return RateLimitResult(
                        allowed=False,
                        count=g_count,
                        limit=self._global_limit,
                        window_seconds=self._window,
                        client_id=client_id,
                        intent_key=intent_key,
                        retry_after=max(0.0, retry_after),
                        remaining=0,
                    )

            # Allowed — record this request
            bucket.append(now)
            if self._global_limit is not None:
                self._global_buckets[intent_key].append(now)

            self._total_allowed += 1
            remaining = self._limit - (count + 1)
            return RateLimitResult(
                allowed=True,
                count=count + 1,
                limit=self._limit,
                window_seconds=self._window,
                client_id=client_id,
                intent_key=intent_key,
                retry_after=0.0,
                remaining=remaining,
            )

    def reset(self, client_id: str | None = None, intent_key: str | None = None) -> int:
        """Reset counters for a specific client, intent, or all.

        Returns the number of window buckets cleared.
        """
        cleared = 0
        with self._lock:
            if client_id is None and intent_key is None:
                cleared = len(self._buckets) + len(self._global_buckets)
                self._buckets.clear()
                self._global_buckets.clear()
            elif client_id is not None and intent_key is not None:
                key = (client_id, intent_key)
                if key in self._buckets:
                    del self._buckets[key]
                    cleared = 1
            elif client_id is not None:
                keys = [k for k in self._buckets if k[0] == client_id]
                for k in keys:
                    del self._buckets[k]
                cleared = len(keys)
            else:
                # intent_key only
                keys = [k for k in self._buckets if k[1] == intent_key]
                for k in keys:
                    del self._buckets[k]
                cleared = len(keys)
                if intent_key in self._global_buckets:
                    del self._global_buckets[intent_key]
                    cleared += 1
        return cleared

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return current rate limiter statistics."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            active_intents = sum(
                1 for bucket in self._buckets.values()
                if any(t > cutoff for t in bucket)
            )
        return {
            "total_allowed": self._total_allowed,
            "total_blocked": self._total_blocked,
            "limit": self._limit,
            "window_seconds": self._window,
            "global_limit": self._global_limit,
            "active_intent_client_pairs": active_intents,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_intent_key(self, prompt: str) -> str:
        """Return the hex E8 lattice key for this prompt.

        Uses cache.get() to compute the key without storing an entry.
        The lattice_key field is populated on both hit and miss.
        """
        result = self._cache.get(prompt)
        if result.lattice_key:
            return result.lattice_key.hex()
        # Fallback: hash the raw text (exact-match only, no semantic grouping)
        import hashlib
        return hashlib.sha1(prompt.encode()).hexdigest()
