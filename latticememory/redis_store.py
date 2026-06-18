"""LatticeRedisStore — Redis-backed cache entry storage for RFSnapSemanticCache.

Replaces the in-process `_entries` dict with a Redis hash so that multiple
proxy instances share a single cache without coordination. Each cache entry is
stored as a JSON blob at `{namespace}:{e8_key_hex}`.

Usage::

    from latticememory.redis_store import LatticeRedisStore, patch_cache_with_redis
    from latticememory.semantic_cache import RFSnapSemanticCache

    cache = RFSnapSemanticCache(...)
    patch_cache_with_redis(cache, redis_url="redis://localhost:6379", namespace="helpdesk")

Or via the proxy::

    proxy = LatticeLLMProxy(
        upstream_url="...",
        upstream_api_key="sk-...",
        redis_url="redis://localhost:6379",
        redis_namespace="helpdesk",
    )

Requirements::

    pip install 'lattice-memory-e8[redis]'
    # which adds: redis>=4.0.0
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator

from latticememory.semantic_cache import SemanticCacheEntry


class _RedisEntriesProxy:
    """A MutableMapping-like proxy over a Redis hash that stores SemanticCacheEntry values.

    The interface matches what RFSnapSemanticCache._entries expects:
      - __getitem__, __setitem__, __delitem__, __contains__, __len__, values(), items(), get()

    Entries are serialized as JSON. The `lattice_key` bytes field is hex-encoded.
    """

    def __init__(self, redis_client: Any, namespace: str, ttl: int | None = None) -> None:
        self._r = redis_client
        self._ns = namespace
        self._ttl = ttl  # seconds; None = no expiry

    def _key(self, cache_id: str) -> str:
        return f"{self._ns}:{cache_id}"

    def _serialize(self, entry: SemanticCacheEntry) -> str:
        return json.dumps({
            "cache_id":    entry.cache_id,
            "prompt":      entry.prompt,
            "value":       entry.value,
            "lattice_key": entry.lattice_key.hex() if entry.lattice_key else None,
            "metadata":    entry.metadata,
            "created_at":  entry.created_at,
            "updated_at":  entry.updated_at,
            "ttl_seconds": entry.ttl_seconds,
        })

    def _deserialize(self, data: bytes | str) -> SemanticCacheEntry:
        d = json.loads(data)
        lk = bytes.fromhex(d["lattice_key"]) if d.get("lattice_key") else b""
        return SemanticCacheEntry(
            cache_id=d["cache_id"],
            prompt=d["prompt"],
            value=d["value"],
            lattice_key=lk,
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            ttl_seconds=d.get("ttl_seconds"),
        )

    def __contains__(self, cache_id: object) -> bool:
        return bool(self._r.exists(self._key(str(cache_id))))

    def __getitem__(self, cache_id: str) -> SemanticCacheEntry:
        data = self._r.get(self._key(cache_id))
        if data is None:
            raise KeyError(cache_id)
        return self._deserialize(data)

    def __setitem__(self, cache_id: str, entry: SemanticCacheEntry) -> None:
        serialized = self._serialize(entry)
        if self._ttl:
            self._r.setex(self._key(cache_id), self._ttl, serialized)
        else:
            self._r.set(self._key(cache_id), serialized)

    def __delitem__(self, cache_id: str) -> None:
        self._r.delete(self._key(cache_id))

    def __len__(self) -> int:
        # SCAN is O(N) but avoids blocking KEYS command in production
        cursor = 0
        count = 0
        pattern = f"{self._ns}:*"
        while True:
            cursor, keys = self._r.scan(cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def get(self, cache_id: str, default: Any = None) -> SemanticCacheEntry | None:
        try:
            return self[cache_id]
        except KeyError:
            return default

    def values(self) -> Iterator[SemanticCacheEntry]:
        pattern = f"{self._ns}:*"
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match=pattern, count=100)
            for k in keys:
                data = self._r.get(k)
                if data:
                    yield self._deserialize(data)
            if cursor == 0:
                break

    def items(self) -> Iterator[tuple[str, SemanticCacheEntry]]:
        for entry in self.values():
            yield entry.cache_id, entry

    def keys(self) -> Iterator[str]:
        pattern = f"{self._ns}:*"
        cursor = 0
        prefix_len = len(self._ns) + 1
        while True:
            cursor, keys = self._r.scan(cursor, match=pattern, count=100)
            for k in keys:
                key_str = k.decode() if isinstance(k, bytes) else k
                yield key_str[prefix_len:]  # strip namespace prefix
            if cursor == 0:
                break


class LatticeRedisStore:
    """Factory for Redis-backed cache entry storage.

    Call ``patch_cache()`` to swap the in-memory dict in a live
    ``RFSnapSemanticCache`` for a Redis-backed proxy.

    Parameters
    ----------
    redis_url:
        Redis connection URL. Default ``redis://localhost:6379``.
    namespace:
        Key prefix used to isolate this deployment's entries.
        Use different namespaces for different domains or environments.
    ttl:
        Optional TTL in seconds for each cache entry. Default None (no expiry).
    db:
        Redis database index. Default 0.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        namespace: str = "lattice",
        ttl: int | None = None,
        db: int = 0,
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "Redis support requires: pip install 'lattice-memory-e8[redis]'"
            ) from exc

        self._client = redis_lib.from_url(redis_url, db=db, decode_responses=False)
        self.namespace = namespace
        self.ttl = ttl

    def make_entries_proxy(self) -> _RedisEntriesProxy:
        """Return a proxy object that can replace RFSnapSemanticCache._entries."""
        return _RedisEntriesProxy(self._client, self.namespace, self.ttl)

    def patch_cache(self, cache: Any) -> None:
        """Swap the in-memory _entries dict in `cache` for Redis-backed storage.

        This is a live operation — existing in-memory entries are migrated to Redis.
        """
        existing_entries: dict = cache._entries
        proxy = self.make_entries_proxy()

        # Migrate any in-memory entries that aren't already in Redis
        for cache_id, entry in existing_entries.items():
            if cache_id not in proxy:
                proxy[cache_id] = entry

        cache._entries = proxy

    def flush(self) -> int:
        """Delete all entries in this namespace. Returns count deleted."""
        pattern = f"{self.namespace}:*"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=pattern, count=100)
            if keys:
                deleted += self._client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            self._client.ping()
            return True
        except Exception:
            return False


def patch_cache_with_redis(
    cache: Any,
    redis_url: str = "redis://localhost:6379",
    namespace: str = "lattice",
    ttl: int | None = None,
) -> LatticeRedisStore:
    """Convenience function: patch a cache instance and return the store object."""
    store = LatticeRedisStore(redis_url=redis_url, namespace=namespace, ttl=ttl)
    store.patch_cache(cache)
    return store
