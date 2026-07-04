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
import hashlib
import math
import time
from typing import Any, Iterator

from latticememory.semantic_cache import SemanticCacheEntry


class _RedisEntriesProxy:
    """A MutableMapping-like proxy over a Redis hash that stores SemanticCacheEntry values.

    The interface matches what RFSnapSemanticCache._entries expects:
      - __getitem__, __setitem__, __delitem__, __contains__, __len__, values(), items(), get()

    Entries are serialized as JSON. The `lattice_key` bytes field is hex-encoded.
    """

    def __init__(
        self,
        redis_client: Any | list[Any],
        namespace: str,
        ttl: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._clients = list(redis_client) if isinstance(redis_client, list) else [redis_client]
        if not self._clients:
            raise ValueError("redis_client must contain at least one client")
        self._r = self._clients[0]
        self._ns = namespace
        self._ttl = ttl  # seconds; None = no expiry
        self._max_entries = max_entries
        self._max_entries_per_shard = (
            max(1, math.ceil(max_entries / len(self._clients))) if max_entries else None
        )

    def _client_and_idx(self, cache_id: str) -> tuple[Any, int]:
        digest = hashlib.md5(cache_id.encode("utf-8")).digest()
        shard_idx = int.from_bytes(digest, byteorder="big") % len(self._clients)
        return self._clients[shard_idx], shard_idx

    def _key(self, cache_id: str) -> str:
        return f"{self._ns}:{cache_id}"

    def _lru_key(self, shard_idx: int) -> str:
        return f"{self._ns}:shard:{shard_idx}:lru"

    def _is_lru_key(self, key: bytes | str) -> bool:
        key_str = key.decode() if isinstance(key, bytes) else key
        return key_str.startswith(f"{self._ns}:shard:") and key_str.endswith(":lru")

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
        client, _ = self._client_and_idx(str(cache_id))
        return bool(client.exists(self._key(str(cache_id))))

    def __getitem__(self, cache_id: str) -> SemanticCacheEntry:
        client, shard_idx = self._client_and_idx(cache_id)
        data = client.get(self._key(cache_id))
        if data is None:
            raise KeyError(cache_id)
        self._touch(client, shard_idx, cache_id)
        return self._deserialize(data)

    def __setitem__(self, cache_id: str, entry: SemanticCacheEntry) -> None:
        client, shard_idx = self._client_and_idx(cache_id)
        serialized = self._serialize(entry)
        if self._ttl:
            client.setex(self._key(cache_id), self._ttl, serialized)
        else:
            client.set(self._key(cache_id), serialized)
        self._touch(client, shard_idx, cache_id)
        self._evict_if_needed(client, shard_idx)

    def __delitem__(self, cache_id: str) -> None:
        client, shard_idx = self._client_and_idx(cache_id)
        client.delete(self._key(cache_id))
        if hasattr(client, "zrem"):
            client.zrem(self._lru_key(shard_idx), cache_id)

    def _touch(self, client: Any, shard_idx: int, cache_id: str) -> None:
        if not hasattr(client, "zadd"):
            return
        client.zadd(self._lru_key(shard_idx), {cache_id: time.time()})

    def _evict_if_needed(self, client: Any, shard_idx: int) -> None:
        if self._max_entries_per_shard is None:
            return
        if not all(hasattr(client, attr) for attr in ("zcard", "zrange", "zrem")):
            return
        lru_key = self._lru_key(shard_idx)
        count = int(client.zcard(lru_key))
        overflow = count - self._max_entries_per_shard
        if overflow <= 0:
            return
        oldest = client.zrange(lru_key, 0, overflow - 1)
        for member in oldest:
            cache_id = member.decode() if isinstance(member, bytes) else member
            client.delete(self._key(cache_id))
            client.zrem(lru_key, cache_id)

    def __len__(self) -> int:
        # SCAN is O(N) but avoids blocking KEYS command in production
        count = 0
        for client in self._clients:
            cursor = 0
            pattern = f"{self._ns}:*"
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                count += sum(1 for key in keys if not self._is_lru_key(key))
                if cursor == 0:
                    break
        return count

    def get(self, cache_id: str, default: Any = None) -> SemanticCacheEntry | None:
        try:
            return self[cache_id]
        except KeyError:
            return default

    def values(self) -> Iterator[SemanticCacheEntry]:
        for client in self._clients:
            pattern = f"{self._ns}:*"
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    if self._is_lru_key(key):
                        continue
                    data = client.get(key)
                    if data:
                        yield self._deserialize(data)
                if cursor == 0:
                    break

    def items(self) -> Iterator[tuple[str, SemanticCacheEntry]]:
        for entry in self.values():
            yield entry.cache_id, entry

    def keys(self) -> Iterator[str]:
        pattern = f"{self._ns}:*"
        prefix_len = len(self._ns) + 1
        for client in self._clients:
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    if self._is_lru_key(key):
                        continue
                    key_str = key.decode() if isinstance(key, bytes) else key
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
        Redis connection URL, a comma-separated URL list, or a list of URLs.
        Multiple URLs shard cache entries by cache id. Default
        ``redis://localhost:6379``.
    namespace:
        Key prefix used to isolate this deployment's entries.
        Use different namespaces for different domains or environments.
    ttl:
        Optional TTL in seconds for each cache entry. Default None (no expiry).
    max_entries:
        Optional approximate maximum entries, distributed evenly across shards
        with per-shard LRU eviction.
    db:
        Redis database index. Default 0.
    """

    def __init__(
        self,
        redis_url: str | list[str] = "redis://localhost:6379",
        namespace: str = "lattice",
        ttl: int | None = None,
        max_entries: int | None = None,
        db: int = 0,
    ) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "Redis support requires: pip install 'lattice-memory-e8[redis]'"
            ) from exc

        if isinstance(redis_url, str):
            urls = [url.strip() for url in redis_url.split(",") if url.strip()]
        else:
            urls = list(redis_url)
        if not urls:
            raise ValueError("redis_url must contain at least one URL")

        self._clients = [
            redis_lib.from_url(url, db=db, decode_responses=False) for url in urls
        ]
        self._client = self._clients[0]
        self.namespace = namespace
        self.ttl = ttl
        self.max_entries = max_entries

    def make_entries_proxy(self) -> _RedisEntriesProxy:
        """Return a proxy object that can replace RFSnapSemanticCache._entries."""
        return _RedisEntriesProxy(
            self._clients,
            self.namespace,
            self.ttl,
            max_entries=self.max_entries,
        )

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
        deleted = 0
        for client in self._clients:
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += client.delete(*keys)
                if cursor == 0:
                    break
        return deleted

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            for client in self._clients:
                client.ping()
            return True
        except Exception:
            return False


def patch_cache_with_redis(
    cache: Any,
    redis_url: str | list[str] = "redis://localhost:6379",
    namespace: str = "lattice",
    ttl: int | None = None,
    max_entries: int | None = None,
) -> LatticeRedisStore:
    """Convenience function: patch a cache instance and return the store object."""
    store = LatticeRedisStore(
        redis_url=redis_url,
        namespace=namespace,
        ttl=ttl,
        max_entries=max_entries,
    )
    store.patch_cache(cache)
    return store
