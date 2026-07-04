from __future__ import annotations

import time

from latticememory.redis_store import _RedisEntriesProxy
from latticememory.semantic_cache import SemanticCacheEntry


class MockRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._store:
                deleted += 1
            self._store.pop(key, None)
        return deleted

    def scan(self, cursor: int, match: str | None = None, count: int = 100) -> tuple[int, list[str]]:
        prefix = match.rstrip("*") if match else ""
        return 0, [key for key in self._store if not prefix or key.startswith(prefix)]

    def zadd(self, name: str, mapping: dict[str, float]) -> None:
        zset = self._zsets.setdefault(name, {})
        zset.update(mapping)

    def zcard(self, name: str) -> int:
        return len(self._zsets.get(name, {}))

    def zrange(self, name: str, start: int, end: int) -> list[str]:
        zset = self._zsets.get(name, {})
        sorted_members = sorted(zset, key=zset.get)
        if end == -1:
            return sorted_members[start:]
        return sorted_members[start : end + 1]

    def zrem(self, name: str, member: str) -> None:
        self._zsets.get(name, {}).pop(member, None)

    def zscan_iter(self, name: str):
        for member, score in self._zsets.get(name, {}).items():
            yield member, score


def _entry(cache_id: str) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        cache_id=cache_id,
        prompt=f"prompt {cache_id}",
        value=f"value {cache_id}",
        lattice_key=b"\x00" * 128,
    )


def test_redis_entries_proxy_shards_entries_across_multiple_clients() -> None:
    redis_a = MockRedis()
    redis_b = MockRedis()
    proxy = _RedisEntriesProxy([redis_a, redis_b], namespace="test-ns")

    for index in range(20):
        cache_id = f"entry-{index}"
        proxy[cache_id] = _entry(cache_id)

    assert len(redis_a._store) + len(redis_b._store) == 20
    assert len(redis_a._store) > 0
    assert len(redis_b._store) > 0
    assert len(proxy) == 20
    assert set(proxy.keys()) == {f"entry-{index}" for index in range(20)}


def test_redis_entries_proxy_evicts_least_recently_used_entry() -> None:
    redis = MockRedis()
    proxy = _RedisEntriesProxy(redis, namespace="test-ns", max_entries=3)

    for index in range(3):
        proxy[f"entry-{index}"] = _entry(f"entry-{index}")

    assert len(proxy) == 3

    time.sleep(0.01)
    _ = proxy["entry-1"]
    proxy["entry-3"] = _entry("entry-3")

    assert len(proxy) == 3
    assert "entry-0" not in proxy
    assert "entry-1" in proxy
    assert "entry-2" in proxy
    assert "entry-3" in proxy
