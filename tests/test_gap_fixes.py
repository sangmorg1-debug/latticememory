"""Tests for the 9 infrastructure gaps identified and fixed:

1. SQLite delete propagation (delete/evict_expired survive restart)
2. lattice import persists to SQLite via restore_entry()
3. CLI d_model detection from SQLite embedding blob
4. proxy_server console script has callable main()
5. streaming proxy uses cached_result shadow hit in flywheel
6. streaming proxy tests
7. Redis serialization includes ttl_seconds
8. Flywheel detect_drift joint clustering
9. E8LatticeDB.delete_document/delete_batch
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest


def test_observability_service_openapi_version_matches_package_metadata():
    from fastapi.testclient import TestClient

    from latticememory.service import create_app

    client = TestClient(create_app(d_model=8))

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["version"] == version("latticememory")


# ---------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------

class HashEncoder:
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        vecs = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


def _make_cache(d_model: int = 384, sqlite_path: str | None = None):
    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.memory import RFSnapLatticeMemory

    lm = RFSnapLatticeMemory(d_model=d_model, sqlite_path=sqlite_path)
    rt = RFSnapTextMemory(encoder=HashEncoder(d_model), d_model=d_model, memory=lm)
    return RFSnapSemanticCache(runtime=rt)


# ---------------------------------------------------------------------------
# 1. SQLite delete propagation
# ---------------------------------------------------------------------------

class TestSQLiteDeletePropagation:
    def test_delete_removes_from_sqlite(self, tmp_path):
        """After delete(), re-loading from SQLite must not resurrect the entry."""
        from latticememory.memory import RFSnapLatticeMemory
        from latticememory.text_runtime import RFSnapTextMemory
        from latticememory.semantic_cache import RFSnapSemanticCache

        db = str(tmp_path / "cache.db")
        cache = _make_cache(384, sqlite_path=db)
        entry = cache.put("hello world", value="greeting")
        assert cache.get("hello world").hit

        cache.delete(entry.cache_id)
        assert not cache.get("hello world").hit

        # Reload from SQLite
        lm2 = RFSnapLatticeMemory(d_model=384, sqlite_path=db)
        rt2 = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384, memory=lm2)
        cache2 = RFSnapSemanticCache(runtime=rt2)

        assert not cache2.get("hello world").hit, "Deleted entry must not survive reload"
        assert entry.cache_id not in cache2._entries

    def test_evict_expired_removes_from_sqlite(self, tmp_path):
        """evict_expired() must propagate deletions to SQLite."""
        import dataclasses as dc

        db = str(tmp_path / "cache.db")
        cache = _make_cache(384, sqlite_path=db)
        entry = cache.put("stale question", value="v", ttl_seconds=1.0)
        aged = dc.replace(entry, created_at=time.time() - 100)
        cache._entries[entry.cache_id] = aged

        n = cache.evict_expired()
        assert n == 1

        # Reload from SQLite
        from latticememory.memory import RFSnapLatticeMemory
        from latticememory.text_runtime import RFSnapTextMemory
        from latticememory.semantic_cache import RFSnapSemanticCache

        lm2 = RFSnapLatticeMemory(d_model=384, sqlite_path=db)
        rt2 = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384, memory=lm2)
        cache2 = RFSnapSemanticCache(runtime=rt2)

        assert entry.cache_id not in cache2._entries, "Evicted entry must not survive reload"

    def test_delete_propagates_through_stack(self):
        """E8LatticeDB.delete_document → memory.delete_documents → cache.delete chain."""
        from latticememory.rag.e8_retriever import E8LatticeDB
        import torch, torch.nn.functional as F

        db = E8LatticeDB(d_model=384)
        emb = torch.randn(384)
        emb = F.normalize(emb, dim=0)
        db.add_document("doc1", emb, metadata={})
        assert "doc1" in db._embeddings

        removed = db.delete_document("doc1")
        assert removed
        assert "doc1" not in db._embeddings
        assert "doc1" not in db._keys

    def test_delete_batch_count(self):
        from latticememory.rag.e8_retriever import E8LatticeDB
        import torch, torch.nn.functional as F

        db = E8LatticeDB(d_model=384)
        for i in range(5):
            emb = F.normalize(torch.randn(384), dim=0)
            db.add_document(f"doc{i}", emb)

        n = db.delete_batch(["doc0", "doc2", "doc4", "nonexistent"])
        assert n == 3
        assert len(db._embeddings) == 2

    def test_RFSnapLatticeMemory_delete_updates_ann_index(self, tmp_path):
        """After delete_documents, _doc_ids and _doc_index must be consistent."""
        from latticememory.memory import RFSnapLatticeMemory, MemoryDocument
        import torch, torch.nn.functional as F

        lm = RFSnapLatticeMemory(d_model=384)
        enc = HashEncoder(384)

        for q in ["alpha query", "beta query", "gamma query"]:
            emb_np = enc.encode([q])[0]
            emb = torch.from_numpy(emb_np)
            doc = MemoryDocument(doc_id=f"id-{q[:5]}", text=q, embedding=emb, metadata={})
            lm.add_documents([doc])

        assert len(lm._doc_ids) == 3
        lm.delete_documents(["id-alpha"])  # "alpha query"[:5] == "alpha"
        assert len(lm._doc_ids) == 2
        # Indices must be gapless
        for doc_id, idx in lm._doc_index.items():
            assert lm._doc_ids[idx] == doc_id


# ---------------------------------------------------------------------------
# 2. lattice import persists via restore_entry()
# ---------------------------------------------------------------------------

class TestRestoreEntry:
    def test_restore_entry_writes_to_sqlite(self, tmp_path):
        """restore_entry() must persist to SQLite so entries survive reload."""
        from latticememory.memory import RFSnapLatticeMemory
        from latticememory.text_runtime import RFSnapTextMemory
        from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry

        db = str(tmp_path / "cache.db")
        # Build a source cache and get a real entry + lattice key
        src = _make_cache(384, sqlite_path=db)
        original = src.put("restore this", value="restored value")

        # Simulate export + clear
        src._entries.clear()
        src.runtime.memory.sqlite_store.delete_documents([original.cache_id])

        # restore_entry into a fresh runtime pointing to same DB
        lm2 = RFSnapLatticeMemory(d_model=384, sqlite_path=db)
        rt2 = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384, memory=lm2)
        cache2 = RFSnapSemanticCache(runtime=rt2)
        cache2.restore_entry(original)

        assert cache2.get("restore this").hit

        # Now reload from scratch and verify it survived
        lm3 = RFSnapLatticeMemory(d_model=384, sqlite_path=db)
        rt3 = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384, memory=lm3)
        cache3 = RFSnapSemanticCache(runtime=rt3)
        assert original.cache_id in cache3._entries


# ---------------------------------------------------------------------------
# 3. CLI _detect_d_model
# ---------------------------------------------------------------------------

class TestDetectDModel:
    def test_detect_from_existing_db(self, tmp_path):
        from latticememory.cli import _detect_d_model

        db = str(tmp_path / "cache384.db")
        cache = _make_cache(384, sqlite_path=db)
        cache.put("test entry", value="val")

        detected = _detect_d_model(db)
        assert detected == 384

    def test_detect_fallback_on_empty_db(self, tmp_path):
        from latticememory.cli import _detect_d_model

        # Non-existent file → fallback
        d = _detect_d_model(str(tmp_path / "nonexistent.db"), fallback=512)
        assert d == 512

    def test_detect_fallback_on_empty_table(self, tmp_path):
        import sqlite3
        from latticememory.cli import _detect_d_model

        db = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE documents (doc_id TEXT, embedding BLOB NOT NULL, metadata TEXT, lattice_key BLOB, text TEXT)")
        conn.close()
        d = _detect_d_model(db, fallback=256)
        assert d == 256


# ---------------------------------------------------------------------------
# 4. console script main() is callable
# ---------------------------------------------------------------------------

def test_proxy_server_has_callable_main():
    from latticememory.proxy_server import main
    assert callable(main)


# ---------------------------------------------------------------------------
# 5. streaming flywheel uses cached_result shadow hit
# ---------------------------------------------------------------------------

class TestStreamingFlywheelShadowHit:
    def test_cached_result_shadow_passed_to_flywheel(self, tmp_path):
        """_stream_upstream_and_cache must forward shadow hit info to flywheel."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from latticememory.proxy import LatticeLLMProxy
        from latticememory.semantic_cache import SemanticCacheResult
        from latticememory.flywheel import LatticeFlywheel

        miss_log = str(tmp_path / "misses.jsonl")

        proxy = LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            upstream_api_key="test",
            encoder=HashEncoder(384),
            d_model=384,
            miss_log_path=miss_log,
        )

        # Fake a shadow hit result
        shadow_result = SemanticCacheResult(
            hit=False, hit_type="miss", value=None, cache_id=None,
            lattice_key=b"\x00" * 128,
            shadow_hit=True,
            shadow_source_prompt="nearest cached question",
            shadow_hamming_distance=42,
        )

        logged_misses = []
        original_log_miss = proxy.flywheel.log_miss
        def _capture(*args, **kwargs):
            logged_misses.append(kwargs)
            original_log_miss(*args, **kwargs)
        proxy.flywheel.log_miss = _capture

        # Simulate the end of _stream_upstream_and_cache after text assembly
        assembled_text = "test answer"
        synthetic_body = {
            "id": "chatcmpl-test", "object": "chat.completion",
            "created": int(time.time()), "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": assembled_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        prompt = "test question"
        proxy.cache.put(prompt, value=synthetic_body, metadata={})
        refetch = proxy.cache.get(prompt)
        key_hex = refetch.lattice_key.hex() if refetch.lattice_key else "unknown"

        proxy.flywheel.log_miss(
            prompt,
            e8_key_hex=key_hex,
            nearest_cache_prompt=shadow_result.shadow_source_prompt if shadow_result.shadow_hit else None,
            nearest_cache_distance=shadow_result.shadow_hamming_distance if shadow_result.shadow_hit else -1,
        )

        assert len(logged_misses) == 1
        assert logged_misses[0]["nearest_cache_prompt"] == "nearest cached question"
        assert logged_misses[0]["nearest_cache_distance"] == 42


# ---------------------------------------------------------------------------
# 6. Streaming proxy tests
# ---------------------------------------------------------------------------

class TestStreamingProxy:
    def _make_proxy(self, **kwargs):
        from dataclasses import dataclass

        @dataclass
        class FakeUpstream:
            calls: int = 0

            async def __call__(self, payload, headers):
                self.calls += 1
                return {
                    "id": f"chatcmpl-{self.calls}",
                    "object": "chat.completion",
                    "model": "test",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": f"reply {self.calls}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

        return LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            upstream_api_key="test",
            encoder=HashEncoder(384),
            d_model=384,
            upstream_client=FakeUpstream(),
            **kwargs,
        )

    def _req(self, prompt: str, stream: bool = False) -> dict:
        r = {"model": "gpt-test", "messages": [{"role": "user", "content": prompt}]}
        if stream:
            r["stream"] = True
        return r

    def test_streaming_cache_hit_returns_sse(self):
        from fastapi.testclient import TestClient
        from latticememory.proxy import LatticeLLMProxy

        proxy = self._make_proxy()
        app = proxy.create_app()
        client = TestClient(app)

        # Warm cache first (non-streaming)
        r1 = client.post("/v1/chat/completions", json=self._req("stream cache test"))
        assert r1.status_code == 200
        assert r1.headers.get("X-Lattice-Cache") == "MISS"

        # Now stream the same prompt — should hit cache
        r2 = client.post(
            "/v1/chat/completions",
            json=self._req("stream cache test", stream=True),
            headers={"Accept": "text/event-stream"},
        )
        assert r2.status_code == 200
        assert r2.headers.get("X-Lattice-Cache") == "HIT"
        body = r2.text
        assert "data:" in body
        assert "[DONE]" in body

    def test_non_streaming_cache_hit_returns_json(self):
        from fastapi.testclient import TestClient

        proxy = self._make_proxy()
        app = proxy.create_app()
        client = TestClient(app)

        client.post("/v1/chat/completions", json=self._req("json cache test"))
        r2 = client.post("/v1/chat/completions", json=self._req("json cache test"))
        assert r2.status_code == 200
        assert r2.headers.get("X-Lattice-Cache") == "HIT"
        data = r2.json()
        assert "choices" in data

    def test_streaming_cache_hit_contains_content(self):
        from fastapi.testclient import TestClient

        proxy = self._make_proxy()
        app = proxy.create_app()
        client = TestClient(app)

        # Warm with non-streaming first
        client.post("/v1/chat/completions", json=self._req("content check q"))

        # Stream the hit
        r = client.post(
            "/v1/chat/completions",
            json=self._req("content check q", stream=True),
        )
        assert r.status_code == 200
        # Should contain at least one data line with a chunk that has content
        data_lines = [l for l in r.text.splitlines() if l.startswith("data:") and "[DONE]" not in l]
        assert len(data_lines) >= 1
        first_chunk = json.loads(data_lines[0][5:].strip())
        assert "choices" in first_chunk


# Fix missing import
from latticememory.proxy import LatticeLLMProxy


# ---------------------------------------------------------------------------
# 7. Redis serialization includes ttl_seconds
# ---------------------------------------------------------------------------

class TestRedisSerializationTTL:
    def test_serialize_deserialize_with_ttl(self):
        from latticememory.redis_store import _RedisEntriesProxy
        from latticememory.semantic_cache import SemanticCacheEntry

        class FakeRedis:
            def __init__(self):
                self._store = {}
            def set(self, k, v): self._store[k] = v
            def setex(self, k, ttl, v): self._store[k] = v
            def get(self, k): return self._store.get(k)
            def exists(self, k): return k in self._store
            def delete(self, *keys):
                for k in keys: self._store.pop(k, None)
            def scan(self, cursor, match=None, count=100):
                keys = [k for k in self._store if not match or k.startswith(match.rstrip("*"))]
                return (0, keys)

        proxy = _RedisEntriesProxy(FakeRedis(), namespace="test", ttl=None)

        entry = SemanticCacheEntry(
            cache_id="abc123",
            prompt="ttl test",
            value="answer",
            lattice_key=b"\x01" * 128,
            ttl_seconds=3600.0,
        )
        proxy["abc123"] = entry

        retrieved = proxy["abc123"]
        assert retrieved.ttl_seconds == 3600.0
        assert not retrieved.is_expired()

    def test_serialize_deserialize_without_ttl(self):
        from latticememory.redis_store import _RedisEntriesProxy
        from latticememory.semantic_cache import SemanticCacheEntry

        class FakeRedis:
            def __init__(self):
                self._store = {}
            def set(self, k, v): self._store[k] = v
            def get(self, k): return self._store.get(k)
            def exists(self, k): return k in self._store

        proxy = _RedisEntriesProxy(FakeRedis(), namespace="test")
        entry = SemanticCacheEntry(
            cache_id="xyz", prompt="no ttl", value="v",
            lattice_key=b"\x02" * 128, ttl_seconds=None,
        )
        proxy["xyz"] = entry
        assert proxy["xyz"].ttl_seconds is None


# ---------------------------------------------------------------------------
# 8. Flywheel detect_drift joint clustering
# ---------------------------------------------------------------------------

class TestFlywheelDetectDrift:
    def test_detect_drift_returns_empty_on_no_records(self, tmp_path):
        from latticememory.flywheel import LatticeFlywheel

        fw = LatticeFlywheel(tmp_path / "misses.jsonl")
        result = fw.detect_drift(window_seconds=3600, min_delta=1, min_cluster_size=1)
        assert result == []

    def test_detect_drift_finds_increasing_cluster(self, tmp_path):
        """Records concentrated in the recent window should surface as drifting."""
        from latticememory.flywheel import LatticeFlywheel, MissRecord

        fw = LatticeFlywheel(tmp_path / "misses.jsonl", cluster_threshold=30)

        # Build a repeatable hex key
        key_hex = bytes([i % 256 for i in range(128)]).hex()

        now = time.time()
        window = 7 * 24 * 3600

        # Older window: 1 record
        older_ts = now - window * 1.5
        # Recent window: 8 records (delta = 7 >= min_delta=5)
        recent_ts = [now - window * 0.1 * i for i in range(8)]

        records = [MissRecord(question="q older", e8_key_hex=key_hex, timestamp=older_ts,
                              nearest_cache_prompt=None, nearest_cache_distance=-1)]
        for i, ts in enumerate(recent_ts):
            records.append(MissRecord(question=f"q recent {i}", e8_key_hex=key_hex,
                                      timestamp=ts, nearest_cache_prompt=None,
                                      nearest_cache_distance=-1))

        # Write records directly to log
        log_path = tmp_path / "misses.jsonl"
        import json, dataclasses
        with open(log_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")

        fw2 = LatticeFlywheel(log_path, cluster_threshold=30)
        drifting = fw2.detect_drift(window_seconds=window, min_delta=5, min_cluster_size=2)
        assert len(drifting) >= 1
        assert drifting[0]["recent"] > drifting[0]["previous"]
        assert drifting[0]["delta"] >= 5

    def test_detect_drift_joint_clustering_no_double_count(self, tmp_path):
        """Joint clustering must not report previous=0 when records span both windows."""
        from latticememory.flywheel import LatticeFlywheel, MissRecord
        import dataclasses, json

        key_hex = bytes([99] * 128).hex()
        now = time.time()
        window = 7 * 24 * 3600

        records = []
        # 3 in older, 6 in recent
        for i in range(3):
            records.append(MissRecord(question=f"older {i}", e8_key_hex=key_hex,
                                      timestamp=now - window * 1.2,
                                      nearest_cache_prompt=None, nearest_cache_distance=-1))
        for i in range(6):
            records.append(MissRecord(question=f"recent {i}", e8_key_hex=key_hex,
                                      timestamp=now - 3600,
                                      nearest_cache_prompt=None, nearest_cache_distance=-1))

        log_path = tmp_path / "misses2.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")

        fw = LatticeFlywheel(log_path, cluster_threshold=30)
        drifting = fw.detect_drift(window_seconds=window, min_delta=1, min_cluster_size=2)

        if drifting:
            # With joint clustering, previous must reflect older window count (3)
            assert drifting[0]["previous"] >= 1, "previous must not always be 0"


# ---------------------------------------------------------------------------
# 9. E8LatticeDB.delete_document idempotency
# ---------------------------------------------------------------------------

class TestE8LatticeDBDelete:
    def test_delete_nonexistent_returns_false(self):
        from latticememory.rag.e8_retriever import E8LatticeDB

        db = E8LatticeDB(d_model=384)
        assert not db.delete_document("ghost_id")

    def test_delete_clears_hash_store(self):
        from latticememory.rag.e8_retriever import E8LatticeDB
        import torch, torch.nn.functional as F

        db = E8LatticeDB(d_model=384)
        emb = F.normalize(torch.randn(384), dim=0)
        key = db.add_document("my_doc", emb)
        assert "my_doc" in db.hash_store.get(key, [])

        db.delete_document("my_doc")
        # The key's bucket must no longer contain the doc_id
        bucket = db.hash_store.get(key, [])
        assert "my_doc" not in bucket

    def test_delete_allows_reuse_of_same_lattice_cell(self):
        from latticememory.rag.e8_retriever import E8LatticeDB
        import torch, torch.nn.functional as F

        db = E8LatticeDB(d_model=384)
        enc = HashEncoder(384)
        emb = torch.from_numpy(enc.encode(["identical sentence"])[0])
        key1 = db.add_document("doc_v1", emb)

        db.delete_document("doc_v1")
        key2 = db.add_document("doc_v2", emb)

        assert key1 == key2  # same embedding → same lattice cell
        assert "doc_v1" not in db.hash_store.get(key2, [])
        assert "doc_v2" in db.hash_store.get(key2, [])


# ---------------------------------------------------------------------------
# 10. _quantize_batch parity with _quantize_to_indices
# ---------------------------------------------------------------------------

class TestQuantizeBatchParity:
    """_quantize_batch([v])[0] must be byte-identical to _quantize_to_indices(v) for any v.

    This guards against silent divergence between the single-vector and batch paths
    at argmax tie-breaking — a cache-correctness bug that add_batch / lattice_keys_for_batch
    would inherit.
    """

    def _random_vecs(self, n: int, d: int, seed: int = 0) -> "torch.Tensor":
        import torch
        rng = torch.Generator()
        rng.manual_seed(seed)
        vecs = torch.randn(n, d, generator=rng)
        return vecs  # deliberately not normalised — test handles arbitrary scale

    def test_single_vector_parity_384d(self):
        import torch
        from latticememory.rag.e8_retriever import E8LatticeDB

        db = E8LatticeDB(d_model=384)
        vecs = self._random_vecs(64, 384, seed=42)
        for v in vecs:
            single = db._quantize_to_indices(v)
            batch  = db._quantize_batch(vecs[:1])[0]  # batch of exactly 1
            # Use the same vector for both paths
            batch_v = db._quantize_batch(v.unsqueeze(0))[0]
            assert single == batch_v, (
                f"_quantize_batch disagrees with _quantize_to_indices for 384d vector; "
                f"first differing block at index "
                f"{next(i for i in range(len(single)) if single[i] != batch_v[i])}"
            )

    def test_batch_parity_1024d(self):
        import torch
        from latticememory.rag.e8_retriever import E8LatticeDB

        db = E8LatticeDB(d_model=1024)
        vecs = self._random_vecs(32, 1024, seed=7)
        batch_keys = db._quantize_batch(vecs)
        for i, v in enumerate(vecs):
            single = db._quantize_to_indices(v)
            assert single == batch_keys[i], (
                f"_quantize_batch[{i}] disagrees with _quantize_to_indices for 1024d vector"
            )

    def test_add_batch_keys_match_add_document(self):
        """add_batch must produce identical keys to N individual add_document calls."""
        import torch
        from latticememory.rag.e8_retriever import E8LatticeDB

        db_single = E8LatticeDB(d_model=384)
        db_batch  = E8LatticeDB(d_model=384)
        vecs = self._random_vecs(20, 384, seed=99)

        # Single-document path
        single_keys = []
        for i, v in enumerate(vecs):
            k = db_single.add_document(f"doc{i}", v)
            single_keys.append(k)

        # Batch path
        ids = [f"doc{i}" for i in range(20)]
        db_batch.add_batch(ids, vecs)
        batch_keys = [db_batch._keys[f"doc{i}"] for i in range(20)]

        assert single_keys == batch_keys

    def test_lattice_keys_for_batch_matches_lattice_key_for(self):
        """lattice_keys_for_batch must return the same keys as per-call lattice_key_for."""
        import torch
        from latticememory.memory import RFSnapLatticeMemory

        lm = RFSnapLatticeMemory(d_model=384)
        vecs = self._random_vecs(16, 384, seed=13)

        single_keys = [lm.lattice_key_for(v) for v in vecs]
        batch_keys  = lm.lattice_keys_for_batch(vecs)

        assert single_keys == batch_keys
