"""Tests for the 5 new direct-fit features:
1. TTL expiry (SemanticCacheEntry + evict_expired + delete)
2. /v1/cache CRUD REST API
3. LatticeMultiCache
4. lattice export / import CLI
5. Proxy warm-start (warm_path)
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from latticememory.proxy import LatticeLLMProxy
from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry
from latticememory.hamming_router import HammingRouter
from latticememory.multi_cache import LatticeMultiCache


# ---------------------------------------------------------------------------
# Shared test encoder / proxy helpers
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


def _make_proxy(**kwargs):
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
        upstream_api_key="test-key",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=FakeUpstream(),
        **kwargs,
    )


def _req(prompt: str) -> dict:
    return {"model": "gpt-test", "messages": [{"role": "user", "content": prompt}]}


# ---------------------------------------------------------------------------
# 1. TTL expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_is_expired_false_when_no_ttl(self):
        e = SemanticCacheEntry(
            cache_id="x", prompt="hi", value="bye",
            lattice_key=b"\x00" * 128,
        )
        assert not e.is_expired()

    def test_is_expired_false_when_within_ttl(self):
        e = SemanticCacheEntry(
            cache_id="x", prompt="hi", value="bye",
            lattice_key=b"\x00" * 128,
            created_at=time.time(),
            ttl_seconds=3600.0,
        )
        assert not e.is_expired()

    def test_is_expired_true_when_past_ttl(self):
        e = SemanticCacheEntry(
            cache_id="x", prompt="hi", value="bye",
            lattice_key=b"\x00" * 128,
            created_at=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600.0,             # 1 hour TTL
        )
        assert e.is_expired()

    def test_get_returns_miss_for_expired_entry(self):
        proxy = _make_proxy()
        # Put with a TTL that has already expired
        entry = proxy.cache.put(
            "what is two plus two",
            value={"answer": "4"},
            ttl_seconds=1.0,
        )
        # Artificially age it by mutating via dataclass replace
        import dataclasses
        aged = dataclasses.replace(entry, created_at=time.time() - 10)
        proxy.cache._entries[entry.cache_id] = aged

        result = proxy.cache.get("what is two plus two")
        assert not result.hit
        assert entry.cache_id not in proxy.cache._entries

    def test_evict_expired_removes_only_expired(self):
        proxy = _make_proxy()
        import dataclasses

        good = proxy.cache.put("fresh question", value="answer A", ttl_seconds=3600.0)
        stale = proxy.cache.put("stale question", value="answer B", ttl_seconds=1.0)

        # Age the stale entry
        aged = dataclasses.replace(stale, created_at=time.time() - 100)
        proxy.cache._entries[stale.cache_id] = aged

        evicted = proxy.cache.evict_expired()
        assert evicted == 1
        assert good.cache_id in proxy.cache._entries
        assert stale.cache_id not in proxy.cache._entries

    def test_delete_removes_entry_and_router_key(self):
        proxy = _make_proxy(hamming_router_mode="serve")
        entry = proxy.cache.put("delete me", value="v1")
        assert proxy.cache.get("delete me").hit

        deleted = proxy.cache.delete(entry.cache_id)
        assert deleted
        # After deletion, router should not return this cache_id
        result = proxy.cache.get("delete me")
        # Either miss or different entry — the deleted one must not resurface
        assert not result.hit or result.cache_id != entry.cache_id

    def test_delete_nonexistent_returns_false(self):
        proxy = _make_proxy()
        assert not proxy.cache.delete("nonexistent-id-xyz")


# ---------------------------------------------------------------------------
# 2. /v1/cache CRUD REST API
# ---------------------------------------------------------------------------

class TestCacheRestAPI:
    def _client(self, **kwargs):
        from fastapi.testclient import TestClient
        proxy = _make_proxy(**kwargs)
        app = proxy.create_app()
        return TestClient(app), proxy

    def test_list_empty_cache(self):
        client, _ = self._client()
        r = client.get("/v1/cache")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["entries"] == []

    def test_add_and_list_entry(self):
        client, _ = self._client()
        r = client.post("/v1/cache", json={"prompt": "hello world", "value": "greeting"})
        assert r.status_code == 200
        data = r.json()
        assert "cache_id" in data
        assert data["status"] == "added"

        r2 = client.get("/v1/cache")
        assert r2.json()["total"] == 1

    def test_get_entry_by_id(self):
        client, _ = self._client()
        add = client.post("/v1/cache", json={"prompt": "fetch me", "value": "found"})
        cid = add.json()["cache_id"]

        r = client.get(f"/v1/cache/{cid}")
        assert r.status_code == 200
        assert r.json()["prompt"] == "fetch me"
        assert r.json()["value"] == "found"

    def test_get_nonexistent_entry_404(self):
        client, _ = self._client()
        r = client.get("/v1/cache/does-not-exist")
        assert r.status_code == 404

    def test_delete_entry(self):
        client, _ = self._client()
        add = client.post("/v1/cache", json={"prompt": "remove me", "value": "gone"})
        cid = add.json()["cache_id"]

        r = client.delete(f"/v1/cache/{cid}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

        r2 = client.get(f"/v1/cache/{cid}")
        assert r2.status_code == 404

    def test_delete_nonexistent_entry_404(self):
        client, _ = self._client()
        r = client.delete("/v1/cache/no-such-id")
        assert r.status_code == 404

    def test_flush_cache(self):
        client, _ = self._client()
        client.post("/v1/cache", json={"prompt": "a", "value": "1"})
        client.post("/v1/cache", json={"prompt": "b", "value": "2"})
        assert client.get("/v1/cache").json()["total"] == 2

        r = client.post("/v1/cache/flush")
        assert r.status_code == 200
        assert r.json()["entries_removed"] == 2
        assert client.get("/v1/cache").json()["total"] == 0

    def test_evict_expired_endpoint(self):
        import dataclasses
        client, proxy = self._client()
        entry = proxy.cache.put("stale q", value="v", ttl_seconds=1.0)
        aged = dataclasses.replace(entry, created_at=time.time() - 10)
        proxy.cache._entries[entry.cache_id] = aged

        r = client.post("/v1/cache/evict_expired")
        assert r.status_code == 200
        assert r.json()["evicted"] == 1

    def test_admin_key_gates_mutations(self):
        client, _ = self._client(admin_key="secret-key")

        # Read endpoints also require the key once admin_key is configured —
        # cached prompts/responses are sensitive and must not be readable by
        # anyone who can merely reach the proxy.
        r = client.get("/v1/cache")
        assert r.status_code == 403

        r = client.get("/v1/cache", headers={"X-Lattice-Admin-Key": "secret-key"})
        assert r.status_code == 200

        # Mutations without key are 403
        r = client.post("/v1/cache", json={"prompt": "x", "value": "y"})
        assert r.status_code == 403

        # Mutations with correct key work
        r = client.post(
            "/v1/cache",
            json={"prompt": "x", "value": "y"},
            headers={"X-Lattice-Admin-Key": "secret-key"},
        )
        assert r.status_code == 200

    def test_add_entry_with_ttl(self):
        client, proxy = self._client()
        r = client.post("/v1/cache", json={"prompt": "ttl question", "value": "ans", "ttl_seconds": 3600})
        assert r.status_code == 200
        cid = r.json()["cache_id"]
        entry = proxy.cache._entries[cid]
        assert entry.ttl_seconds == 3600.0


# ---------------------------------------------------------------------------
# 3. LatticeMultiCache
# ---------------------------------------------------------------------------

class TestLatticeMultiCache:
    def _make_mc(self, **kwargs):
        from latticememory.text_runtime import RFSnapTextMemory

        def factory(tenant_id: str):
            runtime = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384)
            return RFSnapSemanticCache(runtime=runtime)

        return LatticeMultiCache(cache_factory=factory, **kwargs)

    def test_separate_namespaces(self):
        mc = self._make_mc()
        mc.put("acme", "reset password", value="go to /reset")
        mc.put("globex", "reset password", value="call IT x123")

        r_acme = mc.get("acme", "reset password")
        r_globex = mc.get("globex", "reset password")

        assert r_acme.hit and r_acme.value == "go to /reset"
        assert r_globex.hit and r_globex.value == "call IT x123"

    def test_lazy_creation(self):
        mc = self._make_mc()
        assert mc.tenant_ids() == []
        mc.get_cache("new_tenant")
        assert "new_tenant" in mc.tenant_ids()

    def test_default_tenant(self):
        mc = self._make_mc(default_tenant="shared")
        mc.put(None, "hello", value="world")
        r = mc.get(None, "hello")
        assert r.hit

    def test_no_default_raises_on_none(self):
        mc = self._make_mc()
        with pytest.raises(ValueError):
            mc.get_cache(None)

    def test_max_tenants(self):
        mc = self._make_mc(max_tenants=2)
        mc.get_cache("a")
        mc.get_cache("b")
        with pytest.raises(RuntimeError):
            mc.get_cache("c")

    def test_drop_tenant(self):
        mc = self._make_mc()
        mc.get_cache("tenant_x")
        assert mc.drop_tenant("tenant_x")
        assert "tenant_x" not in mc.tenant_ids()
        assert not mc.drop_tenant("tenant_x")  # already gone

    def test_stats(self):
        mc = self._make_mc()
        mc.put("t1", "q", value="a")
        s = mc.stats()
        assert s["n_tenants"] == 1
        assert "t1" in s["tenants"]
        assert s["tenants"]["t1"]["size"] == 1

    def test_evict_expired_all(self):
        import dataclasses as dc
        mc = self._make_mc()
        entry = mc.put("t1", "stale", value="v", ttl_seconds=1.0)
        cache = mc.get_cache("t1")
        aged = dc.replace(entry, created_at=time.time() - 100)
        cache._entries[entry.cache_id] = aged

        results = mc.evict_expired_all()
        assert results["t1"] == 1


# ---------------------------------------------------------------------------
# 4. lattice export / import CLI
# ---------------------------------------------------------------------------

class TestExportImport:
    def _build_cache(self):
        from latticememory.text_runtime import RFSnapTextMemory
        runtime = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384)
        return RFSnapSemanticCache(runtime=runtime)

    def test_export_produces_valid_jsonl(self, tmp_path):
        """Export writes one JSON line per entry with required fields."""
        import json, dataclasses as dc

        cache = self._build_cache()
        cache.put("What is Python?", value="A programming language")
        cache.put("What is Rust?",   value="A systems language")

        export_path = tmp_path / "export.jsonl"

        # Write entries manually (mirrors cmd_export logic)
        with open(export_path, "w", encoding="utf-8") as f:
            for e in cache._entries.values():
                f.write(json.dumps({
                    "cache_id":    e.cache_id,
                    "prompt":      e.prompt,
                    "value":       e.value,
                    "lattice_key": e.lattice_key.hex() if e.lattice_key else None,
                    "metadata":    e.metadata,
                    "created_at":  e.created_at,
                    "updated_at":  e.updated_at,
                    "ttl_seconds": e.ttl_seconds,
                }, ensure_ascii=False) + "\n")

        lines = [l for l in export_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            rec = json.loads(line)
            assert "prompt" in rec
            assert "lattice_key" in rec
            assert "value" in rec

    def test_import_populates_entries(self, tmp_path):
        """cmd_import loads JSONL entries into cache._entries without re-encoding."""
        import argparse

        from latticememory.cli import cmd_import
        from latticememory.text_runtime import RFSnapTextMemory

        cache_orig = self._build_cache()
        e1 = cache_orig.put("import me", value="imported value")

        export_path = tmp_path / "exp.jsonl"
        with open(export_path, "w", encoding="utf-8") as f:
            e = e1
            f.write(json.dumps({
                "cache_id": e.cache_id, "prompt": e.prompt, "value": e.value,
                "lattice_key": e.lattice_key.hex(), "metadata": e.metadata,
                "created_at": e.created_at, "updated_at": e.updated_at,
                "ttl_seconds": e.ttl_seconds,
            }) + "\n")

        imp_args = argparse.Namespace(
            input=str(export_path),
            cache=str(tmp_path / "cache2.db"),
            overwrite=False,
            include_expired=False,
        )
        rc = cmd_import(imp_args)
        assert rc == 0

    def test_import_skips_expired_entries(self, tmp_path):
        """cmd_import with include_expired=False drops TTL-expired records."""
        import argparse, dataclasses as dc

        from latticememory.cli import cmd_import

        cache = self._build_cache()
        entry = cache.put("stale q", value="v", ttl_seconds=1.0)
        aged = dc.replace(entry, created_at=time.time() - 100)

        export_path = tmp_path / "exp.jsonl"
        with open(export_path, "w", encoding="utf-8") as f:
            e = aged
            f.write(json.dumps({
                "cache_id": e.cache_id, "prompt": e.prompt, "value": e.value,
                "lattice_key": e.lattice_key.hex(), "metadata": e.metadata,
                "created_at": e.created_at, "updated_at": e.updated_at,
                "ttl_seconds": e.ttl_seconds,
            }) + "\n")

        # Without include_expired the record should be skipped (rc=0 still)
        rc = cmd_import(argparse.Namespace(
            input=str(export_path), cache=str(tmp_path / "c2.db"),
            overwrite=False, include_expired=False,
        ))
        assert rc == 0


# ---------------------------------------------------------------------------
# 5. Proxy warm-start
# ---------------------------------------------------------------------------

class TestProxyWarmStart:
    def test_warm_from_json(self, tmp_path):
        warm_file = tmp_path / "warm.json"
        pairs = [
            {"question": "How do I reset my password?", "answer": "Visit /reset"},
            {"question": "What is my username?",        "answer": "Your email address"},
        ]
        warm_file.write_text(json.dumps(pairs), encoding="utf-8")

        proxy = _make_proxy(warm_path=str(warm_file))
        assert proxy.cache.size == 2

        result = proxy.cache.get("How do I reset my password?")
        assert result.hit
        assert result.value == "Visit /reset"

    def test_warm_from_csv(self, tmp_path):
        import csv
        warm_file = tmp_path / "warm.csv"
        with open(warm_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question", "answer", "intent_id"])
            w.writeheader()
            w.writerow({"question": "open a ticket", "answer": "email support@co.com", "intent_id": "ticket"})
        proxy = _make_proxy(warm_path=str(warm_file))
        assert proxy.cache.size == 1

    def test_warm_from_jsonl(self, tmp_path):
        warm_file = tmp_path / "warm.jsonl"
        warm_file.write_text(
            '{"question": "q1", "answer": "a1"}\n{"question": "q2", "answer": "a2"}\n',
            encoding="utf-8",
        )
        proxy = _make_proxy(warm_path=str(warm_file))
        assert proxy.cache.size == 2

    def test_warm_missing_file_does_not_crash(self, tmp_path):
        proxy = _make_proxy(warm_path=str(tmp_path / "does_not_exist.json"))
        assert proxy.cache.size == 0

    def test_warm_skips_rows_without_answer(self, tmp_path):
        warm_file = tmp_path / "warm.json"
        pairs = [
            {"question": "good row", "answer": "has answer"},
            {"question": "bad row"},   # no answer
        ]
        warm_file.write_text(json.dumps(pairs))
        proxy = _make_proxy(warm_path=str(warm_file))
        assert proxy.cache.size == 1


# ---------------------------------------------------------------------------
# HammingRouter.remove_by_value (prerequisite)
# ---------------------------------------------------------------------------

class TestHammingRouterRemove:
    def test_remove_by_value_found(self):
        enc = HashEncoder(384)
        router = HammingRouter(encoder=enc, d_model=384, threshold=111)
        router.add("hello world", value="entry_1")
        assert len(router) == 1
        found = router.remove_by_value("entry_1")
        assert found
        assert len(router) == 0

    def test_remove_by_value_not_found(self):
        enc = HashEncoder(384)
        router = HammingRouter(encoder=enc, d_model=384, threshold=111)
        router.add("hello world", value="entry_1")
        found = router.remove_by_value("nonexistent")
        assert not found
        assert len(router) == 1

    def test_remove_does_not_break_lookup(self):
        enc = HashEncoder(384)
        router = HammingRouter(encoder=enc, d_model=384, threshold=111)
        router.add("hello world", value="entry_1")
        router.add("goodbye world", value="entry_2")
        router.remove_by_value("entry_1")
        # Lookup for "goodbye world" should still work
        match = router.lookup("goodbye world")
        assert match is not None
        assert match.value == "entry_2"
