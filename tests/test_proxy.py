from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from latticememory.proxy import LatticeLLMProxy


class CanonicalPromptEncoder:
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        vectors = []
        for sentence in sentences:
            canonical = str(sentence).lower().replace("which city is france's capital?", "what is the capital of france?")
            seed = int(hashlib.md5(canonical.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.d_model).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        return np.stack(vectors)


@dataclass
class FakeUpstream:
    calls: int = 0

    async def __call__(self, payload, headers):
        self.calls += 1
        return {
            "id": f"chatcmpl-{self.calls}",
            "object": "chat.completion",
            "model": payload.get("model", "test-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"answer {self.calls}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }


def _request(prompt: str) -> dict:
    return {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": prompt}],
    }


class HashEncoder:
    """MD5-hash deterministic encoder — same text → same vector, different text → different vector."""

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        vectors = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vectors.append(v)
        return np.stack(vectors)


def test_chat_completion_miss_calls_upstream_and_sets_miss_header():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
    )
    client = TestClient(proxy.create_app())

    response = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))

    assert response.status_code == 200
    assert response.headers["X-Lattice-Cache"] == "MISS"
    assert upstream.calls == 1
    assert response.json()["choices"][0]["message"]["content"] == "answer 1"


def test_semantic_cache_hit_returns_cached_response_without_upstream_call():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
    )
    client = TestClient(proxy.create_app())

    first = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    second = client.post("/v1/chat/completions", json=_request("Which city is France's capital?"))

    assert first.headers["X-Lattice-Cache"] == "MISS"
    assert second.status_code == 200
    assert second.headers["X-Lattice-Cache"] == "HIT"
    assert float(second.headers["X-Lattice-Savings-USD"]) > 0.0
    assert upstream.calls == 1
    assert second.json()["choices"][0]["message"]["content"] == "answer 1"


def test_chat_completion_rejects_requests_without_user_prompt():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
    )
    client = TestClient(proxy.create_app())

    response = client.post("/v1/chat/completions", json={"model": "gpt-test", "messages": []})

    assert response.status_code == 400
    assert "messages must include" in response.json()["detail"]
    assert upstream.calls == 0


def test_proxy_is_exported_from_top_level_package():
    from latticememory import LatticeLLMProxy as ExportedProxy

    assert ExportedProxy is LatticeLLMProxy


def test_compliance_validation_flow():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
        compliance_mode=True,
        validation_required=True
    )
    client = TestClient(proxy.create_app())

    # 1. First request is a MISS, cached but unvalidated
    response1 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert response1.headers["X-Lattice-Cache"] == "MISS"
    assert response1.headers["X-Lattice-Compliance"] == "PENDING_VALIDATION"
    assert upstream.calls == 1

    # 2. Second request is also a MISS because validation is required but entry is unvalidated
    response2 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert response2.headers["X-Lattice-Cache"] == "MISS"
    assert response2.headers["X-Lattice-Compliance"] == "PENDING_VALIDATION"
    assert upstream.calls == 2

    # 3. Call validate endpoint to approve the entry
    val_res = client.post("/v1/compliance/validate", json={"prompt": "What is the capital of France?"})
    assert val_res.status_code == 200
    assert val_res.json()["status"] == "approved"

    # 4. Third request is a HIT because it is now validated!
    response3 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert response3.headers["X-Lattice-Cache"] == "HIT"
    assert response3.headers["X-Lattice-Compliance"] == "APPROVED"
    assert upstream.calls == 2  # no new upstream call!


def test_tamper_evident_audit_log():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
        compliance_mode=True,
    )
    client = TestClient(proxy.create_app())

    client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    client.post("/v1/compliance/validate", json={"prompt": "What is the capital of France?"})
    client.post("/v1/chat/completions", json=_request("What is the capital of France?"))

    # Fetch audit log
    log_res = client.get("/v1/compliance/audit-log")
    assert log_res.status_code == 200
    data = log_res.json()
    assert data["total_events"] == 3
    assert data["chain_integrity_valid"] is True

    # Tamper with an event in-memory to verify chain check detection
    proxy.audit_events[1]["response"] = "TAMPERED_CONTENT"
    log_res_tampered = client.get("/v1/compliance/audit-log")
    assert log_res_tampered.json()["chain_integrity_valid"] is False


def test_divergence_detection():
    class VariableFakeUpstream:
        def __init__(self):
            self.calls = 0
        async def __call__(self, payload, headers):
            self.calls += 1
            if self.calls == 1:
                content = "Paris is the capital of France."
            else:
                content = "Completely different text: frogs and baguettes."
            return {
                "id": "chatcmpl",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": content}}],
            }

    upstream = VariableFakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
        compliance_mode=True,
        divergence_threshold=0.1
    )
    client = TestClient(proxy.create_app())

    # 1. Seed the cache (MISS)
    res1 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert res1.headers["X-Lattice-Cache"] == "MISS"

    # Approve the cache entry so it is compliance-validated
    client.post("/v1/compliance/validate", json={"prompt": "What is the capital of France?"})

    # 2. Get it again. It has a cache hit, but because divergence_threshold is set, it checks
    # divergence by calling upstream. Upstream returns a different answer, triggering divergence!
    res2 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert res2.headers["X-Lattice-Cache"] == "HIT_DIVERGED"
    assert res2.headers["X-Lattice-Compliance"] == "DIVERGENCE_REVIEW_REQUIRED"
    assert res2.headers["X-Lattice-Hamming-Distance"] == "-1"


def test_hamming_router_disabled_by_default_gives_paraphrase_miss():
    """With enable_hamming_router=False (default), distinct prompts that don't share an E8 key miss."""
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=upstream,
        # enable_hamming_router defaults to False
    )
    client = TestClient(proxy.create_app())

    client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    res = client.post("/v1/chat/completions", json=_request("Which city is France's capital?"))

    # Different text → different E8 key → no router → miss
    assert res.headers["X-Lattice-Cache"] == "MISS"
    assert upstream.calls == 2


def test_hamming_router_enabled_returns_hamming_nn_hit():
    """With enable_hamming_router=True and threshold=128, any stored entry matches any query."""
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=upstream,
        enable_hamming_router=True,
        hamming_threshold=128,  # always hit — exercises the Hamming-NN code path
    )
    client = TestClient(proxy.create_app())

    # Seed with one prompt
    client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    # Query with a different prompt — threshold=128 guarantees a hit
    res = client.post("/v1/chat/completions", json=_request("Which city is France's capital?"))

    assert res.headers["X-Lattice-Cache"] == "HIT"
    assert res.headers["X-Lattice-Retrieval-Path"] == "hamming_nn"
    assert int(res.headers["X-Lattice-Hamming-Distance"]) >= 0
    assert upstream.calls == 1  # no second upstream call


def test_hamming_router_deduplicates_repeated_puts():
    """Putting the same prompt multiple times adds it to the router only once."""
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=upstream,
        enable_hamming_router=True,
        hamming_threshold=128,
    )
    client = TestClient(proxy.create_app())

    for _ in range(3):
        client.post("/v1/chat/completions", json=_request("What is the capital of France?"))

    router = proxy.cache._hamming_router
    assert router is not None
    assert len(router) == 1  # deduplicated: 3 puts → 1 router entry


def test_proxy_calibration_success(tmp_path):
    import json
    data_path = tmp_path / "calibration.json"
    data = {
        "paraphrases": [
            ["What is the capital of France?", "What is the capital of France?"],
            ["Which city is France's capital?", "What is the capital of France?"]
        ],
        "near_misses": [
            ["What is the capital of France?", "Reset my password"],
            ["Which city is France's capital?", "Reset my password"]
        ]
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")

    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=upstream,
        enable_hamming_router=True,
        calibration_data_path=str(data_path),
        fp_budget=0.1,
    )
    assert proxy.hamming_router_calibrated is True
    assert proxy.hamming_router_n_paraphrase_pairs == 2
    assert proxy.hamming_router_n_near_miss_pairs == 2
    assert proxy.cache._hamming_threshold >= 0

    client = TestClient(proxy.create_app())
    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["hamming_router"]["enabled"] is True
    assert health_data["hamming_router"]["calibrated"] is True
    assert health_data["hamming_router"]["threshold"] == proxy.cache._hamming_threshold
    assert health_data["hamming_router"]["fp_budget"] == 0.1
    assert health_data["hamming_router"]["n_paraphrase_pairs"] == 2
    assert health_data["hamming_router"]["n_near_miss_pairs"] == 2
    assert health_data["hamming_router"]["reliable"] is False  # Below 100 pairs


def test_proxy_calibration_schema_validation(tmp_path):
    import json
    import pytest
    data_path = tmp_path / "invalid_calibration.json"
    
    # 1. Invalid dict structure
    data_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Calibration JSON must be a dictionary"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(data_path),
            require_calibration=True,
        )

    # 2. Missing keys
    data_path.write_text(json.dumps({"paraphrases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain both 'paraphrases' and 'near_misses'"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(data_path),
            require_calibration=True,
        )

    # 3. Empty lists
    data_path.write_text(json.dumps({"paraphrases": [], "near_misses": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain at least 1 pair"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(data_path),
            require_calibration=True,
        )

    # 4. Invalid pair length
    data_path.write_text(json.dumps({
        "paraphrases": [["one text only"]],
        "near_misses": [["one", "two"]]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a pair of length 2"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(data_path),
            require_calibration=True,
        )


def test_proxy_calibration_fail_closed(tmp_path):
    import pytest
    import json
    # Non-existent file
    non_existent = tmp_path / "does_not_exist.json"

    # With require_calibration=True -> Raises ValueError
    with pytest.raises(ValueError, match="Calibration file does not exist"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(non_existent),
            require_calibration=True,
        )

    # With require_calibration=False -> Fails closed, disables router
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        enable_hamming_router=True,
        calibration_data_path=str(non_existent),
        require_calibration=False,
    )
    assert proxy.cache._hamming_router is None
    
    client = TestClient(proxy.create_app())
    health_res = client.get("/health")
    assert health_res.json()["hamming_router"]["enabled"] is False
    assert health_res.json()["hamming_router"]["threshold"] is None


def test_proxy_calibration_injected_cache(tmp_path):
    import json
    import pytest
    data_path = tmp_path / "calibration.json"
    data = {
        "paraphrases": [
            ["What is the capital of France?", "What is the capital of France?"]
        ],
        "near_misses": [
            ["What is the capital of France?", "Reset my password"]
        ]
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")

    # Injected cache WITHOUT a router, require_calibration=True -> Raises
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory
    
    runtime = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384)
    cache_no_router = RFSnapSemanticCache(runtime=runtime, hamming_router=None)
    
    with pytest.raises(ValueError, match="no Hamming router is configured"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            semantic_cache=cache_no_router,
            calibration_data_path=str(data_path),
            require_calibration=True,
        )

    # Injected cache WITH a router -> Successfully calibrates
    from latticememory.hamming_router import HammingRouter
    router = HammingRouter(encoder=runtime.encoder, d_model=runtime.d_model, threshold=70)
    cache_with_router = RFSnapSemanticCache(runtime=runtime, hamming_router=router)
    
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        semantic_cache=cache_with_router,
        calibration_data_path=str(data_path),
        require_calibration=True,
    )
    assert proxy.hamming_router_calibrated is True
    assert proxy.cache._hamming_threshold == router.threshold


def test_proxy_calibration_warnings(tmp_path):
    import pytest
    import warnings
    # 1. Warning when no Hamming router configured but optional calibration requested
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory
    
    runtime = RFSnapTextMemory(encoder=HashEncoder(384), d_model=384)
    cache_no_router = RFSnapSemanticCache(runtime=runtime, hamming_router=None)
    
    data_path = tmp_path / "calibration.json"
    data_path.write_text("{}", encoding="utf-8") # structure check is bypassed if router check fails
    
    with pytest.warns(UserWarning, match="no Hamming router is configured"):
        proxy1 = LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            semantic_cache=cache_no_router,
            calibration_data_path=str(data_path),
            require_calibration=False,
        )
    assert proxy1.cache._hamming_router is None

    # 2. Warning when optional calibration fails
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.warns(UserWarning, match="calibration failed"):
        proxy2 = LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            calibration_data_path=str(non_existent),
            require_calibration=False,
        )
    assert proxy2.cache._hamming_router is None


def test_proxy_shadow_mode():
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder=HashEncoder(384),
        d_model=384,
        upstream_client=upstream,
        hamming_router_mode="shadow",
        hamming_threshold=128,  # very high to ensure hit
    )
    client = TestClient(proxy.create_app())

    # 1. Populate the cache with a miss
    r1 = client.post("/v1/chat/completions", json=_request("What is the capital of France?"))
    assert r1.status_code == 200
    assert r1.headers["X-Lattice-Cache"] == "MISS"
    assert upstream.calls == 1

    # 2. Query again using a paraphrase/same key. Since shadow mode is active, E8 matching triggers
    # but still calls upstream (returns MISS, with shadow headers and shadow_hit in audit log)
    r2 = client.post("/v1/chat/completions", json=_request("Which city is France's capital?"))
    assert r2.status_code == 200
    assert r2.headers["X-Lattice-Cache"] == "MISS"
    assert r2.headers["X-Lattice-Shadow-Hit"] == "true"
    assert "X-Lattice-Shadow-Distance" in r2.headers
    assert upstream.calls == 2

    # Check that a shadow hit audit event was logged
    shadow_events = [e for e in proxy.audit_events if e["event_type"] == "SHADOW_HIT"]
    assert len(shadow_events) == 1
    assert shadow_events[0]["prompt"] == "Which city is France's capital?"


def test_proxy_calibration_version_lock_success(tmp_path):
    import json
    data_path = tmp_path / "precalibrated.json"
    artifact = {
        "artifact_type": "latticememory_hamming_calibration",
        "artifact_version": 1,
        "model": "gpt-test-model-name",
        "d_model": 384,
        "calibration_data_sha256": "fake-sha256-string",
        "created_at": "2026-06-07T12:00:00Z",
        "fp_budget": 0.05,
        "calibration": {
            "threshold": 65,
            "recall": 0.95,
            "fp_rate": 0.01,
            "n_paraphrase_pairs": 100,
            "n_near_miss_pairs": 100
        },
        "gap_stats": {}
    }
    data_path.write_text(json.dumps(artifact), encoding="utf-8")

    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder_model="gpt-test-model-name",
        encoder=HashEncoder(384),
        d_model=384,
        hamming_router_mode="serve",
        calibration_data_path=str(data_path),
        require_calibration=True,
    )
    assert proxy.hamming_router_calibrated is True
    assert proxy.cache._hamming_threshold == 65
    assert proxy.hamming_router_recall == 0.95
    assert proxy.hamming_router_fp_rate == 0.01
    assert proxy.hamming_router_fp_budget == 0.05
    assert proxy.hamming_router_calibration_file_hash == "fake-sha256-string"


def test_proxy_calibration_version_lock_mismatch_raise(tmp_path):
    import json
    import pytest
    data_path = tmp_path / "precalibrated_mismatch.json"
    artifact = {
        "artifact_type": "latticememory_hamming_calibration",
        "artifact_version": 1,
        "model": "wrong-model-name",
        "d_model": 384,
        "calibration_data_sha256": "fake-sha256-string",
        "created_at": "2026-06-07T12:00:00Z",
        "fp_budget": 0.05,
        "calibration": {
            "threshold": 65,
            "recall": 0.95,
            "fp_rate": 0.01,
            "n_paraphrase_pairs": 100,
            "n_near_miss_pairs": 100
        },
        "gap_stats": {}
    }
    data_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="Calibration version lock mismatch"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder_model="gpt-test-model-name",
            encoder=HashEncoder(384),
            d_model=384,
            hamming_router_mode="serve",
            calibration_data_path=str(data_path),
            require_calibration=True,
        )


def test_proxy_calibration_version_lock_mismatch_warn(tmp_path):
    import json
    import pytest
    import warnings
    data_path = tmp_path / "precalibrated_mismatch.json"
    artifact = {
        "artifact_type": "latticememory_hamming_calibration",
        "artifact_version": 1,
        "model": "wrong-model-name",
        "d_model": 384,
        "calibration_data_sha256": "fake-sha256-string",
        "created_at": "2026-06-07T12:00:00Z",
        "fp_budget": 0.05,
        "calibration": {
            "threshold": 65,
            "recall": 0.95,
            "fp_rate": 0.01,
            "n_paraphrase_pairs": 100,
            "n_near_miss_pairs": 100
        },
        "gap_stats": {}
    }
    data_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.warns(UserWarning, match="Calibration version lock mismatch"):
        proxy = LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder_model="gpt-test-model-name",
            encoder=HashEncoder(384),
            d_model=384,
            hamming_router_mode="serve",
            calibration_data_path=str(data_path),
            require_calibration=False,
        )
    assert proxy.hamming_router_mode == "off"
    assert proxy.cache.hamming_router_mode == "off"
    assert proxy.cache._hamming_router is None


def test_proxy_mode_override_conflict():
    import pytest
    with pytest.raises(ValueError, match="Conflict: enable_hamming_router=True"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=True,
            hamming_router_mode="shadow",
        )

    with pytest.raises(ValueError, match="Conflict: enable_hamming_router=False"):
        LatticeLLMProxy(
            upstream_url="https://example.test/v1/chat/completions",
            encoder=HashEncoder(384),
            d_model=384,
            enable_hamming_router=False,
            hamming_router_mode="serve",
        )


# ---------------------------------------------------------------------------
# proxy_server.py ASGI entrypoint tests
# ---------------------------------------------------------------------------

def test_proxy_server_imports_without_error_when_no_api_key(monkeypatch):
    """proxy_server.py must not raise at import time even with no API key set."""
    import sys
    import warnings

    for mod_name in list(sys.modules.keys()):
        if "latticememory.proxy_server" in mod_name:
            del sys.modules[mod_name]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LATTICE_API_KEY", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import latticememory.proxy_server  # noqa: F401

    warning_msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("OPENAI_API_KEY" in msg for msg in warning_msgs), (
        f"Expected RuntimeWarning about missing key, got: {[str(w.message) for w in caught]}"
    )


def test_proxy_server_emits_runtime_warning_not_error_on_missing_key(monkeypatch):
    """Importing proxy_server without API key should emit RuntimeWarning, not raise."""
    import sys
    import pytest

    for mod_name in list(sys.modules.keys()):
        if "latticememory.proxy_server" in mod_name:
            del sys.modules[mod_name]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LATTICE_API_KEY", raising=False)

    with pytest.warns(RuntimeWarning, match="OPENAI_API_KEY"):
        import latticememory.proxy_server  # noqa: F401


def test_proxy_server_health_endpoint_responds(monkeypatch):
    """The ASGI app in proxy_server responds to /health with status=healthy."""
    import sys

    for mod_name in list(sys.modules.keys()):
        if "latticememory.proxy_server" in mod_name:
            del sys.modules[mod_name]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-health-check")

    import latticememory.proxy_server as ps
    client = TestClient(ps.app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "latticememory-proxy"


# ---------------------------------------------------------------------------
# Compliance: reviewer key and pending queue
# ---------------------------------------------------------------------------

def _make_compliance_proxy(*, admin_key=None, reviewer_key=None):
    upstream = FakeUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
        compliance_mode=True,
        validation_required=True,
        admin_key=admin_key,
        reviewer_key=reviewer_key,
    )
    return proxy, TestClient(proxy.create_app())


def test_compliance_pending_lists_unvalidated_entries():
    proxy, client = _make_compliance_proxy()
    # Populate one unvalidated entry
    client.post("/v1/chat/completions", json=_request("What is gravity?"))
    res = client.get("/v1/compliance/pending")
    assert res.status_code == 200
    data = res.json()
    assert data["total_pending"] == 1
    assert data["entries"][0]["prompt"] == "What is gravity?"
    assert data["entries"][0]["preview"] is not None


def test_compliance_pending_empties_after_validation():
    proxy, client = _make_compliance_proxy()
    client.post("/v1/chat/completions", json=_request("What is gravity?"))
    assert client.get("/v1/compliance/pending").json()["total_pending"] == 1
    client.post("/v1/compliance/validate", json={"prompt": "What is gravity?"})
    assert client.get("/v1/compliance/pending").json()["total_pending"] == 0


def test_compliance_reviewer_key_allows_validate():
    proxy, client = _make_compliance_proxy(admin_key="admin-secret", reviewer_key="reviewer-secret")
    client.post("/v1/chat/completions", json=_request("What is photosynthesis?"))
    # Reviewer key accepted for /validate
    res = client.post(
        "/v1/compliance/validate",
        json={"prompt": "What is photosynthesis?"},
        headers={"X-Lattice-Reviewer-Key": "reviewer-secret"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "approved"


def test_compliance_reviewer_key_rejected_for_admin_mutation():
    proxy, client = _make_compliance_proxy(admin_key="admin-secret", reviewer_key="reviewer-secret")
    # Reviewer key must NOT allow cache deletion (admin-only)
    client.post("/v1/chat/completions", json=_request("What is a quasar?"))
    entry_id = list(proxy.cache._entries.keys())[0]
    res = client.delete(
        f"/v1/cache/{entry_id}",
        headers={"X-Lattice-Reviewer-Key": "reviewer-secret"},
    )
    assert res.status_code == 403


def test_compliance_pending_requires_compliance_mode():
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="test-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=FakeUpstream(),
        compliance_mode=False,
    )
    client = TestClient(proxy.create_app())
    res = client.get("/v1/compliance/pending")
    assert res.status_code == 400


def test_compliance_reviewer_key_wrong_key_rejected():
    proxy, client = _make_compliance_proxy(admin_key="admin-secret", reviewer_key="reviewer-secret")
    client.post("/v1/chat/completions", json=_request("What is a neutron star?"))
    res = client.post(
        "/v1/compliance/validate",
        json={"prompt": "What is a neutron star?"},
        headers={"X-Lattice-Reviewer-Key": "wrong-key"},
    )
    assert res.status_code == 403
