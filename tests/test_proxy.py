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

