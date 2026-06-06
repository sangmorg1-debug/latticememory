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
