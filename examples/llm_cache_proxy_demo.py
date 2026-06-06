"""Phase 6 demo: OpenAI-compatible LLM cache proxy."""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.proxy import LatticeLLMProxy


class CanonicalPromptEncoder:
    """Maps hand-written paraphrases to the same deterministic embedding."""

    CANONICAL = {
        "which city is france's capital?": "what is the capital of france?",
        "name france's capital city.": "what is the capital of france?",
        "what city is the capital of france?": "what is the capital of france?",
        "france capital?": "what is the capital of france?",
    }

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        vectors = []
        for sentence in sentences:
            prompt = str(sentence).strip().lower()
            canonical = self.CANONICAL.get(prompt, prompt)
            seed = int(hashlib.md5(canonical.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.d_model).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        return np.stack(vectors)


@dataclass
class MockUpstream:
    calls: int = 0

    async def __call__(self, payload, headers):
        self.calls += 1
        prompt = payload["messages"][-1]["content"]
        return {
            "id": f"mock-{self.calls}",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Mock answer for: {prompt}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 24, "completion_tokens": 12, "total_tokens": 36},
        }


def run_demo() -> None:
    prompts = [
        "What is the capital of France?",
        "Which city is France's capital?",
        "Name France's capital city.",
        "What city is the capital of France?",
        "France capital?",
        "Explain E8 lattice quantization.",
        "What is semantic caching?",
        "Define retrieval augmented generation.",
        "How does SQLite persistence work?",
        "What is a vector database?",
    ]
    upstream = MockUpstream()
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="demo-key",
        encoder=CanonicalPromptEncoder(384),
        d_model=384,
        upstream_client=upstream,
    )
    client = TestClient(proxy.create_app())

    hits = 0
    total_savings = 0.0
    for prompt in prompts:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-demo", "messages": [{"role": "user", "content": prompt}]},
        )
        cache_state = response.headers["X-Lattice-Cache"]
        if cache_state == "HIT":
            hits += 1
        total_savings += float(response.headers["X-Lattice-Savings-USD"])
        print(f"{cache_state:4} | {prompt}")

    print("\n--- Phase 6: LLM Cache Proxy Demo ---")
    print(f"Prompts sent through proxy: {len(prompts)}")
    print(f"Upstream API calls without proxy: {len(prompts)}")
    print(f"Upstream API calls with proxy: {upstream.calls}")
    print(f"Cache hits: {hits}")
    print(f"Estimated cost saved: ${total_savings:.6f}")


if __name__ == "__main__":
    run_demo()
