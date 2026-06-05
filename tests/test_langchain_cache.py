"""Tests for LatticeMemoryCache LangChain integration."""
from __future__ import annotations

import hashlib
import numpy as np
import pytest

try:
    from langchain_core.outputs import Generation
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


class FakeEncoder:
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def get_embedding_dimension(self):
        return self.d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        result = []
        for s in sentences:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            result.append(v)
        return np.stack(result)


@pytest.fixture()
def cache():
    from latticememory.integrations.langchain import LatticeMemoryCache
    from latticememory.index import LatticeIndex

    index = LatticeIndex.__new__(LatticeIndex)
    index._init_with_encoder(FakeEncoder(384), d_model=384)
    return LatticeMemoryCache(index=index)


def test_lookup_miss_returns_none(cache):
    result = cache.lookup("What is the capital of France?", "gpt-4")
    assert result is None


def test_update_then_lookup_returns_generations(cache):
    gens = [Generation(text="Paris")]
    cache.update("What is the capital of France?", "gpt-4", gens)
    result = cache.lookup("What is the capital of France?", "gpt-4")
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "Paris"


def test_lookup_different_llm_string_returns_none(cache):
    gens = [Generation(text="Paris")]
    cache.update("What is the capital of France?", "gpt-4", gens)
    result = cache.lookup("What is the capital of France?", "gpt-3.5-turbo")
    assert result is None


def test_update_idempotent(cache):
    gens = [Generation(text="Paris")]
    cache.update("capital of France?", "gpt-4", gens)
    cache.update("capital of France?", "gpt-4", [Generation(text="Paris updated")])
    result = cache.lookup("capital of France?", "gpt-4")
    assert result[0].text == "Paris updated"


def test_cache_accepts_injected_index(cache):
    from latticememory.integrations.langchain import LatticeMemoryCache
    assert isinstance(cache, LatticeMemoryCache)


def test_clear_empties_cache(cache):
    cache.update("hello", "gpt-4", [Generation(text="world")])
    cache.clear()
    result = cache.lookup("hello", "gpt-4")
    assert result is None
