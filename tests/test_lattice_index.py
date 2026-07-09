"""Tests for LatticeIndex public API.

Uses a FakeEncoder to avoid downloading model weights in unit tests.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pytest
import torch

from latticememory.index import LatticeIndex, LatticeStats, SearchResult


class FakeEncoder:
    """Deterministic fake encoder that hashes text to a fixed-dim vector."""

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
def index():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    return idx


def test_add_returns_hex_addresses(index):
    addresses = index.add(["Paris is the capital of France."])
    assert len(addresses) == 1
    assert isinstance(addresses[0], str)
    # 384 dims / 8 per E8 block = 48 blocks = 48 bytes = 96 hex chars
    assert len(addresses[0]) == 96


def test_add_multiple_returns_one_address_per_doc(index):
    texts = ["doc one", "doc two", "doc three"]
    addresses = index.add(texts)
    assert len(addresses) == 3
    assert len(set(addresses)) == 3


def test_same_text_always_gives_same_address(index):
    a1 = index.add(["What is the capital of France?"])[0]
    a2 = index.add(["What is the capital of France?"])[0]
    assert a1 == a2


def test_snap_returns_hex_string(index):
    addr = index.snap("hello world")
    assert isinstance(addr, str)
    assert len(addr) == 96


def test_snap_is_deterministic(index):
    a1 = index.snap("What is the capital of France?")
    a2 = index.snap("What is the capital of France?")
    assert a1 == a2


def test_search_returns_search_results(index):
    index.add(["Paris is the capital of France.", "London is the capital of the UK."])
    results = index.search("capital of France", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert isinstance(results[0].text, str)
    assert isinstance(results[0].score, float)
    assert isinstance(results[0].address, str)
    assert isinstance(results[0].retrieval_path, str)


def test_search_top_k_is_respected(index):
    for i in range(5):
        index.add([f"document number {i}"])
    results = index.search("document", top_k=3)
    assert len(results) <= 3


def test_search_empty_index_returns_empty(index):
    results = index.search("anything", top_k=5)
    assert results == []


def test_stats_doc_count(index):
    index.add(["doc a", "doc b"])
    stats = index.stats()
    assert isinstance(stats, LatticeStats)
    assert stats.docs == 2


def test_stats_compression_ratio_positive(index):
    index.add(["some text"])
    stats = index.stats()
    # Key-only mode (no fallback configured): real measured ratio is 32x
    # (4 bytes/dim float32 -> 1 byte per 8-dim E8 block). See
    # docs/honest_product_review.md's "32x (key-only)" figure.
    assert 31.0 <= stats.compression_vs_float32 <= 33.0


def test_stats_exposes_key_fallback_and_total_bytes():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._init_with_encoder(FakeEncoder(384), d_model=384, fallback_quantization=8)
    idx.add(["doc a", "doc b"])

    stats = idx.stats()

    assert stats.e8_key_bytes == 2 * (384 // 8)
    assert stats.fallback_bytes == 2 * 384
    assert stats.total_index_bytes == stats.e8_key_bytes + stats.fallback_bytes
    assert stats.float32_embedding_bytes == 2 * 384 * 4
    assert stats.fallback_quantization == 8
    assert stats.compression_mode == "hybrid_int8_fallback"


def test_init_with_encoder_stores_d_model():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._init_with_encoder(FakeEncoder(1024), d_model=1024)
    assert idx._d_model == 1024


def test_snap_matches_add_address(index):
    text = "What is the capital of France?"
    add_addr = index.add([text])[0]
    snap_addr = index.snap(text)
    assert snap_addr == add_addr


def test_search_exact_text_hits_lattice_exact_path(index):
    text = "Paris is the capital of France."
    index.add([text])
    results = index.search(text, top_k=1)
    assert len(results) == 1
    assert results[0].retrieval_path == "lattice_exact"
    stats = index.stats()
    assert stats.exact_hit_rate == 1.0


def test_public_imports():
    """Verify the top-level package exports the new public API."""
    from latticememory import LatticeIndex, SearchResult, LatticeStats
    assert LatticeIndex is not None
    assert SearchResult is not None
    assert LatticeStats is not None
