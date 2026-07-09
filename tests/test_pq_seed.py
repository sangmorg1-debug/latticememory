"""Tests for latticememory.pq_seed's Q&A-pairs file loader."""
from __future__ import annotations

import json

from latticememory.pq_seed import load_qa_pairs_file


def test_load_qa_pairs_file_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"question": "What is the refund policy?", "answer": "30 days."}\n'
        '{"question": "How do I reset my password?", "answer": "Use the reset link."}\n',
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert len(rows) == 2
    assert rows[0]["question"] == "What is the refund policy?"
    assert rows[0]["answer"] == "30 days."


def test_load_qa_pairs_file_reads_json_list(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([{"prompt": "Hi", "value": "Hello!"}]),
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"prompt": "Hi", "value": "Hello!"}]


def test_load_qa_pairs_file_reads_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("question,answer,intent_id\nHi,Hello!,greeting\n", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"question": "Hi", "answer": "Hello!", "intent_id": "greeting"}]


def test_load_qa_pairs_file_missing_returns_empty(tmp_path):
    rows = load_qa_pairs_file(str(tmp_path / "does_not_exist.jsonl"))

    assert rows == []


def test_load_qa_pairs_file_unsupported_extension_returns_empty(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("not a supported format", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == []


# Tests for Task 2: build_pq_cache_from_qa_pairs and build_pq_cache_from_qa_file

import numpy as np
import hashlib


class FakeEncoder:
    """Deterministic fake encoder that hashes text to a fixed-dim vector.

    Same pattern as tests/test_lattice_index.py's FakeEncoder -- avoids
    downloading model weights in unit tests.
    """

    def __init__(self, d_model: int = 32):
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


def test_build_pq_cache_from_qa_pairs_seeds_entries():
    from latticememory.pq_seed import build_pq_cache_from_qa_pairs
    from latticememory.rag.pq_retriever import PQLatticeDB

    qa_pairs = [
        {"question": "What is the refund policy?", "answer": "30 days."},
        {"question": "How do I reset my password?", "answer": "Use the reset link."},
        {"question": "Where is my order?", "answer": "Check your email for tracking."},
        {"question": "Can I cancel my subscription?", "answer": "Yes, anytime."},
        {"question": "What are your hours?", "answer": "9am-5pm EST."},
        {"question": "Do you have a mobile app?", "answer": "Yes, iOS and Android."},
        {"question": "What's your return policy?", "answer": "30 days, no questions asked."},
        {"question": "How do I contact support?", "answer": "Email support@example.com."},
    ]

    cache = build_pq_cache_from_qa_pairs(
        qa_pairs, encoder=FakeEncoder(32), d_model=32, pq_num_blocks=4, pq_codebook_size=8,
    )

    assert isinstance(cache.runtime.memory.lattice, PQLatticeDB)
    result = cache.get("What is the refund policy?")
    assert result.hit is True
    assert result.value == "30 days."


def test_build_pq_cache_from_qa_pairs_uses_default_validated_pq_config():
    from latticememory.pq_seed import (
        DEFAULT_PQ_CODEBOOK_SIZE,
        DEFAULT_PQ_NUM_BLOCKS,
        build_pq_cache_from_qa_pairs,
    )

    qa_pairs = [
        {"question": "Hi", "answer": "Hello!"},
        {"question": "How are you?", "answer": "I'm good."},
        {"question": "What's up?", "answer": "Not much."},
        {"question": "How's it going?", "answer": "Great!"},
        {"question": "What time is it?", "answer": "2pm."},
        {"question": "What day is it?", "answer": "Monday."},
        {"question": "Where are you?", "answer": "Home."},
        {"question": "What are you doing?", "answer": "Working."},
    ]

    cache = build_pq_cache_from_qa_pairs(qa_pairs, encoder=FakeEncoder(32), d_model=32, pq_num_blocks=4, pq_codebook_size=8)
    lattice = cache.runtime.memory.lattice

    # Explicit args above override the defaults -- this test just proves
    # the DEFAULT_* constants exist and equal the validated sweet spot,
    # which is what Task 3's proxy_server.py wiring will actually rely on.
    assert DEFAULT_PQ_NUM_BLOCKS == 8
    assert DEFAULT_PQ_CODEBOOK_SIZE == 256
    assert lattice.num_blocks == 4  # explicit override was honored, not the default
    assert lattice.codebook_size == 8


def test_build_pq_cache_from_qa_pairs_empty_list_raises():
    from latticememory.pq_seed import build_pq_cache_from_qa_pairs

    import pytest
    with pytest.raises(ValueError, match="no Q&A pairs"):
        build_pq_cache_from_qa_pairs([], encoder=FakeEncoder(32), d_model=32)
