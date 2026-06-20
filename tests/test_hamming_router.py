"""Tests for HammingRouter — no real model downloads, uses FakeEncoder."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from latticememory.hamming_router import HammingMatch, HammingRouter
from latticememory.rag.e8_retriever import E8LatticeDB


# ---------------------------------------------------------------------------
# Fake encoder (deterministic, no downloads)
# ---------------------------------------------------------------------------

class FakeEncoder:
    def __init__(self, d_model: int = 1024):
        self.d_model = d_model

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        out = []
        for s in sentences:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            if normalize_embeddings:
                v /= np.linalg.norm(v) + 1e-9
            out.append(v)
        return np.stack(out)


D = 1024


@pytest.fixture
def router():
    return HammingRouter(encoder=FakeEncoder(D), d_model=D, threshold=111)


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------

def test_empty_lookup_returns_none(router):
    assert router.lookup("anything") is None


def test_len_tracks_additions(router):
    assert len(router) == 0
    router.add("hello", "v1")
    assert len(router) == 1
    router.add("world", "v2")
    assert len(router) == 2


def test_clear_resets_store(router):
    router.add("hello", "v1")
    router.clear()
    assert len(router) == 0
    assert router.lookup("hello") is None


def test_same_text_always_hits(router):
    router.add("What is the weather today?", "sunny")
    result = router.lookup("What is the weather today?")
    assert result is not None
    assert result.value == "sunny"
    assert result.hamming_distance == 0


def test_hit_returns_hamming_match(router):
    router.add("Reset my password", "reset_response")
    result = router.lookup("Reset my password")
    assert isinstance(result, HammingMatch)
    assert result.value == "reset_response"
    assert result.hamming_distance == 0
    assert len(result.stored_key) == D // 8  # 128 bytes


# ---------------------------------------------------------------------------
# Threshold control
# ---------------------------------------------------------------------------

def test_threshold_zero_requires_exact_match(router):
    router.add("exact text here", "hit")
    assert router.lookup("exact text here", threshold=0) is not None
    # Different text → different key → no hit at threshold=0
    assert router.lookup("slightly different text here", threshold=0) is None


def test_threshold_128_always_hits(router):
    router.add("any stored text", "value")
    result = router.lookup("completely unrelated query sentence", threshold=128)
    assert result is not None  # always within 128 blocks


def test_per_lookup_threshold_overrides_router_threshold(router):
    router.threshold = 0
    router.add("hello world", "val")
    assert router.lookup("hello world", threshold=128) is not None


# ---------------------------------------------------------------------------
# add_from_key / lookup_key
# ---------------------------------------------------------------------------

def test_add_from_key_then_lookup_key(router):
    key = router.e8_key("canonical query text")
    router.add_from_key(key, "cached_response")
    result = router.lookup_key(key, threshold=0)
    assert result is not None
    assert result.value == "cached_response"
    assert result.hamming_distance == 0


def test_e8_key_is_deterministic(router):
    k1 = router.e8_key("hello")
    k2 = router.e8_key("hello")
    assert k1 == k2


def test_e8_key_length(router):
    key = router.e8_key("any text")
    assert len(key) == D // 8  # 128 bytes for 1024D


def test_lookup_key_empty_returns_none(router):
    key = router.e8_key("whatever")
    assert router.lookup_key(key) is None


# ---------------------------------------------------------------------------
# Multiple stored keys — nearest-neighbor selection
# ---------------------------------------------------------------------------

def test_nearest_key_selected(router):
    # Store two keys; lookup same text as first — should return first
    k1 = router.add("first query text", "first_value")
    k2 = router.add("second query text", "second_value")
    result = router.lookup("first query text", threshold=128)
    assert result is not None
    assert result.value == "first_value"
    assert result.hamming_distance == 0


def test_lookup_returns_none_when_all_beyond_threshold():
    # Use threshold=0 so only exact matches count, look up different text
    enc = FakeEncoder(D)
    r = HammingRouter(encoder=enc, d_model=D, threshold=0)
    r.add("stored query", "value")
    assert r.lookup("completely different text that wont match") is None


# ---------------------------------------------------------------------------
# fragmentation_score
# ---------------------------------------------------------------------------

def test_fragmentation_score_single_text(router):
    score = router.fragmentation_score(["only one text"])
    assert score == 1.0


def test_fragmentation_score_identical_texts(router):
    text = "same text repeated"
    score = router.fragmentation_score([text, text, text])
    assert score == 1.0  # all identical → all pairs match at d=0


def test_fragmentation_score_threshold_128(router):
    texts = ["alpha phrase here", "beta phrase there", "gamma text now"]
    score = router.fragmentation_score(texts, threshold=128)
    assert score == 1.0  # everything matches within 128


def test_fragmentation_score_threshold_0_different_texts(router):
    texts = ["alpha phrase here", "completely different text"]
    score = router.fragmentation_score(texts, threshold=0)
    assert score == 0.0  # different texts → different keys → no match


def test_fragmentation_score_range(router):
    texts = ["word one", "word two", "word three", "word four"]
    score = router.fragmentation_score(texts)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# separation_score
# ---------------------------------------------------------------------------

def test_separation_score_single_anchor(router):
    score = router.separation_score(["only one"])
    assert score == 1.0


def test_separation_score_threshold_0_different_texts(router):
    texts = ["alpha topic", "beta topic"]
    score = router.separation_score(texts, threshold=0)
    assert score == 1.0  # both differ by >0 → separated


def test_separation_score_threshold_128_same_key(router):
    text = "identical text"
    score = router.separation_score([text, text], threshold=128)
    assert score == 0.0  # d=0 <= 128 → not separated


def test_separation_score_range(router):
    anchors = ["refunds policy", "order tracking", "password reset"]
    score = router.separation_score(anchors)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Public import check
# ---------------------------------------------------------------------------

def test_public_imports():
    from latticememory import HammingRouter, HammingMatch
    assert HammingRouter is not None
    assert HammingMatch is not None


# ---------------------------------------------------------------------------
# _batch_pair_hamming
# ---------------------------------------------------------------------------

def test_batch_pair_hamming_returns_correct_count(router):
    pairs = [("hello world", "goodbye world"), ("foo bar", "baz qux")]
    dists = router._batch_pair_hamming(pairs)
    assert len(dists) == 2


def test_batch_pair_hamming_empty_returns_empty(router):
    assert router._batch_pair_hamming([]) == []


def test_batch_pair_hamming_distances_in_valid_range(router):
    pairs = [("alpha text here", "beta text there"), ("cancel my plan", "reset password")]
    dists = router._batch_pair_hamming(pairs)
    n_blocks = D // 8
    for d in dists:
        assert 0 <= d <= n_blocks


def test_batch_pair_hamming_identical_text_is_zero(router):
    pairs = [("exact same text", "exact same text")]
    dists = router._batch_pair_hamming(pairs)
    assert dists[0] == 0


# ---------------------------------------------------------------------------
# gap_stats
# ---------------------------------------------------------------------------

_IDENTICAL_PARAPHRASE_PAIRS = [
    ("what is the weather", "what is the weather"),
    ("cancel my order", "cancel my order"),
    ("reset my password", "reset my password"),
]

_DISTINCT_NEAR_MISS_PAIRS = [
    ("cancel my Netflix subscription", "cancel my gym membership"),
    ("weather in Paris tomorrow", "weather in London tomorrow"),
    ("refund policy for Amazon", "refund policy for eBay"),
]


def test_gap_stats_returns_required_keys(router):
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    for key in ("paraphrase", "near_miss", "gap", "threshold_table",
                "n_paraphrase_pairs", "n_near_miss_pairs"):
        assert key in result


def test_gap_stats_n_pairs_match_input(router):
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    assert result["n_paraphrase_pairs"] == len(_IDENTICAL_PARAPHRASE_PAIRS)
    assert result["n_near_miss_pairs"] == len(_DISTINCT_NEAR_MISS_PAIRS)


def test_gap_stats_identical_paraphrase_has_p95_zero(router):
    # Same text → Hamming=0, so all percentiles including p95 are 0
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    assert result["paraphrase"]["min"] == 0
    assert result["paraphrase"]["p95"] == 0.0


def test_gap_stats_positive_gap_for_identical_paraphrases(router):
    # Identical paraphrase pairs → p95=0; near-miss pairs are different texts → p5 > 0
    # Therefore gap = near_miss_p5 - paraphrase_p95 > 0
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    assert result["gap"] > 0.0


def test_gap_stats_threshold_table_has_entries(router):
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    table = result["threshold_table"]
    assert len(table) > 0
    # First entry is threshold=0, last is threshold=n_blocks
    assert table[0]["threshold"] == 0
    assert table[-1]["threshold"] == D // 8


def test_gap_stats_threshold_table_recall_monotone(router):
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    recalls = [r["recall"] for r in result["threshold_table"]]
    assert recalls == sorted(recalls)


def test_gap_stats_threshold_table_fp_rate_monotone(router):
    result = router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS)
    fp_rates = [r["fp_rate"] for r in result["threshold_table"]]
    assert fp_rates == sorted(fp_rates)


def test_gap_stats_rejects_empty_pair_sets(router):
    with pytest.raises(ValueError, match="paraphrase_pairs"):
        router.gap_stats([], _DISTINCT_NEAR_MISS_PAIRS)
    with pytest.raises(ValueError, match="near_miss_pairs"):
        router.gap_stats(_IDENTICAL_PARAPHRASE_PAIRS, [])


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------

def test_calibrate_threshold_returns_required_keys(router):
    result = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS
    )
    for key in ("threshold", "recall", "fp_rate", "fp_budget",
                "n_paraphrase_pairs", "n_near_miss_pairs"):
        assert key in result


def test_calibrate_threshold_fp_rate_within_budget(router):
    for budget in (0.0, 0.1, 0.5):
        result = router.calibrate_threshold(
            _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS, fp_budget=budget
        )
        assert result["fp_rate"] <= budget + 1e-9, (
            f"fp_rate {result['fp_rate']} exceeded budget {budget}"
        )


def test_calibrate_threshold_identical_paraphrases_recall_one(router):
    # Identical text pairs → Hamming=0 → recall=1.0 at any threshold that passes FP budget
    result = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS, fp_budget=0.0
    )
    assert result["recall"] == 1.0


def test_calibrate_threshold_fp_budget_recorded(router):
    result = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS, fp_budget=0.1
    )
    assert result["fp_budget"] == 0.1


def test_calibrate_threshold_n_pairs_match_input(router):
    result = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS
    )
    assert result["n_paraphrase_pairs"] == len(_IDENTICAL_PARAPHRASE_PAIRS)
    assert result["n_near_miss_pairs"] == len(_DISTINCT_NEAR_MISS_PAIRS)


def test_calibrate_threshold_higher_budget_gives_higher_or_equal_threshold(router):
    r0 = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS, fp_budget=0.0
    )
    r1 = router.calibrate_threshold(
        _IDENTICAL_PARAPHRASE_PAIRS, _DISTINCT_NEAR_MISS_PAIRS, fp_budget=0.5
    )
    assert r1["threshold"] >= r0["threshold"]


def test_calibrate_threshold_rejects_empty_pair_sets(router):
    with pytest.raises(ValueError, match="paraphrase_pairs"):
        router.calibrate_threshold([], _DISTINCT_NEAR_MISS_PAIRS)
    with pytest.raises(ValueError, match="near_miss_pairs"):
        router.calibrate_threshold(_IDENTICAL_PARAPHRASE_PAIRS, [])


# ---------------------------------------------------------------------------
# cosine calibration
# ---------------------------------------------------------------------------


class CosineFixtureEncoder:
    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = str(sentence).lower()
            if "paraphrase" in text or "same" in text:
                vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "near" in text:
                vector = np.array([0.4, 0.916515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                vector = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            vector /= np.linalg.norm(vector) + 1e-9
            vectors.append(vector)
        return np.stack(vectors)


def test_cosine_gap_stats_returns_positive_gap():
    router = HammingRouter(encoder=CosineFixtureEncoder(), d_model=8)

    result = router.cosine_gap_stats(
        [("same canonical", "same paraphrase")],
        [("same canonical", "near miss")],
    )

    assert result["paraphrase"]["min"] == pytest.approx(1.0)
    assert result["near_miss"]["max"] == pytest.approx(0.4, abs=1e-5)
    assert result["gap"] > 0.5
    assert result["n_paraphrase_pairs"] == 1
    assert result["n_near_miss_pairs"] == 1


def test_calibrate_cosine_threshold_satisfies_fp_budget():
    router = HammingRouter(encoder=CosineFixtureEncoder(), d_model=8)

    result = router.calibrate_cosine_threshold(
        [("same canonical", "same paraphrase")],
        [("same canonical", "near miss")],
        fp_budget=0.0,
    )

    assert result["threshold"] > 0.4
    assert result["threshold"] <= 1.0
    assert result["recall"] == 1.0
    assert result["fp_rate"] == 0.0


def test_evaluate_threshold_detects_false_accept_cosine():
    router = HammingRouter(encoder=CosineFixtureEncoder(), d_model=8)
    paraphrase_pairs = [("same canonical", "same paraphrase")]
    near_miss_pairs = [("same canonical", "near miss")]

    too_loose = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, 0.3, metric="cosine")
    assert too_loose["false_accepts"] == 1
    assert too_loose["false_accept_rate"] == 1.0
    assert too_loose["false_rejects"] == 0
    assert too_loose["n_paraphrase_pairs"] == 1
    assert too_loose["n_near_miss_pairs"] == 1

    safe = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, 0.5, metric="cosine")
    assert safe["false_accepts"] == 0
    assert safe["false_accept_rate"] == 0.0
    assert safe["false_rejects"] == 0


def test_evaluate_threshold_detects_false_reject_cosine():
    router = HammingRouter(encoder=CosineFixtureEncoder(), d_model=8)
    paraphrase_pairs = [("same canonical", "same paraphrase")]
    near_miss_pairs = [("same canonical", "near miss")]

    too_strict = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, 1.5, metric="cosine")
    assert too_strict["false_rejects"] == 1
    assert too_strict["false_reject_rate"] == 1.0


def test_evaluate_threshold_detects_false_accept_hamming(router):
    paraphrase_pairs = [("question one", "question one restated")]
    near_miss_pairs = [("topic alpha", "topic beta")]
    [near_miss_distance] = router._batch_pair_hamming(near_miss_pairs)

    too_loose = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, near_miss_distance, metric="hamming")
    assert too_loose["false_accepts"] == 1

    safe = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, near_miss_distance - 1, metric="hamming")
    assert safe["false_accepts"] == 0


def test_evaluate_threshold_defaults_to_hamming_metric(router):
    paraphrase_pairs = [("question one", "question one restated")]
    near_miss_pairs = [("topic alpha", "topic beta")]
    [near_miss_distance] = router._batch_pair_hamming(near_miss_pairs)

    result = router.evaluate_threshold(paraphrase_pairs, near_miss_pairs, near_miss_distance)
    assert result["false_accepts"] == 1


def test_evaluate_threshold_rejects_unknown_metric():
    router = HammingRouter(encoder=CosineFixtureEncoder(), d_model=8)
    with pytest.raises(ValueError, match="metric"):
        router.evaluate_threshold([("a", "b")], [("c", "d")], 0.5, metric="euclidean")
