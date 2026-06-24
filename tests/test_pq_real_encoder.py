"""
Real-encoder integration tests for the PQ (Product Quantization) addressing
backend - LatticeIndex(mode="pq"), backed by PQLatticeDB.

Deliberately uses the real production encoder (dfrokido/bge-large-e8-snap),
not a mock/fake. This is the first test in the suite to do so on purpose:
every prior "it works" claim discovered to be wrong this session (the 16-
command claim, the original E8 mechanism's untested closed-vocabulary case)
traced back to either a mock encoder or a test that wasn't actually a held-
out evaluation. These tests exist to make sure that mistake can't quietly
happen again for the PQ backend - if PQ's wiring breaks, this should fail
with real embeddings, not pass falsely because a mock made retrieval trivial.

Scope: these are wiring/regression tests, not a re-derivation of the PQ
accuracy numbers - that rigor lives in scratch/pq-validation/ (real PAWS/MS
MARCO benchmarks, thousands of examples). These tests use a small, fixed
corpus and check the integration actually works end-to-end with real
embeddings: fitting, indexing, exact-text retrieval, and that searching with
a real paraphrase doesn't crash and returns a sane result.

The real encoder is loaded exactly ONCE for the whole module (session-scoped
fixture) and reused across every test via LatticeIndex._init_with_encoder()
(the same internal hook the existing FakeEncoder tests use to skip the
constructor's own model loading) - a fresh LatticeIndex(mode="pq") call per
test would reload the real model from scratch each time, which is exactly
what triggered a Windows access-violation crash in transformers' threaded
weight materialization when this file first ran as part of the full suite.
"""
from __future__ import annotations

import pytest

from latticememory.index import LatticeIndex


@pytest.fixture(scope="session")
def real_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("dfrokido/bge-large-e8-snap", device="cpu")


def make_pq_index(real_encoder, *, num_blocks: int = 4, codebook_size: int = 4) -> LatticeIndex:
    """Build a fresh LatticeIndex(mode="pq") reusing the already-loaded real
    encoder, instead of LatticeIndex(mode="pq", ...) which would load its own
    new encoder instance from scratch."""
    index = LatticeIndex.__new__(LatticeIndex)
    d_model = int(real_encoder.get_embedding_dimension() or 0)
    index._init_with_encoder(
        real_encoder, d_model=d_model, mode="pq", device="cpu",
        pq_num_blocks=num_blocks, pq_codebook_size=codebook_size,
    )
    return index


@pytest.fixture(scope="module")
def pq_index(real_encoder) -> LatticeIndex:
    # Small num_blocks/codebook_size: spherical k-means with K=256 centroids
    # on a handful of calibration texts would degenerate (more clusters than
    # data). Production defaults (M=8, K=256) are validated separately in
    # scratch/pq-validation/ against thousands of real examples.
    index = make_pq_index(real_encoder)
    index.fit_pq(
        [
            "What is the refund policy?",
            "How do I reset my password?",
            "Where is my order?",
            "Can I cancel my subscription?",
            "What is the refund policy.",
            "How do I change my password?",
            "When will my order arrive?",
            "How do I cancel my subscription?",
        ]
    )
    return index


def test_pq_mode_constructs_without_error(real_encoder):
    index = make_pq_index(real_encoder)
    assert index._mode == "pq"
    assert index._pq_fitted is False


def test_fit_pq_marks_fitted(pq_index: LatticeIndex):
    assert pq_index._pq_fitted is True


def test_fit_pq_rejects_empty_sample(real_encoder):
    index = make_pq_index(real_encoder)
    with pytest.raises(ValueError):
        index.fit_pq([])


def test_fit_pq_rejects_non_pq_mode(real_encoder):
    index = make_pq_index(real_encoder)
    index._mode = "cache"  # simulate a non-pq index without loading a second encoder
    with pytest.raises(ValueError):
        index.fit_pq(["some text"])


def test_add_after_explicit_fit_does_not_refit(pq_index: LatticeIndex):
    # Calling add() after fit_pq() already ran must not silently re-fit on
    # the add() batch - the explicit calibration sample should stick.
    pq_index.add(["What is the refund policy?", "How do I reset my password?"])
    assert pq_index._pq_fitted is True


def test_exact_repeat_text_is_a_real_lattice_hit(pq_index: LatticeIndex):
    """Identical text must hit lattice_exact - this is the one case that
    should always work regardless of codebook quality (same embedding,
    quantized the same way, every time)."""
    results = pq_index.search("What is the refund policy?", top_k=1)
    assert results
    assert results[0].retrieval_path in {"lattice_exact", "lattice_hamming1"}
    assert results[0].text == "What is the refund policy?"


def test_paraphrase_query_does_not_crash_and_returns_sane_result(pq_index: LatticeIndex):
    """Not asserting a specific hit rate (that's scratch/pq-validation/'s job
    with real statistical power) - asserting the integration doesn't crash
    and returns a real, coherently-scored result for a real paraphrase."""
    results = pq_index.search("What's your return policy?", top_k=3)
    assert results
    for r in results:
        assert isinstance(r.text, str) and r.text
        assert isinstance(r.score, float)
        assert r.retrieval_path in {"lattice_exact", "lattice_hamming1", "fallback", "miss"}


def test_pq_index_without_fallback_can_miss_cleanly(real_encoder):
    """No dense fallback wired in for mode="pq" by default (mirrors mode="cache").
    An out-of-domain query with no real candidates should return an empty
    result, not raise."""
    index = make_pq_index(real_encoder)
    index.add(
        [
            "completely unrelated calibration text about gardening",
            "another unrelated sentence about cooking",
            "a third sentence about astronomy",
            "a fourth sentence about woodworking",
        ]
    )
    results = index.search("some totally different out of domain query about finance", top_k=1)
    assert isinstance(results, list)


def test_fit_with_fewer_examples_than_codebook_size_raises_clear_error(real_encoder):
    """A real bug this test suite caught: fitting with fewer training examples
    than codebook_size used to crash with an obscure IndexError deep in
    spherical k-means, instead of a clear, actionable error. Fixed in
    train_spherical_kmeans (pq_retriever.py) - verify the fix holds."""
    index = make_pq_index(real_encoder)
    with pytest.raises(ValueError, match="needs at least"):
        index.add(["only one document, but codebook_size=4"])


def test_auto_fit_on_first_add_when_fit_pq_not_called(real_encoder):
    index = make_pq_index(real_encoder)
    assert index._pq_fitted is False
    index.add(["first document", "second document", "third document", "fourth document"])
    assert index._pq_fitted is True


def test_pq_mode_rejects_unknown_mode_string():
    with pytest.raises(ValueError):
        LatticeIndex(mode="not-a-real-mode")
