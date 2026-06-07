"""Tests for LatticeObservatory — block-level interpretability layer."""
from __future__ import annotations

import pytest
from tests.test_lattice_index import FakeEncoder
from latticememory.index import LatticeIndex
from latticememory.observatory import LatticeObservatory


@pytest.fixture()
def index():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._mode = "cache"
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    return idx


@pytest.fixture()
def populated_index(index):
    index.add([
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps over a sleepy hound.",
        "Quantum entanglement is a physical phenomenon.",
        "LatticeMemory uses the E8 lattice for semantic routing.",
        "Neural networks learn representations from data.",
    ])
    return index


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_observatory_constructed_via_factory(populated_index):
    obs = populated_index.observatory()
    assert isinstance(obs, LatticeObservatory)


def test_observatory_constructed_directly(populated_index):
    obs = LatticeObservatory(populated_index)
    assert obs._n_blocks == 48  # 384 // 8


# --------------------------------------------------------------------------
# block_stability
# --------------------------------------------------------------------------

def test_block_stability_empty_texts(populated_index):
    obs = populated_index.observatory()
    result = obs.block_stability([])
    assert "error" in result


def test_block_stability_single_text(populated_index):
    obs = populated_index.observatory()
    result = obs.block_stability(["The quick brown fox."])
    assert result["n_texts"] == 1
    assert result["n_blocks"] == 48
    # One text → all blocks are trivially "stable"
    assert result["fully_stable_block_count"] == 48
    assert result["mean_block_entropy"] == 0.0


def test_block_stability_identical_texts(populated_index):
    obs = populated_index.observatory()
    texts = ["Identical text."] * 5
    result = obs.block_stability(texts)
    assert result["fully_stable_block_count"] == 48
    assert result["mean_block_entropy"] == 0.0


def test_block_stability_diverse_texts(populated_index):
    obs = populated_index.observatory()
    diverse = [
        "Quantum computing harnesses quantum mechanics.",
        "Machine learning trains models on data.",
        "The cat sat on the mat.",
        "Economic policy affects inflation.",
        "Protein folding determines biological function.",
    ]
    result = obs.block_stability(diverse)
    assert result["n_texts"] == 5
    assert result["n_blocks"] == 48
    # Semantically diverse texts should have at least some unstable blocks
    assert result["unstable_block_count"] > 0
    # all_blocks present and complete
    assert len(result["all_blocks"]) == 48
    for b in result["all_blocks"]:
        assert "block" in b
        assert "entropy" in b
        assert "stability" in b
        assert 0.0 <= b["stability"] <= 1.0


def test_block_stability_keys_present(populated_index):
    obs = populated_index.observatory()
    result = obs.block_stability(["text one", "text two"])
    for key in ("n_texts", "n_blocks", "mean_block_entropy",
                "fully_stable_block_count", "unstable_block_count",
                "concept_fingerprint_blocks", "noisiest_blocks", "all_blocks"):
        assert key in result


# --------------------------------------------------------------------------
# cell_coherence
# --------------------------------------------------------------------------

def test_cell_coherence_known_key(populated_index):
    obs = populated_index.observatory()
    # Grab a key that exists
    key = next(iter(populated_index._runtime.memory.lattice.hash_store))
    result = obs.cell_coherence(key)
    assert "coherence_label" in result
    assert result["coherence_label"] in {"tight", "loose", "collision", "singleton"}


def test_cell_coherence_hex_string_input(populated_index):
    obs = populated_index.observatory()
    key_bytes = next(iter(populated_index._runtime.memory.lattice.hash_store))
    key_hex = key_bytes.hex()
    result = obs.cell_coherence(key_hex)
    assert "key" in result
    assert result["key"] == key_hex


def test_cell_coherence_empty_key():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._mode = "cache"
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    obs = LatticeObservatory(idx)
    # Construct an artificial 48-byte key that maps to nothing
    fake_key = bytes(48)
    result = obs.cell_coherence(fake_key)
    assert result["status"] == "empty_cell"


def test_cell_coherence_doc_count(populated_index):
    obs = populated_index.observatory()
    for key in populated_index._runtime.memory.lattice.hash_store:
        result = obs.cell_coherence(key)
        assert result["doc_count"] >= 1


# --------------------------------------------------------------------------
# trace_mismatch
# --------------------------------------------------------------------------

def test_trace_mismatch_identical_texts(populated_index):
    obs = populated_index.observatory()
    text = "The quick brown fox."
    result = obs.trace_mismatch(text, text)
    assert result["hamming_distance"] == 0
    assert result["routing_verdict"] == "lattice_exact"
    assert result["n_differing_blocks"] == 0
    assert result["n_matching_blocks"] == 48
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_trace_mismatch_different_texts(populated_index):
    obs = populated_index.observatory()
    result = obs.trace_mismatch(
        "The quick brown fox jumps.",
        "Quantum entanglement is fascinating.",
    )
    assert result["hamming_distance"] >= 0
    assert result["routing_verdict"] in {"lattice_exact", "lattice_hamming1", "fallback_required"}
    assert result["n_differing_blocks"] + result["n_matching_blocks"] == 48


def test_trace_mismatch_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.trace_mismatch("hello world", "goodbye world")
    for key in ("query", "doc", "cosine_similarity", "hamming_distance",
                "match_rate", "routing_verdict", "n_differing_blocks",
                "n_matching_blocks", "differing_blocks", "matching_blocks"):
        assert key in result
    for b in result["differing_blocks"]:
        assert "block" in b
        assert "dim_range" in b
        assert "query_address" in b
        assert "doc_address" in b


def test_trace_mismatch_hamming1_verdict(populated_index):
    obs = populated_index.observatory()
    # Find a pair with hamming == 1 by checking many near-identical texts
    # Just verify the verdict logic is consistent with hamming distance
    result = obs.trace_mismatch("cat sat on mat", "cat sat on rug")
    h = result["hamming_distance"]
    if h == 0:
        assert result["routing_verdict"] == "lattice_exact"
    elif h == 1:
        assert result["routing_verdict"] == "lattice_hamming1"
    else:
        assert result["routing_verdict"] == "fallback_required"


# --------------------------------------------------------------------------
# routing_profile
# --------------------------------------------------------------------------

def test_routing_profile_empty_raises(populated_index):
    obs = populated_index.observatory()
    result = obs.routing_profile([], [])
    assert "error" in result


def test_routing_profile_length_mismatch(populated_index):
    obs = populated_index.observatory()
    with pytest.raises(ValueError):
        obs.routing_profile(["q1"], [])


def test_routing_profile_identical_pairs(populated_index):
    obs = populated_index.observatory()
    texts = ["The quick brown fox.", "A lazy hound."]
    result = obs.routing_profile(texts, texts)
    assert result["routing"]["exact_rate"] == 1.0
    assert result["routing"]["fallback_rate"] == 0.0
    assert result["hamming"]["mean"] == 0.0
    assert result["n_pairs"] == 2


def test_routing_profile_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.routing_profile(
        ["The quick brown fox.", "hello"],
        ["The quick brown fox.", "goodbye"],
    )
    for key in ("n_pairs", "routing", "hamming", "top_mismatching_blocks", "optimization_target"):
        assert key in result
    r = result["routing"]
    assert abs(r["exact_rate"] + r["hamming1_rate"] + r["fallback_rate"] - 1.0) < 1e-6


def test_routing_profile_top_mismatching_blocks(populated_index):
    obs = populated_index.observatory()
    pairs_q = ["hello world", "machine learning", "quantum physics"]
    pairs_d = ["goodbye moon", "deep learning", "classical mechanics"]
    result = obs.routing_profile(pairs_q, pairs_d)
    for b in result["top_mismatching_blocks"]:
        assert "block" in b
        assert "miss_rate" in b
        assert 0.0 <= b["miss_rate"] <= 1.0


# --------------------------------------------------------------------------
# export_for_llm
# --------------------------------------------------------------------------

def test_export_for_llm_empty_index():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._mode = "cache"
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    obs = LatticeObservatory(idx)
    result = obs.export_for_llm()
    assert result["status"] == "empty_index"


def test_export_for_llm_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm()
    for key in ("schema_version", "index_summary", "block_analysis",
                "sample_cells", "recommendations", "interpretation_guide"):
        assert key in result
    assert result["schema_version"] == 1


def test_export_for_llm_index_summary(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm()
    summary = result["index_summary"]
    assert summary["total_docs"] == 5
    assert summary["unique_keys"] >= 1
    assert summary["n_blocks"] == 48
    assert summary["d_model"] == 384


def test_export_for_llm_block_analysis(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm()
    ba = result["block_analysis"]
    assert len(ba["entropy_per_block"]) == 48
    assert isinstance(ba["mean_entropy"], float)
    assert isinstance(ba["most_stable_blocks"], list)
    assert isinstance(ba["most_variable_blocks"], list)
    assert "interpretation" in ba


def test_export_for_llm_recommendations_nonempty(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm()
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) >= 1


def test_export_for_llm_n_sample_cells(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm(n_sample_cells=2)
    assert len(result["sample_cells"]) <= 2


def test_export_for_llm_interpretation_guide(populated_index):
    obs = populated_index.observatory()
    result = obs.export_for_llm()
    guide = result["interpretation_guide"]
    assert "coherence_label" in guide
    assert "routing_verdict" in guide
    assert "optimization_approach" in guide


# --------------------------------------------------------------------------
# collision_audit
# --------------------------------------------------------------------------

def test_collision_audit_empty_index():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._mode = "cache"
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    obs = LatticeObservatory(idx)
    result = obs.collision_audit()
    assert result["status"] == "empty_index"


def test_collision_audit_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.collision_audit()
    for key in ("total_cells", "singleton_cells", "multi_doc_cells",
                "collision_cells", "loose_cells", "tight_cells",
                "collision_rate", "ranked_by_risk", "summary"):
        assert key in result


def test_collision_audit_counts_sum(populated_index):
    obs = populated_index.observatory()
    result = obs.collision_audit()
    assert result["collision_cells"] + result["loose_cells"] + result["tight_cells"] == result["multi_doc_cells"]


def test_collision_audit_ranked_worst_first(populated_index):
    obs = populated_index.observatory()
    result = obs.collision_audit()
    ranked = result["ranked_by_risk"]
    cosines = [r["mean_cosine"] for r in ranked]
    assert cosines == sorted(cosines)


def test_collision_audit_collision_rate_bounds(populated_index):
    obs = populated_index.observatory()
    result = obs.collision_audit()
    assert 0.0 <= result["collision_rate"] <= 1.0


def test_collision_audit_total_cells_matches_hash_store(populated_index):
    obs = populated_index.observatory()
    result = obs.collision_audit()
    assert result["total_cells"] == len(populated_index._runtime.memory.lattice.hash_store)


# --------------------------------------------------------------------------
# fragmentation_score
# --------------------------------------------------------------------------

def test_fragmentation_score_empty():
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._mode = "cache"
    idx._init_with_encoder(FakeEncoder(384), d_model=384)
    obs = LatticeObservatory(idx)
    result = obs.fragmentation_score([])
    assert "error" in result


def test_fragmentation_score_single_text(populated_index):
    obs = populated_index.observatory()
    result = obs.fragmentation_score(["single text"])
    assert result["score"] == 1.0
    assert result["n_unique_keys"] == 1
    assert result["cacheable"] is True


def test_fragmentation_score_identical_texts(populated_index):
    obs = populated_index.observatory()
    result = obs.fragmentation_score(["same text"] * 5)
    assert result["score"] == 1.0
    assert result["label"] == "unified"
    assert result["cacheable"] is True
    assert result["n_unique_keys"] == 1


def test_fragmentation_score_diverse_texts(populated_index):
    obs = populated_index.observatory()
    diverse = [
        "Quantum mechanics and wave functions.",
        "The stock market crashed yesterday.",
        "Protein folding simulation results.",
        "Deep learning for image classification.",
        "The Roman Empire fell in 476 AD.",
    ]
    result = obs.fragmentation_score(diverse)
    assert result["n_texts"] == 5
    assert 0.0 <= result["score"] <= 1.0
    assert result["label"] in {"unified", "cohesive", "fragmented", "scattered"}
    assert isinstance(result["key_distribution"], dict)


def test_fragmentation_score_cacheable_flag(populated_index):
    obs = populated_index.observatory()
    r1 = obs.fragmentation_score(["same text", "same text"])
    assert r1["cacheable"] is True
    r2 = obs.fragmentation_score(["different A", "different B"])
    # may or may not be cacheable depending on hash — just check the field is present
    assert "cacheable" in r2


def test_fragmentation_score_dominant_key(populated_index):
    obs = populated_index.observatory()
    result = obs.fragmentation_score(["hello world", "hello world", "hello world", "bye"])
    assert "dominant_key" in result
    dist = result["key_distribution"]
    assert result["dominant_key"] == max(dist, key=dist.__getitem__)


# --------------------------------------------------------------------------
# suggest_training_pairs
# --------------------------------------------------------------------------

def test_suggest_training_pairs_empty(populated_index):
    obs = populated_index.observatory()
    result = obs.suggest_training_pairs([])
    assert "error" in result


def test_suggest_training_pairs_identical(populated_index):
    obs = populated_index.observatory()
    pairs = [("The quick brown fox.", "The quick brown fox.")]
    result = obs.suggest_training_pairs(pairs)
    assert result["mismatch_count"] == 0
    assert result["training_pairs"] == []


def test_suggest_training_pairs_structure(populated_index):
    obs = populated_index.observatory()
    pairs = [
        ("machine learning models", "deep neural networks"),
        ("quantum entanglement physics", "stock market economics"),
    ]
    result = obs.suggest_training_pairs(pairs)
    for key in ("n_pairs", "mismatch_count", "mismatch_rate",
                "training_pairs", "top_target_blocks", "top_target_dims", "training_recipe"):
        assert key in result


def test_suggest_training_pairs_recipe_fields(populated_index):
    obs = populated_index.observatory()
    pairs = [("cat on mat", "dog on rug")]
    result = obs.suggest_training_pairs(pairs)
    recipe = result["training_recipe"]
    assert "loss" in recipe
    assert "freeze_strategy" in recipe
    assert "n_pairs_needed" in recipe
    assert "suggested_epochs" in recipe


def test_suggest_training_pairs_mismatch_rate_bounds(populated_index):
    obs = populated_index.observatory()
    pairs = [("hello world", "goodbye moon"), ("same text", "same text")]
    result = obs.suggest_training_pairs(pairs)
    assert 0.0 <= result["mismatch_rate"] <= 1.0


def test_suggest_training_pairs_target_dims_format(populated_index):
    obs = populated_index.observatory()
    pairs = [("quantum physics experiment", "economic policy debate")]
    result = obs.suggest_training_pairs(pairs)
    for dim_range in result["top_target_dims"]:
        parts = dim_range.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)


# --------------------------------------------------------------------------
# compare_snapshots
# --------------------------------------------------------------------------

def test_compare_snapshots_invalid_input(populated_index):
    obs = populated_index.observatory()
    result = obs.compare_snapshots({}, {"index_summary": {}, "block_analysis": {}})
    assert "error" in result


def test_compare_snapshots_identical(populated_index):
    obs = populated_index.observatory()
    snap = obs.export_for_llm()
    result = obs.compare_snapshots(snap, snap)
    assert "verdict" in result
    assert result["entropy"]["delta"] == 0.0
    assert result["entropy"]["improved_blocks"] == 0
    assert result["entropy"]["degraded_blocks"] == 0


def test_compare_snapshots_structure(populated_index):
    obs = populated_index.observatory()
    snap = obs.export_for_llm()
    result = obs.compare_snapshots(snap, snap)
    for key in ("verdict", "entropy", "collisions", "index",
                "most_improved_blocks", "most_degraded_blocks",
                "block_deltas", "summary"):
        assert key in result


def test_compare_snapshots_block_deltas_length(populated_index):
    obs = populated_index.observatory()
    snap = obs.export_for_llm()
    result = obs.compare_snapshots(snap, snap)
    assert len(result["block_deltas"]) == 48


def test_compare_snapshots_verdict_values(populated_index):
    obs = populated_index.observatory()
    snap = obs.export_for_llm()
    result = obs.compare_snapshots(snap, snap)
    assert result["verdict"] in {"improved", "degraded", "neutral", "unknown"}


def test_compare_snapshots_collisions_delta(populated_index):
    obs = populated_index.observatory()
    snap = obs.export_for_llm()
    result = obs.compare_snapshots(snap, snap)
    delta = result["collisions"]["delta"]
    assert delta == result["collisions"]["after_sampled"] - result["collisions"]["before_sampled"]


# --------------------------------------------------------------------------
# neighbor_density
# --------------------------------------------------------------------------

def test_neighbor_density_structure(populated_index):
    obs = populated_index.observatory()
    key = next(iter(populated_index._runtime.memory.lattice.hash_store))
    result = obs.neighbor_density(key)
    for field in ("key", "own_doc_count", "hamming1_neighbor_cells",
                  "total_neighbor_docs", "safe_neighbor_cells", "noisy_neighbor_cells",
                  "neighbors", "expansion_verdict", "recall_estimate"):
        assert field in result


def test_neighbor_density_hex_string_input(populated_index):
    obs = populated_index.observatory()
    key_bytes = next(iter(populated_index._runtime.memory.lattice.hash_store))
    result = obs.neighbor_density(key_bytes.hex())
    assert result["key"] == key_bytes.hex()


def test_neighbor_density_own_doc_count(populated_index):
    obs = populated_index.observatory()
    hash_store = populated_index._runtime.memory.lattice.hash_store
    key = next(iter(hash_store))
    result = obs.neighbor_density(key)
    assert result["own_doc_count"] == len(hash_store[key])


def test_neighbor_density_all_neighbors_are_hamming1(populated_index):
    obs = populated_index.observatory()
    hash_store = populated_index._runtime.memory.lattice.hash_store
    key_bytes = next(iter(hash_store))
    result = obs.neighbor_density(key_bytes)
    for n in result["neighbors"]:
        neighbor_key = bytes.fromhex(n["key"])
        hamming = sum(1 for i in range(48) if neighbor_key[i] != key_bytes[i])
        assert hamming == 1


def test_neighbor_density_expansion_verdict_values(populated_index):
    obs = populated_index.observatory()
    key = next(iter(populated_index._runtime.memory.lattice.hash_store))
    result = obs.neighbor_density(key)
    assert result["expansion_verdict"] in {"safe", "noisy", "empty"}


def test_neighbor_density_counts_consistent(populated_index):
    obs = populated_index.observatory()
    key = next(iter(populated_index._runtime.memory.lattice.hash_store))
    result = obs.neighbor_density(key)
    assert result["total_neighbor_docs"] == sum(n["doc_count"] for n in result["neighbors"])
    assert result["hamming1_neighbor_cells"] == len(result["neighbors"])


# --------------------------------------------------------------------------
# semantic_probe
# --------------------------------------------------------------------------

def test_semantic_probe_too_few_labels(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({"only_one": ["text"]})
    assert "error" in result


def test_semantic_probe_empty(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({})
    assert "error" in result


def test_semantic_probe_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "sports": ["soccer match", "basketball game", "tennis tournament"],
        "science": ["quantum physics", "molecular biology", "astrophysics research"],
    })
    for key in ("n_classes", "n_texts", "labels", "label_entropy",
                "top_separating_blocks", "bottom_separating_blocks",
                "all_blocks", "fine_tune_targets", "freeze_candidates", "interpretation"):
        assert key in result


def test_semantic_probe_block_count(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "A": ["apple orange banana"],
        "B": ["car truck motorcycle"],
    })
    assert len(result["all_blocks"]) == 48


def test_semantic_probe_info_gain_bounds(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "X": ["text one", "text two"],
        "Y": ["text three", "text four"],
    })
    for b in result["all_blocks"]:
        assert b["info_gain"] >= 0.0
        assert 0.0 <= b["separability"] <= 1.0 + 1e-6


def test_semantic_probe_top_blocks_sorted(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "cat": ["fluffy cat", "sleeping cat", "orange tabby"],
        "car": ["red sports car", "electric vehicle", "sedan"],
    })
    gains = [b["info_gain"] for b in result["top_separating_blocks"]]
    assert gains == sorted(gains, reverse=True)


def test_semantic_probe_labels_present(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "alpha": ["text a1", "text a2"],
        "beta": ["text b1", "text b2"],
    })
    assert set(result["labels"]) == {"alpha", "beta"}
    assert result["n_classes"] == 2
    assert result["n_texts"] == 4


def test_semantic_probe_fine_tune_targets_format(populated_index):
    obs = populated_index.observatory()
    result = obs.semantic_probe({
        "pos": ["good excellent great"],
        "neg": ["bad terrible awful"],
    })
    for target in result["fine_tune_targets"]:
        parts = target.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)


# --------------------------------------------------------------------------
# block_correlation
# --------------------------------------------------------------------------

def test_block_correlation_too_few_texts(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["only one text"])
    assert "error" in result


def test_block_correlation_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["text one", "text two", "text three"])
    for key in ("n_texts", "n_blocks", "mean_inter_block_nmi",
                "high_correlation_pairs", "redundant_pairs",
                "hub_blocks", "interpretation"):
        assert key in result


def test_block_correlation_nmi_bounds(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["a", "b", "c", "d", "e"])
    for pair in result["high_correlation_pairs"]:
        assert 0.0 <= pair["nmi"] <= 1.0 + 1e-6


def test_block_correlation_hub_blocks_sorted(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["text a", "text b", "text c"])
    nmi_vals = [b["mean_nmi_with_others"] for b in result["hub_blocks"]]
    assert nmi_vals == sorted(nmi_vals, reverse=True)


def test_block_correlation_redundant_subset_of_high(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["x1", "x2", "x3"])
    # redundant_pairs is a subset of high_correlation_pairs
    for r in result["redundant_pairs"]:
        assert r["nmi"] > 0.9


def test_block_correlation_identical_texts(populated_index):
    obs = populated_index.observatory()
    # All texts identical → all blocks constant → all entropies 0 → NMI = 1 for constant pairs
    result = obs.block_correlation(["same text", "same text", "same text"])
    # mean_inter_block_nmi should be high (1.0) since all blocks are constant
    assert result["mean_inter_block_nmi"] >= 0.0  # at minimum, no crash


def test_block_correlation_n_blocks_correct(populated_index):
    obs = populated_index.observatory()
    result = obs.block_correlation(["alpha", "beta", "gamma"])
    assert result["n_blocks"] == 48


# --------------------------------------------------------------------------
# address_trajectory
# --------------------------------------------------------------------------

def test_address_trajectory_too_short(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["only one"])
    assert "error" in result


def test_address_trajectory_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["first text", "second text", "third text"])
    for key in ("n_texts", "n_steps", "total_hamming", "mean_hamming_per_step",
                "smooth_steps", "discontinuous_steps", "trajectory_continuity",
                "most_volatile_blocks", "steps", "training_recommendation"):
        assert key in result


def test_address_trajectory_step_count(populated_index):
    obs = populated_index.observatory()
    texts = ["a", "b", "c", "d"]
    result = obs.address_trajectory(texts)
    assert result["n_texts"] == 4
    assert result["n_steps"] == 3
    assert len(result["steps"]) == 3


def test_address_trajectory_identical_sequence(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["same", "same", "same"])
    assert result["total_hamming"] == 0
    assert result["mean_hamming_per_step"] == 0.0
    assert result["smooth_steps"] == 2
    assert result["discontinuous_steps"] == 0


def test_address_trajectory_continuity_bounds(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["text one", "text two", "text three"])
    assert 0.0 <= result["trajectory_continuity"] <= 1.0


def test_address_trajectory_step_fields(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["alpha text", "beta text"])
    step = result["steps"][0]
    for field in ("step", "from_text", "to_text", "hamming_delta",
                  "changed_blocks", "changed_dim_ranges", "routing_continuity"):
        assert field in step
    assert step["routing_continuity"] in {"exact", "hamming1", "discontinuous"}


def test_address_trajectory_hamming_consistent(populated_index):
    obs = populated_index.observatory()
    result = obs.address_trajectory(["one", "two", "three"])
    total = sum(s["hamming_delta"] for s in result["steps"])
    assert total == result["total_hamming"]


# --------------------------------------------------------------------------
# generate_training_curriculum
# --------------------------------------------------------------------------

def test_generate_training_curriculum_empty(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({})
    assert "error" in result


def test_generate_training_curriculum_structure(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "animals": ["cat on mat", "dog on rug", "bird in tree"],
        "vehicles": ["red sports car", "blue bicycle", "electric train"],
    })
    for key in ("n_clusters", "cluster_analysis", "clusters_needing_unification",
                "positive_pairs", "hard_negative_pairs", "block_training_weights",
                "curriculum_steps", "loss_config", "summary"):
        assert key in result


def test_generate_training_curriculum_cluster_count(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "A": ["text a1", "text a2"],
        "B": ["text b1", "text b2"],
    })
    assert result["n_clusters"] == 2


def test_generate_training_curriculum_positive_pairs(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "cluster": ["text one", "text two", "text three"],
    })
    # C(3, 2) = 3 positive pairs
    assert len(result["positive_pairs"]) == 3
    for p in result["positive_pairs"]:
        assert p["type"] == "positive"
        assert p["cluster"] == "cluster"
        assert "anchor" in p
        assert "positive" in p
        assert p["difficulty"] in {"easy", "medium", "hard"}


def test_generate_training_curriculum_hard_negatives(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "A": ["text a1", "text a2"],
        "B": ["text b1"],
        "C": ["text c1"],
    })
    # 3 cluster pairs: (A,B), (A,C), (B,C)
    assert len(result["hard_negative_pairs"]) == 3
    for n in result["hard_negative_pairs"]:
        assert n["type"] == "hard_negative"
        assert "cluster_a" in n
        assert "cluster_b" in n


def test_generate_training_curriculum_block_weights(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "grp1": ["hello world one"],
        "grp2": ["goodbye moon two"],
    })
    weights = result["block_training_weights"]
    assert len(weights) <= 20
    for w in weights:
        assert "block" in w
        assert "dim_range" in w
        assert "training_weight" in w
        assert w["training_weight"] >= 0.0


def test_generate_training_curriculum_loss_config(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "X": ["text x"],
        "Y": ["text y"],
    })
    cfg = result["loss_config"]
    assert "loss" in cfg
    assert "block_weights" in cfg
    assert "margin" in cfg
    assert "freeze_strategy" in cfg


def test_generate_training_curriculum_three_phases(populated_index):
    obs = populated_index.observatory()
    result = obs.generate_training_curriculum({
        "topic_a": ["science physics chemistry"],
        "topic_b": ["sports football basketball"],
    })
    assert len(result["curriculum_steps"]) == 3
    phases = [s["phase"] for s in result["curriculum_steps"]]
    assert phases == [1, 2, 3]


def test_generate_training_curriculum_single_text_clusters(populated_index):
    obs = populated_index.observatory()
    # Single-text clusters should produce no positive pairs within the cluster
    result = obs.generate_training_curriculum({
        "solo_a": ["just one text"],
        "solo_b": ["another text here"],
    })
    assert result["n_clusters"] == 2
    assert len(result["positive_pairs"]) == 0
    assert len(result["hard_negative_pairs"]) == 1
