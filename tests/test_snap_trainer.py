"""Tests for SnapTrainer — symmetric E8 snap encoder training with Observatory eval.

These tests use FakeEncoder (no real model downloads) and tiny synthetic data so
they run in CI without GPU or internet access. The training loop itself is tested
with a TrainableFakeEncoder that adds tokenize()/forward() on top of FakeEncoder.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from latticememory.index import LatticeIndex
from latticememory.snap_trainer import (
    SnapEpochMetrics,
    SnapObsCheckpoint,
    SnapTrainResult,
    SnapTrainer,
    SnapTrainingConfig,
    _is_better_snap_checkpoint,
)
from latticememory.training import RoutingTrainingExample


# ---------------------------------------------------------------------------
# Fake encoders
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Deterministic hash-based encoder for observatory/eval tests (no grad)."""

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        result = []
        for s in sentences:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            result.append(v)
        return np.stack(result)


class TrainableFakeEncoder(nn.Module):
    """FakeEncoder that also supports tokenize()+forward() for training loop tests.

    tokenize() returns a dict with token IDs derived from MD5 hashes.
    forward() ignores those IDs and returns a fixed linear projection of
    hash-based vectors — still deterministic, just gradient-capable.
    """

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.d_model = d_model
        self._proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self._proj.weight)
        self._fake = FakeEncoder(d_model)

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        return self._fake.encode(sentences, batch_size=batch_size)

    def tokenize(self, texts: list[str]) -> dict:
        vecs = torch.tensor(self._fake.encode(texts), dtype=torch.float32)
        return {"_raw": vecs}

    def forward(self, features: dict) -> dict:
        raw = features["_raw"].to(next(self._proj.parameters()).device)
        emb = self._proj(raw)
        return {"sentence_embedding": emb}

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=recurse)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VAL_CLUSTERS = {
    "greetings": ["Hello!", "Hi there!", "Hey!", "Good day!"],
    "farewells": ["Goodbye!", "See you later!", "Bye!", "Take care!"],
}

D_MODEL = 384


@pytest.fixture
def fake_trainer():
    return SnapTrainer._from_encoder(
        FakeEncoder(D_MODEL), val_clusters=VAL_CLUSTERS, d_model=D_MODEL
    )


@pytest.fixture
def trainable_trainer():
    enc = TrainableFakeEncoder(D_MODEL)
    return SnapTrainer._from_encoder(enc, val_clusters=VAL_CLUSTERS, d_model=D_MODEL)


def _tiny_examples(n: int = 8) -> list[RoutingTrainingExample]:
    texts = [
        ("I want a refund", "Can I get my money back?"),
        ("What is the weather?", "How's the forecast today?"),
        ("Reset my password", "I forgot my login credentials"),
        ("Track my order", "Where is my package?"),
    ]
    out = []
    for i in range(n):
        q, p = texts[i % len(texts)]
        out.append(RoutingTrainingExample(query=q, positive=p, negatives=[]))
    return out


# ---------------------------------------------------------------------------
# SnapTrainingConfig tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SnapTrainingConfig()
    assert cfg.epochs == 5
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.lr == 2e-5
    assert cfg.lambda_address == 0.0
    assert cfg.lambda_negative == 1.0
    assert cfg.lambda_near_miss == 1.0
    assert cfg.near_miss_margin == 80.0
    assert cfg.lambda_address_hinge == 3.0
    assert cfg.address_hinge_margin == 0.2
    assert cfg.lambda_address_mse == 3.0
    assert cfg.lambda_soft_hard == 0.0
    assert cfg.soft_hard_temperature_start == 1.0
    assert cfg.soft_hard_temperature_end == 0.05
    assert cfg.soft_hard_straight_through is False
    assert cfg.soft_hard_focal_gamma == 0.0
    assert cfg.soft_hard_top_k_blocks == 0
    assert cfg.freeze_layers == 20
    assert cfg.fragmentation_target == 0.75
    assert cfg.separation_target == 0.80
    assert cfg.zero_fp_recall_target == 0.8056
    assert cfg.fp16 is True
    assert cfg.gradient_checkpointing is True


def test_config_product_gate_override():
    cfg = SnapTrainingConfig(zero_fp_recall_target=0.9, fragmentation_target=0.25)
    assert cfg.zero_fp_recall_target == 0.9
    assert cfg.fragmentation_target == 0.25


def test_config_override():
    cfg = SnapTrainingConfig(
        epochs=3,
        lr=1e-5,
        lambda_address=10.0,
        fp16=False,
        lambda_address_hinge=5.0,
        address_hinge_margin=0.4,
        lambda_address_mse=6.0,
    )
    assert cfg.epochs == 3
    assert cfg.lr == 1e-5
    assert cfg.lambda_address == 10.0
    assert cfg.fp16 is False
    assert cfg.lambda_address_hinge == 5.0
    assert cfg.address_hinge_margin == 0.4
    assert cfg.lambda_address_mse == 6.0


def test_config_near_miss_override():
    cfg = SnapTrainingConfig(lambda_near_miss=2.5, near_miss_margin=96.0)
    assert cfg.lambda_near_miss == 2.5
    assert cfg.near_miss_margin == 96.0


def test_config_soft_hard_override():
    cfg = SnapTrainingConfig(
        lambda_soft_hard=4.0,
        soft_hard_temperature_start=0.8,
        soft_hard_temperature_end=0.1,
        soft_hard_straight_through=True,
    )
    assert cfg.lambda_soft_hard == 4.0
    assert cfg.soft_hard_temperature_start == 0.8
    assert cfg.soft_hard_temperature_end == 0.1
    assert cfg.soft_hard_straight_through is True


def test_config_soft_hard_focal_override():
    cfg = SnapTrainingConfig(
        soft_hard_focal_gamma=2.0,
        soft_hard_top_k_blocks=8,
    )
    assert cfg.soft_hard_focal_gamma == 2.0
    assert cfg.soft_hard_top_k_blocks == 8


def test_best_checkpoint_prioritizes_zero_fp_recall_over_fragmentation():
    assert _is_better_snap_checkpoint(
        mean_fragmentation=0.0,
        hamming_gap=1.0,
        zero_fp_recall=0.92,
        best_fragmentation=0.4,
        best_hamming_gap=20.0,
        best_zero_fp_recall=0.78,
    )


def test_best_checkpoint_tie_breaks_on_hamming_gap_after_equal_zero_fp():
    assert _is_better_snap_checkpoint(
        mean_fragmentation=0.0,
        hamming_gap=4.0,
        zero_fp_recall=0.92,
        best_fragmentation=0.4,
        best_hamming_gap=2.0,
        best_zero_fp_recall=0.92,
    )


def test_best_checkpoint_uses_fragmentation_only_as_final_tie_breaker():
    assert _is_better_snap_checkpoint(
        mean_fragmentation=0.2,
        hamming_gap=2.0,
        zero_fp_recall=0.92,
        best_fragmentation=0.1,
        best_hamming_gap=2.0,
        best_zero_fp_recall=0.92,
    )


def test_best_checkpoint_rejects_worse_zero_fp_recall():
    assert not _is_better_snap_checkpoint(
        mean_fragmentation=1.0,
        hamming_gap=5.0,
        zero_fp_recall=0.7,
        best_fragmentation=0.0,
        best_hamming_gap=2.1,
        best_zero_fp_recall=0.8,
    )


# ---------------------------------------------------------------------------
# SnapTrainer._from_encoder tests
# ---------------------------------------------------------------------------

def test_from_encoder_sets_attributes():
    enc = FakeEncoder(D_MODEL)
    trainer = SnapTrainer._from_encoder(enc, val_clusters=VAL_CLUSTERS, d_model=D_MODEL)
    assert trainer.encoder is enc
    assert trainer.d_model == D_MODEL
    assert trainer.val_clusters == VAL_CLUSTERS
    assert trainer.base_model == "custom"


def test_from_encoder_empty_val_clusters_raises():
    with pytest.raises(Exception):
        SnapTrainer._from_encoder(FakeEncoder(D_MODEL), val_clusters={}, d_model=D_MODEL)


# ---------------------------------------------------------------------------
# eval_with_observatory tests
# ---------------------------------------------------------------------------

def test_eval_returns_required_keys(fake_trainer):
    result = fake_trainer.eval_with_observatory()
    assert "mean_fragmentation_score" in result
    assert "separation_score" in result
    assert "hamming_gap" in result
    assert "zero_fp_recall" in result
    assert "zero_fp_threshold" in result
    assert "paraphrase_hamming_mean" in result
    assert "near_miss_hamming_mean" in result
    assert "is_collapsed" in result
    assert "cluster_scores" in result
    assert "mean_hamming_per_step" in result
    assert "trajectory_continuity" in result
    assert "mean_inter_block_nmi" in result


def test_eval_cluster_scores_match_val_clusters(fake_trainer):
    result = fake_trainer.eval_with_observatory()
    assert set(result["cluster_scores"].keys()) == set(VAL_CLUSTERS.keys())


def test_eval_mean_frag_is_average_of_clusters(fake_trainer):
    result = fake_trainer.eval_with_observatory()
    scores = list(result["cluster_scores"].values())
    expected = round(float(np.mean(scores)), 4)
    assert abs(result["mean_fragmentation_score"] - expected) < 0.01


def test_eval_scores_are_in_0_1_range(fake_trainer):
    result = fake_trainer.eval_with_observatory()
    assert 0.0 <= result["mean_fragmentation_score"] <= 1.0
    assert 0.0 <= result["separation_score"] <= 1.0
    for v in result["cluster_scores"].values():
        assert 0.0 <= v <= 1.0
    assert -D_MODEL // 8 <= result["hamming_gap"] <= D_MODEL // 8
    assert 0.0 <= result["zero_fp_recall"] <= 1.0
    assert 0 <= result["zero_fp_threshold"] <= D_MODEL // 8
    assert 0.0 <= result["paraphrase_hamming_mean"] <= D_MODEL // 8
    assert 0.0 <= result["near_miss_hamming_mean"] <= D_MODEL // 8
    assert 0.0 <= result["trajectory_continuity"] <= 1.0
    assert result["mean_hamming_per_step"] >= 0.0
    assert result["mean_inter_block_nmi"] >= 0.0


def test_eval_is_deterministic(fake_trainer):
    r1 = fake_trainer.eval_with_observatory()
    r2 = fake_trainer.eval_with_observatory()
    assert r1["mean_fragmentation_score"] == r2["mean_fragmentation_score"]
    assert r1["cluster_scores"] == r2["cluster_scores"]


def test_eval_with_single_cluster():
    trainer = SnapTrainer._from_encoder(
        FakeEncoder(D_MODEL),
        val_clusters={"solo": ["Only one cluster", "Has some texts", "In here"]},
        d_model=D_MODEL,
    )
    result = trainer.eval_with_observatory()
    assert "solo" in result["cluster_scores"]
    assert result["mean_fragmentation_score"] == result["cluster_scores"]["solo"]


def test_eval_with_large_d_model():
    trainer = SnapTrainer._from_encoder(
        FakeEncoder(1024),
        val_clusters={"test": ["a", "b", "c", "d"]},
        d_model=1024,
    )
    result = trainer.eval_with_observatory()
    assert isinstance(result["mean_fragmentation_score"], float)


# ---------------------------------------------------------------------------
# RoutingTrainingExample construction
# ---------------------------------------------------------------------------

def test_tiny_examples_structure():
    examples = _tiny_examples(8)
    assert len(examples) == 8
    for ex in examples:
        assert isinstance(ex.query, str)
        assert isinstance(ex.positive, str)
        assert isinstance(ex.negatives, list)


# ---------------------------------------------------------------------------
# SnapTrainer.train() smoke tests (CPU, no AMP, 1 epoch, 8 examples)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_returns_snap_train_result(trainable_trainer, tmp_path):
    examples = _tiny_examples(8)
    config = SnapTrainingConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    result = trainable_trainer.train(examples, config)
    assert isinstance(result, SnapTrainResult)


@pytest.mark.slow
def test_train_result_fields(trainable_trainer, tmp_path):
    examples = _tiny_examples(8)
    config = SnapTrainingConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    result = trainable_trainer.train(examples, config)
    assert result.best_global_step >= 0
    assert 0.0 <= result.best_fragmentation_score <= 1.0
    assert 0.0 <= result.best_separation_score <= 1.0
    assert isinstance(result.reached_target, bool)
    assert result.d_model == D_MODEL
    assert result.examples_trained > 0
    assert len(result.epoch_metrics) == 1
    assert isinstance(result.obs_checkpoints, list)


@pytest.mark.slow
def test_train_epoch_metrics_structure(trainable_trainer, tmp_path):
    examples = _tiny_examples(8)
    config = SnapTrainingConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    result = trainable_trainer.train(examples, config)
    m = result.epoch_metrics[0]
    assert isinstance(m, SnapEpochMetrics)
    assert m.epoch == 1
    assert m.train_loss >= 0.0
    assert m.address_loss >= 0.0
    assert 0.0 <= m.mean_fragmentation_score <= 1.0
    assert 0.0 <= m.separation_score <= 1.0
    assert -D_MODEL // 8 <= m.hamming_gap <= D_MODEL // 8
    assert 0.0 <= m.zero_fp_recall <= 1.0
    assert 0 <= m.zero_fp_threshold <= D_MODEL // 8
    assert m.mean_hamming_per_step >= 0.0
    assert 0.0 <= m.trajectory_continuity <= 1.0


@pytest.mark.slow
def test_train_writes_summary_json(trainable_trainer, tmp_path):
    examples = _tiny_examples(8)
    config = SnapTrainingConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    trainable_trainer.train(examples, config)
    summary_path = tmp_path / "snap_training_summary.json"
    assert summary_path.exists()
    import json
    summary = json.loads(summary_path.read_text())
    assert summary["epochs_trained"] == 1
    assert "best_fragmentation_score" in summary
    assert "best_hamming_gap" in summary
    assert "best_zero_fp_recall" in summary
    assert summary["soft_hard_focal_gamma"] == 0.0
    assert summary["soft_hard_top_k_blocks"] == 0
    assert summary["product_gate"] == "zero_fp_recall"
    assert summary["zero_fp_recall_target"] == 0.8056
    assert summary["fragmentation_metric_role"] == "research_exact_snap"
    assert "epoch_metrics" in summary
    assert "obs_checkpoints" in summary
    assert len(summary["epoch_metrics"]) == 1
    assert "hamming_gap" in summary["epoch_metrics"][0]
    assert "zero_fp_recall" in summary["epoch_metrics"][0]
    assert "soft_hard_loss" in summary["epoch_metrics"][0]
    assert "target_cell_probability" in summary["epoch_metrics"][0]
    assert "soft_hard_temperature" in summary["epoch_metrics"][0]
    assert "soft_hard_loss_recent" in summary["obs_checkpoints"][0]
    assert "target_cell_probability_recent" in summary["obs_checkpoints"][0]
    assert "soft_hard_temperature" in summary["obs_checkpoints"][0]


@pytest.mark.slow
def test_train_scheduler_handles_partial_gradient_accumulation_step(trainable_trainer, tmp_path):
    examples = _tiny_examples(10)
    config = SnapTrainingConfig(
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=3,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    result = trainable_trainer.train(examples, config)
    assert result.best_global_step == 2


@pytest.mark.slow
def test_train_early_stop_when_target_reached(trainable_trainer, tmp_path):
    """The product gate is zero-FP recall + separation, not exact fragmentation."""
    examples = _tiny_examples(8)
    config = SnapTrainingConfig(
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=1,
        lr=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        freeze_layers=0,
        log_every_batches=0,
        output_dir=str(tmp_path),
        fragmentation_target=1.0,
        zero_fp_recall_target=0.0,
        separation_target=0.0,
        device="cpu",
    )
    result = trainable_trainer.train(examples, config)
    assert result.reached_target is True
    assert result.best_fragmentation_score < config.fragmentation_target
    assert len(result.epoch_metrics) <= 3  # stopped at or before epoch 3


@pytest.mark.slow
def test_train_empty_examples_raises(trainable_trainer):
    config = SnapTrainingConfig(epochs=1, fp16=False, gradient_checkpointing=False, device="cpu")
    with pytest.raises(ValueError, match="examples must not be empty"):
        trainable_trainer.train([], config)


# ---------------------------------------------------------------------------
# SnapTrainResult dataclass
# ---------------------------------------------------------------------------

def test_snap_train_result_fields():
    result = SnapTrainResult(
        epoch_metrics=[],
        obs_checkpoints=[],
        best_global_step=200,
        best_fragmentation_score=0.75,
        best_separation_score=0.9,
        reached_target=False,
        output_dir=None,
        base_model="test",
        d_model=384,
        examples_trained=100,
    )
    assert result.best_global_step == 200
    assert result.best_fragmentation_score == 0.75
    assert result.best_separation_score == 0.9
    assert result.reached_target is False


def test_snap_epoch_metrics_fields():
    m = SnapEpochMetrics(
        epoch=1,
        train_loss=0.5,
        address_loss=0.3,
        contrastive_loss=0.2,
        mean_fragmentation_score=0.6,
        separation_score=0.9,
        hamming_gap=12.0,
        zero_fp_recall=0.4,
        zero_fp_threshold=42,
        mean_hamming_per_step=12.0,
        trajectory_continuity=0.7,
        mean_inter_block_nmi=0.4,
        cluster_scores={"a": 0.6, "b": 0.7},
        elapsed_sec=30.0,
        is_best=True,
    )
    assert m.epoch == 1
    assert m.is_best is True
    assert m.separation_score == 0.9
    assert m.hamming_gap == 12.0
    assert m.zero_fp_recall == 0.4
    assert m.zero_fp_threshold == 42
    assert m.cluster_scores == {"a": 0.6, "b": 0.7}


def test_snap_obs_checkpoint_fields():
    ckpt = SnapObsCheckpoint(
        epoch=1,
        global_step=10,
        batch_in_epoch=2,
        examples_seen=32,
        mean_fragmentation_score=0.5,
        separation_score=0.8,
        hamming_gap=5.5,
        zero_fp_recall=0.25,
        zero_fp_threshold=30,
        cluster_scores={"a": 0.5},
        mean_hamming_per_step=20.0,
        trajectory_continuity=0.1,
        mean_inter_block_nmi=0.6,
        address_loss_recent=1.2,
        train_loss_recent=2.3,
    )
    assert ckpt.hamming_gap == 5.5
    assert ckpt.zero_fp_recall == 0.25
    assert ckpt.zero_fp_threshold == 30


def test_load_banking77_mocked():
    from unittest.mock import patch
    mock_data = [
        {"text": "query A1", "label": 1, "label_text": "intent_one"},
        {"text": "query A2", "label": 1, "label_text": "intent_one"},
        {"text": "query B1", "label": 2, "label_text": "intent_two"},
    ]
    with patch("datasets.load_dataset", return_value=mock_data):
        trainer = SnapTrainer._from_encoder(
            FakeEncoder(384), val_clusters={"dummy": ["dummy"]}, d_model=384
        )
        examples = trainer.load_banking77(limit=10)
        assert len(examples) == 3
        
        # Mapping checks: queries should map to their own intent's first text as positive
        assert examples[0].query == "query A1"
        assert examples[0].positive == "query A1"
        assert examples[1].query == "query A2"
        assert examples[1].positive == "query A1"  # "query A1" is canonical for label 1
        assert examples[2].query == "query B1"
        assert examples[2].positive == "query B1"
        
        # Seeded negative sampling checks (should be from other labels)
        assert all(neg in ["query B1"] for neg in examples[0].negatives)
        assert all(neg in ["query A1", "query A2"] for neg in examples[2].negatives)


def test_load_clinc150_mocked():
    from unittest.mock import patch
    mock_data = [
        {"text": "query X1", "intent": 10},
        {"text": "query X2", "intent": 10},
        {"text": "query Y1", "intent": 20},
    ]
    with patch("datasets.load_dataset", return_value=mock_data):
        trainer = SnapTrainer._from_encoder(
            FakeEncoder(384), val_clusters={"dummy": ["dummy"]}, d_model=384
        )
        examples = trainer.load_clinc150(limit=10)
        assert len(examples) == 3
        
        # Mapping checks
        assert examples[0].query == "query X1"
        assert examples[0].positive == "query X1"
        assert examples[1].query == "query X2"
        assert examples[1].positive == "query X1"  # "query X1" is canonical for intent 10
        assert examples[2].query == "query Y1"
        assert examples[2].positive == "query Y1"
        
        # Seeded negative sampling checks (should be from other intents)
        assert all(neg in ["query Y1"] for neg in examples[0].negatives)
        assert all(neg in ["query X1", "query X2"] for neg in examples[2].negatives)


def test_load_val_clusters():
    val_clusters = {
        "refund": ["refund A", "refund B"],
        "password": ["password A", "password B"],
    }
    trainer = SnapTrainer._from_encoder(
        FakeEncoder(384), val_clusters=val_clusters, d_model=384
    )
    examples = trainer.load_val_clusters()
    assert len(examples) == 4
    
    # Mapping checks: query should map to canonical (first item in cluster)
    # Alphabetically "password" comes first, then "refund"
    assert examples[0].query == "password A"
    assert examples[0].positive == "password A"
    assert examples[1].query == "password B"
    assert examples[1].positive == "password A"
    assert examples[2].query == "refund A"
    assert examples[2].positive == "refund A"
    assert examples[3].query == "refund B"
    assert examples[3].positive == "refund A"

    # Seeded negative sampling checks (should draw from refund for password, and vice versa)
    assert all(neg in ["refund A", "refund B"] for neg in examples[0].negatives)
    assert all(neg in ["password A", "password B"] for neg in examples[2].negatives)


# ---------------------------------------------------------------------------
# __init__.py export check
# ---------------------------------------------------------------------------

def test_public_imports():
    from latticememory import SnapTrainer, SnapTrainingConfig, SnapTrainResult, SnapEpochMetrics, SnapObsCheckpoint
    assert SnapTrainer is not None
    assert SnapTrainingConfig is not None
    assert SnapTrainResult is not None
    assert SnapEpochMetrics is not None
    assert SnapObsCheckpoint is not None
