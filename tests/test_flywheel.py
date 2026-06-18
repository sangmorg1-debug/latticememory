"""Tests for LatticeFlywheel — including detect_drift and should_finetune."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from latticememory.flywheel import LatticeFlywheel


# ── Key fixtures ────────────────────────────────────────────────────────────

# Three well-separated 128-byte fake E8 keys (all bytes identical within
# each cluster → Hamming distance between clusters = 128, far above threshold).
_KEY_A = ("01" * 128)
_KEY_B = ("80" * 128)
_KEY_C = ("ff" * 128)
# A near-variant of KEY_A (8 bytes differ → Hamming distance = 8 < threshold 25)
_KEY_A2 = ("01" * 120 + "02" * 8)

_NOW = time.time()
_WINDOW = 7 * 86400  # 7 days


def _recent(offset_hours: float = 24) -> float:
    """Timestamp within the 'recent' half of the drift window."""
    return _NOW - offset_hours * 3600


def _previous(offset_hours: float = 24) -> float:
    """Timestamp within the 'previous' half of the drift window."""
    return _NOW - _WINDOW - offset_hours * 3600


# ── log_miss / load_log ─────────────────────────────────────────────────────

def test_log_miss_and_load(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log)
    fw.log_miss("What is the capital of France?", e8_key_hex=_KEY_A)
    fw.log_miss("How do I reset my password?", e8_key_hex=_KEY_B)

    records = fw.load_log()
    assert len(records) == 2
    assert records[0].question == "What is the capital of France?"
    assert records[0].e8_key_hex == _KEY_A


def test_clear_log(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log)
    fw.log_miss("foo", e8_key_hex=_KEY_A)
    fw.clear_log()
    assert fw.load_log() == []


# ── cluster_gaps ─────────────────────────────────────────────────────────────

def test_cluster_gaps_groups_similar_keys(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # 5 records with KEY_A and 1 variant (KEY_A2 differs by 8 bytes < threshold)
    for i in range(4):
        fw.log_miss(f"question A-{i}", e8_key_hex=_KEY_A)
    fw.log_miss("question A-variant", e8_key_hex=_KEY_A2)

    # 4 records with KEY_B (separate cluster)
    for i in range(4):
        fw.log_miss(f"question B-{i}", e8_key_hex=_KEY_B)

    clusters = fw.cluster_gaps(min_cluster_size=3)
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [4, 5]  # cluster A has 5 (incl. variant), cluster B has 4


def test_cluster_gaps_empty_log(tmp_path):
    fw = LatticeFlywheel(tmp_path / "empty.jsonl", cluster_threshold=25)
    assert fw.cluster_gaps() == []


def test_top_gaps_limits_results(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)
    for key in [_KEY_A, _KEY_B, _KEY_C]:
        for i in range(5):
            fw.log_miss(f"q {key[:2]}-{i}", e8_key_hex=key)
    gaps = fw.top_gaps(n=2, min_cluster_size=3)
    assert len(gaps) == 2


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log)
    fw.log_miss("alpha", e8_key_hex=_KEY_A)
    fw.log_miss("alpha", e8_key_hex=_KEY_A)  # duplicate question
    fw.log_miss("beta", e8_key_hex=_KEY_B)

    s = fw.stats()
    assert s["total_misses"] == 3
    assert s["unique_questions"] == 2
    assert s["log_exists"] is True


# ── detect_drift ─────────────────────────────────────────────────────────────

def _log_drifting_cluster(fw: LatticeFlywheel, key: str, n_recent: int, n_previous: int) -> None:
    """Helper: log n_recent records in the recent window and n_previous in the old window."""
    for i in range(n_recent):
        fw.log_miss(f"recent-{key[:4]}-{i}", e8_key_hex=key)
        # patch timestamp to 'recent'
        records = fw.load_log()
        records[-1].timestamp = _recent(offset_hours=i + 1)
        # rewrite log with updated timestamp
        fw.log_path.write_text(
            "\n".join(json.dumps({
                "question": r.question,
                "e8_key_hex": r.e8_key_hex,
                "timestamp": r.timestamp,
                "nearest_cache_prompt": r.nearest_cache_prompt,
                "nearest_cache_distance": r.nearest_cache_distance,
                "metadata": r.metadata,
            }) for r in records) + "\n",
            encoding="utf-8",
        )

    for i in range(n_previous):
        fw.log_miss(f"prev-{key[:4]}-{i}", e8_key_hex=key)
        records = fw.load_log()
        records[-1].timestamp = _previous(offset_hours=i + 1)
        fw.log_path.write_text(
            "\n".join(json.dumps({
                "question": r.question,
                "e8_key_hex": r.e8_key_hex,
                "timestamp": r.timestamp,
                "nearest_cache_prompt": r.nearest_cache_prompt,
                "nearest_cache_distance": r.nearest_cache_distance,
                "metadata": r.metadata,
            }) for r in records) + "\n",
            encoding="utf-8",
        )


def test_detect_drift_identifies_growing_clusters(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # Cluster A: 7 recent vs 1 previous (delta = 6 >= 5)
    _log_drifting_cluster(fw, _KEY_A, n_recent=7, n_previous=1)
    # Cluster B: 6 recent vs 0 previous (delta = 6 >= 5)
    _log_drifting_cluster(fw, _KEY_B, n_recent=6, n_previous=0)
    # Cluster C: stable (3 recent, 3 previous, delta = 0 < 5)
    _log_drifting_cluster(fw, _KEY_C, n_recent=3, n_previous=3)

    drifting = fw.detect_drift(
        window_seconds=_WINDOW,
        min_delta=5,
        min_cluster_size=3,
    )
    assert len(drifting) == 2
    deltas = sorted(r["delta"] for r in drifting)
    assert all(d >= 5 for d in deltas)


def test_detect_drift_empty_log(tmp_path):
    fw = LatticeFlywheel(tmp_path / "empty.jsonl", cluster_threshold=25)
    assert fw.detect_drift() == []


def test_detect_drift_no_drift(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # Both windows equal → delta = 0
    _log_drifting_cluster(fw, _KEY_A, n_recent=4, n_previous=4)

    drifting = fw.detect_drift(window_seconds=_WINDOW, min_delta=5, min_cluster_size=3)
    assert drifting == []


# ── should_finetune ──────────────────────────────────────────────────────────

def test_should_finetune_true_when_enough_drift(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # 3 drifting clusters: each has 7 recent vs 1 previous (delta=6 >= 5)
    for key in [_KEY_A, _KEY_B, _KEY_C]:
        _log_drifting_cluster(fw, key, n_recent=7, n_previous=1)

    assert fw.should_finetune(
        min_drifting_clusters=3,
        min_delta=5,
        window_seconds=_WINDOW,
        min_cluster_size=3,
    ) is True


def test_should_finetune_false_when_too_few_clusters(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # Only 2 drifting clusters; threshold is 3
    for key in [_KEY_A, _KEY_B]:
        _log_drifting_cluster(fw, key, n_recent=7, n_previous=1)

    assert fw.should_finetune(
        min_drifting_clusters=3,
        min_delta=5,
        window_seconds=_WINDOW,
        min_cluster_size=3,
    ) is False


def test_should_finetune_false_on_empty_log(tmp_path):
    fw = LatticeFlywheel(tmp_path / "empty.jsonl")
    assert fw.should_finetune() is False


def test_should_finetune_respects_min_delta(tmp_path):
    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)

    # 3 clusters each with delta = 3 (< min_delta=5)
    for key in [_KEY_A, _KEY_B, _KEY_C]:
        _log_drifting_cluster(fw, key, n_recent=4, n_previous=1)  # delta=3

    assert fw.should_finetune(
        min_drifting_clusters=1,
        min_delta=5,
        window_seconds=_WINDOW,
        min_cluster_size=3,
    ) is False

    # Same data but lower bar: min_delta=2 → should trigger
    assert fw.should_finetune(
        min_drifting_clusters=1,
        min_delta=2,
        window_seconds=_WINDOW,
        min_cluster_size=3,
    ) is True


# ── drift CLI command ─────────────────────────────────────────────────────────

def test_cmd_drift_no_drift(tmp_path, capsys):
    from latticememory.cli import cmd_drift

    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)
    _log_drifting_cluster(fw, _KEY_A, n_recent=2, n_previous=2)

    args = argparse.Namespace(
        log=str(log),
        window=_WINDOW,
        min_delta=5,
        min_cluster_size=3,
        min_clusters=3,
        threshold=25,
        export=None,
    )
    rc = cmd_drift(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No drifting intents" in out


def test_cmd_drift_with_drift(tmp_path, capsys):
    from latticememory.cli import cmd_drift

    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)
    for key in [_KEY_A, _KEY_B, _KEY_C]:
        _log_drifting_cluster(fw, key, n_recent=7, n_previous=1)

    args = argparse.Namespace(
        log=str(log),
        window=_WINDOW,
        min_delta=5,
        min_cluster_size=3,
        min_clusters=3,
        threshold=25,
        export=None,
    )
    rc = cmd_drift(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Drifting intents" in out
    assert "YES" in out


def test_cmd_drift_export(tmp_path, capsys):
    from latticememory.cli import cmd_drift

    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)
    for key in [_KEY_A, _KEY_B, _KEY_C]:
        _log_drifting_cluster(fw, key, n_recent=7, n_previous=1)

    export_path = tmp_path / "drift.json"
    args = argparse.Namespace(
        log=str(log),
        window=_WINDOW,
        min_delta=5,
        min_cluster_size=3,
        min_clusters=3,
        threshold=25,
        export=str(export_path),
    )
    cmd_drift(args)
    assert export_path.exists()
    data = json.loads(export_path.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1


def test_cmd_drift_missing_log(tmp_path, capsys):
    from latticememory.cli import cmd_drift

    args = argparse.Namespace(
        log=str(tmp_path / "nonexistent.jsonl"),
        window=_WINDOW,
        min_delta=5,
        min_cluster_size=3,
        min_clusters=3,
        threshold=25,
        export=None,
    )
    rc = cmd_drift(args)
    assert rc == 1


# ── review CLI command ─────────────────────────────────────────────────────

class _FakeSentenceTransformer:
    """Deterministic MD5-seeded fake encoder — no real model downloads."""

    def __init__(self, *_args, **_kwargs):
        self.d_model = 384

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, sentences, **kwargs):
        import hashlib

        import numpy as np

        vecs = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


def test_cmd_review_export_writes_review_queue(tmp_path, capsys):
    from latticememory.cli import cmd_review

    log = tmp_path / "misses.jsonl"
    fw = LatticeFlywheel(log, cluster_threshold=25)
    for i in range(4):
        fw.log_miss(f"how do I reset my password variant {i}", e8_key_hex=_KEY_A)

    output = tmp_path / "review_queue.json"
    args = argparse.Namespace(
        review_command="export",
        log=str(log),
        output=str(output),
        top=10,
        min_size=3,
        threshold=25,
    )
    rc = cmd_review(args)
    assert rc == 0
    data = json.loads(output.read_text())
    assert len(data) == 1
    assert data[0]["size"] == 4
    assert data[0]["answer"] is None
    out = capsys.readouterr().out
    assert "Review queue exported" in out


def test_cmd_review_export_missing_log(tmp_path, capsys):
    from latticememory.cli import cmd_review

    args = argparse.Namespace(
        review_command="export",
        log=str(tmp_path / "nonexistent.jsonl"),
        output=str(tmp_path / "review_queue.json"),
        top=10,
        min_size=3,
        threshold=25,
    )
    rc = cmd_review(args)
    assert rc == 1


def test_cmd_review_import_loads_answered_items_into_cache(tmp_path, capsys, monkeypatch):
    from latticememory.cli import cmd_review

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)

    log = tmp_path / "misses.jsonl"
    queue_path = tmp_path / "review_queue.json"
    queue_path.write_text(json.dumps([
        {
            "cluster_id": 0,
            "representative_question": "how do I reset my password",
            "sample_questions": ["how do I reset my password", "reset my password please"],
            "size": 2,
            "answer": "Go to Settings > Security > Reset Password.",
            "intent_id": "password_reset",
            "centroid_key_hex": _KEY_A,
        },
        {
            "cluster_id": 1,
            "representative_question": "unanswered question",
            "sample_questions": ["unanswered question"],
            "size": 1,
            "answer": None,
            "intent_id": None,
            "centroid_key_hex": _KEY_B,
        },
    ]), encoding="utf-8")

    cache_path = tmp_path / "cache.db"
    args = argparse.Namespace(
        review_command="import",
        input=str(queue_path),
        log=str(log),
        cache=str(cache_path),
        encoder="fake-model",
    )
    rc = cmd_review(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Imported 1" in out

    from latticememory.memory import RFSnapLatticeMemory
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory

    lm = RFSnapLatticeMemory(d_model=384, sqlite_path=str(cache_path))
    runtime = RFSnapTextMemory(encoder=None, d_model=384, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)
    assert cache.size >= 1


def test_cmd_review_import_missing_input(tmp_path, capsys, monkeypatch):
    from latticememory.cli import cmd_review

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)

    args = argparse.Namespace(
        review_command="import",
        input=str(tmp_path / "nonexistent.json"),
        log=str(tmp_path / "misses.jsonl"),
        cache=str(tmp_path / "cache.db"),
        encoder="fake-model",
    )
    rc = cmd_review(args)
    assert rc == 1


# ── federated CLI command ────────────────────────────────────────────────────

def test_cmd_federated_aggregates_multiple_logs(tmp_path, capsys):
    from latticememory.cli import cmd_federated

    log1 = tmp_path / "node1.jsonl"
    log2 = tmp_path / "node2.jsonl"
    fw1 = LatticeFlywheel(log1)
    fw2 = LatticeFlywheel(log2)
    fw1.log_miss("alpha", e8_key_hex=_KEY_A)
    fw1.log_miss("alpha again", e8_key_hex=_KEY_A)
    fw2.log_miss("beta", e8_key_hex=_KEY_B)

    args = argparse.Namespace(logs=[str(log1), str(log2)], export=None)
    rc = cmd_federated(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert str(log1) in out
    assert str(log2) in out


def test_cmd_federated_exports_json(tmp_path, capsys):
    from latticememory.cli import cmd_federated

    log1 = tmp_path / "node1.jsonl"
    fw1 = LatticeFlywheel(log1)
    fw1.log_miss("alpha", e8_key_hex=_KEY_A)

    export_path = tmp_path / "federated.json"
    args = argparse.Namespace(logs=[str(log1)], export=str(export_path))
    rc = cmd_federated(args)
    assert rc == 0
    data = json.loads(export_path.read_text())
    assert isinstance(data, dict)


def test_cmd_federated_missing_log(tmp_path, capsys):
    from latticememory.cli import cmd_federated

    args = argparse.Namespace(logs=[str(tmp_path / "nonexistent.jsonl")], export=None)
    rc = cmd_federated(args)
    assert rc == 1
