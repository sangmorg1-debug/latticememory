"""Tests for the `lattice calibrate` CLI subcommand — no real model downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from latticememory.hamming_router import HammingRouter


class HashEncoder:
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, **kwargs):
        vecs = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


@pytest.fixture(autouse=True)
def fake_from_model(monkeypatch):
    def _from_model(cls, model_name_or_path, threshold=111):
        return HammingRouter(encoder=HashEncoder(384), d_model=384, threshold=threshold)

    monkeypatch.setattr(HammingRouter, "from_model", classmethod(_from_model))


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text("\n".join(f"{a}|||{b}" for a, b in pairs), encoding="utf-8")


def test_cmd_calibrate_prints_threshold_table_and_recommendation(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    args = argparse.Namespace(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "HAMMING DISTANCE STATISTICS" in out
    assert "Paraphrase pairs: 20" in out
    assert "Near-miss pairs: 20" in out
    assert "THRESHOLD" in out
    assert "RECOMMEND" in out


def test_cmd_calibrate_exports_json(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    export_path = tmp_path / "calibration.json"
    args = argparse.Namespace(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=str(export_path),
    )
    cmd_calibrate(args)

    data = json.loads(export_path.read_text())
    assert "calibration" in data
    assert "threshold" in data["calibration"]
    assert "gap_stats" in data


def test_cmd_calibrate_export_is_a_valid_precalibrated_artifact(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate
    from latticememory.hamming_router import validate_precalibrated_artifact_schema

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    export_path = tmp_path / "calibration.json"
    args = argparse.Namespace(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=str(export_path),
    )
    cmd_calibrate(args)

    data = json.loads(export_path.read_text())
    # Must not raise -- this is the exact check latticememory.proxy.LatticeLLMProxy
    # runs before trusting a calibration file enough to set a live threshold from it.
    validate_precalibrated_artifact_schema(data)
    assert data["model"] == "fake-model"
    assert data["d_model"] == 384


def test_cmd_calibrate_export_loads_into_a_live_proxy(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate
    from latticememory.proxy import LatticeLLMProxy

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    export_path = tmp_path / "calibration.json"
    args = argparse.Namespace(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=str(export_path),
    )
    cmd_calibrate(args)

    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        encoder_model="fake-model",
        encoder=HashEncoder(384),
        d_model=384,
        enable_hamming_router=True,
        calibration_data_path=str(export_path),
        require_calibration=True,
        fp_budget=0.0,
    )

    assert proxy.hamming_router_calibrated is True
    assert proxy.hamming_router_n_paraphrase_pairs == 20
    assert proxy.hamming_router_n_near_miss_pairs == 20
    assert proxy.cache._hamming_threshold == proxy.cache._hamming_router.threshold


def test_cmd_calibrate_missing_paraphrases_file(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(near_misses, [("a", "b")])

    args = argparse.Namespace(
        paraphrases=str(tmp_path / "nonexistent.txt"),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 1


def test_cmd_calibrate_empty_pairs_file_errors(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    paraphrases.write_text("", encoding="utf-8")
    _write_pairs(near_misses, [("a", "b")])

    args = argparse.Namespace(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 1


def _calibrate_args(tmp_path, paraphrases, near_misses, **overrides):
    defaults = dict(
        paraphrases=str(paraphrases),
        near_misses=str(near_misses),
        encoder="fake-model",
        fp_budget=0.0,
        export=None,
        metric="hamming",
        holdout_paraphrases=None,
        holdout_near_misses=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cmd_calibrate_cosine_metric_prints_cosine_statistics(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    args = _calibrate_args(tmp_path, paraphrases, near_misses, metric="cosine")
    rc = cmd_calibrate(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "COSINE SIMILARITY STATISTICS" in out
    assert "HAMMING DISTANCE STATISTICS" not in out


def test_cmd_calibrate_cosine_metric_exports_distinct_artifact_type(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    export_path = tmp_path / "cosine_calibration.json"
    args = _calibrate_args(tmp_path, paraphrases, near_misses, metric="cosine", export=str(export_path))
    cmd_calibrate(args)

    data = json.loads(export_path.read_text())
    assert data["artifact_type"] == "latticememory_hamming_cosine_calibration"
    assert data["metric"] == "cosine"
    # Not the live-loadable Hamming schema -- there is no cosine-loading path yet.
    from latticememory.hamming_router import validate_precalibrated_artifact_schema

    with pytest.raises(ValueError):
        validate_precalibrated_artifact_schema(data)


def test_cmd_calibrate_without_holdout_prints_in_sample_warning(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    args = _calibrate_args(tmp_path, paraphrases, near_misses)
    rc = cmd_calibrate(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "in-sample" in out.lower()
    assert "held-out" in out.lower() or "--holdout" in out


def test_cmd_calibrate_with_holdout_reports_held_out_evaluation(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    holdout_paraphrases = tmp_path / "holdout_paraphrases.txt"
    holdout_near_misses = tmp_path / "holdout_near_misses.txt"
    _write_pairs(holdout_paraphrases, [(f"held-out question {i}", f"held-out question {i} restated") for i in range(10)])
    _write_pairs(holdout_near_misses, [(f"held-out topic {i} alpha", f"held-out topic {i} beta") for i in range(10)])

    export_path = tmp_path / "calibration.json"
    args = _calibrate_args(
        tmp_path,
        paraphrases,
        near_misses,
        holdout_paraphrases=str(holdout_paraphrases),
        holdout_near_misses=str(holdout_near_misses),
        export=str(export_path),
    )
    rc = cmd_calibrate(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "HELD-OUT EVALUATION" in out
    assert "WARNING" not in out or "in-sample" not in out.lower()

    data = json.loads(export_path.read_text())
    assert "held_out_evaluation" in data
    assert "false_accepts" in data["held_out_evaluation"]
    assert "false_rejects" in data["held_out_evaluation"]
    assert data["held_out_evaluation"]["n_paraphrase_pairs"] == 10
    assert data["held_out_evaluation"]["n_near_miss_pairs"] == 10


def test_cmd_calibrate_holdout_requires_both_files_together(tmp_path, capsys):
    from latticememory.cli import cmd_calibrate

    paraphrases = tmp_path / "paraphrases.txt"
    near_misses = tmp_path / "near_misses.txt"
    _write_pairs(paraphrases, [(f"question {i}", f"question {i} restated") for i in range(20)])
    _write_pairs(near_misses, [(f"topic {i} alpha", f"topic {i} beta") for i in range(20)])

    holdout_paraphrases = tmp_path / "holdout_paraphrases.txt"
    _write_pairs(holdout_paraphrases, [("held-out question", "held-out question restated")])

    args = _calibrate_args(tmp_path, paraphrases, near_misses, holdout_paraphrases=str(holdout_paraphrases))
    rc = cmd_calibrate(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "holdout" in out.lower()
