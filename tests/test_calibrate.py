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
