from __future__ import annotations

import json
from unittest.mock import patch

from examples.proxy_replay_demo import run_replay_demo
from latticememory.proof_pack import _InMemoryRedis, build_support_dataset


def test_proxy_replay_demo_reports_validated_cache_behavior(tmp_path):
    dataset = build_support_dataset(
        seed_count=8,
        calibration_count=8,
        evaluation_count=24,
        adversarial_count=8,
    )
    dataset_path = tmp_path / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for rows in dataset.values():
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    redis_client = _InMemoryRedis()
    with patch("redis.from_url", return_value=redis_client):
        summary = run_replay_demo(
            dataset_path=dataset_path,
            redis_url="redis://localhost:6379/15",
            redis_namespace="demo-test",
            output_json=tmp_path / "summary.json",
            output_md=tmp_path / "summary.md",
        )

    assert summary["total_requests"] == 32
    assert summary["false_positive_rate"] == 0.0
    assert summary["adversarial_false_positive_rate"] == 0.0
    assert summary["upstream_call_rate"] > 0.0
    assert summary["redis_url"] == "redis://localhost:6379/15"
    assert (tmp_path / "summary.json").exists()
    assert "LatticeMemory Proxy Replay Demo" in (tmp_path / "summary.md").read_text(encoding="utf-8")
