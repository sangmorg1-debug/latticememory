from __future__ import annotations

import json
from unittest.mock import patch

from latticememory.proof_pack import (
    build_support_dataset_from_qa_records,
    build_support_dataset,
    load_support_dataset_jsonl,
    run_exact_string_baseline,
    run_proxy_pq_redis_flywheel_proof_pack,
)


def test_support_dataset_contains_required_splits_and_row_fields():
    dataset = build_support_dataset(
        seed_count=6,
        calibration_count=8,
        evaluation_count=18,
        adversarial_count=6,
    )

    assert set(dataset) == {"cache_seed", "calibration", "evaluation", "adversarial"}
    assert len(dataset["cache_seed"]) == 6
    assert len(dataset["calibration"]) == 8
    assert len(dataset["evaluation"]) == 18
    assert len(dataset["adversarial"]) == 6

    row = dataset["evaluation"][0]
    assert {
        "id",
        "intent_id",
        "prompt",
        "canonical_answer",
        "expected_cache_id",
        "is_repeat",
        "is_paraphrase",
        "is_adversarial",
    }.issubset(row)

    assert any(r["is_repeat"] for r in dataset["evaluation"])
    assert any(r["is_paraphrase"] for r in dataset["evaluation"])
    assert all(r["is_adversarial"] for r in dataset["adversarial"])


def test_exact_string_baseline_reports_cache_and_safety_metrics():
    dataset = build_support_dataset(
        seed_count=4,
        calibration_count=4,
        evaluation_count=12,
        adversarial_count=4,
    )

    row = run_exact_string_baseline(dataset)

    assert row["run_id"] == "exact_string"
    assert row["total_requests"] == 16
    assert 0.0 < row["exact_hit_rate"] < 1.0
    assert row["approximate_hit_rate"] == 0.0
    assert row["false_positive_rate"] == 0.0
    assert row["adversarial_false_positive_rate"] == 0.0
    assert row["upstream_call_rate"] > 0.0
    assert row["estimated_cost_saved_usd"] > 0.0
    assert row["status"] == "ok"


def test_proxy_pq_redis_flywheel_proof_pack_writes_artifacts(tmp_path):
    summary = run_proxy_pq_redis_flywheel_proof_pack(
        tmp_path,
        seed_count=8,
        calibration_count=8,
        evaluation_count=24,
        adversarial_count=8,
    )

    summary_path = tmp_path / "proof_pack_summary.json"
    report_path = tmp_path / "proof_pack_report.md"
    review_path = tmp_path / "flywheel_review_queue.json"
    import_path = tmp_path / "flywheel_review_import_result.json"

    assert summary_path.exists()
    assert report_path.exists()
    assert review_path.exists()
    assert import_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    run_ids = {row["run_id"] for row in loaded["runs"]}
    assert {
        "exact_string",
        "dense_cosine",
        "lattice_pq_local",
        "lattice_pq_validated_cosine",
        "lattice_pq_redis",
    }.issubset(run_ids)

    exact_row = next(row for row in loaded["runs"] if row["run_id"] == "exact_string")
    raw_pq_row = next(row for row in loaded["runs"] if row["run_id"] == "lattice_pq_local")
    validated_row = next(row for row in loaded["runs"] if row["run_id"] == "lattice_pq_validated_cosine")
    assert validated_row["status"] == "ok"
    assert validated_row["validation_gate"] == "cosine"
    assert validated_row["hit_rate"] >= exact_row["hit_rate"]
    assert validated_row["adversarial_false_positive_rate"] <= raw_pq_row["adversarial_false_positive_rate"]

    redis_row = next(row for row in loaded["runs"] if row["run_id"] == "lattice_pq_redis")
    assert redis_row["status"] == "ok"
    assert redis_row["cache_entries"] >= 8
    assert redis_row["redis_memory_mb"] > 0.0
    assert redis_row["flywheel_miss_clusters"] >= 1
    assert redis_row["reviewed_answers_loaded"] >= 1

    assert summary["artifact_dir"] == str(tmp_path)
    assert "LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack" in report_path.read_text(encoding="utf-8")
    policy_report = tmp_path / "operating_policy_report.md"
    assert policy_report.exists()
    policy_text = policy_report.read_text(encoding="utf-8")
    assert "conservative_zero_fp" in policy_text
    assert "balanced_validated_pq" in policy_text
    assert "aggressive_raw_pq" in policy_text


def test_external_support_dataset_jsonl_round_trips_required_splits(tmp_path):
    dataset = build_support_dataset(
        seed_count=8,
        calibration_count=8,
        evaluation_count=16,
        adversarial_count=8,
    )
    path = tmp_path / "external_support_dataset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rows in dataset.values():
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    loaded = load_support_dataset_jsonl(path)

    assert {split: len(rows) for split, rows in loaded.items()} == {
        "cache_seed": 8,
        "calibration": 8,
        "evaluation": 16,
        "adversarial": 8,
    }
    assert loaded["evaluation"][0]["source"] == "external_jsonl"


def test_third_party_qa_records_build_public_support_splits():
    records = []
    for intent_idx, intent in enumerate(("cancel_order", "track_refund", "payment_issue")):
        for row_idx in range(8):
            records.append(
                {
                    "instruction": f"third party question {row_idx} for {intent}",
                    "response": f"canonical support response for {intent}",
                    "intent": intent,
                    "category": "ORDER" if intent == "cancel_order" else "PAYMENT",
                    "flags": "synthetic-test",
                }
            )

    dataset = build_support_dataset_from_qa_records(
        records,
        seed_count=3,
        calibration_count=6,
        evaluation_count=9,
        adversarial_count=3,
        source_name="bitext_fixture",
    )

    assert {split: len(rows) for split, rows in dataset.items()} == {
        "cache_seed": 3,
        "calibration": 6,
        "evaluation": 9,
        "adversarial": 3,
    }
    assert all(row["source"] == "bitext_fixture" for rows in dataset.values() for row in rows)
    assert any(row["is_repeat"] for row in dataset["evaluation"])
    assert any(row["is_paraphrase"] for row in dataset["evaluation"])
    assert all(row["is_adversarial"] for row in dataset["adversarial"])
    assert all(row["source_category"] for rows in dataset.values() for row in rows)


def test_proof_pack_reports_real_redis_shared_cache_when_reachable(tmp_path):
    from latticememory.proof_pack import _InMemoryRedis

    redis_client = _InMemoryRedis()
    with patch("redis.from_url", return_value=redis_client):
        summary = run_proxy_pq_redis_flywheel_proof_pack(
            tmp_path,
            seed_count=8,
            calibration_count=8,
            evaluation_count=24,
            adversarial_count=8,
            redis_url="redis://localhost:6379/15",
        )

    redis_row = next(row for row in summary["runs"] if row["run_id"] == "lattice_pq_redis_real")
    assert redis_row["status"] == "ok"
    assert redis_row["redis_backend"] == "real"
    assert redis_row["redis_persistence_verified"] is True
    assert redis_row["multi_proxy_shared_cache_verified"] is True
    assert redis_row["redis_memory_mb"] > 0.0


def test_proof_pack_reports_validated_real_redis_pq_when_reachable(tmp_path):
    from latticememory.proof_pack import _InMemoryRedis

    redis_client = _InMemoryRedis()
    with patch("redis.from_url", return_value=redis_client):
        summary = run_proxy_pq_redis_flywheel_proof_pack(
            tmp_path,
            seed_count=8,
            calibration_count=8,
            evaluation_count=24,
            adversarial_count=8,
            redis_url="redis://localhost:6379/15",
            redis_namespace="validated-proof",
        )

    raw_row = next(row for row in summary["runs"] if row["run_id"] == "lattice_pq_redis_real")
    validated_row = next(
        row for row in summary["runs"] if row["run_id"] == "lattice_pq_redis_validated_cosine"
    )
    assert validated_row["status"] == "ok"
    assert validated_row["validation_gate"] == "cosine"
    assert validated_row["redis_backend"] == "real"
    assert validated_row["redis_persistence_verified"] is True
    assert validated_row["multi_proxy_shared_cache_verified"] is True
    assert validated_row["redis_memory_mb"] > 0.0
    assert validated_row["adversarial_false_positive_rate"] <= raw_row["adversarial_false_positive_rate"]
    assert (tmp_path / "lattice_pq_redis_validated_cosine.json").exists()

    policy_text = (tmp_path / "operating_policy_report.md").read_text(encoding="utf-8")
    assert "balanced_validated_pq | lattice_pq_redis_validated_cosine" in policy_text


def test_proof_pack_writes_progress_profile_rows(tmp_path):
    from latticememory.proof_pack import _InMemoryRedis

    redis_client = _InMemoryRedis()
    progress_path = tmp_path / "progress.jsonl"
    with patch("redis.from_url", return_value=redis_client):
        summary = run_proxy_pq_redis_flywheel_proof_pack(
            tmp_path,
            seed_count=8,
            calibration_count=8,
            evaluation_count=24,
            adversarial_count=8,
            redis_url="redis://localhost:6379/15",
            progress_path=progress_path,
        )

    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [row["run_id"] for row in summary["runs"]]
    assert [row["run_id"] for row in progress_rows[::2]] == run_ids
    assert [row["event"] for row in progress_rows[::2]] == ["started"] * len(run_ids)
    assert [row["run_id"] for row in progress_rows[1::2]] == run_ids
    assert [row["event"] for row in progress_rows[1::2]] == ["finished"] * len(run_ids)
    assert all(row["status"] == "running" for row in progress_rows[::2])
    assert all(row["status"] in {"ok", "skipped"} for row in progress_rows[1::2])
    assert all(row["elapsed_s"] >= 0.0 for row in progress_rows[1::2])


def test_real_redis_persistence_verification_allows_duplicate_seed_cache_keys(tmp_path):
    from latticememory.proof_pack import _InMemoryRedis

    records = []
    for intent in ("cancel_order", "track_refund", "payment_issue"):
        for row_idx in range(8):
            records.append(
                {
                    "instruction": f"third party question {row_idx} for {intent}",
                    "response": f"canonical support response for {intent}",
                    "intent": intent,
                    "category": "SUPPORT",
                    "flags": "synthetic-test",
                }
            )
    dataset = build_support_dataset_from_qa_records(
        records,
        seed_count=9,
        calibration_count=6,
        evaluation_count=18,
        adversarial_count=6,
        source_name="duplicate_seed_fixture",
    )
    dataset_path = tmp_path / "duplicate_seed_fixture.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for rows in dataset.values():
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    redis_client = _InMemoryRedis()
    with patch("redis.from_url", return_value=redis_client):
        summary = run_proxy_pq_redis_flywheel_proof_pack(
            tmp_path,
            dataset_path=dataset_path,
            redis_url="redis://localhost:6379/15",
            redis_namespace="duplicate-seed-proof",
        )

    raw_row = next(row for row in summary["runs"] if row["run_id"] == "lattice_pq_redis_real")
    validated_row = next(
        row for row in summary["runs"] if row["run_id"] == "lattice_pq_redis_validated_cosine"
    )
    assert raw_row["cache_entries"] < 9
    assert raw_row["multi_proxy_shared_cache_verified"] is True
    assert raw_row["redis_persistence_verified"] is True
    assert validated_row["cache_entries"] < 9
    assert validated_row["multi_proxy_shared_cache_verified"] is True
    assert validated_row["redis_persistence_verified"] is True


def test_redis_memory_mb_counts_lattice_redis_store_client_namespace():
    from latticememory.proof_pack import _InMemoryRedis, _redis_memory_mb
    from latticememory.redis_store import LatticeRedisStore

    redis_client = _InMemoryRedis()
    redis_client.set("proof-pack:item", "cached-value")
    redis_client.set("other:item", "not-counted")
    with patch("redis.from_url", return_value=redis_client):
        store = LatticeRedisStore(redis_url="redis://localhost:6379/15", namespace="proof-pack")

    assert _redis_memory_mb(store, namespace="proof-pack") > 0.0


def test_proof_pack_writes_skipped_baselines_and_public_claim_card(tmp_path):
    summary = run_proxy_pq_redis_flywheel_proof_pack(
        tmp_path,
        seed_count=8,
        calibration_count=8,
        evaluation_count=24,
        adversarial_count=8,
        include_competitor_baselines=True,
    )

    run_ids = {row["run_id"] for row in summary["runs"]}
    assert {"redisvl_direct", "gptcache_direct", "upstash_semantic_cache"}.issubset(run_ids)
    skipped = [row for row in summary["runs"] if row["status"] == "skipped"]
    assert any("not installed" in row["skip_reason"] or "credentials" in row["skip_reason"] for row in skipped)

    claim_card = tmp_path / "public_claim_card.md"
    assert claim_card.exists()
    text = claim_card.read_text(encoding="utf-8")
    assert "reduces repeated/paraphrased upstream calls" in text
    assert "does not replace general-purpose vector databases" in text
    assert "Safest Zero-FP Measured Row" in text
    assert "Highest-Hit Measured Row" in text
    assert "Target Product Policy Row" in text
