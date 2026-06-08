from __future__ import annotations

import json


def test_compression_benchmark_reports_actual_and_theoretical_bytes(tmp_path):
    from benchmarks.benchmark_compression import run_benchmark

    result = run_benchmark(n_docs=8, d_model=384, output_path=tmp_path / "compression.json")

    assert result["benchmark"] == "compression"
    assert result["n_docs"] == 8
    assert result["d_model"] == 384
    assert result["float32_embedding_bytes"] == 8 * 384 * 4
    assert result["stored_lattice_key_bytes"] == 8 * (384 // 8)
    assert result["theoretical_lattice_payload_bytes"] == 8 * (384 // 8) * 3
    assert result["theoretical_compression_vs_float32"] == 10.7
    assert result["sqlite_document_count"] == 8
    assert json.loads((tmp_path / "compression.json").read_text())["benchmark"] == "compression"


def test_dedup_benchmark_reports_throughput_and_duplicate_rate(tmp_path):
    from benchmarks.benchmark_dedup import run_benchmark

    result = run_benchmark(
        n_docs=100,
        duplicate_rate=0.2,
        d_model=384,
        brute_force_limit=100,
        output_path=tmp_path / "dedup.json",
    )

    assert result["benchmark"] == "dedup"
    assert result["model"] == "synthetic"
    assert result["n_docs"] == 100
    assert result["duplicate_count"] == 20
    assert result["unique_count"] == 80
    assert result["dedup_rate"] == 0.2
    assert result["e8_docs_per_second"] > 0
    assert result["bruteforce_pairwise_comparisons"] == 100 * 99 // 2
    assert result["bruteforce_elapsed_seconds"] > 0
    assert result["bruteforce_docs_per_second"] > 0
    assert result["speedup_vs_bruteforce"] > 0
    assert result["speedup_docs_measured"] == 100
    assert json.loads((tmp_path / "dedup.json").read_text())["benchmark"] == "dedup"


def test_retrieval_benchmark_reports_latency_paths_and_recall(tmp_path):
    from benchmarks.benchmark_retrieval import run_benchmark

    result = run_benchmark(n_docs=50, n_queries=10, top_k=10, d_model=384, output_path=tmp_path / "retrieval.json")

    assert result["benchmark"] == "retrieval"
    assert result["model"] == "synthetic"
    assert result["n_docs"] == 50
    assert result["n_queries"] == 10
    assert result["top_k"] == 10
    assert result["query_mode"] == "exact"
    assert result["recall_at_k_vs_float32"] == 0.1
    assert result["top1_agreement_vs_float32"] == 1.0
    assert result["path_counts"]["lattice_exact"] == 10
    assert set(result["per_path_latency_ms"]) >= {"lattice_exact", "lattice_hamming1", "fallback", "miss"}
    assert result["per_path_latency_ms"]["lattice_exact"]["count"] == 10
    assert result["per_path_lookup_only_latency_ms"]["lattice_exact"]["count"] == 10
    assert result["latency_ms"]["p50"] >= 0
    assert result["latency_ms"]["p95"] >= result["latency_ms"]["p50"]
    assert result["latency_ms"]["p99"] >= result["latency_ms"]["p95"]
    assert result["lookup_only_latency_ms"]["p50"] >= 0
    assert result["lookup_only_latency_ms"]["p50"] <= result["latency_ms"]["p50"]
    assert json.loads((tmp_path / "retrieval.json").read_text())["benchmark"] == "retrieval"


def test_fallback_quantization_benchmark_reports_quality_size_and_latency(tmp_path):
    from benchmarks.benchmark_fallback_quantization import run_benchmark

    result = run_benchmark(
        n_docs=300,
        n_queries=40,
        d_model=128,
        top_k=10,
        output_path=tmp_path / "fallback_quantization.json",
    )

    assert result["benchmark"] == "fallback_quantization"
    assert result["baseline"] == "float32"
    assert set(result["variants"]) == {"float32", "int8", "int4"}
    for name, variant in result["variants"].items():
        assert variant["index_bytes"] > 0
        assert variant["latency_ms"]["p50"] >= 0
        assert variant["recall_at_10_vs_float32"] >= 0
        assert variant["top1_agreement_vs_float32"] >= 0
        assert variant["top_k_overlap_vs_float32"] >= 0
        if name == "float32":
            assert variant["recall_at_10_vs_float32"] == 1.0
            assert variant["top1_agreement_vs_float32"] == 1.0
    assert result["variants"]["int8"]["compression_vs_float32_fallback"] > 3.5
    assert result["variants"]["int4"]["compression_vs_float32_fallback"] > 7.0
    assert json.loads((tmp_path / "fallback_quantization.json").read_text())["benchmark"] == "fallback_quantization"


def test_retrieval_benchmark_supports_paraphrase_query_mode(tmp_path):
    from benchmarks.benchmark_retrieval import run_benchmark

    result = run_benchmark(
        n_docs=30,
        n_queries=5,
        top_k=5,
        d_model=384,
        query_mode="paraphrase",
        output_path=tmp_path / "retrieval_paraphrase.json",
    )

    assert result["query_mode"] == "paraphrase"
    assert result["queries_are_exact_document_copies"] is False
    assert result["semantic_recall_requires_real_model"] is True


def test_retrieval_helpers_use_cosine_and_actual_recall_at_k():
    from benchmarks.benchmark_retrieval import _recall_at_k, _top_k_doc_ids_by_cosine

    doc_embeddings = [
        [10.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
    ]
    query_embedding = [1.0, 0.5]

    assert _top_k_doc_ids_by_cosine(doc_embeddings, query_embedding, top_k=2) == ["doc-2", "doc-0"]
    assert _recall_at_k(["doc-9", "doc-0"], ["doc-2", "doc-0"], top_k=2) == 0.5


def test_routing_adapter_benchmark_reports_train_and_heldout_metrics(tmp_path):
    from benchmarks.benchmark_routing_adapter import run_benchmark

    result = run_benchmark(
        model="synthetic",
        d_model=384,
        preset="synthetic-copy",
        train_count=8,
        heldout_count=4,
        epochs=2,
        output_path=tmp_path / "routing_adapter.json",
    )

    assert result["benchmark"] == "routing_adapter"
    assert result["model"] == "synthetic"
    assert result["preset"] == "synthetic-copy"
    assert result["train"]["count"] == 8
    assert result["heldout"]["count"] == 4
    assert 0.0 <= result["train"]["lattice_exact_accuracy"] <= 1.0
    assert 0.0 <= result["heldout"]["lattice_exact_accuracy"] <= 1.0
    assert "lattice_exact" in result["train"]["path_counts"]
    assert "lattice_exact" in result["heldout"]["path_counts"]
    assert "final_train_accuracy_reported_by_trainer" in result
    assert result["claim_supported"] in {
        "heldout lattice_exact routing",
        "trained-pair lattice_exact routing only; heldout generalization not shown",
        "no lattice_exact routing shown",
    }
    assert json.loads((tmp_path / "routing_adapter.json").read_text())["benchmark"] == "routing_adapter"


def test_routing_adapter_benchmark_supports_paraphrase_split_preset(tmp_path):
    from benchmarks.benchmark_routing_adapter import run_benchmark

    result = run_benchmark(
        model="synthetic",
        d_model=384,
        preset="natural-qa-paraphrase-split",
        train_count=3,
        heldout_count=2,
        epochs=1,
        output_path=tmp_path / "routing_adapter_split.json",
    )

    assert result["preset"] == "natural-qa-paraphrase-split"
    assert result["documents_indexed"] == 3
    assert result["train"]["count"] == 9
    assert result["heldout"]["count"] == 2
    assert result["heldout"]["rows"][0]["expected"] in result["indexed_documents"]

def test_diagnose_msmarco_smoke(tmp_path):
    import torch
    import unittest.mock
    from benchmarks.diagnose_msmarco import main
    
    adapter_path = tmp_path / "query_adapter.pt"
    torch.save(
        {
            "_version": 1,
            "d_model": 384,
            "weight": torch.eye(384),
            "bias": torch.zeros(384),
            "batch_size": 64,
        },
        adapter_path,
    )
    
    mock_argv = [
        "diagnose_msmarco.py",
        "--adapter", str(adapter_path),
        "--model", "synthetic",
        "--limit", "3",
        "--device", "cpu"
    ]
    with unittest.mock.patch("sys.argv", mock_argv):
        exit_code = main()
        assert exit_code == 0


def test_hamming_router_benchmark_load_pairs_selects_requested_key(tmp_path):
    from benchmarks.benchmark_hamming_router import load_pairs

    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            {
                "paraphrases": [["same a", "same b"]],
                "near_misses": [["near a", "near b"]],
            }
        ),
        encoding="utf-8",
    )

    assert load_pairs(str(pairs_path), key="paraphrases") == [("same a", "same b")]
    assert load_pairs(str(pairs_path), key="near_misses") == [("near a", "near b")]


def test_hamming_router_report_adds_product_gate_and_index_bytes():
    from benchmarks.benchmark_hamming_router import build_product_proof_report, recall_at_fp_budget

    report = build_product_proof_report(
        model="demo-model",
        d_model=1024,
        calibration_sha="abc123",
        fp_budget=0.0,
        chosen_threshold=83,
        calibration_results={
            "threshold": 83,
            "recall": 0.98,
            "fp_rate": 0.0,
            "fp_budget": 0.0,
            "n_paraphrase_pairs": 100,
            "n_near_miss_pairs": 100,
        },
        held_out_recall=0.98,
        held_out_fp_rate=0.0,
        held_out_tp=49,
        held_out_fp=0,
        para_stats={"n": 50, "mean": 60.0},
        nm_stats={"n": 50, "mean": 95.0},
        cache_hits=20,
        cache_misses=30,
        total_prompts=50,
        cache_hit_rate=0.4,
        mean_latency_ms=7.5,
        n_cached_keys=100,
        threshold_curve=[],
        product_recall_target=0.8056,
        held_out_budget_metrics=recall_at_fp_budget(
            paraphrase_dists=[10, 12, 18],
            near_miss_dists=[20, 40],
            fp_budget=0.0,
            max_threshold=40,
        ),
    )

    assert report["product_gate"]["name"] == "recall_at_FP=0"
    assert report["product_gate"]["passed"] is True
    assert report["product_gate"]["recall_target"] == 0.8056
    assert report["product_gate"]["exact_snap_required"] is False
    assert report["metrics"]["held_out_true_positives"] == 49
    assert report["metrics"]["held_out_false_positives"] == 0
    assert report["metrics"]["held_out_recall_at_fp_budget"] == 1.0
    assert report["metrics"]["held_out_threshold_at_fp_budget"] == 19
    assert report["index"]["key_bytes_per_entry"] == 128
    assert report["index"]["stored_key_bytes"] == 12800
    assert report["index"]["float32_embedding_bytes_equivalent"] == 409600


def test_nvidia_demo_dataset_split_writes_benchmark_inputs(tmp_path):
    from benchmarks.generate_nvidia_demo_dataset import write_demo_files

    intents = []
    for idx in range(12):
        intents.append(
            {
                "intent_id": f"intent_{idx}",
                "category": "support",
                "canonical_prompt": f"canonical question {idx}",
                "safe_answer": f"safe answer {idx}",
                "paraphrases": [f"variant {idx}-{j}" for j in range(8)],
            }
        )
    source = {
        "domain": "customer_support_ecommerce_saas",
        "intents": intents,
        "near_miss_pairs": [
            {"a_intent": f"intent_{idx % 12}", "b_intent": f"intent_{(idx + 1) % 12}", "reason": "close"}
            for idx in range(20)
        ],
    }

    paths = write_demo_files(source, tmp_path)

    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    heldout_para = json.loads(paths["heldout_paraphrases"].read_text(encoding="utf-8"))
    heldout_near = json.loads(paths["heldout_near_misses"].read_text(encoding="utf-8"))
    prompts = json.loads(paths["prompts_responses"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert len(calibration["paraphrases"]) == 12 * 8
    assert len(calibration["near_misses"]) == 20 * 4
    assert heldout_para["paraphrases"] == []
    assert len(heldout_near["near_misses"]) == 20 * 2
    assert len(prompts) == 12 * 7
    assert manifest["artifact_type"] == "latticememory_nvidia_product_demo_dataset"


def test_render_hamming_router_results_page_contains_key_metrics():
    from benchmarks.render_hamming_router_results_page import render_results_page

    html = render_results_page(
        {
            "artifact_type": "latticememory_hamming_proof_results",
            "artifact_version": 1,
            "model": "demo-model",
            "d_model": 1024,
            "created_at": "2026-06-07T00:00:00Z",
            "calibration_data_sha256": "abc123",
            "fp_budget": 0.0,
            "calibrated_threshold": 64,
            "product_gate": {
                "name": "recall_at_FP=0",
                "passed": True,
                "recall_target": 0.8056,
                "exact_snap_required": False,
            },
            "metrics": {
                "held_out_recall": 0.875,
                "held_out_fp_rate": 0.0,
                "mean_latency_ms": 12.34,
            },
            "index": {
                "stored_key_bytes": 1280,
                "compression_vs_float32_keys_only": 32.0,
            },
            "cache_simulation": {
                "total_prompts": 10,
                "hit_rate": 0.4,
            },
            "distributions": {
                "paraphrase": {"n": 8, "mean": 55.0},
                "near_miss": {"n": 8, "mean": 90.0},
            },
            "threshold_curve": [{"threshold": 64, "recall": 0.875, "fp_rate": 0.0}],
        }
    )

    assert "LatticeMemory HammingRouter Demo" in html
    assert "87.5%" in html
    assert "0.0%" in html
    assert "PASS" in html
    assert "1,280 bytes" in html
    assert "demo-model" in html


def test_intent_cache_product_report_passes_gate():
    from benchmarks.benchmark_intent_cache_product_proof import build_intent_cache_report

    report = build_intent_cache_report(
        model="demo-model",
        d_model=1024,
        n_intents=12,
        held_out_recall=0.8889,
        held_out_wrong_route_rate=0.0,
        held_out_correct=32,
        held_out_wrong=0,
        total_paraphrases=36,
        total_near_miss_queries=60,
        mean_latency_ms=2.5,
        cache_hits=48,
        cache_misses=24,
        total_prompts=72,
        product_recall_target=0.8056,
    )

    assert report["artifact_type"] == "latticememory_intent_cache_proof_results"
    assert report["product_gate"]["passed"] is True
    assert report["product_gate"]["name"] == "intent_recall_at_zero_wrong_routes"
    assert report["metrics"]["held_out_recall"] == 0.8889
    assert report["metrics"]["held_out_fp_rate"] == 0.0
    assert report["index"]["centroid_bytes"] == 12 * 1024 * 4


def test_hard_near_miss_challenge_has_safety_shape(tmp_path):
    from benchmarks.hard_near_miss_challenge import build_challenge_dataset, write_challenge_dataset

    dataset = build_challenge_dataset()
    assert dataset["domain"] == "hard_customer_support_near_misses"
    assert len(dataset["intents"]) >= 12
    assert len(dataset["paraphrases"]) >= 60
    assert len(dataset["near_misses"]) >= 40

    pair_text = {" || ".join(pair) for pair in dataset["near_misses"]}
    assert any("cancel" in text and "pause" in text for text in pair_text)
    assert any("refund" in text and "return" in text for text in pair_text)
    assert any("password" in text and "email" in text for text in pair_text)

    paths = write_challenge_dataset(tmp_path)
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    heldout_para = json.loads(paths["heldout_paraphrases"].read_text(encoding="utf-8"))
    heldout_near = json.loads(paths["heldout_near_misses"].read_text(encoding="utf-8"))
    prompts = json.loads(paths["prompts_responses"].read_text(encoding="utf-8"))

    assert calibration["paraphrases"]
    assert calibration["near_misses"]
    assert heldout_para["paraphrases"]
    assert heldout_near["near_misses"]
    assert len(prompts) >= len(dataset["intents"]) * 4
    assert all("prompt" in row and "response" in row and "intent_id" in row for row in prompts)


def test_recall_zero_fp_budget_metrics_choose_best_threshold():
    from benchmarks.benchmark_recall_zero_fp import recall_at_fp_budgets

    result = recall_at_fp_budgets(
        paraphrase_dists=[10, 12, 14, 30],
        near_miss_dists=[20, 25, 35, 40],
        fp_budgets=[0.0, 0.25, 1.0],
        max_threshold=40,
    )

    assert result["0.0"]["threshold"] == 19
    assert result["0.0"]["recall"] == 0.75
    assert result["0.0"]["fp_rate"] == 0.0
    assert result["0.25"]["threshold"] == 24
    assert result["0.25"]["recall"] == 0.75
    assert result["0.25"]["fp_rate"] == 0.25
    assert result["1.0"]["threshold"] == 40
    assert result["1.0"]["recall"] == 1.0
    assert result["1.0"]["fp_rate"] == 1.0


def test_block_failure_audit_counts_false_negative_and_confusion_blocks():
    from benchmarks.benchmark_block_failure_audit import block_failure_summary

    paraphrase_rows = [
        {
            "a": "same 1",
            "b": "same 2",
            "hamming": 2,
            "diff_blocks": [1, 3],
        },
        {
            "a": "same 3",
            "b": "same 4",
            "hamming": 1,
            "diff_blocks": [3],
        },
    ]
    near_miss_rows = [
        {
            "a": "near 1",
            "b": "near 2",
            "hamming": 1,
            "diff_blocks": [1],
        },
        {
            "a": "near 3",
            "b": "near 4",
            "hamming": 4,
            "diff_blocks": [4, 5, 6, 7],
        },
    ]

    summary = block_failure_summary(
        paraphrase_rows=paraphrase_rows,
        near_miss_rows=near_miss_rows,
        threshold=1,
        n_blocks=8,
        closest_near_miss_count=1,
    )

    assert summary["false_negative_count"] == 1
    assert summary["near_miss_confusion_count"] == 1
    assert summary["closest_near_miss_count"] == 1
    assert summary["false_negative_blocks"][0]["block"] == 1
    assert summary["false_negative_blocks"][0]["count"] == 1
    assert summary["stable_but_confusing_blocks"][0]["block"] == 0
    assert summary["closest_near_miss_same_blocks"][0]["block"] == 0


def test_canonical_key_training_builds_intent_rows():
    from benchmarks.benchmark_canonical_key_training import build_intent_lookup, build_positive_rows

    source = {
        "intents": [
            {
                "intent_id": "cancel",
                "canonical": "cancel subscription",
                "paraphrases": ["stop plan", "end membership", "turn off renewal"],
            },
            {
                "intent_id": "pause",
                "canonical": "pause subscription",
                "paraphrases": ["hold plan", "freeze membership", "pause renewal"],
            },
        ]
    }

    lookup = build_intent_lookup(source)
    assert lookup["stop plan"] == "cancel"
    assert lookup["pause subscription"] == "pause"

    train_rows, heldout_rows = build_positive_rows(source, train_per_intent=2)
    assert train_rows == [
        {"intent_id": "cancel", "prompt": "cancel subscription", "target": "cancel subscription"},
        {"intent_id": "cancel", "prompt": "stop plan", "target": "cancel subscription"},
        {"intent_id": "cancel", "prompt": "end membership", "target": "cancel subscription"},
        {"intent_id": "pause", "prompt": "pause subscription", "target": "pause subscription"},
        {"intent_id": "pause", "prompt": "hold plan", "target": "pause subscription"},
        {"intent_id": "pause", "prompt": "freeze membership", "target": "pause subscription"},
    ]
    assert heldout_rows == [
        {"intent_id": "cancel", "prompt": "turn off renewal", "target": "cancel subscription"},
        {"intent_id": "pause", "prompt": "pause renewal", "target": "pause subscription"},
    ]


def test_exact_snap_block_audit_schema():
    from benchmarks.benchmark_exact_snap_block_audit import summarize_exact_snap_blocks

    query_keys = [[1, 2, 3, 4], [1, 2, 0, 4]]
    target_keys = [[1, 9, 3, 4], [1, 2, 3, 5]]
    target_probs = [[0.9, 0.1, 0.8, 0.7], [0.95, 0.92, 0.2, 0.3]]

    report = summarize_exact_snap_blocks(
        query_keys=query_keys,
        target_keys=target_keys,
        target_probabilities=target_probs,
        cluster_name="refund_policy",
    )

    assert report["cluster_name"] == "refund_policy"
    assert report["n_pairs"] == 2
    assert report["exact_same_key_rate"] == 0.0
    assert report["mean_hamming"] == 1.5
    assert report["mean_correct_blocks"] == 2.5
    assert report["worst_blocks"][0]["block"] in {1, 2, 3}
    assert "repeated_wrong_blocks" in report
