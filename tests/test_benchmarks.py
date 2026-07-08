from __future__ import annotations

import json


def test_compression_benchmark_reports_actual_and_hybrid_bytes(tmp_path):
    from benchmarks.benchmark_compression import run_benchmark

    result = run_benchmark(n_docs=8, d_model=384, output_path=tmp_path / "compression.json")

    assert result["benchmark"] == "compression"
    assert result["n_docs"] == 8
    assert result["d_model"] == 384
    assert result["float32_embedding_bytes"] == 8 * 384 * 4
    assert result["stored_lattice_key_bytes"] == 8 * (384 // 8)
    assert result["stored_key_compression_vs_float32"] == 32.0
    assert result["hybrid_fallback_payload_bytes"] == 8 * (384 // 8) * 3
    assert result["hybrid_compression_vs_float32"] == 10.7
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
        },
        title="LatticeMemory HammingRouter Demo",
    )

    assert "LatticeMemory HammingRouter Demo" in html
    assert "87.5%" in html
    assert "0.0%" in html
    assert "PASS" in html
    assert "demo-model" in html


def test_intent_cache_product_report_passes_gate_v2():
    from benchmarks.benchmark_intent_cache_product_proof import build_intent_cache_report

    report = build_intent_cache_report(
        model="demo-model",
        domain="customer_support",
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
        threshold_curve=[],
        paraphrase_examples=[],
        near_miss_examples=[]
    )

    assert report["artifact_type"] == "latticememory_intent_cache_proof_results"
    assert report["artifact_version"] == 2
    assert report["domain"] == "customer_support"
    assert report["product_gate"]["passed"] is True
    assert report["product_gate"]["name"] == "intent_recall_at_zero_wrong_routes"
    assert report["metrics"]["held_out_recall"] == 0.8889
    assert report["metrics"]["held_out_fp_rate"] == 0.0
    assert report["index"]["centroid_bytes"] == 12 * 1024 * 4
    assert "threshold_curve" in report
    assert "paraphrase_examples" in report
    assert "near_miss_examples" in report


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


def test_hard_near_miss_helpdesk_has_correct_shape(tmp_path):
    from benchmarks.hard_near_miss_helpdesk import build_challenge_dataset, write_challenge_dataset

    dataset = build_challenge_dataset()
    assert dataset["domain"] == "hard_helpdesk_near_misses"
    assert len(dataset["intents"]) >= 12
    assert len(dataset["paraphrases"]) >= 60
    assert len(dataset["near_misses"]) >= 40

    pair_text = {" || ".join(pair) for pair in dataset["near_misses"]}
    assert any("vpn" in text.lower() or "laptop" in text.lower() for text in pair_text)
    assert any("license" in text.lower() or "software" in text.lower() for text in pair_text)
    assert any("expense" in text.lower() or "leave" in text.lower() for text in pair_text)

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
    assert all("intent_id" in row for row in prompts)


def test_shadow_mode_demo_runs_with_synthetic_encoder(tmp_path):
    from benchmarks.benchmark_shadow_mode_demo import run_shadow_demo

    prompts_responses = [
        {"prompt": "Where is my order?", "response": "Check order history.", "intent_id": "order_status"},
        {"prompt": "Track my package", "response": "Check order history.", "intent_id": "order_status"},
        {"prompt": "How do I cancel?", "response": "Cancel from billing.", "intent_id": "cancel"},
        {"prompt": "Stop my subscription", "response": "Cancel from billing.", "intent_id": "cancel"},
        {"prompt": "Reset my password", "response": "Use the reset link.", "intent_id": "reset_password"},
    ]
    calibration_data = {
        "paraphrases": [
            ["Where is my order?", "Track my package"],
            ["How do I cancel?", "Stop my subscription"],
        ],
        "near_misses": [
            ["Where is my order?", "How do I cancel?"],
        ],
    }
    output_json = tmp_path / "shadow.json"
    report = run_shadow_demo(
        model="synthetic",
        prompts_responses=prompts_responses,
        calibration_data=calibration_data,
        hamming_threshold=70,
        output_path=output_json,
    )

    assert report["artifact_type"] == "latticememory_shadow_mode_demo"
    assert report["stream"]["total_prompts"] == 5
    assert report["stream"]["shadow_hit_rate"] >= 0.0
    assert report["stream"]["would_be_savings_usd"] >= 0.0
    assert report["latency"]["mean_lookup_ms"] >= 0.0
    assert report["index"]["total_key_bytes"] > 0
    assert output_json.exists()


def test_shadow_mode_html_render_contains_key_sections(tmp_path):
    from benchmarks.benchmark_shadow_mode_demo import render_shadow_demo_html

    report = {
        "artifact_type": "latticememory_shadow_mode_demo",
        "artifact_version": 1,
        "model": "test-model",
        "d_model": 128,
        "hamming_threshold": 70,
        "created_at": "2026-06-08T00:00:00Z",
        "stream": {
            "total_prompts": 100,
            "shadow_hits": 45,
            "true_misses": 55,
            "shadow_hit_rate": 0.45,
            "cost_per_query_usd": 0.00025,
            "would_be_savings_usd": 0.01125,
            "would_be_savings_pct": 45.0,
        },
        "latency": {"mean_lookup_ms": 0.05, "p95_lookup_ms": 0.12, "probe_n": 200},
        "hamming_distribution": {"n": 45, "mean": 30.0, "exact_hits": 10, "histogram": {"0-9": 10, "30-39": 35}},
        "index": {"n_keys": 20, "key_bytes_per_entry": 16, "total_key_bytes": 320},
        "examples": {"shadow_hits": [], "misses": []},
        "calibration": {"n_canonical_texts": 10, "n_calibration_pairs": 20},
    }
    html = render_shadow_demo_html(report)
    assert "45.0%" in html
    assert "Shadow-Mode" in html
    assert "test-model" in html
    assert "0.0500" in html or "0.05" in html


def test_premium_html_renderer_handles_intent_cache_v2():
    from benchmarks.render_hamming_router_results_page import render_results_page

    report = {
        "artifact_type": "latticememory_intent_cache_proof_results",
        "artifact_version": 2,
        "route_type": "closed_set_intent_centroid_cache",
        "model": "demo-model",
        "d_model": 1024,
        "domain": "customer_support",
        "created_at": "2026-06-08T00:00:00Z",
        "calibrated_threshold": "nearest_centroid",
        "fp_budget": 0.0,
        "product_gate": {
            "name": "intent_recall_at_zero_wrong_routes",
            "passed": True,
            "recall_target": 0.8056,
            "fp_budget": 0.0,
            "exact_snap_required": False,
            "fragmentation_metric_role": "research_exact_snap",
        },
        "metrics": {
            "held_out_recall": 0.8889,
            "held_out_fp_rate": 0.0,
            "held_out_true_positives": 32,
            "held_out_false_positives": 0,
            "total_held_out_paraphrases": 36,
            "total_near_miss_queries": 60,
            "mean_latency_ms": 0.072,
        },
        "index": {
            "n_intents": 12,
            "centroid_bytes": 49152,
            "stored_key_bytes": 49152,
            "compression_vs_float32_keys_only": 1.0,
        },
        "cache_simulation": {"total_prompts": 72, "hits": 60, "misses": 12, "hit_rate": 0.8333},
        "distributions": {"paraphrase": {"n": 36}, "near_miss": {"n": 60}},
        "threshold_curve": [
            {"min_score_threshold": 0.80, "recall": 0.95, "fp_rate": 0.0, "coverage": 0.9},
            {"min_score_threshold": 0.90, "recall": 0.88, "fp_rate": 0.0, "coverage": 0.7},
            {"min_score_threshold": 0.95, "recall": 0.70, "fp_rate": 0.0, "coverage": 0.5},
        ],
        "paraphrase_examples": [
            {"query": "Stop my plan", "expected": "cancel_subscription", "predicted": "cancel_subscription", "result": "correct"},
        ],
        "near_miss_examples": [],
        "splits_summary": {
            "n_splits": 5,
            "seed": 42,
            "recall": {"mean": 0.89, "std": 0.03, "min": 0.85, "max": 0.93, "n_splits": 5, "values": [0.85, 0.88, 0.90, 0.91, 0.93]},
            "wrong_route_rate": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_splits": 5, "values": [0.0, 0.0, 0.0, 0.0, 0.0]},
            "cache_hit_rate": {"mean": 0.82, "std": 0.04, "min": 0.78, "max": 0.87, "n_splits": 5, "values": [0.78, 0.80, 0.82, 0.84, 0.87]},
        },
    }
    html = render_results_page(report, title="LatticeMemory Test")
    assert "PASS" in html
    assert "88.9%" in html
    assert "0.0%" in html
    assert "Cross-Validation" in html
    assert "83.3%" in html  # cache hit rate
    assert "Routing Confidence" in html  # threshold curve section


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


# ---------------------------------------------------------------------------
# Multi-domain runner tests
# ---------------------------------------------------------------------------

def test_build_multidomain_report_aggregates_correctly():
    from benchmarks.benchmark_multidomain_runner import build_multidomain_report

    domain_results = [
        {
            "domain": "customer_support",
            "held_out_recall": 0.90,
            "held_out_fp_rate": 0.0,
            "cache_hit_rate": 0.80,
            "n_intents": 12,
            "passed": True,
        },
        {
            "domain": "it_helpdesk",
            "held_out_recall": 0.85,
            "held_out_fp_rate": 0.0,
            "cache_hit_rate": 0.75,
            "n_intents": 15,
            "passed": True,
        },
    ]

    report = build_multidomain_report(
        model="test-model",
        domain_results=domain_results,
        product_recall_target=0.8056,
    )

    assert report["artifact_type"] == "latticememory_multidomain_proof_results"
    assert report["artifact_version"] == 2
    assert report["product_gate"]["passed"] is True
    assert report["product_gate"]["all_domains_pass"] is True
    assert report["aggregate_metrics"]["n_domains"] == 2
    assert abs(report["aggregate_metrics"]["mean_held_out_recall"] - 0.875) < 1e-4
    assert report["aggregate_metrics"]["max_held_out_fp_rate"] == 0.0
    assert abs(report["aggregate_metrics"]["mean_cache_hit_rate"] - 0.775) < 1e-4


def test_build_multidomain_report_fails_gate_when_any_domain_fails():
    from benchmarks.benchmark_multidomain_runner import build_multidomain_report

    domain_results = [
        {
            "domain": "a",
            "held_out_recall": 0.90,
            "held_out_fp_rate": 0.0,
            "cache_hit_rate": 0.80,
            "n_intents": 10,
            "passed": True,
        },
        {
            "domain": "b",
            "held_out_recall": 0.70,   # below target
            "held_out_fp_rate": 0.02,
            "cache_hit_rate": 0.50,
            "n_intents": 8,
            "passed": False,
        },
    ]

    report = build_multidomain_report(
        model="test-model",
        domain_results=domain_results,
        product_recall_target=0.8056,
    )

    assert report["product_gate"]["passed"] is False
    assert report["product_gate"]["all_domains_pass"] is False
    assert report["aggregate_metrics"]["max_held_out_fp_rate"] == 0.02


def test_build_multidomain_report_raises_on_empty_domains():
    from benchmarks.benchmark_multidomain_runner import build_multidomain_report
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        build_multidomain_report(model="m", domain_results=[], product_recall_target=0.8)


def test_run_multidomain_proof_validates_config_keys(tmp_path):
    from benchmarks.benchmark_multidomain_runner import run_multidomain_proof
    import pytest

    # Missing required keys
    bad_config = [{"name": "x", "source": "x.json"}]  # missing calibration, etc.
    with pytest.raises(ValueError, match="missing keys"):
        run_multidomain_proof(
            model="synthetic",
            config=bad_config,
            output_path=tmp_path / "out.json",
        )


def test_run_multidomain_proof_with_existing_demo_data_synthetic(tmp_path):
    """End-to-end run using both existing demo domains with synthetic encoder."""
    from pathlib import Path
    from benchmarks.benchmark_multidomain_runner import run_multidomain_proof
    import json

    # Resolve absolute paths from this file's location so CWD doesn't matter
    _root = Path(__file__).parent.parent
    _challenge = _root / "benchmarks" / "demo_data" / "hard_near_miss_challenge"
    _helpdesk = _root / "benchmarks" / "demo_data" / "hard_near_miss_helpdesk"

    config = [
        {
            "name": "customer_support",
            "source": str(_challenge / "hard_near_miss_source.json"),
            "calibration": str(_challenge / "calibration_data.json"),
            "paraphrases": str(_challenge / "heldout_paraphrases.json"),
            "near_misses": str(_challenge / "heldout_near_misses.json"),
            "prompts_responses": str(_challenge / "prompts_responses.json"),
        },
        {
            "name": "it_hr_helpdesk",
            "source": str(_helpdesk / "hard_near_miss_source.json"),
            "calibration": str(_helpdesk / "calibration_data.json"),
            "paraphrases": str(_helpdesk / "heldout_paraphrases.json"),
            "near_misses": str(_helpdesk / "heldout_near_misses.json"),
            "prompts_responses": str(_helpdesk / "prompts_responses.json"),
        },
    ]

    report = run_multidomain_proof(
        model="synthetic",
        config=config,
        output_path=tmp_path / "multidomain_proof.json",
        n_splits=2,  # fast: 2-fold instead of default 5
    )

    # Schema checks
    assert report["artifact_type"] == "latticememory_multidomain_proof_results"
    assert report["aggregate_metrics"]["n_domains"] == 2
    assert 0.0 <= report["aggregate_metrics"]["mean_held_out_recall"] <= 1.0
    assert report["aggregate_metrics"]["max_held_out_fp_rate"] >= 0.0
    assert len(report["domains"]) == 2
    for domain in report["domains"]:
        assert "domain" in domain
        assert "held_out_recall" in domain
        assert "held_out_fp_rate" in domain
        assert "passed" in domain

    # Output file written
    assert (tmp_path / "multidomain_proof.json").exists()
    saved = json.loads((tmp_path / "multidomain_proof.json").read_text(encoding="utf-8"))
    assert saved["artifact_type"] == "latticememory_multidomain_proof_results"


# ---------------------------------------------------------------------------
# Proxy stream replay benchmark tests
# ---------------------------------------------------------------------------

def test_build_replay_report_schema():
    from benchmarks.benchmark_proxy_stream_replay import build_replay_report
    import pytest

    report = build_replay_report(
        model="test-model",
        d_model=128,
        hamming_threshold=70,
        total_prompts=100,
        cache_hits=40,
        true_misses=60,
        hamming_distances=[0, 0, 15, 22, 30, 50],
        lookup_latencies_ms=[0.05, 0.06, 0.04, 0.07, 0.05],
        encode_latencies_ms=[1.2, 1.3, 1.1, 1.4],
        n_keys_in_index=60,
        cost_per_query_usd=0.00025,
    )

    assert report["artifact_type"] == "latticememory_proxy_stream_replay"
    assert report["artifact_version"] == 1
    assert report["stream"]["total_prompts"] == 100
    assert report["stream"]["cache_hits"] == 40
    assert report["stream"]["hit_rate"] == 0.4
    assert report["stream"]["would_be_savings_pct"] == 40.0
    assert report["stream"]["would_be_savings_usd"] == pytest.approx(40 * 0.00025, abs=1e-6)

    assert report["lookup_latency_ms"]["n"] == 5
    assert report["lookup_latency_ms"]["p50"] > 0
    assert report["lookup_latency_ms"]["p95"] >= report["lookup_latency_ms"]["p50"]
    assert report["encode_latency_ms"]["n"] == 4
    assert report["encode_latency_ms"]["p50"] > 0

    assert report["hamming_distribution"]["n"] == 6
    assert report["hamming_distribution"]["exact_hits"] == 2

    key_bytes = 60 * (128 // 8)
    assert report["index"]["total_key_bytes"] == key_bytes
    assert report["index"]["compression_vs_float32_keys_only"] == pytest.approx(
        60 * 128 * 4 / key_bytes, abs=0.01
    )


def test_build_replay_report_empty_latencies():
    from benchmarks.benchmark_proxy_stream_replay import build_replay_report
    import pytest

    report = build_replay_report(
        model="m",
        d_model=64,
        hamming_threshold=50,
        total_prompts=5,
        cache_hits=2,
        true_misses=3,
        hamming_distances=[],
        lookup_latencies_ms=[],
        encode_latencies_ms=[],
        n_keys_in_index=3,
    )

    assert report["stream"]["hit_rate"] == pytest.approx(0.4, abs=1e-4)
    assert report["lookup_latency_ms"]["n"] == 0
    assert report["lookup_latency_ms"]["p50"] == 0.0
    assert report["hamming_distribution"]["n"] == 0
    assert report["hamming_distribution"]["mean"] == 0.0


def test_run_proxy_stream_replay_with_synthetic_encoder(tmp_path):
    """End-to-end: synthetic encoder, small prompt set, verify report schema and file."""
    import json
    from benchmarks.benchmark_proxy_stream_replay import run_proxy_stream_replay

    prompts_responses = [
        {"prompt": "Where is my order?", "response": "Check your order history.", "intent_id": "order_status"},
        {"prompt": "Track my package", "response": "Check your order history.", "intent_id": "order_status"},
        {"prompt": "How do I cancel?", "response": "Go to billing settings.", "intent_id": "cancel"},
        {"prompt": "Stop my subscription", "response": "Go to billing settings.", "intent_id": "cancel"},
        {"prompt": "Reset my password", "response": "Use the reset link.", "intent_id": "reset"},
        {"prompt": "I forgot my login", "response": "Use the reset link.", "intent_id": "reset"},
        {"prompt": "How do I get a refund?", "response": "Submit a refund form.", "intent_id": "refund"},
    ]
    calibration_data = {
        "paraphrases": [
            ["Where is my order?", "Track my package"],
            ["How do I cancel?", "Stop my subscription"],
            ["Reset my password", "I forgot my login"],
        ],
        "near_misses": [
            ["Where is my order?", "How do I cancel?"],
        ],
    }
    output_json = tmp_path / "proxy_replay.json"

    report = run_proxy_stream_replay(
        model="synthetic",
        prompts_responses=prompts_responses,
        calibration_data=calibration_data,
        hamming_threshold=70,
        output_path=output_json,
        latency_probe_n=20,   # small for speed
    )

    assert report["artifact_type"] == "latticememory_proxy_stream_replay"
    assert report["stream"]["total_prompts"] == 7
    assert report["stream"]["cache_hits"] + report["stream"]["true_misses"] == 7
    assert 0.0 <= report["stream"]["hit_rate"] <= 1.0
    assert report["lookup_latency_ms"]["n"] == 20
    assert report["lookup_latency_ms"]["p50"] >= 0.0
    assert report["index"]["total_key_bytes"] > 0
    assert report["index"]["compression_vs_float32_keys_only"] > 1.0  # E8 is always smaller
    assert output_json.exists()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["artifact_type"] == "latticememory_proxy_stream_replay"


def test_run_proxy_stream_replay_with_existing_demo_files(tmp_path):
    """End-to-end: uses on-disk demo data, synthetic encoder, full pipeline."""
    import json
    from pathlib import Path
    from benchmarks.benchmark_proxy_stream_replay import run_proxy_stream_replay

    _root = Path(__file__).parent.parent
    _challenge = _root / "benchmarks" / "demo_data" / "hard_near_miss_challenge"

    prompts_responses = json.loads(
        (_challenge / "prompts_responses.json").read_text(encoding="utf-8")
    )
    calibration_data = json.loads(
        (_challenge / "calibration_data.json").read_text(encoding="utf-8")
    )

    report = run_proxy_stream_replay(
        model="synthetic",
        prompts_responses=prompts_responses,
        calibration_data=calibration_data,
        hamming_threshold=70,
        output_path=tmp_path / "proxy_replay_demo.json",
        latency_probe_n=50,
    )

    assert report["stream"]["total_prompts"] > 0
    assert report["index"]["n_keys"] > 0
    # artifact_type field present and correct
    assert report["artifact_type"] == "latticememory_proxy_stream_replay"
    assert (tmp_path / "proxy_replay_demo.json").exists()
