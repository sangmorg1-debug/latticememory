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
