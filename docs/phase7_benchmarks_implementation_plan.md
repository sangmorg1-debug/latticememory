# Implementation Plan: Phase 7 (Reproducible Benchmark Suite)

This plan outlines the design and integration of **Phase 7: Reproducible Benchmark Suite** to empirically verify LatticeMemory performance, compression, and retrieval qualities.

## Goal Description
To validate LatticeMemory's commercial claims, we need automated benchmark scripts measuring:
1. Compression (stored bytes vs raw float32).
2. Deduplication throughput (avoided pairwise comparisons vs brute-force $O(N^2)$).
3. Retrieval (build latency, query latency p50/p95/p99, path counts, and Recall@K vs float32).

## Proposed Changes
* Create `benchmarks/common.py` providing unified text encoding loading.
* Create `benchmarks/benchmark_compression.py` evaluating stored E8 key + scale payload bytes.
* Create `benchmarks/benchmark_dedup.py` measuring semantic dedup rates.
* Create `benchmarks/benchmark_retrieval.py` testing build/query times and Recall@K with CPU-friendly CLI execution and custom `--beam-radius` options.
* Add unit test verification in `tests/test_benchmarks.py`.

## Verification Plan
* Run full benchmarks on synthetic datasets.
* Run real-model benchmarks on snapshot-trained embeddings (e.g. `dfrokido/bge-large-e8-snap`) on CPU.
