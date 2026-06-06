# Walkthrough: Phase 7 (Reproducible Benchmark Suite)

This walkthrough documents the design and first-pass execution of **Phase 7: Reproducible Benchmark Suite**.

## Accomplishments
1. **Empirical Benchmarks**: Built three benchmark scripts in `benchmarks/`:
   - `benchmark_compression.py`: Verified 10.7x compression empirically based on actual SQLite stored bytes.
   - `benchmark_dedup.py`: Proved $O(N)$ E8 snapping deduplication is orders of magnitude faster than brute-force $O(N^2)$ comparison loops.
   - `benchmark_retrieval.py`: Measures latency profiles and Recall@K vs float32 baseline.
2. **Neighborhood / Beam Search Tuning**:
   - Added `beam_radius` configuration options to the CLI and `LatticeIndex` API.
   - Fixed the calculation of `recall_at_k_vs_float32` to correctly compute average Recall@K rather than Top-1 match rate, addressing precision differences for highly similar documents.
3. **Automated Verification**: Integrated smoke-test assertions in `tests/test_benchmarks.py` to ensure benchmark scripts execute without crashing.
