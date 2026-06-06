from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from latticememory.index import LatticeIndex
from latticememory.memory import MemoryQuery

from benchmarks.common import load_text_encoder, percentile, write_json_result


def _make_index(encoder, d_model: int, batch_size: int, beam_radius: int = 1) -> LatticeIndex:
    index = LatticeIndex.__new__(LatticeIndex)
    index._init_with_encoder(encoder, d_model=d_model, batch_size=batch_size, beam_radius=beam_radius)
    return index


def run_benchmark(
    *,
    n_docs: int = 100_000,
    n_queries: int = 1_000,
    top_k: int = 10,
    d_model: int | None = 1024,
    model: str = "synthetic",
    query_mode: str = "exact",
    output_path: str | Path | None = None,
    batch_size: int = 64,
    beam_radius: int = 1,
) -> dict:
    if n_docs < 1:
        raise ValueError("n_docs must be positive")
    if n_queries < 1:
        raise ValueError("n_queries must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if n_queries > n_docs:
        raise ValueError("n_queries must be <= n_docs for exact-query benchmark")
    if query_mode not in {"exact", "paraphrase"}:
        raise ValueError("query_mode must be 'exact' or 'paraphrase'")

    docs = [f"benchmark document {idx:06d}" for idx in range(n_docs)]
    queries = _build_queries(docs, n_queries=n_queries, query_mode=query_mode)
    encoder, runtime_dim = load_text_encoder(model, d_model, batch_size=batch_size)
    index = _make_index(encoder, runtime_dim, batch_size, beam_radius=beam_radius)

    build_start = time.perf_counter()
    index.add(docs)
    build_seconds = time.perf_counter() - build_start

    # Pre-encode once — reused for recall baseline, full-latency loop, and isolated-lookup loop.
    doc_embeddings = index._runtime._encode_texts(docs).numpy()
    query_tensors = index._runtime._encode_texts(queries)
    query_embeddings_np = query_tensors.numpy()

    # Full search latency: text encoding + E8 lookup + result re-encoding.
    latency_ms: list[float] = []
    latency_by_path: dict[str, list[float]] = _empty_path_samples()
    path_counts: Counter[str] = Counter()
    top1_matches = 0
    recall_scores: list[float] = []

    for query_idx, query in enumerate(queries):
        start = time.perf_counter()
        results = index.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latency_ms.append(elapsed_ms)
        if results:
            path = results[0].retrieval_path
            path_counts[path] += 1
            latency_by_path.setdefault(path, []).append(elapsed_ms)

        baseline_doc_ids = _top_k_doc_ids_by_cosine(doc_embeddings, query_embeddings_np[query_idx], top_k=top_k)
        result_doc_ids = [result.doc_id for result in results]
        recall_scores.append(_recall_at_k(result_doc_ids, baseline_doc_ids, top_k=top_k))
        if results and results[0].doc_id == baseline_doc_ids[0]:
            top1_matches += 1

    # Isolated lookup latency: pre-encoded embedding → E8 key → hash lookup only.
    # This is the number the paper cares about; latency_ms above conflates encoding time.
    lookup_only_ms: list[float] = []
    lookup_only_by_path: dict[str, list[float]] = _empty_path_samples()
    for q_idx, q_emb in enumerate(query_tensors):
        mq = MemoryQuery(text=queries[q_idx], embedding=q_emb, top_k=top_k)
        start = time.perf_counter()
        lookup_result = index._runtime.memory.retrieve(mq)
        elapsed_ms = (time.perf_counter() - start) * 1000
        lookup_only_ms.append(elapsed_ms)
        lookup_only_by_path.setdefault(lookup_result.path, []).append(elapsed_ms)

    result = {
        "benchmark": "retrieval",
        "n_docs": int(n_docs),
        "n_queries": int(n_queries),
        "top_k": int(top_k),
        "model": model,
        "query_mode": query_mode,
        "queries_are_exact_document_copies": query_mode == "exact",
        "semantic_recall_requires_real_model": model == "synthetic" and query_mode != "exact",
        "d_model": int(runtime_dim),
        "build_seconds": build_seconds,
        "docs_per_second_build": n_docs / max(build_seconds, 1e-12),
        "latency_ms": {
            "p50": percentile(latency_ms, 50),
            "p95": percentile(latency_ms, 95),
            "p99": percentile(latency_ms, 99),
        },
        "lookup_only_latency_ms": {
            "p50": percentile(lookup_only_ms, 50),
            "p95": percentile(lookup_only_ms, 95),
            "p99": percentile(lookup_only_ms, 99),
        },
        "per_path_latency_ms": _summarize_path_samples(latency_by_path),
        "per_path_lookup_only_latency_ms": _summarize_path_samples(lookup_only_by_path),
        "path_counts": dict(path_counts),
        "recall_at_k_vs_float32": round(float(np.mean(recall_scores)) if recall_scores else 0.0, 6),
        "top1_agreement_vs_float32": round(top1_matches / n_queries, 6),
        "index_stats": index.stats().__dict__,
    }
    return write_json_result(result, output_path)


def _top_k_doc_ids_by_cosine(doc_embeddings, query_embedding, *, top_k: int) -> list[str]:
    docs = np.asarray(doc_embeddings, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
    docs = docs / np.clip(doc_norms, 1e-9, None)
    query = query / max(float(np.linalg.norm(query)), 1e-9)
    scores = docs @ query
    k = min(top_k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    return [f"doc-{int(idx)}" for idx in order]


def _recall_at_k(result_doc_ids: list[str], baseline_doc_ids: list[str], *, top_k: int) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    expected = set(baseline_doc_ids[:top_k])
    if not expected:
        return 0.0
    observed = set(result_doc_ids[:top_k])
    return len(observed & expected) / len(expected)


def _build_queries(docs: list[str], *, n_queries: int, query_mode: str) -> list[str]:
    if query_mode == "exact":
        return docs[:n_queries]
    return [f"Find information about: {doc}" for doc in docs[:n_queries]]


def _empty_path_samples() -> dict[str, list[float]]:
    return {
        "lattice_exact": [],
        "lattice_hamming1": [],
        "fallback": [],
        "miss": [],
    }


def _summarize_path_samples(samples_by_path: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        path: {
            "count": len(samples),
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
            "p99": percentile(samples, 99),
        }
        for path, samples in samples_by_path.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LatticeMemory retrieval latency and recall.")
    parser.add_argument("--n-docs", type=int, default=100_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--model", default="synthetic")
    parser.add_argument("--query-mode", choices=["exact", "paraphrase"], default="exact")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-radius", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/retrieval.json"))
    args = parser.parse_args()
    result = run_benchmark(
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        top_k=args.top_k,
        d_model=args.d_model,
        model=args.model,
        query_mode=args.query_mode,
        batch_size=args.batch_size,
        beam_radius=args.beam_radius,
        output_path=args.output,
    )
    print(f"Wrote {result['benchmark']} benchmark to {args.output}")


if __name__ == "__main__":
    main()
