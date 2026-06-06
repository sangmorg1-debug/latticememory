from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.common import generate_unit_embeddings, load_text_encoder, percentile, write_json_result
from latticememory.memory import DenseVectorFallback, MemoryDocument


def run_benchmark(
    *,
    n_docs: int = 10_000,
    n_queries: int = 1_000,
    d_model: int = 384,
    top_k: int = 10,
    model: str = "random",
    query_mode: str = "random",
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 123,
    output_path: str | Path | None = None,
) -> dict:
    if n_docs < 1:
        raise ValueError("n_docs must be positive")
    if n_queries < 1:
        raise ValueError("n_queries must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if d_model < 1:
        raise ValueError("d_model must be positive")
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for int4 fallback benchmark")
    if query_mode not in {"random", "exact", "paraphrase"}:
        raise ValueError("query_mode must be 'random', 'exact', or 'paraphrase'")

    doc_embeddings, query_embeddings, runtime_dim = _build_embeddings(
        n_docs=n_docs,
        n_queries=n_queries,
        d_model=d_model,
        model=model,
        query_mode=query_mode,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    if runtime_dim % 2 != 0:
        raise ValueError("runtime model dimension must be even for int4 fallback benchmark")
    documents = [
        MemoryDocument(doc_id=f"doc-{idx}", text=f"benchmark document {idx:06d}", embedding=doc_embeddings[idx])
        for idx in range(n_docs)
    ]

    fallbacks = {
        "float32": DenseVectorFallback(d_model=runtime_dim),
        "int8": DenseVectorFallback(d_model=runtime_dim, quantization_bits=8),
        "int4": DenseVectorFallback(d_model=runtime_dim, quantization_bits=4),
    }
    build_seconds: dict[str, float] = {}
    for name, fallback in fallbacks.items():
        start = time.perf_counter()
        fallback.add_documents(documents)
        build_seconds[name] = time.perf_counter() - start

    baseline_rows = _search_all(fallbacks["float32"], query_embeddings, top_k=top_k)
    baseline_ids = [[doc_id for doc_id, _score in row["hits"]] for row in baseline_rows]
    baseline_score_maps = [
        {doc_id: score for doc_id, score in row["hits"]}
        for row in baseline_rows
    ]
    float32_bytes = n_docs * runtime_dim * 4

    variants = {}
    for name, fallback in fallbacks.items():
        rows = baseline_rows if name == "float32" else _search_all(fallback, query_embeddings, top_k=top_k)
        doc_ids = [[doc_id for doc_id, _score in row["hits"]] for row in rows]
        variants[name] = {
            "quantization_bits": getattr(fallback, "quantization_bits", None),
            "build_seconds": build_seconds[name],
            "index_bytes": int(fallback.get_index_size_bytes()),
            "compression_vs_float32_fallback": round(
                float32_bytes / max(int(fallback.get_index_size_bytes()), 1),
                6,
            ),
            "latency_ms": _summarize_latency([row["latency_ms"] for row in rows]),
            f"recall_at_{top_k}_vs_float32": round(_mean_recall(doc_ids, baseline_ids, top_k=top_k), 6),
            "top1_agreement_vs_float32": round(_top1_agreement(doc_ids, baseline_ids), 6),
            "top_k_overlap_vs_float32": round(_mean_recall(doc_ids, baseline_ids, top_k=top_k), 6),
            "score_error_vs_float32": _score_error(rows, baseline_score_maps),
        }

    result = {
        "benchmark": "fallback_quantization",
        "baseline": "float32",
        "n_docs": int(n_docs),
        "n_queries": int(n_queries),
        "d_model": int(runtime_dim),
        "top_k": int(top_k),
        "model": model,
        "query_mode": query_mode,
        "seed": int(seed),
        "variants": variants,
    }
    return write_json_result(result, output_path)


def _build_embeddings(
    *,
    n_docs: int,
    n_queries: int,
    d_model: int,
    model: str,
    query_mode: str,
    batch_size: int,
    device: str | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if model == "random":
        if query_mode != "random":
            raise ValueError("query_mode must be 'random' when model='random'")
        return (
            torch.from_numpy(generate_unit_embeddings(n_docs, d_model, seed=seed)),
            torch.from_numpy(generate_unit_embeddings(n_queries, d_model, seed=seed + 1)),
            d_model,
        )

    if query_mode == "random":
        raise ValueError("query_mode must be 'exact' or 'paraphrase' when using a text model")
    if n_queries > n_docs:
        raise ValueError("n_queries must be <= n_docs for text fallback benchmark")
    docs = [f"benchmark document {idx:06d}" for idx in range(n_docs)]
    queries = docs[:n_queries] if query_mode == "exact" else [f"Find information about: {doc}" for doc in docs[:n_queries]]
    encoder, runtime_dim = load_text_encoder(model, d_model, batch_size=batch_size, device=device)
    doc_embeddings = torch.as_tensor(encoder.encode(docs, batch_size=batch_size), dtype=torch.float32)
    query_embeddings = torch.as_tensor(encoder.encode(queries, batch_size=batch_size), dtype=torch.float32)
    return doc_embeddings, query_embeddings, runtime_dim


def _search_all(fallback: DenseVectorFallback, query_embeddings: torch.Tensor, *, top_k: int) -> list[dict]:
    rows = []
    for query in query_embeddings:
        start = time.perf_counter()
        hits = fallback.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "latency_ms": elapsed_ms,
                "hits": [(hit.doc_id, float(hit.score)) for hit in hits],
            }
        )
    return rows


def _mean_recall(observed_rows: list[list[str]], baseline_rows: list[list[str]], *, top_k: int) -> float:
    if not baseline_rows:
        return 0.0
    scores = []
    for observed, baseline in zip(observed_rows, baseline_rows):
        expected = set(baseline[:top_k])
        if not expected:
            scores.append(0.0)
            continue
        scores.append(len(set(observed[:top_k]) & expected) / len(expected))
    return float(np.mean(scores)) if scores else 0.0


def _top1_agreement(observed_rows: list[list[str]], baseline_rows: list[list[str]]) -> float:
    if not baseline_rows:
        return 0.0
    matches = 0
    total = 0
    for observed, baseline in zip(observed_rows, baseline_rows):
        if not observed or not baseline:
            continue
        matches += int(observed[0] == baseline[0])
        total += 1
    return matches / total if total else 0.0


def _score_error(rows: list[dict], baseline_score_maps: list[dict[str, float]]) -> dict[str, float]:
    errors = []
    for row, baseline_scores in zip(rows, baseline_score_maps):
        for doc_id, score in row["hits"]:
            if doc_id in baseline_scores:
                errors.append(abs(float(score) - float(baseline_scores[doc_id])))
    return {
        "mae": round(float(np.mean(errors)) if errors else 0.0, 6),
        "p95": round(percentile(errors, 95), 6),
    }


def _summarize_latency(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark float32 vs int8 vs int4 fallback retrieval quality.")
    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default="random")
    parser.add_argument("--query-mode", choices=["random", "exact", "paraphrase"], default="random")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/fallback_quantization.json"))
    args = parser.parse_args()
    result = run_benchmark(
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        d_model=args.d_model,
        top_k=args.top_k,
        model=args.model,
        query_mode=args.query_mode,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Wrote {result['benchmark']} benchmark to {args.output}")


if __name__ == "__main__":
    main()
