from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from latticememory.rag.e8_retriever import E8LatticeDB


def summarize_exact_snap_blocks(
    *,
    query_keys: Sequence[Sequence[int]],
    target_keys: Sequence[Sequence[int]],
    target_probabilities: Sequence[Sequence[float]],
    cluster_name: str,
) -> dict:
    if len(query_keys) != len(target_keys) or len(query_keys) != len(target_probabilities):
        raise ValueError("query_keys, target_keys, and target_probabilities must have the same length")
    if not query_keys:
        raise ValueError("at least one key pair is required")

    n_blocks = len(query_keys[0])
    wrong_counter: Counter[int] = Counter()
    probability_by_block: dict[int, list[float]] = defaultdict(list)
    hamming_values: list[int] = []
    correct_counts: list[int] = []

    for q_key, t_key, probs in zip(query_keys, target_keys, target_probabilities):
        if len(q_key) != n_blocks or len(t_key) != n_blocks or len(probs) != n_blocks:
            raise ValueError("all keys and probability rows must have the same block count")
        diff = [idx for idx, (q, t) in enumerate(zip(q_key, t_key)) if int(q) != int(t)]
        wrong_counter.update(diff)
        for idx, prob in enumerate(probs):
            probability_by_block[idx].append(float(prob))
        hamming_values.append(len(diff))
        correct_counts.append(n_blocks - len(diff))

    exact_count = sum(1 for hamming in hamming_values if hamming == 0)
    block_probabilities = [
        (block, float(np.mean(values)))
        for block, values in probability_by_block.items()
    ]
    block_probabilities.sort(key=lambda row: row[1])

    return {
        "cluster_name": cluster_name,
        "n_pairs": len(query_keys),
        "n_blocks": n_blocks,
        "exact_same_key_rate": round(exact_count / len(query_keys), 4),
        "mean_hamming": round(float(np.mean(hamming_values)), 4),
        "mean_correct_blocks": round(float(np.mean(correct_counts)), 4),
        "worst_blocks": [
            {"block": int(block), "mean_target_probability": round(float(prob), 6)}
            for block, prob in block_probabilities[:20]
        ],
        "repeated_wrong_blocks": [
            {"block": int(block), "wrong_count": int(count)}
            for block, count in wrong_counter.most_common(20)
        ],
    }


def _block_target_probabilities(
    embeddings: torch.Tensor,
    target_keys: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    n_blocks = embeddings.shape[1] // 8
    blocks = F.normalize(embeddings.reshape(embeddings.shape[0], n_blocks, 8), dim=-1)
    code_vectors = F.normalize(codebook.to(embeddings.device), dim=-1)
    logits = torch.einsum("bnd,kd->bnk", blocks, code_vectors) / float(temperature)
    probs = F.softmax(logits, dim=-1)
    return probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)


def _keys_for(lattice: E8LatticeDB, rows: torch.Tensor) -> list[list[int]]:
    return [list(lattice._quantize_to_indices(row)) for row in rows]


def run_audit(*, model: str, temperature: float = 0.05, output: str | Path | None = None) -> dict:
    from sentence_transformers import SentenceTransformer
    from examples.train_snap_encoder import VAL_CLUSTERS

    encoder = SentenceTransformer(model)
    d_model = int(encoder.get_sentence_embedding_dimension())
    lattice = E8LatticeDB(d_model=d_model)
    codebook = lattice._codebook.float()

    cluster_reports = []
    for cluster_name, texts in VAL_CLUSTERS.items():
        target = texts[0]
        queries = texts[1:]
        q_emb = torch.tensor(
            encoder.encode(queries, normalize_embeddings=True),
            dtype=torch.float32,
        )
        t_emb = torch.tensor(
            encoder.encode([target] * len(queries), normalize_embeddings=True),
            dtype=torch.float32,
        )
        q_keys = _keys_for(lattice, q_emb)
        t_keys = _keys_for(lattice, t_emb)
        target_key_tensor = torch.tensor(t_keys, dtype=torch.long)
        target_probs = _block_target_probabilities(
            q_emb,
            target_key_tensor,
            codebook,
            temperature,
        )
        cluster_reports.append(
            summarize_exact_snap_blocks(
                query_keys=q_keys,
                target_keys=t_keys,
                target_probabilities=target_probs.tolist(),
                cluster_name=cluster_name,
            )
        )

    report = {
        "artifact_type": "latticememory_exact_snap_block_audit",
        "artifact_version": 1,
        "model": model,
        "d_model": d_model,
        "temperature": temperature,
        "clusters": cluster_reports,
        "mean_exact_same_key_rate": round(float(np.mean([c["exact_same_key_rate"] for c in cluster_reports])), 4),
        "mean_hamming": round(float(np.mean([c["mean_hamming"] for c in cluster_reports])), 4),
        "mean_correct_blocks": round(float(np.mean([c["mean_correct_blocks"] for c in cluster_reports])), 4),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact E8 block failures for validation clusters")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmarks/results/exact_snap_block_audit.json")
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    report = run_audit(model=args.model, temperature=args.temperature, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
