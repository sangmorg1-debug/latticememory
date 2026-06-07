"""Canonical-key projection training for exact-snap experiments.

This benchmark keeps the sentence encoder frozen and trains a small projection
head over embeddings. The projection is supervised to map every same-intent
prompt to the canonical prompt's E8 key while hard near-miss pairs are pushed
away from each other's canonical keys.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_hamming_router import load_pairs
from benchmarks.benchmark_recall_zero_fp import recall_at_fp_budgets
from latticememory.rag.e8_retriever import E8LatticeDB


def build_intent_lookup(source: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for intent in source["intents"]:
        intent_id = intent["intent_id"]
        lookup[intent["canonical"]] = intent_id
        for paraphrase in intent.get("paraphrases", []):
            lookup[paraphrase] = intent_id
    return lookup


def build_positive_rows(
    source: dict[str, Any],
    *,
    train_per_intent: int = 5,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    heldout_rows: list[dict[str, str]] = []
    for intent in source["intents"]:
        intent_id = intent["intent_id"]
        canonical = intent["canonical"]
        train_rows.append({"intent_id": intent_id, "prompt": canonical, "target": canonical})
        for paraphrase in intent.get("paraphrases", [])[:train_per_intent]:
            train_rows.append({"intent_id": intent_id, "prompt": paraphrase, "target": canonical})
        for paraphrase in intent.get("paraphrases", [])[train_per_intent:]:
            heldout_rows.append({"intent_id": intent_id, "prompt": paraphrase, "target": canonical})
    return train_rows, heldout_rows


class IdentityProjection(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1)


def _keys_for_embeddings(embeddings: np.ndarray, *, lattice: E8LatticeDB) -> np.ndarray:
    keys = [
        np.frombuffer(lattice._quantize_to_indices(torch.tensor(emb, dtype=torch.float32)), dtype=np.uint8).copy()
        for emb in embeddings
    ]
    return np.stack(keys)


def _target_keys_for_texts(
    *,
    text_to_embedding: dict[str, np.ndarray],
    target_texts: list[str],
    lattice: E8LatticeDB,
) -> torch.Tensor:
    embs = np.stack([text_to_embedding[text] for text in target_texts])
    return torch.tensor(_keys_for_embeddings(embs, lattice=lattice), dtype=torch.long)


def _block_logits(projected: torch.Tensor, codebook: torch.Tensor, *, temperature: float) -> torch.Tensor:
    n_blocks = projected.shape[1] // 8
    blocks = F.normalize(projected.reshape(projected.shape[0], n_blocks, 8), dim=-1)
    code_vectors = F.normalize(codebook.to(projected.device), dim=-1)
    return torch.einsum("bnd,kd->bnk", blocks, code_vectors) / temperature


def _address_ce(projected: torch.Tensor, target_keys: torch.Tensor, codebook: torch.Tensor, *, temperature: float) -> torch.Tensor:
    logits = _block_logits(projected, codebook, temperature=temperature)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_keys.to(projected.device).reshape(-1))


def _expected_hamming_to_keys(
    projected: torch.Tensor,
    target_keys: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    probabilities = F.softmax(_block_logits(projected, codebook, temperature=temperature), dim=-1)
    target_probability = probabilities.gather(2, target_keys.to(projected.device).unsqueeze(-1)).squeeze(-1)
    return (1.0 - target_probability).sum(dim=-1)


def _expected_pair_hamming(projected_pairs: torch.Tensor, codebook: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if projected_pairs.shape[0] % 2 != 0:
        raise ValueError("projected_pairs must contain an even number of rows")
    probabilities = F.softmax(_block_logits(projected_pairs, codebook, temperature=temperature), dim=-1)
    left = probabilities[0::2]
    right = probabilities[1::2]
    same_probability = (left * right).sum(dim=-1)
    return (1.0 - same_probability).sum(dim=-1)


def _hamming_to_target(projected: np.ndarray, target_keys: np.ndarray, *, lattice: E8LatticeDB) -> list[int]:
    keys = _keys_for_embeddings(projected, lattice=lattice)
    return [int(np.sum(keys[i] != target_keys[i])) for i in range(len(keys))]


def _pair_hamming(projected_pairs: np.ndarray, *, lattice: E8LatticeDB) -> list[int]:
    keys = _keys_for_embeddings(projected_pairs, lattice=lattice)
    return [int(np.sum(keys[2 * i] != keys[2 * i + 1])) for i in range(len(keys) // 2)]


def evaluate_projection(
    projection: IdentityProjection,
    *,
    text_to_embedding: dict[str, np.ndarray],
    heldout_rows: list[dict[str, str]],
    near_miss_pairs: list[tuple[str, str]],
    lattice: E8LatticeDB,
    threshold: int,
) -> dict[str, Any]:
    projection.eval()
    with torch.no_grad():
        heldout_inputs = torch.tensor(np.stack([text_to_embedding[row["prompt"]] for row in heldout_rows]), dtype=torch.float32)
        heldout_projected = projection(heldout_inputs).cpu().numpy()
        heldout_target_keys = _target_keys_for_texts(
            text_to_embedding=text_to_embedding,
            target_texts=[row["target"] for row in heldout_rows],
            lattice=lattice,
        ).numpy()
        paraphrase_dists = _hamming_to_target(heldout_projected, heldout_target_keys, lattice=lattice)

        pair_texts = [text for pair in near_miss_pairs for text in pair]
        pair_inputs = torch.tensor(np.stack([text_to_embedding[text] for text in pair_texts]), dtype=torch.float32)
        pair_projected = projection(pair_inputs).cpu().numpy()
        near_dists = _pair_hamming(pair_projected, lattice=lattice)

    budgets = recall_at_fp_budgets(
        paraphrase_dists=paraphrase_dists,
        near_miss_dists=near_dists,
        fp_budgets=[0.0, 0.001, 0.01],
        max_threshold=lattice.d_model // 8,
    )
    return {
        "paraphrase_hamming_mean": round(float(np.mean(paraphrase_dists)), 3) if paraphrase_dists else 0.0,
        "near_miss_hamming_mean": round(float(np.mean(near_dists)), 3) if near_dists else 0.0,
        "exact_same_cell_rate": round(sum(1 for dist in paraphrase_dists if dist == 0) / len(paraphrase_dists), 4),
        "recall_at_threshold": round(sum(1 for dist in paraphrase_dists if dist <= threshold) / len(paraphrase_dists), 4),
        "near_miss_confusion_at_threshold": round(sum(1 for dist in near_dists if dist <= threshold) / len(near_dists), 4),
        "budget_metrics": budgets,
    }


def run_experiment(
    *,
    source_path: str | Path,
    train_near_misses_path: str | Path,
    eval_near_misses_path: str | Path,
    model: str,
    output_path: str | Path,
    threshold: int = 102,
    train_per_intent: int = 5,
    epochs: int = 30,
    lr: float = 1e-4,
    lambda_near: float = 0.2,
    near_margin: float = 112.0,
    lambda_preserve: float = 0.05,
    temperature: float = 0.05,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    train_rows, heldout_rows = build_positive_rows(source, train_per_intent=train_per_intent)
    train_near_miss_pairs = load_pairs(str(train_near_misses_path), key="near_misses")
    eval_near_miss_pairs = load_pairs(str(eval_near_misses_path), key="near_misses")
    lattice = E8LatticeDB(d_model=1024)
    encoder = SentenceTransformer(model)

    all_texts = sorted({
        row["prompt"] for row in train_rows + heldout_rows
    } | {
        row["target"] for row in train_rows + heldout_rows
    } | {
        text for pair in train_near_miss_pairs + eval_near_miss_pairs for text in pair
    })
    embeddings = encoder.encode(all_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    text_to_embedding = {text: emb.astype(np.float32) for text, emb in zip(all_texts, embeddings, strict=True)}
    d_model = embeddings.shape[1]
    lattice = E8LatticeDB(d_model=d_model)
    codebook = lattice._codebook.float()

    projection = IdentityProjection(d_model)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=1e-4)
    train_inputs = torch.tensor(np.stack([text_to_embedding[row["prompt"]] for row in train_rows]), dtype=torch.float32)
    train_targets = _target_keys_for_texts(
        text_to_embedding=text_to_embedding,
        target_texts=[row["target"] for row in train_rows],
        lattice=lattice,
    )
    near_texts = [text for pair in train_near_miss_pairs for text in pair]
    near_inputs = torch.tensor(np.stack([text_to_embedding[text] for text in near_texts]), dtype=torch.float32)

    baseline = evaluate_projection(
        projection,
        text_to_embedding=text_to_embedding,
        heldout_rows=heldout_rows,
        near_miss_pairs=eval_near_miss_pairs,
        lattice=lattice,
        threshold=threshold,
    )
    history = []
    for epoch in range(1, epochs + 1):
        projection.train()
        optimizer.zero_grad()
        projected = projection(train_inputs)
        address_loss = _address_ce(projected, train_targets, codebook, temperature=temperature)
        near_projected = projection(near_inputs)
        near_expected_hamming = _expected_pair_hamming(
            near_projected,
            codebook,
            temperature=temperature,
        )
        near_loss = F.relu(near_margin - near_expected_hamming).mean()
        preserve_loss = F.mse_loss(projected, train_inputs)
        loss = address_loss + lambda_near * near_loss + lambda_preserve * preserve_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            metrics = evaluate_projection(
                projection,
                text_to_embedding=text_to_embedding,
                heldout_rows=heldout_rows,
                near_miss_pairs=eval_near_miss_pairs,
                lattice=lattice,
                threshold=threshold,
            )
            metrics.update({
                "epoch": epoch,
                "loss": round(float(loss.item()), 6),
                "address_loss": round(float(address_loss.item()), 6),
                "near_loss": round(float(near_loss.item()), 6),
                "preserve_loss": round(float(preserve_loss.item()), 6),
            })
            history.append(metrics)

    final = evaluate_projection(
        projection,
        text_to_embedding=text_to_embedding,
        heldout_rows=heldout_rows,
        near_miss_pairs=eval_near_miss_pairs,
        lattice=lattice,
        threshold=threshold,
    )
    report = {
        "artifact_type": "latticememory_canonical_key_projection_training",
        "artifact_version": 1,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "model": model,
        "d_model": d_model,
        "threshold": threshold,
        "train_per_intent": train_per_intent,
        "epochs": epochs,
        "lr": lr,
        "lambda_near": lambda_near,
        "near_margin": near_margin,
        "lambda_preserve": lambda_preserve,
        "baseline": baseline,
        "final": final,
        "history": history,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a projection head toward canonical E8 keys")
    parser.add_argument("--source", default="benchmarks/demo_data/hard_near_miss_challenge/hard_near_miss_source.json")
    parser.add_argument("--train-near-misses", default="benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json")
    parser.add_argument("--eval-near-misses", default="benchmarks/demo_data/hard_near_miss_challenge/heldout_near_misses.json")
    parser.add_argument("--model", default="dfrokido/bge-large-e8-snap")
    parser.add_argument("--threshold", type=int, default=102)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-near", type=float, default=0.2)
    parser.add_argument("--near-margin", type=float, default=112.0)
    parser.add_argument("--output", default="benchmarks/results/canonical_key_training.json")
    args = parser.parse_args()
    report = run_experiment(
        source_path=args.source,
        train_near_misses_path=args.train_near_misses,
        eval_near_misses_path=args.eval_near_misses,
        model=args.model,
        output_path=args.output,
        threshold=args.threshold,
        epochs=args.epochs,
        lr=args.lr,
        lambda_near=args.lambda_near,
        near_margin=args.near_margin,
    )
    print(json.dumps({
        "output": args.output,
        "baseline": report["baseline"],
        "final": report["final"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
