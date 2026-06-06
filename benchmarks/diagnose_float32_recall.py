"""Diagnose float32-only recall@1 for the fine-tuned model vs the base model.

Run from the repo root:
    python benchmarks/diagnose_float32_recall.py

This answers: did full-encoder fine-tuning degrade float32 retrieval quality?
If fine-tuned recall << base recall, the embedding space is compressed and the
two-phase training approach is required.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

FINE_TUNED_PATH = "benchmarks/results/msmarco_full_encoder_100_5ep_stsb_b4_hard_lam8/full_encoder"
BASE_MODEL = "dfrokido/bge-large-e8-snap"
EVAL_LIMIT = 50


def evaluate_float32_recall(model_name_or_path: str, examples) -> dict:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_name_or_path, device=device)

    queries = [e.query for e in examples]
    docs = [e.positive for e in examples]

    q_embs = encoder.encode(queries, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    d_embs = encoder.encode(docs, normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    sim = np.dot(q_embs, d_embs.T)  # [N, N]
    top1 = np.argmax(sim, axis=1)
    n = len(examples)
    correct = int(np.sum(top1 == np.arange(n)))
    diag_cosine = float(np.mean(np.diag(sim)))
    mean_cosine = float(np.mean(sim))

    return {
        "recall_at_1": correct / n,
        "correct": correct,
        "n": n,
        "mean_diagonal_cosine": diag_cosine,
        "mean_all_pairs_cosine": mean_cosine,
    }


def main():
    from latticememory.training import load_msmarco_examples

    print("Loading 50 validation examples from MS MARCO v1.1 ...", flush=True)
    examples = load_msmarco_examples(
        dataset_name="microsoft/ms_marco",
        dataset_config="v1.1",
        split="validation",
        streaming=True,
        min_negatives=1,
        limit=EVAL_LIMIT,
    )
    print(f"  Loaded {len(examples)} examples\n")

    for label, path in [("Base model", BASE_MODEL), ("Fine-tuned model", FINE_TUNED_PATH)]:
        print(f"=== {label}: {path} ===", flush=True)
        r = evaluate_float32_recall(path, examples)
        print(f"  recall@1              = {r['recall_at_1']:.3f}  ({r['correct']}/{r['n']})")
        print(f"  mean diagonal cosine  = {r['mean_diagonal_cosine']:.4f}  (query vs its own positive)")
        print(f"  mean all-pairs cosine = {r['mean_all_pairs_cosine']:.4f}  (compression indicator)")
        print()


if __name__ == "__main__":
    main()
