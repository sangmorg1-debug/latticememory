from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from benchmarks.common import load_text_encoder, write_json_result
from latticememory.dual_encoder import RFSnapDualTextMemory, train_lattice_contrastive_encoder


NATURAL_QA_PAIRS = [
    ("What city is the capital of France?", "Paris is the capital of France."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain on Earth."),
    ("What planet is known as the red planet?", "Mars is known as the red planet."),
    ("What gas do plants absorb?", "Plants absorb carbon dioxide during photosynthesis."),
    ("What is H2O commonly called?", "H2O is commonly called water."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What animal barks?", "A dog is an animal that barks."),
    ("What instrument has keys and pedals?", "A piano has keys and pedals."),
    ("What do bees produce?", "Bees produce honey."),
    ("Where do fish live?", "Fish live in water."),
    ("What is the freezing point of water?", "Water freezes at zero degrees Celsius."),
    ("What organ pumps blood?", "The heart pumps blood through the body."),
    ("What star is closest to Earth?", "The Sun is the closest star to Earth."),
    ("What language is spoken in Spain?", "Spanish is spoken in Spain."),
    ("What do cows produce?", "Cows produce milk."),
    ("What do birds use to fly?", "Birds use wings to fly."),
    ("What do people use to tell time?", "People use clocks to tell time."),
    ("What is the opposite of hot?", "Cold is the opposite of hot."),
    ("What shape has three sides?", "A triangle has three sides."),
]

NATURAL_QA_PARAPHRASE_GROUPS = [
    (
        "Paris is the capital of France.",
        ["What city is France capital?", "Name the capital city of France.", "Which city serves as France capital?"],
        ["What is the capital of France?"],
    ),
    (
        "William Shakespeare wrote Hamlet.",
        ["Who is the author of Hamlet?", "Name the playwright behind Hamlet.", "Hamlet was written by whom?"],
        ["Who wrote Hamlet?"],
    ),
    (
        "Mount Everest is the tallest mountain on Earth.",
        ["Which mountain is tallest?", "Name Earth tallest mountain.", "What is the highest mountain in the world?"],
        ["What mountain is the tallest on Earth?"],
    ),
    (
        "Mars is known as the red planet.",
        ["Which planet is called the red planet?", "Name the red planet.", "What planet has the red planet nickname?"],
        ["What is the red planet?"],
    ),
    (
        "Plants absorb carbon dioxide during photosynthesis.",
        ["What gas do plants take in?", "Which gas is absorbed by plants?", "Plants absorb what gas in photosynthesis?"],
        ["What gas do plants absorb?"],
    ),
    (
        "H2O is commonly called water.",
        ["What is H2O called?", "Name the common name for H2O.", "H2O refers to what substance?"],
        ["What do we call H2O?"],
    ),
    (
        "Leonardo da Vinci painted the Mona Lisa.",
        ["Who painted Mona Lisa?", "Name the artist of Mona Lisa.", "Mona Lisa was painted by whom?"],
        ["Who made the Mona Lisa painting?"],
    ),
    (
        "A dog is an animal that barks.",
        ["What animal barks?", "Name an animal known for barking.", "Which pet makes a bark sound?"],
        ["What animal makes barking sounds?"],
    ),
    (
        "A piano has keys and pedals.",
        ["What instrument has keys and pedals?", "Name a keyed instrument with pedals.", "Which musical instrument includes pedals and keys?"],
        ["What instrument uses both keys and pedals?"],
    ),
    (
        "Bees produce honey.",
        ["What do bees make?", "Name the food produced by bees.", "Bees create what sweet substance?"],
        ["What substance do bees produce?"],
    ),
    (
        "Fish live in water.",
        ["Where do fish live?", "Fish usually live in what environment?", "What habitat do fish occupy?"],
        ["Where are fish found?"],
    ),
    (
        "Water freezes at zero degrees Celsius.",
        ["At what Celsius temperature does water freeze?", "What is water freezing point in Celsius?", "Water becomes ice at what Celsius degree?"],
        ["When does water freeze in Celsius?"],
    ),
    (
        "The heart pumps blood through the body.",
        ["What organ pumps blood?", "Which organ moves blood through the body?", "Name the blood pumping organ."],
        ["What body organ circulates blood?"],
    ),
    (
        "The Sun is the closest star to Earth.",
        ["What star is closest to Earth?", "Name Earth nearest star.", "Which star is nearest our planet?"],
        ["What is the nearest star to Earth?"],
    ),
    (
        "Spanish is spoken in Spain.",
        ["What language is spoken in Spain?", "Name the language people speak in Spain.", "Spain commonly uses what language?"],
        ["Which language is used in Spain?"],
    ),
    (
        "Cows produce milk.",
        ["What do cows produce?", "Cows are known for producing what?", "Name the drink cows provide."],
        ["What product comes from cows?"],
    ),
]


def run_benchmark(
    *,
    model: str = "synthetic",
    d_model: int | None = 1024,
    preset: str = "natural-qa",
    train_count: int = 12,
    heldout_count: int = 4,
    epochs: int = 30,
    adapter_kind: str = "residual_mlp",
    adapter_hidden_multiplier: float = 0.5,
    lr: float = 1e-2,
    lambda_address: float = 10.0,
    lambda_neighborhood: float = 0.5,
    batch_size: int = 8,
    seed: int = 7,
    output_path: str | Path | None = None,
) -> dict:
    train_pairs, heldout_pairs, indexed_docs = _build_dataset(
        preset=preset,
        train_count=train_count,
        heldout_count=heldout_count,
    )
    if train_count < 1:
        raise ValueError("train_count must be positive")
    if heldout_count < 0:
        raise ValueError("heldout_count must be non-negative")
    encoder, runtime_dim = load_text_encoder(model, d_model, batch_size=batch_size)

    start = time.perf_counter()
    train_result = train_lattice_contrastive_encoder(
        base_encoder=encoder,
        pairs=train_pairs,
        d_model=runtime_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=0.05,
        lambda_address=lambda_address,
        lambda_neighborhood=lambda_neighborhood,
        adapter_kind=adapter_kind,
        adapter_hidden_multiplier=adapter_hidden_multiplier,
        encode_batch_size=max(batch_size, 16),
        seed=seed,
    )
    train_seconds = time.perf_counter() - start

    runtime = RFSnapDualTextMemory(
        document_encoder=train_result.dual_encoder.document_encoder,
        query_encoder=train_result.dual_encoder.query_encoder,
        d_model=runtime_dim,
    )
    runtime.add_texts(indexed_docs, doc_ids=[f"doc-{idx}" for idx in range(len(indexed_docs))])

    train_eval = _evaluate_pairs(runtime, train_pairs)
    heldout_eval = _evaluate_pairs(runtime, heldout_pairs)
    result = {
        "benchmark": "routing_adapter",
        "model": model,
        "d_model": runtime_dim,
        "preset": preset,
        "adapter_kind": adapter_kind,
        "adapter_hidden_multiplier": adapter_hidden_multiplier,
        "epochs": epochs,
        "documents_indexed": len(indexed_docs),
        "indexed_documents": indexed_docs,
        "train_seconds": train_seconds,
        "final_train_accuracy_reported_by_trainer": train_result.final_train_accuracy,
        "train_loss_first": train_result.train_loss_history[0] if train_result.train_loss_history else None,
        "train_loss_last": train_result.train_loss_history[-1] if train_result.train_loss_history else None,
        "train": train_eval,
        "heldout": heldout_eval,
        "claim_supported": _claim_supported(train_eval, heldout_eval),
    }
    return write_json_result(result, output_path)


def _build_dataset(
    *,
    preset: str,
    train_count: int,
    heldout_count: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    if preset == "natural-qa-paraphrase-split":
        groups = NATURAL_QA_PARAPHRASE_GROUPS[:train_count]
        if train_count > len(NATURAL_QA_PARAPHRASE_GROUPS):
            raise ValueError("requested train_count exceeds available paraphrase groups")
        train_pairs = [(query, doc) for doc, train_queries, _heldout_queries in groups for query in train_queries]
        heldout_pairs = [
            (query, doc)
            for doc, _train_queries, heldout_queries in groups[:heldout_count]
            for query in heldout_queries
        ]
        return train_pairs, heldout_pairs, [doc for doc, _train_queries, _heldout_queries in groups]

    pairs = _build_pairs(preset, train_count + heldout_count)
    if train_count + heldout_count > len(pairs):
        raise ValueError("requested split exceeds available preset pairs")
    train_pairs = pairs[:train_count]
    heldout_pairs = pairs[train_count : train_count + heldout_count]
    docs = list(dict.fromkeys(doc for _query, doc in train_pairs + heldout_pairs))
    return train_pairs, heldout_pairs, docs


def _build_pairs(preset: str, count: int) -> list[tuple[str, str]]:
    if preset == "natural-qa":
        return NATURAL_QA_PAIRS[:count]
    if preset == "synthetic-copy":
        return [
            (f"training query {idx:06d}", f"training query {idx:06d}")
            for idx in range(count)
        ]
    if preset == "synthetic-paraphrase":
        return [
            (f"Find information about: benchmark document {idx:06d}", f"benchmark document {idx:06d}")
            for idx in range(count)
        ]
    raise ValueError(
        "preset must be 'natural-qa', 'natural-qa-paraphrase-split', "
        "'synthetic-copy', or 'synthetic-paraphrase'"
    )


def _evaluate_pairs(runtime: RFSnapDualTextMemory, pairs: list[tuple[str, str]]) -> dict:
    rows = []
    path_counts: Counter[str] = Counter()
    correct = 0
    for query, expected in pairs:
        result = runtime.retrieve_text(query, top_k=1)
        path_counts[result.path] += 1
        hit_text = result.hits[0].text if result.hits else None
        is_correct = bool(result.hits) and result.path == "lattice_exact" and hit_text == expected
        correct += int(is_correct)
        rows.append(
            {
                "query": query,
                "expected": expected,
                "path": result.path,
                "hit": hit_text,
                "correct_lattice_exact": is_correct,
            }
        )
    count = len(pairs)
    return {
        "count": count,
        "correct_lattice_exact": correct,
        "lattice_exact_accuracy": correct / count if count else 0.0,
        "path_counts": {path: path_counts.get(path, 0) for path in ["lattice_exact", "lattice_hamming", "fallback", "miss"]},
        "rows": rows,
    }


def _claim_supported(train_eval: dict, heldout_eval: dict) -> str:
    if heldout_eval["count"] > 0 and heldout_eval["lattice_exact_accuracy"] == 1.0:
        return "heldout lattice_exact routing"
    if train_eval["lattice_exact_accuracy"] == 1.0:
        return "trained-pair lattice_exact routing only; heldout generalization not shown"
    return "no lattice_exact routing shown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate lattice-native routing adapters.")
    parser.add_argument("--model", default="synthetic")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument(
        "--preset",
        choices=["natural-qa", "natural-qa-paraphrase-split", "synthetic-copy", "synthetic-paraphrase"],
        default="natural-qa",
    )
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-kind", choices=["linear", "residual_mlp"], default="residual_mlp")
    parser.add_argument("--adapter-hidden-multiplier", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-address", type=float, default=10.0)
    parser.add_argument("--lambda-neighborhood", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/routing_adapter.json"))
    args = parser.parse_args()
    result = run_benchmark(
        model=args.model,
        d_model=args.d_model,
        preset=args.preset,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
        epochs=args.epochs,
        adapter_kind=args.adapter_kind,
        adapter_hidden_multiplier=args.adapter_hidden_multiplier,
        lr=args.lr,
        lambda_address=args.lambda_address,
        lambda_neighborhood=args.lambda_neighborhood,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Wrote {result['benchmark']} benchmark to {args.output}")


if __name__ == "__main__":
    main()
