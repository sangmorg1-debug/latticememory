from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from latticememory.dual_encoder import (
    ContrastiveTrainResult,
    LatticeDualEncoder,
    RFSnapDualTextMemory,
    TextEncoder,
    _e8_address_cross_entropy,
    _e8_expected_hamming_loss,
    train_lattice_contrastive_encoder,
)
from latticememory.rag.e8_retriever import E8LatticeDB


@dataclass(frozen=True)
class RoutingTrainingExample:
    query: str
    positive: str
    negatives: list[str]


@dataclass(frozen=True)
class E8RoutingLossOutput:
    total: torch.Tensor
    contrastive: torch.Tensor
    address: torch.Tensor
    hamming: torch.Tensor
    negative: torch.Tensor


@dataclass(frozen=True)
class LatticeRoutingTrainResult:
    dual_encoder: LatticeDualEncoder
    train_result: ContrastiveTrainResult
    examples_seen: int
    negatives_seen: int
    dataset_name: str | None = None
    dataset_config: str | None = None
    train_split: str | None = None


def build_msmarco_examples(
    rows: Iterable[dict],
    *,
    min_negatives: int = 0,
    limit: int | None = None,
) -> list[RoutingTrainingExample]:
    """Convert MS MARCO QA or passage-ranking rows into routing examples."""

    examples: list[RoutingTrainingExample] = []
    for row in rows:
        for example in _examples_from_row(row, min_negatives=min_negatives):
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def load_msmarco_examples(
    *,
    dataset_name: str = "microsoft/ms_marco",
    dataset_config: str = "v1.1",
    split: str = "train",
    streaming: bool = True,
    min_negatives: int = 1,
    limit: int | None = None,
) -> list[RoutingTrainingExample]:
    """Load real MS MARCO examples through Hugging Face datasets.

    The default `microsoft/ms_marco` rows contain a query plus candidate passages
    with `is_selected` labels. The `sentence-transformers/msmarco` triplets
    subset is also accepted when its rows expose query/positive/negative text.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - covered only without optional dep
        raise ImportError(
            "datasets is required for MS MARCO loading. Install with: "
            "pip install 'latticememory[hf]'"
        ) from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    return build_msmarco_examples(dataset, min_negatives=min_negatives, limit=limit)


class E8RoutingLoss(torch.nn.Module):
    """Contrastive retrieval loss plus direct E8 cell-address supervision."""

    def __init__(
        self,
        *,
        d_model: int,
        temperature: float = 0.05,
        lambda_address: float = 1.0,
        lambda_hamming: float = 0.0,
        lambda_negative: float = 1.0,
    ):
        super().__init__()
        if d_model <= 0 or d_model % 8 != 0:
            raise ValueError("d_model must be positive and divisible by 8")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        lattice = E8LatticeDB(d_model=d_model)
        self.d_model = int(d_model)
        self.temperature = float(temperature)
        self.lambda_address = float(lambda_address)
        self.lambda_hamming = float(lambda_hamming)
        self.lambda_negative = float(lambda_negative)
        self.register_buffer("codebook", lattice._codebook.float(), persistent=False)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
    ) -> E8RoutingLossOutput:
        if query_embeddings.shape != positive_embeddings.shape:
            raise ValueError("query_embeddings and positive_embeddings must have the same shape")
        if query_embeddings.dim() != 2 or query_embeddings.shape[1] != self.d_model:
            raise ValueError(f"embeddings must have shape [batch, {self.d_model}]")

        query_norm = F.normalize(query_embeddings, dim=-1)
        positive_norm = F.normalize(positive_embeddings, dim=-1)
        logits = (query_norm @ positive_norm.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = F.cross_entropy(logits, labels)

        target_keys = self._target_keys(positive_embeddings.detach()).to(query_embeddings.device)
        address = _e8_address_cross_entropy(
            adapted_queries=query_embeddings,
            target_keys=target_keys,
            codebook=self.codebook,
            temperature=self.temperature,
        )
        hamming = _e8_expected_hamming_loss(
            adapted_queries=query_embeddings,
            target_keys=target_keys,
            codebook=self.codebook,
            temperature=self.temperature,
        )
        negative = self._negative_loss(query_embeddings, positive_embeddings, negative_embeddings)
        total = (
            contrastive
            + (self.lambda_address * address)
            + (self.lambda_hamming * hamming)
            + (self.lambda_negative * negative)
        )
        return E8RoutingLossOutput(
            total=total,
            contrastive=contrastive,
            address=address,
            hamming=hamming,
            negative=negative,
        )

    def _target_keys(self, positive_embeddings: torch.Tensor) -> torch.Tensor:
        lattice = E8LatticeDB(d_model=self.d_model)
        return torch.tensor(
            [list(lattice._quantize_to_indices(embedding)) for embedding in positive_embeddings.cpu()],
            dtype=torch.long,
        )

    def _negative_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if negative_embeddings is None or negative_embeddings.numel() == 0:
            return query_embeddings.sum() * 0.0
        if negative_embeddings.dim() == 2:
            negative_embeddings = negative_embeddings.unsqueeze(1)
        if negative_embeddings.dim() != 3 or negative_embeddings.shape[0] != query_embeddings.shape[0]:
            raise ValueError("negative_embeddings must have shape [batch, negatives, d_model]")
        if negative_embeddings.shape[-1] != self.d_model:
            raise ValueError(f"negative embeddings must have last dimension {self.d_model}")

        query_norm = F.normalize(query_embeddings, dim=-1)
        positive_norm = F.normalize(positive_embeddings, dim=-1)
        negative_norm = F.normalize(negative_embeddings.to(query_embeddings.device), dim=-1)
        positive_scores = (query_norm * positive_norm).sum(dim=-1) / self.temperature
        negative_scores = torch.einsum("bd,bnd->bn", query_norm, negative_norm) / self.temperature
        return F.softplus(negative_scores - positive_scores.unsqueeze(1)).mean()


def train_lattice_adapter_from_examples(
    *,
    base_encoder: TextEncoder,
    examples: Sequence[RoutingTrainingExample],
    d_model: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.05,
    lambda_address: float = 1.0,
    lambda_neighborhood: float = 0.0,
    lambda_hard: float = 1.0,
    adapter_kind: str = "residual_mlp",
    adapter_hidden_multiplier: float = 1.0,
    seed: int = 42,
    device: str | torch.device = "cpu",
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    train_split: str | None = None,
) -> LatticeRoutingTrainResult:
    if not examples:
        raise ValueError("examples must not be empty")

    pairs = [(example.query, example.positive) for example in examples]
    hard_negatives = [negative for example in examples for negative in example.negatives]
    train_result = train_lattice_contrastive_encoder(
        base_encoder=base_encoder,
        pairs=pairs,
        hard_negative_texts=hard_negatives,
        d_model=d_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        lambda_address=lambda_address,
        lambda_neighborhood=lambda_neighborhood,
        lambda_hard=lambda_hard,
        adapter_kind=adapter_kind,
        adapter_hidden_multiplier=adapter_hidden_multiplier,
        seed=seed,
        device=device,
    )
    return LatticeRoutingTrainResult(
        dual_encoder=train_result.dual_encoder,
        train_result=train_result,
        examples_seen=len(examples),
        negatives_seen=len(hard_negatives),
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        train_split=train_split,
    )


def evaluate_routing_examples(
    dual_encoder: LatticeDualEncoder,
    examples: Sequence[RoutingTrainingExample],
    *,
    top_k: int = 1,
) -> dict:
    docs = list(dict.fromkeys(example.positive for example in examples))
    runtime = RFSnapDualTextMemory(
        document_encoder=dual_encoder.document_encoder,
        query_encoder=dual_encoder.query_encoder,
        d_model=dual_encoder.d_model,
    )
    runtime.add_texts(docs, doc_ids=[f"doc-{idx}" for idx in range(len(docs))])

    # Compute query-to-positive-doc Hamming distance distribution
    queries = [example.query for example in examples]
    positives = [example.positive for example in examples]
    
    import numpy as np
    hamming_distances = []
    with torch.no_grad():
        doc_embeddings = runtime._encode_documents(positives)
        query_embeddings = runtime._encode_queries(queries)
        
    lattice = runtime.memory.lattice
    for i in range(len(examples)):
        doc_key = lattice._quantize_to_indices(doc_embeddings[i])
        query_key = lattice._quantize_to_indices(query_embeddings[i])
        dist = sum(1 for a, b in zip(query_key, doc_key) if a != b)
        hamming_distances.append(dist)
        
    mean_dist = float(np.mean(hamming_distances)) if hamming_distances else 0.0
    min_dist = int(np.min(hamming_distances)) if hamming_distances else 0
    max_dist = int(np.max(hamming_distances)) if hamming_distances else 0
    p50_dist = float(np.percentile(hamming_distances, 50)) if hamming_distances else 0.0
    p95_dist = float(np.percentile(hamming_distances, 95)) if hamming_distances else 0.0
    p99_dist = float(np.percentile(hamming_distances, 99)) if hamming_distances else 0.0
    
    hist = Counter()
    for d in hamming_distances:
        if d >= 5:
            hist[">=5"] += 1
        else:
            hist[str(d)] += 1
            
    hist_dict = {}
    for k in ["0", "1", "2", "3", "4", ">=5"]:
        hist_dict[k] = hist[k]

    path_counts: Counter[str] = Counter()
    rows = []
    correct_at_1 = 0
    lattice_routed = 0
    for idx, example in enumerate(examples):
        result = runtime.retrieve_text(example.query, top_k=top_k)
        path_counts[result.path] += 1
        if result.path in {"lattice_exact", "lattice_hamming1"}:
            lattice_routed += 1
        hit_text = result.hits[0].text if result.hits else None
        is_correct = hit_text == example.positive
        correct_at_1 += int(is_correct)
        rows.append(
            {
                "query": example.query,
                "expected": example.positive,
                "hit": hit_text,
                "path": result.path,
                "correct_at_1": is_correct,
                "hamming_distance_to_positive": hamming_distances[idx],
            }
        )

    total = len(examples)
    return {
        "total": total,
        "correct_at_1": correct_at_1,
        "recall_at_1": correct_at_1 / total if total else 0.0,
        "lattice_routed": lattice_routed,
        "lattice_route_rate": lattice_routed / total if total else 0.0,
        "path_counts": dict(path_counts),
        "mean_hamming_distance": mean_dist,
        "min_hamming_distance": min_dist,
        "max_hamming_distance": max_dist,
        "p50_hamming_distance": p50_dist,
        "p95_hamming_distance": p95_dist,
        "p99_hamming_distance": p99_dist,
        "hamming_distance_histogram": hist_dict,
        "rows": rows,
    }


def train_and_evaluate_msmarco(
    *,
    model: str = "dfrokido/bge-large-e8-snap",
    dataset_name: str = "microsoft/ms_marco",
    dataset_config: str = "v1.1",
    train_split: str = "train",
    eval_split: str = "validation",
    train_limit: int = 2000,
    eval_limit: int = 200,
    min_negatives: int = 1,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_address: float = 10.0,
    lambda_neighborhood: float = 0.5,
    lambda_hard: float = 1.0,
    adapter_kind: str = "residual_mlp",
    adapter_hidden_multiplier: float = 1.0,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> dict:
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model, device=device)
    embedding_dim_fn = getattr(encoder, "get_embedding_dimension", None)
    sentence_dim_fn = getattr(encoder, "get_sentence_embedding_dimension", None)
    d_model = int(
        (embedding_dim_fn() if embedding_dim_fn is not None else None)
        or (sentence_dim_fn() if sentence_dim_fn is not None else None)
        or 0
    )
    if d_model <= 0:
        probe = encoder.encode(["dimension probe"])
        d_model = int(torch.as_tensor(probe).shape[-1])

    train_examples = load_msmarco_examples(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=train_split,
        min_negatives=min_negatives,
        limit=train_limit,
    )
    eval_examples = load_msmarco_examples(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=eval_split,
        min_negatives=min_negatives,
        limit=eval_limit,
    )
    train_result = train_lattice_adapter_from_examples(
        base_encoder=encoder,
        examples=train_examples,
        d_model=d_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_address=lambda_address,
        lambda_neighborhood=lambda_neighborhood,
        lambda_hard=lambda_hard,
        adapter_kind=adapter_kind,
        adapter_hidden_multiplier=adapter_hidden_multiplier,
        device=device,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        train_split=train_split,
    )
    train_metrics = evaluate_routing_examples(train_result.dual_encoder, train_examples, top_k=1)
    eval_metrics = evaluate_routing_examples(train_result.dual_encoder, eval_examples, top_k=1)

    result = {
        "model": model,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "train_split": train_split,
        "eval_split": eval_split,
        "d_model": d_model,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "adapter_kind": adapter_kind,
        "adapter_hidden_multiplier": adapter_hidden_multiplier,
        "epochs": epochs,
        "lambda_address": lambda_address,
        "lambda_neighborhood": lambda_neighborhood,
        "lambda_hard": lambda_hard,
        "examples_seen": train_result.examples_seen,
        "negatives_seen": train_result.negatives_seen,
        "train_loss_history": train_result.train_result.train_loss_history,
        "address_loss_history": train_result.train_result.address_loss_history,
        "hard_loss_history": train_result.train_result.hard_loss_history,
        "final_train_accuracy": train_result.train_result.final_train_accuracy,
        "train": _metrics_without_rows(train_metrics),
        "eval": _metrics_without_rows(eval_metrics),
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        adapter_path = output_path / "query_adapter.pt"
        metrics_path = output_path / "metrics.json"
        train_result.dual_encoder.query_encoder.save_adapter(adapter_path)
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["adapter_path"] = str(adapter_path)
        result["metrics_path"] = str(metrics_path)
    return result


def _examples_from_row(row: dict, *, min_negatives: int) -> list[RoutingTrainingExample]:
    query = _first_text(row, ("query", "query_text", "question"))
    if not query:
        return []

    triplet_positive = _first_text(row, ("positive", "positive_passage", "pos", "pos_text", "passage"))
    triplet_negative = _negative_texts_from_row(row)
    if triplet_positive:
        if len(triplet_negative) < min_negatives:
            return []
        return [RoutingTrainingExample(query=query, positive=triplet_positive, negatives=triplet_negative)]

    passages = row.get("passages")
    if not isinstance(passages, dict):
        return []
    texts = list(passages.get("passage_text") or passages.get("text") or [])
    selected = list(passages.get("is_selected") or [])
    positives = [
        str(text).strip()
        for text, label in zip(texts, selected)
        if str(text).strip() and int(label) == 1
    ]
    negatives = [
        str(text).strip()
        for text, label in zip(texts, selected)
        if str(text).strip() and int(label) == 0
    ]
    if len(negatives) < min_negatives:
        return []
    return [RoutingTrainingExample(query=query, positive=positive, negatives=negatives) for positive in positives]


def _first_text(row: dict, keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _negative_texts_from_row(row: dict) -> list[str]:
    negatives: list[str] = []
    for key in ("negative", "negative_passage", "neg", "neg_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            negatives.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            negatives.extend(str(item).strip() for item in value if str(item).strip())
    return negatives


def _metrics_without_rows(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "rows"}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate LatticeMemory E8 routing on MS MARCO.")
    parser.add_argument("--model", default="dfrokido/bge-large-e8-snap")
    parser.add_argument("--dataset-name", default="microsoft/ms_marco")
    parser.add_argument("--dataset-config", default="v1.1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-address", type=float, default=10.0)
    parser.add_argument("--lambda-neighborhood", type=float, default=0.5)
    parser.add_argument("--lambda-hard", type=float, default=1.0)
    parser.add_argument("--adapter-kind", choices=["linear", "residual_mlp"], default="residual_mlp")
    parser.add_argument("--adapter-hidden-multiplier", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    result = train_and_evaluate_msmarco(
        model=args.model,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_address=args.lambda_address,
        lambda_neighborhood=args.lambda_neighborhood,
        lambda_hard=args.lambda_hard,
        adapter_kind=args.adapter_kind,
        adapter_hidden_multiplier=args.adapter_hidden_multiplier,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
