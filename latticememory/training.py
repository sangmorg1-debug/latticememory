from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from latticememory.dual_encoder import (
    ContrastiveTrainResult,
    LatticeDualEncoder,
    RFSnapDualTextMemory,
    TextEncoder,
    _check_dim,
    _e8_address_cross_entropy,
    _e8_expected_hamming_loss,
    _hamming_distance,
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
    near_miss: torch.Tensor
    address_hinge: torch.Tensor
    address_mse: torch.Tensor
    soft_hard: torch.Tensor
    target_cell_probability: torch.Tensor


@dataclass(frozen=True)
class LatticeRoutingTrainResult:
    dual_encoder: LatticeDualEncoder
    train_result: ContrastiveTrainResult
    examples_seen: int
    negatives_seen: int
    dataset_name: str | None = None
    dataset_config: str | None = None
    train_split: str | None = None


@dataclass(frozen=True)
class FullEncoderTrainResult:
    dual_encoder: LatticeDualEncoder
    encoder: TextEncoder
    training_mode: str
    train_loss_history: list[float]
    contrastive_loss_history: list[float]
    address_loss_history: list[float]
    neighborhood_loss_history: list[float]
    negative_loss_history: list[float]
    train_min_hamming_history: list[int]
    train_mean_hamming_history: list[float]
    train_lattice_route_rate_history: list[float]
    epoch_metrics: list[dict]
    epochs_trained: int
    examples_seen: int
    negatives_seen: int
    final_train_accuracy: float
    dataset_name: str | None = None
    dataset_config: str | None = None
    train_split: str | None = None
    address_hinge_loss_history: list[float] = field(default_factory=list)
    address_mse_loss_history: list[float] = field(default_factory=list)


BUILTIN_STS_EXAMPLES: tuple[tuple[str, str, float], ...] = (
    ("A person is playing a piano.", "Someone is performing music on a piano.", 4.8),
    ("A dog is running through grass.", "An animal is moving outdoors.", 3.9),
    ("The capital of France is Paris.", "Paris is the capital city of France.", 5.0),
    ("A man is cooking dinner.", "Someone is preparing a meal.", 4.4),
    ("Children are playing in a park.", "Kids are outside playing.", 4.7),
    ("A woman is reading a book.", "A person reads a novel.", 4.3),
    ("The stock market closed higher today.", "Financial markets ended the day up.", 4.1),
    ("A cat sleeps on a sofa.", "A vehicle is parked near a building.", 0.4),
    ("Two people are riding bicycles.", "People are biking together.", 4.6),
    ("The weather is cold and rainy.", "It is warm and sunny outside.", 0.8),
    ("A scientist looks through a microscope.", "A researcher uses lab equipment.", 4.2),
    ("A football player kicks the ball.", "An athlete plays soccer.", 3.8),
)


def load_sts_examples(
    *,
    source: str = "builtin",
    limit: int | None = None,
    split: str = "validation",
) -> list[tuple[str, str, float]]:
    if source == "builtin":
        examples = list(BUILTIN_STS_EXAMPLES)
    elif source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("datasets is required for HuggingFace STS-B loading") from exc
        rows = load_dataset("glue", "stsb", split=split)
        examples = [
            (str(row["sentence1"]), str(row["sentence2"]), float(row["label"]))
            for row in rows
        ]
    else:
        raise ValueError("source must be 'builtin' or 'hf'")
    return examples[:limit] if limit is not None else examples


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


def build_canonical_cluster_examples(
    examples: Sequence[RoutingTrainingExample],
) -> list[RoutingTrainingExample]:
    """Map positive targets to deterministic canonical nodes per paraphrase component.

    Positive query-document pairs define an undirected graph. Each connected
    component is treated as one paraphrase cluster. The canonical representative
    is the highest-degree node in that component, with alphabetical order as a
    deterministic tie-breaker. Queries and negatives are preserved; only the
    positive target is replaced.
    """
    if not examples:
        return []

    adjacency: dict[str, set[str]] = {}
    for example in examples:
        query = str(example.query)
        positive = str(example.positive)
        adjacency.setdefault(query, set()).add(positive)
        adjacency.setdefault(positive, set()).add(query)

    canonical_by_node: dict[str, str] = {}
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        canonical = sorted(
            component,
            key=lambda node: (-len(adjacency[node]), node),
        )[0]
        for node in component:
            canonical_by_node[node] = canonical

    return [
        RoutingTrainingExample(
            query=example.query,
            positive=canonical_by_node.get(str(example.positive), example.positive),
            negatives=list(example.negatives),
        )
        for example in examples
    ]


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
        lambda_near_miss: float = 0.0,
        near_miss_margin: float = 80.0,
        lambda_address_hinge: float = 0.0,
        address_hinge_margin: float = 0.2,
        lambda_address_mse: float = 0.0,
        lambda_soft_hard: float = 0.0,
        soft_hard_temperature: float = 1.0,
        soft_hard_straight_through: bool = False,
        soft_hard_focal_gamma: float = 0.0,
        soft_hard_top_k_blocks: int = 0,
    ):
        super().__init__()
        if d_model <= 0 or d_model % 8 != 0:
            raise ValueError("d_model must be positive and divisible by 8")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if soft_hard_temperature <= 0:
            raise ValueError("soft_hard_temperature must be > 0")
        if soft_hard_focal_gamma < 0:
            raise ValueError("soft_hard_focal_gamma must be >= 0")
        if soft_hard_top_k_blocks < 0:
            raise ValueError("soft_hard_top_k_blocks must be >= 0")
        lattice = E8LatticeDB(d_model=d_model)
        self.d_model = int(d_model)
        self.temperature = float(temperature)
        self.lambda_address = float(lambda_address)
        self.lambda_hamming = float(lambda_hamming)
        self.lambda_negative = float(lambda_negative)
        self.lambda_near_miss = float(lambda_near_miss)
        self.near_miss_margin = float(near_miss_margin)
        self.lambda_address_hinge = float(lambda_address_hinge)
        self.address_hinge_margin = float(address_hinge_margin)
        self.lambda_address_mse = float(lambda_address_mse)
        self.lambda_soft_hard = float(lambda_soft_hard)
        self.soft_hard_temperature = float(soft_hard_temperature)
        self.soft_hard_straight_through = bool(soft_hard_straight_through)
        self.soft_hard_focal_gamma = float(soft_hard_focal_gamma)
        self.soft_hard_top_k_blocks = int(soft_hard_top_k_blocks)
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

        if (self.lambda_address > 0 or 
            self.lambda_hamming > 0 or 
            self.lambda_address_hinge > 0 or 
            self.lambda_address_mse > 0 or
            self.lambda_soft_hard > 0):
            target_keys = self._target_keys(positive_embeddings.detach()).to(query_embeddings.device)
        else:
            target_keys = None

        if self.lambda_address > 0:
            address = _e8_address_cross_entropy(
                adapted_queries=query_embeddings,
                target_keys=target_keys,
                codebook=self.codebook,
                temperature=self.temperature,
            )
        else:
            address = query_embeddings.sum() * 0.0

        if self.lambda_hamming > 0:
            hamming = _e8_expected_hamming_loss(
                adapted_queries=query_embeddings,
                target_keys=target_keys,
                codebook=self.codebook,
                temperature=self.temperature,
            )
        else:
            hamming = query_embeddings.sum() * 0.0

        if self.lambda_address_hinge > 0:
            num_blocks = query_embeddings.shape[1] // 8
            blocks = query_embeddings.reshape(query_embeddings.shape[0], num_blocks, 8)
            block_vectors = F.normalize(blocks, dim=-1)
            code_vectors = F.normalize(self.codebook.to(query_embeddings.device), dim=-1)
            block_logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors) / self.temperature
            target_logits = block_logits.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
            masked_logits = block_logits.clone()
            masked_logits.scatter_(2, target_keys.unsqueeze(-1), -10000.0)
            hardest_negative_logits = masked_logits.max(dim=-1)[0]
            address_hinge = F.relu(hardest_negative_logits - target_logits + self.address_hinge_margin).mean()
        else:
            address_hinge = query_embeddings.sum() * 0.0

        if self.lambda_address_mse > 0:
            num_blocks = query_embeddings.shape[1] // 8
            blocks = query_embeddings.reshape(query_embeddings.shape[0], num_blocks, 8)
            block_vectors = F.normalize(blocks, dim=-1)
            target_vectors = self.codebook[target_keys].to(query_embeddings.device)
            target_vectors_norm = F.normalize(target_vectors, dim=-1)
            address_mse = F.mse_loss(block_vectors, target_vectors_norm)
        else:
            address_mse = query_embeddings.sum() * 0.0

        if self.lambda_soft_hard > 0:
            soft_hard, target_cell_probability = self._soft_hard_loss(
                query_embeddings=query_embeddings,
                target_keys=target_keys,
            )
        else:
            soft_hard = query_embeddings.sum() * 0.0
            target_cell_probability = query_embeddings.sum() * 0.0

        negative = self._negative_loss(query_embeddings, positive_embeddings, negative_embeddings)
        near_miss = self._near_miss_loss(query_embeddings, negative_embeddings)
        total = (
            contrastive
            + (self.lambda_address * address)
            + (self.lambda_hamming * hamming)
            + (self.lambda_negative * negative)
            + (self.lambda_near_miss * near_miss)
            + (self.lambda_address_hinge * address_hinge)
            + (self.lambda_address_mse * address_mse)
            + (self.lambda_soft_hard * soft_hard)
        )
        return E8RoutingLossOutput(
            total=total,
            contrastive=contrastive,
            address=address,
            hamming=hamming,
            negative=negative,
            near_miss=near_miss,
            address_hinge=address_hinge,
            address_mse=address_mse,
            soft_hard=soft_hard,
            target_cell_probability=target_cell_probability,
        )

    def _target_keys(self, positive_embeddings: torch.Tensor) -> torch.Tensor:
        lattice = E8LatticeDB(d_model=self.d_model)
        return torch.tensor(
            [list(lattice._quantize_to_indices(embedding)) for embedding in positive_embeddings.cpu()],
            dtype=torch.long,
        )

    def _soft_hard_loss(
        self,
        *,
        query_embeddings: torch.Tensor,
        target_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_blocks = query_embeddings.shape[1] // 8
        blocks = query_embeddings.reshape(query_embeddings.shape[0], num_blocks, 8)
        block_vectors = F.normalize(blocks, dim=-1)
        code_vectors = F.normalize(self.codebook.to(query_embeddings.device), dim=-1)
        logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors)
        logits = logits / float(self.soft_hard_temperature)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
        per_block_loss = -target_log_probs
        if self.soft_hard_focal_gamma > 0:
            per_block_loss = per_block_loss * (1.0 - target_probs).pow(self.soft_hard_focal_gamma)
        if self.soft_hard_top_k_blocks > 0:
            k = min(int(self.soft_hard_top_k_blocks), per_block_loss.shape[1])
            soft_hard = per_block_loss.topk(k, dim=1).values.mean()
        else:
            soft_hard = per_block_loss.mean()

        if self.soft_hard_straight_through:
            soft_vectors = torch.einsum("bnk,kd->bnd", probs, code_vectors)
            hard_indices = probs.argmax(dim=-1)
            hard_one_hot = F.one_hot(hard_indices, num_classes=code_vectors.shape[0]).to(probs.dtype)
            hard_vectors = torch.einsum("bnk,kd->bnd", hard_one_hot, code_vectors)
            straight_through_vectors = hard_vectors.detach() - soft_vectors.detach() + soft_vectors
            target_vectors = code_vectors[target_keys]
            soft_hard = soft_hard + F.mse_loss(straight_through_vectors, target_vectors)

        return soft_hard, target_probs.mean()

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

    def _near_miss_loss(
        self,
        query_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if negative_embeddings is None or negative_embeddings.numel() == 0 or self.near_miss_margin <= 0:
            return query_embeddings.sum() * 0.0
        if negative_embeddings.dim() == 2:
            negative_embeddings = negative_embeddings.unsqueeze(1)
        if negative_embeddings.dim() != 3 or negative_embeddings.shape[0] != query_embeddings.shape[0]:
            raise ValueError("negative_embeddings must have shape [batch, negatives, d_model]")
        if negative_embeddings.shape[-1] != self.d_model:
            raise ValueError(f"negative embeddings must have last dimension {self.d_model}")

        bsz, n_neg, _ = negative_embeddings.shape
        flat_negative = negative_embeddings.reshape(bsz * n_neg, self.d_model)
        negative_keys = self._target_keys(flat_negative.detach()).to(query_embeddings.device)
        negative_keys = negative_keys.reshape(bsz, n_neg, self.d_model // 8)

        num_blocks = self.d_model // 8
        blocks = query_embeddings.reshape(bsz, num_blocks, 8)
        block_vectors = F.normalize(blocks, dim=-1)
        code_vectors = F.normalize(self.codebook.to(query_embeddings.device), dim=-1)
        logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors) / self.temperature
        probabilities = F.softmax(logits, dim=-1)
        expanded = probabilities.unsqueeze(1).expand(-1, n_neg, -1, -1)
        target_probability = expanded.gather(3, negative_keys.unsqueeze(-1)).squeeze(-1)
        expected_hamming = (1.0 - target_probability).sum(dim=-1)
        return F.relu(self.near_miss_margin - expected_hamming).mean()


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
    sts_examples: Sequence[tuple[str, str, float]] | None = None,
    sts_batch_size: int = 64,
    epoch_metrics_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    train_split: str | None = None,
) -> LatticeRoutingTrainResult:
    if not examples:
        raise ValueError("examples must not be empty")

    pairs = [(example.query, example.positive) for example in examples]
    hard_negatives = [negative for example in examples for negative in example.negatives] if lambda_hard > 0 else []
    epoch_evaluator = None
    if sts_examples:
        epoch_evaluator = _build_sts_epoch_evaluator(
            base_encoder=base_encoder,
            sts_examples=sts_examples,
            d_model=d_model,
            device=device,
            batch_size=sts_batch_size,
            epoch_metrics_path=epoch_metrics_path,
        )
    train_result = train_lattice_contrastive_encoder(
        base_encoder=base_encoder,
        pairs=pairs,
        hard_negative_texts=hard_negatives or None,
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
        epoch_evaluator=epoch_evaluator,
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


def train_full_encoder_from_examples(
    *,
    encoder: TextEncoder,
    examples: Sequence[RoutingTrainingExample],
    d_model: int,
    epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    temperature: float = 0.05,
    lambda_address: float = 50.0,
    lambda_neighborhood: float = 1.0,
    lambda_hard: float = 1.0,
    seed: int = 42,
    device: str | torch.device = "cpu",
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    sts_examples: Sequence[tuple[str, str, float]] | None = None,
    sts_batch_size: int = 64,
    epoch_metrics_path: str | Path | None = None,
    progress_metrics_path: str | Path | None = None,
    log_every_batches: int = 0,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    train_split: str | None = None,
    checkpoint_dir: str | Path | None = None,
    lambda_address_hinge: float = 0.0,
    address_hinge_margin: float = 0.2,
    lambda_address_mse: float = 0.0,
) -> FullEncoderTrainResult:
    """Fine-tune a trainable text encoder for direct E8 query-to-passage routing.

    Unlike the adapter path, this function never precomputes document target
    keys. Query and passage embeddings are produced live in each batch, and the
    E8 target key is derived from the current passage embedding after detach().
    """

    if not examples:
        raise ValueError("examples must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if log_every_batches < 0:
        raise ValueError("log_every_batches must be >= 0")
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be positive and divisible by 8")

    train_device = torch.device(device)
    torch.manual_seed(seed)
    _move_encoder_to_device(encoder, train_device)
    if gradient_checkpointing:
        _enable_gradient_checkpointing(encoder)

    loss_fn = E8RoutingLoss(
        d_model=d_model,
        temperature=temperature,
        lambda_address=lambda_address,
        lambda_hamming=lambda_neighborhood,
        lambda_negative=lambda_hard,
        lambda_address_hinge=lambda_address_hinge,
        address_hinge_margin=address_hinge_margin,
        lambda_address_mse=lambda_address_mse,
    ).to(train_device)
    parameters = [parameter for parameter in _encoder_parameters(encoder) if parameter.requires_grad]
    if not parameters:
        raise ValueError("encoder has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    use_amp = bool(fp16 and train_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = len(examples)
    negatives_seen = sum(len(example.negatives) for example in examples) if lambda_hard > 0 else 0

    epoch_evaluator = None
    if sts_examples:
        epoch_evaluator = _build_full_encoder_sts_epoch_evaluator(
            encoder=encoder,
            sts_examples=sts_examples,
            d_model=d_model,
            device=train_device,
            batch_size=sts_batch_size,
            epoch_metrics_path=epoch_metrics_path,
        )
    progress_path = Path(progress_metrics_path) if progress_metrics_path is not None else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    train_loss_history: list[float] = []
    contrastive_loss_history: list[float] = []
    address_loss_history: list[float] = []
    neighborhood_loss_history: list[float] = []
    negative_loss_history: list[float] = []
    address_hinge_loss_history: list[float] = []
    address_mse_loss_history: list[float] = []
    train_min_hamming_history: list[int] = []
    train_mean_hamming_history: list[float] = []
    train_lattice_route_rate_history: list[float] = []
    epoch_metrics: list[dict] = []

    for epoch in range(epochs):
        _set_encoder_train(encoder)
        optimizer.zero_grad(set_to_none=True)
        perm = torch.randperm(n, generator=rng)
        total_batches = int((n + batch_size - 1) // batch_size)
        epoch_started = time.perf_counter()
        epoch_losses: list[float] = []
        epoch_contrastive: list[float] = []
        epoch_address: list[float] = []
        epoch_hamming: list[float] = []
        epoch_negative: list[float] = []
        epoch_address_hinge: list[float] = []
        epoch_address_mse: list[float] = []

        for step, start in enumerate(range(0, n, batch_size), start=1):
            batch_idx = perm[start : start + batch_size]
            batch_examples = [examples[int(idx)] for idx in batch_idx.tolist()]
            queries = [example.query for example in batch_examples]
            positives = [example.positive for example in batch_examples]
            negative_texts = _select_batch_negatives(batch_examples, epoch=epoch) if lambda_hard > 0 else []

            with torch.autocast(device_type=train_device.type, dtype=torch.float16, enabled=use_amp):
                query_embeddings = _encode_trainable_texts(
                    encoder,
                    queries,
                    device=train_device,
                    d_model=d_model,
                )
                positive_embeddings = _encode_trainable_texts(
                    encoder,
                    positives,
                    device=train_device,
                    d_model=d_model,
                )
                negative_embeddings = (
                    _encode_trainable_texts(
                        encoder,
                        negative_texts,
                        device=train_device,
                        d_model=d_model,
                    ).unsqueeze(1)
                    if negative_texts
                    else None
                )
                output = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)
                scaled_loss = output.total / gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            should_step = step % gradient_accumulation_steps == 0 or start + batch_size >= n
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.append(float(output.total.detach()))
            epoch_contrastive.append(float(output.contrastive.detach()))
            epoch_address.append(float(output.address.detach()))
            epoch_hamming.append(float(output.hamming.detach()))
            epoch_negative.append(float(output.negative.detach()))
            epoch_address_hinge.append(float(output.address_hinge.detach()))
            epoch_address_mse.append(float(output.address_mse.detach()))
            if log_every_batches and (step % log_every_batches == 0 or step == total_batches):
                elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                batches_per_sec = step / elapsed
                row = {
                    "metric": "train_progress",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "batch": step,
                    "total_batches": total_batches,
                    "elapsed_sec": round(elapsed, 3),
                    "batches_per_sec": round(batches_per_sec, 6),
                    "eta_epoch_sec": round((total_batches - step) / batches_per_sec, 3)
                    if batches_per_sec > 0
                    else 0.0,
                    "loss": round(float(output.total.detach()), 6),
                    "address_loss": round(float(output.address.detach()), 6),
                    "hamming_loss": round(float(output.hamming.detach()), 6),
                }
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                if progress_path is not None:
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")

        train_loss_history.append(_mean(epoch_losses))
        contrastive_loss_history.append(_mean(epoch_contrastive))
        address_loss_history.append(_mean(epoch_address))
        neighborhood_loss_history.append(_mean(epoch_hamming))
        negative_loss_history.append(_mean(epoch_negative))
        address_hinge_loss_history.append(_mean(epoch_address_hinge))
        address_mse_loss_history.append(_mean(epoch_address_mse))

        hamming_summary = _routing_hamming_summary_for_encoder(
            encoder=encoder,
            examples=examples,
            d_model=d_model,
            batch_size=sts_batch_size,
        )
        train_min_hamming_history.append(int(hamming_summary["min_hamming_distance"]))
        train_mean_hamming_history.append(float(hamming_summary["mean_hamming_distance"]))
        train_lattice_route_rate_history.append(float(hamming_summary["lattice_route_rate"]))

        if epoch_evaluator is not None:
            _set_encoder_eval(encoder)
            with torch.no_grad():
                epoch_metrics.append(epoch_evaluator(epoch + 1))

        if checkpoint_dir is not None:
            _set_encoder_eval(encoder)
            ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch + 1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            save = getattr(encoder, "save", None)
            if callable(save):
                save(str(ckpt_path))
                ckpt_row = {
                    "metric": "checkpoint",
                    "epoch": epoch + 1,
                    "path": str(ckpt_path),
                    "mean_hamming": train_mean_hamming_history[-1] if train_mean_hamming_history else None,
                    "train_loss": train_loss_history[-1] if train_loss_history else None,
                }
                print(json.dumps(ckpt_row, sort_keys=True), flush=True)

    _set_encoder_eval(encoder)
    dual = LatticeDualEncoder(
        document_encoder=encoder,
        query_encoder=encoder,
        d_model=d_model,
        training_pairs=n,
        ridge=0.0,
    )
    final_route_rate = train_lattice_route_rate_history[-1] if train_lattice_route_rate_history else 0.0
    return FullEncoderTrainResult(
        dual_encoder=dual,
        encoder=encoder,
        training_mode="full_encoder",
        train_loss_history=train_loss_history,
        contrastive_loss_history=contrastive_loss_history,
        address_loss_history=address_loss_history,
        neighborhood_loss_history=neighborhood_loss_history,
        negative_loss_history=negative_loss_history,
        train_min_hamming_history=train_min_hamming_history,
        train_mean_hamming_history=train_mean_hamming_history,
        train_lattice_route_rate_history=train_lattice_route_rate_history,
        epoch_metrics=epoch_metrics,
        epochs_trained=epochs,
        examples_seen=len(examples),
        negatives_seen=negatives_seen,
        final_train_accuracy=final_route_rate,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        train_split=train_split,
        address_hinge_loss_history=address_hinge_loss_history,
        address_mse_loss_history=address_mse_loss_history,
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
    with torch.no_grad():
        indexed_doc_embeddings = runtime._encode_documents(docs)
    lattice = runtime.memory.lattice
    indexed_doc_keys = [bytes(lattice._quantize_to_indices(embedding)) for embedding in indexed_doc_embeddings]
    key_counts = Counter(indexed_doc_keys)
    collision_key_count = sum(1 for count in key_counts.values() if count > 1)
    documents_in_collision_keys = sum(count for count in key_counts.values() if count > 1)

    # Compute query-to-positive-doc Hamming distance distribution
    queries = [example.query for example in examples]
    positives = [example.positive for example in examples]
    
    import numpy as np
    hamming_distances = []
    with torch.no_grad():
        doc_embeddings = runtime._encode_documents(positives)
        query_embeddings = runtime._encode_queries(queries)
        
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
        "unique_document_keys": len(key_counts),
        "max_documents_per_key": max(key_counts.values()) if key_counts else 0,
        "collision_key_count": collision_key_count,
        "documents_in_collision_keys": documents_in_collision_keys,
        "rows": rows,
    }


def train_and_evaluate_msmarco(
    *,
    training_mode: str = "adapter",
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
    lambda_address: float | None = None,
    lambda_neighborhood: float = 0.5,
    lambda_hard: float = 1.0,
    adapter_kind: str = "residual_mlp",
    adapter_hidden_multiplier: float = 1.0,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    log_every_batches: int = 0,
    sts_eval_each_epoch: bool = False,
    sts_source: str = "builtin",
    sts_limit: int = 64,
    sts_split: str = "validation",
    output_dir: str | Path | None = None,
    checkpoint_every_epoch: bool = False,
    device: str = "auto",
) -> dict:
    from sentence_transformers import SentenceTransformer

    if training_mode not in {"adapter", "full_encoder"}:
        raise ValueError("training_mode must be 'adapter' or 'full_encoder'")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if lambda_address is None:
        lambda_address = 50.0 if training_mode == "full_encoder" else 10.0
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
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    sts_examples = (
        load_sts_examples(source=sts_source, limit=sts_limit, split=sts_split)
        if sts_eval_each_epoch
        else None
    )
    epoch_metrics_path = (output_path / "epoch_metrics.jsonl") if output_path is not None and sts_examples else None
    progress_metrics_path = (output_path / "progress_metrics.jsonl") if output_path is not None else None
    if training_mode == "adapter":
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
            sts_examples=sts_examples,
            sts_batch_size=batch_size,
            epoch_metrics_path=epoch_metrics_path,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            train_split=train_split,
        )
        dual_encoder = train_result.dual_encoder
        train_loss_history = train_result.train_result.train_loss_history
        address_loss_history = train_result.train_result.address_loss_history
        hard_loss_history = train_result.train_result.hard_loss_history
        epoch_metrics = train_result.train_result.epoch_metrics
        final_train_accuracy = train_result.train_result.final_train_accuracy
        examples_seen = train_result.examples_seen
        negatives_seen = train_result.negatives_seen
        extra_result_fields = {
            "adapter_kind": adapter_kind,
            "adapter_hidden_multiplier": adapter_hidden_multiplier,
        }
    else:
        checkpoint_dir = (output_path / "checkpoints") if (checkpoint_every_epoch and output_path is not None) else None
        train_result = train_full_encoder_from_examples(
            encoder=encoder,
            examples=train_examples,
            d_model=d_model,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=lr,
            lambda_address=lambda_address,
            lambda_neighborhood=lambda_neighborhood,
            lambda_hard=lambda_hard,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            device=device,
            sts_examples=sts_examples,
            sts_batch_size=batch_size,
            epoch_metrics_path=epoch_metrics_path,
            progress_metrics_path=progress_metrics_path,
            log_every_batches=log_every_batches,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            train_split=train_split,
            checkpoint_dir=checkpoint_dir,
        )
        dual_encoder = train_result.dual_encoder
        train_loss_history = train_result.train_loss_history
        address_loss_history = train_result.address_loss_history
        hard_loss_history = train_result.negative_loss_history
        epoch_metrics = train_result.epoch_metrics
        final_train_accuracy = train_result.final_train_accuracy
        examples_seen = train_result.examples_seen
        negatives_seen = train_result.negatives_seen
        extra_result_fields = {
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": bool(fp16),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "train_min_hamming_history": train_result.train_min_hamming_history,
            "train_mean_hamming_history": train_result.train_mean_hamming_history,
            "train_lattice_route_rate_history": train_result.train_lattice_route_rate_history,
            "contrastive_loss_history": train_result.contrastive_loss_history,
            "neighborhood_loss_history": train_result.neighborhood_loss_history,
            "log_every_batches": int(log_every_batches),
        }

    train_metrics = evaluate_routing_examples(dual_encoder, train_examples, top_k=1)
    eval_metrics = evaluate_routing_examples(dual_encoder, eval_examples, top_k=1)

    result = {
        "training_mode": training_mode,
        "model": model,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "train_split": train_split,
        "eval_split": eval_split,
        "d_model": d_model,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "epochs": epochs,
        "lambda_address": lambda_address,
        "lambda_neighborhood": lambda_neighborhood,
        "lambda_hard": lambda_hard,
        "examples_seen": examples_seen,
        "negatives_seen": negatives_seen,
        "train_loss_history": train_loss_history,
        "address_loss_history": address_loss_history,
        "hard_loss_history": hard_loss_history,
        "epoch_metrics": epoch_metrics,
        "final_train_accuracy": final_train_accuracy,
        "train": _metrics_without_rows(train_metrics),
        "eval": _metrics_without_rows(eval_metrics),
    }
    result.update(extra_result_fields)
    if output_dir is not None:
        metrics_path = output_path / "metrics.json"
        if training_mode == "adapter":
            adapter_path = output_path / "query_adapter.pt"
            train_result.dual_encoder.query_encoder.save_adapter(adapter_path)
            result["adapter_path"] = str(adapter_path)
        else:
            model_path = output_path / "full_encoder"
            save = getattr(encoder, "save", None)
            if not callable(save):
                raise TypeError("full-encoder training requires encoder.save(path) for output_dir")
            save(str(model_path))
            result["model_path"] = str(model_path)
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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


def _build_sts_epoch_evaluator(
    *,
    base_encoder: TextEncoder,
    sts_examples: Sequence[tuple[str, str, float]],
    d_model: int,
    device: str | torch.device,
    batch_size: int,
    epoch_metrics_path: str | Path | None,
):
    left_texts = [left for left, _right, _score in sts_examples]
    right_texts = [right for _left, right, _score in sts_examples]
    gold = [float(score) / 5.0 for _left, _right, score in sts_examples]
    left_embeddings = _encode_texts(base_encoder, left_texts, batch_size=batch_size)
    right_embeddings = _encode_texts(base_encoder, right_texts, batch_size=batch_size)
    if left_embeddings.shape != right_embeddings.shape:
        raise ValueError("STS left and right embeddings must have the same shape")
    if left_embeddings.dim() != 2 or left_embeddings.shape[1] != d_model:
        raise ValueError(f"STS encoder dim must be {d_model}, got {tuple(left_embeddings.shape)}")
    eval_device = torch.device(device)
    left_embeddings = left_embeddings.to(eval_device)
    right_embeddings = right_embeddings.to(eval_device)
    metrics_path = Path(epoch_metrics_path) if epoch_metrics_path is not None else None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

    def evaluate(epoch: int, adapter: torch.nn.Module) -> dict:
        adapted_left = adapter(left_embeddings)
        predicted = (
            F.normalize(adapted_left, dim=-1)
            * F.normalize(right_embeddings, dim=-1)
        ).sum(dim=-1).detach().cpu().tolist()
        metrics = _sts_metrics(predicted, gold)
        row = {
            "epoch": int(epoch),
            "metric": "sts",
            "source": "builtin_or_hf",
            "count": len(gold),
            **metrics,
        }
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return row

    return evaluate


def _build_full_encoder_sts_epoch_evaluator(
    *,
    encoder: TextEncoder,
    sts_examples: Sequence[tuple[str, str, float]],
    d_model: int,
    device: str | torch.device,
    batch_size: int,
    epoch_metrics_path: str | Path | None,
):
    left_texts = [left for left, _right, _score in sts_examples]
    right_texts = [right for _left, right, _score in sts_examples]
    gold = [float(score) / 5.0 for _left, _right, score in sts_examples]
    eval_device = torch.device(device)
    metrics_path = Path(epoch_metrics_path) if epoch_metrics_path is not None else None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

    def evaluate(epoch: int) -> dict:
        left_embeddings = _encode_trainable_texts(
            encoder,
            left_texts,
            device=eval_device,
            d_model=d_model,
            batch_size=batch_size,
        )
        right_embeddings = _encode_trainable_texts(
            encoder,
            right_texts,
            device=eval_device,
            d_model=d_model,
            batch_size=batch_size,
        )
        predicted = (
            F.normalize(left_embeddings, dim=-1)
            * F.normalize(right_embeddings, dim=-1)
        ).sum(dim=-1).detach().cpu().tolist()
        metrics = _sts_metrics(predicted, gold)
        row = {
            "epoch": int(epoch),
            "metric": "sts",
            "source": "builtin_or_hf",
            "count": len(gold),
            **metrics,
        }
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return row

    return evaluate


def _encode_texts(encoder: TextEncoder, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
    return torch.as_tensor(encoder.encode(list(texts), batch_size=batch_size), dtype=torch.float32)


def _encode_trainable_texts(
    encoder: TextEncoder,
    texts: Sequence[str],
    *,
    device: torch.device,
    d_model: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    text_list = list(texts)
    if not text_list:
        return torch.empty((0, d_model), dtype=torch.float32, device=device)
    tokenize = getattr(encoder, "tokenize", None)
    if tokenize is None:
        raise TypeError("full-encoder training requires an encoder with tokenize() and forward()")
    rows = []
    step = batch_size or len(text_list)
    for start in range(0, len(text_list), step):
        features = tokenize(text_list[start : start + step])
        features = _move_features_to_device(features, device)
        output = encoder(features)  # type: ignore[misc]
        if hasattr(output, "keys") and "sentence_embedding" in output:
            embeddings = output["sentence_embedding"]
        elif isinstance(output, dict):
            embeddings = output.get("sentence_embedding")
        else:
            embeddings = output
        if embeddings is None:
            raise ValueError("encoder forward output must include sentence_embedding")
        embeddings = torch.as_tensor(embeddings)
        _check_dim(embeddings, d_model)
        rows.append(embeddings)
    return torch.cat(rows, dim=0)


def _move_features_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    move = getattr(value, "to", None)
    if callable(move) and not isinstance(value, (dict, list, tuple)):
        return move(device)
    if isinstance(value, dict):
        return {key: _move_features_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_features_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_features_to_device(item, device) for item in value)
    return value


def _move_encoder_to_device(encoder: TextEncoder, device: torch.device) -> None:
    move = getattr(encoder, "to", None)
    if callable(move):
        move(device)


def _set_encoder_train(encoder: TextEncoder) -> None:
    train = getattr(encoder, "train", None)
    if callable(train):
        train()


def _set_encoder_eval(encoder: TextEncoder) -> None:
    eval_fn = getattr(encoder, "eval", None)
    if callable(eval_fn):
        eval_fn()


def _encoder_parameters(encoder: TextEncoder):
    parameters = getattr(encoder, "parameters", None)
    if not callable(parameters):
        return []
    return list(parameters())


def _enable_gradient_checkpointing(encoder: TextEncoder) -> bool:
    enabled = False
    modules = list(encoder.modules()) if callable(getattr(encoder, "modules", None)) else [encoder]
    for module in modules:
        method = getattr(module, "gradient_checkpointing_enable", None)
        if callable(method):
            method()
            enabled = True
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    return enabled


def _select_batch_negatives(examples: Sequence[RoutingTrainingExample], *, epoch: int) -> list[str]:
    negatives = []
    for position, example in enumerate(examples):
        if not example.negatives:
            return []
        negatives.append(example.negatives[(epoch + position) % len(example.negatives)])
    return negatives


def _routing_hamming_summary_for_encoder(
    *,
    encoder: TextEncoder,
    examples: Sequence[RoutingTrainingExample],
    d_model: int,
    batch_size: int,
) -> dict:
    from latticememory.rag.e8_retriever import E8LatticeDB

    queries = [example.query for example in examples]
    positives = [example.positive for example in examples]
    query_embeddings = _encode_texts(encoder, queries, batch_size=batch_size)
    positive_embeddings = _encode_texts(encoder, positives, batch_size=batch_size)
    _check_dim(query_embeddings, d_model)
    _check_dim(positive_embeddings, d_model)
    lattice = E8LatticeDB(d_model=d_model)
    distances = []
    for query_embedding, positive_embedding in zip(query_embeddings, positive_embeddings):
        query_key = lattice._quantize_to_indices(query_embedding)
        positive_key = lattice._quantize_to_indices(positive_embedding)
        distances.append(_hamming_distance(query_key, positive_key))
    routed = sum(1 for distance in distances if distance <= 1)
    return {
        "min_hamming_distance": min(distances) if distances else 0,
        "mean_hamming_distance": _mean([float(distance) for distance in distances]),
        "lattice_route_rate": routed / len(distances) if distances else 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sts_metrics(predicted: Sequence[float], gold: Sequence[float]) -> dict:
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not predicted:
        return {"pearson": 0.0, "spearman": 0.0, "mse": 0.0, "mean_cosine": 0.0}
    pred = [float(value) for value in predicted]
    target = [float(value) for value in gold]
    mse = sum((p - g) ** 2 for p, g in zip(pred, target)) / len(pred)
    return {
        "pearson": round(_pearson(pred, target), 6),
        "spearman": round(_pearson(_average_ranks(pred), _average_ranks(target)), 6),
        "mse": round(float(mse), 6),
        "mean_cosine": round(sum(pred) / len(pred), 6),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n == 0:
        return 0.0
    left_mean = sum(left) / n
    right_mean = sum(right) / n
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = (left_var * right_var) ** 0.5
    return float(numerator / denom) if denom else 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate LatticeMemory E8 routing on MS MARCO.")
    parser.add_argument("--training-mode", choices=["adapter", "full_encoder"], default="adapter")
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
    parser.add_argument("--lambda-address", type=float, default=None)
    parser.add_argument("--lambda-neighborhood", type=float, default=0.5)
    parser.add_argument("--lambda-hard", type=float, default=1.0)
    parser.add_argument("--adapter-kind", choices=["linear", "residual_mlp"], default="residual_mlp")
    parser.add_argument("--adapter-hidden-multiplier", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=0)
    parser.add_argument("--sts-eval-each-epoch", action="store_true")
    parser.add_argument("--sts-source", choices=["builtin", "hf"], default="builtin")
    parser.add_argument("--sts-limit", type=int, default=64)
    parser.add_argument("--sts-split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    result = train_and_evaluate_msmarco(
        training_mode=args.training_mode,
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        log_every_batches=args.log_every_batches,
        sts_eval_each_epoch=args.sts_eval_each_epoch,
        sts_source=args.sts_source,
        sts_limit=args.sts_limit,
        sts_split=args.sts_split,
        output_dir=args.output_dir,
        checkpoint_every_epoch=args.checkpoint_every_epoch,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
