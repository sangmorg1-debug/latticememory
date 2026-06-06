from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
import torch.nn.functional as F

from .memory import MemoryDocument, MemoryQuery, MemoryResult, RFSnapLatticeMemory, VectorFallback


class TextEncoder(Protocol):
    def encode(self, sentences, batch_size: int = 64, **kwargs):
        ...


class LinearAdapterEncoder:
    """Text encoder wrapper with a learned linear adapter on top of base embeddings."""

    def __init__(
        self,
        *,
        base_encoder: TextEncoder,
        weight: torch.Tensor,
        bias: torch.Tensor,
        batch_size: int = 64,
    ):
        self.base_encoder = base_encoder
        self.weight = torch.as_tensor(weight, dtype=torch.float32).detach().cpu()
        self.bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()
        self.batch_size = batch_size
        if self.weight.dim() != 2 or self.weight.shape[0] != self.weight.shape[1]:
            raise ValueError("weight must be a square [d_model, d_model] matrix")
        if self.bias.shape != (self.weight.shape[0],):
            raise ValueError("bias must have shape [d_model]")

    @property
    def d_model(self) -> int:
        return int(self.weight.shape[0])

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        encoded = _encode_with(self.base_encoder, sentences, batch_size=batch_size or self.batch_size)
        adapted = encoded @ self.weight.T + self.bias
        return adapted.cpu().numpy().astype("float32")

    def save_adapter(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "_version": 1,
                "d_model": self.d_model,
                "weight": self.weight,
                "bias": self.bias,
                "batch_size": self.batch_size,
            },
            path,
        )

    @classmethod
    def load_adapter(
        cls,
        *,
        base_encoder: TextEncoder,
        path: str | Path,
        batch_size: int | None = None,
    ) -> "LinearAdapterEncoder":
        state = torch.load(path, weights_only=False)
        version = state.get("_version", 0)
        if version != 1:
            raise ValueError(f"Unsupported LinearAdapterEncoder version {version}")
        return cls(
            base_encoder=base_encoder,
            weight=state["weight"],
            bias=state["bias"],
            batch_size=batch_size or int(state.get("batch_size", 64)),
        )


class ResidualMLPAdapterEncoder:
    """Text encoder wrapper with a learned residual MLP adapter on top of base embeddings."""

    def __init__(
        self,
        *,
        base_encoder: TextEncoder,
        d_model: int,
        hidden_dim: int,
        state_dict: dict[str, torch.Tensor],
        batch_size: int = 64,
    ):
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.base_encoder = base_encoder
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = batch_size
        self.adapter = _ResidualMLPAdapter(self.d_model, self.hidden_dim)
        self.adapter.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32).cpu() for k, v in state_dict.items()})
        self.adapter.eval()

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        encoded = _encode_with(self.base_encoder, sentences, batch_size=batch_size or self.batch_size).float()
        with torch.no_grad():
            adapted = self.adapter(encoded)
        return adapted.cpu().numpy().astype("float32")

    def save_adapter(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "_version": 1,
                "adapter_kind": "residual_mlp",
                "d_model": self.d_model,
                "hidden_dim": self.hidden_dim,
                "state_dict": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                "batch_size": self.batch_size,
            },
            path,
        )

    @classmethod
    def load_adapter(
        cls,
        *,
        base_encoder: TextEncoder,
        path: str | Path,
        batch_size: int | None = None,
    ) -> "ResidualMLPAdapterEncoder":
        state = torch.load(path, weights_only=False)
        version = state.get("_version", 0)
        if version != 1:
            raise ValueError(f"Unsupported ResidualMLPAdapterEncoder version {version}")
        if state.get("adapter_kind") != "residual_mlp":
            raise ValueError("adapter_kind must be 'residual_mlp'")
        return cls(
            base_encoder=base_encoder,
            d_model=int(state["d_model"]),
            hidden_dim=int(state["hidden_dim"]),
            state_dict=state["state_dict"],
            batch_size=batch_size or int(state.get("batch_size", 64)),
        )


class QueryCanonicalizingEncoder:
    """Text encoder wrapper that strips retrieval-instruction prefixes from queries."""

    DEFAULT_PREFIXES = (
        "Find information about:",
        "Answer this question:",
        "Relevant passage for:",
        "Document that answers:",
        "Search result for:",
        "Locate a passage about:",
        "Which document discusses:",
        "Retrieve context for:",
    )

    def __init__(
        self,
        base_encoder: TextEncoder,
        *,
        prefixes: Sequence[str] | None = None,
    ):
        self.base_encoder = base_encoder
        self.prefixes = tuple(prefixes or self.DEFAULT_PREFIXES)

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        is_single = isinstance(sentences, str)
        inputs = [sentences] if is_single else list(sentences)
        canonical = [self.canonicalize(str(text), self.prefixes) for text in inputs]
        return self.base_encoder.encode(canonical, batch_size=batch_size, **kwargs)

    @staticmethod
    def canonicalize(text: str, prefixes: Sequence[str] | None = None) -> str:
        stripped = text.strip()
        for prefix in prefixes or QueryCanonicalizingEncoder.DEFAULT_PREFIXES:
            if stripped.lower().startswith(prefix.lower()):
                return stripped[len(prefix):].strip()
        return stripped


@dataclass(frozen=True)
class LatticeDualEncoder:
    document_encoder: TextEncoder
    query_encoder: TextEncoder
    d_model: int
    training_pairs: int
    ridge: float


class RFSnapDualTextMemory:
    """Text runtime with separate document and query encoders.

    This is the product path for lattice-native retrieval: documents are indexed
    into E8 addresses with one encoder, while queries can use a separately
    trained adapter whose job is to land positives in the same E8 address.
    """

    def __init__(
        self,
        *,
        document_encoder: TextEncoder,
        query_encoder: TextEncoder,
        d_model: int,
        memory: RFSnapLatticeMemory | None = None,
        fallback: VectorFallback | None = None,
        model_id: str | None = None,
        batch_size: int = 64,
        hamming_pool_multiplier: int = 10,
        rerank_mode: str = "cosine",
        lexical_weight: float = 0.0,
        beam_radius: int = 1,
    ):
        self.document_encoder = document_encoder
        self.query_encoder = query_encoder
        self.d_model = d_model
        self.memory = memory or RFSnapLatticeMemory(
            d_model=d_model,
            fallback=fallback,
            hamming_pool_multiplier=hamming_pool_multiplier,
            rerank_mode=rerank_mode,
            lexical_weight=lexical_weight,
            beam_radius=beam_radius,
        )
        if self.memory.d_model != d_model:
            raise ValueError(f"memory d_model {self.memory.d_model} does not match runtime d_model {d_model}")
        self.model_id = model_id
        self.batch_size = batch_size

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        text_list = list(texts)
        ids = list(doc_ids) if doc_ids is not None else [f"doc-{self.memory.num_documents + i}" for i in range(len(text_list))]
        if len(ids) != len(text_list):
            raise ValueError("doc_ids length must match texts length")
        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in text_list]
        if len(metadata_list) != len(text_list):
            raise ValueError("metadatas length must match texts length")

        embeddings = self._encode_documents(text_list, batch_size=batch_size)
        docs = []
        for text, doc_id, metadata, embedding in zip(text_list, ids, metadata_list, embeddings):
            enriched = dict(metadata)
            if dataset is not None:
                enriched.setdefault("dataset", dataset)
            if index_id is not None:
                enriched.setdefault("index_id", index_id)
            docs.append(MemoryDocument(doc_id=doc_id, text=text, embedding=embedding, metadata=enriched))
        self.memory.add_documents(docs)
        return ids

    def retrieve_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        request_id: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        model_id: str | None = None,
        index_id: str | None = None,
        quality_tags: list[str] | None = None,
        batch_size: int | None = None,
    ) -> MemoryResult:
        embedding = self._encode_queries([text], batch_size=batch_size)[0]
        return self.memory.retrieve(
            MemoryQuery(
                text=text,
                embedding=embedding,
                top_k=top_k,
                request_id=request_id,
                product=product,
                dataset=dataset,
                model_id=model_id or self.model_id,
                index_id=index_id,
                quality_tags=list(quality_tags or []),
            )
        )

    def _encode_documents(self, texts: Sequence[str], *, batch_size: int | None = None) -> torch.Tensor:
        embeddings = _encode_with(self.document_encoder, texts, batch_size=batch_size or self.batch_size)
        _check_dim(embeddings, self.d_model)
        return embeddings

    def _encode_queries(self, texts: Sequence[str], *, batch_size: int | None = None) -> torch.Tensor:
        embeddings = _encode_with(self.query_encoder, texts, batch_size=batch_size or self.batch_size)
        _check_dim(embeddings, self.d_model)
        return embeddings


def fit_lattice_dual_encoder(
    *,
    base_encoder: TextEncoder,
    pairs: Sequence[tuple[str, str]],
    d_model: int,
    ridge: float = 1e-4,
    batch_size: int = 64,
) -> LatticeDualEncoder:
    """Fit a query-side linear adapter from labeled query/document pairs.

    The document side remains the base encoder. The query adapter is fit with
    ridge regression to map query embeddings onto their positive document
    embeddings, which makes positives share the document E8 address when the
    pair signal is learnable by a linear map.
    """

    if not pairs:
        raise ValueError("pairs must not be empty")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    queries = [q for q, _doc in pairs]
    docs = [doc for _q, doc in pairs]
    query_embeddings = _encode_with(base_encoder, queries, batch_size=batch_size)
    doc_embeddings = _encode_with(base_encoder, docs, batch_size=batch_size)
    _check_dim(query_embeddings, d_model)
    _check_dim(doc_embeddings, d_model)

    ones = torch.ones(query_embeddings.shape[0], 1, dtype=torch.float32)
    design = torch.cat([query_embeddings.float(), ones], dim=1)
    targets = doc_embeddings.float()
    gram = design.T @ design
    if ridge:
        gram = gram + torch.eye(gram.shape[0], dtype=torch.float32) * float(ridge)
        gram[-1, -1] -= float(ridge)
    rhs = design.T @ targets
    solution = torch.linalg.pinv(gram) @ rhs
    weight = solution[:-1, :].T.contiguous()
    bias = solution[-1, :].contiguous()
    query_encoder = LinearAdapterEncoder(
        base_encoder=base_encoder,
        weight=weight,
        bias=bias,
        batch_size=batch_size,
    )
    return LatticeDualEncoder(
        document_encoder=base_encoder,
        query_encoder=query_encoder,
        d_model=d_model,
        training_pairs=len(pairs),
        ridge=float(ridge),
    )


@dataclass(frozen=True)
class ContrastiveTrainResult:
    dual_encoder: LatticeDualEncoder
    adapter_kind: str
    adapter_hidden_dim: int | None
    train_loss_history: list[float]  # per-epoch mean loss
    easy_loss_history: list[float]
    address_loss_history: list[float]
    neighborhood_loss_history: list[float]
    hard_loss_history: list[float]
    synthetic_hard_loss_history: list[float]
    epochs_trained: int
    final_train_accuracy: float  # fraction of training pairs where query E8 key == doc E8 key
    hard_negative_pairs: int
    synthetic_hard_negative_pairs: int
    epoch_metrics: list[dict]


def train_lattice_contrastive_encoder(
    *,
    base_encoder: TextEncoder,
    pairs: Sequence[tuple[str, str]],  # (query_text, positive_doc_text)
    hard_negative_texts: Sequence[str] | None = None,
    d_model: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.05,
    weight_decay: float = 1e-4,
    encode_batch_size: int = 64,
    seed: int = 42,
    hard_negative_k: int = 2,
    hard_negative_radius: int = 2,
    lambda_address: float = 1.0,
    lambda_neighborhood: float = 0.0,
    lambda_hard: float = 1.0,
    synthetic_hard_negative_k: int = 0,
    synthetic_hard_negative_radius: int = 1,
    lambda_synthetic_hard: float = 0.0,
    adapter_kind: str = "linear",
    adapter_hidden_multiplier: float = 2.0,
    device: str | torch.device = "cpu",
    epoch_evaluator: Callable[[int, torch.nn.Module], dict] | None = None,
) -> ContrastiveTrainResult:
    """Train a query-side linear adapter for discrete E8 routing.

    The objective combines three signals:
      - easy in-batch InfoNCE against positive document embeddings
      - E8 block address cross-entropy against the positive document key
      - optional hard-negative ranking against documents at nearby E8 addresses
    """
    if not pairs:
        raise ValueError("pairs must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if d_model % 8 != 0:
        raise ValueError("d_model must be divisible by 8 for E8 block address loss")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if hard_negative_k < 0:
        raise ValueError("hard_negative_k must be >= 0")
    if hard_negative_radius < 0:
        raise ValueError("hard_negative_radius must be >= 0")
    if synthetic_hard_negative_k < 0:
        raise ValueError("synthetic_hard_negative_k must be >= 0")
    if synthetic_hard_negative_radius < 1:
        raise ValueError("synthetic_hard_negative_radius must be >= 1")
    if adapter_kind not in {"linear", "residual_mlp"}:
        raise ValueError("adapter_kind must be 'linear' or 'residual_mlp'")
    if adapter_hidden_multiplier <= 0:
        raise ValueError("adapter_hidden_multiplier must be > 0")

    train_device = torch.device(device)

    # Step 1: encode all texts upfront with the base encoder (not the adapter)
    query_embs = _encode_with(base_encoder, [q for q, _ in pairs], batch_size=encode_batch_size)  # [N, d_model]
    doc_embs = _encode_with(base_encoder, [d for _, d in pairs], batch_size=encode_batch_size)    # [N, d_model]
    _check_dim(query_embs, d_model)
    _check_dim(doc_embs, d_model)

    from latticememory.rag.e8_retriever import E8LatticeDB

    tmp_db = E8LatticeDB(d_model=d_model)
    target_keys = torch.tensor(
        [list(tmp_db._quantize_to_indices(doc_emb)) for doc_emb in doc_embs],
        dtype=torch.long,
    )
    codebook = tmp_db._codebook.float().to(train_device)

    hard_doc_embs: torch.Tensor | None = None
    hard_negative_indices = [[] for _ in pairs]
    if hard_negative_k and hard_negative_texts:
        hard_texts = list(dict.fromkeys(str(text) for text in hard_negative_texts))
        hard_doc_embs = _encode_with(base_encoder, hard_texts, batch_size=encode_batch_size)
        _check_dim(hard_doc_embs, d_model)
        hard_keys = [tmp_db._quantize_to_indices(emb) for emb in hard_doc_embs]
        positive_texts = [doc for _query, doc in pairs]
        positive_keys = [bytes(key.tolist()) for key in target_keys]
        for pair_idx, (positive_text, positive_key) in enumerate(zip(positive_texts, positive_keys)):
            candidates = []
            for candidate_idx, (candidate_text, candidate_key) in enumerate(zip(hard_texts, hard_keys)):
                if candidate_text == positive_text:
                    continue
                distance = _hamming_distance(positive_key, candidate_key)
                if 0 < distance <= hard_negative_radius:
                    candidates.append((distance, candidate_idx))
            candidates.sort(key=lambda item: item[0])
            hard_negative_indices[pair_idx] = [idx for _distance, idx in candidates[:hard_negative_k]]
    hard_negative_pairs = sum(len(indices) for indices in hard_negative_indices)
    synthetic_negative_keys = _build_synthetic_negative_keys(
        target_keys=target_keys,
        negative_k=synthetic_hard_negative_k,
        radius=synthetic_hard_negative_radius,
    )
    synthetic_hard_negative_pairs = int(target_keys.shape[0] * synthetic_hard_negative_k)
    query_embs = query_embs.to(train_device)
    doc_embs = doc_embs.to(train_device)
    target_keys = target_keys.to(train_device)
    synthetic_negative_keys = synthetic_negative_keys.to(train_device)
    if hard_doc_embs is not None:
        hard_doc_embs = hard_doc_embs.to(train_device)

    # Step 2: initialize adapter near identity
    torch.manual_seed(seed)
    adapter, adapter_hidden_dim = _build_trainable_adapter(
        adapter_kind=adapter_kind,
        d_model=d_model,
        hidden_multiplier=adapter_hidden_multiplier,
    )
    adapter = adapter.to(train_device)

    # Step 3: AdamW optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(pairs)
    train_loss_history: list[float] = []
    easy_loss_history: list[float] = []
    address_loss_history: list[float] = []
    hard_loss_history: list[float] = []
    neighborhood_loss_history: list[float] = []
    synthetic_hard_loss_history: list[float] = []
    epoch_metrics: list[dict] = []
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Step 4: training loop
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=rng)
        epoch_losses: list[float] = []
        epoch_easy_losses: list[float] = []
        epoch_address_losses: list[float] = []
        epoch_neighborhood_losses: list[float] = []
        epoch_hard_losses: list[float] = []
        epoch_synthetic_hard_losses: list[float] = []
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            adapted_queries = adapter(query_embs[batch_idx])  # [B, d_model]
            adapted_q = F.normalize(adapted_queries, dim=-1)
            doc_b = F.normalize(doc_embs[batch_idx], dim=-1)  # [B, d_model]
            sim = (adapted_q @ doc_b.T) / temperature          # [B, B]
            labels = torch.arange(sim.shape[0], device=sim.device)
            easy_loss = F.cross_entropy(sim, labels)
            address_loss = _e8_address_cross_entropy(
                adapted_queries=adapted_queries,
                target_keys=target_keys[batch_idx],
                codebook=codebook,
                temperature=temperature,
            )
            neighborhood_loss = _e8_expected_hamming_loss(
                adapted_queries=adapted_queries,
                target_keys=target_keys[batch_idx],
                codebook=codebook,
                temperature=temperature,
            )
            hard_loss = _hard_negative_loss(
                adapted_queries=adapted_queries,
                positive_docs=doc_embs[batch_idx],
                hard_doc_embs=hard_doc_embs,
                hard_negative_indices=hard_negative_indices,
                batch_indices=batch_idx,
                epoch=epoch,
                temperature=temperature,
            )
            synthetic_hard_loss = _synthetic_address_negative_loss(
                adapted_queries=adapted_queries,
                target_keys=target_keys[batch_idx],
                synthetic_negative_keys=synthetic_negative_keys[batch_idx],
                codebook=codebook,
                temperature=temperature,
            )
            loss = (
                easy_loss
                + (lambda_address * address_loss)
                + (lambda_neighborhood * neighborhood_loss)
                + (lambda_hard * hard_loss)
                + (lambda_synthetic_hard * synthetic_hard_loss)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            epoch_easy_losses.append(float(easy_loss.detach()))
            epoch_address_losses.append(float(address_loss.detach()))
            epoch_neighborhood_losses.append(float(neighborhood_loss.detach()))
            epoch_hard_losses.append(float(hard_loss.detach()))
            epoch_synthetic_hard_losses.append(float(synthetic_hard_loss.detach()))
        train_loss_history.append(float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0)
        easy_loss_history.append(float(sum(epoch_easy_losses) / len(epoch_easy_losses)) if epoch_easy_losses else 0.0)
        address_loss_history.append(float(sum(epoch_address_losses) / len(epoch_address_losses)) if epoch_address_losses else 0.0)
        neighborhood_loss_history.append(
            float(sum(epoch_neighborhood_losses) / len(epoch_neighborhood_losses)) if epoch_neighborhood_losses else 0.0
        )
        hard_loss_history.append(float(sum(epoch_hard_losses) / len(epoch_hard_losses)) if epoch_hard_losses else 0.0)
        synthetic_hard_loss_history.append(
            float(sum(epoch_synthetic_hard_losses) / len(epoch_synthetic_hard_losses))
            if epoch_synthetic_hard_losses
            else 0.0
        )
        if epoch_evaluator is not None:
            adapter.eval()
            with torch.no_grad():
                epoch_metrics.append(epoch_evaluator(epoch + 1, adapter))
            adapter.train()

    # Step 6: compute final train accuracy (exact E8 key match)
    with torch.no_grad():
        adapted_q_all = adapter(query_embs)  # [N, d_model] — raw adapter output, no global normalize
    correct = 0
    for q_emb, d_emb in zip(adapted_q_all, doc_embs):
        qk = tmp_db._quantize_to_indices(q_emb)
        dk = tmp_db._quantize_to_indices(d_emb)
        correct += int(qk == dk)
    final_train_accuracy = correct / n

    # Step 7: build runtime query encoder from trained weights
    if adapter_kind == "linear":
        linear = adapter
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError("linear adapter expected torch.nn.Linear")
        query_encoder = LinearAdapterEncoder(
            base_encoder=base_encoder,
            weight=linear.weight.data.detach().cpu(),
            bias=linear.bias.data.detach().cpu(),
            batch_size=encode_batch_size,
        )
    else:
        query_encoder = ResidualMLPAdapterEncoder(
            base_encoder=base_encoder,
            d_model=d_model,
            hidden_dim=int(adapter_hidden_dim or d_model),
            state_dict={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            batch_size=encode_batch_size,
        )

    dual = LatticeDualEncoder(
        document_encoder=base_encoder,
        query_encoder=query_encoder,
        d_model=d_model,
        training_pairs=n,
        ridge=0.0,
    )

    return ContrastiveTrainResult(
        dual_encoder=dual,
        adapter_kind=adapter_kind,
        adapter_hidden_dim=adapter_hidden_dim,
        train_loss_history=train_loss_history,
        easy_loss_history=easy_loss_history,
        address_loss_history=address_loss_history,
        neighborhood_loss_history=neighborhood_loss_history,
        hard_loss_history=hard_loss_history,
        synthetic_hard_loss_history=synthetic_hard_loss_history,
        epochs_trained=epochs,
        final_train_accuracy=final_train_accuracy,
        hard_negative_pairs=hard_negative_pairs,
        synthetic_hard_negative_pairs=synthetic_hard_negative_pairs,
        epoch_metrics=epoch_metrics,
    )


class _ResidualMLPAdapter(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.fc1 = torch.nn.Linear(d_model, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, d_model)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def _build_trainable_adapter(
    *,
    adapter_kind: str,
    d_model: int,
    hidden_multiplier: float,
) -> tuple[torch.nn.Module, int | None]:
    if adapter_kind == "linear":
        adapter = torch.nn.Linear(d_model, d_model, bias=True)
        torch.nn.init.eye_(adapter.weight)
        torch.nn.init.zeros_(adapter.bias)
        return adapter, None
    if adapter_kind == "residual_mlp":
        hidden_dim = max(1, int(round(d_model * hidden_multiplier)))
        return _ResidualMLPAdapter(d_model, hidden_dim), hidden_dim
    raise ValueError("adapter_kind must be 'linear' or 'residual_mlp'")


def _e8_address_cross_entropy(
    *,
    adapted_queries: torch.Tensor,
    target_keys: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    num_blocks = adapted_queries.shape[1] // 8
    blocks = adapted_queries.reshape(adapted_queries.shape[0], num_blocks, 8)
    block_vectors = F.normalize(blocks, dim=-1)
    code_vectors = F.normalize(codebook.to(adapted_queries.device), dim=-1)
    logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors) / temperature
    return F.cross_entropy(
        logits.reshape(-1, code_vectors.shape[0]),
        target_keys.to(adapted_queries.device).reshape(-1),
    )


def _e8_expected_hamming_loss(
    *,
    adapted_queries: torch.Tensor,
    target_keys: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    num_blocks = adapted_queries.shape[1] // 8
    blocks = adapted_queries.reshape(adapted_queries.shape[0], num_blocks, 8)
    block_vectors = F.normalize(blocks, dim=-1)
    code_vectors = F.normalize(codebook.to(adapted_queries.device), dim=-1)
    logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors) / temperature
    probabilities = F.softmax(logits, dim=-1)
    target_probability = probabilities.gather(2, target_keys.to(adapted_queries.device).unsqueeze(-1)).squeeze(-1)
    return (1.0 - target_probability).sum(dim=-1).mean()


def _synthetic_address_negative_loss(
    *,
    adapted_queries: torch.Tensor,
    target_keys: torch.Tensor,
    synthetic_negative_keys: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if synthetic_negative_keys.shape[1] == 0:
        return adapted_queries.sum() * 0.0

    num_blocks = adapted_queries.shape[1] // 8
    blocks = adapted_queries.reshape(adapted_queries.shape[0], num_blocks, 8)
    block_vectors = F.normalize(blocks, dim=-1)
    code_vectors = F.normalize(codebook.to(adapted_queries.device), dim=-1)
    logits = torch.einsum("bnd,kd->bnk", block_vectors, code_vectors) / temperature

    targets = target_keys.to(adapted_queries.device)
    negatives = synthetic_negative_keys.to(adapted_queries.device)
    positive_scores = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    expanded_logits = logits.unsqueeze(1).expand(-1, negatives.shape[1], -1, -1)
    negative_scores = expanded_logits.gather(3, negatives.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    return F.softplus(negative_scores - positive_scores.unsqueeze(1)).mean()


def _build_synthetic_negative_keys(
    *,
    target_keys: torch.Tensor,
    negative_k: int,
    radius: int,
) -> torch.Tensor:
    num_pairs, num_blocks = target_keys.shape
    negatives = torch.empty((num_pairs, negative_k, num_blocks), dtype=torch.long)
    for pair_idx in range(num_pairs):
        base = target_keys[pair_idx]
        for negative_idx in range(negative_k):
            candidate = base.clone()
            for step in range(radius):
                block_idx = (pair_idx + negative_idx + step) % num_blocks
                candidate[block_idx] = (candidate[block_idx] + 1 + negative_idx + step) % 240
            negatives[pair_idx, negative_idx] = candidate
    return negatives


def _hard_negative_loss(
    *,
    adapted_queries: torch.Tensor,
    positive_docs: torch.Tensor,
    hard_doc_embs: torch.Tensor | None,
    hard_negative_indices: Sequence[Sequence[int]],
    batch_indices: torch.Tensor,
    epoch: int,
    temperature: float,
) -> torch.Tensor:
    if hard_doc_embs is None:
        return adapted_queries.sum() * 0.0

    owner_positions = []
    selected_negative_indices = []
    for batch_position, pair_idx in enumerate(batch_indices.tolist()):
        candidates = hard_negative_indices[pair_idx]
        if not candidates:
            continue
        selected_negative_indices.append(candidates[(epoch + batch_position) % len(candidates)])
        owner_positions.append(batch_position)

    if not selected_negative_indices:
        return adapted_queries.sum() * 0.0

    owner_tensor = torch.tensor(owner_positions, dtype=torch.long, device=adapted_queries.device)
    negative_tensor = torch.tensor(selected_negative_indices, dtype=torch.long, device=adapted_queries.device)
    query_vecs = F.normalize(adapted_queries[owner_tensor], dim=-1)
    positive_vecs = F.normalize(positive_docs.to(adapted_queries.device)[owner_tensor], dim=-1)
    negative_vecs = F.normalize(hard_doc_embs.to(adapted_queries.device)[negative_tensor], dim=-1)
    positive_scores = (query_vecs * positive_vecs).sum(dim=-1)
    negative_scores = (query_vecs * negative_vecs).sum(dim=-1)
    return F.softplus((negative_scores - positive_scores) / temperature).mean()


def _hamming_distance(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("E8 address keys must have the same length")
    return sum(1 for a, b in zip(left, right) if a != b)


def _encode_with(encoder: TextEncoder, texts, *, batch_size: int) -> torch.Tensor:
    encoded = encoder.encode(list(texts) if not isinstance(texts, str) else [texts], batch_size=batch_size)
    return torch.as_tensor(encoded, dtype=torch.float32)


def _check_dim(embeddings: torch.Tensor, d_model: int) -> None:
    if embeddings.dim() != 2 or embeddings.shape[1] != d_model:
        raise ValueError(f"expected encoder dim {d_model}, got shape {tuple(embeddings.shape)}")

