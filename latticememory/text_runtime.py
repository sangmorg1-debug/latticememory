from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import torch

from .memory import MemoryDocument, MemoryQuery, MemoryResult, RFSnapLatticeMemory, TextPairReranker, VectorFallback


class TextEncoder(Protocol):
    def encode(self, sentences, batch_size: int = 64, **kwargs):
        ...


@dataclass(frozen=True)
class TextIndexResult:
    indexed: int
    total_documents: int
    doc_ids: list[str]


class RFSnapTextMemory:
    """Text-level product runtime for RF-Snap LatticeMemory.

    Products should use this when they have text and an RF-Snap encoder rather
    than precomputed embeddings. The lower `RFSnapLatticeMemory` layer remains
    embedding-first for benchmarks and systems that already own embeddings.
    """

    def __init__(
        self,
        *,
        encoder: TextEncoder,
        d_model: int,
        memory: RFSnapLatticeMemory | None = None,
        fallback: VectorFallback | None = None,
        model_id: str | None = None,
        batch_size: int = 64,
        hamming_pool_multiplier: int = 10,
        rerank_mode: str = "cosine",
        lexical_weight: float = 0.0,
        neural_reranker: TextPairReranker | None = None,
        neural_rerank_top_n: int | None = None,
        beam_radius: int = 1,
    ):
        self.encoder = encoder
        self.d_model = d_model
        self.memory = memory or RFSnapLatticeMemory(
            d_model=d_model,
            fallback=fallback,
            hamming_pool_multiplier=hamming_pool_multiplier,
            rerank_mode=rerank_mode,
            lexical_weight=lexical_weight,
            neural_reranker=neural_reranker,
            neural_rerank_top_n=neural_rerank_top_n,
            beam_radius=beam_radius,
        )
        if self.memory.d_model != d_model:
            raise ValueError(f"memory d_model {self.memory.d_model} does not match runtime d_model {d_model}")
        self.model_id = model_id
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "e8_bge_large_snaptrained",
        *,
        d_model: int | None = None,
        device: str | None = None,
        batch_size: int = 64,
        trust_remote_code: bool = True,
        fallback: VectorFallback | None = None,
        hamming_pool_multiplier: int = 10,
        rerank_mode: str = "cosine",
        lexical_weight: float = 0.0,
        neural_reranker: TextPairReranker | None = None,
        neural_rerank_top_n: int | None = None,
        beam_radius: int = 1,
    ) -> "RFSnapTextMemory":
        from hf_model.modeling_e8_snap import E8SnapEncoder

        encoder = E8SnapEncoder.from_pretrained(
            model_path,
            batch_size=batch_size,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        detected_dim = int(getattr(encoder, "_embed_dim", 0) or 0)
        runtime_dim = d_model or detected_dim
        if runtime_dim <= 0:
            probe = cls._encode_with(encoder, ["dimension probe"], batch_size=batch_size)
            runtime_dim = int(probe.shape[1])
        return cls(
            encoder=encoder,
            d_model=runtime_dim,
            fallback=fallback,
            model_id=model_path,
            batch_size=batch_size,
            hamming_pool_multiplier=hamming_pool_multiplier,
            rerank_mode=rerank_mode,
            lexical_weight=lexical_weight,
            neural_reranker=neural_reranker,
            neural_rerank_top_n=neural_rerank_top_n,
            beam_radius=beam_radius,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        doc_ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        dataset: str | None = None,
        index_id: str | None = None,
        batch_size: int | None = None,
    ) -> TextIndexResult:
        text_list = list(texts)
        if not text_list:
            return TextIndexResult(indexed=0, total_documents=self.memory.num_documents, doc_ids=[])
        ids = list(doc_ids) if doc_ids is not None else [f"doc-{self.memory.num_documents + i}" for i in range(len(text_list))]
        if len(ids) != len(text_list):
            raise ValueError("doc_ids length must match texts length")
        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in text_list]
        if len(metadata_list) != len(text_list):
            raise ValueError("metadatas length must match texts length")

        embeddings = self._encode_texts(text_list, batch_size=batch_size)
        docs = []
        for text, doc_id, metadata, embedding in zip(text_list, ids, metadata_list, embeddings):
            enriched = dict(metadata)
            if dataset is not None:
                enriched.setdefault("dataset", dataset)
            if index_id is not None:
                enriched.setdefault("index_id", index_id)
            docs.append(MemoryDocument(doc_id=doc_id, text=text, embedding=embedding, metadata=enriched))
        self.memory.add_documents(docs)
        return TextIndexResult(indexed=len(docs), total_documents=self.memory.num_documents, doc_ids=ids)

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
        embedding = self._encode_texts([text], batch_size=batch_size)[0]
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

    def _encode_texts(self, texts: Sequence[str], *, batch_size: int | None = None) -> torch.Tensor:
        embeddings = self._encode_with(self.encoder, list(texts), batch_size=batch_size or self.batch_size)
        if embeddings.dim() != 2 or embeddings.shape[1] != self.d_model:
            raise ValueError(f"expected encoder dim {self.d_model}, got shape {tuple(embeddings.shape)}")
        return embeddings

    @staticmethod
    def _encode_with(encoder: TextEncoder, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
        encoded = encoder.encode(list(texts), batch_size=batch_size)
        return torch.as_tensor(encoded, dtype=torch.float32)

