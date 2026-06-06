from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from latticememory.rag.e8_retriever import RetrievalHit

from .memory import MemoryDocument


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FaissVectorFallback requires faiss. Install the API extra or faiss-cpu."
        ) from exc
    return faiss


class FaissVectorFallback:
    """FAISS-backed cosine retrieval adapter for LatticeMemory fallback.

    Uses a normalized inner-product index, supporting optional QT_8bit and QT_4bit
    scalar quantization for memory compression.
    """

    def __init__(self, d_model: int, quantization_bits: int | None = None):
        if quantization_bits is not None and quantization_bits not in (4, 8):
            raise ValueError("quantization_bits must be 4, 8, or None")
        if quantization_bits == 4 and d_model % 2 != 0:
            raise ValueError("d_model must be even for 4-bit quantization")
            
        self.d_model = d_model
        self.quantization_bits = quantization_bits
        self._faiss = _require_faiss()
        
        if quantization_bits == 8:
            self._index = self._faiss.IndexScalarQuantizer(d_model, self._faiss.ScalarQuantizer.QT_8bit, self._faiss.METRIC_INNER_PRODUCT)
        elif quantization_bits == 4:
            self._index = self._faiss.IndexScalarQuantizer(d_model, self._faiss.ScalarQuantizer.QT_4bit, self._faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = self._faiss.IndexFlatIP(d_model)
            
        self._doc_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}

    @property
    def num_documents(self) -> int:
        return len(self._doc_ids)

    def add_documents(self, documents: Iterable[MemoryDocument]) -> None:
        import numpy as np
        vectors = []
        for doc in documents:
            vector = torch.as_tensor(doc.embedding, dtype=torch.float32)
            if vector.dim() != 1 or vector.numel() != self.d_model:
                raise ValueError(f"document {doc.doc_id!r} embedding must have dim {self.d_model}")
            self._doc_ids.append(doc.doc_id)
            self._texts[doc.doc_id] = doc.text
            self._metadata[doc.doc_id] = dict(doc.metadata)
            vectors.append(F.normalize(vector, p=2, dim=0))
            
        if not vectors:
            return
            
        batch_tensor = torch.stack(vectors).contiguous().cpu()
        batch_np = batch_tensor.numpy().astype("float32")
        
        if not self._index.is_trained:
            if batch_np.shape[0] >= 256:
                self._index.train(batch_np)
            else:
                needed = 256 - batch_np.shape[0]
                synthetic = np.random.randn(needed, self.d_model).astype("float32")
                norms = np.linalg.norm(synthetic, axis=1, keepdims=True)
                synthetic = np.where(norms > 0, synthetic / norms, synthetic)
                train_data = np.concatenate([batch_np, synthetic], axis=0)
                self._index.train(train_data)
                
        self._index.add(batch_np)

    def search(self, query: torch.Tensor | Iterable[float], top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._doc_ids:
            return []
        vector = torch.as_tensor(query, dtype=torch.float32)
        if vector.dim() != 1 or vector.numel() != self.d_model:
            raise ValueError(f"query embedding must have dim {self.d_model}")
        q = F.normalize(vector, p=2, dim=0).reshape(1, -1).contiguous().cpu().numpy().astype("float32")
        k = min(top_k, self.num_documents)
        scores, indices = self._index.search(q, k)
        hits = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            doc_id = self._doc_ids[idx]
            hits.append(RetrievalHit(doc_id=doc_id, score=float(score), metadata=self._metadata.get(doc_id, {})))
        return hits

    def text_for(self, doc_id: str) -> str:
        return self._texts.get(doc_id, "")

    def get_index_size_bytes(self) -> int:
        if not self._doc_ids:
            return 0
        n = len(self._doc_ids)
        if self.quantization_bits == 8:
            return n * self.d_model
        elif self.quantization_bits == 4:
            return n * (self.d_model // 2)
        else:
            return n * self.d_model * 4

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))
        torch.save(
            {
                "_version": 1,
                "d_model": self.d_model,
                "quantization_bits": self.quantization_bits,
                "doc_ids": list(self._doc_ids),
                "texts": dict(self._texts),
                "metadata": {doc_id: dict(meta) for doc_id, meta in self._metadata.items()},
            },
            str(path) + ".meta.pt",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorFallback":
        path = Path(path)
        meta = torch.load(str(path) + ".meta.pt", weights_only=False)
        if meta.get("_version") != 1:
            raise ValueError(f"Unsupported FaissVectorFallback save version: {meta.get('_version')}")
        fallback = cls(d_model=int(meta["d_model"]), quantization_bits=meta.get("quantization_bits"))
        fallback._index = fallback._faiss.read_index(str(path))
        fallback._doc_ids = list(meta["doc_ids"])
        fallback._texts = dict(meta["texts"])
        fallback._metadata = dict(meta["metadata"])
        return fallback

