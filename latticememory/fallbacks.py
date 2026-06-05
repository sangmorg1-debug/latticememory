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

    Uses a normalized inner-product `IndexFlatIP`, which is equivalent to cosine
    similarity for normalized vectors. This is intentionally the first simple
    production adapter; IVF/HNSW variants can be added behind the same protocol.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self._faiss = _require_faiss()
        self._index = self._faiss.IndexFlatIP(d_model)
        self._doc_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}

    @property
    def num_documents(self) -> int:
        return len(self._doc_ids)

    def add_documents(self, documents: Iterable[MemoryDocument]) -> None:
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
        batch = torch.stack(vectors).contiguous().cpu().numpy().astype("float32")
        self._index.add(batch)

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))
        torch.save(
            {
                "_version": 1,
                "d_model": self.d_model,
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
        fallback = cls(d_model=int(meta["d_model"]))
        fallback._index = fallback._faiss.read_index(str(path))
        fallback._doc_ids = list(meta["doc_ids"])
        fallback._texts = dict(meta["texts"])
        fallback._metadata = dict(meta["metadata"])
        return fallback

