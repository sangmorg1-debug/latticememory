from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from latticememory.dedup import LatticeDedup
from latticememory.memory import RFSnapLatticeMemory


@dataclass(frozen=True)
class _LinearEmbeddingAdapter:
    weight: torch.Tensor
    bias: torch.Tensor

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.float() @ self.weight.T + self.bias


class LatticeDataPipeline:
    """High-level ML training-data workflows backed by E8 lattice addresses.

    The pipeline intentionally works with precomputed embeddings for image-caption
    quality and sharding so large dataset jobs can keep model inference outside of
    the hot path.
    """

    def __init__(
        self,
        *,
        text_encoder: Any | None = None,
        image_encoder: Any | None = None,
        d_model: int = 1024,
        device: str = "cpu",
    ):
        if d_model <= 0 or d_model % 8 != 0:
            raise ValueError("d_model must be a positive multiple of 8")
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.d_model = int(d_model)
        self.device = torch.device(device)
        self._memory = RFSnapLatticeMemory(d_model=self.d_model)
        self._cross_modal_adapter: _LinearEmbeddingAdapter | None = None

    def fit_cross_modal_adapter(
        self,
        *,
        image_embeddings,
        text_embeddings,
        epochs: int = 50,
        ridge: float = 0.0,
    ) -> _LinearEmbeddingAdapter:
        """Fit a linear image->text embedding adapter for exact E8 key matching.

        `epochs` is accepted for API compatibility with training-loop based
        adapters. This ridge solve is deterministic and has no iterative state.
        """

        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        image = _as_2d_tensor(image_embeddings, name="image_embeddings", device=self.device)
        text = _as_2d_tensor(text_embeddings, name="text_embeddings", device=self.device)
        if image.shape[0] != text.shape[0]:
            raise ValueError("image_embeddings and text_embeddings must have the same row count")
        if text.shape[1] != self.d_model:
            raise ValueError(f"text_embeddings must have dim {self.d_model}, got {text.shape[1]}")

        ones = torch.ones(image.shape[0], 1, dtype=torch.float32, device=self.device)
        design = torch.cat([image.float(), ones], dim=1)
        targets = text.float()
        gram = design.T @ design
        if ridge:
            gram = gram + torch.eye(gram.shape[0], dtype=torch.float32, device=self.device) * float(ridge)
            gram[-1, -1] -= float(ridge)
        solution = torch.linalg.pinv(gram) @ (design.T @ targets)
        adapter = _LinearEmbeddingAdapter(
            weight=solution[:-1, :].T.detach().cpu(),
            bias=solution[-1, :].detach().cpu(),
        )
        self._cross_modal_adapter = adapter
        return adapter

    def deduplicate_text(self, documents: Sequence[str]) -> dict[str, Any]:
        """Deduplicate text documents with the existing O(N) lattice deduper."""

        deduper = LatticeDedup(d_model=self.d_model)
        if self.text_encoder is not None:
            deduper.encoder = self.text_encoder
        return deduper.deduplicate(list(documents))

    def filter_caption_quality(self, *, images, captions) -> dict[str, Any]:
        """Return clean and mismatched image-caption row indices by E8 key equality."""

        image_embeddings = _as_2d_tensor(images, name="images", device=self.device)
        caption_embeddings = _as_2d_tensor(captions, name="captions", device=self.device)
        if image_embeddings.shape[0] != caption_embeddings.shape[0]:
            raise ValueError("images and captions must have the same row count")
        if caption_embeddings.shape[1] != self.d_model:
            raise ValueError(f"captions must have dim {self.d_model}, got {caption_embeddings.shape[1]}")
        if self._cross_modal_adapter is not None:
            image_embeddings = self._cross_modal_adapter.transform(image_embeddings.cpu()).to(self.device)
        elif image_embeddings.shape[1] != self.d_model:
            raise ValueError(
                f"images must have dim {self.d_model} when no cross-modal adapter is fitted, "
                f"got {image_embeddings.shape[1]}"
            )

        clean_pairs: list[int] = []
        mismatch_pairs: list[int] = []
        for idx, (image_embedding, caption_embedding) in enumerate(zip(image_embeddings, caption_embeddings)):
            image_key = self._memory.lattice_key_for(image_embedding)
            caption_key = self._memory.lattice_key_for(caption_embedding)
            if image_key == caption_key:
                clean_pairs.append(idx)
            else:
                mismatch_pairs.append(idx)

        total = int(image_embeddings.shape[0])
        mismatch_rate = (len(mismatch_pairs) / total) if total else 0.0
        return {
            "clean_pairs": clean_pairs,
            "mismatch_pairs": mismatch_pairs,
            "mismatch_rate": mismatch_rate,
            "total_pairs": total,
        }

    def assign_shards(self, embeddings, *, num_shards: int) -> np.ndarray:
        """Assign deterministic shard IDs from lattice addresses."""

        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        vectors = _as_2d_tensor(embeddings, name="embeddings", device=self.device)
        if vectors.shape[1] != self.d_model:
            raise ValueError(f"embeddings must have dim {self.d_model}, got {vectors.shape[1]}")

        shard_ids = []
        for vector in vectors:
            key = self._memory.lattice_key_for(vector)
            shard_ids.append(int.from_bytes(key, byteorder="big") % int(num_shards))
        return np.asarray(shard_ids, dtype=np.int64)


def _as_2d_tensor(value, *, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(value, list):
        value = np.asarray(value, dtype=np.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2-D embedding array")
    return tensor
