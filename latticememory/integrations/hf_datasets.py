from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch

from latticememory.pipeline import LatticeDataPipeline


class LatticeDatasetFilter:
    """Streaming dataset filters for HuggingFace-style row iterables."""

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        d_model: int = 1024,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.d_model = int(d_model)
        self.pipeline = LatticeDataPipeline(text_encoder=encoder, d_model=d_model, device=device)

    def deduplicate_streaming(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        text_column: str,
        batch_size: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Yield only the first row observed for each text lattice key."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        encoder = self._get_encoder()
        seen_keys: set[bytes] = set()
        for batch in _batched(dataset, batch_size):
            texts = [str(row[text_column]) for row in batch]
            embeddings = torch.as_tensor(encoder.encode(texts, batch_size=batch_size), dtype=torch.float32)
            if embeddings.dim() != 2 or embeddings.shape[1] != self.d_model:
                raise ValueError(f"encoder returned embeddings with shape {tuple(embeddings.shape)}")
            for row, embedding in zip(batch, embeddings):
                key = self.pipeline._memory.lattice_key_for(embedding)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                yield row

    def filter_caption_quality(
        self,
        dataset: Iterable[dict[str, Any]],
        *,
        image_column: str,
        caption_column: str,
        adapter: Any | None = None,
        batch_size: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows whose image and caption embeddings snap to the same key.

        `adapter` may be a fitted object returned by
        `LatticeDataPipeline.fit_cross_modal_adapter`.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if adapter is not None:
            self.pipeline._cross_modal_adapter = adapter
        for batch in _batched(dataset, batch_size):
            images = [row[image_column] for row in batch]
            captions = [row[caption_column] for row in batch]
            report = self.pipeline.filter_caption_quality(images=images, captions=captions)
            clean = set(report["clean_pairs"])
            for idx, row in enumerate(batch):
                if idx in clean:
                    yield row

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer("dfrokido/bge-large-e8-snap")
        return self.encoder


def _batched(dataset: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in dataset:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
