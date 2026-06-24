from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from latticememory.rag.e8_retriever import RetrievalHit, RetrievalResult, FallbackSearch


def train_spherical_kmeans(
    x: torch.Tensor,
    k: int,
    num_iters: int = 30,
    tol: float = 1e-4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Vectorized Spherical K-Means in PyTorch.

    Args:
        x: Input tensor of shape (N, d_sub)
        k: Number of cluster centroids to learn
        num_iters: Maximum number of iterations
        tol: Convergence tolerance for centroid change
        device: Device to run the clustering on

    Returns:
        Centroid tensor of shape (k, d_sub), unit-normalized.
    """
    N, d_sub = x.shape
    if N < k:
        raise ValueError(
            f"train_spherical_kmeans needs at least k={k} training vectors to fit "
            f"k={k} centroids, got only N={N}. Use a larger calibration sample, or "
            f"reduce codebook_size."
        )
    x = F.normalize(x.to(device=device), p=2, dim=-1)

    # Initialize centroids randomly from data points with a fixed generator seed
    g = torch.Generator(device=x.device)
    g.manual_seed(42)
    indices = torch.randperm(N, generator=g, device=x.device)[:k]
    centroids = x[indices].clone()

    for iteration in range(num_iters):
        # Compute cosine similarity: (N, k)
        sims = x @ centroids.T
        labels = sims.argmax(dim=-1)

        # Update centroids in a fully vectorized manner using index_add_
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.float32, device=x.device)
        ones = torch.ones(N, dtype=torch.float32, device=x.device)
        counts.index_add_(0, labels, ones)

        new_centroids.index_add_(0, labels, x)

        # Normalize non-empty centroids, and retain old ones for empty clusters
        empty_mask = counts == 0
        non_empty_mask = ~empty_mask

        if non_empty_mask.any():
            new_centroids[non_empty_mask] = F.normalize(new_centroids[non_empty_mask], p=2, dim=-1)
        if empty_mask.any():
            new_centroids[empty_mask] = centroids[empty_mask]

        if torch.allclose(centroids, new_centroids, atol=tol):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


class PQLatticeDB:
    """In-memory Product Quantization Lattice Cache.

    Divides embeddings into coarser blocks and quantizes each block to a learned
    codebook of size K using Spherical K-Means. Exposes identical retrieval
    methods (exact, radius-1, beam search) as E8LatticeDB.
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_blocks: int = 16,
        codebook_size: int = 256,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if d_model % num_blocks != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_blocks={num_blocks}")
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.codebook_size = codebook_size
        self.block_dim = d_model // num_blocks
        self.device = torch.device(device)
        self.dtype = dtype

        # Codebooks shape: (num_blocks, codebook_size, block_dim)
        self._codebooks: torch.Tensor | None = None
        self.hash_store: defaultdict[bytes, list[str]] = defaultdict(list)
        self._embeddings: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, dict] = {}
        self._keys: dict[str, bytes] = {}

    @property
    def num_documents(self) -> int:
        return len(self._embeddings)

    def _as_vector(self, embedding: torch.Tensor | Iterable[float]) -> torch.Tensor:
        vector = torch.as_tensor(embedding, dtype=self.dtype, device=self.device)
        if vector.dim() == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        if vector.dim() != 1:
            raise ValueError(f"expected a single vector, got shape {tuple(vector.shape)}")
        if vector.numel() != self.d_model:
            raise ValueError(f"expected vector dim {self.d_model}, got {vector.numel()}")
        return vector.float()

    def fit(self, embeddings: torch.Tensor, num_iters: int = 30) -> None:
        """Train Spherical K-Means codebooks on a sample of embeddings."""
        N = embeddings.shape[0]
        vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        blocks = vectors.reshape(N, self.num_blocks, self.block_dim)

        self._codebooks = torch.zeros(
            self.num_blocks,
            self.codebook_size,
            self.block_dim,
            device=self.device,
            dtype=self.dtype,
        )

        for b in range(self.num_blocks):
            sub_vecs = blocks[:, b, :]
            centroids = train_spherical_kmeans(
                sub_vecs,
                self.codebook_size,
                num_iters=num_iters,
                device=self.device,
            )
            self._codebooks[b] = centroids

    @torch.no_grad()
    def _quantize_to_indices(self, embedding: torch.Tensor | Iterable[float]) -> bytes:
        if self._codebooks is None:
            raise RuntimeError("Index not fitted. Run fit() first.")
        vector = self._as_vector(embedding)
        blocks = vector.reshape(self.num_blocks, self.block_dim)
        unit_blocks = F.normalize(blocks, p=2, dim=-1)  # (num_blocks, block_dim)

        # Batch dot-product for the query blocks: (num_blocks, codebook_size)
        sims = torch.sum(unit_blocks.unsqueeze(1) * self._codebooks, dim=-1)
        indices = sims.argmax(dim=-1).tolist()
        return bytes(indices)

    @torch.no_grad()
    def _quantize_batch(self, embeddings: torch.Tensor) -> list[bytes]:
        if self._codebooks is None:
            raise RuntimeError("Index not fitted. Run fit() first.")
        vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        N = vectors.shape[0]
        blocks = vectors.reshape(N, self.num_blocks, self.block_dim)
        unit_blocks = F.normalize(blocks, p=2, dim=-1)  # (N, num_blocks, block_dim)

        # Batched matrix multiplication: (num_blocks, N, codebook_size)
        # unit_blocks_perm: (num_blocks, N, block_dim)
        # codebooks_trans: (num_blocks, block_dim, codebook_size)
        unit_blocks_perm = unit_blocks.permute(1, 0, 2)
        codebooks_trans = self._codebooks.transpose(1, 2)
        sims = torch.bmm(unit_blocks_perm, codebooks_trans)

        indices = sims.argmax(dim=-1)  # (num_blocks, N)
        rows = indices.transpose(0, 1).tolist()  # (N, num_blocks)
        return [bytes(row) for row in rows]

    def add_document(
        self,
        doc_id: str,
        embedding: torch.Tensor | Iterable[float],
        metadata: dict | None = None,
    ) -> bytes:
        vector = self._as_vector(embedding)
        key = self._quantize_to_indices(vector)
        if doc_id in self._keys:
            old_key = self._keys[doc_id]
            self.hash_store[old_key] = [existing for existing in self.hash_store[old_key] if existing != doc_id]
        self.hash_store[key].append(doc_id)
        self._embeddings[doc_id] = F.normalize(vector, p=2, dim=0)
        self._metadata[doc_id] = dict(metadata or {})
        self._keys[doc_id] = key
        return key

    def add_batch(
        self,
        doc_ids: Iterable[str],
        embeddings: torch.Tensor | Iterable[Iterable[float]],
        metadatas: Iterable[dict | None] | None = None,
    ) -> None:
        vectors = torch.as_tensor(embeddings, dtype=self.dtype, device=self.device)
        meta_list = list(metadatas) if metadatas is not None else [None] * vectors.shape[0]
        keys = self._quantize_batch(vectors)
        for doc_id_raw, vector, key, metadata in zip(doc_ids, vectors, keys, meta_list):
            doc_id = str(doc_id_raw)
            if doc_id in self._keys:
                old_key = self._keys[doc_id]
                self.hash_store[old_key] = [e for e in self.hash_store[old_key] if e != doc_id]
            self.hash_store[key].append(doc_id)
            self._embeddings[doc_id] = F.normalize(vector, p=2, dim=0)
            self._metadata[doc_id] = dict(metadata or {})
            self._keys[doc_id] = key

    def retrieve_exact(self, query: torch.Tensor | Iterable[float]) -> list[str]:
        key = self._quantize_to_indices(query)
        return list(self.hash_store.get(key, []))

    def retrieve_radius_1(self, query: torch.Tensor | Iterable[float]) -> list[str]:
        key_bytes = self._quantize_to_indices(query)
        result: set[str] = set(self.hash_store.get(key_bytes, []))
        probe = bytearray(key_bytes)
        for block_idx in range(self.num_blocks):
            original = probe[block_idx]
            for alt in range(self.codebook_size):
                if alt == original:
                    continue
                probe[block_idx] = alt
                candidates = self.hash_store.get(bytes(probe), None)
                if candidates:
                    result.update(candidates)
            probe[block_idx] = original
        return list(result)

    def retrieve_within_radius(
        self,
        query: torch.Tensor | Iterable[float],
        radius: int = 1,
        top_k_alts: int = 5,
        max_candidates: int = 500,
    ) -> list[str]:
        """Return doc_ids within Hamming distance `radius`, generalizing
        retrieve_radius_1 the same way E8LatticeDB.retrieve_within_radius
        generalizes its own radius-1 path - exhaustive for radius<=1, then
        top-K learned-codebook-cosine-nearest alternatives per block for
        radius>=2. Mirrors E8LatticeDB's interface/semantics exactly so
        RFSnapLatticeMemory can use either backend interchangeably.

        radius=0  : exact match only.
        radius>=1 : exhaustive Hamming-1, preserving retrieve_radius_1 behavior.
        radius>=2 : top-K cosine-nearest alternatives per block across block pairs/triples.
        max_candidates : early-exit cap on total distinct candidates returned.
        """
        if self._codebooks is None:
            raise RuntimeError("Index not fitted. Run fit() first.")
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        vector = self._as_vector(query)
        key_bytes = self._quantize_to_indices(vector)
        result: list[str] = []
        seen: set[str] = set()

        def _add(doc_ids: list[str] | None) -> None:
            if not doc_ids:
                return
            for did in doc_ids:
                if did not in seen:
                    seen.add(did)
                    result.append(did)

        _add(list(self.hash_store.get(key_bytes, [])))
        if radius == 0 or (max_candidates > 0 and len(result) >= max_candidates):
            return result

        probe = bytearray(key_bytes)

        for b in range(self.num_blocks):
            original = probe[b]
            for alt in range(self.codebook_size):
                if alt == original:
                    continue
                probe[b] = alt
                _add(list(self.hash_store.get(bytes(probe), [])))
                if max_candidates > 0 and len(result) >= max_candidates:
                    probe[b] = original
                    return result
            probe[b] = original

        if radius == 1 or (max_candidates > 0 and len(result) >= max_candidates):
            return result

        blocks = vector.reshape(self.num_blocks, self.block_dim)
        unit_blocks = F.normalize(blocks, p=2, dim=-1)
        sims = torch.sum(unit_blocks.unsqueeze(1) * self._codebooks, dim=-1)  # (num_blocks, codebook_size)
        key_list = list(key_bytes)

        k = min(int(top_k_alts), self.codebook_size - 1)
        top_alts: list[list[int]] = []
        for b in range(self.num_blocks):
            row = sims[b].tolist()
            current_idx = key_list[b]
            ranked = sorted(
                (i for i in range(self.codebook_size) if i != current_idx),
                key=lambda i, r=row: r[i],
                reverse=True,
            )
            top_alts.append(ranked[:k])

        from itertools import combinations as _comb

        for b1, b2 in _comb(range(self.num_blocks), 2):
            orig1, orig2 = probe[b1], probe[b2]
            for alt1 in top_alts[b1]:
                probe[b1] = alt1
                for alt2 in top_alts[b2]:
                    probe[b2] = alt2
                    _add(list(self.hash_store.get(bytes(probe), [])))
                    if max_candidates > 0 and len(result) >= max_candidates:
                        probe[b1] = orig1
                        probe[b2] = orig2
                        return result
                probe[b2] = orig2
            probe[b1] = orig1

        if radius == 2 or (max_candidates > 0 and len(result) >= max_candidates):
            return result

        k3 = min(k, 3)
        for b1, b2, b3 in _comb(range(self.num_blocks), 3):
            orig1, orig2, orig3 = probe[b1], probe[b2], probe[b3]
            for alt1 in top_alts[b1][:k3]:
                probe[b1] = alt1
                for alt2 in top_alts[b2][:k3]:
                    probe[b2] = alt2
                    for alt3 in top_alts[b3][:k3]:
                        probe[b3] = alt3
                        _add(list(self.hash_store.get(bytes(probe), [])))
                        if max_candidates > 0 and len(result) >= max_candidates:
                            probe[b1] = orig1
                            probe[b2] = orig2
                            probe[b3] = orig3
                            return result
                    probe[b3] = orig3
                probe[b2] = orig2
            probe[b1] = orig1

        return result

    def retrieve_probabilistic_beam(
        self,
        query: torch.Tensor | Iterable[float],
        top_k_per_block: int = 3,
        max_candidates: int = 500,
    ) -> list[str]:
        """Probabilistic multi-address beam search using learned codebooks."""
        if self._codebooks is None:
            raise RuntimeError("Index not fitted.")
        vector = self._as_vector(query)
        blocks = vector.reshape(self.num_blocks, self.block_dim)
        unit_blocks = F.normalize(blocks, p=2, dim=-1)  # (num_blocks, block_dim)

        # sims: (num_blocks, codebook_size)
        sims = torch.sum(unit_blocks.unsqueeze(1) * self._codebooks, dim=-1)

        k = min(int(top_k_per_block), self.codebook_size)
        top_sims, top_indices = sims.topk(k, dim=-1)  # (num_blocks, k)
        probs = top_sims.softmax(dim=-1)  # (num_blocks, k)

        top1_key = bytes(top_indices[:, 0].tolist())
        result: list[str] = []
        seen: set[str] = set()

        def _add(doc_ids: list[str] | None) -> None:
            if not doc_ids:
                return
            for did in doc_ids:
                if did not in seen:
                    seen.add(did)
                    result.append(did)

        _add(list(self.hash_store.get(top1_key, [])))

        # Single-block swaps ordered by probability
        swap_candidates = []
        for b in range(self.num_blocks):
            for rank in range(1, k):
                swap_candidates.append(
                    (
                        float(probs[b, rank].item()),
                        b,
                        int(top_indices[b, rank].item()),
                    )
                )
        swap_candidates.sort(key=lambda x: x[0], reverse=True)

        probe = bytearray(top1_key)
        for _, b, alt_idx in swap_candidates:
            if max_candidates > 0 and len(result) >= max_candidates:
                break
            orig = probe[b]
            probe[b] = alt_idx
            _add(list(self.hash_store.get(bytes(probe), [])))
            probe[b] = orig

        return result

    def _rerank(self, query: torch.Tensor, doc_ids: list[str], top_k: int) -> list[RetrievalHit]:
        if not doc_ids:
            return []
        q = F.normalize(query, p=2, dim=0)
        hits = []
        for doc_id in dict.fromkeys(doc_ids):
            score = float(torch.dot(q, self._embeddings[doc_id]).item())
            hits.append(RetrievalHit(doc_id=doc_id, score=score, metadata=self._metadata.get(doc_id, {})))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def search(
        self,
        query: torch.Tensor | Iterable[float],
        top_k: int = 5,
        *,
        fallback: FallbackSearch | None = None,
    ) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = self._as_vector(query)
        candidates = self.retrieve_radius_1(vector)
        if candidates:
            return RetrievalResult(
                hits=self._rerank(vector, candidates, top_k),
                path="pq",
                fallback_required=False,
                candidate_count=len(candidates),
            )
        if fallback is not None:
            fallback_hits = fallback(vector.detach().cpu(), top_k)
            return RetrievalResult(
                hits=fallback_hits[:top_k],
                path="fallback",
                fallback_required=True,
                candidate_count=0,
            )
        return RetrievalResult(hits=[], path="miss", fallback_required=True, candidate_count=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        doc_ids = list(self._embeddings.keys())
        if doc_ids:
            emb_tensor = torch.stack([self._embeddings[d] for d in doc_ids]).cpu()
        else:
            emb_tensor = torch.zeros(0, self.d_model)

        state = {
            "_version": 1,
            "d_model": self.d_model,
            "num_blocks": self.num_blocks,
            "codebook_size": self.codebook_size,
            "codebooks": self._codebooks.cpu() if self._codebooks is not None else None,
            "doc_ids": doc_ids,
            "embeddings": emb_tensor,
            "metadata": [self._metadata.get(d, {}) for d in doc_ids],
            "keys": {doc_id: self._keys[doc_id].hex() for doc_id in doc_ids},
            "hash_store": {k.hex(): v for k, v in self.hash_store.items()},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> PQLatticeDB:
        state = torch.load(path, weights_only=False)
        db = cls(
            d_model=state["d_model"],
            num_blocks=state["num_blocks"],
            codebook_size=state["codebook_size"],
        )
        if state["codebooks"] is not None:
            db._codebooks = state["codebooks"].to(db.device)
        doc_ids: list[str] = state["doc_ids"]
        emb_tensor: torch.Tensor = state["embeddings"].to(db.device)
        keys_data = state["keys"]
        for i, doc_id in enumerate(doc_ids):
            db._embeddings[doc_id] = emb_tensor[i]
            db._metadata[doc_id] = state["metadata"][i]
            db._keys[doc_id] = bytes.fromhex(keys_data[doc_id])
        for k_hex, v in state["hash_store"].items():
            db.hash_store[bytes.fromhex(k_hex)] = list(v)
        return db
