from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


FallbackSearch = Callable[[torch.Tensor, int], list["RetrievalHit"]]


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    path: str
    fallback_required: bool
    candidate_count: int


def _decode_d8(x: torch.Tensor) -> torch.Tensor:
    z = torch.round(x)
    parity = z.sum(dim=-1) % 2
    diff = x - z
    worst = diff.abs().argmax(dim=-1)
    adj = torch.sign(diff)
    adj = torch.where(adj == 0, torch.ones_like(adj), adj)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, worst.unsqueeze(-1), 1.0)
    z_fixed = z + adj * mask
    return torch.where(parity.unsqueeze(-1) == 1, z_fixed, z)


def _e8_nearest(x: torch.Tensor) -> torch.Tensor:
    z0 = _decode_d8(x)
    z1 = _decode_d8(x - 0.5) + 0.5
    d0 = (x - z0).pow(2).sum(dim=-1, keepdim=True)
    d1 = (x - z1).pow(2).sum(dim=-1, keepdim=True)
    return torch.where(d0 <= d1, z0, z1)


def _build_shell1_codebook(device: torch.device) -> torch.Tensor:
    vecs: list[list[float]] = []
    for i, j in combinations(range(8), 2):
        for si in (1.0, -1.0):
            for sj in (1.0, -1.0):
                v = [0.0] * 8
                v[i] = si
                v[j] = sj
                vecs.append(v)
    for mask in range(256):
        signs = [1.0 if (mask >> bit) & 1 == 0 else -1.0 for bit in range(8)]
        if signs.count(-1.0) % 2 == 0:
            vecs.append([s * 0.5 for s in signs])
    codebook = torch.tensor(vecs, dtype=torch.float32, device=device)
    if codebook.shape != (240, 8):
        raise RuntimeError(f"expected E8 shell-1 codebook shape (240, 8), got {tuple(codebook.shape)}")
    return codebook


class E8LatticeDB:
    """
    In-memory RF-Snap LatticeCache.

    Documents are stored under deterministic E8 block-address keys. Exact lookup
    is O(1); radius-1 lookup checks all one-block E8-address mutations; search
    reranks candidates with cosine similarity over stored embeddings and can
    delegate misses to an optional fallback hook.
    """

    def __init__(self, d_model: int = 1024, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        if d_model % 8 != 0:
            raise ValueError(f"d_model={d_model} must be divisible by 8")
        self.d_model = d_model
        self.num_blocks = d_model // 8
        self.device = torch.device(device)
        self.dtype = dtype
        self._codebook = _build_shell1_codebook(self.device)
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

    @torch.no_grad()
    def _quantize_to_indices(self, embedding: torch.Tensor | Iterable[float]) -> bytes:
        vector = self._as_vector(embedding)
        blocks = vector.reshape(-1, 8)
        beta = blocks.norm(p=2, dim=-1).clamp_min(1e-8) / math.sqrt(2.0)
        snapped = _e8_nearest(blocks / beta.unsqueeze(-1))
        snapped_unit = snapped / snapped.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dots = snapped_unit @ self._codebook.T / math.sqrt(2.0)
        return bytes(dots.argmax(dim=-1).tolist())

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
        if vectors.dim() != 2 or vectors.shape[1] != self.d_model:
            raise ValueError(f"expected embeddings shape [N, {self.d_model}], got {tuple(vectors.shape)}")
        meta_list = list(metadatas) if metadatas is not None else [None] * vectors.shape[0]
        for doc_id, vector, metadata in zip(doc_ids, vectors, meta_list):
            self.add_document(str(doc_id), vector, metadata)

    def retrieve_exact(self, query: torch.Tensor | Iterable[float]) -> list[str]:
        key = self._quantize_to_indices(query)
        return list(self.hash_store.get(key, []))

    def retrieve_radius_1(self, query: torch.Tensor | Iterable[float]) -> list[str]:
        key_bytes = self._quantize_to_indices(query)
        result: set[str] = set(self.hash_store.get(key_bytes, []))
        probe = bytearray(key_bytes)
        for block_idx in range(self.num_blocks):
            original = probe[block_idx]
            for alt in range(240):
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
        query: "torch.Tensor | Iterable[float]",
        radius: int = 1,
        top_k_alts: int = 5,
        max_candidates: int = 500,
    ) -> list[str]:
        """Return doc_ids within E8 Hamming distance `radius` using top-K alternatives per block.

        radius=0  : exact match only.
        radius>=1 : exhaustive Hamming-1, preserving retrieve_radius_1 behavior.
        radius>=2 : top-K cosine-nearest alternatives per block across block pairs/triples.
        max_candidates : early-exit cap on total distinct candidates returned.
        """
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}")
        vector = self._as_vector(query)
        key_bytes = self._quantize_to_indices(vector)
        result: list[str] = []
        seen: set[str] = set()

        def _add(doc_ids: "list[str] | None") -> None:
            if not doc_ids:
                return
            for did in doc_ids:
                if did not in seen:
                    seen.add(did)
                    result.append(did)

        # R=0: exact match
        _add(list(self.hash_store.get(key_bytes, [])))
        if radius == 0 or (max_candidates > 0 and len(result) >= max_candidates):
            return result

        probe = bytearray(key_bytes)

        # R=1: exhaustive one-block swap. This intentionally ignores top_k_alts
        # so beam modes never regress from the original Hamming-1 product path.
        for b in range(self.num_blocks):
            original = probe[b]
            for alt in range(240):
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

        # Precompute per-block ranked alternatives by cosine similarity for
        # approximate radius-2+ expansion.
        blocks = vector.reshape(-1, 8)
        block_norms = blocks.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        blocks_unit = blocks / block_norms
        cb_unit = self._codebook / self._codebook.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sims = blocks_unit @ cb_unit.T  # (num_blocks, 240)
        key_list = list(key_bytes)

        k = min(int(top_k_alts), 239)
        top_alts: list[list[int]] = []
        for b in range(self.num_blocks):
            row = sims[b].tolist()
            current_idx = key_list[b]
            ranked = sorted(
                (i for i in range(240) if i != current_idx),
                key=lambda i, r=row: r[i],
                reverse=True,
            )
            top_alts.append(ranked[:k])

        # R=2: swap two blocks simultaneously
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

        # R=3: swap three blocks (cap alts at 3 for speed)
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
        query: "torch.Tensor | Iterable[float]",
        top_k_per_block: int = 3,
        max_candidates: int = 500,
    ) -> list[str]:
        """Probabilistic multi-address beam search.

        Instead of committing to one hard E8 address per block (argmax), this
        generates the top-K most probable codebook entries per block weighted by
        cosine similarity, then walks the joint probability space by exploring
        all single-block alternatives ranked by their individual block probability.

        This is the architecture fix for exact-address brittleness: a query near
        a Voronoi boundary emits high probability for 2-3 nearby codewords per
        block, and the beam naturally explores those neighbors in priority order
        rather than treating all Hamming-1 neighbors as equally likely.

        Returns doc_ids sorted by joint block probability (highest first).
        """
        vector = self._as_vector(query)
        blocks = vector.reshape(-1, 8)
        block_norms = blocks.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        blocks_unit = blocks / block_norms
        cb_unit = self._codebook / self._codebook.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sims = blocks_unit @ cb_unit.T  # (num_blocks, 240) — per-block cosine scores

        k = min(int(top_k_per_block), 240)
        # Top-K entries per block with their probabilities (softmax of cosine scores)
        top_sims, top_indices = sims.topk(k, dim=-1)  # (num_blocks, k)
        probs = top_sims.softmax(dim=-1)               # (num_blocks, k) — probability per entry

        # Start with the top-1 address (greedy) and expand using single-block swaps
        # ordered by the probability of the swapped block entry
        top1_key = bytes(top_indices[:, 0].tolist())
        result: list[str] = []
        seen: set[str] = set()

        def _add(doc_ids: "list[str] | None") -> None:
            if not doc_ids:
                return
            for did in doc_ids:
                if did not in seen:
                    seen.add(did)
                    result.append(did)

        _add(list(self.hash_store.get(top1_key, [])))

        # Single-block swaps ordered by block probability (highest prob first)
        # Build list of (prob, block_idx, alt_codebook_idx) for all rank-2..k entries
        swap_candidates = []
        for b in range(self.num_blocks):
            for rank in range(1, k):  # rank 0 is already in top1
                swap_candidates.append((float(probs[b, rank].item()), b, int(top_indices[b, rank].item())))
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
                path="e8",
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
        """Serialize the index to a single file at `path` (e.g. 'index.pt').

        The format is a torch-saved dict containing all state needed to
        reconstruct the index with `load()`. Embeddings are stored as a
        stacked float32 tensor; structural data (doc order, hash_store,
        metadata, keys) is stored as plain Python objects.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        doc_ids = list(self._embeddings.keys())
        if doc_ids:
            emb_tensor = torch.stack([self._embeddings[d] for d in doc_ids]).cpu()
        else:
            emb_tensor = torch.zeros(0, self.d_model)
        state = {
            "_version": 2,
            "d_model": self.d_model,
            "doc_ids": doc_ids,
            "embeddings": emb_tensor,
            "metadata": [self._metadata.get(d, {}) for d in doc_ids],
            "keys": {doc_id: self._keys[doc_id].hex() for doc_id in doc_ids},
            "hash_store": {k.hex(): v for k, v in self.hash_store.items()},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "E8LatticeDB":
        """Reconstruct an `E8LatticeDB` from a file saved with `save()`."""
        state = torch.load(path, weights_only=False)
        version = state.get("_version", 0)
        if version != 2:
            raise ValueError(
                f"Unsupported E8LatticeDB save version {version}. "
                "Re-index your corpus and save again with the current version."
            )
        db = cls(d_model=state["d_model"])
        doc_ids: list[str] = state["doc_ids"]
        emb_tensor: torch.Tensor = state["embeddings"].to(db.device)
        keys_data = state["keys"]  # dict[doc_id, hex_str]
        for i, doc_id in enumerate(doc_ids):
            db._embeddings[doc_id] = emb_tensor[i]
            db._metadata[doc_id] = state["metadata"][i]
            db._keys[doc_id] = bytes.fromhex(keys_data[doc_id])
        for k_hex, v in state["hash_store"].items():
            db.hash_store[bytes.fromhex(k_hex)] = list(v)
        return db
