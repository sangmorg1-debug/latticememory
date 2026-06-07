"""HammingRouter — semantic cache using E8 lattice Hamming distance nearest-neighbor.

Key insight from empirical measurement on BAAI/bge-large-en-v1.5:
- Intra-cluster (paraphrase) Hamming distances: 84-119, mean 110
- Inter-cluster (different topic) Hamming distances: 112-127, mean 122
- Gap: inter_min (112) > most intra distances

At threshold=111: ~60% recall, 0% false positives on diverse topic clusters.
The threshold is tunable — lower = higher precision, higher = higher recall.

IMPORTANT — near-miss calibration finding (empirical, 2026-06):
On adjacent-but-distinct same-domain prompts (e.g. "cancel Netflix" vs "cancel gym"),
the Hamming distribution OVERLAPS with paraphrase pairs (both ~97-100 mean). No threshold
gives FP=0 with useful recall on such pairs using the base model. Use ``gap_stats()`` on your
domain's near-miss pairs before choosing a threshold; use ``calibrate_threshold()`` for a
principled, data-driven threshold rather than guessing. The proxy default threshold=70 is
conservative; the router class default remains 111 for standalone experiments.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rag.e8_retriever import E8LatticeDB


@dataclass(frozen=True)
class HammingMatch:
    value: Any
    hamming_distance: int
    stored_key: bytes


class HammingRouter:
    """Nearest-neighbor semantic cache using E8 Hamming distance.

    Store canonical query→value pairs with ``add()``. At query time, ``lookup()``
    finds the stored key with minimum Hamming distance. If that distance is within
    ``threshold``, it is a cache hit and the associated value is returned.

    The default threshold of 111 works well with BAAI/bge-large-en-v1.5 for
    semantically distinct topic clusters. For tighter domains (more similar topics),
    lower the threshold. For more paraphrase recall, raise it carefully.
    """

    def __init__(self, encoder: Any, d_model: int, threshold: int = 111) -> None:
        self._encoder = encoder
        self._d_model = d_model
        self._lattice = E8LatticeDB(d_model=d_model)
        self.threshold = threshold
        self._keys: list[np.ndarray] = []   # each is uint8 shape (n_blocks,)
        self._values: list[Any] = []

    @classmethod
    def from_model(cls, model_name_or_path: str, threshold: int = 111) -> "HammingRouter":
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_name_or_path)
        return cls(encoder=enc, d_model=enc.get_embedding_dimension(), threshold=threshold)

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def _emb_to_key_arr(self, embedding: np.ndarray) -> np.ndarray:
        key_bytes = self._lattice._quantize_to_indices(
            torch.tensor(embedding, dtype=torch.float32)
        )
        return np.frombuffer(key_bytes, dtype=np.uint8).copy()

    def e8_key(self, text: str) -> bytes:
        """Return the raw 128-byte E8 key for a text string."""
        emb = self._encoder.encode([text], normalize_embeddings=True)[0]
        return self._emb_to_key_arr(emb).tobytes()

    # ------------------------------------------------------------------
    # Store operations
    # ------------------------------------------------------------------

    def add(self, key_text: str, value: Any) -> bytes:
        """Encode key_text to an E8 key, store it with value. Returns the raw key."""
        emb = self._encoder.encode([key_text], normalize_embeddings=True)[0]
        key_arr = self._emb_to_key_arr(emb)
        self._keys.append(key_arr)
        self._values.append(value)
        return key_arr.tobytes()

    def add_from_key(self, e8_key: bytes, value: Any) -> None:
        """Store a pre-computed E8 key with value (avoids re-encoding)."""
        self._keys.append(np.frombuffer(e8_key, dtype=np.uint8).copy())
        self._values.append(value)

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()

    def __len__(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, query_text: str, threshold: int | None = None) -> HammingMatch | None:
        """Find the nearest stored key for query_text. Returns None on miss."""
        if not self._keys:
            return None
        emb = self._encoder.encode([query_text], normalize_embeddings=True)[0]
        query_arr = self._emb_to_key_arr(emb)
        return self._nearest(query_arr, threshold)

    def lookup_key(self, e8_key: bytes, threshold: int | None = None) -> HammingMatch | None:
        """Find the nearest stored key for a pre-computed E8 key. Returns None on miss."""
        if not self._keys:
            return None
        query_arr = np.frombuffer(e8_key, dtype=np.uint8)
        return self._nearest(query_arr, threshold)

    def _nearest(self, query_arr: np.ndarray, threshold: int | None) -> HammingMatch | None:
        thresh = threshold if threshold is not None else self.threshold
        stored = np.stack(self._keys)               # [N, n_blocks]
        dists = np.sum(stored != query_arr, axis=1) # [N] — block-level Hamming
        min_idx = int(np.argmin(dists))
        min_dist = int(dists[min_idx])
        if min_dist > thresh:
            return None
        return HammingMatch(
            value=self._values[min_idx],
            hamming_distance=min_dist,
            stored_key=self._keys[min_idx].tobytes(),
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def fragmentation_score(
        self, cluster_texts: list[str], threshold: int | None = None
    ) -> float:
        """Fraction of within-cluster text pairs that match within threshold (recall proxy)."""
        thresh = threshold if threshold is not None else self.threshold
        embs = self._encoder.encode(cluster_texts, normalize_embeddings=True)
        keys = np.stack([self._emb_to_key_arr(e) for e in embs])  # [N, n_blocks]
        n = len(keys)
        if n < 2:
            return 1.0
        total = same = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if int(np.sum(keys[i] != keys[j])) <= thresh:
                    same += 1
        return same / total

    def separation_score(
        self, anchor_texts: list[str], threshold: int | None = None
    ) -> float:
        """Fraction of cross-cluster anchor pairs that stay beyond threshold (precision proxy).

        anchor_texts should contain one representative text per cluster.
        """
        thresh = threshold if threshold is not None else self.threshold
        embs = self._encoder.encode(anchor_texts, normalize_embeddings=True)
        keys = np.stack([self._emb_to_key_arr(e) for e in embs])  # [N, n_blocks]
        n = len(keys)
        if n < 2:
            return 1.0
        total = diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if int(np.sum(keys[i] != keys[j])) > thresh:
                    diff += 1
        return diff / total

    # ------------------------------------------------------------------
    # Threshold calibration from labeled pairs
    # ------------------------------------------------------------------

    def _batch_pair_hamming(self, pairs: list[tuple[str, str]]) -> list[int]:
        """Batch-encode text pairs; return per-pair block-level Hamming distances."""
        if not pairs:
            return []
        flat = [text for pair in pairs for text in pair]
        embs = self._encoder.encode(flat, normalize_embeddings=True)
        keys = [self._emb_to_key_arr(e) for e in embs]
        return [int(np.sum(keys[2 * i] != keys[2 * i + 1])) for i in range(len(pairs))]

    def gap_stats(
        self,
        paraphrase_pairs: list[tuple[str, str]],
        near_miss_pairs: list[tuple[str, str]],
    ) -> dict:
        """Measure the Hamming gap between paraphrase and near-miss pair distributions.

        ``gap = near_miss_p5 - paraphrase_p95``.
        A positive gap means a threshold exists where FP=0 at non-zero recall on these pairs.
        A gap <= 0 means the base model cannot separate the populations — training is needed
        before a safe threshold can be chosen.

        Always check ``n_paraphrase_pairs`` / ``n_near_miss_pairs``: a result from 10 pairs
        is not reliable enough for production deployment.
        """
        if not paraphrase_pairs:
            raise ValueError("paraphrase_pairs must not be empty")
        if not near_miss_pairs:
            raise ValueError("near_miss_pairs must not be empty")

        para_dists = self._batch_pair_hamming(paraphrase_pairs)
        nm_dists = self._batch_pair_hamming(near_miss_pairs)

        def _dist_stats(dists: list[int]) -> dict:
            arr = np.array(dists, dtype=float)
            return {
                "n": len(dists),
                "min": int(arr.min()),
                "p5": float(np.percentile(arr, 5)),
                "mean": round(float(arr.mean()), 2),
                "p95": float(np.percentile(arr, 95)),
                "max": int(arr.max()),
            }

        para_s = _dist_stats(para_dists)
        nm_s = _dist_stats(nm_dists)
        gap = nm_s["p5"] - para_s["p95"]

        n_blocks = self._d_model // 8
        step = max(1, n_blocks // 20)
        thresholds = sorted(set(range(0, n_blocks + 1, step)) | {n_blocks})
        threshold_table = []
        for t in thresholds:
            tp = sum(1 for d in para_dists if d <= t)
            fp = sum(1 for d in nm_dists if d <= t)
            threshold_table.append({
                "threshold": t,
                "recall": round(tp / len(para_dists), 3) if para_dists else 0.0,
                "fp_rate": round(fp / len(nm_dists), 3) if nm_dists else 0.0,
            })

        return {
            "paraphrase": para_s,
            "near_miss": nm_s,
            "gap": round(gap, 2),
            "threshold_table": threshold_table,
            "n_paraphrase_pairs": len(para_dists),
            "n_near_miss_pairs": len(nm_dists),
        }

    def calibrate_threshold(
        self,
        paraphrase_pairs: list[tuple[str, str]],
        near_miss_pairs: list[tuple[str, str]],
        *,
        fp_budget: float = 0.0,
    ) -> dict:
        """Find the highest-recall threshold where FP rate does not exceed fp_budget.

        fp_budget=0.0 (default) means zero false positives — strictest and safest
        for production LLM caches where wrong answers are worse than cache misses.

        Returns ``threshold=-1`` with ``recall=0.0`` only if not even threshold=0
        satisfies the budget (practically impossible; indicates data error).
        Check ``n_paraphrase_pairs`` and ``n_near_miss_pairs`` in the result — a
        threshold calibrated from fewer than ~100 pairs per type is unreliable.
        """
        if not paraphrase_pairs:
            raise ValueError("paraphrase_pairs must not be empty")
        if not near_miss_pairs:
            raise ValueError("near_miss_pairs must not be empty")

        para_dists = sorted(self._batch_pair_hamming(paraphrase_pairs))
        nm_dists = sorted(self._batch_pair_hamming(near_miss_pairs))

        n_para = len(para_dists)
        n_nm = len(nm_dists)
        n_blocks = self._d_model // 8
        best_threshold = -1
        best_recall = 0.0
        best_fp_rate = 0.0

        for t in range(n_blocks + 1):
            fp = bisect.bisect_right(nm_dists, t)
            fp_rate = fp / n_nm if n_nm else 0.0
            if fp_rate > fp_budget:
                break
            tp = bisect.bisect_right(para_dists, t)
            recall = tp / n_para if n_para else 0.0
            if recall >= best_recall:
                best_recall = recall
                best_threshold = t
                best_fp_rate = fp_rate

        return {
            "threshold": best_threshold,
            "recall": round(best_recall, 4),
            "fp_rate": round(best_fp_rate, 4),
            "fp_budget": fp_budget,
            "n_paraphrase_pairs": len(paraphrase_pairs),
            "n_near_miss_pairs": len(near_miss_pairs),
        }


def validate_calibration_data_schema(data: dict) -> None:
    """Validate that the given dictionary conforms to the calibration dataset schema."""
    if not isinstance(data, dict):
        raise ValueError("Calibration JSON must be a dictionary.")

    if "paraphrases" not in data or "near_misses" not in data:
        raise ValueError("Calibration data must contain both 'paraphrases' and 'near_misses' keys.")

    paraphrases = data["paraphrases"]
    near_misses = data["near_misses"]

    if not isinstance(paraphrases, list) or not isinstance(near_misses, list):
        raise ValueError("'paraphrases' and 'near_misses' must be lists.")

    if len(paraphrases) < 1 or len(near_misses) < 1:
        raise ValueError("Calibration data lists must contain at least 1 pair.")

    for list_name, pair_list in [("paraphrases", paraphrases), ("near_misses", near_misses)]:
        for idx, pair in enumerate(pair_list):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each item in '{list_name}' must be a pair of length 2 (at index {idx}).")
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise ValueError(f"Each item in pair must be a string in '{list_name}' (at index {idx}).")


def compute_calibration_data_sha256(data: dict) -> str:
    """Compute deterministic SHA-256 hash of calibration data."""
    import hashlib
    import json
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_precalibrated_artifact_schema(data: dict) -> None:
    """Validate that the given dictionary conforms to the pre-calibrated artifact schema."""
    if not isinstance(data, dict):
        raise ValueError("Pre-calibrated artifact must be a dictionary.")

    required_keys = {
        "artifact_type",
        "artifact_version",
        "model",
        "d_model",
        "calibration_data_sha256",
        "created_at",
        "fp_budget",
        "calibration",
    }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Pre-calibrated artifact is missing required keys: {missing}")

    if data["artifact_type"] != "latticememory_hamming_calibration":
        raise ValueError(f"Invalid artifact_type: expected 'latticememory_hamming_calibration', got {data['artifact_type']!r}")

    if data["artifact_version"] != 1:
        raise ValueError(f"Invalid artifact_version: expected 1, got {data['artifact_version']!r}")

    calibration = data["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("'calibration' key in pre-calibrated artifact must be a dictionary.")

    required_cal_keys = {
        "threshold",
        "recall",
        "fp_rate",
        "n_paraphrase_pairs",
        "n_near_miss_pairs",
    }
    missing_cal = required_cal_keys - calibration.keys()
    if missing_cal:
        raise ValueError(f"Calibration dictionary is missing required keys: {missing_cal}")

