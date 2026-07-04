"""Coordinate drift snapshots for semantic-cache encoder changes.

This module intentionally stays small: it records lattice keys for stable
reference texts, then compares a later cache/runtime against that snapshot.
It is useful for catching encoder/model upgrades that would silently change
cache coordinates and make old entries unsafe to serve.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def create_cache_snapshot(cache: Any, reference_texts: list[str]) -> dict[str, Any]:
    """Create a lattice-key snapshot for reference texts.

    Parameters
    ----------
    cache:
        An ``RFSnapSemanticCache``-compatible object with ``runtime._encode_texts``
        and ``runtime.memory.lattice_key_for``.
    reference_texts:
        Stable anchor texts used to detect whether future runtime coordinates
        still match the saved mapping.
    """

    model_id = getattr(cache.runtime, "model_id", None)
    d_model = getattr(cache.runtime, "d_model", None)
    snapshots: list[dict[str, str]] = []

    for text in reference_texts:
        embedding = cache.runtime._encode_texts([text])[0]
        key = cache.runtime.memory.lattice_key_for(embedding)
        snapshots.append({"text": text, "key_hex": key.hex()})

    return {
        "model_id": model_id,
        "d_model": d_model,
        "reference_count": len(reference_texts),
        "snapshots": snapshots,
    }


def compare_cache_snapshot(
    cache: Any,
    snapshot: dict[str, Any],
    drift_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compare current cache coordinates against a saved snapshot."""

    current_model = getattr(cache.runtime, "model_id", None)
    current_dim = getattr(cache.runtime, "d_model", None)
    snapshot_model = snapshot.get("model_id")
    snapshot_dim = snapshot.get("d_model")
    snapshots = snapshot.get("snapshots", [])

    if snapshot_dim is not None and current_dim != snapshot_dim:
        return {
            "mean_hamming_distance": float(current_dim or 0),
            "exact_match_ratio": 0.0,
            "drift_percentage": 100.0,
            "needs_reindexing": True,
            "details": f"Dimension mismatch: snapshot={snapshot_dim}, current={current_dim}",
        }

    model_changed = snapshot_model is not None and current_model != snapshot_model
    if not snapshots:
        return {
            "mean_hamming_distance": 0.0,
            "exact_match_ratio": 1.0,
            "drift_percentage": 0.0,
            "needs_reindexing": model_changed,
            "details": "Empty snapshot reference list",
        }

    distances: list[int] = []
    exact_matches = 0
    for item in snapshots:
        text = item["text"]
        old_key = bytes.fromhex(item["key_hex"])
        embedding = cache.runtime._encode_texts([text])[0]
        new_key = cache.runtime.memory.lattice_key_for(embedding)
        distance = _byte_hamming_distance(old_key, new_key)
        distances.append(distance)
        if distance == 0:
            exact_matches += 1

    mean_distance = float(np.mean(distances))
    exact_ratio = exact_matches / len(snapshots)
    drift_percentage = (1.0 - exact_ratio) * 100.0
    needs_reindexing = mean_distance > drift_threshold or model_changed
    details = (
        f"Mean Hamming: {mean_distance:.2f}, "
        f"Exact matches: {exact_ratio * 100.0:.1f}%. "
        f"Needs reindexing: {needs_reindexing} (threshold={drift_threshold})"
    )
    if model_changed:
        details += f" (Model changed: {snapshot_model} -> {current_model})"

    return {
        "mean_hamming_distance": mean_distance,
        "exact_match_ratio": exact_ratio,
        "drift_percentage": drift_percentage,
        "needs_reindexing": needs_reindexing,
        "details": details,
    }


def _byte_hamming_distance(left: bytes, right: bytes) -> int:
    left_arr = np.frombuffer(left, dtype=np.uint8)
    right_arr = np.frombuffer(right, dtype=np.uint8)
    if left_arr.shape != right_arr.shape:
        max_len = max(left_arr.size, right_arr.size)
        padded_left = np.zeros(max_len, dtype=np.uint8)
        padded_right = np.zeros(max_len, dtype=np.uint8)
        padded_left[: left_arr.size] = left_arr
        padded_right[: right_arr.size] = right_arr
        left_arr = padded_left
        right_arr = padded_right
    return int(np.sum(left_arr != right_arr))
