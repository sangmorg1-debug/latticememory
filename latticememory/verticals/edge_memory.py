"""LatticeEdgeMemory — semantic memory for constrained environments.

New capability class (no direct predecessor at this size/speed point).

Existing approaches:
  - Store full float32 embeddings on device: 4096 bytes each, needs 4MB for
    1000 "memories," requires transformer at inference time → infeasible on edge.
  - Store nothing, query cloud on every encounter → latency + connectivity required.

LatticeEdgeMemory stores 128-byte E8 keys only. The encoder runs once during a
"learning" phase (on server or workstation), producing key_hex values. The key
store is then deployed to the edge device. At recognition time: Hamming scan
over the key table, O(N) on 128-byte keys, no encoder, no GPU, no network.

4MB of storage = 32,768 semantic memories. Each recognition call is ~0.01ms
for 1000 entries on any modern microcontroller.

Persistence format: compact binary.
  Header: 4 bytes magic (0x4C4D454D), 4 bytes uint32 entry count
  Per entry: 128 bytes key, 2 bytes label_len, N bytes label UTF-8, 1 byte meta_len, M bytes meta JSON
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MAGIC = b"LMEM"
_KEY_LEN = 128


@dataclass
class EdgeRecognitionResult:
    recognized: bool
    label: str | None
    hamming_distance: int
    confidence: float  # 1 - (distance / _KEY_LEN)


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


class LatticeEdgeMemory:
    """Compact semantic key store for resource-constrained environments.

    No encoder is required at recognition time. key_hex values are produced
    by a full model upstream and stored here as 128-byte binary keys.
    """

    def __init__(
        self,
        *,
        capacity: int = 10_000,
        threshold: int = 20,
        persist_path: str | None = None,
    ) -> None:
        self._capacity = capacity
        self.threshold = threshold
        self._keys: list[bytes] = []
        self._labels: list[str] = []
        self._metadata: list[dict] = []
        if persist_path is not None:
            p = Path(persist_path)
            if p.exists():
                self.load(persist_path)

    # ------------------------------------------------------------------
    # Learning (offline / server-side)
    # ------------------------------------------------------------------

    def learn(self, key_hex: str, label: str, metadata: dict | None = None) -> None:
        """Store a key → label mapping. key_hex is the 256-char E8 key hex string."""
        if len(self) >= self._capacity:
            raise OverflowError(f"LatticeEdgeMemory capacity ({self._capacity}) reached.")
        key = bytes.fromhex(key_hex)
        if len(key) != _KEY_LEN:
            raise ValueError(f"E8 key must be {_KEY_LEN} bytes, got {len(key)}.")
        self._keys.append(key)
        self._labels.append(label)
        self._metadata.append(metadata or {})

    # ------------------------------------------------------------------
    # Recognition (edge / inference-time)
    # ------------------------------------------------------------------

    def recognize(self, key_hex: str) -> EdgeRecognitionResult:
        """Find the nearest stored key by Hamming distance."""
        key = bytes.fromhex(key_hex)
        best_dist = _KEY_LEN + 1
        best_idx = -1
        for i, stored in enumerate(self._keys):
            d = _hamming(key, stored)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx == -1 or best_dist > self.threshold:
            return EdgeRecognitionResult(
                recognized=False,
                label=None,
                hamming_distance=best_dist if best_idx >= 0 else -1,
                confidence=0.0,
            )
        return EdgeRecognitionResult(
            recognized=True,
            label=self._labels[best_idx],
            hamming_distance=best_dist,
            confidence=1.0 - best_dist / _KEY_LEN,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def forget(self, label: str) -> int:
        """Remove all entries with the given label. Returns count removed."""
        keep = [(k, l, m) for k, l, m in zip(self._keys, self._labels, self._metadata) if l != label]
        removed = len(self._keys) - len(keep)
        if keep:
            self._keys, self._labels, self._metadata = map(list, zip(*keep))
        else:
            self._keys, self._labels, self._metadata = [], [], []
        return removed

    def memory_bytes(self) -> int:
        """Total bytes used by the key store (keys only, no metadata)."""
        return len(self._keys) * _KEY_LEN

    def __len__(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------
    # Persistence (compact binary)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack(">I", len(self._keys)))
            for key, label, meta in zip(self._keys, self._labels, self._metadata):
                label_b = label.encode("utf-8")
                meta_b = json.dumps(meta, separators=(",", ":")).encode("utf-8")
                f.write(key)
                f.write(struct.pack(">H", len(label_b)))
                f.write(label_b)
                f.write(struct.pack(">H", len(meta_b)))
                f.write(meta_b)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"Not a LatticeEdgeMemory file: {path}")
            (count,) = struct.unpack(">I", f.read(4))
            self._keys, self._labels, self._metadata = [], [], []
            for _ in range(count):
                key = f.read(_KEY_LEN)
                (label_len,) = struct.unpack(">H", f.read(2))
                label = f.read(label_len).decode("utf-8")
                (meta_len,) = struct.unpack(">H", f.read(2))
                meta = json.loads(f.read(meta_len).decode("utf-8"))
                self._keys.append(key)
                self._labels.append(label)
                self._metadata.append(meta)
