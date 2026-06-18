"""B10-shells: Cell B accuracy vs E8 codebook size (shells=1/2/3).

Tests whether extending the E8 codebook from 240 (shell-1) to 2048 (shell-2)
or 8192 (shell-3) vectors per 8-dim block improves kNN classification accuracy
on BANKING77 20-shot without any training.

Quantization: unit-normalize each 8-dim block, argmax over unit-normalized codebook.
Keys stored as uint16[128] for uniform handling across all shell counts.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_MODEL = "dfrokido/bge-large-e8-snap"
SHOTS      = 20
SEED       = 42
RESULTS_DIR = Path(__file__).parent / "results"


# ── E8 shell generators ───────────────────────────────────────────────────────

def _shell1_vecs() -> np.ndarray:
    vecs = []
    for i, j in itertools.combinations(range(8), 2):
        for si, sj in itertools.product([1.0, -1.0], repeat=2):
            v = [0.0] * 8; v[i] = si; v[j] = sj
            vecs.append(v)
    for mask in range(256):
        signs = [1.0 if (mask >> b) & 1 == 0 else -1.0 for b in range(8)]
        if signs.count(-1.0) % 2 == 0:
            vecs.append([s * 0.5 for s in signs])
    a = np.array(vecs, dtype=np.float32)
    assert a.shape == (240, 8), f"shell1: expected (240,8), got {a.shape}"
    return a


def _shell2_vecs() -> np.ndarray:
    vecs = set()
    for i in range(8):
        for s in [2.0, -2.0]:
            r = [0.0] * 8; r[i] = s; vecs.add(tuple(r))
    for idxs in itertools.combinations(range(8), 4):
        for signs in itertools.product([1.0, -1.0], repeat=4):
            r = [0.0] * 8
            for idx, s in zip(idxs, signs): r[idx] = s
            vecs.add(tuple(r))
    for i in range(8):
        for signs in itertools.product([0.5, -0.5], repeat=8):
            r = list(signs)
            r[i] = 1.5 if signs[i] > 0 else -1.5
            if sum(1 for s in r if s < 0) % 2 == 0:
                vecs.add(tuple(r))
    return np.array(sorted(vecs), dtype=np.float32)


def _shell3_vecs() -> np.ndarray:
    vecs = set()
    for base in [[2, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]]:
        for p in set(itertools.permutations(base)):
            for signs in itertools.product([1, -1], repeat=8):
                vecs.add(tuple(p[k] * signs[k] for k in range(8)))
    base3 = [0.5] * 6 + [1.5, 1.5]
    for p in set(itertools.permutations(base3)):
        for signs in itertools.product([1, -1], repeat=8):
            r = tuple(p[k] * signs[k] for k in range(8))
            if sum(1 for s in r if s < 0) % 2 == 0:
                vecs.add(r)
    return np.array(sorted(vecs), dtype=np.float32)


def build_codebook(shells: int) -> np.ndarray:
    """Unit-normalized E8 codebook from N shells.

    shells=1 ->  240 codewords
    shells=2 -> 2048 codewords (2400 total, truncated to nearest power-of-2)
    shells=3 -> 8192 codewords (~9120 total, truncated to nearest power-of-2)
    """
    print(f"  Building shell-{shells} codebook ...", end=" ", flush=True)
    t0 = time.perf_counter()
    parts = [_shell1_vecs()]
    if shells >= 2:
        parts.append(_shell2_vecs())
    if shells >= 3:
        parts.append(_shell3_vecs())
    raw = np.unique(np.concatenate(parts, axis=0), axis=0)
    if shells > 1:
        target = 2 ** int(math.log2(raw.shape[0]))
        raw = raw[:target]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cb = (raw / np.maximum(norms, 1e-9)).astype(np.float32)
    print(f"{cb.shape[0]} codewords  ({time.perf_counter()-t0:.1f}s)")
    return cb


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_batched(embs_np: np.ndarray, codebook: np.ndarray, device: str) -> np.ndarray:
    """Quantize (N, 1024) float embeddings using codebook -> (N, 128) uint16 keys.

    Approach: unit-normalize each 8-dim block, argmax over unit-normalized codebook.
    No Babai snapping — pure cosine-nearest-codeword.
    """
    N, D = embs_np.shape
    n_blocks = D // 8  # 128
    K = codebook.shape[0]

    cb_t = torch.tensor(codebook, dtype=torch.float32, device=device)  # (K, 8)
    keys = np.zeros((N, n_blocks), dtype=np.uint16)

    CHUNK = 128  # texts per chunk; 128*128=16384 blocks per matmul
    for start in range(0, N, CHUNK):
        batch_np = embs_np[start:start + CHUNK]
        batch = torch.tensor(batch_np, dtype=torch.float32, device=device)  # (B, 1024)
        blocks = batch.reshape(-1, 8)  # (B*128, 8)
        norms = blocks.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        unit_blocks = blocks / norms  # (B*128, 8)
        dots = unit_blocks @ cb_t.T  # (B*128, K)
        idx = dots.argmax(dim=-1).cpu().numpy().reshape(-1, n_blocks)
        keys[start:start + CHUNK] = idx.astype(np.uint16)

    return keys


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_banking77(shots: int, seed: int):
    from datasets import load_dataset
    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    test_ds  = load_dataset("legacy-datasets/banking77", split="test")
    label_names = train_ds.features["label"].names
    rng = random.Random(seed)
    by_intent: dict[str, list[str]] = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])
    train_index = {i: rng.sample(t, min(shots, len(t))) for i, t in by_intent.items()}
    test_queries = [(ex["text"], label_names[ex["label"]]) for ex in test_ds]
    return train_index, test_queries


# ── Per-shell eval ────────────────────────────────────────────────────────────

def eval_shells(
    train_embs: np.ndarray,
    train_labels: list[str],
    test_embs: np.ndarray,
    test_labels: list[str],
    shells: int,
    device: str,
) -> tuple[float, float, list]:
    """Return (accuracy, ms_per_query, top_confusions)."""
    codebook = build_codebook(shells)
    t0 = time.perf_counter()
    tr_keys = encode_batched(train_embs, codebook, device)
    te_keys = encode_batched(test_embs, codebook, device)
    enc_s = time.perf_counter() - t0
    print(f"  Encode {len(train_embs)}+{len(test_embs)} texts: {enc_s:.1f}s")

    t0 = time.perf_counter()
    dists = np.sum(te_keys[:, None, :] != tr_keys[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1)
    lookup_ms = (time.perf_counter() - t0) * 1000

    confusions: dict[tuple[str, str], int] = defaultdict(int)
    correct = 0
    for pred_idx, true_label in zip(preds, test_labels):
        if train_labels[pred_idx] == true_label:
            correct += 1
        else:
            confusions[(true_label, train_labels[pred_idx])] += 1

    acc = correct / len(test_labels)
    ms_per_q = lookup_ms / len(test_labels)
    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:5]
    print(f"  Lookup: {lookup_ms:.0f}ms  acc={acc*100:.2f}%")
    return acc, ms_per_q, top_conf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("B10-SHELLS: Cell B accuracy vs E8 codebook depth")
    print(f"  base model: {BASE_MODEL}")
    print(f"  {SHOTS}-shot BANKING77  device={device}")
    print("=" * 65)

    print("\nLoading BANKING77 ...")
    train_index, test_queries = load_banking77(SHOTS, SEED)
    intents      = list(train_index.keys())
    train_texts  = [t for i in intents for t in train_index[i]]
    train_labels = [i for i in intents for _ in train_index[i]]
    test_texts   = [q for q, _ in test_queries]
    test_labels  = [l for _, l in test_queries]
    print(f"  {len(train_texts)} train  {len(test_texts)} test  {len(intents)} intents")

    from sentence_transformers import SentenceTransformer
    print(f"\nLoading encoder ({BASE_MODEL}) ...")
    model = SentenceTransformer(BASE_MODEL, device=device)

    print("Encoding all texts (float) ...")
    t0 = time.perf_counter()
    train_embs = model.encode(train_texts, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=False, convert_to_numpy=True)
    test_embs  = model.encode(test_texts,  batch_size=64, show_progress_bar=False,
                              normalize_embeddings=False, convert_to_numpy=True)
    print(f"  Done in {time.perf_counter()-t0:.1f}s  "
          f"train={train_embs.shape}  test={test_embs.shape}")

    results = {}
    for shells in [1, 2, 3]:
        print(f"\n--- Shell-{shells} ---")
        acc, ms, conf = eval_shells(
            train_embs, train_labels,
            test_embs, test_labels,
            shells, device,
        )
        results[f"shell{shells}"] = {"accuracy": round(acc, 4), "ms_per_query": round(ms, 4)}
        print("  Top confusions:")
        for (t, p), c in conf:
            print(f"    {t:35s} -> {p:35s} ({c}x)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("RESULTS: Cell B accuracy vs E8 shell depth  (20-shot BANKING77)")
    print("=" * 65)
    print(f"  {'Shells':>8}  {'Codebook':>10}  {'Accuracy':>10}  {'vs Shell-1':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    cb_sizes = {1: 240, 2: 2048, 3: 8192}
    base_acc = results["shell1"]["accuracy"]
    for s in [1, 2, 3]:
        acc = results[f"shell{s}"]["accuracy"]
        delta = (acc - base_acc) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {s:>8}  {cb_sizes[s]:>10}  {acc*100:>9.2f}%  {sign}{delta:.2f} pp")
    print()
    print(f"  Cell A (cosine kNN, 20-shot): 90.42%")
    print(f"  Shell-1 gap vs cosine: {(0.9042 - base_acc)*100:.2f} pp")
    if "shell3" in results:
        shell3_gap = (0.9042 - results["shell3"]["accuracy"]) * 100
        print(f"  Shell-3 gap vs cosine: {shell3_gap:.2f} pp")

    (RESULTS_DIR / "b10_shells.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to results/b10_shells.json")


if __name__ == "__main__":
    main()
