"""Validate that proxy/router Hamming thresholds still hold after the quantizer change.

The quantizer was simplified from Babai-snapping to direct unit-normalize→argmax.
That changes keys near Voronoi boundaries, potentially shifting the Hamming distance
distributions that the proxy threshold (default=70) and router default (111) were
calibrated against.

This script:
1. Samples BANKING77 test pairs (intra-intent = paraphrase proxy, inter-intent = near-miss proxy)
2. Computes block-level Hamming distances with the new quantizer
3. Reports mean/p5/p95 per group and the gap
4. Checks whether the documented thresholds still make sense

Expected output (for sanity check):
  intra-intent:  p95 < 120 (low-distance, same intent)
  inter-intent:  p5  > 100 (high-distance, different intents)
  gap > 0 means a separating threshold exists

Run:
  python benchmarks/validate_hamming_thresholds.py
"""
from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_MODEL = "dfrokido/bge-large-e8-snap"
SEED = 42
N_INTRA_PAIRS = 200   # paraphrase pairs (same intent, different texts)
N_INTER_PAIRS = 200   # near-miss pairs (different intents, random cross-pairs)


def load_pairs(seed: int):
    from datasets import load_dataset

    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    label_names = train_ds.features["label"].names
    rng = random.Random(seed)

    by_intent: dict[str, list[str]] = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])

    intents = list(by_intent.keys())

    # Intra-intent pairs: random 2-text sample from same intent
    intra = []
    while len(intra) < N_INTRA_PAIRS:
        intent = rng.choice(intents)
        texts = by_intent[intent]
        if len(texts) >= 2:
            a, b = rng.sample(texts, 2)
            intra.append((a, b))

    # Inter-intent pairs: random texts from different intents
    inter = []
    while len(inter) < N_INTER_PAIRS:
        i1, i2 = rng.sample(intents, 2)
        a = rng.choice(by_intent[i1])
        b = rng.choice(by_intent[i2])
        inter.append((a, b))

    return intra, inter


def encode_all(texts: list[str], model, device: str) -> np.ndarray:
    return model.encode(
        texts, batch_size=64, show_progress_bar=False,
        normalize_embeddings=False, convert_to_numpy=True,
    )


def compute_e8_keys_numpy(embs: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Unit-normalize each 8-dim block, argmax over unit-normalized codebook. Returns (N, 128) uint8."""
    N, D = embs.shape
    n_blocks = D // 8
    # (N, n_blocks, 8)
    blocks = embs.reshape(N, n_blocks, 8).astype(np.float32)
    norms = np.linalg.norm(blocks, axis=-1, keepdims=True).clip(min=1e-8)
    unit_blocks = blocks / norms                             # (N, n_blocks, 8)
    # codebook: (240, 8) already unit-normalised
    dots = np.einsum("nbi,ki->nbk", unit_blocks, codebook)  # (N, n_blocks, 240)
    indices = dots.argmax(axis=-1).astype(np.uint8)          # (N, n_blocks)
    return indices


def _shell1_codebook() -> np.ndarray:
    from itertools import combinations
    vecs = []
    for i, j in combinations(range(8), 2):
        for si in [1.0, -1.0]:
            for sj in [1.0, -1.0]:
                v = [0.0] * 8; v[i] = si; v[j] = sj
                vecs.append(v)
    for mask in range(256):
        signs = [1.0 if (mask >> b) & 1 == 0 else -1.0 for b in range(8)]
        if signs.count(-1.0) % 2 == 0:
            vecs.append([s * 0.5 for s in signs])
    raw = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / norms).astype(np.float32)


def dist_stats(dists: np.ndarray) -> dict:
    return {
        "n":    len(dists),
        "min":  int(dists.min()),
        "p5":   float(np.percentile(dists, 5)),
        "mean": round(float(dists.mean()), 2),
        "p95":  float(np.percentile(dists, 95)),
        "max":  int(dists.max()),
    }


def main():
    print("=" * 65)
    print("Hamming Threshold Validation — new quantizer vs proxy defaults")
    print(f"  model: {BASE_MODEL}")
    print(f"  {N_INTRA_PAIRS} intra-intent pairs  {N_INTER_PAIRS} inter-intent pairs")
    print("=" * 65)

    print("\nLoading BANKING77 pairs ...")
    intra_pairs, inter_pairs = load_pairs(SEED)
    print(f"  {len(intra_pairs)} intra-intent (paraphrase proxy)")
    print(f"  {len(inter_pairs)} inter-intent (near-miss proxy)")

    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading encoder ({BASE_MODEL}, device={device}) ...")
    model = SentenceTransformer(BASE_MODEL, device=device)

    print("Building codebook ...")
    codebook = _shell1_codebook()  # (240, 8) unit-normalised

    print("Encoding all texts ...")
    all_texts = [t for pair in intra_pairs + inter_pairs for t in pair]
    all_embs = encode_all(all_texts, model, device)

    print("Computing E8 keys ...")
    all_keys = compute_e8_keys_numpy(all_embs, codebook)  # (N*2, 128)

    n_intra = len(intra_pairs)
    n_inter = len(inter_pairs)
    intra_keys = all_keys[:n_intra * 2]
    inter_keys = all_keys[n_intra * 2:]

    intra_dists = np.array([
        int(np.sum(intra_keys[2*i] != intra_keys[2*i+1]))
        for i in range(n_intra)
    ])
    inter_dists = np.array([
        int(np.sum(inter_keys[2*i] != inter_keys[2*i+1]))
        for i in range(n_inter)
    ])

    intra_s = dist_stats(intra_dists)
    inter_s = dist_stats(inter_dists)
    gap = inter_s["p5"] - intra_s["p95"]

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Intra-intent (paraphrase proxy): n={intra_s['n']}")
    print(f"    min={intra_s['min']}  p5={intra_s['p5']:.1f}  "
          f"mean={intra_s['mean']}  p95={intra_s['p95']:.1f}  max={intra_s['max']}")
    print()
    print(f"  Inter-intent (near-miss proxy):  n={inter_s['n']}")
    print(f"    min={inter_s['min']}  p5={inter_s['p5']:.1f}  "
          f"mean={inter_s['mean']}  p95={inter_s['p95']:.1f}  max={inter_s['max']}")
    print()
    print(f"  Gap (inter_p5 - intra_p95): {gap:+.1f}")
    print()

    # Threshold sweep
    print("  Threshold sweep (recall = intra within threshold, fp_rate = inter within threshold):")
    print(f"  {'threshold':>10}  {'recall':>8}  {'fp_rate':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for t in [60, 70, 80, 90, 100, 105, 110, 111, 115, 120, 125, 128]:
        recall  = float(np.mean(intra_dists <= t))
        fp_rate = float(np.mean(inter_dists <= t))
        marker = " <-- proxy default" if t == 70 else (" <-- router default" if t == 111 else "")
        print(f"  {t:>10}  {recall:>7.1%}  {fp_rate:>7.1%}{marker}")

    print()
    if gap > 0:
        print("  PASS: gap > 0 — a separating threshold exists with the new quantizer.")
    else:
        print("  WARNING: gap <= 0 — paraphrase and near-miss distributions overlap.")
        print("  Proxy default thresholds may produce false positives.")
        print("  Consider re-calibrating with calibrate_proxy.py on your domain.")

    print()
    print("  Reference (pre-change empirical, bge-large-en-v1.5, diverse topics):")
    print("    intra: mean ~110, inter: mean ~122, gap: ~2pp at t=111")
    print("  If your numbers are similar, the quantizer change did not shift distributions.")


if __name__ == "__main__":
    main()
