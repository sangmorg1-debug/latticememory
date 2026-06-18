"""B10 Cell D only: E8 Hamming kNN with trained encoder.

Standalone script -- no WandB, no progress bars.
Loads existing trained checkpoint and evaluates on BANKING77 test set.
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR   = Path(__file__).parent / "results"
TRAINED_PATH  = RESULTS_DIR / "b10_banking77_trained" / "best_snap_encoder"
SHOTS = 10
SEED  = 42


def load_banking77(shots, seed):
    from datasets import load_dataset
    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    test_ds  = load_dataset("legacy-datasets/banking77", split="test")
    label_names = train_ds.features["label"].names
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])
    train_index = {intent: rng.sample(texts, min(shots, len(texts)))
                   for intent, texts in by_intent.items()}
    test_queries = [(ex["text"], label_names[ex["label"]]) for ex in test_ds]
    return train_index, test_queries, label_names


def main():
    print("B10 Cell D: E8 Hamming kNN, trained encoder")
    print(f"Checkpoint: {TRAINED_PATH}")

    if not TRAINED_PATH.exists():
        print(f"ERROR: trained checkpoint not found at {TRAINED_PATH}")
        sys.exit(1)

    print("Loading BANKING77 ...")
    train_index, test_queries, _ = load_banking77(SHOTS, SEED)
    intents = list(train_index.keys())
    train_texts  = [t for intent in intents for t in train_index[intent]]
    train_labels = [intent for intent in intents for _ in train_index[intent]]
    test_texts   = [q for q, _ in test_queries]
    test_labels  = [lbl for _, lbl in test_queries]
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    # --- Encode with trained SentenceTransformer (one call, no WandB) ---
    print("Loading trained encoder ...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(TRAINED_PATH), device=device)
    print(f"  Device: {device}")

    print("Encoding train texts ...")
    t0 = time.perf_counter()
    train_embs = model.encode(train_texts, batch_size=64,
                              show_progress_bar=False, normalize_embeddings=False,
                              convert_to_numpy=False)  # keep as tensor
    print(f"  {len(train_texts)} texts in {time.perf_counter()-t0:.1f}s")

    print("Encoding test texts ...")
    t0 = time.perf_counter()
    test_embs = model.encode(test_texts, batch_size=64,
                             show_progress_bar=False, normalize_embeddings=False,
                             convert_to_numpy=False)
    print(f"  {len(test_texts)} texts in {time.perf_counter()-t0:.1f}s")

    # --- Apply E8 quantization ---
    print("Applying E8 quantization ...")
    from latticememory import LatticeIndex
    idx = LatticeIndex(model=str(TRAINED_PATH))
    mem = idx._runtime.memory

    t0 = time.perf_counter()
    train_keys = np.zeros((len(train_texts), 128), dtype=np.uint8)
    for i, emb in enumerate(train_embs):
        key_bytes = mem.lattice_key_for(emb)
        train_keys[i] = np.frombuffer(key_bytes, dtype=np.uint8)

    test_keys = np.zeros((len(test_texts), 128), dtype=np.uint8)
    for i, emb in enumerate(test_embs):
        key_bytes = mem.lattice_key_for(emb)
        test_keys[i] = np.frombuffer(key_bytes, dtype=np.uint8)

    quant_sec = time.perf_counter() - t0
    print(f"  E8 quantization: {quant_sec:.1f}s for {len(train_texts)+len(test_texts)} texts")

    # --- Hamming kNN ---
    print("Computing Hamming kNN ...")
    t0 = time.perf_counter()
    dists = np.sum(test_keys[:, None, :] != train_keys[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1)
    lookup_ms = (time.perf_counter() - t0) * 1000

    confusions = defaultdict(int)
    correct = 0
    for pred_idx, true_label in zip(preds, test_labels):
        if train_labels[pred_idx] == true_label:
            correct += 1
        else:
            confusions[(true_label, train_labels[pred_idx])] += 1

    cell_d_acc = correct / len(test_labels)
    ms_per_query = lookup_ms / len(test_labels)

    print(f"\nCell D accuracy: {cell_d_acc*100:.2f}%  ({ms_per_query:.4f} ms/query lookup)")

    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:10]
    print("Top confusions (true -> pred):")
    for (true_i, pred_i), cnt in top_conf[:8]:
        print(f"  {true_i:40s} -> {pred_i:40s}  ({cnt}x)")

    # --- 2x2 summary from saved intermediate + cell D ---
    inter_path = RESULTS_DIR / "b10_banking77_intermediate.json"
    saved = json.loads(inter_path.read_text()) if inter_path.exists() else {}
    cell_a_acc = saved.get("cell_A", {}).get("accuracy", None)
    cell_b_acc = saved.get("cell_B", {}).get("accuracy", None)

    print()
    print("=" * 65)
    print("  BANKING77 2x2 FINAL RESULTS")
    print("=" * 65)
    print(f"  {'':25s}  {'Cosine kNN':>12}  {'E8 Hamming':>12}")
    if cell_a_acc and cell_b_acc:
        print(f"  {'Base encoder':25s}  {cell_a_acc*100:>11.2f}%  {cell_b_acc*100:>11.2f}%")
        print(f"  {'SnapTrained (1ep)':25s}  {'82.40':>12}%  {cell_d_acc*100:>11.2f}%")
        print()
        print(f"  Training benefit (E8): {(cell_d_acc - cell_b_acc)*100:+.2f} pp")
        print(f"  vs cosine baseline:    {(cell_d_acc - cell_a_acc)*100:+.2f} pp vs Cell A ({cell_a_acc*100:.2f}%)")
        print(f"  Published 10-shot SOTA (SetFit): ~85-88%")
        if cell_d_acc >= cell_a_acc:
            print(f"\n  *** Cell D ({cell_d_acc*100:.2f}%) >= Cell A ({cell_a_acc*100:.2f}%) ***")
            print(f"  Trained E8 Hamming routing MATCHES or BEATS cosine kNN.")
        elif cell_d_acc >= 0.85:
            print(f"\n  Cell D ({cell_d_acc*100:.2f}%) is competitive with published SOTA.")
        else:
            print(f"\n  Cell D ({cell_d_acc*100:.2f}%) improved over base E8 by "
                  f"{(cell_d_acc - cell_b_acc)*100:+.2f} pp.")
            print(f"  Gap to Cell A: {(cell_a_acc - cell_d_acc)*100:.2f} pp. More epochs would help.")

    result = {
        "cell_D": {"accuracy": round(cell_d_acc, 4), "ms_per_query": round(ms_per_query, 6),
                   "top_confusions": [{"true": a, "pred": b, "count": c}
                                      for (a, b), c in top_conf]},
        "cell_C": {"accuracy": 0.824, "note": "from prior run"},
    }
    if saved:
        result.update(saved)
    (RESULTS_DIR / "b10_banking77_2x2.json").write_text(json.dumps(result, indent=2))
    print("\nSaved to b10_banking77_2x2.json")
    print("DONE")


if __name__ == "__main__":
    main()
