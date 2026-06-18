"""B10 smoke test: 2 epochs with all 4 fixes, immediate Cell D eval.

Gate: does Cell D (trained E8) beat Cell B (base E8) after 2 epochs?
If yes -> evidence training works, proceed to full 10-epoch run.
If no  -> training is harmful at this scale, Cell B is the product story.

Cell A (20-shot cosine baseline): 90.42% (from prior partial run).
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

RESULTS_DIR  = Path(__file__).parent / "results"
BASE_MODEL   = "dfrokido/bge-large-e8-snap"
SHOTS        = 20
SEED         = 42
CELL_A_ACC   = 0.9042   # from prior run
CELL_A_MS    = 0.0188


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


def load_banking77(shots, seed):
    from datasets import load_dataset
    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    test_ds  = load_dataset("legacy-datasets/banking77", split="test")
    label_names = train_ds.features["label"].names
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])
    train_index = {i: rng.sample(t, min(shots, len(t))) for i, t in by_intent.items()}
    test_queries = [(ex["text"], label_names[ex["label"]]) for ex in test_ds]
    return train_index, test_queries


def encode_e8_fast(texts, model_path=None):
    from sentence_transformers import SentenceTransformer
    from latticememory import LatticeIndex
    mp = model_path or BASE_MODEL
    model = SentenceTransformer(mp, device="cuda" if _cuda() else "cpu")
    embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=False, convert_to_numpy=False)
    idx = LatticeIndex(model=mp)
    mem = idx._runtime.memory
    keys = np.zeros((len(texts), 128), dtype=np.uint8)
    for i, emb in enumerate(embs):
        keys[i] = np.frombuffer(mem.lattice_key_for(emb), dtype=np.uint8)
    return keys


def hamming_knn_acc(train_index, test_queries, model_path=None):
    intents      = list(train_index.keys())
    train_texts  = [t for i in intents for t in train_index[i]]
    train_labels = [i for i in intents for _ in train_index[i]]
    test_texts   = [q for q, _ in test_queries]
    test_labels  = [l for _, l in test_queries]
    print(f"  Encoding {len(train_texts)} train + {len(test_texts)} test ...")
    t0 = time.perf_counter()
    tr_keys = encode_e8_fast(train_texts, model_path)
    te_keys = encode_e8_fast(test_texts, model_path)
    enc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    dists = np.sum(te_keys[:, None, :] != tr_keys[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1)
    look_ms = (time.perf_counter() - t0) * 1000
    correct = sum(train_labels[p] == l for p, l in zip(preds, test_labels))
    confusions = defaultdict(int)
    for p, (_, l) in zip(preds, test_queries):
        if train_labels[p] != l:
            confusions[(l, train_labels[p])] += 1
    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:6]
    print(f"  Encode: {enc_s:.1f}s  lookup: {look_ms:.0f}ms")
    return correct / len(test_labels), look_ms / len(test_labels), top_conf


def train_2ep(train_index):
    from latticememory import LatticeIndex
    from latticememory.training import RoutingTrainingExample
    from latticememory.snap_trainer import SnapTrainer, SnapTrainingConfig

    print("  Building observatory ...")
    index = LatticeIndex()
    for texts in train_index.values():
        index.add(texts)
    obs = index.observatory()

    print("  Generating curriculum ...")
    curriculum = obs.generate_training_curriculum(train_index)

    # Build examples: cap at 20 positive pairs per intent
    intent_names = list(train_index.keys())
    other_reps = {n: [train_index[k][0] for k in intent_names if k != n][:3] for n in intent_names}
    cluster_can = {n: train_index[n][0] for n in intent_names}

    by_cluster = defaultdict(list)
    for pp in curriculum["positive_pairs"]:
        by_cluster[pp["cluster"]].append(pp)

    examples = []
    for name, pairs in by_cluster.items():
        for pp in pairs[:20]:
            negs = other_reps.get(name, [])
            if negs:
                examples.append(RoutingTrainingExample(
                    query=pp["anchor"], positive=pp["positive"], negatives=negs))
    for hn in curriculum["hard_negative_pairs"]:
        pos = cluster_can.get(hn["cluster_a"], hn["anchor"])
        others = [t for t in train_index.get(hn["cluster_a"], []) if t != hn["anchor"]]
        if hn["anchor"] == pos and others:
            pos = others[0]
        if hn["anchor"] != pos:
            examples.append(RoutingTrainingExample(
                query=hn["anchor"], positive=pos, negatives=[hn["negative"]]))

    print(f"  {len(examples)} training examples")

    # ALL 77 intents, 5 examples each for val
    rng = random.Random(SEED)
    val_clusters = {n: rng.sample(t, min(5, len(t))) for n, t in train_index.items()}
    print(f"  Val: {len(val_clusters)} intents x 5 = {sum(len(v) for v in val_clusters.values())} texts")

    output_dir = RESULTS_DIR / "b10_smoke_trained"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SnapTrainingConfig(
        epochs=2,
        batch_size=4,
        gradient_accumulation_steps=4,
        freeze_layers=20,
        output_dir=str(output_dir),
        obs_eval_every_steps=200,
        zero_fp_recall_target=2.0,   # no early stop
        separation_target=2.0,
        seed=SEED,
    )

    trainer = SnapTrainer(base_model=BASE_MODEL, val_clusters=val_clusters)
    t0 = time.perf_counter()
    result = trainer.train(examples, config=config)
    elapsed = time.perf_counter() - t0
    print(f"  Training: {elapsed:.0f}s ({elapsed/60:.1f}min)  "
          f"best_zfp={result.best_zero_fp_recall:.4f}  "
          f"gap={max((c.hamming_gap for c in result.obs_checkpoints), default=0):.2f}")
    for c in result.obs_checkpoints:
        print(f"    step {c.global_step:4d} ep{c.epoch}  "
              f"zfp={c.zero_fp_recall:.4f}  gap={c.hamming_gap:.1f}  "
              f"ph={c.paraphrase_hamming_mean:.1f}  nh={c.near_miss_hamming_mean:.1f}")
    return str(output_dir / "best_snap_encoder")


def main():
    print("=" * 65)
    print("B10 SMOKE TEST: 2ep, all fixes, gate on Cell D > Cell B")
    print("=" * 65)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading BANKING77 ...")
    train_index, test_queries = load_banking77(SHOTS, SEED)
    print(f"  {sum(len(v) for v in train_index.values())} train, {len(test_queries)} test")

    print(f"\nCell A (20-shot cosine, from prior run): {CELL_A_ACC*100:.2f}%")

    print("\n--- Cell B: E8 Hamming, base encoder ---")
    cell_b_acc, cell_b_ms, b_conf = hamming_knn_acc(train_index, test_queries, None)
    print(f"Cell B: {cell_b_acc*100:.2f}%")

    print("\n--- Training: 2 epochs, all 4 fixes ---")
    trained_path = train_2ep(train_index)

    print("\n--- Cell D: E8 Hamming, trained encoder ---")
    cell_d_acc, cell_d_ms, d_conf = hamming_knn_acc(train_index, test_queries, trained_path)
    print(f"Cell D: {cell_d_acc*100:.2f}%")
    print("Top confusions:")
    for (t, p), c in d_conf:
        print(f"  {t:40s} -> {p:40s} ({c}x)")

    print()
    print("=" * 65)
    print("SMOKE TEST RESULT")
    print("=" * 65)
    print(f"  Cell A (cosine, base):    {CELL_A_ACC*100:.2f}%  <- bar to beat")
    print(f"  Cell B (E8, base):        {cell_b_acc*100:.2f}%  <- training must beat this")
    print(f"  Cell D (E8, 2ep trained): {cell_d_acc*100:.2f}%")
    delta = cell_d_acc - cell_b_acc
    print(f"  Delta D-B: {delta*100:+.2f} pp")
    print()
    if delta > 0.02:
        print("  GATE PASSED: training is helping. Proceed to 10-epoch full run.")
    elif delta > 0:
        print("  MARGINAL: training barely helped. Need more epochs to confirm.")
    else:
        print("  GATE FAILED: training hurt E8 accuracy even with all fixes.")
        print("  Cell B (base E8, no training) is the product story.")
        print("  Do NOT proceed to 10-epoch run without diagnosing further.")

    result = {"cell_A": CELL_A_ACC, "cell_B": round(cell_b_acc, 4),
              "cell_D_2ep": round(cell_d_acc, 4), "delta_D_minus_B": round(delta, 4)}
    (RESULTS_DIR / "b10_smoke.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
