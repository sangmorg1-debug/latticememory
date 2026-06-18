"""B10 v2: BANKING77 with all 4 training fixes applied.

Fixes vs v1:
  1. val_clusters = ALL 77 intents (was 10)
  2. 10 epochs (was 3)
  3. 20 examples per intent for training (was 10)
  4. zero_fp_recall_target=2.0 -- no early stopping (was 0.7)

Pair capping: C(20,2)=190 pairs per intent is too many (~38hr training).
MAX_PAIRS_PER_INTENT=20 keeps positive pairs to 20x77=1540, total ~2300-3000
examples, training ~5 hours on GTX 1660 Ti.

Cells A+B are recomputed for 20-shot index (different from v1 10-shot).
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

RESULTS_DIR = Path(__file__).parent / "results"
BASE_MODEL   = "dfrokido/bge-large-e8-snap"

SHOTS                    = 20    # fix 3: 20 examples per intent
EPOCHS                   = 10    # fix 2: 10 epochs
VAL_EXAMPLES_PER_INTENT  = 5     # all 77 intents, 5 examples each -> 385 val texts, fast obs_eval
ZERO_FP_TARGET           = 2.0   # fix 4: impossible target -> run all epochs
OBS_EVAL_STEPS           = 1000  # checkpoint interval
MAX_PAIRS_PER_INTENT     = 20    # cap positive pairs (from C(20,2)=190) to keep training feasible
FREEZE_LAYERS            = 20
SEED                     = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_banking77(shots: int, seed: int):
    from datasets import load_dataset
    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    test_ds  = load_dataset("legacy-datasets/banking77", split="test")
    label_names = train_ds.features["label"].names
    rng = random.Random(seed)
    by_intent: dict[str, list[str]] = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])
    train_index = {
        intent: rng.sample(texts, min(shots, len(texts)))
        for intent, texts in by_intent.items()
    }
    test_queries = [(ex["text"], label_names[ex["label"]]) for ex in test_ds]
    return train_index, test_queries, label_names


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


def encode_continuous(texts, model_path):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device="cuda" if _cuda() else "cpu")
    return model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def encode_e8_fast(texts, model_path=None):
    """Encode texts to E8 keys: one batch encode call, then quantize."""
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


# ---------------------------------------------------------------------------
# kNN evaluation
# ---------------------------------------------------------------------------

def cosine_knn(train_index, test_queries, model_path):
    intents     = list(train_index.keys())
    train_texts = [t for intent in intents for t in train_index[intent]]
    train_labels= [intent for intent in intents for _ in train_index[intent]]
    print(f"    Encoding {len(train_texts)} train + {len(test_queries)} test ...")
    t0 = time.perf_counter()
    tr_embs  = encode_continuous(train_texts, model_path)
    te_embs  = encode_continuous([q for q, _ in test_queries], model_path)
    enc_sec  = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = te_embs @ tr_embs.T
    preds  = np.argmax(scores, axis=1)
    look_ms = (time.perf_counter() - t0) * 1000
    correct = sum(train_labels[p] == lbl for p, (_, lbl) in zip(preds, test_queries))
    print(f"    Encode: {enc_sec:.1f}s  lookup: {look_ms:.1f}ms total")
    return correct / len(test_queries), look_ms / len(test_queries)


def hamming_knn(train_index, test_queries, model_path=None):
    intents     = list(train_index.keys())
    train_texts = [t for intent in intents for t in train_index[intent]]
    train_labels= [intent for intent in intents for _ in train_index[intent]]
    test_texts  = [q for q, _ in test_queries]
    print(f"    Encoding {len(train_texts)} train + {len(test_texts)} test (E8) ...")
    t0 = time.perf_counter()
    tr_keys = encode_e8_fast(train_texts, model_path)
    te_keys = encode_e8_fast(test_texts,  model_path)
    enc_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    dists = np.sum(te_keys[:, None, :] != tr_keys[None, :, :], axis=2)
    preds = np.argmin(dists, axis=1)
    look_ms = (time.perf_counter() - t0) * 1000
    confusions = defaultdict(int)
    correct = 0
    for p, (_, lbl) in zip(preds, test_queries):
        if train_labels[p] == lbl: correct += 1
        else: confusions[(lbl, train_labels[p])] += 1
    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:10]
    print(f"    Encode: {enc_sec:.1f}s  lookup: {look_ms:.1f}ms total")
    return correct / len(test_queries), look_ms / len(test_queries), top_conf


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_curriculum_examples(curriculum, clusters, max_pairs_per_intent=20):
    from latticememory.training import RoutingTrainingExample
    intent_names = list(clusters.keys())
    other_reps = {
        name: [clusters[n][0] for n in intent_names if n != name][:3]
        for name in intent_names
    }
    cluster_canonical = {name: clusters[name][0] for name in intent_names}

    # Group positive pairs by cluster, cap per intent
    by_cluster: dict[str, list] = defaultdict(list)
    for pp in curriculum["positive_pairs"]:
        by_cluster[pp["cluster"]].append(pp)

    phase1, phase2 = [], []
    for cluster_name, pairs in by_cluster.items():
        # Sort by difficulty (easy first), then cap
        easy = [p for p in pairs if p["difficulty"] == "easy"]
        hard = [p for p in pairs if p["difficulty"] != "easy"]
        selected = (easy + hard)[:max_pairs_per_intent]
        for pp in selected:
            negs = other_reps.get(pp["cluster"], [])
            if not negs: continue
            ex = RoutingTrainingExample(
                query=pp["anchor"], positive=pp["positive"], negatives=negs
            )
            (phase1 if pp["difficulty"] == "easy" else phase2).append(ex)

    phase3 = []
    for hn in curriculum["hard_negative_pairs"]:
        anchor   = hn["anchor"]
        positive = cluster_canonical.get(hn["cluster_a"], anchor)
        if positive == anchor:
            others = [t for t in clusters.get(hn["cluster_a"], []) if t != anchor]
            positive = others[0] if others else anchor
        if anchor == positive: continue
        phase3.append(RoutingTrainingExample(
            query=anchor, positive=positive, negatives=[hn["negative"]]
        ))

    all_examples = phase1 + phase2 + phase3
    print(f"    Phase 1 (easy):         {len(phase1):5d}")
    print(f"    Phase 2 (hard pos):     {len(phase2):5d}  (capped at {max_pairs_per_intent}/intent)")
    print(f"    Phase 3 (hard neg):     {len(phase3):5d}")
    print(f"    Total:                  {len(all_examples):5d}")
    return all_examples


def run_training(clusters, output_dir):
    from latticememory import LatticeIndex
    from latticememory.snap_trainer import SnapTrainer, SnapTrainingConfig

    print("  Building observatory ...")
    index = LatticeIndex()
    for texts in clusters.values():
        index.add(texts)
    obs = index.observatory()

    print("  Generating curriculum ...")
    curriculum = obs.generate_training_curriculum(clusters)
    print(f"    Raw positive pairs: {len(curriculum['positive_pairs'])}  "
          f"hard negatives: {len(curriculum['hard_negative_pairs'])}")

    examples = build_curriculum_examples(
        curriculum, clusters, max_pairs_per_intent=MAX_PAIRS_PER_INTENT
    )
    if len(examples) < 4:
        raise RuntimeError("Too few training examples")

    # Fix 1: ALL 77 intents in val_clusters, VAL_EXAMPLES_PER_INTENT each
    rng = random.Random(SEED)
    val_clusters = {
        intent: rng.sample(texts, min(VAL_EXAMPLES_PER_INTENT, len(texts)))
        for intent, texts in clusters.items()
    }
    n_val_texts = sum(len(v) for v in val_clusters.values())
    print(f"  Val clusters: {len(val_clusters)} intents x {VAL_EXAMPLES_PER_INTENT} = {n_val_texts} texts")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = SnapTrainingConfig(
        epochs=EPOCHS,                                   # fix 2
        batch_size=4,
        gradient_accumulation_steps=4,
        freeze_layers=FREEZE_LAYERS,
        output_dir=str(output_dir),
        obs_eval_every_steps=OBS_EVAL_STEPS,
        zero_fp_recall_target=ZERO_FP_TARGET,            # fix 4
        separation_target=ZERO_FP_TARGET,
        seed=SEED,
    )

    print(f"  SnapTrainer: {EPOCHS} epochs, freeze={FREEZE_LAYERS}, "
          f"eval_every={OBS_EVAL_STEPS}, target={ZERO_FP_TARGET} (no early stop)")

    trainer = SnapTrainer(base_model=BASE_MODEL, val_clusters=val_clusters)
    t0 = time.perf_counter()
    result = trainer.train(examples, config=config)
    elapsed = time.perf_counter() - t0

    print(f"  Done in {elapsed:.0f}s ({elapsed/3600:.1f}h)  "
          f"best zero_fp_recall={result.best_zero_fp_recall:.4f}  "
          f"sep={result.best_separation_score:.4f}  reached={result.reached_target}")

    if result.obs_checkpoints:
        print("  Checkpoint progression (zero_fp_recall):")
        for c in result.obs_checkpoints:
            print(f"    step {c.global_step:4d}  ep{c.epoch}  "
                  f"zfp={c.zero_fp_recall:.4f}  sep={c.separation_score:.4f}  "
                  f"gap={c.hamming_gap:.1f}  ph={c.paraphrase_hamming_mean:.1f}  "
                  f"nh={c.near_miss_hamming_mean:.1f}")

    trained_path = str(output_dir / "best_snap_encoder")
    return trained_path, {
        "elapsed_sec": round(elapsed, 1),
        "best_zero_fp_recall": result.best_zero_fp_recall,
        "best_separation_score": result.best_separation_score,
        "reached_target": result.reached_target,
        "n_examples": len(examples),
        "epochs": EPOCHS,
        "checkpoints": [
            {"step": c.global_step, "epoch": c.epoch,
             "zero_fp_recall": c.zero_fp_recall,
             "separation_score": c.separation_score,
             "hamming_gap": c.hamming_gap,
             "paraphrase_hamming_mean": c.paraphrase_hamming_mean,
             "near_miss_hamming_mean": c.near_miss_hamming_mean}
            for c in result.obs_checkpoints
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("B10 v2: BANKING77 2x2 (all 4 training fixes)")
    print(f"  {SHOTS}-shot | {EPOCHS} epochs | 77 val intents | no early stop")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading BANKING77 ...")
    train_index, test_queries, label_names = load_banking77(SHOTS, SEED)
    n_train = sum(len(v) for v in train_index.values())
    print(f"  Train index: {n_train} examples ({SHOTS}/intent, {len(train_index)} intents)")
    print(f"  Test:        {len(test_queries)} queries")

    results: dict = {"version": "b10_v2", "shots": SHOTS, "seed": SEED,
                     "n_train": n_train, "n_test": len(test_queries),
                     "n_intents": len(label_names)}

    inter_path = RESULTS_DIR / "b10_v2_intermediate.json"
    saved = json.loads(inter_path.read_text()) if inter_path.exists() else {}

    # Cell A
    if "cell_A" in saved:
        cell_a_acc = saved["cell_A"]["accuracy"]
        cell_a_ms  = saved["cell_A"]["ms_per_query"]
        results["cell_A"] = saved["cell_A"]
        print(f"\n  Cell A (cached): {cell_a_acc*100:.2f}%")
    else:
        print("\n  --- CELL A: Cosine kNN, base encoder ---")
        cell_a_acc, cell_a_ms = cosine_knn(train_index, test_queries, BASE_MODEL)
        print(f"  Cell A: {cell_a_acc*100:.2f}%  ({cell_a_ms:.4f} ms/query)")
        results["cell_A"] = {"accuracy": round(cell_a_acc, 4), "ms_per_query": round(cell_a_ms, 6)}
        inter_path.write_text(json.dumps(results, indent=2))

    # Cell B
    if "cell_B" in saved:
        cell_b_acc = saved["cell_B"]["accuracy"]
        cell_b_ms  = saved["cell_B"]["ms_per_query"]
        b_confusions = saved["cell_B"].get("top_confusions", [])
        results["cell_B"] = saved["cell_B"]
        print(f"  Cell B (cached): {cell_b_acc*100:.2f}%")
    else:
        print("\n  --- CELL B: E8 Hamming kNN, base encoder ---")
        cell_b_acc, cell_b_ms, b_confusions_raw = hamming_knn(
            train_index, test_queries, model_path=None
        )
        b_confusions = [{"true": a, "pred": b, "count": c} for (a, b), c in b_confusions_raw]
        print(f"  Cell B: {cell_b_acc*100:.2f}%  ({cell_b_ms:.4f} ms/query)")
        results["cell_B"] = {"accuracy": round(cell_b_acc, 4),
                              "ms_per_query": round(cell_b_ms, 6),
                              "top_confusions": b_confusions}
        inter_path.write_text(json.dumps(results, indent=2))

    print()
    print(f"  Base: A={cell_a_acc*100:.2f}% (cosine)  B={cell_b_acc*100:.2f}% (E8)")
    print(f"  Training target: Cell D > {cell_a_acc*100:.2f}%")
    print(f"  Published {SHOTS}-shot SOTA: ~{85 + (SHOTS-10)*0.3:.0f}-{88 + (SHOTS-10)*0.3:.0f}%")

    # Training
    print("\n  --- TRAINING (all fixes applied) ---")
    train_output_dir = RESULTS_DIR / "b10_v2_trained"
    trained_path, train_meta = run_training(train_index, train_output_dir)
    results["training"] = train_meta
    inter_path.write_text(json.dumps(results, indent=2))

    # Cell C
    print("\n  --- CELL C: Cosine kNN, trained encoder ---")
    cell_c_acc, cell_c_ms = cosine_knn(train_index, test_queries, trained_path)
    print(f"  Cell C: {cell_c_acc*100:.2f}%  ({cell_c_ms:.4f} ms/query)")
    results["cell_C"] = {"accuracy": round(cell_c_acc, 4), "ms_per_query": round(cell_c_ms, 6)}
    inter_path.write_text(json.dumps(results, indent=2))

    # Cell D
    print("\n  --- CELL D: E8 Hamming kNN, trained encoder ---")
    cell_d_acc, cell_d_ms, d_confusions_raw = hamming_knn(
        train_index, test_queries, model_path=trained_path
    )
    d_confusions = [{"true": a, "pred": b, "count": c} for (a, b), c in d_confusions_raw]
    print(f"  Cell D: {cell_d_acc*100:.2f}%  ({cell_d_ms:.4f} ms/query)")
    print("  Top confusions (true -> pred):")
    for conf in d_confusions[:6]:
        print(f"    {conf['true']:40s} -> {conf['pred']:40s}  ({conf['count']}x)")
    results["cell_D"] = {"accuracy": round(cell_d_acc, 4),
                          "ms_per_query": round(cell_d_ms, 6),
                          "top_confusions": d_confusions}

    # Final table
    print()
    print("=" * 70)
    print(f"  B10 v2 FINAL RESULTS  ({SHOTS}-shot, 3080 test queries, 77 intents)")
    print("=" * 70)
    print(f"  {'':25s}  {'Cosine kNN':>12}  {'E8 Hamming':>12}  {'delta':>8}")
    print(f"  {'Base encoder':25s}  {cell_a_acc*100:>11.2f}%  {cell_b_acc*100:>11.2f}%  "
          f"{(cell_b_acc-cell_a_acc)*100:>+7.2f}")
    print(f"  {f'SnapTrained ({EPOCHS}ep)':25s}  {cell_c_acc*100:>11.2f}%  {cell_d_acc*100:>11.2f}%  "
          f"{(cell_d_acc-cell_c_acc)*100:>+7.2f}")
    print(f"  {'Training benefit':25s}  {(cell_c_acc-cell_a_acc)*100:>+11.2f}   "
          f"{(cell_d_acc-cell_b_acc)*100:>+11.2f}")
    print()
    if cell_d_acc >= cell_a_acc:
        print(f"  *** Cell D ({cell_d_acc*100:.2f}%) >= Cell A ({cell_a_acc*100:.2f}%) ***")
        print("  Trained E8 Hamming routing BEATS unquantized cosine kNN baseline.")
    elif cell_d_acc >= 0.85:
        print(f"  Cell D ({cell_d_acc*100:.2f}%) is competitive with published SOTA.")
    else:
        print(f"  Cell D ({cell_d_acc*100:.2f}%), Cell A ({cell_a_acc*100:.2f}%). "
              f"Gap: {(cell_a_acc-cell_d_acc)*100:.2f} pp")

    (RESULTS_DIR / "b10_v2_2x2.json").write_text(json.dumps(results, indent=2))
    print("\nB10 v2 COMPLETE")


if __name__ == "__main__":
    main()
