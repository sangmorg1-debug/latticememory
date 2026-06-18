"""B10: BANKING77 2×2 Routing Benchmark.

2×2 evaluation grid:
                    | Cosine kNN  | E8 Hamming kNN |
  Base encoder      |     A       |       B        |
  SnapTrained       |     C       |       D        |

A = continuous BGE baseline (published comparison point)
B = E8 with no training (quantization cost)
C = continuous after SnapTrainer domain adaptation
D = E8 Hamming routing after SnapTrainer (the product claim)

Setup: 10-shot index (10 examples per intent, seed=42), evaluated on full
test split (3,080 queries, 40 per intent). Published 10-shot SetFit SOTA
on BANKING77 is ~85-88%.

Stages run:
  1. Cells A + B -- no training, ~5 minutes, printed as intermediate results
  2. Curriculum training (SnapTrainer, 3 epochs, freeze_layers=20)
  3. Cells C + D -- same harness, trained encoder
  4. Full 2×2 table + confusion pairs + latency comparison
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
BASE_MODEL   = "dfrokido/bge-large-e8-snap"
SHOTS        = 10          # examples per intent in train index
SEED         = 42
N_SEEDS      = 1           # set to 3 to average over seeds (slower)
VAL_INTENTS  = 10          # intents used for obs_eval during SnapTrainer
OBS_EVAL_STEPS = 300
EPOCHS       = 3
FREEZE_LAYERS = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_banking77(shots: int, seed: int):
    """Returns train_index, test_queries, label_names."""
    from datasets import load_dataset
    train_ds = load_dataset("legacy-datasets/banking77", split="train")
    test_ds  = load_dataset("legacy-datasets/banking77", split="test")
    label_names = train_ds.features["label"].names

    # Group train by intent
    rng = random.Random(seed)
    by_intent: dict[str, list[str]] = defaultdict(list)
    for ex in train_ds:
        by_intent[label_names[ex["label"]]].append(ex["text"])

    # Sample `shots` per intent for the train index
    train_index: dict[str, list[str]] = {}
    for intent, texts in by_intent.items():
        sampled = rng.sample(texts, min(shots, len(texts)))
        train_index[intent] = sampled

    # Test queries
    test_queries = [(ex["text"], label_names[ex["label"]]) for ex in test_ds]
    return train_index, test_queries, label_names


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_continuous(texts: list[str], model_path: str) -> np.ndarray:
    """Encode texts to L2-normalised float32 vectors using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(model_path, device="cuda" if _cuda() else "cpu")
    embs = model.encode(
        texts, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def encode_e8(texts: list[str], model_path: str | None = None) -> np.ndarray:
    """Encode texts to 128-byte E8 keys, batched via _runtime._encode_texts()."""
    from latticememory import LatticeIndex
    BATCH = 64
    idx = LatticeIndex() if model_path is None else LatticeIndex(model=model_path)
    keys = np.zeros((len(texts), 128), dtype=np.uint8)
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        embs = idx._runtime._encode_texts(batch)   # (B, 1024) tensor, GPU-batched
        for j, emb in enumerate(embs):
            key_bytes = idx._runtime.memory.lattice_key_for(emb)
            keys[start + j] = np.frombuffer(key_bytes, dtype=np.uint8)
    return keys


def _cuda() -> bool:
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# kNN evaluation
# ---------------------------------------------------------------------------

def cosine_knn_accuracy(
    train_index: dict[str, list[str]],
    test_queries: list[tuple[str, str]],
    embs_func,
) -> tuple[float, float]:
    """Returns (accuracy, latency_ms_per_query)."""
    intents = list(train_index.keys())
    train_texts = [t for intent in intents for t in train_index[intent]]
    train_labels = [intent for intent in intents for _ in train_index[intent]]

    print(f"    Encoding {len(train_texts)} train + {len(test_queries)} test texts ...")
    t0 = time.perf_counter()
    train_embs = embs_func(train_texts)
    test_embs  = embs_func([q for q, _ in test_queries])
    encode_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = test_embs @ train_embs.T             # (N_test, N_train)
    preds  = np.argmax(scores, axis=1)
    lookup_sec = time.perf_counter() - t0

    correct = sum(
        train_labels[p] == true_label
        for p, (_, true_label) in zip(preds, test_queries)
    )
    acc = correct / len(test_queries)
    latency_ms = (lookup_sec / len(test_queries)) * 1000
    print(f"    Encode: {encode_sec:.1f}s  |  lookup: {lookup_sec*1000:.1f}ms total  "
          f"|  {latency_ms:.4f} ms/query")
    return acc, latency_ms


def hamming_knn_accuracy(
    train_index: dict[str, list[str]],
    test_queries: list[tuple[str, str]],
    keys_func,
) -> tuple[float, float, list[tuple[str, str, str]]]:
    """Returns (accuracy, latency_ms_per_query, top_confusions)."""
    intents = list(train_index.keys())
    train_texts = [t for intent in intents for t in train_index[intent]]
    train_labels = [intent for intent in intents for _ in train_index[intent]]

    print(f"    Computing E8 keys for {len(train_texts)} train + {len(test_queries)} test texts ...")
    t0 = time.perf_counter()
    train_keys = keys_func(train_texts)
    test_keys  = keys_func([q for q, _ in test_queries])
    encode_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Vectorised Hamming: count differing bytes per pair
    dists = np.sum(
        test_keys[:, None, :] != train_keys[None, :, :], axis=2
    )  # (N_test, N_train)
    preds = np.argmin(dists, axis=1)
    lookup_sec = time.perf_counter() - t0

    confusions: dict[tuple[str, str], int] = defaultdict(int)
    correct = 0
    for pred_idx, (_, true_label) in zip(preds, test_queries):
        pred_label = train_labels[pred_idx]
        if pred_label == true_label:
            correct += 1
        else:
            confusions[(true_label, pred_label)] += 1

    acc = correct / len(test_queries)
    latency_ms = (lookup_sec / len(test_queries)) * 1000
    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:10]
    top_conf_list = [(a, b, str(c)) for (a, b), c in top_conf]

    print(f"    Encode: {encode_sec:.1f}s  |  lookup: {lookup_sec*1000:.1f}ms total  "
          f"|  {latency_ms:.4f} ms/query")
    return acc, latency_ms, top_conf_list


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------

def build_training_examples(curriculum: dict, clusters: dict[str, list[str]]):
    """Convert curriculum → RoutingTrainingExample list (curriculum-ordered)."""
    from latticememory.training import RoutingTrainingExample
    intent_names = list(clusters.keys())
    other_reps = {
        name: [clusters[n][0] for n in intent_names if n != name][:3]
        for name in intent_names
    }
    cluster_canonical = {name: clusters[name][0] for name in intent_names}

    phase1, phase2, phase3 = [], [], []
    for pp in curriculum["positive_pairs"]:
        negs = other_reps.get(pp["cluster"], [])
        if not negs:
            continue
        ex = RoutingTrainingExample(query=pp["anchor"], positive=pp["positive"], negatives=negs)
        (phase1 if pp["difficulty"] == "easy" else phase2).append(ex)

    for hn in curriculum["hard_negative_pairs"]:
        anchor   = hn["anchor"]
        positive = cluster_canonical.get(hn["cluster_a"], anchor)
        if positive == anchor:
            others = [t for t in clusters.get(hn["cluster_a"], []) if t != anchor]
            positive = others[0] if others else anchor
        if anchor == positive:
            continue
        phase3.append(RoutingTrainingExample(
            query=anchor, positive=positive, negatives=[hn["negative"]]
        ))

    return phase1 + phase2 + phase3


def run_curriculum_training(
    clusters: dict[str, list[str]],
    intent_names: list[str],
    output_dir: Path,
) -> str:
    """Run Observatory curriculum → SnapTrainer. Returns path to trained encoder."""
    from latticememory import LatticeIndex
    from latticememory.snap_trainer import SnapTrainer, SnapTrainingConfig

    print("  [Train] Building base-encoder observatory ...")
    index = LatticeIndex()
    for texts in clusters.values():
        index.add(texts)
    obs = index.observatory()

    print("  [Train] Generating curriculum ...")
    curriculum = obs.generate_training_curriculum(clusters)
    print(f"    Positive pairs: {len(curriculum['positive_pairs'])}  "
          f"hard negatives: {len(curriculum['hard_negative_pairs'])}")
    for step in curriculum["curriculum_steps"]:
        print(f"    Phase {step['phase']} [{step['name']}]: {step['n_pairs']} pairs")

    examples = build_training_examples(curriculum, clusters)
    print(f"  [Train] {len(examples)} training examples (curriculum-ordered)")

    # Val clusters: random subset of VAL_INTENTS intents
    rng = random.Random(SEED)
    val_intent_names = rng.sample(intent_names, min(VAL_INTENTS, len(intent_names)))
    val_clusters = {k: clusters[k] for k in val_intent_names}

    output_dir.mkdir(parents=True, exist_ok=True)
    config = SnapTrainingConfig(
        epochs=EPOCHS,
        batch_size=4,
        gradient_accumulation_steps=4,
        freeze_layers=FREEZE_LAYERS,
        output_dir=str(output_dir),
        obs_eval_every_steps=OBS_EVAL_STEPS,
        zero_fp_recall_target=0.7,
        separation_target=0.8,
        seed=SEED,
    )

    print(f"  [Train] SnapTrainer: {EPOCHS} epochs, freeze_layers={FREEZE_LAYERS}, "
          f"obs_eval_every_steps={OBS_EVAL_STEPS}")
    print(f"  [Train] Val intents: {val_intent_names}")

    trainer = SnapTrainer(base_model=BASE_MODEL, val_clusters=val_clusters)
    t0 = time.perf_counter()
    result = trainer.train(examples, config=config)
    elapsed = time.perf_counter() - t0

    best_zfp = result.best_zero_fp_recall
    best_sep = result.best_separation_score
    print(f"  [Train] Done in {elapsed:.0f}s  |  "
          f"best zero_fp_recall={best_zfp:.4f}  separation={best_sep:.4f}  "
          f"reached_target={result.reached_target}")

    trained_path = str(output_dir / "best_snap_encoder")
    return trained_path, {
        "elapsed_sec": round(elapsed, 1),
        "best_zero_fp_recall": best_zfp,
        "best_separation_score": best_sep,
        "reached_target": result.reached_target,
        "n_examples": len(examples),
        "epochs": EPOCHS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("B10: BANKING77 2x2 ROUTING BENCHMARK")
    print("10-shot index (10 examples/intent)  |  3080 test queries  |  77 intents")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print("\n  Loading BANKING77 ...")
    train_index, test_queries, label_names = load_banking77(SHOTS, SEED)
    n_train = sum(len(v) for v in train_index.values())
    print(f"  Train index: {n_train} examples ({SHOTS}/intent, {len(train_index)} intents)")
    print(f"  Test:        {len(test_queries)} queries")

    results: dict = {
        "step": "B10_banking77_2x2",
        "shots": SHOTS, "seed": SEED,
        "n_train": n_train, "n_test": len(test_queries),
        "n_intents": len(label_names),
    }

    intermediate_path = RESULTS_DIR / "b10_banking77_intermediate.json"

    # -----------------------------------------------------------------------
    # CELL A: Continuous cosine kNN, base encoder
    # -----------------------------------------------------------------------
    saved_ab = json.loads(intermediate_path.read_text()) if intermediate_path.exists() else {}
    if "cell_A" in saved_ab:
        cell_a_acc = saved_ab["cell_A"]["accuracy"]
        cell_a_ms  = saved_ab["cell_A"]["ms_per_query"]
        results["cell_A"] = saved_ab["cell_A"]
        print(f"\n  --- CELL A: (cached) {cell_a_acc*100:.2f}%  {cell_a_ms:.4f} ms/query ---")
    else:
        print("\n  --- CELL A: Cosine kNN, base encoder ---")
        base_embs_fn = lambda texts: encode_continuous(texts, BASE_MODEL)
        cell_a_acc, cell_a_ms = cosine_knn_accuracy(train_index, test_queries, base_embs_fn)
        print(f"  Cell A accuracy: {cell_a_acc*100:.2f}%  ({cell_a_ms:.4f} ms/query)")
        results["cell_A"] = {"accuracy": round(cell_a_acc, 4), "ms_per_query": round(cell_a_ms, 6)}
        intermediate_path.write_text(json.dumps(results, indent=2))

    # -----------------------------------------------------------------------
    # CELL B: E8 Hamming kNN, base encoder (no training)
    # -----------------------------------------------------------------------
    if "cell_B" in saved_ab:
        cell_b_acc = saved_ab["cell_B"]["accuracy"]
        cell_b_ms  = saved_ab["cell_B"]["ms_per_query"]
        b_confusions = saved_ab["cell_B"].get("top_confusions", [])
        results["cell_B"] = saved_ab["cell_B"]
        print(f"\n  --- CELL B: (cached) {cell_b_acc*100:.2f}%  {cell_b_ms:.4f} ms/query ---")
    else:
        print("\n  --- CELL B: E8 Hamming kNN, base encoder ---")
        base_e8_fn = lambda texts: encode_e8(texts)
        cell_b_acc, cell_b_ms, b_confusions = hamming_knn_accuracy(train_index, test_queries, base_e8_fn)
        results["cell_B"] = {
            "accuracy": round(cell_b_acc, 4),
            "ms_per_query": round(cell_b_ms, 6),
            "top_confusions": b_confusions,
        }
        intermediate_path.write_text(json.dumps(results, indent=2))
        print(f"  Cell B accuracy: {cell_b_acc*100:.2f}%  ({cell_b_ms:.4f} ms/query)")
        print(f"  Top confusions (true->pred):")
        for true_i, pred_i, cnt in b_confusions[:5]:
            print(f"    {true_i:40s} -> {pred_i:40s}  ({cnt}x)")

    # -----------------------------------------------------------------------
    # Intermediate report: A vs B -- calibrate before training
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  INTERMEDIATE: Base encoder (no training)")
    print("=" * 70)
    quant_cost = cell_a_acc - cell_b_acc
    speedup = cell_a_ms / cell_b_ms if cell_b_ms > 0 else float("inf")
    print(f"  Cell A (cosine, continuous) : {cell_a_acc*100:.2f}%  -- bar to beat")
    print(f"  Cell B (Hamming, E8)        : {cell_b_acc*100:.2f}%  -- quant cost: {quant_cost*100:.2f}pp")
    print(f"  E8 Hamming speedup          : {speedup:.1f}x  ({cell_a_ms:.4f} vs {cell_b_ms:.4f} ms/query)")
    print()
    print("  Published 10-shot BANKING77 SOTA (SetFit): ~85-88%")
    to_beat = max(cell_a_acc, 0.85)
    gap_to_sota = to_beat - cell_b_acc
    print(f"  Gap for Cell D to close:    {gap_to_sota*100:.2f} pp (from {cell_b_acc*100:.2f}% to >{to_beat*100:.1f}%)")

    # -----------------------------------------------------------------------
    # TRAINING: curriculum-ordered SnapTrainer on 10-shot BANKING77
    # -----------------------------------------------------------------------
    print()
    print("  --- TRAINING: curriculum SnapTrainer on BANKING77 ---")
    train_output_dir = RESULTS_DIR / "b10_banking77_trained"
    trained_path, train_meta = run_curriculum_training(
        train_index,
        list(label_names),
        train_output_dir,
    )
    results["training"] = train_meta

    # -----------------------------------------------------------------------
    # CELL C: Continuous cosine kNN, trained encoder
    # -----------------------------------------------------------------------
    print()
    print("  --- CELL C: Cosine kNN, trained encoder ---")
    trained_embs_fn = lambda texts: encode_continuous(texts, trained_path)
    cell_c_acc, cell_c_ms = cosine_knn_accuracy(train_index, test_queries, trained_embs_fn)
    print(f"  Cell C accuracy: {cell_c_acc*100:.2f}%  ({cell_c_ms:.4f} ms/query)")
    results["cell_C"] = {"accuracy": round(cell_c_acc, 4), "ms_per_query": round(cell_c_ms, 6)}

    # -----------------------------------------------------------------------
    # CELL D: E8 Hamming kNN, trained encoder
    # -----------------------------------------------------------------------
    print()
    print("  --- CELL D: E8 Hamming kNN, trained encoder ---")
    trained_e8_fn = lambda texts: encode_e8(texts, trained_path)
    cell_d_acc, cell_d_ms, d_confusions = hamming_knn_accuracy(train_index, test_queries, trained_e8_fn)
    print(f"  Cell D accuracy: {cell_d_acc*100:.2f}%  ({cell_d_ms:.4f} ms/query)")
    print(f"  Top confusions (true->pred):")
    for true_i, pred_i, cnt in d_confusions[:8]:
        print(f"    {true_i:40s} -> {pred_i:40s}  ({cnt}x)")
    results["cell_D"] = {
        "accuracy": round(cell_d_acc, 4),
        "ms_per_query": round(cell_d_ms, 6),
        "top_confusions": d_confusions,
    }

    # -----------------------------------------------------------------------
    # Final 2×2 table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  BANKING77 2x2 RESULTS  (10-shot index, 3080 test queries)")
    print("=" * 70)
    print()
    print(f"  {'':25s}  {'Cosine kNN':>12}  {'E8 Hamming kNN':>16}  {'Δ pp':>8}")
    print(f"  {'Base encoder':25s}  {cell_a_acc*100:>11.2f}%  {cell_b_acc*100:>15.2f}%  "
          f"{(cell_b_acc-cell_a_acc)*100:>+7.2f}")
    print(f"  {'SnapTrained (3ep)':25s}  {cell_c_acc*100:>11.2f}%  {cell_d_acc*100:>15.2f}%  "
          f"{(cell_d_acc-cell_c_acc)*100:>+7.2f}")
    print(f"  {'Δ pp (train benefit)':25s}  {(cell_c_acc-cell_a_acc)*100:>+11.2f}   {(cell_d_acc-cell_b_acc)*100:>+15.2f}")
    print()

    speedup_ad = cell_a_ms / cell_d_ms if cell_d_ms > 0 else 0
    print(f"  Latency: cosine={cell_a_ms:.4f}ms  E8={cell_d_ms:.4f}ms  speedup={speedup_ad:.0f}x")
    print()

    # Verdict
    sota = 0.87  # SetFit 10-shot SOTA midpoint
    if cell_d_acc >= sota:
        print(f"  VERDICT: Cell D ({cell_d_acc*100:.2f}%) MATCHES or BEATS 10-shot SetFit SOTA ({sota*100:.0f}%).")
        print(f"  This is a record claim: equivalent accuracy at {speedup_ad:.0f}x lookup speed.")
    elif cell_d_acc >= cell_a_acc * 0.95:
        print(f"  VERDICT: Cell D ({cell_d_acc*100:.2f}%) retains "
              f"{cell_d_acc/cell_a_acc*100:.1f}% of Cell A accuracy at {speedup_ad:.0f}x speed.")
        print(f"  Pareto win on latency; {(sota-cell_d_acc)*100:.1f}pp gap to close for SOTA claim.")
    else:
        print(f"  VERDICT: Cell D ({cell_d_acc*100:.2f}%) needs more training or epochs.")
        print(f"  Training improved E8 routing by {(cell_d_acc-cell_b_acc)*100:.2f}pp. Run more epochs.")

    # Save final
    (RESULTS_DIR / "b10_banking77_2x2.json").write_text(json.dumps(results, indent=2))
    print()
    print("B10 COMPLETE")


if __name__ == "__main__":
    main()
