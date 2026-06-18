"""B6: Flywheel Fine-tune Smoke -- validates the flywheel->SnapTrainer loop.

NOTE: This is a SMOKE TEST of the training pipeline, not the quality benchmark.
The published quality result (zero_fp_recall=0.8056+) lives in the existing
checkpoint at benchmarks/results/snap_product_gate_hard_near_miss_10ep/.

Seeds flywheel with trained-encoder keys (cluster_threshold=60 to match the
HammingRouter's 70-block accept window), then runs 1 epoch of finetune() from
BASE encoder. Confirms the loop: cluster -> generate_pairs -> SnapTrainer.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"
BASE_MODEL = "dfrokido/bge-large-e8-snap"


def main() -> None:
    print("=" * 60)
    print("B6: FLYWHEEL FINE-TUNE SMOKE")
    print("(1-epoch validation -- not the quality checkpoint)")
    print("=" * 60)

    from latticememory import LatticeIndex
    from latticememory.flywheel import LatticeFlywheel
    from latticememory.snap_trainer import SnapTrainingConfig

    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    intents = src.get("intents", [])

    # Use trained encoder for key computation.
    # Base encoder puts same-intent paraphrases ~99 blocks apart (won't cluster).
    # Trained encoder brings them within 60 blocks, enabling clustering.
    TRAINED = RESULTS_DIR / "snap_product_gate_hard_near_miss_10ep" / "best_snap_encoder"
    if TRAINED.exists():
        print("  Loading trained encoder (10ep) for key computation ...")
        key_index = LatticeIndex(model=str(TRAINED))
    else:
        print("  WARNING: trained checkpoint not found, using base encoder")
        key_index = LatticeIndex()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tf:
        log_path = tf.name

    # cluster_threshold=60 matches the HammingRouter's 70-block accept window
    flywheel = LatticeFlywheel(log_path, cluster_threshold=60)
    total_seeded = 0
    for intent in intents:
        texts = [intent["canonical"]] + intent.get("paraphrases", [])
        for text in texts:
            key_hex = key_index.snap(text)
            flywheel.log_miss(text, e8_key_hex=key_hex,
                              metadata={"intent_id": intent["intent_id"]})
            total_seeded += 1

    print(f"  Seeded {total_seeded} miss records")

    clusters = flywheel.cluster_gaps(min_cluster_size=3)
    print(f"  Formed {len(clusters)} clusters")

    if len(clusters) < 2:
        print("  SKIP: need >=2 clusters for cross-cluster hard negatives")
        Path(log_path).unlink(missing_ok=True)
        return

    pairs = flywheel.generate_training_pairs(min_examples_per_cluster=3)
    print(f"  Generated {len(pairs)} training pairs")

    output_dir = RESULTS_DIR / "b6_flywheel_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SnapTrainingConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=4,
        freeze_layers=20,
        output_dir=str(output_dir),
        zero_fp_recall_target=0.5,  # lower bar -- smoke run
        separation_target=0.5,
    )

    print("  Running finetune: 1 epoch, BASE encoder, freeze_layers=20, no PAWS ...")
    result = flywheel.finetune(
        base_model=BASE_MODEL,
        output_dir=output_dir,
        config=config,
        min_examples_per_cluster=3,
        add_paws=False,
    )

    Path(log_path).unlink(missing_ok=True)

    # Extract best zero_fp_recall from obs checkpoints
    best_zero_fp = max(
        (c.zero_fp_recall for c in result.obs_checkpoints), default=0.0
    )

    out = {
        "step": "B6_flywheel_smoke",
        "n_clusters": len(clusters),
        "n_training_pairs": len(pairs),
        "epochs_run": 1,
        "best_zero_fp_recall": best_zero_fp,
        "best_separation_score": result.best_separation_score,
        "best_fragmentation_score": result.best_fragmentation_score,
        "reached_product_gate": result.reached_target,
        "smoke_note": "1-epoch smoke; full quality checkpoint at snap_product_gate_hard_near_miss_10ep/",
    }
    (RESULTS_DIR / "b6_flywheel_smoke.json").write_text(json.dumps(out, indent=2))

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Clusters used          : {len(clusters)}")
    print(f"  Training pairs         : {len(pairs)}")
    print(f"  Best zero_fp_recall    : {best_zero_fp:.4f}")
    print(f"  Best separation_score  : {result.best_separation_score:.4f}")
    print(f"  Best fragmentation     : {result.best_fragmentation_score:.4f}")
    print(f"  Reached product gate   : {result.reached_target}")
    print()
    print("  NOTE: 1-epoch smoke on demo data -- for published metrics see")
    print("  benchmarks/results/snap_product_gate_hard_near_miss_10ep/")
    print()
    print("B6 COMPLETE")


if __name__ == "__main__":
    main()
