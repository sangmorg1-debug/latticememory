"""B7: Semantic Diff -- compare_snapshots(base encoder, 10ep trained encoder).

Loads the B1 baseline snapshot and builds a new snapshot with the validated
10-epoch checkpoint. compare_snapshots() shows the training impact block-by-block.

Requires B1 to have run first (loads benchmarks/results/b1_baseline_snapshot.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_SNAPSHOT = RESULTS_DIR / "b1_baseline_snapshot.json"
TRAINED_CHECKPOINT = RESULTS_DIR / "snap_product_gate_hard_near_miss_10ep" / "best_snap_encoder"


def load_demo_texts() -> list[str]:
    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    paraphrases_raw = json.loads((DEMO_DIR / "heldout_paraphrases.json").read_text())

    texts: list[str] = []
    seen: set[str] = set()

    for intent in src.get("intents", []):
        for t in [intent["canonical"]] + intent.get("paraphrases", []):
            if t not in seen:
                seen.add(t)
                texts.append(t)

    items = paraphrases_raw if isinstance(paraphrases_raw, list) else paraphrases_raw.get("items", [])
    for item in items:
        q = item.get("query") or item.get("question") or item.get("paraphrase", "")
        if q and q not in seen:
            seen.add(q)
            texts.append(q)

    return texts


def main() -> None:
    print("=" * 60)
    print("B7: SEMANTIC DIFF")
    print("Before: base encoder  |  After: 10ep trained checkpoint")
    print("=" * 60)

    from latticememory import LatticeIndex

    # Load or regenerate baseline snapshot
    if BASELINE_SNAPSHOT.exists():
        print(f"  Loading baseline snapshot from {BASELINE_SNAPSHOT.name} ...")
        before_snapshot = json.loads(BASELINE_SNAPSHOT.read_text())
    else:
        print("  B1 snapshot not found -- regenerating baseline ...")
        from latticememory.observatory import LatticeObservatory
        texts = load_demo_texts()
        idx = LatticeIndex()
        idx.add(texts)
        before_snapshot = idx.observatory().export_for_llm(n_sample_cells=10)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        BASELINE_SNAPSHOT.write_text(json.dumps(before_snapshot, indent=2))
        print(f"  Saved to {BASELINE_SNAPSHOT}")

    if "status" in before_snapshot:
        print("  ERROR: baseline snapshot is empty")
        return

    # Validate trained checkpoint exists
    if not TRAINED_CHECKPOINT.exists():
        print(f"  ERROR: trained checkpoint not found at {TRAINED_CHECKPOINT}")
        print("  Expected: benchmarks/results/snap_product_gate_hard_near_miss_10ep/best_snap_encoder/")
        return

    print(f"  Loading 10ep trained encoder from {TRAINED_CHECKPOINT} ...")
    texts = load_demo_texts()

    after_index = LatticeIndex(model=str(TRAINED_CHECKPOINT))
    print(f"  Indexing {len(texts)} texts with trained encoder ...")
    after_index.add(texts)

    print("  Running export_for_llm() on trained encoder ...")
    after_obs = after_index.observatory()
    after_snapshot = after_obs.export_for_llm(n_sample_cells=10)

    print("  Running compare_snapshots() ...")
    diff = after_obs.compare_snapshots(before_snapshot, after_snapshot)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "b7_semantic_diff.json").write_text(json.dumps(diff, indent=2))

    # Extract key stats
    verdict = diff.get("verdict", "unknown")
    entropy = diff.get("entropy", {})
    collisions = diff.get("collisions", {})
    most_improved = diff.get("most_improved_blocks", [])[:5]
    most_degraded = diff.get("most_degraded_blocks", [])[:5]

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Verdict                  : {verdict.upper()}")
    print(f"  Mean entropy before      : {entropy.get('before_mean')}")
    print(f"  Mean entropy after       : {entropy.get('after_mean')}")
    print(f"  Entropy delta            : {entropy.get('delta')}  (negative = better)")
    print(f"  Blocks improved          : {entropy.get('improved_blocks')}")
    print(f"  Blocks degraded          : {entropy.get('degraded_blocks')}")
    if collisions:
        print(f"  Collision cells before   : {collisions.get('before_sampled')}")
        print(f"  Collision cells after    : {collisions.get('after_sampled')}")
        print(f"  Collision delta          : {collisions.get('delta')}")

    if most_improved:
        print(f"  Most improved blocks (top 5):")
        for b in most_improved:
            print(f"    block {b['block']:3d} dims {b['dim_range']:10s}  D={b['delta']:+.4f}")
    if most_degraded and most_degraded[0].get("delta", 0) > 0:
        print(f"  Most degraded blocks (top 5):")
        for b in most_degraded:
            if b.get("delta", 0) > 0:
                print(f"    block {b['block']:3d} dims {b['dim_range']:10s}  D={b['delta']:+.4f}")

    summary = diff.get("summary", "")
    if summary:
        print(f"  Summary: {summary}")

    print()
    print("B7 COMPLETE")


if __name__ == "__main__":
    main()
