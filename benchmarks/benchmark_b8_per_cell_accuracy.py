"""B8: Per-Cell Accuracy -- semantic_probe() before and after training.

Maps each E8 block to the semantic dimensions it encodes via mutual information.
Before/after comparison shows which blocks the training sharpened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"
TRAINED_CHECKPOINT = RESULTS_DIR / "snap_product_gate_hard_near_miss_10ep" / "best_snap_encoder"


def build_labeled_texts(src: dict) -> dict[str, list[str]]:
    labeled: dict[str, list[str]] = {}
    for intent in src.get("intents", []):
        iid = intent["intent_id"]
        texts = [intent["canonical"]] + intent.get("paraphrases", [])
        labeled[iid] = texts
    return labeled


def _mean_sep(p: dict) -> float | None:
    blks = p.get("all_blocks", [])
    return round(sum(b.get("separability", 0) for b in blks) / len(blks), 4) if blks else None


def _print_probe(probe: dict) -> None:
    if "error" in probe:
        print(f"    ERROR: {probe['error']}")
        return
    top_blocks = probe.get("top_separating_blocks", [])[:8]
    freeze_cands = probe.get("freeze_candidates", [])[:5]
    finetune_targets = probe.get("fine_tune_targets", [])[:5]

    all_blocks = probe.get("all_blocks", [])
    mean_sep = round(sum(b.get("separability", 0) for b in all_blocks) / len(all_blocks), 4) if all_blocks else None

    print(f"    Mean separability      : {mean_sep}")
    top_block_nums = [b['block'] for b in top_blocks]
    top_block_seps = [round(b['separability'], 3) for b in top_blocks]
    print(f"    Top separating blocks  : {top_block_nums}")
    print(f"    Top block sep scores   : {top_block_seps}")
    print(f"    Fine-tune targets dims : {finetune_targets}")
    print(f"    Freeze candidates dims : {freeze_cands}")
    if probe.get("interpretation"):
        print(f"    {probe['interpretation'][:120]}")


def main() -> None:
    print("=" * 60)
    print("B8: PER-CELL ACCURACY (semantic_probe)")
    print("Before: base encoder  |  After: 10ep trained checkpoint")
    print("=" * 60)

    from latticememory import LatticeIndex

    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    labeled_texts = build_labeled_texts(src)

    print(f"  Labels: {list(labeled_texts.keys())}")
    total_texts = sum(len(v) for v in labeled_texts.values())
    print(f"  Total labeled texts: {total_texts}")

    # ---------- BASE ENCODER ----------
    print()
    print("  [BASE ENCODER] Loading ...")
    base_index = LatticeIndex()
    for texts in labeled_texts.values():
        base_index.add(texts)

    base_obs = base_index.observatory()
    print("  Running semantic_probe (base) ...")
    base_probe = base_obs.semantic_probe(labeled_texts)

    # ---------- TRAINED ENCODER ----------
    if not TRAINED_CHECKPOINT.exists():
        print(f"  WARNING: trained checkpoint not found at {TRAINED_CHECKPOINT}")
        print("  Skipping after-training comparison.")
        after_probe = None
    else:
        print()
        print("  [TRAINED ENCODER] Loading 10ep checkpoint ...")
        after_index = LatticeIndex(model=str(TRAINED_CHECKPOINT))
        for texts in labeled_texts.values():
            after_index.add(texts)

        after_obs = after_index.observatory()
        print("  Running semantic_probe (trained) ...")
        after_probe = after_obs.semantic_probe(labeled_texts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step": "B8_per_cell_accuracy",
        "base_probe_summary": {
            "mean_sep": _mean_sep(base_probe) if base_probe else None,
            "top_blocks": [b["block"] for b in base_probe.get("top_separating_blocks", [])[:10]] if base_probe else [],
            "fine_tune_targets": base_probe.get("fine_tune_targets", []) if base_probe else [],
        },
        "after_probe_summary": {
            "mean_sep": _mean_sep(after_probe) if after_probe else None,
            "top_blocks": [b["block"] for b in after_probe.get("top_separating_blocks", [])[:10]] if after_probe else [],
            "fine_tune_targets": after_probe.get("fine_tune_targets", []) if after_probe else [],
        } if after_probe else None,
    }
    (RESULTS_DIR / "b8_per_cell_accuracy.json").write_text(json.dumps(out, indent=2))

    print()
    print("-- RESULTS ---------------------------------------------")
    print("  BASE ENCODER:")
    _print_probe(base_probe)

    if after_probe:
        print()
        print("  TRAINED ENCODER (10ep):")
        _print_probe(after_probe)

        # Compare top separating blocks (extract block numbers from dicts)
        base_top_nums = set(b["block"] for b in base_probe.get("top_separating_blocks", [])[:10])
        after_top_nums = set(b["block"] for b in after_probe.get("top_separating_blocks", [])[:10])
        newly_active = after_top_nums - base_top_nums
        lost_active = base_top_nums - after_top_nums
        stable = base_top_nums & after_top_nums

        print()
        print("  DELTA:")
        print(f"    Stable top blocks      : {sorted(stable)}")
        print(f"    Newly activated blocks : {sorted(newly_active)}")
        print(f"    Lost from top          : {sorted(lost_active)}")

        base_sep = _mean_sep(base_probe)
        after_sep = _mean_sep(after_probe)
        if base_sep is not None and after_sep is not None:
            delta = round(after_sep - base_sep, 4)
            print(f"    Mean separability delta: {delta:+.4f}  (positive = better)")

    print()
    print("B8 COMPLETE")


if __name__ == "__main__":
    main()
