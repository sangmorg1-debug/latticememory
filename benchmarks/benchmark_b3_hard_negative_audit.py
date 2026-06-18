"""B3: Hard Negative Audit -- generate_training_pairs() Hamming distribution.

Seeds the flywheel with demo records keyed by the TRAINED encoder, clusters
them, then inspects the Hamming distribution of the generated hard negatives.
We use the trained encoder because on the base encoder same-intent paraphrases
are ~99 blocks apart (won't cluster); training is what makes them cluster.
Tight negatives (Hamming 10-50) are the best training signal.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"


def hamming_bytes(a: bytes, b: bytes) -> int:
    # Block-level Hamming: count differing bytes (0-128 range, matching router semantics)
    return sum(1 for x, y in zip(a, b) if x != y)


def main() -> None:
    print("=" * 60)
    print("B3: HARD NEGATIVE AUDIT")
    print("=" * 60)

    from latticememory import LatticeIndex
    from latticememory.flywheel import LatticeFlywheel

    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    near_misses_raw = json.loads((DEMO_DIR / "heldout_near_misses.json").read_text())

    # Collect texts to seed flywheel: paraphrases by intent
    intents = src.get("intents", [])
    intent_texts: dict[str, list[str]] = {}
    for intent in intents:
        iid = intent["intent_id"]
        intent_texts[iid] = [intent["canonical"]] + intent.get("paraphrases", [])

    # Load near misses
    nm_items = near_misses_raw if isinstance(near_misses_raw, list) else near_misses_raw.get("items", [])

    TRAINED_CHECKPOINT = RESULTS_DIR / "snap_product_gate_hard_near_miss_10ep" / "best_snap_encoder"
    if TRAINED_CHECKPOINT.exists():
        print(f"  Loading trained encoder (10ep) for E8 key computation ...")
        index = LatticeIndex(model=str(TRAINED_CHECKPOINT))
    else:
        print("  WARNING: trained checkpoint not found, using base encoder")
        print("  (clusters may not form — base encoder maps paraphrases ~99 blocks apart)")
        index = LatticeIndex()

    print("  Computing E8 keys for all texts ...")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tf:
        log_path = tf.name

    # Use cluster_threshold=60 to match the HammingRouter's 70-block accept window.
    # Default threshold=25 is too tight for same-intent paraphrases (Hamming 20-60 apart).
    flywheel = LatticeFlywheel(log_path, cluster_threshold=60)

    total_logged = 0
    for intent_id, texts in intent_texts.items():
        for text in texts:
            key_hex = index.snap(text)
            flywheel.log_miss(text, e8_key_hex=key_hex, metadata={"intent_id": intent_id})
            total_logged += 1

    # Also log near-miss queries (these represent cross-intent hard cases)
    for item in nm_items[:30]:
        q = item.get("query") or item.get("question", "")
        if q:
            key_hex = index.snap(q)
            flywheel.log_miss(q, e8_key_hex=key_hex, metadata={"source": "near_miss"})
            total_logged += 1

    print(f"  Seeded {total_logged} records into flywheel")

    clusters = flywheel.cluster_gaps(min_cluster_size=3)
    print(f"  Formed {len(clusters)} clusters (min_size=3)")
    for c in clusters[:5]:
        print(f"    cluster_{c.cluster_id}: size={c.size}, rep={c.representative[:50]!r}")

    print("  Generating training pairs ...")
    pairs = flywheel.generate_training_pairs(min_examples_per_cluster=3)
    print(f"  Generated {len(pairs)} training pairs")

    if not pairs:
        print("  No pairs -- likely all misses landed in same cluster or too few clusters")
        return

    # Analyze Hamming distances between anchor and each negative
    hamming_dists: list[int] = []
    for ex in pairs:
        anchor_key = bytes.fromhex(index.snap(ex.query))
        for neg in ex.negatives:
            neg_key = bytes.fromhex(index.snap(neg))
            d = hamming_bytes(anchor_key, neg_key)
            hamming_dists.append(d)

    # Bucket distribution
    buckets = {"0-10": 0, "11-30": 0, "31-50": 0, "51-80": 0, "81-128": 0}
    for d in hamming_dists:
        if d <= 10:
            buckets["0-10"] += 1
        elif d <= 30:
            buckets["11-30"] += 1
        elif d <= 50:
            buckets["31-50"] += 1
        elif d <= 80:
            buckets["51-80"] += 1
        else:
            buckets["81-128"] += 1

    import statistics
    mean_h = statistics.mean(hamming_dists) if hamming_dists else 0
    median_h = statistics.median(hamming_dists) if hamming_dists else 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step": "B3_hard_negative_audit",
        "n_records_logged": total_logged,
        "n_clusters": len(clusters),
        "n_pairs": len(pairs),
        "n_negatives_analyzed": len(hamming_dists),
        "hamming_mean": round(mean_h, 2),
        "hamming_median": float(median_h),
        "hamming_buckets": buckets,
    }
    (RESULTS_DIR / "b3_hard_negative_audit.json").write_text(json.dumps(out, indent=2))

    # Clean up temp file
    Path(log_path).unlink(missing_ok=True)

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Training pairs generated : {len(pairs)}")
    print(f"  Negatives analyzed       : {len(hamming_dists)}")
    print(f"  Hamming mean             : {mean_h:.2f}")
    print(f"  Hamming median           : {float(median_h):.1f}")
    print(f"  Distribution:")
    for bucket, count in buckets.items():
        pct = count / len(hamming_dists) * 100 if hamming_dists else 0
        bar = "#" * int(pct / 2)
        print(f"    {bucket:>8}  {count:4d}  {pct:5.1f}%  {bar}")

    tight = buckets["11-30"] + buckets["31-50"]
    total = len(hamming_dists)
    if total:
        print(f"  'Hard' negatives (Hamming 11-50): {tight}/{total} = {tight/total*100:.1f}%")
        if tight / total > 0.4:
            print("  -> STRONG hard negative signal -- training will see genuinely confusable pairs")
        else:
            print("  -> Negatives are mostly far apart -- consider tighter cluster threshold")

    print()
    print("B3 COMPLETE")


if __name__ == "__main__":
    main()
