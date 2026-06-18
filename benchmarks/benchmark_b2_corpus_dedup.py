"""B2: Corpus Dedup -- LatticeDedup on training corpus.

Measures how much semantic redundancy exists in the raw paraphrase+near-miss corpus.
High compression -> fewer unique concepts than texts (good: training data was clean).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    print("=" * 60)
    print("B2: CORPUS DEDUP")
    print("=" * 60)

    from latticememory import LatticeDedup

    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    paraphrases_raw = json.loads((DEMO_DIR / "heldout_paraphrases.json").read_text())
    near_misses_raw = json.loads((DEMO_DIR / "heldout_near_misses.json").read_text())

    corpus: list[str] = []

    # All canonicals + paraphrases from source
    for intent in src.get("intents", []):
        corpus.append(intent["canonical"])
        corpus.extend(intent.get("paraphrases", []))

    # Heldout paraphrases
    items = paraphrases_raw if isinstance(paraphrases_raw, list) else paraphrases_raw.get("items", [])
    for item in items:
        q = item.get("query") or item.get("question") or item.get("paraphrase", "")
        if q:
            corpus.append(q)

    # Near-misses
    items = near_misses_raw if isinstance(near_misses_raw, list) else near_misses_raw.get("items", [])
    for item in items:
        q = item.get("query") or item.get("question", "")
        if q:
            corpus.append(q)

    print(f"  Corpus size: {len(corpus)} texts")
    print("  Loading encoder and running dedup ...")

    dedup = LatticeDedup()
    result = dedup.deduplicate(corpus)

    unique = result["unique_documents"]
    dupes = result["duplicates"]
    ratio = result["compression_ratio"]

    n_dupe_clusters = len(dupes)
    total_dupe_texts = sum(len(v) for v in dupes.values())

    # Per-cluster stats
    cluster_sizes = sorted([len(v) + 1 for v in dupes.values()], reverse=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step": "B2_corpus_dedup",
        "n_input": len(corpus),
        "n_unique": len(unique),
        "n_dupe_clusters": n_dupe_clusters,
        "total_duplicates_removed": total_dupe_texts,
        "compression_ratio": ratio,
        "top_cluster_sizes": cluster_sizes[:10],
    }
    (RESULTS_DIR / "b2_corpus_dedup.json").write_text(json.dumps(out, indent=2))

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Input texts         : {len(corpus)}")
    print(f"  Unique E8 keys      : {len(unique)}")
    print(f"  Duplicate clusters  : {n_dupe_clusters}")
    print(f"  Texts removed       : {total_dupe_texts}")
    print(f"  Compression ratio   : {ratio:.4f}  ({ratio*100:.1f}% removed)")
    print(f"  Largest clusters    : {cluster_sizes[:5]}")

    if ratio > 0.3:
        print("  -> HIGH redundancy: significant semantic overlap in corpus")
    elif ratio > 0.1:
        print("  -> MODERATE redundancy: some paraphrases share E8 keys")
    else:
        print("  -> LOW redundancy: corpus is semantically diverse")

    print()
    print("B2 COMPLETE")


if __name__ == "__main__":
    main()
