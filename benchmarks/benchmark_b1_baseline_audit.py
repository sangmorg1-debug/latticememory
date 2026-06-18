"""B1: Baseline Audit -- export_for_llm() + routing_profile() on base encoder.

Captures the 'before' snapshot needed by B7 semantic diff.
Saved to: benchmarks/results/b1_baseline_snapshot.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"
SNAPSHOT_PATH = RESULTS_DIR / "b1_baseline_snapshot.json"


def load_demo_texts() -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (all_texts, query_doc_pairs) from demo data."""
    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    paraphrases = json.loads((DEMO_DIR / "heldout_paraphrases.json").read_text())
    near_misses = json.loads((DEMO_DIR / "heldout_near_misses.json").read_text())

    all_texts: list[str] = []
    pairs: list[tuple[str, str]] = []

    # Index all canonicals + paraphrases
    for intent in src.get("intents", []):
        canonical = intent["canonical"]
        all_texts.append(canonical)
        for p in intent.get("paraphrases", []):
            all_texts.append(p)
            pairs.append((p, canonical))

    # Heldout paraphrases -> (query, doc) pairs
    for item in paraphrases if isinstance(paraphrases, list) else paraphrases.get("items", []):
        q = item.get("query") or item.get("question") or item.get("paraphrase", "")
        doc = item.get("canonical") or item.get("answer", "")
        if q and doc:
            pairs.append((q, doc))
        if q:
            all_texts.append(q)

    # Near-misses: we index them so they exist in the corpus
    for item in near_misses if isinstance(near_misses, list) else near_misses.get("items", []):
        q = item.get("query") or item.get("question", "")
        if q:
            all_texts.append(q)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_texts: list[str] = []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    return unique_texts, pairs


def main() -> None:
    print("=" * 60)
    print("B1: BASELINE AUDIT")
    print("Encoder: dfrokido/bge-large-e8-snap (base)")
    print("=" * 60)

    from latticememory import LatticeIndex
    from latticememory.observatory import LatticeObservatory

    texts, pairs = load_demo_texts()
    print(f"  Loaded {len(texts)} unique texts, {len(pairs)} query-doc pairs")

    print("  Loading base encoder ...")
    index = LatticeIndex()  # dfrokido/bge-large-e8-snap
    print("  Indexing texts ...")
    index.add(texts)
    print(f"  Indexed {index._runtime.memory.num_documents} documents")

    obs = index.observatory()

    print("  Running export_for_llm() ...")
    snapshot = obs.export_for_llm(n_sample_cells=10)

    if "status" in snapshot and snapshot.get("docs", 1) == 0:
        print("  ERROR: empty index -- export failed")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"  Snapshot saved -> {SNAPSHOT_PATH}")

    # Print results
    summary = snapshot.get("index_summary", {})
    block_analysis = snapshot.get("block_analysis", {})
    collision_cells = sum(
        1 for c in snapshot.get("sample_cells", [])
        if c.get("coherence_label") == "collision"
    )

    routing_result: dict = {}
    if pairs:
        queries = [p[0] for p in pairs[:50]]
        docs = [p[1] for p in pairs[:50]]
        print("  Running routing_profile() ...")
        routing_result = obs.routing_profile(queries, docs)

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Total docs indexed : {summary.get('total_docs')}")
    print(f"  Unique E8 keys     : {summary.get('unique_keys')}")
    print(f"  Multi-doc cells    : {summary.get('multi_doc_keys')}")
    print(f"  Mean entropy       : {block_analysis.get('mean_entropy'):.4f}")
    print(f"  Most stable blocks : {block_analysis.get('most_stable_blocks', [])[:5]}")
    print(f"  Most variable blocks: {block_analysis.get('most_variable_blocks', [])[:5]}")
    print(f"  Collision cells (sample): {collision_cells}")

    if routing_result:
        routing = routing_result.get("routing", {})
        print(f"  Routing exact_rate : {routing.get('exact_rate'):.4f}")
        print(f"  Routing hamming1   : {routing.get('hamming1_rate'):.4f}")
        print(f"  Routing fallback   : {routing.get('fallback_rate'):.4f}")
        print(f"  Hamming mean       : {routing_result.get('hamming', {}).get('mean')}")
        print(f"  Top mismatching blocks: {[b['block'] for b in routing_result.get('top_mismatching_blocks', [])[:5]]}")

    recs = snapshot.get("recommendations", [])
    if recs:
        print(f"  Recommendations ({len(recs)}):")
        for r in recs[:3]:
            print(f"    - {r}")

    print()
    print(f"  Snapshot written to {SNAPSHOT_PATH}")
    print("B1 COMPLETE")


if __name__ == "__main__":
    main()
