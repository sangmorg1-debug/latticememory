"""B5: Context Compression -- LatticeStreamDedup on prompts_responses.json.

Simulates a real-time stream of prompts and measures how many are semantic
duplicates within a sliding window. Compression = fewer LLM calls needed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_DIR = Path(__file__).parent / "demo_data" / "hard_near_miss_challenge"
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    print("=" * 60)
    print("B5: CONTEXT COMPRESSION")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    from latticememory.stream import LatticeStreamDedup
    from latticememory.text_runtime import RFSnapTextMemory

    pr_raw = json.loads((DEMO_DIR / "prompts_responses.json").read_text())

    prompts: list[str] = []
    if isinstance(pr_raw, list):
        for item in pr_raw:
            p = item.get("prompt") or item.get("question") or item.get("query", "")
            if p:
                prompts.append(p)
    elif isinstance(pr_raw, dict):
        for item in pr_raw.get("items", pr_raw.get("prompts", [])):
            p = item.get("prompt") or item.get("question") or item.get("query", "")
            if p:
                prompts.append(p)

    if not prompts:
        print("  ERROR: no prompts found in prompts_responses.json")
        return

    # Also append paraphrases to get a realistic stream with repetition
    src = json.loads((DEMO_DIR / "hard_near_miss_source.json").read_text())
    for intent in src.get("intents", []):
        prompts.append(intent["canonical"])
        prompts.extend(intent.get("paraphrases", []))

    print(f"  Stream size: {len(prompts)} prompts (including paraphrases)")
    print("  Loading encoder ...")

    encoder = SentenceTransformer("dfrokido/bge-large-e8-snap")
    d_model = int(encoder.get_embedding_dimension() or 1024)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model)

    stream_dedup = LatticeStreamDedup(
        runtime=runtime,
        time_window_seconds=3600.0,
        allow_neighborhood=True,
    )

    print("  Processing stream ...")
    t0 = time.perf_counter()
    n_dup = 0
    n_unique = 0
    exact_matches = 0
    neighborhood_matches = 0

    for prompt in prompts:
        result = stream_dedup.process(prompt)
        if result["is_duplicate"]:
            n_dup += 1
            if result["match_path"] == "exact":
                exact_matches += 1
            else:
                neighborhood_matches += 1
        else:
            n_unique += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total = len(prompts)
    dedup_rate = n_dup / total if total else 0

    # Rough token estimate (avg 15 tokens per prompt)
    avg_tokens = 15
    total_tokens = total * avg_tokens
    saved_tokens = n_dup * avg_tokens
    token_reduction = saved_tokens / total_tokens if total_tokens else 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "step": "B5_context_compression",
        "total_prompts": total,
        "unique": n_unique,
        "duplicates": n_dup,
        "exact_matches": exact_matches,
        "neighborhood_matches": neighborhood_matches,
        "dedup_rate": round(dedup_rate, 4),
        "token_reduction_estimate": round(token_reduction, 4),
        "total_elapsed_ms": round(elapsed_ms, 2),
        "throughput_per_sec": round(total / (elapsed_ms / 1000), 1),
    }
    (RESULTS_DIR / "b5_context_compression.json").write_text(json.dumps(out, indent=2))

    print()
    print("-- RESULTS ---------------------------------------------")
    print(f"  Total prompts          : {total}")
    print(f"  Unique (non-duplicate) : {n_unique}")
    print(f"  Duplicates caught      : {n_dup}  ({dedup_rate*100:.1f}%)")
    print(f"    Exact matches        : {exact_matches}")
    print(f"    Neighborhood matches : {neighborhood_matches}")
    print(f"  Est. token reduction   : {token_reduction*100:.1f}%")
    print(f"  Throughput             : {out['throughput_per_sec']:.0f} prompts/sec")
    print(f"  Total elapsed          : {elapsed_ms:.1f} ms")

    if dedup_rate > 0.3:
        print(f"  -> HIGH dedup: {dedup_rate*100:.0f}% of prompts were semantic duplicates")
        print(f"    Serving those from cache saves ~{token_reduction*100:.0f}% of LLM token spend")
    elif dedup_rate > 0.1:
        print(f"  -> MODERATE dedup: {dedup_rate*100:.0f}% duplicates found in stream")
    else:
        print(f"  -> LOW dedup: stream is mostly unique content ({dedup_rate*100:.0f}% dups)")

    print()
    print("B5 COMPLETE")


if __name__ == "__main__":
    main()
