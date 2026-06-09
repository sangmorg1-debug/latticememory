"""Demo: LatticeQABot on the IT helpdesk dataset.

Loads all Q&A pairs from the helpdesk prompts_responses.json into the bot's
cache, then runs every prompt through ask() to measure cache hit rate, source
distribution, and latency.  Also exercises the flywheel: any miss is logged,
clustered, and a review queue is exported.

Usage (no real model, synthetic encoder):
    python -m benchmarks.demo_qa_bot --synthetic

Usage (trained checkpoint):
    python -m benchmarks.demo_qa_bot \\
        --model benchmarks/results/snap_product_gate_hard_symmetric_8ep/best_snap_encoder

Usage (interactive REPL):
    python -m benchmarks.demo_qa_bot --model ... --interactive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic encoder (no model download required for CI / demo)
# ---------------------------------------------------------------------------

class _SyntheticEncoder:
    """Deterministic hash-based embedding — for testing only, not semantic."""

    def __init__(self, d_model: int = 1024) -> None:
        self.d_model = d_model

    def get_sentence_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **_kw):
        import hashlib
        import numpy as np

        out = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.d_model).astype("float32")
            if normalize_embeddings:
                n = float(np.linalg.norm(vec))
                vec = vec / max(n, 1e-8)
            out.append(vec)
        import numpy as np
        return np.stack(out)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

HELPDESK_QA    = Path("benchmarks/demo_data/hard_near_miss_helpdesk/prompts_responses.json")
HELPDESK_HELD  = Path("benchmarks/demo_data/hard_near_miss_helpdesk/heldout_paraphrases.json")
HELPDESK_NM    = Path("benchmarks/demo_data/hard_near_miss_helpdesk/heldout_near_misses.json")
MISS_LOG       = Path("benchmarks/results/flywheel_demo_misses.jsonl")
REVIEW_QUEUE   = Path("benchmarks/results/flywheel_demo_review_queue.json")
RESULTS_OUT    = Path("benchmarks/results/demo_qa_bot_results.json")


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_demo(
    *,
    encoder,
    interactive: bool = False,
    verbose: bool = True,
) -> dict:
    from latticememory.qa_bot import LatticeQABot
    from latticememory.flywheel import LatticeFlywheel

    # Wipe previous miss log for clean run
    if MISS_LOG.exists():
        MISS_LOG.unlink()

    flywheel = LatticeFlywheel(MISS_LOG, encoder=encoder)

    # Threshold will be swept below — start with a low value so all answers are collected
    bot = LatticeQABot(
        routing_mode="centroid",     # cosine-centroid matching — matches benchmark recall
        encoder=encoder,
        flywheel=flywheel,
        confidence_threshold=0.0,   # capture raw scores; threshold applied in sweep
        escalation_threshold=0.0,
        escalation_response="[No answer available — please contact IT support]",
    )

    # --- Load Q&A pairs --------------------------------------------------
    qa_data = _load_json(HELPDESK_QA)
    # Deduplicate by (intent_id, response) so each intent has one canonical answer
    seen: set[str] = set()
    canonical_pairs = []
    for item in qa_data:
        key = f"{item['intent_id']}::{item['response']}"
        if key not in seen:
            seen.add(key)
            canonical_pairs.append(
                {"question": item["prompt"], "answer": item["response"], "intent_id": item["intent_id"]}
            )

    # Load ALL training Q&A pairs (all paraphrase variants per intent).
    # This gives the cache enough coverage to match held-out paraphrases.
    train_pairs = [
        {"question": item["prompt"], "answer": item["response"], "intent_id": item["intent_id"]}
        for item in qa_data
    ]
    n_loaded = bot.load_qa(train_pairs)
    train_questions = {item["prompt"] for item in qa_data}
    if verbose:
        mode_label = "centroid" if bot.routing_mode == "centroid" else "E8-exact"
        print(f"\nLoaded {n_loaded} training Q&A pairs [{mode_label} mode, {bot.n_intents} intents].")

    # --- Test only on HELD-OUT paraphrases (not training data) -----------
    held_data = _load_json(HELPDESK_HELD).get("paraphrases", [])
    # Build intent lookup from training data
    prompt_to_intent = {item["prompt"]: item["intent_id"] for item in qa_data}
    intent_to_answer = {item["intent_id"]: item["response"] for item in qa_data}

    held_prompts = []
    for left, right in held_data:
        held_prompts.append(left)
        held_prompts.append(right)

    # Near-miss PAIRS — a "wrong route" is when both questions in a pair
    # route to the SAME intent (same as the intent cache benchmark metric).
    nm_data = _load_json(HELPDESK_NM).get("near_misses", [])

    # Run held-out paraphrase test
    t_total = 0.0
    para_records = []
    for q in held_prompts:
        resp = bot.ask(q)
        t_total += resp.latency_ms
        para_records.append({"question": q, "score": resp.confidence, "intent_id": resp.intent_id,
                              "latency_ms": resp.latency_ms})

    # Run near-miss pair test — two queries per pair
    nm_pair_records = []
    for left, right in nm_data:
        resp_l = bot.ask(left)
        resp_r = bot.ask(right)
        t_total += resp_l.latency_ms + resp_r.latency_ms
        nm_pair_records.append({
            "left": left,  "left_score": resp_l.confidence,  "left_intent": resp_l.intent_id,
            "right": right, "right_score": resp_r.confidence, "right_intent": resp_r.intent_id,
        })

    mean_latency = t_total / (len(para_records) + 2 * len(nm_pair_records)) if para_records else 0.0
    para_scores = [r["score"] for r in para_records]

    # Wrong-route = both sides of a pair route to the SAME intent at the given threshold
    def _wrong_routes(threshold: float) -> int:
        count = 0
        for p in nm_pair_records:
            served_l = p["left_score"]  >= threshold
            served_r = p["right_score"] >= threshold
            if served_l and served_r and p["left_intent"] == p["right_intent"]:
                count += 1
        return count

    # --- Threshold sweep -------------------------------------------------
    all_scores = para_scores + [p["left_score"] for p in nm_pair_records] + [p["right_score"] for p in nm_pair_records]
    thresholds = sorted(set(all_scores), reverse=True)
    threshold_curve = []
    for t in thresholds:
        tp = sum(1 for s in para_scores if s >= t)
        wr = _wrong_routes(t)
        recall_t  = tp / len(para_scores)    if para_scores    else 0.0
        fp_rate_t = wr / len(nm_pair_records) if nm_pair_records else 0.0
        threshold_curve.append({"threshold": round(t, 4), "recall": round(recall_t, 4),
                                  "fp_rate": round(fp_rate_t, 4), "tp": tp, "wrong_routes": wr})

    zero_fp_points = [p for p in threshold_curve if p["fp_rate"] == 0.0]
    zero_fp = max(zero_fp_points, key=lambda p: p["recall"]) if zero_fp_points else \
              {"threshold": 1.0, "recall": 0.0, "fp_rate": 0.0, "tp": 0, "wrong_routes": 0}

    default_points = [p for p in threshold_curve if p["recall"] >= 0.85]
    default_op = min(default_points, key=lambda p: (p["fp_rate"], -p["recall"])) if default_points else \
                 (threshold_curve[-1] if threshold_curve else zero_fp)

    recall     = default_op["recall"]
    nm_fp      = default_op["wrong_routes"]
    nm_total   = len(nm_pair_records)
    cache_hits = default_op["tp"]

    results = para_records  # for legacy summary key

    # --- Flywheel stats --------------------------------------------------
    fw_stats = flywheel.stats()
    clusters = flywheel.cluster_gaps(min_cluster_size=2)
    n_exported = flywheel.export_review_queue(REVIEW_QUEUE, n=10, min_cluster_size=2)

    # --- Print summary ---------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"  LatticeQABot Demo — IT Helpdesk Dataset")
        print(f"{'='*60}")
        print(f"  Held-out queries   : {len(para_scores)}")
        print(f"  Cache recall       : {cache_hits} / {len(para_scores)} ({recall*100:.1f}%)  [threshold={default_op['threshold']:.4f}]")
        print(f"  Near-miss pairs    : {nm_total}  wrong-routes: {nm_fp} ({nm_fp/max(nm_total,1)*100:.1f}%)")
        print(f"  Zero-FP point      : recall={zero_fp['recall']*100:.1f}%  threshold={zero_fp['threshold']:.4f}")
        print(f"  Mean latency       : {mean_latency:.4f} ms")
        print(f"  Misses logged      : {fw_stats['total_misses']}")
        print(f"  Miss clusters      : {len(clusters)}")
        print(f"  Review queue items : {n_exported} -> {REVIEW_QUEUE}")
        print(f"{'='*60}\n")

        if clusters:
            print("  Top miss clusters (coverage gaps to fill):")
            for c in clusters[:5]:
                print(f"    [{c.size:>3} misses] {c.representative!r}")
            print()

    # --- Serialise results -----------------------------------------------
    report = {
        "summary": {
            "n_training_pairs": n_loaded,
            "n_intents": bot.n_intents,
            "n_held_out_queries": len(para_scores),
            "n_near_miss_pairs": nm_total,
            "default_operating_point": default_op,
            "zero_fp_operating_point": zero_fp,
            "cache_recall": round(recall, 4),
            "wrong_route_rate": round(nm_fp / max(nm_total, 1), 4),
            "wrong_routes": nm_fp,
            "mean_latency_ms": round(mean_latency, 4),
            "flywheel_misses": fw_stats["total_misses"],
            "flywheel_clusters": len(clusters),
            "review_queue_path": str(REVIEW_QUEUE),
        },
        "threshold_curve": threshold_curve,
        "per_question": results,
    }

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"  Results -> {RESULTS_OUT}")

    # --- Interactive REPL ------------------------------------------------
    if interactive:
        print("\nInteractive mode. Type a question or 'quit' to exit.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            resp = bot.ask(q)
            print(f"Bot ({resp.source}, conf={resp.confidence:.2f}, {resp.latency_ms:.2f}ms): {resp.answer}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LatticeQABot demo on the IT helpdesk dataset"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Encoder model name/path. Omit to use synthetic encoder.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic encoder (fast, no downloads).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After benchmark, drop into interactive REPL.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output.")
    args = parser.parse_args()

    if args.synthetic or args.model is None:
        encoder = _SyntheticEncoder()
        if not args.quiet:
            print("[demo] Using synthetic encoder (hash-based, not semantic).")
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.model)

    report = run_demo(
        encoder=encoder,
        interactive=args.interactive,
        verbose=not args.quiet,
    )

    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
