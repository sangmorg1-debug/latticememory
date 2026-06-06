"""
GPTCache-style semantic cache benchmark.

Simulates a realistic enterprise support query log and compares:
  - No cache baseline (every query → LLM)
  - Exact-string cache (naive dict, catches repeats only)
  - LatticeMemory E8 cache (catches repeats + semantic paraphrases)

Query log mix (configurable):
  exact_repeat  — same text as a cached canonical question
  paraphrase    — different wording, same intent (tests E8 neighborhood routing)
  novel         — unrelated question (should miss → LLM)

Modes:
  --model synthetic   offline smoke test; paraphrases use the suffix-strip trick
                      so SyntheticTextEncoder maps them to the same E8 cell.
  --model <hf-id>     real model; natural-language paraphrases test genuine
                      Hamming-1 neighborhood routing.

Usage:
  python -m benchmarks.benchmark_semantic_cache --model synthetic
  python -m benchmarks.benchmark_semantic_cache --model dfrokido/bge-large-e8-snap
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common import load_text_encoder, write_json_result

# ---------------------------------------------------------------------------
# Curated enterprise support Q&A pairs
# ---------------------------------------------------------------------------
QUERY_GROUPS = [
    {
        "canonical": "What is the refund policy?",
        "paraphrases": [
            "Can I return an item for a refund?",
            "How do I get my money back?",
            "What's your return policy?",
            "Do you offer refunds?",
            "I want to return my purchase",
        ],
        "answer": "Our refund policy allows returns within 30 days of purchase.",
    },
    {
        "canonical": "How do I reset my password?",
        "paraphrases": [
            "I forgot my password",
            "Can't log in, need password help",
            "Password reset instructions please",
            "How to change my password",
            "Send me a password reset link",
        ],
        "answer": "Click Forgot Password on the login page to reset your password.",
    },
    {
        "canonical": "Where is my order?",
        "paraphrases": [
            "What's the status of my order?",
            "Has my package shipped?",
            "Track my order please",
            "When will my order arrive?",
            "My order hasn't arrived yet",
        ],
        "answer": "Track your order using the link in your confirmation email.",
    },
    {
        "canonical": "How do I cancel my subscription?",
        "paraphrases": [
            "I want to stop my subscription",
            "How do I unsubscribe?",
            "Cancel my account please",
            "How to turn off auto-renewal?",
            "I don't want to be charged anymore",
        ],
        "answer": "Cancel via Account Settings > Subscription > Cancel.",
    },
    {
        "canonical": "What payment methods do you accept?",
        "paraphrases": [
            "Do you take PayPal?",
            "Can I pay with credit card?",
            "What are your payment options?",
            "Do you accept Apple Pay?",
            "Is there a way to pay by invoice?",
        ],
        "answer": "We accept Visa, Mastercard, PayPal, Apple Pay, and bank transfers.",
    },
    {
        "canonical": "How do I contact customer support?",
        "paraphrases": [
            "I need to speak to someone",
            "What's your support phone number?",
            "How can I reach your team?",
            "Customer service contact please",
            "I have a complaint who do I call?",
        ],
        "answer": "Reach support at support@example.com or 1-800-EXAMPLE.",
    },
    {
        "canonical": "What are your business hours?",
        "paraphrases": [
            "When are you open?",
            "Are you available on weekends?",
            "What time does support close?",
            "Do you have 24/7 support?",
            "When can I call you?",
        ],
        "answer": "We are available Monday to Friday, 9 AM to 6 PM EST.",
    },
    {
        "canonical": "How do I update my billing information?",
        "paraphrases": [
            "I need to change my credit card",
            "Update payment method please",
            "My card expired how do I change it?",
            "How to update billing address?",
            "Change my payment details",
        ],
        "answer": "Update billing in Account Settings > Payment Methods.",
    },
    {
        "canonical": "Do you offer a free trial?",
        "paraphrases": [
            "Can I try before I buy?",
            "Is there a demo available?",
            "Free tier available?",
            "How long is the trial period?",
            "Can I test the product first?",
        ],
        "answer": "Yes, we offer a 14-day free trial with no credit card required.",
    },
    {
        "canonical": "How do I download my invoice?",
        "paraphrases": [
            "I need a receipt for my purchase",
            "Where can I find my billing history?",
            "Download invoice for tax purposes",
            "Get a copy of my payment confirmation",
            "Need proof of purchase",
        ],
        "answer": "Download invoices from Account Settings > Billing > Invoice History.",
    },
]

NOVEL_QUERIES = [
    "Tell me about quantum computing applications in finance",
    "What is the difference between TCP and UDP protocols?",
    "Explain gradient descent in machine learning",
    "How does the human immune system respond to vaccines?",
    "What were the main causes of the French Revolution?",
    "Describe the architecture of a transformer neural network",
    "How do black holes form from stellar collapse?",
    "What are the environmental impacts of lithium mining?",
    "Explain the difference between supervised and unsupervised learning",
    "What is the significance of the Turing test?",
    "How does CRISPR gene editing work?",
    "What is the role of mitochondria in cellular respiration?",
    "Explain the blockchain consensus mechanism",
    "What are the main tenets of stoic philosophy?",
    "How is a Fourier transform used in signal processing?",
]

SIMULATED_LLM_LATENCY_MS = 650.0


def _build_query_log(
    *,
    rng: random.Random,
    repeat_rate: float,
    paraphrase_rate: float,
    novel_rate: float,
    n_queries: int,
    synthetic_mode: bool,
) -> list[dict]:
    n_repeat = int(n_queries * repeat_rate)
    n_paraphrase = int(n_queries * paraphrase_rate)
    n_novel = n_queries - n_repeat - n_paraphrase
    log = []

    for _ in range(n_repeat):
        group = rng.choice(QUERY_GROUPS)
        log.append({"text": group["canonical"], "type": "exact_repeat", "canonical": group["canonical"]})

    for _ in range(n_paraphrase):
        group = rng.choice(QUERY_GROUPS)
        if synthetic_mode:
            # Suffix trick: SyntheticTextEncoder strips " :: duplicate variant" → same MD5 as canonical
            text = f"{group['canonical']} :: duplicate variant"
        else:
            text = rng.choice(group["paraphrases"])
        log.append({"text": text, "type": "paraphrase", "canonical": group["canonical"]})

    for _ in range(n_novel):
        log.append({"text": rng.choice(NOVEL_QUERIES), "type": "novel", "canonical": None})

    rng.shuffle(log)
    return log


def run_benchmark(
    *,
    model: str = "synthetic",
    d_model: int | None = None,
    n_queries: int = 500,
    repeat_rate: float = 0.30,
    paraphrase_rate: float = 0.30,
    device: str | None = None,
    seed: int = 42,
    llm_latency_ms: float = SIMULATED_LLM_LATENCY_MS,
    output_path: str | Path | None = None,
) -> dict:
    novel_rate = round(1.0 - repeat_rate - paraphrase_rate, 10)
    if novel_rate < -1e-9:
        raise ValueError("repeat_rate + paraphrase_rate cannot exceed 1.0")
    novel_rate = max(novel_rate, 0.0)

    rng = random.Random(seed)
    synthetic_mode = model == "synthetic"

    encoder, detected_dim = load_text_encoder(model, d_model, device=device)

    from latticememory.index import LatticeIndex
    idx = LatticeIndex.__new__(LatticeIndex)
    idx._init_with_encoder(encoder, d_model=detected_dim)

    string_cache: dict[str, str] = {}
    warmup_texts = [g["canonical"] for g in QUERY_GROUPS]
    idx.add(warmup_texts)
    for group in QUERY_GROUPS:
        string_cache[group["canonical"]] = group["answer"]

    query_log = _build_query_log(
        rng=rng,
        repeat_rate=repeat_rate,
        paraphrase_rate=paraphrase_rate,
        novel_rate=novel_rate,
        n_queries=n_queries,
        synthetic_mode=synthetic_mode,
    )

    counts: dict[str, dict[str, int]] = {
        qt: {"exact_string": 0, "e8_exact": 0, "e8_hamming1": 0, "miss": 0}
        for qt in ("exact_repeat", "paraphrase", "novel")
    }
    latencies: dict[str, list[float]] = defaultdict(list)

    for entry in query_log:
        text = entry["text"]
        qtype = entry["type"]

        if text in string_cache:
            counts[qtype]["exact_string"] += 1

        t0 = time.perf_counter()
        hits = idx.search(text, top_k=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if hits and hits[0].retrieval_path == "lattice_exact":
            counts[qtype]["e8_exact"] += 1
            latencies["e8_exact"].append(elapsed_ms)
        elif hits and hits[0].retrieval_path == "lattice_hamming1":
            counts[qtype]["e8_hamming1"] += 1
            latencies["e8_hamming1"].append(elapsed_ms)
        else:
            counts[qtype]["miss"] += 1
            latencies["miss"].append(elapsed_ms)

    n = len(query_log)

    def _pct(key: str) -> float:
        total = sum(counts[qt][key] for qt in counts)
        return round(total / n, 4)

    def _p(tier: str, q: int) -> float | None:
        vals = latencies[tier]
        return round(float(np.percentile(vals, q)), 3) if vals else None

    e8_hit_rate = _pct("e8_exact") + _pct("e8_hamming1")
    string_hit_rate = _pct("exact_string")
    miss_rate = _pct("miss")

    result = {
        "benchmark": "semantic_cache",
        "model": model,
        "synthetic_mode": synthetic_mode,
        "n_canonical_cached": len(QUERY_GROUPS),
        "n_queries_served": n,
        "query_mix": {
            "exact_repeat_rate": repeat_rate,
            "paraphrase_rate": paraphrase_rate,
            "novel_rate": novel_rate,
        },
        "cache_hit_rate": {
            "no_cache_baseline": 0.0,
            "exact_string_cache": string_hit_rate,
            "lattice_e8_total": e8_hit_rate,
            "lattice_exact": _pct("e8_exact"),
            "lattice_hamming1": _pct("e8_hamming1"),
            "miss_rate": miss_rate,
        },
        "paraphrase_uplift": round(e8_hit_rate - string_hit_rate, 4),
        "hit_rate_by_query_type": {
            qtype: {
                "n": sum(v for v in r.values()),
                "exact_string": r["exact_string"],
                "e8_exact": r["e8_exact"],
                "e8_hamming1": r["e8_hamming1"],
                "miss": r["miss"],
            }
            for qtype, r in counts.items()
        },
        "latency_ms": {
            "e8_exact_p50": _p("e8_exact", 50),
            "e8_exact_p95": _p("e8_exact", 95),
            "e8_hamming1_p50": _p("e8_hamming1", 50),
            "e8_hamming1_p95": _p("e8_hamming1", 95),
            "miss_p50": _p("miss", 50),
            "miss_p95": _p("miss", 95),
        },
        "api_savings": {
            "assumed_llm_latency_ms": llm_latency_ms,
            "lattice_e8_savings_pct": round(e8_hit_rate * 100, 1),
            "exact_string_savings_pct": round(string_hit_rate * 100, 1),
            "paraphrase_uplift_pct": round((e8_hit_rate - string_hit_rate) * 100, 1),
        },
    }
    return write_json_result(result, output_path)


def _print_report(r: dict) -> None:
    hit = r["cache_hit_rate"]
    api = r["api_savings"]
    lat = r["latency_ms"]
    mix = r["query_mix"]
    mode_note = "  [synthetic - paraphrases use suffix-strip trick]" if r["synthetic_mode"] else ""

    print()
    print("=" * 62)
    print("  LatticeMemory Semantic Cache Benchmark")
    print("=" * 62)
    print(f"  Model    : {r['model']}{mode_note}")
    print(f"  Cached   : {r['n_canonical_cached']} canonical Q&A pairs")
    print(f"  Queries  : {r['n_queries_served']}  "
          f"({mix['exact_repeat_rate']:.0%} repeat / "
          f"{mix['paraphrase_rate']:.0%} paraphrase / "
          f"{mix['novel_rate']:.0%} novel)")
    print()
    print("  CACHE HIT RATE")
    print(f"    No cache (baseline)         {hit['no_cache_baseline']:>6.1%}   all -> LLM")
    print(f"    Exact-string cache          {hit['exact_string_cache']:>6.1%}   repeats only")
    print(f"    LatticeMemory E8            {hit['lattice_e8_total']:>6.1%}   "
          f"(+{r['paraphrase_uplift']:.1%} from paraphrases)")
    print()
    print("  E8 PATH BREAKDOWN")
    print(f"    lattice_exact  (O(1))       {hit['lattice_exact']:>6.1%}")
    print(f"    lattice_hamming1 (O(1))     {hit['lattice_hamming1']:>6.1%}")
    print(f"    miss -> LLM                 {hit['miss_rate']:>6.1%}")
    print()
    print("  LATENCY (p50 / p95)")
    e8e = f"{lat['e8_exact_p50']} / {lat['e8_exact_p95']} ms" if lat["e8_exact_p50"] else "n/a"
    e8h = f"{lat['e8_hamming1_p50']} / {lat['e8_hamming1_p95']} ms" if lat["e8_hamming1_p50"] else "n/a"
    mis = f"{lat['miss_p50']} / {lat['miss_p95']} ms" if lat["miss_p50"] else "n/a"
    print(f"    E8 exact lookup             {e8e}")
    print(f"    E8 hamming1 lookup          {e8h}")
    print(f"    Miss (cache check only)     {mis} + {api['assumed_llm_latency_ms']:.0f}ms LLM")
    print()
    print("  ESTIMATED API CALL SAVINGS vs NO CACHE")
    print(f"    LatticeMemory E8            {api['lattice_e8_savings_pct']:.1f}% fewer LLM calls")
    print(f"    Exact-string cache          {api['exact_string_savings_pct']:.1f}% fewer LLM calls")
    print(f"    Paraphrase uplift           +{api['paraphrase_uplift_pct']:.1f}pp from E8 semantics")
    print("=" * 62)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTCache-style semantic cache benchmark.")
    parser.add_argument("--model", default="synthetic",
                        help="'synthetic' for offline smoke test or a HuggingFace model ID")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--repeat-rate", type=float, default=0.30)
    parser.add_argument("--paraphrase-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-latency-ms", type=float, default=SIMULATED_LLM_LATENCY_MS)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/results/semantic_cache.json"))
    args = parser.parse_args()

    result = run_benchmark(
        model=args.model,
        d_model=args.d_model,
        n_queries=args.n_queries,
        repeat_rate=args.repeat_rate,
        paraphrase_rate=args.paraphrase_rate,
        seed=args.seed,
        llm_latency_ms=args.llm_latency_ms,
        output_path=args.output,
    )
    _print_report(result)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
