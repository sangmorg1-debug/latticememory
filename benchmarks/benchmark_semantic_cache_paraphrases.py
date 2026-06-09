"""
Semantic cache benchmark with real natural-language paraphrases.

This benchmark measures the cache hit rate for a realistic distribution:
- 30% repeat queries (exact text that was previously cached)
- 30% paraphrase queries (semantically equivalent but different wording)
- 40% novel queries (first time seeing this)

Results are saved to benchmarks/results/semantic_cache_paraphrase_benchmark.json
"""
import json
import time
from pathlib import Path

from latticememory import LatticeIndex


# Real paraphrase pairs from customer support, documentation, and common queries
PARAPHRASE_PAIRS = [
    ("What is your refund policy?", "How do I get a refund?"),
    ("What is your refund policy?", "Can I return my purchase?"),
    ("What is your refund policy?", "What's your return window?"),
    ("How do I reset my password?", "I forgot my password"),
    ("How do I reset my password?", "Password reset instructions?"),
    ("How do I reset my password?", "Can't log in, need to reset"),
    ("Where is my order?", "Track my order"),
    ("Where is my order?", "What's the status of my purchase?"),
    ("Where is my order?", "When will my order arrive?"),
    ("Do you offer free shipping?", "Is shipping free?"),
    ("Do you offer free shipping?", "Any shipping discount?"),
    ("Do you offer free shipping?", "Shipping cost?"),
    ("What payment methods do you accept?", "Can I pay with credit card?"),
    ("What payment methods do you accept?", "Do you take PayPal?"),
    ("What payment methods do you accept?", "Accepted payment options?"),
    ("How long is your warranty?", "What's the warranty period?"),
    ("How long is your warranty?", "Warranty coverage details?"),
    ("How long is your warranty?", "Does this product have a warranty?"),
    ("Can I cancel my subscription?", "How do I cancel?"),
    ("Can I cancel my subscription?", "Unsubscribe instructions"),
    ("Can I cancel my subscription?", "How to stop the service?"),
    ("Are there bulk discounts?", "Volume pricing available?"),
    ("Are there bulk discounts?", "Do you offer discounts for large orders?"),
    ("Are there bulk discounts?", "Wholesale pricing?"),
]

# Novel queries (40% of distribution)
NOVEL_QUERIES = [
    "What is your company's sustainability policy?",
    "Do you have international offices?",
    "What are your career opportunities?",
    "How can I report a bug?",
    "What third-party integrations do you support?",
    "Is there an API available?",
    "Can I export my data?",
    "Do you offer SSO?",
    "What SLA do you provide?",
    "How often do you release updates?",
]


def run_benchmark() -> dict:
    """Run the semantic cache benchmark and return metrics."""
    print("=" * 70)
    print("LatticeMemory Semantic Cache Benchmark (Real Paraphrases)")
    print("=" * 70)
    print()

    index = LatticeIndex(mode="cache")
    
    # Build initial cache with first item from each paraphrase pair + some novel queries
    print("1. Pre-populating cache with canonical queries...")
    cached_texts = [pair[0] for pair in PARAPHRASE_PAIRS]
    cached_texts.extend(NOVEL_QUERIES[:5])  # Add some novel queries to cache
    
    for text in cached_texts:
        index.add([text])
    
    print(f"   Cache size: {len(cached_texts)} entries")
    print()

    # Simulate realistic query distribution
    print("2. Simulating realistic query traffic:")
    
    # Create query distribution
    import random
    random.seed(42)
    
    repeat_rate = 0.30
    paraphrase_rate = 0.30
    novel_rate = 0.40
    
    # Determine counts based on rates
    total_queries = 100
    n_repeats = int(total_queries * repeat_rate)
    n_paraphrases = int(total_queries * paraphrase_rate)
    n_novel = int(total_queries * novel_rate)
    
    print(f"   - {n_repeats} repeat queries (exact cached text)")
    print(f"   - {n_paraphrases} paraphrase queries (semantically similar)")
    print(f"   - {n_novel} novel queries (not in cache)")
    print()

    # Generate query stream
    queries = []
    
    # Add repeats (exact copies of cached queries)
    for _ in range(n_repeats):
        q = random.choice(cached_texts)
        queries.append(("repeat", q))
    
    # Add paraphrases (second variant of paraphrase pairs)
    for _ in range(n_paraphrases):
        pair = random.choice(PARAPHRASE_PAIRS)
        queries.append(("paraphrase", pair[1]))
    
    # Add novel queries (completely new)
    for _ in range(n_novel):
        q = random.choice(NOVEL_QUERIES[5:])
        queries.append(("novel", q))
    
    # Shuffle
    random.shuffle(queries)

    # Run queries and measure cache hits
    print("3. Running queries through cache...")
    start_time = time.time()
    
    results_by_type = {"repeat": {"hits": 0, "misses": 0}, "paraphrase": {"hits": 0, "misses": 0}, "novel": {"hits": 0, "misses": 0}}
    hit_latencies = []
    miss_latencies = []
    
    for i, (query_type, q) in enumerate(queries):
        query_start = time.time()
        result = index.search(q, top_k=1)
        query_elapsed = (time.time() - query_start) * 1000  # Convert to ms

        # Check if we got a hit on exact or hamming distance
        is_hit = result and result[0].retrieval_path in {"lattice_exact", "lattice_hamming1"}
        
        if is_hit:
            results_by_type[query_type]["hits"] += 1
            hit_latencies.append(query_elapsed)
        else:
            results_by_type[query_type]["misses"] += 1
            miss_latencies.append(query_elapsed)

        if (i + 1) % 25 == 0:
            total_hits = sum(t["hits"] for t in results_by_type.values())
            total_misses = sum(t["misses"] for t in results_by_type.values())
            print(f"   Progress: {i+1}/{len(queries)} queries ({total_hits} hits, {total_misses} misses)")

    total_time = time.time() - start_time

    # Calculate statistics
    total_hits = sum(t["hits"] for t in results_by_type.values())
    total_misses = sum(t["misses"] for t in results_by_type.values())
    hit_rate = total_hits / len(queries) if queries else 0.0
    avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
    avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0

    print()
    print("4. Results by Query Type:")
    for qtype in ["repeat", "paraphrase", "novel"]:
        hits = results_by_type[qtype]["hits"]
        misses = results_by_type[qtype]["misses"]
        total = hits + misses
        hit_pct = (hits / total * 100) if total > 0 else 0
        print(f"   {qtype:12} {hits:3} hits, {misses:3} misses ({hit_pct:5.1f}% hit rate)")
    
    print()
    print("5. Overall Statistics:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Cache hits: {total_hits} ({hit_rate*100:.1f}%)")
    print(f"   Cache misses: {total_misses} ({(1-hit_rate)*100:.1f}%)")
    print(f"   Avg hit latency: {avg_hit_latency:.3f} ms")
    print(f"   Avg miss latency: {avg_miss_latency:.3f} ms")
    print(f"   Total time: {total_time:.2f}s")
    print()

    # Estimate cost savings (OpenAI API pricing)
    cost_per_1k_tokens = 0.005
    avg_tokens_per_query = 50
    cost_per_query = (avg_tokens_per_query / 1000) * cost_per_1k_tokens
    total_cost_without_cache = len(queries) * cost_per_query
    total_cost_with_cache = total_misses * cost_per_query
    total_savings = total_cost_without_cache - total_cost_with_cache

    print(f"6. Cost Analysis (based on OpenAI API pricing):")
    print(f"   Cost per query: ${cost_per_query:.6f}")
    print(f"   Total cost without cache: ${total_cost_without_cache:.2f}")
    print(f"   Total cost with cache: ${total_cost_with_cache:.2f}")
    print(f"   **Total savings: ${total_savings:.2f}** ({(total_savings/total_cost_without_cache)*100:.1f}%)")
    print()

    return {
        "artifact_type": "latticememory_semantic_cache_paraphrase_benchmark",
        "artifact_version": 1,
        "timestamp": time.time(),
        "metrics": {
            "total_queries": len(queries),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(hit_rate, 4),
            "avg_hit_latency_ms": round(avg_hit_latency, 3),
            "avg_miss_latency_ms": round(avg_miss_latency, 3),
            "total_time_seconds": round(total_time, 2),
            "by_type": {
                "repeat": {
                    "hits": results_by_type["repeat"]["hits"],
                    "misses": results_by_type["repeat"]["misses"],
                    "hit_rate": round(results_by_type["repeat"]["hits"] / (results_by_type["repeat"]["hits"] + results_by_type["repeat"]["misses"]), 4) if (results_by_type["repeat"]["hits"] + results_by_type["repeat"]["misses"]) > 0 else 0,
                },
                "paraphrase": {
                    "hits": results_by_type["paraphrase"]["hits"],
                    "misses": results_by_type["paraphrase"]["misses"],
                    "hit_rate": round(results_by_type["paraphrase"]["hits"] / (results_by_type["paraphrase"]["hits"] + results_by_type["paraphrase"]["misses"]), 4) if (results_by_type["paraphrase"]["hits"] + results_by_type["paraphrase"]["misses"]) > 0 else 0,
                },
                "novel": {
                    "hits": results_by_type["novel"]["hits"],
                    "misses": results_by_type["novel"]["misses"],
                    "hit_rate": round(results_by_type["novel"]["hits"] / (results_by_type["novel"]["hits"] + results_by_type["novel"]["misses"]), 4) if (results_by_type["novel"]["hits"] + results_by_type["novel"]["misses"]) > 0 else 0,
                },
            },
        },
        "cost_analysis": {
            "cost_per_1k_tokens_usd": cost_per_1k_tokens,
            "avg_tokens_per_query": avg_tokens_per_query,
            "cost_per_query_usd": round(cost_per_query, 6),
            "total_cost_without_cache_usd": round(total_cost_without_cache, 2),
            "total_cost_with_cache_usd": round(total_cost_with_cache, 2),
            "total_savings_usd": round(total_savings, 2),
            "savings_percentage": round((total_savings / total_cost_without_cache) * 100, 1),
        },
        "cache_stats": {
            "cache_size": len(cached_texts),
            "paraphrase_pairs_tested": len(PARAPHRASE_PAIRS),
            "novel_queries_pool": len(NOVEL_QUERIES),
        },
    }


if __name__ == "__main__":
    results = run_benchmark()
    
    # Save results
    results_path = Path("benchmarks/results/semantic_cache_paraphrase_benchmark.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print()
    print("=" * 70)
