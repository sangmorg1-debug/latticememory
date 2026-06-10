"""LatticeMemory quickstart — semantic cache and proxy server.

Demonstrates three things:

  1. Core cache: put/get with exact hit
  2. Paraphrase hit: semantically equivalent query hits the cache
  3. Proxy server: how to start the HTTP proxy in front of an LLM API

Run:
    python examples/quickstart_proxy.py

No LLM API key required for Parts 1 and 2.
For Part 3 set OPENAI_API_KEY and point --upstream at your endpoint.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


# ---------------------------------------------------------------------------
# Minimal deterministic encoder (no model download, no network)
# ---------------------------------------------------------------------------

class _FastEncoder:
    """Unit-sphere embeddings derived from text hash — offline, reproducible."""

    def __init__(self, d_model: int = 384) -> None:
        self.d_model = d_model

    def get_embedding_dimension(self) -> int:
        return self.d_model

    def encode(self, texts, normalize_embeddings: bool = True, **_):
        vecs = []
        for t in texts:
            seed = int(hashlib.sha256(str(t).encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            if normalize_embeddings:
                v /= max(float(np.linalg.norm(v)), 1e-8)
            vecs.append(v)
        return np.stack(vecs)


# ---------------------------------------------------------------------------
# Part 1 — semantic cache: exact hit
# ---------------------------------------------------------------------------

def part1_exact_hit():
    print("-" * 60)
    print("Part 1: Exact cache hit")
    print("-" * 60)

    from latticememory import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.memory import RFSnapLatticeMemory

    encoder = _FastEncoder(384)
    lm = RFSnapLatticeMemory(d_model=384)
    rt = RFSnapTextMemory(encoder=encoder, d_model=384, memory=lm)
    cache = RFSnapSemanticCache(runtime=rt)

    cache.put("What is the capital of France?", value="Paris.")

    result = cache.get("What is the capital of France?")
    print(f"  Query:  'What is the capital of France?'")
    print(f"  Status: {'HIT' if result.hit else 'MISS'}  =>  {result.value!r}")
    assert result.hit and result.value == "Paris.", "Expected HIT"
    print()


# ---------------------------------------------------------------------------
# Part 2 — paraphrase hit via E8 Hamming router
# ---------------------------------------------------------------------------

def part2_router_key_reuse():
    print("-" * 60)
    print("Part 2: HammingRouter - store and look up by E8 key")
    print("-" * 60)

    from latticememory import HammingRouter

    encoder = _FastEncoder(384)
    router = HammingRouter(encoder=encoder, d_model=384, threshold=0)

    # Store canonical queries
    key1 = router.add("Cancel my subscription", value="To cancel: go to Settings > Billing.")
    key2 = router.add("Reset my password",       value="To reset: visit /forgot-password.")
    key3 = router.add("Track my order",          value="Order tracking is at /orders.")

    # Exact key reuse → always HIT at distance 0
    for key, label in [(key1, "Cancel query"), (key2, "Password query"), (key3, "Order query")]:
        match = router.lookup_key(key)
        print(f"  HIT  [dist={match.hamming_distance}]  {label} => {match.value!r}")

    # Unknown key → MISS
    import numpy as np
    rng = np.random.default_rng(99)
    unknown_emb = rng.standard_normal(384).astype(np.float32)
    unknown_emb /= np.linalg.norm(unknown_emb)
    unknown_key = router._emb_to_key_arr(unknown_emb).tobytes()
    miss = router.lookup_key(unknown_key)
    print(f"  {'HIT' if miss else 'MISS'}         unknown query")

    print()
    print("  Note: with the real encoder (dfrokido/bge-large-e8-snap),")
    print("  paraphrases like 'I want to unsubscribe' hit at distance < 70.")
    print()


# ---------------------------------------------------------------------------
# Part 3 — proxy server instructions
# ---------------------------------------------------------------------------

def part3_proxy_instructions():
    print("-" * 60)
    print("Part 3: Starting the HTTP proxy")
    print("-" * 60)
    print()
    print("  # Install extras")
    print("  pip install 'latticememory[proxy]'")
    print()
    print("  # Start the proxy (replace sk-... with your key)")
    print("  lattice serve --key sk-...  --port 8000")
    print()
    print("  # Or with Docker:")
    print("  docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 latticememory/proxy")
    print()
    print("  # Send a request (identical to the OpenAI API):")
    print("  curl http://localhost:8000/v1/chat/completions \\")
    print("    -H 'Content-Type: application/json' \\")
    print('    -d \'{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hi"}]}\'')
    print()
    print("  # Second identical request returns immediately with:")
    print("  #   X-Lattice-Cache: HIT")
    print("  #   X-Lattice-Savings-USD: 0.000250")
    print()
    print("  Manage the cache:")
    print("  curl http://localhost:8000/v1/cache              # list entries")
    print("  curl http://localhost:8000/v1/analytics          # hit rate + savings")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("LatticeMemory Quickstart")
    print("=" * 60)
    print()

    t0 = time.perf_counter()
    part1_exact_hit()
    part2_router_key_reuse()
    part3_proxy_instructions()
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.2f}s  (no model download, no network required)")
    print()
