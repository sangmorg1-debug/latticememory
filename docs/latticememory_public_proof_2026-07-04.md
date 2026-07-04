# LatticeMemory Public Proof Pack

**Date:** 2026-07-04
**Artifact:** `artifacts/proxy_pq_redis_flywheel_bitext_large_2026-07-04/`
**Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
**Split:** 500 seed / 1,000 calibration / 5,000 evaluation / 1,000 adversarial
**Redis:** `redis/redis-stack-server:latest` via Docker Engine in WSL at `redis://localhost:6382/0`

## Supported Claim

LatticeMemory is an auditable semantic-cache proxy for repeated and paraphrased
support-style LLM requests. It reports hit rate, upstream-call rate, latency,
Redis memory, flywheel review behavior, and false-positive rate together.

This proof does not claim that LatticeMemory replaces vector databases, beats
RedisVL, or solves general RAG retrieval.

## Main Matrix

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB |
|---|---:|---:|---:|---:|---:|---:|
| exact_string | 0.3947 | 0.6053 | 0.0000 | 0.0000 | 0.000 | 0.0000 |
| dense_cosine | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.212 | 0.0000 |
| lattice_pq_local | 1.0000 | 0.0000 | 0.0188 | 0.1110 | 3.089 | 0.0000 |
| lattice_pq_validated_cosine | 0.9812 | 0.0188 | 0.0000 | 0.0000 | 0.311 | 0.0000 |
| lattice_pq_redis_real | 1.0000 | 0.0000 | 0.0188 | 0.1110 | 4.662 | 0.0534 |
| lattice_pq_redis_validated_cosine | 0.9812 | 0.0188 | 0.0000 | 0.0000 | 1.021 | 0.0534 |
| redisvl_direct | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.796 | 0.0000 |
| gptcache_direct | 0.3947 | 0.6053 | 0.0000 | 0.0000 | 0.001 | 0.0000 |

## Product Policy

The target product policy is `lattice_pq_redis_validated_cosine`:

- PQ generates a compressed candidate.
- A cosine gate verifies the source prompt before serving.
- Rejected candidates fall through to upstream instead of being served.
- Redis-backed shared storage and persistence checks are enabled.

On this split, that row avoids the raw PQ false-positive problem:

- Raw Redis PQ adversarial FP rate: `0.1110`
- Validated Redis PQ adversarial FP rate: `0.0000`
- Validated Redis PQ upstream-call rate: `0.0188`

RedisVL direct is also strong on this deterministic proof-pack encoder:
`1.0000` hit rate and `0.0000` false positives. The LatticeMemory advantage to
claim here is not RedisVL superiority; it is the explicit operating-policy
matrix, validation gate, false-positive accounting, and flywheel-ready proxy
shape.

## Live Proxy Replay

`examples/proxy_replay_demo.py` replays the same large dataset through the
OpenAI-compatible proxy path with Redis-backed cache storage and cache cosine
validation enabled.

| Metric | Value |
|---|---:|
| Total requests | 6,000 |
| Hit rate | 0.9917 |
| Upstream call rate | 0.0083 |
| False-positive rate | 0.0000 |
| Adversarial false-positive rate | 0.0000 |
| Rejected candidates | 50 |
| Avg latency ms | 4.383 |

## Profiling Result

The previous large-run issue was observability, not a confirmed Redis deadlock.
The new `proof_pack_progress.jsonl` file records every completed run. In the
large run, `lattice_pq_redis_real` completed in `28.145s`; it was the slowest
row, but it did finish. The validated Redis PQ row completed in `6.126s`.

## Public Wording

Allowed:

> On a 6,000-request public Bitext support workload, LatticeMemory's Redis-backed
> validated PQ cache row served 98.12% of requests with zero measured false
> positives, while reporting adversarial false positives, latency, Redis memory,
> and upstream-call rate beside the hit rate.

Not allowed:

- LatticeMemory replaces RedisVL.
- LatticeMemory replaces vector databases.
- Raw PQ hits are safe without validation.
- This proof establishes general RAG superiority.
