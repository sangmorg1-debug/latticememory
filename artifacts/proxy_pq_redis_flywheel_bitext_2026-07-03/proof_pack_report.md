# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack

This artifact compares exact string caching, dense semantic caching, and LatticeMemory proxy paths.
It is a support-workload proof pack, not a general RAG or vector-database replacement claim.

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_string | 0.335 | 0.665 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 |
| dense_cosine | 1.000 | 0.000 | 0.000 | 0.000 | 0.064 | 0.0000 | 0 |
| lattice_pq_local | 1.000 | 0.000 | 0.034 | 0.150 | 2.835 | 0.0000 | 0 |
| lattice_pq_validated_cosine | 0.966 | 0.034 | 0.000 | 0.000 | 0.314 | 0.0000 | 0 |
| lattice_pq_redis | 1.000 | 0.000 | 0.034 | 0.150 | 2.947 | 0.0490 | 0 |
| lattice_pq_redis_real | 1.000 | 0.000 | 0.034 | 0.150 | 4.264 | 0.0489 | 0 |
| redisvl_direct | skipped | skipped | skipped | skipped | skipped | skipped | skipped |
| gptcache_direct | 0.335 | 0.665 | 0.000 | 0.000 | 0.001 | 0.0000 | 0 |
| upstash_semantic_cache | skipped | skipped | skipped | skipped | skipped | skipped | skipped |

## Skipped Baselines

| Run | Reason |
|---|---|
| redisvl_direct | RedisVL direct baseline unavailable: ResponseError: unknown command 'FT._LIST', with args beginning with:  |
| upstash_semantic_cache | Upstash credentials were not provided; remote semantic-cache baseline skipped. |

Required claim wording: LatticeMemory can reduce repeated/paraphrased upstream calls on this measured workload while reporting false positives and review behavior.
Unsupported wording: LatticeMemory replaces general-purpose vector databases or guarantees accuracy on arbitrary RAG workloads.
