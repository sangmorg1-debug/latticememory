# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack

This artifact compares exact string caching, dense semantic caching, and LatticeMemory proxy paths.
It is a support-workload proof pack, not a general RAG or vector-database replacement claim.

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_string | 0.268 | 0.733 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 |
| dense_cosine | 0.800 | 0.200 | 0.000 | 0.000 | 0.064 | 0.0000 | 0 |
| lattice_pq_local | 0.990 | 0.010 | 0.101 | 0.000 | 3.022 | 0.0000 | 1 |
| lattice_pq_redis | 0.990 | 0.010 | 0.101 | 0.000 | 2.848 | 0.0089 | 1 |
| lattice_pq_redis_real | skipped | skipped | skipped | skipped | skipped | skipped | skipped |
| redisvl_direct | skipped | skipped | skipped | skipped | skipped | skipped | skipped |
| gptcache_direct | skipped | skipped | skipped | skipped | skipped | skipped | skipped |
| upstash_semantic_cache | skipped | skipped | skipped | skipped | skipped | skipped | skipped |

## Skipped Baselines

| Run | Reason |
|---|---|
| lattice_pq_redis_real | redis unavailable: ConnectionError: Redis ping failed for redis://localhost:6379/15 |
| redisvl_direct | redisvl is not installed; install RedisVL and rerun for a direct RedisVL semantic-cache baseline. |
| gptcache_direct | gptcache is not installed; dense_cosine is the local GPTCache-shaped baseline for this run. |
| upstash_semantic_cache | Upstash credentials were not provided; remote semantic-cache baseline skipped. |

Required claim wording: LatticeMemory can reduce repeated/paraphrased upstream calls on this measured workload while reporting false positives and review behavior.
Unsupported wording: LatticeMemory replaces general-purpose vector databases or guarantees accuracy on arbitrary RAG workloads.
