# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack

This artifact compares exact string caching, dense semantic caching, and LatticeMemory proxy paths.
It is a support-workload proof pack, not a general RAG or vector-database replacement claim.

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_string | 0.395 | 0.605 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 |
| dense_cosine | 1.000 | 0.000 | 0.000 | 0.000 | 0.212 | 0.0000 | 0 |
| lattice_pq_local | 1.000 | 0.000 | 0.019 | 0.111 | 3.089 | 0.0000 | 0 |
| lattice_pq_validated_cosine | 0.981 | 0.019 | 0.000 | 0.000 | 0.311 | 0.0000 | 0 |
| lattice_pq_redis | 1.000 | 0.000 | 0.019 | 0.111 | 3.298 | 0.0523 | 0 |
| lattice_pq_redis_real | 1.000 | 0.000 | 0.019 | 0.111 | 4.662 | 0.0534 | 0 |
| lattice_pq_redis_validated_cosine | 0.981 | 0.019 | 0.000 | 0.000 | 1.021 | 0.0534 | 0 |
| redisvl_direct | 1.000 | 0.000 | 0.000 | 0.000 | 0.796 | 0.0000 | 0 |
| gptcache_direct | 0.395 | 0.605 | 0.000 | 0.000 | 0.001 | 0.0000 | 0 |
| upstash_semantic_cache | skipped | skipped | skipped | skipped | skipped | skipped | skipped |

## Skipped Baselines

| Run | Reason |
|---|---|
| upstash_semantic_cache | Upstash credentials were not provided; remote semantic-cache baseline skipped. |

Required claim wording: LatticeMemory can reduce repeated/paraphrased upstream calls on this measured workload while reporting false positives and review behavior.
Unsupported wording: LatticeMemory replaces general-purpose vector databases or guarantees accuracy on arbitrary RAG workloads.
