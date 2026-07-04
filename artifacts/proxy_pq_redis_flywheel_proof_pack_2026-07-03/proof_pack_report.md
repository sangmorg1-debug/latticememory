# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack

This artifact compares exact string caching, dense semantic caching, and LatticeMemory proxy paths.
It is a support-workload proof pack, not a general RAG or vector-database replacement claim.

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_string | 0.268 | 0.733 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 |
| dense_cosine | 0.800 | 0.200 | 0.000 | 0.000 | 0.059 | 0.0000 | 0 |
| lattice_pq_local | 0.980 | 0.020 | 0.000 | 0.000 | 2.747 | 0.0000 | 1 |
| lattice_pq_redis | 0.980 | 0.020 | 0.000 | 0.000 | 2.763 | 0.0094 | 1 |

Required claim wording: LatticeMemory can reduce repeated/paraphrased upstream calls on this measured workload while reporting false positives and review behavior.
Unsupported wording: LatticeMemory replaces general-purpose vector databases or guarantees accuracy on arbitrary RAG workloads.
