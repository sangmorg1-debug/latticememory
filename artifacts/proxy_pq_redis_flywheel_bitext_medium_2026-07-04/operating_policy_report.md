# LatticeMemory Operating Policy Report

| Policy | Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Recommendation |
|---|---|---:|---:|---:|---:|---|
| conservative_zero_fp | dense_cosine | 1.0000 | 0.0000 | 0.0000 | 0.0000 | Use when false positives are more expensive than upstream calls. |
| balanced_validated_pq | lattice_pq_redis_validated_cosine | 0.9600 | 0.0400 | 0.0000 | 0.0000 | Target product path: PQ candidate generation with validation before serving. |
| aggressive_raw_pq | lattice_pq_local | 1.0000 | 0.0000 | 0.0316 | 0.1540 | Research/high-risk mode only; raw PQ can over-hit and must carry FP metrics. |

Policy rule: never publish raw PQ hit rate without the paired false-positive and adversarial false-positive rates.
The balanced path is the one to harden into production serving if it preserves savings while reducing false positives.
