# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack

**Date:** 2026-07-03
**Status:** Implemented proof-pack harness, public Bitext run, validated PQ policy, real Redis run, and RedisVL direct baseline under Redis Stack
**Purpose:** Produce one public, evidence-backed proof pack for LatticeMemory as a semantic cache and AI memory infrastructure layer.

---

## Current Implementation

The local harness is implemented in `latticememory/proof_pack.py` and covered by
`tests/test_proxy_pq_redis_flywheel_proof_pack.py`. It builds deterministic support
data, runs exact string and dense semantic-cache baselines, then runs the real
`LatticeLLMProxy` with a PQ-backed `RFSnapSemanticCache`, local cache entries, a
Redis-compatible shared entry store, upstream miss logging, flywheel review queue
export, and reviewed-answer import.

First artifact run:

- Artifact directory: `artifacts/proxy_pq_redis_flywheel_proof_pack_2026-07-03/`
- Dataset: 80 seed rows, 120 calibration rows, 320 evaluation rows, 80 adversarial rows
- Verification: `python -m pytest -q` -> 595 passed, 6 warnings

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_string | 0.2675 | 0.7325 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0 |
| dense_cosine | 0.8000 | 0.2000 | 0.0000 | 0.0000 | 0.0589 | 0.0000 | 0 |
| lattice_pq_local | 0.9800 | 0.0200 | 0.0000 | 0.0000 | 2.7474 | 0.0000 | 1 |
| lattice_pq_redis | 0.9800 | 0.0200 | 0.0000 | 0.0000 | 2.7631 | 0.0094 | 1 |

Evidence-hardening run:

- Artifact directory: `artifacts/proxy_pq_redis_flywheel_external_2026-07-03/`
- Dataset source: external JSONL loaded through `load_support_dataset_jsonl`
- Dataset: 160 seed rows, 240 calibration rows, 640 evaluation rows, 160 adversarial rows
- Real Redis request: `redis://localhost:6379/15`
- Real Redis result on this machine: skipped, Redis server unavailable
- Optional direct baselines: RedisVL and GPTCache skipped because dependencies are not installed; Upstash skipped because credentials were not provided

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Redis MB | Shared cache |
|---|---:|---:|---:|---:|---:|---|
| exact_string | 0.2675 | 0.7325 | 0.0000 | 0.0000 | 0.0000 | n/a |
| dense_cosine | 0.8000 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| lattice_pq_local | 0.9900 | 0.0100 | 0.1010 | 0.0000 | 0.0000 | false |
| lattice_pq_redis | 0.9900 | 0.0100 | 0.1010 | 0.0000 | 0.0089 | true |
| lattice_pq_redis_real | skipped | skipped | skipped | skipped | skipped | skipped |

Public wording should prefer the zero-FP row for conservative claims and mention the
PQ row only with its measured false-positive rate attached. The generated claim
card follows that split: `public_claim_card.md` reports `dense_cosine` as the
safest zero-FP row and `lattice_pq_local` as the highest-hit row.

Third-party support dataset run:

- Artifact directory: `artifacts/proxy_pq_redis_flywheel_bitext_2026-07-03/`
- Source dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Source URL: <https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset>
- Available source rows: 26,872 customer-service Q/A rows
- Source fields used: `flags`, `instruction`, `category`, `intent`, `response`
- Proof-pack split: 80 seed rows, 240 calibration rows, 640 evaluation rows, 160 adversarial rows
- Real Redis result on this machine: verified with `redis/redis-stack-server:latest` via Docker Engine inside WSL at `redis://localhost:6382/0`
- Direct baselines: RedisVL direct ran against Redis Stack; GPTCache direct exact-cache baseline ran; Upstash remained skipped because credentials were not provided
- RedisVL note: Redis Stack index creation required database 0 in this environment; `redis://localhost:6382/15` failed with `Cannot create index on db != 0`

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Redis MB | Shared cache |
|---|---:|---:|---:|---:|---:|---|
| exact_string | 0.3350 | 0.6650 | 0.0000 | 0.0000 | 0.0000 | n/a |
| dense_cosine | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| lattice_pq_local | 1.0000 | 0.0000 | 0.0338 | 0.1500 | 0.0000 | false |
| lattice_pq_validated_cosine | 0.9663 | 0.0338 | 0.0000 | 0.0000 | 0.0000 | false |
| lattice_pq_redis | 1.0000 | 0.0000 | 0.0338 | 0.1500 | 0.0490 | true |
| lattice_pq_redis_real | 1.0000 | 0.0000 | 0.0338 | 0.1500 | 0.0490 | true |
| redisvl_direct | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| gptcache_direct | 0.3350 | 0.6650 | 0.0000 | 0.0000 | 0.0000 | n/a |

This removes the prior caveat that the external-format run was still generated
from local synthetic rows. The remaining caveat is different and more useful:
the third-party Bitext run shows that unvalidated PQ serving can over-hit and
produce measurable adversarial false positives. Cosine validation keeps most of
the PQ hit-rate lift (`0.9663`) while driving both measured false-positive rates
to `0.0000` on this split. Public claims should therefore lead with measured
cache behavior, validation policy, and false-positive reporting, not raw PQ hit
rate.

Operating policy from this artifact:

| Policy | Run | Hit rate | FP rate | Adv FP rate | Use |
|---|---|---:|---:|---:|---|
| conservative_zero_fp | dense_cosine | 1.0000 | 0.0000 | 0.0000 | safest baseline |
| balanced_validated_pq | lattice_pq_validated_cosine | 0.9663 | 0.0000 | 0.0000 | target product path |
| aggressive_raw_pq | lattice_pq_local | 1.0000 | 0.0338 | 0.1500 | research/high-risk only |

Medium Redis Stack validated-PQ run:

- Artifact directory: `artifacts/proxy_pq_redis_flywheel_bitext_medium_2026-07-04/`
- Source dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Proof-pack split: 250 seed rows, 500 calibration rows, 2,000 evaluation rows, 500 adversarial rows
- Redis runtime: `redis/redis-stack-server:latest` via Docker Engine inside WSL at `redis://localhost:6382/0`
- Persistence/shared-cache check: `lattice_pq_redis_validated_cosine` reports `redis_persistence_verified=true` and `multi_proxy_shared_cache_verified=true`
- Larger attempted split note: a 500/1000/5000/1000 run was interrupted after the local and in-memory Redis rows because the raw real-Redis proxy stage stopped making progress; it was not committed as a completed proof artifact.

| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB |
|---|---:|---:|---:|---:|---:|---:|
| exact_string | 0.4152 | 0.5848 | 0.0000 | 0.0000 | 0.001 | 0.0000 |
| dense_cosine | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.074 | 0.0000 |
| lattice_pq_local | 1.0000 | 0.0000 | 0.0316 | 0.1540 | 4.150 | 0.0000 |
| lattice_pq_validated_cosine | 0.9600 | 0.0400 | 0.0000 | 0.0000 | 0.385 | 0.0000 |
| lattice_pq_redis | 1.0000 | 0.0000 | 0.0316 | 0.1540 | 4.083 | 0.0500 |
| lattice_pq_redis_real | 1.0000 | 0.0000 | 0.0316 | 0.1540 | 6.383 | 0.0511 |
| lattice_pq_redis_validated_cosine | 0.9600 | 0.0400 | 0.0000 | 0.0000 | 1.336 | 0.0511 |
| redisvl_direct | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.140 | 0.0000 |
| gptcache_direct | 0.4152 | 0.5848 | 0.0000 | 0.0000 | 0.003 | 0.0000 |

This run validates the product-shaped serving policy: Redis-backed PQ can be
used as a candidate generator when a cosine gate verifies the candidate before
serving. It does not prove superiority over RedisVL; RedisVL direct remains a
strong zero-FP baseline on this deterministic proof-pack encoder. The LatticeMemory
claim is the auditable operating-policy matrix: raw PQ over-hits, validated PQ
trades 4.0% upstream calls for zero measured false positives on this split, and
all rows report latency, Redis memory, and false-positive rates together.

---

## Claim Boundary

This proof pack must support only the claims the current system can honestly prove:

- LatticeMemory can serve exact-repeat cache hits through deterministic E8/PQ-backed cache keys.
- A PQ-backed semantic cache can catch some real paraphrases without scanning the full corpus.
- Redis-backed storage lets multiple proxy instances share one cache.
- The flywheel can log misses, cluster coverage gaps, export a review queue, and ingest reviewed answers.
- False-positive rate, latency, hit rate, and estimated cost savings are measurable and must be reported together.

This proof pack must not claim:

- LatticeMemory replaces vector databases for general RAG.
- E8 exact lookup solves open-vocabulary paraphrase retrieval by itself.
- PQ/Hamming matches are safe to serve without calibration and false-positive measurement.
- The system has zero recall loss versus dense retrieval.

---

## System Under Test

Primary system:

```text
OpenAI-compatible request
  -> LatticeLLMProxy
  -> PQ-backed RFSnapSemanticCache
  -> Redis-backed shared cache store
  -> upstream LLM only on miss or rejected approximate hit
  -> flywheel miss log and review queue
```

Required LatticeMemory components:

- `latticememory.proxy.LatticeLLMProxy`
- `latticememory.index.LatticeIndex(mode="pq")`
- `latticememory.semantic_cache.RFSnapSemanticCache`
- `latticememory.redis_store.LatticeRedisStore`
- `latticememory.flywheel.LatticeFlywheel`
- `/v1/analytics`
- `/v1/flywheel/*`

The proxy should run in two serving policies:

| Policy | Purpose |
|---|---|
| `exact_only` | Establish deterministic repeat-hit baseline and zero approximate false positives. |
| `pq_validated` | Measure real paraphrase savings with a PQ cache and a strict false-positive gate. |

---

## Baselines

Compare LatticeMemory against locally reproducible baselines. Do not use marketing numbers from competitor docs as benchmark results.

| Baseline | Minimal implementation |
|---|---|
| Exact string cache | Python dict keyed by normalized prompt string. |
| Dense cosine semantic cache | SentenceTransformer embeddings plus brute-force cosine threshold. |
| RedisVL-style semantic cache | Redis-backed vector/cosine cache if Redis vector features are available; otherwise document as skipped. |
| GPTCache-style semantic cache | Local embedding + similarity evaluator; if GPTCache is not installed, reproduce the same shape with the dense cosine baseline and mark GPTCache direct run as skipped. |
| Upstash-style semantic cache | API-compatible comparison only if credentials are provided; otherwise document as skipped. |

The public report must separate:

- **directly run baselines**
- **skipped baselines**
- **qualitative competitor context**

---

## Dataset

Use a customer-support style workload because it matches the product shape: repeated and paraphrased LLM calls, not open-domain RAG.

Required splits:

| Split | Rows | Purpose |
|---|---:|---|
| cache_seed | 200 canonical Q&A prompts | Preload known answers into cache. |
| calibration | 200 labeled prompt pairs | Fit PQ and select serving threshold/policy. |
| evaluation | 1,000 request stream rows | Measure hit rate, savings, latency, and false positives. |
| adversarial | 200 same-template/different-answer rows | Measure unsafe approximate-hit risk. |

Each row should include:

```json
{
  "id": "eval-0001",
  "intent_id": "billing_refund_window",
  "prompt": "Can I still get a refund after 20 days?",
  "canonical_answer": "Refunds are available within 30 days of purchase.",
  "expected_cache_id": "seed-0042",
  "is_repeat": false,
  "is_paraphrase": true,
  "is_adversarial": false
}
```

Rules:

- Calibration rows must not appear in evaluation.
- Adversarial rows must share surface templates with valid rows but require different answers.
- Report exact-repeat and paraphrase rows separately.
- Keep all generated artifacts under `artifacts/proof_pack_proxy_pq_redis_flywheel/`.

---

## Metrics

Every run must report:

| Metric | Definition |
|---|---|
| total_requests | Number of evaluation requests sent through the proxy/cache. |
| exact_hit_rate | Fraction served from exact key/string repeat path. |
| approximate_hit_rate | Fraction served from PQ/Hamming/cosine approximate path. |
| upstream_call_rate | Fraction that called the upstream LLM. |
| false_positive_rate | Fraction of served cache hits whose answer does not match the row label. |
| false_negative_rate | Fraction of expected cacheable rows that missed. |
| p50_latency_ms | Request latency median. |
| p95_latency_ms | Request latency p95. |
| estimated_cost_saved_usd | Upstream calls avoided times configured cost model. |
| redis_memory_mb | Redis memory used by the cache namespace. |
| cache_entries | Number of cache entries stored. |
| flywheel_miss_clusters | Number of clusters produced from misses. |
| reviewed_answers_loaded | Number of reviewed flywheel rows ingested back into cache. |

Public claims may cite hit rate or savings only when false-positive rate is shown beside them.

---

## Benchmark Matrix

Run this matrix:

| Run | Cache backend | Match policy | Redis | Flywheel | Purpose |
|---|---|---|---|---|---|
| A | exact string dict | exact only | no | no | naive baseline |
| B | dense cosine | threshold calibrated | no | no | traditional semantic cache baseline |
| C | LatticeMemory E8/PQ | exact only | no | yes | exact cache + miss review |
| D | LatticeMemory E8/PQ | PQ validated | no | yes | local LatticeMemory semantic cache |
| E | LatticeMemory E8/PQ | PQ validated | yes | yes | production-shaped proof |
| F | RedisVL/GPTCache/Upstash direct | best local config | yes/remote | no | competitor context when available |

Acceptance gate for a public proof pack:

- Run E completes without unhandled errors.
- Run E reports false-positive rate on both normal and adversarial rows.
- Run E beats exact string dict on paraphrase hit rate.
- Run E reports lower upstream call rate than exact string dict.
- Run E writes a flywheel review queue and then successfully ingests at least one reviewed answer.
- No report text claims general RAG/vector-DB replacement.

---

## Required Artifacts

Write:

```text
artifacts/proof_pack_proxy_pq_redis_flywheel/
  dataset_seed.jsonl
  dataset_calibration.jsonl
  dataset_evaluation.jsonl
  dataset_adversarial.jsonl
  run_exact_string.json
  run_dense_cosine.json
  run_lattice_exact_local.json
  run_lattice_pq_local.json
  run_lattice_pq_redis.json
  flywheel_misses.jsonl
  flywheel_review_queue.json
  flywheel_review_import_result.json
  proof_pack_summary.json
  proof_pack_report.md
```

`proof_pack_summary.json` should contain a flat row per run so future docs can be generated mechanically:

```json
{
  "run_id": "lattice_pq_redis",
  "total_requests": 1000,
  "exact_hit_rate": 0.0,
  "approximate_hit_rate": 0.0,
  "upstream_call_rate": 1.0,
  "false_positive_rate": 0.0,
  "adversarial_false_positive_rate": 0.0,
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "estimated_cost_saved_usd": 0.0,
  "redis_memory_mb": 0.0,
  "cache_entries": 0,
  "flywheel_miss_clusters": 0,
  "reviewed_answers_loaded": 0,
  "status": "not_run"
}
```

The zeros above are schema examples, not expected results.

---

## Implementation Order

1. Add dataset builder for support-style repeat/paraphrase/adversarial streams.
2. Add exact string and dense cosine local baselines.
3. Add LatticeMemory proxy runner with a fake upstream client so benchmark cost is deterministic.
4. Add Redis-backed run path using `LatticeRedisStore`; skip cleanly if Redis is unavailable.
5. Add flywheel review export/import check.
6. Generate `proof_pack_summary.json` and `proof_pack_report.md`.
7. Update public docs only from generated summary numbers.

---

## Public Wording Template

Allowed:

> LatticeMemory is an auditable semantic cache and AI memory layer. In the proxy proof pack, it measures exact and approximate cache hit rates, false positives, latency, Redis-backed shared storage, and flywheel review loops on the same labeled request stream.

Allowed if the numbers support it:

> On this customer-support workload, the PQ-backed Redis proxy reduced upstream calls by X% at Y% false-positive rate, while logging misses into a review queue that can be promoted back into the cache.

Not allowed:

> LatticeMemory replaces vector databases.

Not allowed:

> E8 gives every concept a permanent address.

Not allowed:

> LatticeMemory gives zero-recall-loss semantic search.
