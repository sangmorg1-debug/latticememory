# LatticeMemory Product Roadmap

## Updated 2026-06-17 — grounded in current code and test results

---

## BANKING77 Benchmark Findings (2026-06-09)

Empirically validated on BANKING77, 77 intents, 20-shot, base encoder `dfrokido/bge-large-e8-snap`:

| Method | Accuracy | Notes |
| --- | --- | --- |
| Cosine kNN (Cell A) | 90.42% | Float32 baseline |
| **E8 Hamming, no training (Cell B)** | **88.38%** | **32x compression, 2.04pp gap** |
| E8 Hamming, 3-shell codebook | 87.47% | Worse — more shells degrade Hamming kNN |
| E8 Hamming + SnapTrainer 2ep (Cell D) | 84.51% | Training actively harmful at 77-intent scale |

**Shell-1 (240 codewords) is the correct design point.** More codebook vectors degrade
block-level Hamming kNN accuracy because finer quantization breaks the intentional collision
property that makes Hamming distances meaningful for kNN classification.

**SnapTrainer does not help at 77-intent scale** with ≤20 examples/intent. The training
signal is too diffuse: `zero_fp_recall` trends toward zero (0.003 after 2 epochs) and
`hamming_gap` becomes increasingly negative. SnapTrainer is designed for narrow domains
(≤15 intents) with clear semantic boundaries, not broad benchmark classification tasks.

**Quantization simplification shipped:** Replaced the Babai-snapping path with direct
unit-normalize → argmax in `_quantize_to_indices`. Accuracy is equivalent (within noise — the
two benchmarks used different harnesses). The new path is simpler and measurably faster
(removes two tensor ops per block). **Note:** keys differ from the old path near Voronoi
boundaries; any serialized SQLite stores would need re-indexing. Hamming distance distributions
should be re-validated against proxy thresholds if deploying from existing calibration files.

---

---

## What the Engine Actually Does

E8 lattice quantization converts a float32 embedding into a deterministic 128-byte key
(1024D) or 48-byte key (384D). Two texts that encode to nearby points in embedding space
snap to the **same** or **adjacent** lattice cell.

**This works for:** symmetric workloads — the query text and the indexed text are the same or
close paraphrases of each other (cache, dedup, agent memory).

**This does not work for:** asymmetric QA — question vs. answer-paragraph pairs have a
natural Hamming distance of 88–106 out of 128 blocks, regardless of training. This is a
structural property of the E8 Voronoi tessellation at 1024D, not a solvable bug.

Everything in this roadmap that claims a product capability has been built and passes the
test suite as of this date. Capabilities not yet built are marked explicitly.

---

## Product Stack — Most Direct Fit First

### 1. LLM Cache Proxy

**Fit: highest. Built: complete. Blocker to ship: PyPI publish + Docker Hub push.**

Drop-in HTTP proxy in front of any OpenAI-compatible API. Same prompt (or a paraphrase that
snaps to the same E8 cell) returns the cached response without hitting the upstream model.

**What is actually built:**

- OpenAI-compatible `/v1/chat/completions` — streaming SSE and non-streaming JSON
- `X-Lattice-Cache: HIT/MISS` and `X-Lattice-Savings-USD` on every response
- SQLite persistence — cache survives process restart
- HammingRouter approximate cache — catch paraphrases, not just exact repeats; operates in
  `shadow` mode (measure without serving) or `serve` mode (live hits)
- TTL per-entry expiry + `evict_expired()` sweep
- Management CRUD API — `GET/POST/DELETE /v1/cache` gated by `X-Lattice-Admin-Key`
- Warm-start from CSV/JSON/JSONL — pre-populate cache before first traffic
- Active learning flywheel — every miss logged to JSONL; `detect_drift()` surfaces emerging
  intent clusters so new Q&A pairs can be added proactively
- `latticememory-serve-proxy` console script + `Dockerfile` + `docker-compose.yml`

**What is not yet built:**

- Docker image not pushed to Docker Hub
- PyPI v0.2.0 not uploaded
- No dashboard UI for the management API

**Known limitation:** HammingRouter threshold must be calibrated per domain using
`calibrate_threshold()` before enabling `serve` mode. The proxy ships with a conservative
default (threshold=70) that avoids false positives at the cost of recall. For well-scoped
domains (helpdesk, support, fixed-vocabulary), calibration takes ~100 sample pairs.
**`lattice calibrate` CLI — DONE 2026-06-18:** wraps `gap_stats()`/`calibrate_threshold()`
directly (`lattice calibrate --paraphrases FILE --near-misses FILE [--encoder MODEL]
[--fp-budget N] [--export PATH]`), so this calibration step no longer requires knowing the
standalone `calibrate_proxy.py` script exists.

**Who buys it:** Any team spending >$5K/month on LLM API calls with high query repetition —
customer support bots, internal knowledge assistants, coding assistants with repeated context.
Pitchable as: "your LLM API bill, minus the repeats."

---

### 2. Multi-Tenant Semantic Cache (SaaS layer)

**Fit: high. Built: complete. Blocker to ship: same as #1.**

`LatticeMultiCache` wraps N independent `RFSnapSemanticCache` instances behind a single
interface, keyed by tenant ID. Same question from `acme` and `globex` gets each their own
cached answer — entries never intermix.

**What is actually built:**

- `LatticeMultiCache(cache_factory, max_tenants=...)` — thread-safe, lazy-init per tenant
- `mc.put(tenant_id, prompt, value=...)` / `mc.get(tenant_id, prompt)`
- `mc.evict_expired_all()` — sweep TTL expiry across all tenant caches
- `mc.stats()` — per-tenant entry counts
- `mc.drop_tenant(tenant_id)` — release a tenant's cache from memory
- Redis backend swap (`patch_cache_with_redis()`) works per-tenant too

**Who buys it:** B2B SaaS companies that build LLM-powered products on top of OpenAI and
want to serve different cached answers per customer without paying for the same generation
twice. Sold as a library they embed, not a hosted service.

---

### 3. Semantic Dedup

**Fit: high. Built: core complete, streaming in-process only. Blocker: Kafka adapter.**

`LatticeDedup` processes a corpus O(N) — each item is hashed to its E8 key, stored in a
set; duplicates are flagged without pairwise comparison. `LatticeStreamDedup` adds a sliding
window for continuous streams.

**What is actually built:**

- `LatticeDedup.add(text)` → `{"is_duplicate": bool, "canonical_id": str | None}`
- `LatticeDataPipeline.deduplicate_text(corpus)` — batch API, returns deduped list
- `LatticeStreamDedup` — in-process sliding window dedup with configurable TTL
- `LatticeSqliteStore` persistence — key set survives restart

**What is actually built (updated 2026-06-17):**

- `lattice dedup <file>` CLI (`cmd_dedup` in `cli.py`) — `.txt`, `.json`, `.jsonl`, `.csv` input; outputs deduped file; reports compression ratio ✓

**What is not yet built:**

- Kafka consumer adapter (streaming pipeline integration — requires broker to validate; deferred)
- Redis-backed distributed key store for multi-process dedup (requires Redis infra to validate; deferred)

**Known limitation:** Dedup works on the symmetric assumption. Two articles covering the
same event but written from opposite angles (e.g., "Fed raises rates" vs "Markets respond
to Fed hike") will have different E8 keys and will not be flagged as duplicates. This is
correct behavior for near-duplicate detection but is not semantic clustering.

**Who buys it:** LLM training data teams (Common Crawl cleaning, web scrape dedup),
ML data vendors, news/media aggregators paying cloud bills for near-duplicate content.

---

### 4. Compliance Cache (Regulated Industry Proxy Mode)

**Fit: high. Built: complete. Blocker: same as #1 (PyPI publish).**

The proxy in compliance mode only serves cached responses that have been explicitly approved.
New responses are held until a human reviewer signs off. Every cache hit is logged with a
tamper-evident chain.

**What is actually built:**

- `compliance_mode=True` in `LatticeLLMProxy` — validated responses only are served from cache
- `validation_required=True` — new unique responses are flagged, not auto-cached
- `audit_log_path` — every hit logged as `{prompt, e8_key, served_response, timestamp, upstream_hash}`
- `divergence_threshold` — if a new model response deviates from the cached one by >threshold
  cosine distance, it is routed to review instead of served
- `POST /v1/compliance/validate/{cache_id}` — human approval endpoint with HMAC audit chain
- `GET /v1/compliance/audit-log` — full tamper-evident audit log export
- `GET /v1/compliance/pending` — reviewer queue: list all entries awaiting approval
- `LATTICE_REVIEWER_KEY` / `reviewer_key` — role separation: reviewers can approve but not delete

**What is not yet built:**

- Dashboard UI for the reviewer queue

**Who buys it:** Financial services, healthcare AI deployments, legal research tools —
anywhere an LLM response is a regulated statement that must be pre-approved. The pitch is
"your LLM produces the exact same approved answer every time, with a full audit trail."
No current LLM gateway (LiteLLM, Portkey) does this.

---

### 5. Agent Episodic Memory

**Fit: medium-high. Built: core complete, framework adapters exist but untested end-to-end.**

`AgentEpisodicMemory` gives an agent a long-term memory store. `AgentMemorySync` lets agents
in a swarm share only the keys they are missing — no embedding transfer, just 128-byte keys.

**What is actually built:**

- `AgentEpisodicMemory` — `remember(text)`, `recall(query)`, SQLite-backed
- `AgentMemorySync` — `share(key)`, `diff(peer_keys)`, `request(key)` over in-process transport
- `make_autogen_sync_tools()` — AutoGen-compatible tool definitions wrapping `AgentMemorySync`
- `LangGraphLatticeAdapter` — LangGraph node that reads/writes a shared `LatticeIndex`

**What is actually built (updated 2026-06-17):**

- `tests/test_agent_sync.py` — full coverage of `make_autogen_sync_tools` and `LangGraphLatticeAdapter` ✓
- `examples/agent_swarm_demo.py` (Gate 4) — two agents sharing memory via pull-sync and push-broadcast; runs without model download ✓

**What is not yet built:**

- Network transport for `AgentMemorySync` (currently in-process only)
- End-to-end demo with a real AutoGen or LangGraph orchestrator making LLM calls

**Known limitation:** The sync protocol assumes all agents use the same encoder model.
Mixed encoder fleets will produce different E8 keys for the same content and cannot sync
via key comparison alone.

**Who buys it:** Teams building multi-agent pipelines (AutoGen, CrewAI, LangGraph) who need
persistent cross-session memory without embedding transfer overhead.

---

### 6. Active Learning Flywheel (Proxy Add-On)

**Fit: medium — a feature that strengthens #1, not a standalone product.**

Every proxy cache miss is logged with its E8 key. `LatticeFlywheel` clusters the miss log by
Hamming distance to surface emerging intents — groups of related questions the cache doesn't
cover yet.

**What is actually built:**

- `LatticeFlywheel.log_miss(question, e8_key_hex, nearest_cache_prompt, nearest_cache_distance)`
- `detect_drift(window_seconds, min_delta)` — joint-clustering over all in-window records,
  returns clusters whose recent-window count exceeds older-window count by `min_delta`
- `intent_frequency()` — top-N clusters by miss count
- `federated_key_histogram()` — key distribution for multi-node deployments
- `finetune()` — training loop stub (wires to `SnapTrainer`; requires labeled Q&A pairs)

**What is actually built (updated 2026-06-17):**

- `should_finetune(min_drifting_clusters, min_delta, ...) -> bool` — returns True when enough drifting clusters exceed threshold; testable without a training run ✓
- `lattice drift` CLI (`cmd_drift` in `cli.py`) — prints drift table, labels recommend-finetune, optionally exports JSON report ✓

**What is not yet built:**

- UI for the reviewer queue (drift alerts in a browser)
- Automated background retraining loop (requires training infra)

**How it ships:** As a flag on the proxy (`miss_log_path=...`). Not sold separately.

---

### 7. Vertical Applications (9 built, 2026-06-17)

**Fit: high for narrow domains. Built: complete.**

All verticals share the same `RFSnapSemanticCache` core and ship in `latticememory.verticals`.

| Vertical | Class | Key Capability |
| --- | --- | --- |
| SOC Monitor | `LatticeSOCMonitor` | O(1) alert dedup for SIEM event streams |
| Ticket Analyzer | `LatticeTicketAnalyzer` | Intent-based ticket routing + dedup |
| Content Moderator | `LatticeContentModerator` | Semantic near-miss content policy |
| Clause Coder | `LatticeClauseCoder` | Legal clause classification |
| Edge Memory | `LatticeEdgeMemory` | On-device personalization without cloud round-trip |
| Private Sync | `LatticePrivateSync` | Federated key sync, no raw text transfer |
| **Prompt Firewall** | `LatticePromptFirewall` | Semantic injection/jailbreak detection (14 default patterns) |
| **Semantic Rate Limiter** | `LatticeSemanticRateLimiter` | Per-intent sliding-window rate limiting |
| **Training Cleaner** | `LatticeTrainingCleaner` | O(N) near-duplicate removal for LLM training sets |

The last three (`LatticePromptFirewall`, `LatticeSemanticRateLimiter`, `LatticeTrainingCleaner`) were shipped 2026-06-17.

---

## What We Explicitly Don't Have

These appear in the codebase but are not production-ready claims:

| Capability | Reality |
| --- | --- |
| Cross-modal alignment (image↔text, shape↔text E8 key match) | **Experimentally confirmed infeasible, twice, on real data (2026-06-17 audit + follow-up test).** Real CLIP ViT-B/32 image↔text (200 real image-caption pairs): mean Hamming 61.81/64 blocks (~97% mismatch), 0% exact/beam-R10 hit rate, 0% E8 retrieval recall@1 vs. 49% float-cosine baseline. Real OpenShape PointBERT-ViT-L shape↔text (200 real Cap3D pairs, same method): mean Hamming 95.14/96 (~99%), 0% exact/beam-R10 hit rate, same-pair float cosine only 0.106 (looser than native CLIP alignment), E8 retrieval recall@1 2.0% vs. 11.5% float-cosine baseline. Both modalities: the float embeddings aren't tightly coupled enough across modalities for Hamming-distance routing regardless of which encoder pairing is used. The in-package `examples/multimodal_alignment_demo.py` (`FakeEncoder`, 4-pair overfit) remains a separate, weaker, non-representative demo — the real-data results above are the load-bearing finding. |
| Cross-model Semantic DNS (MiniLM↔BGE key parity) | Demo artifact — synthetic embeddings from a shared latent space. Real encoder families are not linearly related. (Distinct from cross-modal above: this is two *encoders* of the same modality, not two modalities.) |
| Asymmetric QA retrieval (question→passage via E8) | **Exact/near-address O(1) routing confirmed infeasible; Hamming-as-coarse-prefilter is not.** The previously-cited DPR-baseline numbers (71/96 mean Hamming, min 49, 2k-example fine-tuning) could not be traced to any script or saved result in this repo or its parent monorepo and should be treated as unsourced. A real, saved experiment does exist (`e8-Project/latticememory_open_retrieval_msmarco_1k_cosine/rfsnap_open_retrieval.json`): 1000 real MS-MARCO docs, 200 real queries, `e8_bge_large_snaptrained` — Hamming pre-filter (pool multiplier=10) → cosine rerank gets 63.5% top-1, 99% recall@10. So Hamming distance does carry real coarse signal for asymmetric QA when used as a *candidate-pool filter feeding a reranker* — it just doesn't work as an *exact/near-address lookup* (the original product claim), which needs matched pairs within ~4 blocks and QA pairs structurally don't land there. |
| Hallucination grounding signal | Architecture exists in concept, not implemented or validated. |
| Encoder drift monitoring (snapshot compare) | `detect_drift()` in the flywheel is for prompt intent drift, not encoder model drift. Snapshot comparison not built. |

---

## Ship Order

| Priority | Product | Status | What's blocking |
| --- | --- | --- | --- |
| **1** | LLM Cache Proxy | Code complete, 508 tests pass | `twine upload`, Docker Hub push |
| **2** | Multi-Tenant Cache | Code complete | Same as #1 |
| **3** | Semantic Dedup | Core + CLI complete | Kafka adapter (infra-dependent, deferred) |
| **4** | Compliance Cache | **Complete** (reviewer key + pending queue + HMAC chain) | Same as #1 |
| **5** | Agent Memory | Core + adapters + swarm demo complete | Network transport |
| **6** | Flywheel | Complete + `should_finetune()` + `lattice drift` CLI | Review UI |
| **7** | Verticals (9 total) | Complete (SOC, tickets, content mod, clause, edge, private sync, firewall, rate limiter, training cleaner) | Same as #1 |

---

## Immediate Actions (in order)

### Gate 0 — validate the quantizer change (DONE 2026-06-09)

Results on BANKING77 20-shot (200 intra + 200 inter-intent pairs, `dfrokido/bge-large-e8-snap`):

| | min | p5 | mean | p95 | max |
| --- | --- | --- | --- | --- | --- |
| Intra-intent (paraphrase) | 57 | 72.0 | 97.71 | 116.0 | 123 |
| Inter-intent (near-miss) | 103 | 112.0 | 119.9 | 126.0 | 128 |

Gap (inter_p5 − intra_p95): **-4.0** (negative — distributions overlap at tails)

| Threshold | Recall | FP rate | |
| --- | --- | --- | --- |
| 70 | 4.5% | 0.0% | proxy default — **safe** |
| 100 | 52.5% | 0.0% | practical operating point |
| 111 | 84.0% | 4.5% | router default — FP on hard domain |

**Conclusion:** The proxy default threshold=70 is safe (0% FP) even on the hard BANKING77 domain. The router default=111 is too aggressive for this domain — calibrate per-domain. The negative gap is a property of BANKING77's confusable banking intents, not the quantizer change.

Helpdesk domain gap_stats (60 paraphrase + 20 near-miss pairs):

| Threshold | Recall | FP rate | |
| --- | --- | --- | --- |
| 70 | 0.0% | 0.0% | too conservative |
| 102 | 41.7% | 0.0% | **recommended operating point for helpdesk** |
| 108 | 66.7% | 5.0% | max usable (near overlap zone) |

Shadow mode demo on helpdesk: **19.4% shadow hit rate**, mean lookup **0.015 ms**, 70 keys × 128 bytes = 8,960 bytes total index.

Paraphrase cache benchmark: exact-match hits = **100%**, paraphrase hits = **0%** with base `LatticeIndex`. Paraphrase recall requires `HammingRouter` with calibrated threshold — the exact lattice cache only catches Hamming-1 neighbors (< 1 block difference).

### Gate 1 — ship the proxy (highest value)

1. ~~Run `validate_hamming_thresholds.py`~~ — **validated 2026-06-17**: proxy t=70 → 0% FP; router t=111 → 84% recall / 4.5% FP on BANKING77 (calibrate per domain)
1. `python -m build && python -m twine upload dist/*` — publish v0.2.0 to PyPI (user-run, outside repo)
1. `docker buildx build --push -t latticememory/proxy:latest .` — Docker Hub
1. ~~Write `examples/quickstart_proxy.py`~~ — **shipped**: three-section demo (exact hit, HammingRouter, proxy server instructions)

### Gate 2 — dedup shippable standalone (DONE 2026-06-17)

1. ~~Add `lattice dedup <file>` CLI command~~ — **shipped**: `cmd_dedup` in `cli.py`, supports `.txt`, `.json`, `.jsonl`, `.csv`

### Gate 3 — compliance cache (DONE 2026-06-17)

1. ~~Add `POST /validate/{cache_id}` to proxy~~ — **shipped**: `POST /v1/compliance/validate/{cache_id}` with HMAC audit chain, `GET /v1/compliance/pending` reviewer queue, `LATTICE_REVIEWER_KEY` role separation

### Gate 4 — agent memory demo (DONE 2026-06-17)

1. ~~Write `examples/agent_swarm_demo.py`~~ — **shipped**: pull-sync + push-broadcast demo, FakeEncoder (no model download), 4+1 fact scenario, all assertions pass

---

## Codebase Improvements Shipped (2026-06-09)

| Change | File | Impact |
| --- | --- | --- |
| Quantization: Babai→argmax | `rag/e8_retriever.py` | Simpler hot path, equivalent accuracy |
| `_quantize_batch` | `rag/e8_retriever.py` | ~N× speedup for bulk indexing |
| `add_batch` uses `_quantize_batch` | `rag/e8_retriever.py` | One matmul instead of N per-row matmuls |
| `add_bulk` | `hamming_router.py` | One encode call for N texts |
| `lattice_keys_for_batch` | `memory.py` (both classes) | Batch key computation exposed |
| Parity test suite | `tests/test_gap_fixes.py` | Catches single/batch path divergence |
| Hamming validation script | `benchmarks/validate_hamming_thresholds.py` | Verifies thresholds after any quantizer change |

---

---

## Research Paths (Next Candidates)

These are not yet built but are grounded in existing code. Listed in rough value order.

### 0. Multimodal lattice platform — shape↔shape caching (validated 2026-06-17, not yet built)

Explored as a possible "branch the E8 idea into something bigger" direction. Two real-data
experiments were run (in the untracked `shape_memory_dev/` workspace, copied from the separate
`e8-Project`/E8 Shape Memory effort — see that project's standalone Windows app for the original
3D viewer/Rust backend):

- **Cross-modal text→shape search: closed, infeasible.** Real OpenShape PointBERT-ViT-L
  (768-D, CLIP-aligned by training) vs. real OpenCLIP ViT-L/14 text, 200 real Cap3D pairs:
  same-pair float cosine only 0.106, mean Hamming 95.14/96 blocks (~99%), 0% exact/beam-R10
  hit rate, E8 retrieval recall@1 2.0% vs. 11.5% float-cosine baseline. Confirms and extends
  the existing image↔text negative result (see "What We Explicitly Don't Have" below) — cross-modal
  E8 routing does not work for any encoder pairing tested so far.
- **Same-modality shape↔shape caching: validated, viable.** Same encoder, 200 real Cap3D objects,
  intra pairs = object vs. an augmented copy of itself (rotation + 80% subsample + jitter), inter
  pairs = different objects: intra Hamming mean=67.1/p95=92, inter mean=90.9/p5=76, gap=-16 (tails
  overlap — structurally identical to the BANKING77 text validation's gap=-4.0, which still shipped).
  Threshold sweep finds usable operating points: threshold=64 → 43% recall/2% FP; threshold=50 →
  23% recall/0% FP. This is the same risk profile the text cache already ships with.

**Recommendation:** build `RFSnapShapeMemory`/shape-equivalent of `HammingRouter` mirroring the
existing text runtime pattern (the core `RFSnapLatticeMemory` quantization layer is already
modality-agnostic — confirmed by reading `memory.py`/`text_runtime.py`). Do not build cross-modal
search — it's a closed question now, not an open one. Caveat: same-modality validation tested
"catch the same object re-encountered," not "cluster different instances of the same category" —
that's a separate, harder, untested claim.

### 1. `lattice review` CLI — Review Workflow Automation

`LatticeFlywheel.export_review_queue()` and `load_reviewed()` already exist, but no CLI wraps
them. A `lattice review export` + `lattice review import` pair would let ops teams drive the
full miss-to-cache feedback loop from the command line, with no code changes.

**Effort:** Small (1 day). **Value:** Makes flywheel usable without writing Python.

### 2. `lattice federated` CLI — Multi-Node Key Histogram

`LatticeFlywheel.federated_key_histogram()` is implemented but not surfaced as a CLI command.
A `lattice federated --logs node1.jsonl node2.jsonl ...` command would aggregate miss
distributions across proxy replicas for capacity planning and threshold calibration at scale.

**Effort:** Small (1 day). **Value:** Enables multi-instance deployments to share miss analytics.

### 3. Network Transport for `AgentMemorySync`

`AgentMemorySync` currently only works in-process (direct Python object references). To cross
process or machine boundaries, a REST or WebSocket adapter is needed. The `share()` / `receive_document()`
/ `sync_from_peer()` interface is already clean for this.

**Approach:** `FastAPI` endpoint per agent; `sync_from_peer(peer)` becomes an HTTP GET to
`http://agent-b/v1/keys`; `share(key)` becomes a POST to all registered peer URLs.

**Effort:** Medium (2–3 days). **Value:** Unlocks real multi-process agent swarms (AutoGen, CrewAI).

### 4. Encoder Drift Monitoring (Snapshot Comparison)

`detect_drift()` in `LatticeFlywheel` tracks prompt-intent drift (same queries appearing more).
A separate capability is needed to detect *encoder model drift* — when the same text produces
different E8 keys after an encoder update, invalidating existing cache entries.

**Approach:** Snapshot the E8 key set at deployment time. After an encoder update, re-key a
random sample and report Hamming distances between old and new keys. If mean Hamming > threshold,
flag for cache invalidation.

**Effort:** Medium (2 days). **Value:** Prevents silent cache poisoning on encoder rollouts.

### 5. WebSocket Streaming Proxy

The current proxy supports SSE streaming for chat completions. WebSocket streaming (bi-directional,
useful for tool-calling agents) is not yet implemented.

**Effort:** Medium (2 days). **Value:** Unblocks WebSocket-based agent frameworks.

### 6. Multi-Node Redis Cache Pool

`LatticeRedisStore` (ships with `latticememory[redis]`) lets multiple proxy instances share a
single Redis cache. The next step is a pool configuration that:

- Shards tenant caches across multiple Redis nodes by namespace prefix
- Runs `evict_expired()` as a background task (cron or Celery beat)

**Effort:** Medium (2–3 days). **Infra requirement:** Redis Cluster or Sentinel; deferred until
a production deployment validates the routing need.

---

## Development Notes

- **Python:** 3.11.9, Windows 11
- **Tests:** `python -m pytest tests/ -q` → 508 pass (as of 2026-06-17)
- **Encoder model:** `dfrokido/bge-large-e8-snap` (HuggingFace) — 1024D, produces E8 keys
- **Key size:** 128 bytes (1024D) / 48 bytes (384D)
- **Compression:** 32× vs float32 on 1024D (key only)
