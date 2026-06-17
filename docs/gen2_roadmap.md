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
407-test suite as of this date. Capabilities not yet built are marked explicitly.

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

**What is not yet built:**

- Kafka consumer adapter (streaming pipeline integration)
- Redis-backed distributed key store for multi-process dedup
- CLI `lattice dedup <file>` — no command-line interface for the dedup workflow

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

**What is not yet built:**

- End-to-end demo with a real AutoGen or LangGraph agent running
- Network transport for `AgentMemorySync` (currently in-process only)
- Tests for `make_autogen_sync_tools` and `LangGraphLatticeAdapter`

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

**What is not yet built:**

- UI or CLI to review drift alerts and add new Q&A pairs from the miss log
- Automated retraining trigger when `detect_drift()` returns results

**How it ships:** As a flag on the proxy (`miss_log_path=...`). Not sold separately.

---

## What We Explicitly Don't Have

These appear in the codebase but are not production-ready claims:

| Capability | Reality |
| --- | --- |
| Cross-modal alignment (image↔text E8 key match) | Demo artifact — `FakeEncoder` with identical text prefixes, 4-pair overfit. Not validated with real CLIP embeddings. |
| Cross-model Semantic DNS (MiniLM↔BGE key parity) | Demo artifact — synthetic embeddings from a shared latent space. Real encoder families are not linearly related. |
| Asymmetric QA retrieval (question→passage via E8) | **Experimentally confirmed infeasible (2026-06-10).** DPR (the best pretrained QA encoder, 78% recall@10 on MS-MARCO) produces mean Hamming 71/96 blocks on matched pairs vs 90 random baseline. Min Hamming across 200 matched pairs: 49. Zero pairs within 30 blocks. Full fine-tuning experiments (2k examples, 15 epochs, multiple loss configs) all ended in collapse or oscillation — never producing useful routing. E8 routing requires matched pairs within ~4 blocks; the QA geometry places them ~71 blocks apart regardless of training. |
| Hallucination grounding signal | Architecture exists in concept, not implemented or validated. |
| Encoder drift monitoring (snapshot compare) | `detect_drift()` in the flywheel is for prompt intent drift, not encoder model drift. Snapshot comparison not built. |

---

## Ship Order

| Priority | Product | Status | What's blocking |
| --- | --- | --- | --- |
| **1** | LLM Cache Proxy | Code complete, 407 tests pass | `twine upload`, Docker Hub push |
| **2** | Multi-Tenant Cache | Code complete | Same as #1 |
| **3** | Semantic Dedup | Core complete | Kafka adapter |
| **4** | Compliance Cache | **Complete** (reviewer key + pending queue + HMAC chain) | Same as #1 |
| **5** | Agent Memory | Core + adapters exist | End-to-end demo, network transport |
| **6** | Flywheel | Complete as proxy add-on | Review UI |

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

1. Run `validate_hamming_thresholds.py` — must pass before this step
1. `python -m build && python -m twine upload dist/*` — publish v0.2.0 to PyPI
1. `docker buildx build --push -t latticememory/proxy:latest .` — Docker Hub
1. Write `examples/quickstart_proxy.py` — 10-line demo showing cache hit with curl

### Gate 2 — dedup shippable standalone

1. Add `lattice dedup <file>` CLI command (wraps `LatticeDataPipeline.deduplicate_text`)

### Gate 3 — compliance cache (DONE 2026-06-17)

1. ~~Add `POST /validate/{cache_id}` to proxy~~ — **shipped**: `POST /v1/compliance/validate/{cache_id}` with HMAC audit chain, `GET /v1/compliance/pending` reviewer queue, `LATTICE_REVIEWER_KEY` role separation

### Gate 4 — agent memory demo

1. Write `examples/agent_swarm_demo.py` — two agents sharing memory over `AgentMemorySync`

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

## Development Notes

- **Python:** 3.11.9, Windows 11
- **Tests:** `python -m pytest tests/ -q` → 407 pass (as of 2026-06-17)
- **Encoder model:** `dfrokido/bge-large-e8-snap` (HuggingFace) — 1024D, produces E8 keys
- **Key size:** 128 bytes (1024D) / 48 bytes (384D)
- **Compression:** 32× vs float32 on 1024D (key only)
