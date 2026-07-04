# LatticeMemory

**Semantic cache, dedup, and hybrid memory — 32× compressed E8 keys for instant repeat-query hits, dense fallback for novel retrieval.**

LatticeMemory uses the [E8 lattice](https://en.wikipedia.org/wiki/E8_lattice) — the densest sphere packing in 8 dimensions — as a deterministic address space for text embeddings. Every 1024-dim embedding snaps to a 128-byte E8 key. Identical or near-identical text lands on the same key; novel queries fall through to a dense float32/Int8 fallback.

[**Live Demo →**](https://huggingface.co/spaces/dfrokido/LatticeMemory) | [**Model →**](https://huggingface.co/dfrokido/bge-large-e8-snap) | [**GitHub →**](https://github.com/sangmorg1-debug/latticememory)

---

## What it's for

| Workload | Recommended mode | Real-model result | Fallback needed? |
| --- | --- | --- | --- |
| Exact-repeat LLM queries (identical text) | `mode="cache"` | ✅ O(1) exact hit, reliable | No |
| **Open-vocabulary paraphrase caching** (general chat, varied phrasing) | `mode="pq"` | ✅ **31% Recall@1** on real PAWS paraphrases, confirmed at ~7x corpus scale ([details](docs/manual-results/2026-06-24-open-vocab-semantic-addressing-redesign.md)) | No (PQ's own candidate pool + rerank is the answer) |
| Semantic deduplication, near-duplicate detection | `mode="cache"` | ✅ Key collision = duplicate | No |
| Dataset quality filtering, semantic sharding | `mode="cache"` | ✅ Stable cluster addresses | No |
| **Asymmetric QA/passage search (RAG)** | `mode="hybrid"` | ❌ 0.0% (E8) / 16.4% best case (PQ, in-domain) - query ≠ passage in embedding space either way | **Yes — Int8 or float32 required** |

**The default E8 lattice mechanism (`mode="cache"`/`mode="hybrid"`'s E8 path) has no demonstrated real-data success outside of literal exact-text repeats** - not on open-vocabulary text, and not even on a genuine closed-vocabulary held-out test (verified 2026-06-24: real encoder, real CLINC150 intents, zero train/test text overlap, scored **0.00%** - a complete miss on all 480 queries; an earlier claim of "100% on a 16-command domain" turned out to be measuring a different mechanism entirely, `qa_bot.py`'s separate centroid-classification mode, tested via re-submitting verbatim training strings rather than real held-out paraphrases - see `docs/honest_product_review.md` gap #6 for the full retraction).

**`mode="pq"`** (Product Quantization, added 2026-06-24) is the actual fix: data-calibrated learned codebooks over 8 coarser blocks instead of E8's fixed 128-block mathematical lattice. Real-model validation against PAWS: **31% Recall@1** (vs. E8's 0.40%), confirmed to hold up at ~7x larger corpus scale, with candidate pools small enough (≈3-7 documents) to keep the actual point of this library - a cheap address-based lookup instead of full vector search - intact. PQ's codebooks need real data to calibrate (unlike E8's fixed geometry): call `index.fit_pq(sample_texts)` with a representative sample (1,000+ texts from your domain, ideally) before `add()`, or just call `add()` directly and the first batch will auto-fit the codebooks (works, but a dedicated calibration sample generalizes better - see the zero-shot vs. in-domain comparison in the linked results doc).

Asymmetric QA/RAG (query text structurally different from passage text) remains genuinely hard for either mechanism - PQ improves it from 0.0% to a best case of 16.4% in-domain, which the project's own findings call "too low for production search." Use `mode="hybrid"` and treat the Int8 dense fallback as the real answer for that workload, not PQ or E8.

**Current proof-pack:** The production-shaped semantic-cache proof path is now implemented as a local harness: LatticeMemory proxy + PQ cache + Redis-compatible shared store + flywheel review, measured against exact string and dense semantic-cache baselines with hit rate, latency, savings, Redis memory, review behavior, and false-positive rate reported together. See [`docs/proxy_pq_redis_flywheel_proof_pack_2026-07-03.md`](docs/proxy_pq_redis_flywheel_proof_pack_2026-07-03.md). The public claim remains scoped: this is evidence for repeated/paraphrased support-workload caching, not a general vector database or RAG replacement claim.

---

## Benchmarks

**Compression (bge-large 1024-dim):**

| Method | Compression | Index / 1M docs | Retrieval p50 @ 100K docs |
| --- | ---: | ---: | ---: |
| Float32 | 1× | 4.1 GB | 20.8 ms |
| **LatticeMemory E8 keys** | **32×** | **0.13 GB** | O(1) on key hit |

**Fallback quality (1K docs, 100 paraphrase queries, recall vs float32):**

| Fallback | Compression vs float32 | Recall@10 overlap | Top-1 agreement | Search p50 |
| --- | ---: | ---: | ---: | ---: |
| Float32 | 1× | 100.0% | 100.0% | 0.14 ms |
| Int8 | 4× | 95.1% | 91.0% | 1.97 ms |
| Int4 | 8× | 12.1% | 1.0% | 4.21 ms |

- **Int8 fallback** is the recommended fallback for RAG/QA — 4× smaller than float32, 95% recall parity.
- **STS quality:** `bge-large-e8-snap` scores 0.8714 vs 0.8637 float baseline (+0.0077).

> **Compression basis:** 1 address byte per 8-dim block × 128 blocks = 128 bytes for 1024-dim vs 4,096 bytes float32 = 32×. This applies to E8 key storage only; hybrid mode also stores the dense index.

---

## Install

```bash
pip install lattice-memory-e8
```

The PyPI distribution is named `lattice-memory-e8` (the plain `latticememory` name
collides with an unrelated existing package on PyPI) — the import name is unaffected:
`import latticememory` works exactly as shown throughout this README.

Optional extras:

```bash
pip install 'lattice-memory-e8[proxy]'   # FastAPI proxy server (fastapi, uvicorn, httpx)
pip install 'lattice-memory-e8[redis]'   # Redis backend for multi-instance caches
pip install 'lattice-memory-e8[hf]'      # HuggingFace datasets integration
pip install 'lattice-memory-e8[faiss]'   # FAISS vector fallback
```

---

## Quickstart

### Semantic cache (the primary use case)

```python
from latticememory import LatticeIndex

# mode="pq" for open-ended phrasing (general chat caching) - this is the
# mechanism that actually delivers real paraphrase hits (31% Recall@1 on
# real PAWS validation, with the production default pq_codebook_size=256).
# Codebook fitting needs at least pq_codebook_size training vectors - this
# demo uses a tiny codebook (pq_codebook_size=8) so it runs with a handful
# of example texts; production use should keep the default 256 and fit on
# 1,000+ representative texts from your actual domain for real quality.
index = LatticeIndex(mode="pq", pq_codebook_size=8)  # downloads dfrokido/bge-large-e8-snap on first run (~500MB)
index.fit_pq([
    "What is the refund policy?", "How do I reset my password?", "Where is my order?",
    "Can I cancel my subscription?", "How long does shipping take?",
    "What is the refund policy.", "How do I change my password?", "When will my order arrive?",
])

index.add([
    "What is the refund policy?",
    "How do I reset my password?",
    "Where is my order?",
])

# Exact text → guaranteed O(1)-ish lattice_exact hit
result = index.search("What is the refund policy?", top_k=1)
print(result[0].retrieval_path)  # lattice_exact

# Real paraphrase → PQ's learned codebooks give this a real shot at a lattice
# hit (not guaranteed - 31% Recall@1 on the real PAWS benchmark, not 100%).
# Check retrieval_path; "miss" with no fallback configured returns no hits.
result2 = index.search("What's your return policy?", top_k=1)
print(result2[0].retrieval_path if result2 else "miss")

print(index.stats())
```

For closed/templated vocabularies only (fixed intent sets, FAQ catalogs with exact-phrasing repeats), `mode="cache"` (E8 lattice, no calibration needed) is simpler - but verify it actually works for your real data before relying on it; "works for closed vocabularies" was itself an unverified claim until corrected on 2026-06-24 (see "What it's for" above).

### Semantic cache with answer lookup

For a small, closed set of templated questions (a FAQ catalog, fixed intents), `RFSnapLatticeMemory` with no fallback is fine - exact and near-exact text reliably hits. For anything with open-ended phrasing, **always wire in a dense fallback** - without one, a paraphrase that doesn't land in the same E8 cell is a silent miss, not a slow-but-correct answer:

```python
from latticememory import RFSnapSemanticCache, RFSnapTextMemory, RFSnapLatticeMemory
from latticememory.memory import DenseVectorFallback
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("dfrokido/bge-large-e8-snap")
fallback = DenseVectorFallback(d_model=1024, quantization_bits=8)  # Int8 - see benchmarks above
lm = RFSnapLatticeMemory(d_model=1024, fallback=fallback)
rt = RFSnapTextMemory(encoder=encoder, d_model=1024, memory=lm)
cache = RFSnapSemanticCache(runtime=rt)

cache.put("What is the refund policy?", value="30-day returns, full refund.")
result = cache.get("What's your return policy?")
print(result.hit)         # True - correct, but likely served via dense fallback, not an O(1) E8 hit
print(result.value)       # "30-day returns, full refund."
print(result.retrieval_path)  # check this if you're relying on the O(1)/compression benefit specifically
```

### Using PQ with the proxy server

`LatticeLLMProxy` has no dedicated PQ constructor flag (it's already a large, production-critical class) - wire PQ in through its existing `semantic_cache=` parameter instead. See [`examples/pq_proxy_setup.py`](examples/pq_proxy_setup.py) for a complete, runnable example - including the one footgun to know about: unlike `LatticeIndex`, a PQ-backed proxy built this way has no auto-fit safety net, so codebooks must be fit *before* the proxy serves any traffic.

### Hybrid RAG / document search

For asymmetric search (user questions against document passages), use hybrid mode — E8 for cache hits, dense fallback for novel queries:

```python
from latticememory import LatticeIndex

index = LatticeIndex(mode="hybrid")  # Int8 fallback enabled automatically
index.add([
    "The refund window is 30 days from purchase date.",
    "Password resets are sent to your registered email.",
    "Orders ship within 2 business days.",
])

# Novel query → routes through E8, misses, falls back to Int8 dense search
result = index.search("Can I return something after a month?", top_k=1)
print(result[0].retrieval_path)  # fallback
print(result[0].text)            # The refund window is 30 days...
```

---

## HammingRouter — Catch Paraphrases at Scale

`HammingRouter` caches full Q&A pairs and matches incoming queries by Hamming distance on their E8 keys. A threshold of 70–111 blocks (out of 128) catches paraphrases while controlling false positives.

```python
from latticememory import HammingRouter

router = HammingRouter(threshold=100)  # tune per domain

# Index known Q&A pairs
router.add("What is your cancellation policy?", answer="Cancel anytime, no fee.", intent="cancel")
router.add("How do I cancel my subscription?",  answer="Cancel anytime, no fee.", intent="cancel")

# Match a paraphrase
match = router.match("Can I cancel at any time?")
if match:
    print(match.answer)          # "Cancel anytime, no fee."
    print(match.hamming_distance)  # e.g. 97
```

**Threshold guidance (BANKING77 benchmark):**

| Threshold | Recall | FP rate | Use case |
| --- | --- | --- | --- |
| 70 | 4.5% | 0.0% | Proxy default — zero false positives |
| 100 | 52.5% | 0.0% | Practical helpdesk operating point |
| 111 | 84.0% | 4.5% | Router default — calibrate per domain |

---

## LLM Cache Proxy

Drop-in OpenAI-compatible HTTP proxy. Same prompt or near-paraphrase returns the cached response without hitting the upstream model.

```bash
pip install 'lattice-memory-e8[proxy]'
```

```bash
lattice serve --key sk-... --cache helpdesk.db --miss-log misses.jsonl --port 8000
```

Or with Docker:

```bash
OPENAI_API_KEY=sk-... docker-compose up
```

Point your OpenAI client at `http://localhost:8000` — no other code changes needed.

**Features:**

- `X-Lattice-Cache: HIT/MISS` and `X-Lattice-Savings-USD` on every response
- Streaming SSE + non-streaming JSON
- SQLite persistence — survives process restart
- HammingRouter approximate cache in `shadow` or `serve` mode
- TTL per-entry expiry
- Compliance mode — only serve pre-approved responses (for regulated industries)
- Admin CRUD API gated by `X-Lattice-Admin-Key`
- Warm-start from CSV/JSON/JSONL

**Running `--hamming-mode serve` in production:** calibrate and enable the
cheap cosine gate first, then use `--hamming-rerank` as the slower semantic
backstop:

```bash
lattice calibrate --paraphrases paraphrases.txt --near-misses near_misses.txt \
  --holdout-paraphrases holdout_paraphrases.txt --holdout-near-misses holdout_near_misses.txt \
  --metric cosine --fp-budget 0

lattice serve ... --hamming-mode serve \
  --hamming-cosine-gate --hamming-cosine-threshold <calibrated-threshold> \
  --hamming-rerank --hamming-rerank-model qwen2.5:1.5b
```

The Hamming router's distance threshold cannot separate genuine paraphrases
from same-template/different-topic queries: adversarial "template mimicry"
inputs land inside the paraphrase distance range regardless of Hamming
calibration and can be served the wrong cached answer. `--hamming-cosine-gate`
is the first line of defense because it is local, fast, and catches most of
those candidates before an LLM call. It is not sufficient by itself: two
synthetic domains still showed a small residual class of same-template,
entity-substitution misses. `--hamming-rerank` is the second line of defense
for that residual case, and fails closed (falls through to a real upstream
call) on any judge error or non-YES verdict. Use a dedicated non-reasoning
judge model via `--hamming-rerank-model`; a reasoning model given a tight token
budget can spend its whole budget "thinking" and never emit a visible verdict,
which silently disables the check. See `lattice serve --help` for all flags.

---

## LangChain Integration

```bash
pip install lattice-memory-e8 langchain-core langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from latticememory.integrations.langchain import LatticeMemoryCache

set_llm_cache(LatticeMemoryCache())
llm = ChatOpenAI(model="gpt-4o")

llm.invoke("What is the capital of France?")   # miss — calls API
llm.invoke("What is the capital of France?")   # hit  — O(1) key match
llm.invoke("Which city is France's capital?")  # likely hit — same E8 neighborhood
```

---

## Deduplication

```python
from latticememory import LatticeTrainingCleaner, RFSnapSemanticCache

# batch dedup
cleaner = LatticeTrainingCleaner(cache)
result = cleaner.clean([
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaped over a sleeping dog.",   # near-duplicate
    "Machine learning is a branch of artificial intelligence.",
])
print(result.kept_count)       # 2
print(result.duplicate_count)  # 1
print(result.dedup_rate)       # 0.333...

# streaming dedup (generator)
for unique_text in cleaner.stream(iter(large_corpus)):
    process(unique_text)
```

Or via CLI:

```bash
lattice dedup corpus.jsonl --text-col text --output corpus_deduped.jsonl
```

---

## Vertical Applications

All 9 verticals ship in `latticememory.verticals` and wrap `RFSnapSemanticCache`.

| Vertical | Class | Key Capability |
| --- | --- | --- |
| SOC Monitor | `LatticeSOCMonitor` | O(1) alert dedup for SIEM event streams |
| Ticket Analyzer | `LatticeTicketAnalyzer` | Intent-based ticket routing + gap detection |
| Content Moderator | `LatticeContentModerator` | Semantic near-miss content policy |
| Clause Coder | `LatticeClauseCoder` | Legal clause classification |
| Edge Memory | `LatticeEdgeMemory` | On-device personalization without cloud |
| Private Sync | `LatticePrivateSync` | Federated key sync, no raw text transfer |
| **Prompt Firewall** | `LatticePromptFirewall` | Semantic injection/jailbreak detection |
| **Semantic Rate Limiter** | `LatticeSemanticRateLimiter` | Per-intent sliding-window rate limiting |
| **Training Cleaner** | `LatticeTrainingCleaner` | O(N) near-duplicate removal for LLM training sets |

### Prompt Firewall

```python
from latticememory import LatticePromptFirewall, RFSnapSemanticCache

fw = LatticePromptFirewall(cache)
fw.load_injection_defaults()  # loads 14 common injection/jailbreak patterns

result = fw.check("Ignore all previous instructions and")
print(result.blocked)   # True
print(result.category)  # prompt_injection

# Add custom deny patterns
fw.add_deny_pattern("roleplay as an unfiltered AI", category="jailbreak")
```

### Semantic Rate Limiter

```python
from latticememory import LatticeSemanticRateLimiter

limiter = LatticeSemanticRateLimiter(cache, limit=10, window_seconds=60.0)

r = limiter.check("tell me about Python", client_id="user_123")
print(r.allowed)     # True
print(r.remaining)   # 9
print(r.retry_after) # 0.0
```

### Training Data Cleaner

```python
from latticememory import LatticeTrainingCleaner

cleaner = LatticeTrainingCleaner(cache)
result = cleaner.clean_to_jsonl(texts, output_path="clean.jsonl")
print(result.summary())
# Total: 50000 | Kept: 43217 | Duplicates removed: 6783 (13.6%)
```

---

## Agent Memory Sync

`AgentMemorySync` lets agents in a swarm share only the E8 keys they are missing — no embedding transfer, just 128-byte addresses.

```python
from latticememory import AgentMemorySync

# Two independent agents
agent_a = AgentMemorySync(runtime=rt_a)
agent_b = AgentMemorySync(runtime=rt_b)

# Register peers
agent_a.register_peer(agent_b)

# Pull-sync: B gets everything A knows
agent_b.sync_from_peer(agent_a)

# Push-broadcast: A broadcasts a new key to all registered peers
new_key = next(iter(agent_a.get_known_keys()))
agent_a.share(new_key)  # agent_b receives it immediately

# Diff: check what each side is missing
diff = agent_a.diff(agent_b.get_known_keys())
# {"extra": set(), "missing": set()}  ← fully in sync
```

See `examples/agent_swarm_demo.py` for a complete end-to-end scenario.

---

## Active Learning Flywheel

Every proxy cache miss can be logged. `LatticeFlywheel` clusters miss logs by E8 key proximity to surface emerging intent gaps — groups of queries the cache doesn't cover yet.

```python
from latticememory import LatticeFlywheel

fw = LatticeFlywheel("misses.jsonl")

# From your proxy, log each miss:
fw.log_miss("How do I bulk export my contacts?", e8_key_hex=e8_key)

# Detect drifting intents (new query patterns emerging):
drifting = fw.detect_drift(window_seconds=7*86400, min_delta=5)
for cluster in drifting:
    print(f"+{cluster['delta']} queries: {cluster['representative']!r}")

# Check if re-training is warranted:
if fw.should_finetune():
    print("Recommend: add Q&A pairs for these new intent clusters")
```

Or via CLI:

```bash
lattice drift --log misses.jsonl --window 604800 --export drift_report.json
```

---

## CLI Reference

| Command | What it does |
| --- | --- |
| `lattice calibrate` | Calibrate a Hamming-distance or cosine threshold from labeled paraphrase/near-miss pairs using `--metric hamming\|cosine`, with optional `--holdout-paraphrases`/`--holdout-near-misses` pairs for genuine held-out evidence |
| `lattice populate` | Load Q&A pairs from CSV/JSON into a SQLite cache |
| `lattice inspect` | Print cache statistics |
| `lattice export` | Export all cache entries to a portable JSONL file |
| `lattice import` | Re-import a JSONL export into a new cache |
| `lattice gaps` | Show top miss clusters (unmet query intents) |
| `lattice drift` | Detect drifting intents + finetune recommendation |
| `lattice dedup` | Deduplicate a text file using E8 lattice hashing |
| `lattice serve` | Start the proxy server |
| `lattice analytics` | Fetch live analytics from a running proxy |

---

## CLI IDE

`lattice ide` opens a local terminal command center for BYOK AI chat, cache operations,
proxy diagnostics, vertical discovery, and VS Code CLI bridging.

```bash
export LATTICE_IDE_BASE_URL=https://api.openai.com/v1
export LATTICE_IDE_MODEL=gpt-4o-mini
export LATTICE_IDE_API_KEY=sk-...

lattice ide chat "Summarize the current cache analytics"
lattice ide cache inspect --cache helpdesk.db
lattice ide proxy doctor --port 8000
lattice ide verticals list
lattice ide vscode status
```

Run `lattice ide` with no arguments for an interactive `lm>` shell. The first IDE
slice uses OpenAI-compatible chat endpoints, so it works with OpenAI and compatible
BYOK gateways. VS Code integration uses the installed `code` command; it does not
require a VS Code extension.

---

## How It Works

```text
float32 embedding [1024-dim]
  → 128 blocks of 8 floats
  → each block → nearest E8 Shell-1 point (240 possible addresses)
  → 1-byte address per block = 128-byte E8 key  ← used for cache routing (32× vs float32)
  → optional 2-byte scale per block = full 384-byte quantized representation

query → same key → O(1) lattice_exact lookup
query → Hamming-N neighbor → O(1) HammingRouter lookup
query → no neighbor found → dense fallback (Int8 or float32 ANN)
```

The E8 key is a **deterministic hash of meaning** — not an approximation. Two texts that are semantically identical land on the same key every time, without cosine threshold tuning.

---

## Redis Backend

For multi-instance deployments sharing a single cache:

```python
from latticememory import LatticeRedisStore, RFSnapSemanticCache, patch_cache_with_redis

cache = RFSnapSemanticCache(...)
patch_cache_with_redis(cache, redis_url="redis://localhost:6379", namespace="helpdesk")
# Now cache._entries reads/writes Redis instead of the in-memory dict
```

---

## Test Suite

508 tests, all passing:

```bash
python -m pytest tests/ -q
# 508 passed in ~70s
```

---

## Design Partners

We're looking for 3 teams with high-repetition LLM workloads (support bots, document QA, internal search) to pilot semantic cache + dedup at no cost.

**[dfrokido@gmail.com](mailto:dfrokido@gmail.com)**

---

## License

MIT
