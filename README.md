# LatticeMemory

**Semantic cache, dedup, and hybrid memory — 32× compressed E8 keys for instant repeat-query hits, dense fallback for novel retrieval.**

LatticeMemory uses the [E8 lattice](https://en.wikipedia.org/wiki/E8_lattice) — the densest sphere packing in 8 dimensions — as a deterministic address space for text embeddings. Every 1024-dim embedding snaps to a 128-byte E8 key. Identical or near-identical text lands on the same key; novel queries fall through to a dense float32/Int8 fallback.

[**Live Demo →**](https://huggingface.co/spaces/dfrokido/LatticeMemory) | [**Model →**](https://huggingface.co/dfrokido/bge-large-e8-snap) | [**GitHub →**](https://github.com/sangmorg1-debug/latticememory)

---

## What it's for

| Workload | E8 path | Fallback needed? |
| --- | --- | --- |
| Repeat / paraphrase LLM queries (cache) | ✅ O(1) exact or Hamming hit | No |
| Semantic deduplication, near-duplicate detection | ✅ Key collision = duplicate | No |
| Dataset quality filtering, semantic sharding | ✅ Stable cluster addresses | No |
| IoT/command normalization (symmetric vocab) | ✅ Fixed command set → fixed keys | No |
| **Asymmetric QA/passage search (RAG)** | ❌ Query ≠ passage in E8 space | **Yes — Int8 or float32 required** |

E8 keys route fast for content that is semantically identical or near-identical. They are not a replacement for vector search on asymmetric workloads where the query text and the correct passage are structurally different.

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
pip install latticememory
```

Optional extras:

```bash
pip install 'latticememory[proxy]'   # FastAPI proxy server (fastapi, uvicorn, httpx)
pip install 'latticememory[redis]'   # Redis backend for multi-instance caches
pip install 'latticememory[hf]'      # HuggingFace datasets integration
pip install 'latticememory[faiss]'   # FAISS vector fallback
```

---

## Quickstart

### Semantic cache (the primary use case)

```python
from latticememory import LatticeIndex

index = LatticeIndex()  # downloads dfrokido/bge-large-e8-snap on first run (~500MB)

index.add([
    "What is the refund policy?",
    "How do I reset my password?",
    "Where is my order?",
])

# Exact text → guaranteed O(1) lattice_exact hit
result = index.search("What is the refund policy?", top_k=1)
print(result[0].retrieval_path)  # lattice_exact

# Near-paraphrase → lattice_exact or Hamming hit (same E8 neighborhood)
result2 = index.search("What's your return policy?", top_k=1)
print(result2[0].retrieval_path)  # lattice_exact or lattice_hamming

print(index.stats())
```

### Semantic cache with answer lookup

```python
from latticememory import RFSnapSemanticCache, RFSnapTextMemory, RFSnapLatticeMemory
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("dfrokido/bge-large-e8-snap")
lm = RFSnapLatticeMemory(d_model=1024)
rt = RFSnapTextMemory(encoder=encoder, d_model=1024, memory=lm)
cache = RFSnapSemanticCache(runtime=rt)

cache.put("What is the refund policy?", value="30-day returns, full refund.")
result = cache.get("What's your return policy?")  # paraphrase hit
print(result.hit)        # True
print(result.value)      # "30-day returns, full refund."
```

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
pip install 'latticememory[proxy]'
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

---

## LangChain Integration

```bash
pip install latticememory langchain-core langchain-openai
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

491 tests, all passing:

```bash
python -m pytest tests/ -q
# 491 passed in ~70s
```

---

## Design Partners

We're looking for 3 teams with high-repetition LLM workloads (support bots, document QA, internal search) to pilot semantic cache + dedup at no cost.

**[dfrokido@gmail.com](mailto:dfrokido@gmail.com)**

---

## License

MIT
