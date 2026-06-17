# LatticeMemory

**Semantic cache, dedup, and hybrid memory — 32× compressed E8 keys for instant repeat-query hits, dense fallback for novel retrieval.**

LatticeMemory uses the [E8 lattice](https://en.wikipedia.org/wiki/E8_lattice) — the densest sphere packing in 8 dimensions — as a deterministic address space for text embeddings. Every 1024-dim embedding snaps to a 128-byte E8 key. Identical or near-identical text lands on the same key; novel queries fall through to a dense float32/Int8 fallback.

[**Live Demo →**](https://huggingface.co/spaces/dfrokido/LatticeMemory) | [**Model →**](https://huggingface.co/dfrokido/bge-large-e8-snap) | [**GitHub →**](https://github.com/sangmorg1-debug/latticememory)

---

## What it's for

| Workload | E8 path | Fallback needed? |
| --- | --- | --- |
| Repeat / paraphrase LLM queries (cache) | ✅ O(1) exact or Hamming-1 hit | No |
| Semantic deduplication, near-duplicate detection | ✅ Key collision = duplicate | No |
| Dataset quality filtering, semantic sharding | ✅ Stable cluster addresses | No |
| IoT/command normalization (symmetric vocab) | ✅ Fixed command set → fixed keys | No |
| **Asymmetric QA/passage search (RAG)** | ❌ Query ≠ passage in E8 space | **Yes — Int8 or float32 required** |

E8 keys route fast for content that is semantically identical or near-identical. They are not a replacement for vector search on asymmetric workloads where the query text and the correct passage are structurally different (e.g. MS MARCO: question vs. answer paragraph).

---

## Benchmarks

**Compression (bge-large 1024-dim):**

| Method | Compression | Index / 1M docs | Retrieval p50 @ 100K docs |
|---|---:|---:|---:|
| Float32 | 1× | 4.1 GB | 20.8 ms |
| **LatticeMemory E8 keys** | **32×** | **0.13 GB** | O(1) on key hit |

**Fallback quality (1K docs, 100 paraphrase queries, recall vs float32):**

| Fallback | Compression vs float32 | Recall@10 overlap | Top-1 agreement | Search p50 |
|---|---:|---:|---:|---:|
| Float32 | 1× | 100.0% | 100.0% | 0.14 ms |
| Int8 | 4× | 95.1% | 91.0% | 1.97 ms |
| Int4 | 8× | 12.1% | 1.0% | 4.21 ms |

- **Int8 fallback** is the recommended fallback for RAG/QA — 4× smaller than float32, 95% recall parity.
- **Int4 fallback** is retrieval-unsafe for QA. Use only for dedup/clustering where approximate grouping is acceptable.
- **STS quality:** `bge-large-e8-snap` scores 0.8714 vs 0.8637 float baseline (+0.0077).

> **Compression basis:** 1 address byte per 8-dim block × 128 blocks = 128 bytes for 1024-dim vs 4,096 bytes float32 = 32×. Ratio applies to E8 key storage only. Hybrid mode (key + fallback index) stores both representations; the E8 layer acts as a fast-path cache in front of the dense index.

---

## Install

```bash
pip install latticememory
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

# Near-paraphrase → lattice_exact or Hamming-1 hit (same E8 neighborhood)
result2 = index.search("What's your return policy?", top_k=1)
print(result2[0].retrieval_path)  # lattice_exact or lattice_hamming1

print(index.stats())
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
print(result[0].retrieval_path)  # fallback (float32 or Int8)
print(result[0].text)            # The refund window is 30 days...
```

---

## LLM Semantic Cache (LangChain)

```bash
pip install latticememory langchain-core langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from latticememory.integrations.langchain import LatticeMemoryCache

set_llm_cache(LatticeMemoryCache())
llm = ChatOpenAI(model="gpt-4o")

llm.invoke("What is the capital of France?")   # miss — calls API (~800ms)
llm.invoke("What is the capital of France?")   # hit  — O(1) exact key match, no API call
llm.invoke("Which city is France's capital?")  # likely hit — same E8 neighborhood
```

Repeated and near-identical prompts hit the same cache entry. Cache hit rate depends on query similarity — symmetric caches (same users, same queries) see near-100% repeat-hit rates. Novel queries miss and fall through to the LLM as normal.

---

## How It Works

```text
float32 embedding [1024-dim]
  → 128 blocks of 8 floats
  → each block → nearest E8 Shell-1 point (240 possible addresses)
  → 1-byte address + 2-byte scale per block = 384-byte key
  → key stored in hash table

query → same key → O(1) lattice_exact lookup
query → Hamming-1 neighbor → O(1) lattice_hamming1 lookup
query → no neighbor found → dense fallback (Int8 or float32 ANN)
```

The E8 key is a **deterministic hash of meaning** — not an approximation. Two texts that are semantically identical land on the same key every time, without cosine threshold tuning.

---

## Deduplication

```python
from latticememory import LatticeIndex

index = LatticeIndex()

docs = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaped over a sleeping dog.",   # near-duplicate
    "Machine learning is a branch of artificial intelligence.",
]

for doc in docs:
    result = index.search(doc, top_k=1)
    if result and result[0].retrieval_path in ("lattice_exact", "lattice_hamming1"):
        print(f"DUPLICATE: {doc[:50]}...")
    else:
        index.add([doc])
```

---

## Production Service

```bash
pip install 'latticememory[proxy]'
```

```python
from latticememory.service import create_app
from latticememory.text_runtime import RFSnapTextMemory
import uvicorn

runtime = RFSnapTextMemory()
app = create_app(text_runtime=runtime)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

REST API with embedding, text, cache, and observability endpoints. Built-in dashboard at `GET /dashboard`.

---

## Design Partners

We're looking for 3 teams with high-repetition LLM workloads (support bots, document QA, internal search) to pilot semantic cache + dedup at no cost.

**[dfrokido@gmail.com](mailto:dfrokido@gmail.com)**

---

## License

MIT
