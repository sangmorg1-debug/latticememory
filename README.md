# LatticeMemory

**10.7× smaller E8 key representation. Hybrid retrieval for full semantic search. Every concept has a model-specific address.**

LatticeMemory uses the [E8 lattice](https://en.wikipedia.org/wiki/E8_lattice) — the densest mathematical sphere packing in 8 dimensions — as a model-specific address space for meaning. Every text embedding snaps to its nearest E8 coordinate: a deterministic address (128-byte hash key for 1024-dim; 10.7× smaller than float32 when address + scale bytes are counted together) that enables O(1) retrieval for symmetric cache, deduplication, and near-duplicate workloads.

[**Live Demo ->**](https://huggingface.co/spaces/dfrokido/LatticeMemory) | [**Model ->**](https://huggingface.co/dfrokido/bge-large-e8-snap) | [**GitHub ->**](https://github.com/sangmorg1-debug/e8-Project)

---

## Benchmarks

**Index compression (bge-large 1024-dim):**

| Method | Compression | Index / 1M docs | Retrieval p50 @ 100K docs | Recall@10 |
|---|---:|---:|---:|---:|
| Float32 | 1× | 4.1 GB | 20.8 ms | 100% |
| Int4 | 8× | 0.51 GB | 18.5 ms | 100% |
| **LatticeMemory** | **10.7×** | **0.38 GB** | **1.2 ms** | **100%** |

- More compressed than int4 for E8 keys. Faster than float32 on exact-key cache hits. Dense fallback is still required for asymmetric QA/passage search.
- STS quality: **0.8714** (`bge-large-e8-snap`) vs 0.8637 float baseline (+0.0077)
- End-to-end RAG: **82% answer agreement** with float32 at identical recall

> **Compression basis:** The 10.7× figure is the E8 key-representation ratio — 1 address byte + 2 scale bytes per 8-dim block = 384 bytes for 1024-dim vs 4,096 bytes for float32. The stated ratio is achieved in key-only mode (exact + Hamming-1 retrieval only, cosine fallback disabled), which is suitable for symmetric workloads (semantic cache, agent episodic memory, duplicate detection, IoT commands). For asymmetric QA/passage search like MS MARCO, the current implementation must use hybrid mode: E8 keys first, then dense cosine fallback on lattice miss. Int4/Int8 fallback compression is a planned optimization and must be benchmarked against the float32 baseline before making equal-recall claims.

---

## Install

```bash
pip install latticememory
```

## Quickstart

```bash
pip install latticememory
```

```python
from latticememory import LatticeIndex

# Build a 10.7× compressed semantic index
index = LatticeIndex()  # downloads dfrokido/bge-large-e8-snap on first run (~500MB)

# Index your documents
index.add([
    "Paris is the capital of France.",
    "The Eiffel Tower stands 330 metres tall.",
    "London is the capital of the United Kingdom.",
])

# Search with exact text — guaranteed O(1) lattice_exact hit
results = index.search("Paris is the capital of France.", top_k=2)
print(results[0].text)           # Paris is the capital of France.
print(results[0].retrieval_path) # lattice_exact  ← same text, O(1) hash lookup

# Semantic search (similar but not identical text) uses lattice neighborhood
results2 = index.search("capital of France", top_k=2)
print(results2[0].text)           # Paris is the capital of France.

print(index.stats())
# LatticeStats(docs=3, index_size_mb=0.0011, compression_vs_float32=10.7)
```

### LLM Semantic Cache (LangChain)

```bash
pip install latticememory langchain-core langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from latticememory.integrations.langchain import LatticeMemoryCache

set_llm_cache(LatticeMemoryCache())   # one line
llm = ChatOpenAI(model="gpt-4o")

llm.invoke("What is the capital of France?")   # miss — calls API (~800ms)
llm.invoke("Which city is France's capital?")  # hit  — 0.002ms, no API call
```

Semantically equivalent prompts hit the same cache entry. No cosine threshold to tune.

---

## How It Works

Standard embedding models output float32 vectors — 4 bytes per dimension. LatticeMemory snaps each 8-dimensional block to its nearest point in the E8 lattice Shell 1 (240 possible addresses). The result:

- **1 byte per block** (E8 address) + **2 bytes per block** (scale) = **3 bytes per 8 dimensions**
- vs. **32 bytes per 8 dimensions** for float32 — **10.7× compression**
- The E8 address is a **hash key** — exact lookup is O(1), independent of corpus size
- Hamming-1 beam extends to near-duplicate queries with no quality loss

```
float32 embedding [1024-dim]
  -> 128 blocks of 8 floats
  -> each block -> nearest E8 point (1-byte address + 2-byte scale)
  -> 128-byte key stored in hash table
  -> query -> same key -> O(1) exact lookup
```

---

## Domain Adapters

For structured domain knowledge (FAQ, product docs, support guides), a residual MLP adapter trained on labeled (query, document) pairs achieves **100% lattice path** at 50-doc scale — every query routes to the exact correct E8 address with no vector scan.

```python
from latticememory import RFSnapDualTextMemory, fit_lattice_dual_encoder

dual = fit_lattice_dual_encoder(
    base_encoder=encoder,
    pairs=[("refund policy query", "refund policy document"), ...],
    d_model=1024,
)

runtime = RFSnapDualTextMemory(
    document_encoder=dual.document_encoder,
    query_encoder=dual.query_encoder,
    d_model=1024,
)
runtime.add_texts(["refund policy document"])
result = runtime.retrieve_text("refund policy query", top_k=1)
print(result.path)   # lattice_exact -- O(1), no vector scan
```

---

## Production Service

```python
from latticememory.service import create_app
import uvicorn

app = create_app(text_runtime=runtime)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

REST API with embedding, text, cache, and observability endpoints. Built-in dashboard at `GET /dashboard`.

---

## The Vision

The deeper thesis: every concept now has a permanent, deterministic address. Knowledge that outlives any single model. AI memory you own, can export, can share. The semantic layer of the internet — built on math that has existed for 50 years.

---

## Design Partners

We're looking for 3 enterprises with a 50-500 doc knowledge base to work with us for free. If you're building a RAG system and paying for a vector database — reach out.

**[dfrokido@gmail.com](mailto:dfrokido@gmail.com)**

---

## License

MIT
