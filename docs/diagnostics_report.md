# E8 Routing Diagnostics & Hamming Distance Analysis

We conducted a comprehensive diagnostic evaluation of the E8 routing failures on the MS MARCO dataset. Here is the summary of our findings and experiments.

---

## 1. Diagnostic Metrics Summary

We ran five separate alignment experiments on the MS MARCO dataset using `dfrokido/bge-large-e8-snap` (1024-dim, 128 blocks of 8D):

| Experiment | Configuration / Model | Train Mean Hamming (128 blocks) | Train Exact Matches | Val Mean Hamming (128 blocks) | Val Exact Matches |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline** | Original embeddings (no adapter) | 99.31 | 0.00% | 99.31 | 0.00% |
| **2. GPU 2K Run** | 5 epochs, MLP, default weights | 99.16 | 0.00% | 99.16 | 0.00% |
| **3. Ridge Regression** | Linear least-squares fit on Train | 9.73 | 79.50% | 102.29 | 0.00% |
| **4. Joint Alignment** | Dual projection (`CrossModelAligner`) | 63.60 | 0.00% | 119.48 | 0.00% |
| **5. Tuned expected Hamming** | 30 epochs, $\lambda_{neighborhood}=10.0$ | 9.39 | 55.50% | 106.26 | 0.00% |

---

## 2. Key Findings & Root Cause Analysis

### A. The Discontinuity of E8 Quantization
E8 lattice snapping is highly sensitive to small coordinate variations. Each 1024D embedding is divided into 128 blocks of 8D, and each block is snapped to one of 240 Voronoi cells on the sphere. Even a minor difference in a single block will snap it to a different codeword.
* At a baseline cosine similarity of ~0.7 (typical for MS MARCO query-passage pairs), the block-level snapping outputs are virtually independent, resulting in a mean Hamming distance of **99.31** (almost entirely random).

### B. Mismatch Asymmetry (Generalization Failure)
* **Symmetric vs Asymmetric:** In paraphrase retrieval or semantic caching, queries and documents represent the *same* text/concept. Thus, they can have very high cosine similarity and can route through the same E8 cells after domain-specific adapter training. The current 100% result is from a small controlled paraphrase split, not a broad production benchmark.
* **MS MARCO:** In passage retrieval, a query is a question and a document is a passage. The mapping from question to passage is highly non-linear, context-dependent, and semantic.
* **Ridge Regression Generalization:** Solving Ridge Regression on the validation set itself reduced Mean Hamming Distance to **9.73** (79.5% exact matches). However, solving it on the train set and evaluating on the validation set yielded **102.29** (worse than baseline). 
* **Capacity Limit:** A linear projection or shallow MLP adapter does not have the capacity to learn a general question-answering mapping on 2,000 examples. To map unseen questions to passages, we rely on the deep transformer layers of the bi-encoder itself.

---

## 3. Recommended Actions & Next Steps

### 1. Reserve Adapters for Symmetric Tasks
Query adapters (e.g., `ResidualMLPAdapter`) should be reserved for symmetric retrieval tasks:
* **Paraphrase retrieval**
* **Semantic Cache** (e.g., LangChain cache)
* **Local IoT Command Normalization**

### 2. Rely on Cosine Fallback for Asymmetric QA
For general QA and passage retrieval (like MS MARCO), our current exact/Hamming-1 key routing did not generalize from train to validation due to semantic asymmetry. The correct current architecture is:
* Use the E8 keys for exact/paraphrase O(1) matching.
* If a query misses the exact/Hamming-1 key, fall back to standard cosine similarity search over stored dense embeddings.
* Float32 dense fallback can match the float32 cosine baseline by construction. Int8 fallback is implemented and currently measures 95.1% Recall@10 overlap vs float32 on the 1K-doc real-model paraphrase fallback benchmark. Int4 fallback is implemented but measured too lossy for QA/RAG fallback quality on that benchmark.
