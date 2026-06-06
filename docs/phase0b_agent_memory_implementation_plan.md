# Implementation Plan: Phase 0b (AI Agent Episodic Memory & Semantic Deduplication)

This plan outlines the steps to build **Phase 0b** of the LatticeMemory roadmap: implementing AI Agent Episodic Memory and LatticeDedup, after executing the three cleanup items to make Phase 0 and Phase 1 fully shippable.

---

## User Review Required

### 1. Agent Episodic Memory Design
We propose creating a dedicated `AgentEpisodicMemory` wrapper around `RFSnapLatticeMemory`. This will support:
* **$O(1)$ Deduplication**: Memories that snap to the exact same E8 address blocks are grouped.
* **Versioning**: Multiple observations sharing the same address are stored as historical versions.
* **Memory Auditing**: Allowing agents to query which E8 addresses have been read or written, creating an access audit trail to help mitigate hallucinations.

### 2. LatticeDedup (Semantic Deduplication) Design
We propose creating a `LatticeDedup` module to perform high-speed corpus deduplication:
* **E8 Block Hashing**: Maps each document to a unique composite E8 key (based on its block-level snapping).
* **Grouping & Clustering**: Documents that share the same E8 key are clustered.
* **Corpus Filtering**: Keeps a single canonical document per E8 key, removing semantic duplicates in $O(N)$ time.

---

## Proposed Changes

### 1. Completed Cleanup Items
* [x] **Compile the Rust/WASM Kernel**: Ran `wasm-pack build --target web` inside the `rust/` directory to generate the WASM package in `rust/pkg/`.
* [x] **Add LlamaIndex Extra**: Updated [pyproject.toml](file:///e:/latticememory/pyproject.toml) to declare `llamaindex = ["llama-index-core"]` under optional dependencies.
* [x] **Prepare PyPI Release**: Executed `build_dist.py` to compile the source distribution and wheel, and validated it using `twine check` (all checks passed).

---

### 2. New Code Components

#### [NEW] [agent_memory.py](file:///e:/latticememory/latticememory/agent_memory.py)
* Implement `AgentEpisodicMemory` class:
  * Wraps `RFSnapLatticeMemory` (backed by SQLite for durability).
  * `add_episode(self, content: str, metadata: dict = None) -> dict`: Snaps content, checks if the address already exists. If yes, registers it as a new version under the same E8 address; if no, creates it.
  * `get_episode_history(self, content: str) -> list`: Retrieves all versioned memories matching the E8 address of the provided text.
  * `audit_trail(self) -> list`: Returns the history of read/write operations on E8 keys to trace memory access.

#### [NEW] [dedup.py](file:///e:/latticememory/latticememory/dedup.py)
* Implement `LatticeDedup` class:
  * `deduplicate(self, documents: list[str], threshold: float = 1.0) -> dict`:
    * Computes E8 snap keys for all documents.
    * Clusters documents by their E8 snap key.
    * Returns a dict with `unique_documents` (list of deduplicated texts) and `duplicates` (mapping of E8 keys to lists of duplicate texts).

---

### 3. Runnable Demos

#### [NEW] [agent_memory_demo.py](file:///e:/latticememory/examples/agent_memory_demo.py)
* Demonstrate agent episodic memory recording observations like:
  * "The user likes coffee in the morning."
  * "The user prefers espresso at 8 AM." (snaps to same E8 address)
* Verify that:
  1. The duplicate observation is versioned under the same E8 address.
  2. Retrieving by query pulls the version history.
  3. The audit trail records memory reads and writes.

#### [NEW] [semantic_deduplication_demo.py](file:///e:/latticememory/examples/semantic_deduplication_demo.py)
* Load a corpus of texts containing semantic duplicates (e.g. differently phrased tech articles).
* Run `LatticeDedup` to filter out duplicates in $O(N)$ time.
* Print statistics showing the percentage of data deduplicated and the groups of duplicate phrasings.

---

## Verification Plan

### Automated Tests
* Run `python examples/agent_memory_demo.py` and inspect console output.
* Run `python examples/semantic_deduplication_demo.py` and inspect console output.
* Run `pytest` to verify no regressions in the core package.
