# Implementation Plan: Phase 1b (Local Edge Products & IoT Cache)

This plan outlines the steps to build **Phase 1b** of the LatticeMemory roadmap: demonstrating local-first semantic routing and edge index search through concrete product exemplars.

---

## Proposed Changes

### 1. IoT Command Normalization & Adapter Demo

We will create a runnable demo that fits a smart home command adapter:

#### [NEW] [examples/iot_command_normalizer.py](file:///e:/latticememory/examples/iot_command_normalizer.py)
* Train a query adapter (`train_lattice_contrastive_encoder`) using a synthetic dataset of smart home command variations (e.g., query: *"make the cooking area brighter"*, target: *"turn_on_kitchen_lights"*).
* Snaps the queries to the exact E8 addresses of canonical targets.
* Logs stats showing the exact hit rate and $O(1)$ lookup path.

---

### 2. Browser Extension Search Demo

We will create a JavaScript script showing how browser extensions load and use the WASM kernel locally:

#### [NEW] [examples/browser_extension_demo.js](file:///e:/latticememory/examples/browser_extension_demo.js)
* Provide a clean JavaScript loader that mocks the compiled WASM kernel from Phase 1.
* Index a list of mock history urls and search query targets.
* Perform local-first semantic history matching in JavaScript.

---

## Verification Plan

### Automated Run
* Run `python examples/iot_command_normalizer.py` to verify the adapter converges and achieves a high exact E8 key match rate (> 90%) for semantically equivalent queries.
* Run `node examples/browser_extension_demo.js` to verify JavaScript E8 snapping is syntactically ready.
