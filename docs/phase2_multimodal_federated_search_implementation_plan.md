# Implementation Plan: Phase 2 (Multimodal Snapping & Federated Search)

This plan outlines the steps to build **Phase 2** of the LatticeMemory roadmap: demonstrating federated zero-leak index queries and aligned cross-modal snapping.

---

## Proposed Changes

### 1. Federated Semantic Search Demo

We will create a runnable demo simulating a B2B consortium search network (e.g., law firms or hospitals):

#### [NEW] [examples/federated_search_consortium.py](file:///e:/latticememory/examples/federated_search_consortium.py)
* Initialize two independent LatticeMemory instances (representing Hospital A and Hospital B) sharing the same model family.
* Hospital A indexes a set of anonymized patient records locally.
* Hospital B queries Hospital A's index by sending **only** a 128-byte E8 snapped key, sharing no raw vectors, text, or central index nodes.
* Hospital A resolves the query locally via $O(1)$ hash lookup and returns matched document IDs.

---

### 2. Multimodal Aligned Training Demo

We will create a script demonstrating cross-modal snapping:

#### [NEW] [examples/multimodal_alignment_demo.py](file:///e:/latticememory/examples/multimodal_alignment_demo.py)
* Simulate unaligned multimodal embeddings (e.g. image and text vectors from CLIP that are close in cosine distance but snap to different E8 coordinates).
* Train a contrastive adapter to project the image embeddings directly onto the E8 coordinates of their positive text counterparts.
* Demonstrate that after training, querying with an image snaps to the exact E8 address of its text description, enabling $O(1)$ cross-modal retrieval.

---

## Verification Plan

### Automated Run
* Run `python examples/federated_search_consortium.py` to verify federated query resolution.
* Run `python examples/multimodal_alignment_demo.py` to verify that multimodal contrastive training successfully aligns the image and text coordinates.
