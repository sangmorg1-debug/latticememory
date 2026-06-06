# Implementation Plan: Phase 5 (ML Training Data Pipeline)

This plan outlines the steps to build **Phase 5** of the LatticeMemory roadmap: the ML Training Data Pipeline (`LatticeDataPipeline`), facilitating high-throughput deduplication, caption quality filtering, and sharding.

## Goal Description
For modern large-scale dataset ingestion (e.g. text/image pairs, multi-million token text corpuses), preprocessing is a bottleneck. Snapping text representations to E8 coordinates allows:
1. $O(1)$ duplicate checking via hash sets.
2. Caption quality gating by calculating E8 address consistency.
3. Multi-worker dataset sharding by hashing the canonical E8 addresses to worker IDs.

## Proposed Changes

### `latticememory/pipeline.py`
* Implement `LatticeDataPipeline` with:
  - `deduplicate_text(texts, threshold)`: deduplicates list of texts by checking for matching E8 keys.
  - `fit_cross_modal_adapter(pairs)`: aligns multimodal spaces.
  - `filter_caption_quality(image_embs, caption_embs)`: filters low-quality text captions based on coordinate distance.
  - `assign_shards(embeddings, num_shards)`: assigns embeddings to deterministic shards based on their E8 addresses.

### `latticememory/integrations/hf_datasets.py`
* Implement `LatticeDatasetFilter` wrapping HuggingFace's `datasets.Dataset` streaming pipelines for E8 deduplication and quality filtering.

## Verification Plan
* Create validation/demo scripts in `examples/`:
  - `examples/training_data_dedup_demo.py`
  - `examples/caption_quality_filter_demo.py`
  - `examples/distributed_shard_demo.py`
* Verify via focused tests: `tests/test_pipeline.py` and `tests/test_hf_datasets_integration.py`.
