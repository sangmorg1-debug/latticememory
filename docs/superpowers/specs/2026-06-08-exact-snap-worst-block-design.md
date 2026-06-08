# Exact Snap Worst-Block Design

## Goal

Improve the exact E8 snapping research path by identifying which blocks prevent full-key agreement, then training against those low-confidence or wrong blocks without sacrificing the calibrated near-zero-false-positive HammingRouter product path.

## Current Evidence

The latest soft-to-hard run showed useful movement but did not solve exact snapping:

- `target_cell_probability_recent` rose from about `0.0183` to `0.5207`.
- `zero_fp_recall` stayed safely above the prior `0.8056` gate and ended at `0.94`.
- `separation_score` stayed `1.0`.
- `mean_fragmentation_score` stayed `0.0`.

This means the model can learn per-block target confidence, but exact snapping requires all 128 blocks to agree at once. The next experiment must stop averaging away the hard blocks.

## Product Interpretation

This work supports two product claims separately:

1. Calibrated HammingRouter caching is the near-term product path. It can claim safe approximate semantic cache hits under a calibrated false-positive budget.
2. Exact same-cell snapping remains a research path. It should not be represented as solved until validation clusters produce nonzero exact fragmentation while preserving near-miss safety.

## Proposed Approach

Add a block-level exactness audit for a trained checkpoint and validation cluster set. The audit must report, per cluster and per pair:

- exact same-key rate
- full-key Hamming distance
- number of correct target blocks
- worst blocks by target probability
- repeated wrong blocks across pairs

Then add a worst-block/focal extension to the soft-to-hard loss. Instead of averaging all blocks equally, the loss focuses on blocks with low target-cell probability. This keeps pressure on the blocks that prevent all-128-block agreement.

## Safety Gates

Product usefulness is gated by safe HammingRouter behavior:

- `zero_fp_recall >= 0.8056`
- `separation_score >= 0.8`
- `is_collapsed == false`
- `mean_inter_block_nmi` does not spike into collapse-like behavior

If exact snapping improves but these gates fail, reject the run.
`mean_fragmentation_score` remains research telemetry, not the product pass/fail
gate.

## Success Criteria

Minimum useful result:

- Produce a JSON audit showing whether exact snapping is blocked by a small or broad set of blocks.
- Produce a gated training run where target-cell probability improves without reducing `zero_fp_recall` below `0.8056`.

Breakthrough result:

- `mean_fragmentation_score > 0.0` on the validation clusters.
- Near-miss safety remains above gate.

## Non-Goals

- Do not claim exact snapping is solved from target probability alone.
- Do not train on asymmetric MS MARCO for this specific experiment.
- Do not relax the false-positive safety gate to make exact snapping look better.
