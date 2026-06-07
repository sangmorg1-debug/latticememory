# Hard Near-Miss Block Failure Audit

This audit explains the block-level failures behind
`benchmarks/results/hard_near_miss_recall_zero_fp.json`.

## Command

```powershell
$env:PYTHONPATH="e:\latticememory"
python benchmarks\benchmark_block_failure_audit.py `
  --paraphrases benchmarks\demo_data\hard_near_miss_challenge\heldout_paraphrases.json `
  --near-misses benchmarks\demo_data\hard_near_miss_challenge\heldout_near_misses.json `
  --threshold 102 `
  --output benchmarks\results\hard_near_miss_block_failure_audit.json
```

## Result

Model: `dfrokido/bge-large-e8-snap`

| Metric | Value |
| --- | ---: |
| Threshold | 102 |
| False negatives | 20 |
| Near-miss confusions at threshold | 0 |
| Closest near-misses audited | 10 |

At the zero-FP threshold, the model has no observed held-out near-miss false
positives. The failure is recall: 20 paraphrase pairs remain above the safe
threshold.

## Blocks That Flip on False Negatives

These blocks differed in all 20 false-negative paraphrase pairs:

| Block | Dimensions | Count |
| ---: | --- | ---: |
| 9 | 72-79 | 20 |
| 21 | 168-175 | 20 |
| 31 | 248-255 | 20 |
| 52 | 416-423 | 20 |
| 71 | 568-575 | 20 |
| 84 | 672-679 | 20 |
| 96 | 768-775 | 20 |
| 97 | 776-783 | 20 |
| 111 | 888-895 | 20 |
| 121 | 968-975 | 20 |

Interpretation: the false negatives are not caused by one isolated bad block.
Many blocks are unstable under paraphrase wording changes. Exact snapping will
need broad block stabilization or a target-cell/soft-to-hard quantizer objective,
not a small local patch.

## Blocks That Stay Same in the Closest Near-Misses

The closest near-miss pairs are the dangerous examples: they are not false
positives at threshold 102, but they are nearest to becoming false positives.
These blocks remained identical most often across the 10 closest near-misses:

| Block | Dimensions | Count |
| ---: | --- | ---: |
| 80 | 640-647 | 8 |
| 76 | 608-615 | 5 |
| 92 | 736-743 | 4 |
| 110 | 880-887 | 4 |
| 116 | 928-935 | 4 |

Interpretation: these blocks may be too generic or too insensitive for the
hard-near-miss distinctions. During exact-snap training, these blocks should not
be blindly pulled together for paraphrases unless near-miss repulsion remains
strong.

## Training Implication

The next exact-snapping experiment should use this audit as a constraint:

- Pull false-negative paraphrases inward by reducing flips across the repeated
  false-negative blocks.
- Keep the closest near-misses separated, especially in blocks that currently
  stay identical too often.
- Track `recall_at_fp=0` after every training epoch.
- Treat any improvement in exact-cell snapping as invalid if near-miss
  confusions appear at the safe threshold.

This points toward canonical-key training with hard near-miss repulsion and
block-level diagnostics, followed by a soft-to-hard quantizer objective.
