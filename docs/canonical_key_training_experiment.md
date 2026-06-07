# Canonical-Key Projection Training Experiment

This experiment tests whether canonical-key supervision can improve the hard
near-miss safety benchmark without modifying the full sentence encoder.

The setup keeps `dfrokido/bge-large-e8-snap` frozen and trains a small linear
projection head over embeddings. Same-intent prompts are trained toward the
canonical prompt's E8 key. Calibration near-misses are pushed apart with a
differentiable expected pair-Hamming repulsion term. Evaluation is performed on
held-out paraphrases and held-out near-misses.

## Command

```powershell
$env:PYTHONPATH="e:\latticememory"
python benchmarks\benchmark_canonical_key_training.py `
  --epochs 30 `
  --lambda-near 0.2 `
  --near-margin 112 `
  --output benchmarks\results\canonical_key_training_pairrepel_30ep.json
```

## Result

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Paraphrase mean Hamming | 102.722 | 74.333 |
| Near-miss mean Hamming | 116.000 | 107.567 |
| Exact same-cell rate | 0.0000 | 0.0000 |
| Recall at original threshold 102 | 0.4444 | 0.9167 |
| Near-miss confusion at threshold 102 | 0.0000 | 0.2333 |
| Best zero-FP threshold | 102 | 93 |
| Recall at FP = 0 | 0.4444 | 0.8056 |
| FP rate at zero-FP threshold | 0.0000 | 0.0000 |

## Interpretation

Canonical-key projection training produced a real improvement on the safety
metric that matters most:

```text
recall at FP = 0: 0.4444 -> 0.8056
```

This means the model can be pushed substantially closer to safe Hamming
routing, as long as the serving threshold is recalibrated after training.

The fixed original threshold 102 became unsafe after training because near-miss
distances also moved closer. That is expected: the projection changes the
geometry, so the old threshold cannot be reused.

Exact same-cell snapping did not improve:

```text
exact_same_cell_rate: 0.0 -> 0.0
```

So this is not the exact-snap breakthrough yet. It is evidence that direct
canonical E8 supervision can improve the safe Hamming routing frontier.

## Next Training Step

The next experiment should replace the plain linear projection with a
soft-to-hard quantizer objective:

- track per-block target-cell probability
- anneal block softmax temperature downward
- optionally use straight-through argmax for hard key assignment
- keep recall-at-zero-FP as the safety gate
- reject any exact-snap gain that increases held-out near-miss false positives

The key target is now:

```text
increase exact_same_cell_rate above 0 while preserving recall_at_FP=0 >= 0.8056
```
