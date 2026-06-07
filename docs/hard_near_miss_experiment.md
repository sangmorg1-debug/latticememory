# Hard Near-Miss Experiment

This experiment is the first strict safety gate for HammingRouter and future
exact-snapping training. It tests whether paraphrases can be recovered while
adjacent-but-different support intents remain outside the serving threshold.

## Dataset

The built-in challenge lives in:

- `benchmarks/demo_data/hard_near_miss_challenge/hard_near_miss_source.json`
- `benchmarks/demo_data/hard_near_miss_challenge/calibration_data.json`
- `benchmarks/demo_data/hard_near_miss_challenge/heldout_paraphrases.json`
- `benchmarks/demo_data/hard_near_miss_challenge/heldout_near_misses.json`

The challenge includes close pairs such as:

- cancel subscription vs pause subscription
- refund request vs return status
- reset password vs change email
- shipping status vs delivery address change
- invoice copy vs update payment method
- delete account vs deactivate account

These pairs are intentionally harder than random negatives. They represent the
false-positive failures a semantic cache must avoid.

## Command

```powershell
$env:PYTHONPATH="e:\latticememory"
python benchmarks\hard_near_miss_challenge.py --output-dir benchmarks\demo_data\hard_near_miss_challenge
python benchmarks\benchmark_recall_zero_fp.py `
  --calibration-data benchmarks\demo_data\hard_near_miss_challenge\calibration_data.json `
  --paraphrases benchmarks\demo_data\hard_near_miss_challenge\heldout_paraphrases.json `
  --near-misses benchmarks\demo_data\hard_near_miss_challenge\heldout_near_misses.json `
  --output benchmarks\results\hard_near_miss_recall_zero_fp.json
```

## Result

Model: `dfrokido/bge-large-e8-snap`

| Metric | Value |
| --- | ---: |
| Held-out paraphrase pairs | 36 |
| Held-out near-miss pairs | 30 |
| Threshold at FP = 0 | 102 |
| Recall at FP = 0 | 0.4444 |
| Recall at FP <= 0.001 | 0.4444 |
| Recall at FP <= 0.01 | 0.4444 |
| Hamming gap, near-miss p5 - paraphrase p95 | -8.7 |
| Paraphrase mean Hamming | 102.72 |
| Near-miss mean Hamming | 116.0 |

## Interpretation

The current snap encoder has useful separation: at threshold 102 it recovers
44.44% of hard held-out paraphrases with zero observed near-miss false
positives.

It is not yet enough for the breakthrough claim. The gap is negative because
the held-out paraphrase p95 is 114.5 while the near-miss p5 is 105.8. In plain
terms: some correct paraphrases are farther away than some dangerous near
misses. Raising the threshold to recover those paraphrases would introduce
false positives.

## Training Implication

This result gives the next training target:

- Pull hard paraphrase false negatives below threshold 102.
- Keep near-miss examples above threshold 102.
- Track recall at FP = 0 as the primary safety metric.
- Only treat exact snapping as progress if this safety metric does not regress.

The next experiment should add block-level failure audit on these false
negatives and near misses, then use those blocks as targets for canonical-key
or soft-to-hard quantizer training.
