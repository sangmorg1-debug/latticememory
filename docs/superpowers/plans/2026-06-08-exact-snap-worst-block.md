# Exact Snap Worst-Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a block-level exactness audit and worst-block/focal soft-to-hard training objective to push exact E8 snapping while preserving HammingRouter safety metrics.

**Architecture:** Add one benchmark/audit script that evaluates saved checkpoints at per-block granularity, then extend `E8RoutingLoss` and `SnapTrainer` with optional focal/worst-block weighting. Keep all new training behavior default-off and expose it through the existing CLI and JSON telemetry.

**Tech Stack:** Python, PyTorch, SentenceTransformers, existing `E8RoutingLoss`, `SnapTrainer`, `E8LatticeDB`, pytest.

---

### Task 1: Add Block Exactness Audit

**Files:**
- Create: `benchmarks/benchmark_exact_snap_block_audit.py`
- Test: `tests/test_benchmarks.py`

- [ ] **Step 1: Write the failing audit schema test**

Add this test to `tests/test_benchmarks.py`:

```python
def test_exact_snap_block_audit_schema():
    from benchmarks.benchmark_exact_snap_block_audit import summarize_exact_snap_blocks

    query_keys = [[1, 2, 3, 4], [1, 2, 0, 4]]
    target_keys = [[1, 9, 3, 4], [1, 2, 3, 5]]
    target_probs = [[0.9, 0.1, 0.8, 0.7], [0.95, 0.92, 0.2, 0.3]]

    report = summarize_exact_snap_blocks(
        query_keys=query_keys,
        target_keys=target_keys,
        target_probabilities=target_probs,
        cluster_name="refund_policy",
    )

    assert report["cluster_name"] == "refund_policy"
    assert report["n_pairs"] == 2
    assert report["exact_same_key_rate"] == 0.0
    assert report["mean_hamming"] == 1.5
    assert report["mean_correct_blocks"] == 2.5
    assert report["worst_blocks"][0]["block"] in {1, 2, 3}
    assert "repeated_wrong_blocks" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_benchmarks.py::test_exact_snap_block_audit_schema -q
```

Expected: import failure because `benchmarks.benchmark_exact_snap_block_audit` does not exist.

- [ ] **Step 3: Create the audit implementation**

Create `benchmarks/benchmark_exact_snap_block_audit.py` with:

```python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from latticememory.rag.e8_retriever import E8LatticeDB


def summarize_exact_snap_blocks(
    *,
    query_keys: Sequence[Sequence[int]],
    target_keys: Sequence[Sequence[int]],
    target_probabilities: Sequence[Sequence[float]],
    cluster_name: str,
) -> dict:
    if len(query_keys) != len(target_keys) or len(query_keys) != len(target_probabilities):
        raise ValueError("query_keys, target_keys, and target_probabilities must have the same length")
    if not query_keys:
        raise ValueError("at least one key pair is required")

    n_blocks = len(query_keys[0])
    wrong_counter: Counter[int] = Counter()
    low_prob_rows: list[tuple[int, float]] = []
    hamming_values: list[int] = []
    correct_counts: list[int] = []

    for q_key, t_key, probs in zip(query_keys, target_keys, target_probabilities):
        if len(q_key) != n_blocks or len(t_key) != n_blocks or len(probs) != n_blocks:
            raise ValueError("all keys and probability rows must have the same block count")
        diff = [idx for idx, (q, t) in enumerate(zip(q_key, t_key)) if int(q) != int(t)]
        wrong_counter.update(diff)
        for idx, prob in enumerate(probs):
            low_prob_rows.append((idx, float(prob)))
        hamming_values.append(len(diff))
        correct_counts.append(n_blocks - len(diff))

    low_prob_rows.sort(key=lambda row: row[1])
    exact_count = sum(1 for h in hamming_values if h == 0)
    return {
        "cluster_name": cluster_name,
        "n_pairs": len(query_keys),
        "n_blocks": n_blocks,
        "exact_same_key_rate": round(exact_count / len(query_keys), 4),
        "mean_hamming": round(float(np.mean(hamming_values)), 4),
        "mean_correct_blocks": round(float(np.mean(correct_counts)), 4),
        "worst_blocks": [
            {"block": int(block), "mean_target_probability": round(float(prob), 6)}
            for block, prob in low_prob_rows[:20]
        ],
        "repeated_wrong_blocks": [
            {"block": int(block), "wrong_count": int(count)}
            for block, count in wrong_counter.most_common(20)
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_benchmarks.py::test_exact_snap_block_audit_schema -q
```

Expected: `1 passed`.

- [ ] **Step 5: Add CLI audit path**

Extend `benchmarks/benchmark_exact_snap_block_audit.py` with:

```python
def _block_target_probabilities(embeddings: torch.Tensor, target_keys: torch.Tensor, codebook: torch.Tensor, temperature: float) -> torch.Tensor:
    n_blocks = embeddings.shape[1] // 8
    blocks = F.normalize(embeddings.reshape(embeddings.shape[0], n_blocks, 8), dim=-1)
    code_vectors = F.normalize(codebook.to(embeddings.device), dim=-1)
    logits = torch.einsum("bnd,kd->bnk", blocks, code_vectors) / float(temperature)
    probs = F.softmax(logits, dim=-1)
    return probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)


def _keys_for(lattice: E8LatticeDB, rows) -> list[list[int]]:
    return [list(lattice._quantize_to_indices(row)) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact E8 block failures for validation clusters")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmarks/results/exact_snap_block_audit.json")
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from examples.train_snap_encoder import VAL_CLUSTERS

    encoder = SentenceTransformer(args.model)
    d_model = int(encoder.get_sentence_embedding_dimension())
    lattice = E8LatticeDB(d_model=d_model)
    codebook = lattice._codebook.float()

    cluster_reports = []
    for cluster_name, texts in VAL_CLUSTERS.items():
        target = texts[0]
        queries = texts[1:]
        q_emb = torch.tensor(encoder.encode(queries, normalize_embeddings=True), dtype=torch.float32)
        t_emb = torch.tensor(encoder.encode([target] * len(queries), normalize_embeddings=True), dtype=torch.float32)
        q_keys = _keys_for(lattice, q_emb)
        t_keys = _keys_for(lattice, t_emb)
        target_key_tensor = torch.tensor(t_keys, dtype=torch.long)
        target_probs = _block_target_probabilities(q_emb, target_key_tensor, codebook, args.temperature)
        cluster_reports.append(
            summarize_exact_snap_blocks(
                query_keys=q_keys,
                target_keys=t_keys,
                target_probabilities=target_probs.tolist(),
                cluster_name=cluster_name,
            )
        )

    report = {
        "artifact_type": "latticememory_exact_snap_block_audit",
        "artifact_version": 1,
        "model": args.model,
        "d_model": d_model,
        "temperature": args.temperature,
        "clusters": cluster_reports,
        "mean_exact_same_key_rate": round(float(np.mean([c["exact_same_key_rate"] for c in cluster_reports])), 4),
        "mean_hamming": round(float(np.mean([c["mean_hamming"] for c in cluster_reports])), 4),
        "mean_correct_blocks": round(float(np.mean([c["mean_correct_blocks"] for c in cluster_reports])), 4),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run audit on the latest soft-to-hard checkpoint**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python benchmarks\benchmark_exact_snap_block_audit.py --model benchmarks\results\snap_soft_hard_from_best_1ep\best_snap_encoder --output benchmarks\results\exact_snap_block_audit_soft_hard_best.json
```

Expected: JSON file exists and reports per-cluster `mean_correct_blocks` and `repeated_wrong_blocks`.

- [ ] **Step 7: Commit audit**

Run:

```powershell
git add benchmarks\benchmark_exact_snap_block_audit.py tests\test_benchmarks.py benchmarks\results\exact_snap_block_audit_soft_hard_best.json
git commit -m "bench: add exact snap block audit"
```

### Task 2: Add Worst-Block Focal Loss

**Files:**
- Modify: `latticememory/training.py`
- Test: `tests/test_training_pipeline.py`

- [ ] **Step 1: Write failing loss test**

Add this test to `tests/test_training_pipeline.py`:

```python
def test_e8_routing_loss_focal_soft_hard_emphasizes_low_probability_blocks():
    torch.manual_seed(44)
    query_embeddings = torch.randn(4, 16, requires_grad=True)
    positive_embeddings = query_embeddings.detach().clone() + 0.01 * torch.randn(4, 16)

    base_loss = E8RoutingLoss(
        d_model=16,
        lambda_soft_hard=1.0,
        soft_hard_temperature=0.2,
        soft_hard_focal_gamma=0.0,
        soft_hard_top_k_blocks=0,
    )
    focal_loss = E8RoutingLoss(
        d_model=16,
        lambda_soft_hard=1.0,
        soft_hard_temperature=0.2,
        soft_hard_focal_gamma=2.0,
        soft_hard_top_k_blocks=1,
    )

    base = base_loss(query_embeddings, positive_embeddings, None)
    focal = focal_loss(query_embeddings, positive_embeddings, None)

    assert focal.soft_hard.item() >= 0.0
    assert focal.target_cell_probability.item() >= 0.0
    assert focal.total.item() != base.total.item()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_training_pipeline.py::test_e8_routing_loss_focal_soft_hard_emphasizes_low_probability_blocks -q
```

Expected: `TypeError` for unknown `soft_hard_focal_gamma`.

- [ ] **Step 3: Extend `E8RoutingLoss` constructor**

In `latticememory/training.py`, add arguments:

```python
soft_hard_focal_gamma: float = 0.0,
soft_hard_top_k_blocks: int = 0,
```

Store them:

```python
self.soft_hard_focal_gamma = float(soft_hard_focal_gamma)
self.soft_hard_top_k_blocks = int(soft_hard_top_k_blocks)
```

- [ ] **Step 4: Replace `_soft_hard_loss` reduction**

Inside `_soft_hard_loss`, replace:

```python
target_log_probs = log_probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
target_probs = probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
soft_hard = -target_log_probs.mean()
```

with:

```python
target_log_probs = log_probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
target_probs = probs.gather(2, target_keys.unsqueeze(-1)).squeeze(-1)
per_block_loss = -target_log_probs
if self.soft_hard_focal_gamma > 0:
    per_block_loss = per_block_loss * (1.0 - target_probs).pow(self.soft_hard_focal_gamma)
if self.soft_hard_top_k_blocks > 0:
    k = min(int(self.soft_hard_top_k_blocks), per_block_loss.shape[1])
    soft_hard = per_block_loss.topk(k, dim=1).values.mean()
else:
    soft_hard = per_block_loss.mean()
```

- [ ] **Step 5: Run loss test to verify it passes**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_training_pipeline.py::test_e8_routing_loss_focal_soft_hard_emphasizes_low_probability_blocks -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit focal loss**

Run:

```powershell
git add latticememory\training.py tests\test_training_pipeline.py
git commit -m "feat: add focal soft-to-hard block loss"
```

### Task 3: Wire Focal Loss Into SnapTrainer and CLI

**Files:**
- Modify: `latticememory/snap_trainer.py`
- Modify: `examples/train_snap_encoder.py`
- Test: `tests/test_snap_trainer.py`

- [ ] **Step 1: Add config tests**

Add to `tests/test_snap_trainer.py`:

```python
def test_config_soft_hard_focal_override():
    cfg = SnapTrainingConfig(
        soft_hard_focal_gamma=2.0,
        soft_hard_top_k_blocks=8,
    )
    assert cfg.soft_hard_focal_gamma == 2.0
    assert cfg.soft_hard_top_k_blocks == 8
```

- [ ] **Step 2: Run config test to verify it fails**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_snap_trainer.py::test_config_soft_hard_focal_override -q
```

Expected: `TypeError` for unknown config fields.

- [ ] **Step 3: Add config fields and pass-through**

Add to `SnapTrainingConfig` in `latticememory/snap_trainer.py`:

```python
soft_hard_focal_gamma: float = 0.0
soft_hard_top_k_blocks: int = 0
```

Pass into `E8RoutingLoss`:

```python
soft_hard_focal_gamma=config.soft_hard_focal_gamma,
soft_hard_top_k_blocks=config.soft_hard_top_k_blocks,
```

Add these fields to `training_init` and `snap_training_summary.json`.

- [ ] **Step 4: Add CLI arguments**

Add to `examples/train_snap_encoder.py`:

```python
p.add_argument("--soft-hard-focal-gamma", type=float, default=0.0)
p.add_argument("--soft-hard-top-k-blocks", type=int, default=0)
```

Pass them into `SnapTrainingConfig` and `training_start` JSON.

- [ ] **Step 5: Run trainer config test**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests\test_snap_trainer.py::test_config_soft_hard_focal_override tests\test_snap_trainer.py::test_train_writes_summary_json -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit trainer wiring**

Run:

```powershell
git add latticememory\snap_trainer.py examples\train_snap_encoder.py tests\test_snap_trainer.py
git commit -m "feat: wire focal soft-to-hard training controls"
```

### Task 4: Run Gated Focal Experiment

**Files:**
- Generated: `benchmarks/results/snap_focal_soft_hard_1ep/`

- [ ] **Step 1: Launch one-epoch focal run from the prior best checkpoint**

Run:

```powershell
$out="e:\latticememory\benchmarks\results\snap_focal_soft_hard_1ep"
python examples\train_snap_encoder.py `
  --base-model e:\latticememory\benchmarks\results\snap_soft_hard_from_best_1ep\best_snap_encoder `
  --data banking77+val `
  --limit 5000 `
  --epochs 1 `
  --batch-size 4 `
  --grad-accum 1 `
  --lr 7e-6 `
  --lambda-address 0 `
  --lambda-address-hinge 0.5 `
  --lambda-address-mse 0.5 `
  --lambda-soft-hard 3.0 `
  --soft-hard-temperature-start 0.12 `
  --soft-hard-temperature-end 0.03 `
  --soft-hard-straight-through `
  --soft-hard-focal-gamma 2.0 `
  --soft-hard-top-k-blocks 16 `
  --lambda-hamming 0.05 `
  --lambda-negative 1.0 `
  --lambda-near-miss 2.0 `
  --near-miss-margin 80 `
  --freeze-layers 20 `
  --fragmentation-target 0.75 `
  --separation-target 0.80 `
  --output $out `
  --device cuda
```

- [ ] **Step 2: Monitor safety**

Watch:

```powershell
Get-Content benchmarks\results\snap_focal_soft_hard_1ep\snap_progress.jsonl -Tail 20
```

Stop the run if either condition repeats for two checkpoints:

- `zero_fp_recall < 0.8056`
- `is_collapsed == true`

- [ ] **Step 3: Evaluate final artifact**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python benchmarks\benchmark_exact_snap_block_audit.py --model benchmarks\results\snap_focal_soft_hard_1ep\best_snap_encoder --output benchmarks\results\exact_snap_block_audit_focal_best.json
```

Expected: JSON file with exactness and worst-block data for the focal best checkpoint.

- [ ] **Step 4: Commit result if useful**

Commit only if the run preserves the safety gate:

```powershell
git add benchmarks\results\exact_snap_block_audit_focal_best.json
git commit -m "bench: record focal exact snap audit"
```

### Task 5: Final Verification

**Files:**
- No new files unless documentation needs a result note.

- [ ] **Step 1: Run full tests**

Run:

```powershell
$env:PYTHONPATH="e:\latticememory"; python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Summarize decision**

Report:

- whether `mean_fragmentation_score` moved above `0.0`
- whether `target_cell_probability_recent` improved over `0.5207`
- whether `zero_fp_recall` stayed above `0.8056`
- whether the audit shows a narrow or broad set of failing blocks

- [ ] **Step 3: Commit docs if updated**

If a result note is added, run:

```powershell
git add docs
git commit -m "docs: summarize focal exact snap experiment"
```
