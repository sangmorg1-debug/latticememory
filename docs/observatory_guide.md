# LatticeObservatory — AI User Guide

This guide is written for an AI agent (IDE assistant, reasoning model, or autonomous optimizer)
operating on the LatticeMemory codebase. It covers every method in `LatticeObservatory`,
what each output means, how to interpret it, and the exact sequence of calls to run in order
to diagnose and fix an encoder.

---

## What the Observatory Is

The E8 lattice converts a 1024-dimensional (or 384d, 768d) float32 embedding into a
128-byte (or 48-byte, 96-byte) key. One byte per 8D block. Each byte is an address in
{0…239} — the 240 Shell-1 vectors of the E8 lattice.

A well-trained encoder produces keys where:
- Semantically equivalent texts → same key (O(1) cache hit)
- Semantically distinct texts → different keys (no collision)
- Semantically adjacent texts → keys differing in ≤1 block (Hamming-1 neighborhood hit)

The Observatory makes the block-level structure fully readable so an AI can identify
exactly which 8D subspaces are working, which are failing, and what training data
would fix the failures.

---

## The Full Optimization Loop

```
1. export_for_llm()          →  global snapshot, feed to reasoning model
2. collision_audit()         →  find cells where false cache hits occur
3. semantic_probe()          →  map blocks to semantic dimensions
4. block_correlation()       →  find redundant/entangled blocks
5. address_trajectory()      →  test encoder continuity on paraphrase chains
6. fragmentation_score()     →  confirm a concept cluster is E8-capturable
7. routing_profile()         →  workload-specific block-level miss map
8. suggest_training_pairs()  →  targeted pairs for noisiest blocks
9. generate_training_curriculum()  →  full training plan + loss config
   [run training]
10. compare_snapshots()      →  verify improvement, close the loop
```

Run steps 1–7 to understand the problem. Run steps 8–9 to build the fix.
Run step 10 after training to verify.

---

## Setup

```python
from latticememory import LatticeIndex
from latticememory.observatory import LatticeObservatory

# If you already have a populated index:
obs = index.observatory()

# Or construct directly:
obs = LatticeObservatory(index)
```

All methods are non-mutating. Nothing is written to the index.

---

## Method Reference

---

### `export_for_llm(n_sample_cells=5) -> dict`

**The first call you should make.** Returns a self-describing JSON snapshot of the
entire index. Feed this directly into a reasoning model (GPT-4o, Claude Opus, Gemini)
with a prompt like: *"Here is a LatticeMemory index snapshot. Diagnose the routing
health and recommend which methods to run next."*

**Key fields:**

| Field | What it tells you |
|---|---|
| `index_summary.total_docs` | How many documents are indexed |
| `index_summary.unique_keys` | How many distinct E8 cells are occupied |
| `index_summary.singleton_keys` | Keys with exactly 1 doc — no cache hits possible |
| `block_analysis.mean_entropy` | 0.0 = all docs in one cell (collapse). ~7.9 = fully random |
| `block_analysis.most_stable_blocks` | Blocks that never change — the semantic fingerprint |
| `block_analysis.most_variable_blocks` | Blocks that always vary — routing failure sources |
| `sample_cells` | Coherence of the largest cells (`tight`/`loose`/`collision`/`singleton`) |
| `recommendations` | Auto-generated actionable findings |

**Healthy index looks like:**
- `mean_entropy` between 1.5 and 4.0 (some structure, not collapsed, not random)
- `sample_cells` all labeled `tight` or `loose`
- `recommendations` returns "No critical issues detected"

**Unhealthy index looks like:**
- `mean_entropy` near 0.0 → key collapse (encoder training failure)
- `mean_entropy` near 7.9 → random routing (encoder not aligned with E8)
- `sample_cells` with `coherence_label: "collision"` → false cache hits

```python
snapshot = obs.export_for_llm(n_sample_cells=10)
import json
print(json.dumps(snapshot, indent=2))
# → paste into your reasoning model
```

---

### `collision_audit() -> dict`

**Run this immediately after `export_for_llm` if any sample cells show collision.**
Scans every cell in the index, ranks by collision risk (worst coherence first).

**Key fields:**

| Field | What it tells you |
|---|---|
| `collision_cells` | Count of cells where mean cosine < 0.80 — false hits guaranteed |
| `loose_cells` | Count of cells where mean cosine 0.80–0.95 — monitor |
| `tight_cells` | Count of cells where mean cosine ≥ 0.95 — safe |
| `collision_rate` | `collision_cells / multi_doc_cells` — your false hit rate |
| `ranked_by_risk` | All multi-doc cells, worst first |

**What to do with results:**
- `collision_rate > 0.1` → add a post-retrieval cosine threshold guard (score ≥ 0.80)
  before returning cached results
- `collision_rate > 0.3` → the encoder needs block-targeted fine-tuning; run
  `semantic_probe` + `generate_training_curriculum` to build the fix
- Take the top 3 entries in `ranked_by_risk` and run `trace_mismatch` on pairs from
  those cells to see exactly which blocks are causing the collision

```python
audit = obs.collision_audit()
print(f"Collision rate: {audit['collision_rate']:.1%}")
print(f"Worst cell: {audit['ranked_by_risk'][0]}")
```

---

### `semantic_probe(labeled_texts: dict[str, list[str]]) -> dict`

**The JEPA-style interpretability call.** Given labeled concept categories, computes
mutual information between each E8 block's address distribution and the label.
Tells you which 8D subspaces encode TOPIC, SENTIMENT, DOMAIN, or any other
semantic dimension you care about.

**Key fields:**

| Field | What it tells you |
|---|---|
| `top_separating_blocks` | Blocks with highest info gain — prime fine-tuning targets |
| `bottom_separating_blocks` | Blocks with near-zero info gain — freeze during training |
| `fine_tune_targets` | Dim ranges to unfreeze (e.g. `"96-103"`) |
| `freeze_candidates` | Dim ranges to freeze — they carry other signal |
| `label_entropy` | Max possible info gain. High = labels are well-distributed |
| `all_blocks[i].separability` | 0.0–1.0. Fraction of label entropy explained by block i |

**How to read it:**
A block with `separability: 0.94` means 94% of the label distinction is captured in
that single 8D subspace. Fine-tuning only that block's dimensions would be nearly
sufficient to align the encoder to this semantic distinction.

A block with `separability: 0.001` is essentially random noise for this label. Leave
it frozen so you don't corrupt whatever it's doing for other tasks.

```python
result = obs.semantic_probe({
    "financial":  ["earnings report Q3", "stock market rally", "interest rate hike"],
    "medical":    ["clinical trial results", "FDA approval", "patient outcomes study"],
    "technology": ["neural network architecture", "GPU memory bandwidth", "API latency"],
})
print("Fine-tune these dims:", result["fine_tune_targets"])
print("Freeze these dims:", result["freeze_candidates"])
```

**Rule of thumb:** If `top_separating_blocks[0].separability < 0.3`, the encoder has
no stable subspace for this distinction. You will need new training data, not just
fine-tuning of existing weights.

---

### `block_correlation(texts: list[str]) -> dict`

**Finds redundant and entangled blocks.** Computes normalized mutual information (NMI)
between every pair of blocks across the text corpus. NMI = 1.0 means two blocks are
carrying identical information. NMI = 0.0 means they're fully independent.

**Key fields:**

| Field | What it tells you |
|---|---|
| `redundant_pairs` | Block pairs with NMI > 0.9 — one is waste |
| `high_correlation_pairs` | Block pairs with NMI > 0.5 — potential entanglement |
| `hub_blocks` | Blocks with high mean NMI — correlated with many others |
| `mean_inter_block_nmi` | Global redundancy metric. < 0.1 = independent, > 0.4 = entangled |

**What to do with results:**
- `redundant_pairs` with NMI > 0.9: when fine-tuning, only touch one block from each
  pair. Training both will cause them to oscillate — the gradient pushes one, which
  pulls the other back.
- `hub_blocks` (high mean NMI): these blocks are dominated by a single surface feature
  (e.g., document length, punctuation density, whitespace patterns). If they correlate
  with your top `semantic_probe` targets, you have a confound — the encoder is using
  surface features as a shortcut. Fix: add training examples that share surface
  features but differ in semantics.

```python
corr = obs.block_correlation(corpus_texts)
print(f"Redundant pairs: {len(corr['redundant_pairs'])}")
for pair in corr["redundant_pairs"][:3]:
    print(f"  Blocks {pair['block_i']} ↔ {pair['block_j']}: NMI={pair['nmi']}")
```

---

### `address_trajectory(text_sequence: list[str]) -> dict`

**Tests encoder continuity on an ordered sequence.** JEPA framing: a well-calibrated
encoder should represent a paraphrase chain as a smooth path through E8 space —
each step changes only 1–2 blocks, not all 48. Large jumps reveal where the encoder
is discontinuous.

**Key fields:**

| Field | What it tells you |
|---|---|
| `mean_hamming_per_step` | Avg blocks changed per step. ≤1 = smooth, ≥10 = discontinuous |
| `trajectory_continuity` | Fraction of steps that are `exact` or `hamming1` |
| `most_volatile_blocks` | Blocks that change most frequently across steps |
| `steps[i].routing_continuity` | `exact` / `hamming1` / `discontinuous` for each step |
| `training_recommendation` | Auto-generated action if discontinuous |

**How to use it:**

Build a paraphrase chain for your target domain — a sequence of rephrased sentences
that all mean the same thing but vary in surface form:

```python
paraphrase_chain = [
    "The quarterly earnings exceeded analyst expectations.",
    "Q3 profits surpassed what analysts had forecast.",
    "The company beat earnings estimates for the third quarter.",
    "Third-quarter results came in above Wall Street projections.",
    "Earnings for Q3 were better than expected.",
]
traj = obs.address_trajectory(paraphrase_chain)
print(f"Mean Hamming per step: {traj['mean_hamming_per_step']}")
print(f"Continuity: {traj['trajectory_continuity']:.0%} smooth steps")
print("Most volatile blocks:", traj["most_volatile_blocks"][:3])
```

A good encoder gives `mean_hamming_per_step ≤ 2` on a paraphrase chain.
If you're seeing 15+, the encoder has no stable semantic axis for this concept
and the volatile blocks are the fine-tuning targets.

---

### `fragmentation_score(texts: list[str]) -> dict`

**Before indexing a concept cluster, predict whether E8 routing will capture it.**
Returns a single score: 1.0 = all texts route to the same key (caching will work).
0.0 = every text in its own cell (caching is impossible for this cluster).

**Key fields:**

| Field | What it tells you |
|---|---|
| `score` | 0.0–1.0. The fraction of the cluster that is unified |
| `label` | `unified` / `cohesive` / `fragmented` / `scattered` |
| `cacheable` | `True` if all texts share one key |
| `n_unique_keys` | How many distinct E8 addresses the cluster spans |
| `dominant_key` | The key that the most texts map to |

**Decision rule:**

| Score | Action |
|---|---|
| 1.0 | Safe to cache. E8 exact path will serve all queries in this cluster. |
| 0.75–1.0 | Good. A few texts miss the dominant key; Hamming-1 will catch most. |
| 0.25–0.75 | Run `generate_training_curriculum` to unify this cluster. |
| < 0.25 | Concept is scattered. Dense fallback required until encoder is fixed. |

```python
frag = obs.fragmentation_score([
    "What is the capital of France?",
    "Which city is the capital of France?",
    "Name the French capital city.",
    "France's capital city is called what?",
])
print(f"Score: {frag['score']} ({frag['label']})")
print(f"Cacheable: {frag['cacheable']}")
```

---

### `routing_profile(queries: list[str], docs: list[str]) -> dict`

**Workload-specific routing analysis.** For known (query, doc) pairs that should
match, measures how often E8 routing actually connects them. Identifies which blocks
are responsible for the most misses.

**Key fields:**

| Field | What it tells you |
|---|---|
| `routing.exact_rate` | Fraction of pairs that hit `lattice_exact` (O(1), Hamming=0) |
| `routing.hamming1_rate` | Fraction that hit `lattice_hamming1` (O(1), Hamming=1) |
| `routing.fallback_rate` | Fraction requiring dense fallback (Hamming > 1) |
| `hamming.mean` | Mean Hamming distance across pairs |
| `hamming.p90` | 90th percentile — your "worst normal case" |
| `top_mismatching_blocks` | Blocks responsible for most misses, sorted by `miss_rate` |
| `optimization_target` | Auto-generated dim range string for fine-tuning |

**Interpreting the routing split:**

- `exact_rate ≥ 0.60` → good cache hit rate for paraphrase workloads
- `exact_rate < 0.20` → encoder is not aligned with this workload; run full curriculum
- `hamming.mean > 10` → structural mismatch (asymmetric workload, e.g. question vs. passage)
  — dense fallback is required regardless of encoder quality

**Critical distinction:** If `hamming.mean` is 40–90, this is an **asymmetric workload**
(questions vs. passages). No amount of fine-tuning will close this gap without causing
key collapse. Use the dense fallback. If `hamming.mean` is 5–15, this is a
**symmetric workload with a training gap** — fine-tuning will fix it.

```python
profile = obs.routing_profile(
    queries=["what causes inflation", "define GDP", "how does monetary policy work"],
    docs=["Inflation is caused by...", "GDP stands for...", "Monetary policy refers to..."],
)
print(f"Exact hit rate: {profile['routing']['exact_rate']:.0%}")
print(f"Mean Hamming: {profile['hamming']['mean']}")
print("Top target dims:", profile["optimization_target"])
```

---

### `trace_mismatch(query: str, doc: str) -> dict`

**Block-by-block routing diff for a single pair.** The most granular diagnostic tool.
Shows exactly which 8D subspaces cause a routing failure between a specific query
and document.

**Key fields:**

| Field | What it tells you |
|---|---|
| `hamming_distance` | Number of blocks where the keys differ |
| `routing_verdict` | `lattice_exact` / `lattice_hamming1` / `fallback_required` |
| `cosine_similarity` | Float32 semantic similarity for reference |
| `differing_blocks` | Each differing block with query address, doc address, dim range |
| `matching_blocks` | Each block where they agree |

```python
mismatch = obs.trace_mismatch(
    "What is the unemployment rate?",
    "The unemployment rate measures the fraction of the labor force without jobs.",
)
print(f"Hamming: {mismatch['hamming_distance']}/48  Verdict: {mismatch['routing_verdict']}")
for b in mismatch["differing_blocks"][:5]:
    print(f"  Block {b['block']} (dims {b['dim_range']}): query={b['query_address']} doc={b['doc_address']}")
```

---

### `cell_coherence(key: str | bytes) -> dict`

**Semantic coherence check for a single cell.** Computes pairwise cosine similarity
of all documents sharing a key. Use this when `collision_audit` flags a cell as
suspicious.

**Key fields:**

| Field | What it tells you |
|---|---|
| `coherence_label` | `tight` ≥0.95 / `loose` 0.80–0.95 / `collision` <0.80 / `singleton` |
| `mean_cosine` | Mean pairwise cosine of all docs in the cell |
| `min_cosine` | Worst-case pair — if < 0.70, this is a hard collision |
| `texts` | Sample text previews so you can visually inspect the contents |

```python
# Check the worst cell from collision_audit
worst_key = audit["ranked_by_risk"][0]["key"]
coh = obs.cell_coherence(worst_key)
print(f"Label: {coh['coherence_label']}  Mean cosine: {coh['mean_cosine']}")
print("Texts in this cell:")
for t in coh["texts"]:
    print(f"  - {t}")
```

---

### `block_stability(texts: list[str]) -> dict`

**Per-block entropy for a semantic cluster.** For each of the 48 (or 128) blocks,
measures how consistently all texts in the cluster map to the same E8 address.
Entropy 0.0 = stable fingerprint. High entropy = routing noise source.

**Key fields:**

| Field | What it tells you |
|---|---|
| `concept_fingerprint_blocks` | Blocks where every text shares the same address |
| `noisiest_blocks` | Top unstable blocks sorted by entropy (worst first) |
| `mean_block_entropy` | Global stability — lower is better for caching |
| `fully_stable_block_count` | How many blocks form a perfect fingerprint |

```python
stability = obs.block_stability([
    "machine learning model training",
    "training a neural network",
    "fitting a deep learning model",
    "supervised learning with gradient descent",
])
print(f"Stable blocks: {stability['fully_stable_block_count']}/48")
print(f"Fingerprint blocks: {stability['concept_fingerprint_blocks']}")
print(f"Noisiest block: {stability['noisiest_blocks'][0]}")
```

---

### `neighbor_density(key: str | bytes) -> dict`

**Maps the Hamming-1 neighborhood of a cell.** For each stored key at Hamming distance
exactly 1, reports how many docs live there and whether the neighbor is coherent.

**Key fields:**

| Field | What it tells you |
|---|---|
| `hamming1_neighbor_cells` | How many occupied cells are exactly 1 block away |
| `total_neighbor_docs` | Total docs reachable via Hamming-1 expansion |
| `expansion_verdict` | `safe` / `noisy` / `empty` |
| `noisy_neighbor_cells` | Count of collision-labeled neighbors |
| `recall_estimate` | Auto-generated description of expansion impact |

Use this when deciding whether to enable Hamming-1 routing for a specific cell.
If `expansion_verdict == "noisy"`, expanding to Hamming-1 will return unrelated
documents. Add a post-retrieval cosine guard or restrict to `lattice_exact` for
that key.

```python
density = obs.neighbor_density(my_key)
print(f"Expansion verdict: {density['expansion_verdict']}")
print(f"Adding Hamming-1 would add {density['total_neighbor_docs']} docs")
```

---

### `suggest_training_pairs(pairs: list[tuple[str, str]]) -> dict`

**Generates targeted contrastive training pairs for the noisiest blocks.**
Given (query, doc) pairs that should route to the same key but currently don't,
returns structured training data and a fine-tuning recipe.

**Key fields:**

| Field | What it tells you |
|---|---|
| `training_pairs` | List of `{anchor, positive, hamming_distance, target_blocks, target_dims}` |
| `top_target_blocks` | Block indices most responsible for failures |
| `top_target_dims` | Dim ranges to unfreeze during fine-tuning |
| `training_recipe.loss` | `E8RoutingLoss` from `latticememory.training` |
| `training_recipe.freeze_strategy` | Which dims to freeze vs. unfreeze |
| `training_recipe.n_pairs_needed` | Recommended training set size |

```python
pairs = obs.suggest_training_pairs([
    ("how does inflation work", "Inflation occurs when the general price level rises"),
    ("what is compound interest", "Compound interest is interest calculated on both principal and accumulated interest"),
])
recipe = pairs["training_recipe"]
print(f"Target dims: {pairs['top_target_dims']}")
print(f"Freeze strategy: {recipe['freeze_strategy']}")
```

---

### `generate_training_curriculum(clusters: dict[str, list[str]]) -> dict`

**The full prescription tool.** Given how you want the E8 lattice to look —
which texts should share a key, which should be separated — generates everything
needed to get there.

This is the primary interface between the observatory and the training pipeline.
An AI agent can call this with a cluster specification derived from the diagnostic
methods above, then pass the output directly to `latticememory.training.E8RoutingLoss`.

**Key fields:**

| Field | What it tells you |
|---|---|
| `cluster_analysis` | Fragmentation score per cluster, flags which need unification |
| `positive_pairs` | Within-cluster pairs sorted hard-first for curriculum learning |
| `hard_negative_pairs` | Cross-cluster pairs, hardest (lowest Hamming) first |
| `block_training_weights` | Per-block weight for `E8RoutingLoss`, highest-weight blocks first |
| `curriculum_steps` | 3-phase training plan: warm-up → unification → separation |
| `loss_config.freeze_strategy` | Exact dim ranges to freeze vs. unfreeze |
| `loss_config.block_weights` | Dict of `{dim_range: weight}` for the loss function |

**The 3 curriculum phases:**

| Phase | What it does | LR multiplier |
|---|---|---|
| 1: warm-up | Easy positives only — establish cluster centers without collapse | 0.1× |
| 2: unification | Hard positives — pull fragmented cluster members together | 1.0× |
| 3: separation | Hard negatives — push cluster boundaries apart | 0.5× |

```python
curriculum = obs.generate_training_curriculum({
    "quarterly_earnings": [
        "Q3 earnings exceeded expectations",
        "Third quarter profits beat forecasts",
        "The company reported strong Q3 results",
    ],
    "product_launch": [
        "New product launches next quarter",
        "The device releases in October",
        "Upcoming product announcement expected soon",
    ],
    "regulatory_risk": [
        "SEC investigation ongoing",
        "Regulatory scrutiny increased this year",
        "Government probe into company practices",
    ],
})

print(curriculum["summary"])
print(f"\nPhase 1: {curriculum['curriculum_steps'][0]['n_pairs']} easy positive pairs")
print(f"Phase 2: {curriculum['curriculum_steps'][1]['n_pairs']} hard positive pairs")
print(f"Phase 3: {curriculum['curriculum_steps'][2]['n_pairs']} hard negative pairs")
print(f"\nTop training targets: {[b['dim_range'] for b in curriculum['block_training_weights'][:5]]}")
print(f"\nFreeze strategy: {curriculum['loss_config']['freeze_strategy']}")
```

---

### `compare_snapshots(before: dict, after: dict) -> dict`

**Closes the optimization loop.** Call `export_for_llm()` before and after training,
pass both snapshots here, get a structured delta report.

**Key fields:**

| Field | What it tells you |
|---|---|
| `verdict` | `improved` / `degraded` / `neutral` / `unknown` |
| `entropy.delta` | Change in mean block entropy (negative = improved) |
| `entropy.improved_blocks` | Blocks that got more stable |
| `entropy.degraded_blocks` | Blocks that got noisier — watch for unintended side effects |
| `collisions.delta` | Change in sampled collision cell count |
| `most_improved_blocks` | Top 5 blocks with biggest entropy reduction |
| `most_degraded_blocks` | Top 5 blocks that got worse — potential collapse signal |
| `summary` | One-line human-readable delta |

```python
snapshot_before = obs.export_for_llm()
# ... run training ...
snapshot_after = obs.export_for_llm()

delta = obs.compare_snapshots(snapshot_before, snapshot_after)
print(f"Verdict: {delta['verdict']}")
print(delta["summary"])

if delta["entropy"]["degraded_blocks"] > delta["entropy"]["improved_blocks"]:
    print("WARNING: More blocks degraded than improved — possible encoder collapse")
    print("Degraded blocks:", [b["block"] for b in delta["most_degraded_blocks"]])
```

---

## Worked Example: Domain-Specific Encoder Audit

```python
from latticememory import LatticeIndex
from latticememory.observatory import LatticeObservatory
import json

# 1. Load your index
index = LatticeIndex(model="dfrokido/bge-large-e8-snap")
# ... add your documents ...
obs = index.observatory()

# 2. Global health check
snapshot = obs.export_for_llm(n_sample_cells=10)
print(json.dumps(snapshot["recommendations"], indent=2))

# 3. Full collision scan
audit = obs.collision_audit()
print(f"\nCollision rate: {audit['collision_rate']:.1%}")
if audit["collision_cells"] > 0:
    worst = audit["ranked_by_risk"][0]
    print(f"Worst cell: key={worst['key'][:16]}... mean_cosine={worst['mean_cosine']}")
    coh = obs.cell_coherence(worst["key"])
    print("Contents:", coh["texts"][:3])

# 4. Map blocks to semantic dimensions (using your domain categories)
probe = obs.semantic_probe({
    "intent_purchase": ["buy now", "add to cart", "checkout", "place order"],
    "intent_research":  ["how does it work", "compare options", "what are alternatives"],
    "intent_support":   ["my order is broken", "need a refund", "contact support"],
})
print(f"\nTop 3 separating blocks: {probe['fine_tune_targets'][:3]}")
print(f"Freeze candidates: {probe['freeze_candidates']}")

# 5. Check for redundancy
corr = obs.block_correlation(your_texts)
print(f"\nRedundant block pairs: {len(corr['redundant_pairs'])}")

# 6. Test continuity on a paraphrase chain
traj = obs.address_trajectory([
    "I want to buy this product",
    "I'd like to purchase this item",
    "Can I order this now?",
    "How do I add this to my cart?",
])
print(f"\nTrajectory continuity: {traj['trajectory_continuity']:.0%}")
print(traj["training_recommendation"])

# 7. Generate training curriculum
curriculum = obs.generate_training_curriculum({
    "purchase_intent": ["buy now", "add to cart", "place order", "checkout"],
    "research_intent": ["how does it work", "compare options", "reviews"],
    "support_intent":  ["broken item", "need refund", "contact support"],
})
print(f"\n{curriculum['summary']}")

# 8. Take snapshot before training
pre_train = obs.export_for_llm()

# 9. Run training (using curriculum output)
#    Pass curriculum["loss_config"]["block_weights"] to E8RoutingLoss
#    Train curriculum["curriculum_steps"] in order

# 10. Verify
post_train = obs.export_for_llm()
delta = obs.compare_snapshots(pre_train, post_train)
print(f"\nPost-training verdict: {delta['verdict']}")
print(delta["summary"])
```

---

## Quick Reference: What to Call When

| Symptom | First call | Follow-up |
|---|---|---|
| Don't know where to start | `export_for_llm()` | Feed JSON to reasoning model |
| Cache hit rate is low | `fragmentation_score()` on your query clusters | `generate_training_curriculum()` |
| Getting wrong cached answers | `collision_audit()` | `cell_coherence()` on worst cells |
| Know which queries fail, want to fix | `suggest_training_pairs()` | `generate_training_curriculum()` |
| Want to know what each block encodes | `semantic_probe()` with your label categories | `block_correlation()` |
| Paraphrases not routing together | `address_trajectory()` on a paraphrase chain | `routing_profile()` |
| Fine-tuning made things worse | `compare_snapshots(before, after)` | `block_correlation()` for collapse check |
| New concept category to add | `fragmentation_score()` first | If < 0.75: `generate_training_curriculum()` |
| Expanding to Hamming-1 routing | `neighbor_density()` for each key | Only expand if `expansion_verdict == "safe"` |

---

## Feeding Output to an AI

The observatory is designed to be the context layer between your index and a reasoning
model. Every method returns structured JSON with self-describing fields. The recommended
pattern for AI-guided optimization:

```python
import json
import anthropic  # or openai, etc.

obs = index.observatory()

# Build a full diagnostic package
diagnostic = {
    "snapshot": obs.export_for_llm(n_sample_cells=20),
    "collision_audit": obs.collision_audit(),
    "semantic_probe": obs.semantic_probe(your_label_dict),
    "block_correlation": obs.block_correlation(sample_texts),
}

prompt = f"""
You are optimizing a LatticeMemory E8 semantic index.
Here is the full diagnostic output from LatticeObservatory:

{json.dumps(diagnostic, indent=2)}

Tasks:
1. Identify the top 3 routing failures and their root causes.
2. Specify which blocks to fine-tune and which to freeze.
3. Describe the training data needed using the generate_training_curriculum() schema.
4. Estimate the expected improvement in cache hit rate after training.
"""

# Send to your reasoning model of choice
```

The reasoning model will have everything it needs: entropy structure, collision map,
semantic block assignments, redundancy graph, and auto-generated recommendations —
all in a single JSON object it can reason over symbolically rather than numerically.

---

## Key Numbers to Know

| Metric | Threshold | Meaning |
|---|---|---|
| `mean_block_entropy` | < 0.5 | Key collapse — encoder training failure |
| `mean_block_entropy` | > 6.0 | Near-random routing — encoder not E8-aligned |
| `coherence_label` | `collision` | Mean cosine < 0.80 — false cache hits |
| `fragmentation_score` | < 0.25 | Concept is scattered — dense fallback required |
| `routing.exact_rate` | ≥ 0.60 | Good cache hit rate for paraphrase workloads |
| `hamming.mean` (profile) | > 20 | Asymmetric workload — structural limit, not a training problem |
| `block_correlation NMI` | > 0.9 | Redundant block pair — fine-tune only one |
| `trajectory continuity` | ≥ 0.80 | Encoder is smooth on this semantic progression |
| `separability` (probe) | < 0.1 | Block carries no signal for this label — freeze it |
