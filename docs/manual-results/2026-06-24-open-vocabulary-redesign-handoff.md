# Handoff prompt: open-vocabulary semantic addressing redesign

> Paste everything below this line into the agent session for this work. This is a genuine research attempt, not a bug fix - read the "what's already failed and why" section carefully before writing any code, because the obvious first three things to try have already been tried and conclusively failed.

---

## Context you need before starting

`latticememory` (PyPI: `lattice-memory-e8`, this repo: `E:\latticememory`) snaps text embeddings onto the E8 lattice - the densest known sphere packing in 8 dimensions - as a deterministic address space for fast semantic caching. The pitch: identical or near-identical text gets an O(1) cache hit via an exact (or Hamming-1) lattice address match, at 32x storage compression versus raw float32 embeddings, instead of a full vector similarity search.

**This works genuinely well for one narrow case and does not work at all for the general case**, and this session's job is to find out whether that can be fixed with a different addressing scheme - not a better-trained adapter (already tried, see below).

### The exact mechanism, precisely

`latticememory/rag/e8_retriever.py`, class `E8LatticeDB`:

- A 1024-dim embedding is split into 128 independent blocks of 8 dimensions each.
- Each 8-dim block is independently quantized to the nearest of 240 codewords (the E8 root system's shell-1 codebook, kissing number 240 - this part is mathematically correct and has been verified).
- The 128 quantized block-indices, concatenated, form a 128-byte "address." Two embeddings land on the same address only if **all (or, for Hamming-1 routing, all but one) of the 128 independent quantization decisions agree.**

### Why this fails for open text - the actual mechanism, not just "it's hard"

Real-model validation (`dfrokido/bge-large-e8-snap`, the actual production encoder) against two real public datasets:

| Dataset | Task | Real-model hit rate (exact + Hamming-1) |
| --- | --- | --- |
| PAWS (`google-research-datasets/paws`, `labeled_final` test, n=3536 true paraphrases) | Symmetric paraphrase retrieval | **0.45%** |
| MS MARCO (`microsoft/ms_marco`) | Asymmetric QA retrieval | **0.00%** |

Root cause: any natural embedding drift between a sentence and its paraphrase - even a trivially meaning-preserving reword - is enough to flip several of the 128 independently-quantized blocks to a *different* codeword. Requiring 127-128 independent categorical agreements out of 128 is an astronomically high bar that real paraphrase-level embedding variance essentially never clears. This is a structural property of the addressing scheme, not a training or calibration problem.

### What's already been tried and has conclusively failed - read this before proposing "train an adapter"

A residual-MLP query adapter (`train_lattice_adapter_from_examples` in `latticememory/training.py`, already-built real infrastructure, contrastive + hard-negative loss, address-prediction loss) was trained and evaluated on PAWS three separate times this session (2026-06-24), isolating different variables each time:

| Config | Train examples | Epochs | Train accuracy | Held-out Recall@1 | What it shows |
| --- | --- | --- | --- | --- | --- |
| A | 3,000 | 8 | 6.3% | 0.00% | Underfit |
| B | 3,000 | 25 | 16.8% (still climbing, loss still smoothly decreasing) | 0.00% (mean Hamming distance got *worse*, 38.99→50.05) | **Overfitting** - more training on the same small set hurt generalization, didn't help it |
| C | 15,000 (5x more data) | 12 | 0.11% (training stalled - too little repetition per example) | 0.00% (but lattice route rate moved 0%→2.15%, still always the *wrong* document) | More data alone, same epoch budget, just shifts the underfit/overfit boundary - doesn't fix anything |

All three land at 0% correct retrieval on held-out data. This matches the project's own pre-existing finding on MS MARCO (`docs/honest_product_review.md` gap #2: adapters reduce *train* Hamming distance to 9.39-9.73 but *validation* Hamming distance stays at 99.16-106.26, essentially random). **Conclusion already drawn and verified twice independently: adapting the embedding before quantization, without changing the quantization scheme itself, does not fix this.** Don't re-attempt this without a genuinely different angle on it.

### One thing that DOES work, for a different reason - possibly relevant inspiration

`latticememory/qa_bot.py` has a `"centroid"` routing mode (cosine similarity to per-intent centroid vectors, plain nearest-centroid classification - not E8 quantization at all) that performs well on closed-vocabulary intent matching. It's a fundamentally different, much coarser-grained mechanism: instead of 128 independent fine-grained quantization decisions, it's one coarse decision (which centroid is closest) over a small number of classes. This isn't a fix for the E8 mechanism, but it's a hint about what direction might actually work: **fewer, coarser, data-calibrated decisions beat many fine-grained fixed-geometry decisions for matching real embedding variance.** That test has now completed (`scratch/closed_vocab_validation/e8_exact_closed_vocab_result.json`, 2026-06-24): the real E8LatticeDB exact/Hamming-1 mechanism, on a genuinely closed-vocabulary domain (CLINC150, 16 real intents, real encoder, genuine held-out paraphrases, zero verbatim overlap with training data), scored **0.00% - a complete miss on all 480 held-out queries.** This means the E8-address mechanism currently has **no demonstrated real-data success on anything except literal exact-text repeats** - not open text, and not even closed-vocabulary text once you require genuine held-out paraphrases rather than mock encoders or re-querying with verbatim training strings (which is what the previously-cited "100% held-out accuracy on 16-command domain" claim actually turned out to be measuring - `qa_bot.py`'s separate centroid mode, not this mechanism). Scope this redesign accordingly: you are not trying to extend a working closed-vocabulary mechanism to open text. You are trying to make the core address-matching mechanism work for the *first* real-data use case at all.

## What to actually try

You are not constrained to these - research current semantic/learned-hashing literature for better starting points if you find one - but here are concrete, well-motivated starting hypotheses given everything above:

1. **Coarser blocks.** Instead of 128 independent 8-dim blocks, use far fewer, larger groupings (e.g. 16-32 blocks covering more dimensions each, using a higher-dimensional lattice or a different quantization scheme per block). Fewer independent categorical decisions means fewer chances for any single one to mismatch - this directly attacks the compounding-probability root cause identified above. Tradeoff to measure honestly: this likely costs some compression ratio and exact-match precision (more documents could collide per address) - quantify both sides, don't just report the hit-rate win.

2. **Data-calibrated codebooks instead of fixed E8 geometry.** Replace the fixed mathematical 240-codeword-per-block scheme with codebooks learned from real embedding data (e.g. product quantization with k-means-trained sub-codebooks, trained on a real corpus of embeddings, not assumed from lattice geometry). The E8 lattice is a beautiful answer to "densest packing in 8D" but has no reason to be the right answer to "where does real sentence-embedding probability mass actually concentrate." Tradeoff to measure: does losing the lattice's nice Hamming-distance structure (used for radius-1 search) cost you the cheap approximate-neighbor lookup that makes this fast in the first place?

3. **Hierarchical / coarse-to-fine addressing.** Coarse cluster assignment first (few decisions, like the qa_bot centroid approach), then fine-grained E8 addressing only *within* a cluster, rather than globally. This could combine the demonstrated strength of coarse data-driven clustering with the compression benefit of fine lattice addressing for the within-cluster case.

## Evaluation methodology - reuse what exists, keep it honest

- **Encoder:** `dfrokido/bge-large-e8-snap`, real, always. No mock/fake encoders for any result you report as validating or refuting an approach - every existing positive claim that turned out wrong in this project's history (see `docs/honest_product_review.md`) traced back to either a mock encoder or a held-out-set contamination issue. Don't repeat that.
- **Symmetric paraphrase baseline to beat:** `scratch/paws-validation/paws_scored.jsonl` (8,000 real PAWS pairs, already scored with cosine similarity) and `scratch/paws-validation/retrieve_paws.py` (the existing measurement script - reuse its methodology, specifically its real-model 0.45% baseline as the number to beat, not a number you re-derive differently).
- **Train/test split discipline:** PAWS's actual train split (fresh, never touched by this project's gate-calibration work, which only ever used the *test* split) is the right place to train anything that needs training. Never evaluate on data used for training, even loosely - `scratch/paws-validation/train_paws_adapter.py` shows the correct split pattern (train on PAWS train split, eval on the held-out `paws_scored.jsonl` test rows).
- **If you build something closed-vocabulary-oriented as part of testing the hierarchical idea:** `scratch/closed_vocab_validation/test_e8_exact_closed_vocab.py` (built this session, also 2026-06-24) is the pattern for a genuine real-encoder, real-mechanism, real-held-out-split test on CLINC150 - reuse it rather than building a new closed-vocabulary harness from scratch.

## Constraints

- **Resource-conscious.** This is meant to be an addressing-scheme/algorithm redesign - k-means, product quantization, and similar are cheap compared to LLM inference. If your approach requires a large model or heavy GPU training that would compete with other active work on this machine, flag that explicitly before running it, the way the original judge-calibration work flagged a large Ollama model pull before doing it.
- **Don't break the 551 passing tests** without explicit justification for what's changing and why.
- **Don't touch the published PyPI version** (`lattice-memory-e8` v0.2.0) without being told to.
- **Report honestly, including a clean negative result.** If this doesn't work either, that's a real and valuable finding - it would mean three independent fix attempts (adapter x3 variations, plus whatever you try here) all fail, which is itself useful evidence that the E8 lattice's exact-address mechanism may be fundamentally unsuited to open-vocabulary text regardless of how the embedding is prepared beforehand, and that the right long-term answer is to lean entirely on the dense Int8 fallback for open text (already correctly documented as the current position in `docs/honest_product_review.md` and `README.md`). Don't force a positive result to avoid reporting that.

## What "done" looks like

A real implementation of at least one of the directions above (or a better one you found), evaluated with the real encoder against the real PAWS baseline (0.45%) using a genuine held-out split, with results written up in `docs/manual-results/` in the same evidence-first style as the rest of this project's history - whether the result is "this beats the baseline by X and here's the real compression/precision tradeoff" or "this also failed, and here's the mechanism that explains why, ruling out this entire class of fix."
