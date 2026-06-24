# Gate calibration roadmap — what we're chasing and what's next

## The actual problem, in plain terms

LatticeMemory serves a cached answer when it decides a new query is equivalent to something already seen. Two ways that decision can be wrong, and they cost you in different currencies:

- **False accept** — serves a cached answer for a query that wasn't actually equivalent. A correctness/trust failure, not just a quality miss. Worst case: an automated action gated on a wrong cache hit.
- **False reject** — refuses to serve a cache hit that was actually valid, falling through to a full LLM call. Doesn't produce a wrong answer, but quietly erodes the cost/latency savings that are this product's entire reason to exist.

There is no universal target number for either rate — it depends on what a given deployment is willing to risk versus pay for. What's NOT context-dependent: a change that improves one rate by making the other one worse isn't progress, it's just moving the failure from one column to the other. **Total error rate (FAR + FRR combined) is the tripwire for telling a real improvement apart from a relabeled trade-off.**

## Bigger-picture tension this work surfaced

The judge only matters for queries that survive the Hamming/cosine gate *before* it. Real-model retrieval validation (2026-06-23) already found that gate barely fires at all on real embeddings without an adapter: **0.45% symmetric hit rate (PAWS), 0.0% asymmetric hit rate (MS MARCO)**. That means all the judge-calibration work below is polishing a decision that only gets made for a small slice of real traffic - the bigger lever may be the retrieval/adapter problem, not the judge. Both are tracked here; priority is discussed at the end.

## Judge calibration: completed experiments

All runs below use the exact same 1,000-pair PAWS sample (cosine-surviving rows from the original validation) for a clean apples-to-apples comparison. `qwen2.5:7b` unless noted.

| Variant | Change tested | FAR | FRR | Total error | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| v1 | Original production prompt ("Question A/B... same answer") | 17.51% | 20.85% | 19.0% | Baseline. Real bug found: PAWS pairs are declarative sentences, not questions - wrong semantic relation being evaluated. |
| v2 | v1's framing fix + strong one-sided role-swap warning | 6.68% | 54.04% | 27.8% | **Worse overall.** Over-sensitized the model into rejecting trivial reorderings/synonyms as if they were role-swaps. |
| **v3** | **Framing fix only - "sentences," no extra hints** | **13.36%** | **21.08%** | **16.8%** | **Best result. Shipped to production** (`latticememory/proxy.py:_rerank_confirms_match`, 2026-06-23). Verified: 551/551 engine tests + 163/163 IDE tests pass with no manual environment overrides. |
| v4 | v3 + calibrated two-sided hint (positive + negative example inline) | 9.03% | 26.91% | 17.0% | Wash. Didn't beat v3. |
| v5 | v3 + forced chain-of-thought reasoning before verdict | 25.99% | 17.04% | 22.0% | **Worse, and ~7-8x slower** (18.0s vs ~2.3s avg latency). Reasoning didn't improve judgment quality on this model size, just added verbose text. |
| v6 | v3's prompt on `qwen3.6:latest` (23GB, already pulled) instead of `qwen2.5:7b` | - | - | - | **Cancelled mid-run** - degraded the host machine's responsiveness. Not yet measured. Most promising untested lever; revisit when it won't compete with other work for resources. |
| v7 | v3 + 4 few-shot examples (2 positive, 2 negative) | 17.69% | 16.82% | 17.3% | Wash. Didn't beat v3. |

**Reading across v2, v4, v5, v7:** five different prompt-engineering strategies on the same small model all land in a narrow 16-17% total-error band, with the simplest fix (v3) still on top. That's the signature of a real capability ceiling for `qwen2.5:7b` on this specific task, not a prompt-wording problem still waiting to be found.

## Open next steps, in priority order

1. **Self-consistency voting** (not yet tried, stays on small models): run v3's prompt 3x at temperature > 0, take the majority verdict. Costs ~3x latency. Worth trying because it's cheap and untested, but unlikely to break the apparent ceiling since the underlying issue looks like capability, not variance.
2. **Bigger judge model** (`qwen3.6:latest`, already pulled, v6 cancelled before producing a result): the most likely real lever left, but resource-intensive on this machine. Revisit when it can run without competing with other active work. If it meaningfully beats v3's 16.8% total error without unacceptable latency, it becomes the new production candidate.
3. **The retrieval/adapter gap** (arguably higher-leverage than further judge tuning): `docs/honest_product_review.md` already shows a trained residual MLP adapter reaches 100% held-out exact-routing accuracy on a small fixed-vocabulary domain (16-command smart-home set), but generalization to PAWS-scale paraphrase or MS MARCO-scale QA has not been attempted. If an adapter approach can lift the 0.45%/0.0% real-model hit rates meaningfully, that increases how much traffic ever reaches the judge stage at all - independent of how good the judge itself is.

## What "done" looks like for this thread

Not a specific number - a documented, evidence-backed answer to: given the actual false-accept/false-reject tradeoff achievable on hardware you're willing to dedicate to this, is the gate's residual risk acceptable for the use cases you intend to support, and is that honestly disclosed (as `SECURITY.md` and `docs/honest_product_review.md` already do)? Continued iteration should keep moving total error down without quietly trading it from one failure mode to the other - and should periodically ask whether the retrieval gap, not the judge, is the better use of the next hour of work.
