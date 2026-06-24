# Patch the Planet application — latticememory

> Paste-ready application content, current as of the verified validation work completed 2026-06-23.

---

**Repository:** latticememory
**URL:** https://github.com/sangmorg1-debug/latticememory
**Description:** Semantic LLM cache/dedup library using the E8 lattice (densest known 8-D sphere packing) as a deterministic address space for text embeddings — 32x compressed cache keys for instant repeat/paraphrase-query hits, with a dense float32/Int8 fallback for novel retrieval.
**Main languages/frameworks:** Python (core library, FastAPI proxy), Rust (WASM component for a browser extension)
**Package/ecosystem:** PyPI (`lattice-memory-e8`, v0.2.0)
**Security-sensitive areas:** AI-agent/LLM tooling, networking (OpenAI-compatible proxy service), local LLM execution (Ollama-based judge-model reranking), third-party framework integrations (LangChain, LlamaIndex)
**Current adoption:** Recently published (PyPI v0.2.0, June 2026); live HuggingFace Space demo and model also published. Early stage — adoption metrics not yet tracked.
**Maintainer contact:** sangmorg1@gmail.com

**I am looking for help with:**

- Vulnerability validation
- Patch development
- Tests/fuzzing/security CI (none currently exists — no `.github/workflows` in the repo yet)
- AI-agent/tooling safety review
- A specific, measured finding I'd like a second opinion on: the cache's gating logic decides whether a cached result is "close enough" to serve a query. Real-world validation against the PAWS adversarial-paraphrase dataset found the cheap cosine-similarity gate alone has a near-99% false-accept rate on hard adversarial pairs (entity/argument-role swaps that look similar but aren't equivalent). Swapping the LLM-judge backstop from a 1.5B to a 7B parameter model brought the measured end-to-end false-accept rate down from an estimated ~50% to **17.5%** (on a 1,000-pair real evaluation: 82.49% catch rate, 20.85% false-reject rate, 2.42s average judge latency). That's real progress, but a 17.5% residual false-accept rate is still a correctness/integrity risk in contexts where serving the wrong cached result has consequences (e.g. gating an automated action on a cache hit) — I'd value expert eyes on whether the two-stage cosine-gate + LLM-judge architecture itself is the right shape, or whether it needs to change.
- Separately, real-model retrieval benchmarking (not the synthetic numbers the project previously cited) found symmetric paraphrase retrieval without an adapter or dense fallback is far weaker than expected (0.45% exact-match hit rate on PAWS) and asymmetric QA retrieval is structurally unsupported out of the box (0.0% on MS MARCO) — both now documented honestly in the repo rather than left as optimistic placeholders.

Full methodology and raw results for all of the above are in the repo's `docs/manual-results/` and `docs/honest_product_review.md` - I'd rather hand over real numbers than a pitch.

I am the maintainer and can review pull requests, coordinate fixes, and add further security documentation as needed.

---

## Notes for whoever submits this (not part of the application text)

- This supersedes the 2026-06-22 draft, which cited the original 60-pair-sample estimate (~50% false-accept rate) rather than the verified 1,000-pair measurement (17.5%) now available.
- `SECURITY.md` and `AGENTS.md` already exist in the repo root and disclose this same finding - the application text above is consistent with what a reviewer will find on arrival, not a separate story.
- If asked for more detail in the application form: `docs/manual-results/2026-06-23-paws-real-world-validation-summary.md` is the single best link to point to.
