# Agent Instructions

This repository welcomes AI-assisted security review, but reports must be evidence-based.

Before reporting an issue:

1. Read `SECURITY.md` first - in particular the "Known limitation" section, which already discloses a real, measured gate-calibration weakness (cosine-gate false-accept rate on adversarial paraphrase pairs). Don't re-report that finding as new; if you have a fix or a materially different angle on it, that's welcome.
2. Confirm the issue is reachable in the supported code path (`main` branch; see `SECURITY.md` for which branches are supported).
3. Provide a minimal reproduction or failing test.
4. Avoid speculative severity ratings.
5. Do not submit public exploit details for unpatched vulnerabilities.

## Where the real risk surface actually is

- `latticememory/proxy.py` - network-facing OpenAI-compatible proxy.
- `latticememory/hamming_router.py` - the gating logic with the disclosed calibration gap above.
- `latticememory/rag/e8_retriever.py` - the core E8 lattice quantization/decode math.
- `latticememory/integrations/` - LangChain, LlamaIndex, and other adapters; trust boundary is whatever the host framework passes in.
- `rust/` - independent Rust/WASM reimplementation of the E8 decode math, consumed only by the browser extension; check `tests/test_wasm_parity.py` for the contract it must hold.
- Local LLM judge invocation via Ollama (`src/lattice_ide/ollama_tools.py` in the sibling `LatticeMemory IDE` repo, not this one) - flagging for awareness since it's the consumer of this library's gating output.

## Test suite reality check

All 551 existing tests pass, but every one of them uses a mock/fake encoder (`FakeEncoder`/`MockEncoder`) - none exercise the real embedding model (`dfrokido/bge-large-e8-snap`). A passing test suite here does not mean real-world retrieval/gating behavior has been validated. See `docs/honest_product_review.md` for what is and isn't actually benchmarked against real data as of the last update.

## Preferred contributions

- Patches with tests
- Fuzzing harnesses
- CI/security workflow improvements
- Dependency and supply-chain hardening
- Documentation that reduces future false positives
- Real-data validation (not synthetic) of retrieval/gating quality claims

## Do not report

- Theoretical issues without reachability
- Duplicate findings (check `docs/manual-results/` and `docs/honest_product_review.md` first - this project already tracks its own known gaps honestly)
- Low-impact lint-only findings unless they prevent a real bug class
