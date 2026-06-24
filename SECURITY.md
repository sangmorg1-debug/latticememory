# Security Policy

## Reporting a vulnerability

Please report security issues privately by emailing: <sangmorg1@gmail.com>

Do not open public GitHub issues for suspected vulnerabilities.

Please include:

- affected version or commit (package versions are published as `lattice-memory-e8` on PyPI)
- reproduction steps
- impact
- suggested fix, if available

## Supported branches

The default branch (`main`) is currently supported. The `release/hamming-router-productization` branch tracks in-progress work and is not yet supported as a release target.

## Scope and known risk areas

This project snaps text embeddings onto the E8 lattice for cache/dedup keys, with a FastAPI-based proxy (`latticememory/proxy.py`) and a hybrid dense fallback for retrieval. Security-relevant surface includes:

- **The proxy service** (`latticememory/proxy.py`) - an OpenAI-compatible HTTP proxy; treat as a network-facing component.
- **The hamming-router / LLM-judge gating logic** (`latticememory/hamming_router.py`) - decides whether a cached result is "close enough" to serve. See "Known limitation" below.
- **Third-party integrations** (`latticememory/integrations/`) - LangChain, LlamaIndex, and other framework adapters trust whatever embeddings/text those frameworks pass through.
- **The Rust/WASM component** (`rust/`) - only consumed by the browser extension, reimplements the E8 decode math independently of the Python path.
- **Local LLM execution via Ollama** for judge-model reranking - treat any locally-run model invocation as executing semi-trusted logic over user-controlled input.

## Known limitation (disclosed, partially mitigated)

Internal validation against the real PAWS adversarial-paraphrase dataset found that the cosine-similarity gate alone has a near-99% false-accept rate on hard real-world adversarial pairs (entity/argument-role swaps). Swapping the LLM-judge backstop from a 1.5B to a 7B parameter model, measured on a 1,000-pair real evaluation, brought the end-to-end false-accept rate down to **17.5%** (82.49% catch rate, 20.85% false-reject rate, 2.42s average judge latency) - a real improvement from the earlier ~50% estimate, but still a residual correctness/integrity risk: the cache can, under real adversarial-style inputs, serve a cached result for text that is not actually equivalent to the query. This matters most in contexts where serving the wrong cached result has security consequences (e.g. gating an automated action on a cache hit). Separately, real-model retrieval benchmarking (not synthetic) found symmetric paraphrase retrieval without an adapter or dense fallback is far weaker than previously estimated (0.45% exact-match hit rate on PAWS), and asymmetric QA retrieval is structurally unsupported out of the box (0.0% on MS MARCO) - both now documented honestly rather than left as "not yet benchmarked." See `docs/manual-results/2026-06-23-paws-real-world-validation-summary.md` for full methodology and results; further remediation (e.g. a high-confidence cosine bypass to reduce judge-latency/false-reject overhead) is recommended but not yet implemented.

## Disclosure

I will acknowledge reports as soon as possible and coordinate fixes before public disclosure.
