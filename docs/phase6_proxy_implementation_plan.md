# Implementation Plan: Phase 6 (LLM Cache Proxy)

This plan outlines the steps to build **Phase 6** of the LatticeMemory roadmap: the LLM Cache Proxy (`LatticeLLMProxy`), an OpenAI-compatible semantic caching gateway.

## Goal Description
Upstream LLM calls are expensive and slow. The LLM Cache Proxy acts as a "Varnish for LLMs", intercepting OpenAI-compatible chat completion calls. Before making an upstream request, it snaps the incoming prompt to its E8 address and checks the local cache. If a semantically equivalent query has been resolved before, it immediately returns the cached completion.

## Proposed Changes
* Create `latticememory/proxy.py` containing:
  - `LatticeLLMProxy`: A FastAPI application.
  - Endpoint `POST /v1/chat/completions` parsing OpenAI request formats.
  - E8 semantic lookup: snaps request prompt to E8 address.
  - Integration headers: returns `X-Lattice-Cache`, `X-Lattice-Retrieval-Path`, and `X-Lattice-Savings-USD`.
  - Upstream handler: forwards missed requests to actual LLM providers.

## Verification Plan
* Develop a demo in `examples/llm_cache_proxy_demo.py` validating cache hits/misses, custom savings headers, and throughput.
* Add comprehensive mock tests in `tests/test_proxy.py`.
