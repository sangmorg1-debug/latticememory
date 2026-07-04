# Parked WIP Branch Review

**Date:** 2026-07-04
**Branch:** `park/main-dirty-before-redis-proof-2026-07-04`
**Commit reviewed:** `bc40591 park unrelated main work before redis proof merge`
**Main baseline:** `13c1c77 add large redis proof and replay demo`

## Verification

The parked branch is not throwaway work. It compiles and its tests pass in
isolation:

- `python -m py_compile latticememory\agent_sync.py latticememory\agent_sync_network.py latticememory\drift_monitor.py latticememory\flywheel.py latticememory\memory.py latticememory\proxy.py latticememory\proxy_server.py latticememory\redis_store.py latticememory\semantic_cache.py latticememory\shape_runtime.py`
- Focused WIP tests: `65 passed, 3 warnings`
- Full parked-branch suite: `602 passed, 6 warnings`

The branch is not merge-ready as one unit:

- It was parked before the Redis proof branch landed, so it conflicts with
  current `main` in `latticememory/proxy.py` and `latticememory/proxy_server.py`.
- The original WIP commit fails `git diff --check` because many new benchmark and
  test files contain trailing whitespace.
- The proxy changes mix WebSocket serving, flywheel REST endpoints, dashboard UI,
  background fine-tuning, Redis capacity controls, and unrelated analytics in one
  large diff.

## Recommended Split

### Track 1: Redis Store Sharding And LRU

**Files:** `latticememory/redis_store.py`, `tests/test_redis_store_sharded.py`

This is a real candidate for promotion. It adds multi-Redis URL support and
per-shard LRU bookkeeping. Keep it separate from proxy changes so it can be
tested against the already-merged Redis proof path.

Required cleanup:

- Rebase onto current `main`, preserving existing `redis_url`, `redis_namespace`,
  and `redis_ttl` behavior.
- Decide whether `max_entries` means global capacity or best-effort per-shard
  capacity when the value is not divisible by shard count.
- Add a real Redis integration smoke if feasible; the current tests use a mock.

### Track 2: Drift Snapshot Monitor

**Files:** `latticememory/drift_monitor.py`, `tests/test_drift_monitor.py`

This is small, well-scoped, and product-relevant. It directly supports public
cache-safety claims by detecting encoder/model coordinate drift before stale
cache entries are trusted.

Required cleanup:

- Add documentation and a CLI/proxy hook only after the module lands.
- Clarify the distance unit in docs: it is byte/block mismatch over lattice key
  bytes, not vector cosine drift.

### Track 3: Shape Runtime

**Files:** `latticememory/shape_runtime.py`, `tests/test_shape_memory.py`

This is promising but should be experimental. It extends lattice retrieval to
precomputed non-text vectors, which is useful for CAD/3D/embedding reuse. It is
not yet evidence for a public claim.

Required cleanup:

- Keep it behind docs that say inputs are precomputed vectors, not raw meshes.
- Add a small benchmark or fixture before advertising it outside developer docs.

### Track 4: Agent Memory Sync

**Files:** `latticememory/agent_sync.py`, `latticememory/agent_sync_network.py`,
`tests/test_agent_sync_network.py`, `tests/test_agent_sync_network_async.py`,
`benchmarks/benchmark_agent_sync.py`

The in-process sync layer is useful. The network layer needs security hardening
before promotion because peer-provided URLs and document payloads cross a trust
boundary.

Required cleanup:

- Add peer allowlisting or signed peer identity before enabling HTTP sync in
  product docs.
- Validate `sender_url` and document payload size.
- Add timeout/error telemetry instead of silently swallowing broad failure modes.
- Keep benchmark scripts out of the first merge unless they are cleaned and
  documented.

### Track 5: Flywheel Proxy Endpoints And Dashboard

**Files:** `latticememory/proxy.py`, `latticememory/flywheel.py`,
`tests/test_proxy_flywheel.py`, `tests/test_flywheel.py`

This is useful, but it is not safe to merge as currently shaped. The endpoints
are valuable for the proof/demo story, but the proxy diff is too broad and
contains a background fine-tuning endpoint that can write to arbitrary output
paths.

Required cleanup:

- Split API endpoints from dashboard UI.
- Keep `/v1/flywheel/stats`, `/gaps`, `/drift`, and `/review` first.
- Defer `/v1/flywheel/finetune` until there is a job runner, output directory
  allowlist, and explicit safety model.
- Rebase carefully over the Redis proof proxy changes, especially the validated
  cache cosine gate and Redis constructor args.

### Track 6: Proxy WebSocket Endpoint

**Files:** `latticememory/proxy.py`, `tests/test_proxy_websocket.py`

This is product-facing and should be delayed until the HTTP proxy path is stable.
It needs close parity with the existing REST handler: auth behavior, cache gate
headers or equivalents, analytics accounting, streaming error handling, and
upstream cancellation behavior.

Required cleanup:

- Implement as a narrow endpoint after the flywheel/proxy split, not in the same
  diff.
- Add tests for auth, cache cosine gate rejection, upstream error propagation,
  and client disconnect.

### Track 7: QA Bot

**Files:** `tests/test_qa_bot.py`

The tests exercise an existing `latticememory/qa_bot.py`. This should not be
merged with the parked branch until the actual QA bot implementation is reviewed
against current `main`.

Required cleanup:

- Review the existing `qa_bot.py` implementation separately.
- Decide whether QA bot belongs in core package docs or only examples.

## Merge Order

1. Redis store sharding/LRU.
2. Drift snapshot monitor.
3. Shape runtime as experimental.
4. Agent sync in-process cleanup, then network sync behind explicit trust rules.
5. Flywheel REST endpoints without dashboard/fine-tune.
6. Dashboard as a separate UI layer.
7. WebSocket proxy.
8. QA bot review/tests.

This order keeps the Redis proof foundation stable while moving the useful WIP
pieces forward in independently reviewable branches.

## Claim Guidance

None of the parked WIP should change the public Redis proof claim yet. The only
current public claim should remain the validated Redis PQ proof from
`docs/latticememory_public_proof_2026-07-04.md`.

The parked WIP can eventually support additional claims, but only after each
track has its own proof:

- Redis sharding: multi-proxy/shared-cache capacity behavior.
- Drift monitor: encoder drift detection before serving stale cache hits.
- Shape runtime: vector-to-vector lattice retrieval for non-text embeddings.
- Agent sync: efficient shared semantic memory between trusted agents.
- Flywheel endpoints: auditable human review loop for cache misses.
