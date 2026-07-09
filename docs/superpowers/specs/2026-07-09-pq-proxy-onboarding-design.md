# PQ Proxy Onboarding & Design-Partner Quickstart Design

## Goal

Close the gap between the proxy+PQ+Redis serving path that's already proven correct (`docs/proxy_pq_redis_flywheel_proof_pack_2026-07-03.md`: 99.17% hit rate, 0% false positives, real Bitext workload, real Redis Stack) and something a design partner can actually run against their own data. This is the first of three planned phases for the LatticeMemory Redis productization roadmap item — the other two (a guided/auto-calibrating onboarding flow, and a hosted zero-install demo) are deliberately deferred until this phase surfaces what design partners actually find difficult.

## Why this is needed, not just "write more docs"

Two real gaps exist in the shipped code today, found by reading it directly rather than trusting the README:

1. **`lattice serve`'s PQ defaults don't match the validated configuration.** `LatticeIndex`'s own defaults (`_DEFAULT_PQ_NUM_BLOCKS = 8`, `_DEFAULT_PQ_CODEBOOK_SIZE = 256`) are documented as "the validated sweet spot from real-model testing." But `proxy_server.py` hardcodes `LATTICE_PQ_NUM_BLOCKS`/`LATTICE_PQ_CODEBOOK_SIZE` defaults of `4`/`4`. A design partner following `lattice serve --pq-proof-dataset ...` as documented gets a materially worse cache than the one the proof-pack measured.
2. **The only documented way to get a PQ-backed cache running requires the proof-pack's dataset schema**, which needs `cache_seed`/`calibration`/`evaluation`/`adversarial` splits with `id`/`intent_id`/`expected_cache_id`/`is_adversarial` fields per row. `build_seeded_pq_cache_from_support_jsonl`'s own docstring says: *"This is intentionally a proof/demo helper, not a general-purpose PQ proxy configuration API."* No design partner has this schema sitting in their support system. What they have is a list of question/answer pairs — which `--warm-path` already accepts, in exactly this simpler shape, but `--warm-path` only seeds an already-constructed cache; it doesn't build a PQ-backed one.

Both gaps are fixable by wiring together pieces that already work correctly, not by building new architecture.

## Scope

**In scope:**
- Correct `proxy_server.py`'s PQ default values to match `LatticeIndex`'s validated defaults (8 blocks / 256-entry codebook).
- Add a `--pq-mode` flag to `lattice serve` that constructs the proxy's semantic cache backed by `PQLatticeDB` (from `latticememory/rag/pq_retriever.py`), calibrated from the same file `--warm-path` already loads — reusing `--warm-path`'s existing CSV/JSON/JSONL loader and simple `question`/`answer` schema, not the proof-pack's schema.
- A new quickstart doc, `docs/getting-started/design-partner-quickstart.md`, walking through three stages in order of increasing setup cost and increasing hit-rate ceiling: exact-cache only (zero config) → calibrated Hamming+cosine (`lattice calibrate` with a small paraphrase/near-miss set) → PQ+Redis (`--warm-path` + `--pq-mode`, corrected defaults). Each stage's expected numbers are cited from the real proof-pack results, not synthetic ones.
- Update the CLI `--help` text for `--pq-num-blocks`/`--pq-codebook-size` (currently says "default 4", which is simply wrong once the code default changes).

**Explicitly out of scope, deferred to the next phase:**
- Any mechanism that infers or calibrates from live traffic automatically. This phase still requires a design partner to hand-provide a small labeled calibration set (paraphrase pairs, near-miss pairs) — that manual step is exactly the friction the next phase (a guided/shadow-mode calibration flow) exists to remove, once real usage shows whether it's actually the blocker.
- `--pq-proof-dataset` and the proof-pack schema are not being removed or changed — they remain the reproducible-benchmark path. `--pq-mode` is a second, simpler door, not a replacement.
- `docker-compose.yml` — confirmed correct as-is. `LatticeRedisStore` only issues plain Redis GET/SET, never RediSearch, so `redis:7-alpine` is sufficient; Redis Stack was only needed for the proof-pack's own RedisVL baseline comparison, not for LatticeMemory's operation.
- A hosted/public demo (third planned phase).

## Architecture

No new modules. Three existing files change:

- **`latticememory/proxy_server.py`**: change the two `os.getenv(..., "4")` defaults to `"8"`/`"256"`. Add the `--pq-mode` branch: when set (and `--pq-proof-dataset` is not), build a `PQLatticeDB`-backed `RFSnapSemanticCache` from `--warm-path`'s entries instead of the plain exact-cache one, using `pq_num_blocks`/`pq_codebook_size` (now correctly defaulted). If both `--pq-mode` and `--pq-proof-dataset` are given, `--pq-proof-dataset` wins (it's the more specific, fully-specified path) and a warning is logged noting `--pq-mode` was ignored. If `--pq-mode` is given without `--warm-path`, fail fast at startup with a clear `RuntimeError` ("--pq-mode requires --warm-path: PQ codebooks are fit from the warm-start file's entries, and there's nothing to fit from without one") rather than starting with an uncalibrated or empty PQ cache — matching this codebase's existing pattern of raising a clear error at construction time instead of failing confusingly on the first request (see `TokenizerBridge`'s starved-state check in the E8 platform sibling project for the same philosophy).
- **`latticememory/cli.py`**: add the `--pq-mode` argument to the `serve` subparser (`action="store_true"`, off by default — exact-cache remains the zero-config default). Fix the `--pq-num-blocks`/`--pq-codebook-size` help strings to say "default 8"/"default 256".
- **`docs/getting-started/design-partner-quickstart.md`** (new file): the three-stage walkthrough described above.

## Data Flow

```text
--warm-path partner_qa.jsonl  (existing loader: {"question": ..., "answer": ...} per line)
  --pq-mode not set  → entries loaded into whatever cache was already constructed (today's behavior, unchanged)
  --pq-mode set      → same file's entries used to (a) fit PQ codebooks (8 blocks / 256 codewords,
                        matching LatticeIndex's validated default) via PQLatticeDB, then
                        (b) seed the resulting PQ-backed cache — one file, one pass, no schema change
                        for the design partner to learn.
```

## Testing

- Unit test: `proxy_server.py` module-level defaults, asserting `pq_num_blocks == 8` and `pq_codebook_size == 256` when the env vars are unset (regression test for the exact bug found — the whole reason this phase exists).
- Unit test: `--pq-mode` with a small in-memory `--warm-path` JSONL fixture produces a cache whose entries are retrievable via `RFSnapSemanticCache.get()`, and whose underlying lattice is a `PQLatticeDB` instance (not the default `E8LatticeDB`) — proves the wiring, not just that *some* cache got built.
- Unit test: `--pq-mode` + `--pq-proof-dataset` together — asserts `--pq-proof-dataset` wins and a warning is logged, per the Architecture section's stated precedence.
- Doc: the quickstart's exact-cache and Hamming-calibration stages are already covered by existing CLI behavior (no new test needed); the PQ+Redis stage's example commands should be run manually once during implementation to confirm they work verbatim as written, since a design partner copy-pasting a broken command is the specific failure mode this phase exists to prevent.

## Open Questions Resolved During Planning

None outstanding — the two gaps motivating this phase (default mismatch, schema-only-via-proof-pack) were both confirmed by reading the code directly (`proxy_server.py`, `proof_pack.py`'s docstring, `pq_retriever.py`'s `PQLatticeDB`, `proxy.py`'s `_warm_cache`), not inferred from documentation.
