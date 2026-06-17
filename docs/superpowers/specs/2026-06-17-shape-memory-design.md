# Shape Memory v1 — Design Spec

## Status

Design only. No production code written yet. Validation experiments complete
(real data, real models) and support proceeding. Workspace for this effort is
`shape_memory_dev/` at the repo root — **untracked** (gitignored), local-only,
moves to its own repo before publishing per explicit prior instruction.

## Context

LatticeMemory ships a mature E8-lattice semantic cache for text (`RFSnapTextMemory`,
`HammingRouter`, `RFSnapSemanticCache`). A separate, more advanced effort
("E8 Shape Memory") already exists as a standalone Windows app applying the same
E8 quantization idea to 3D point clouds (OpenShape PointBERT-ViT-L encoder, Rust
backend, Bevy viewer) — built in the same `e8-Project` monorepo LatticeMemory was
extracted from, but never merged back.

This spec covers extending LatticeMemory's core engine to support 3D shapes as a
second modality, validated by two real-data experiments run this session
(see `shape_memory_dev/CROSSMODAL_RESULTS.md` and `SHAPE_DEDUP_RESULTS.md`):

1. **Cross-modal text→shape search is closed — does not work.** Real OpenShape
   embeddings vs. real CLIP text embeddings on 200 real Cap3D pairs: 0% E8 exact/beam
   hit rate, same-pair cosine only 0.106. Not in scope, not a future v2 either, unless
   a fundamentally different encoder pairing is found.
2. **Same-modality shape↔shape caching is validated, viable.** Same encoder, 200 real
   objects, intra (same object, augmented) vs. inter (different objects) Hamming
   distance: gap is negative (-16) at the tails but the bulk separates cleanly, with
   usable threshold operating points (e.g. 64/96 blocks → 43% recall / 2% FP). This is
   the same risk profile the text cache already ships with.

## Goal

Build `RFSnapShapeMemory` — a 3D-shape-modality runtime mirroring the existing
`RFSnapTextMemory`, reusing the unmodified core quantization engine
(`RFSnapLatticeMemory`, `HammingRouter`). Ship as a same-modality cache/dedup tool
for 3D assets (catch re-submitted/re-exported/re-scanned duplicates of the same
object), not a cross-modal search tool.

## Non-goals (explicitly out of scope for v1)

- Cross-modal search (text query → 3D shape result). Closed question, see above.
- Semantic clustering of *different* objects in the same category (e.g. routing
  two distinct chair models to a shared "chair" cell). Untested, harder claim,
  closer to classification than caching — not validated, not promised.
- Training-data acceleration via storing/training on E8 keys directly. Not how
  the quantization works (lossy address, not a learned reconstructable code) —
  ruled out in conversation, not re-litigated here.
- Integration into the main `latticememory` PyPI package. Stays in
  `shape_memory_dev/` until proven out, per explicit prior instruction.

## Architecture

Confirmed by reading the actual code (`latticememory/memory.py`,
`latticememory/text_runtime.py`): `RFSnapLatticeMemory` is already modality-agnostic
— it operates on float32 embeddings of a configured dimension and has no
text-specific logic. The only modality-specific piece is the thin wrapper
(`RFSnapTextMemory`) that adapts a `TextEncoder` protocol onto that core.

```
RFSnapShapeMemory(encoder, d_model=768, memory=RFSnapLatticeMemory(d_model=768))
        |
        +-- ShapeEncoder protocol: encode(point_clouds, batch_size, **kwargs) -> embeddings
        |       (implemented by OpenShapeEncoderAdapter, wrapping shape_encoder.py's
        |        OpenShapeEncoder from the copied e8-Project source)
        |
        +-- delegates everything else (add_documents, search, lattice key lookup,
            Hamming routing) to the SAME RFSnapLatticeMemory / HammingRouter classes
            text already uses, unmodified.
```

### Components

1. **`ShapeEncoder` protocol** (new, small) — structural type matching the existing
   `TextEncoder` protocol shape: `encode(point_clouds: list[np.ndarray], batch_size: int, **kwargs) -> np.ndarray`.
   Lives alongside `RFSnapShapeMemory` in `shape_memory_dev/` for now.

2. **`OpenShapeEncoderAdapter`** — thin wrapper around the already-working,
   already-tested `OpenShapeEncoder` (from `shape_memory_dev/liora_core/adapters/shape_encoder.py`,
   copied from `e8-Project`) to match the `ShapeEncoder` protocol's batch signature.
   No changes to the underlying encoder — it already works (used in both validation
   experiments this session).

3. **`RFSnapShapeMemory`** — mirrors `RFSnapTextMemory` field-for-field:
   `__init__(self, *, encoder: ShapeEncoder, d_model: int = 768, memory: RFSnapLatticeMemory | None = None, ...)`.
   `MemoryDocument.text` field is reused as the shape's caption/label (Cap3D objects
   already have one) — no change needed to the shared `MemoryDocument` dataclass.
   The actual point-cloud data/file path lives in `MemoryDocument.metadata`.

4. **Threshold calibration** — reuse `HammingRouter.calibrate_threshold()` unchanged,
   re-run against shape data instead of text. Starting point from validation:
   threshold≈64 (43% recall/2% FP) as a balanced default, threshold≈50 (23%/0%) as
   a conservative default — same two-tier pattern the proxy already uses for text
   (default=70 conservative, calibrated higher per-domain).

### Data flow

```
PLY/point-cloud file or (N,6) xyz+rgb array
  -> OpenShapeEncoderAdapter.encode() -> 768-D embedding
  -> RFSnapLatticeMemory._quantize_to_indices() -> 96-byte E8 key  (unchanged core)
  -> same exact/Hamming lookup path text already uses (unchanged core)
  -> MemoryHit / SemanticCacheResult (same result types, unchanged)
```

No new result types needed — `MemoryDocument`, `MemoryHit`, `MemoryResult` are
already modality-neutral (confirmed by reading `memory.py`).

### Testing

Mirrors the existing test pattern (`tests/test_lattice_index.py`,
`tests/test_verticals.py` use a deterministic `HashEncoder`/`FakeEncoder` so tests
don't need network/GPU). For shape tests specifically:

- A `FakeShapeEncoder` (same MD5-seed-to-deterministic-vector pattern already used
  in `test_agent_swarm.py`/`test_verticals.py`) for fast, no-GPU unit tests of the
  `RFSnapShapeMemory` wiring itself (add/search/Hamming routing logic).
- The two real-data validation scripts (`bench_crossmodal_shape_e8.py`,
  `bench_shape_dedup_e8.py`) stay in `shape_memory_dev/` as the empirical-grounding
  evidence — they are slow (GPU, real model downloads) and not part of the unit
  test suite, same way `benchmarks/validate_hamming_thresholds.py` isn't part of
  `tests/` for the text cache either.

### Dependencies

OpenShape requires `torch`, `open_clip_torch`, `open3d`, `torch_redstone`, `einops`
— all confirmed already installed in this environment. These stay scoped to
`shape_memory_dev/` and are not added to the main package's `pyproject.toml` until
(if) this moves toward integration — consistent with the "stays untracked until
ready" instruction.

## Open questions for implementation plan

- Exact calibration threshold to ship as the default (64 vs. 50, or expose both
  as named presets like the text proxy does).
- Whether `RFSnapShapeMemory` should accept `.ply` files directly (matching the
  installed E8 Shape Memory app's `load_ply` helper) or only pre-loaded arrays —
  affects whether `open3d`/PLY-parsing code needs to be ported into this workspace.
- Whether to validate the "same exact threshold works across object categories"
  claim (current validation used 200 objects across mixed categories already, but
  wasn't stratified by category to check for per-category threshold drift).
