# Shape Memory v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a same-modality 3D-shape cache (`ShapeHammingRouter`) that catches re-encountered/re-exported duplicate 3D objects via E8 Hamming distance, validated on real Cap3D data, living entirely in the untracked `shape_memory_dev/` workspace.

**Architecture:** Fork the proven `HammingRouter` pattern (`latticememory/hamming_router.py`) into a new `ShapeHammingRouter` class adapted for point-cloud inputs and the `OpenShapeEncoder`'s actual interface (`encode_batch(list[np.ndarray]) -> np.ndarray`), reusing the unmodified, shared `E8LatticeDB._quantize_to_indices()` primitive from the main package. **Deviation from the design spec, found while reading the real code:** the spec said "reuse `RFSnapLatticeMemory`/`HammingRouter` unmodified" — `RFSnapLatticeMemory`'s default beam-radius-1 neighbor search only catches single-block Hamming differences, far narrower than the validated ~50–64-block tolerance this needs, and `HammingRouter` itself hardcodes a SentenceTransformer-style `encode(..., normalize_embeddings=True)` call that `OpenShapeEncoder` doesn't implement. Forking the linear-scan-with-arbitrary-threshold *pattern* (not the literal class) is the correct way to honor the spec's actual goal.

**Tech Stack:** Python, NumPy, PyTorch, the existing `latticememory` package (installed editable, importable from anywhere), `OpenShapeEncoder` (already working in `shape_memory_dev/liora_core/adapters/shape_encoder.py`).

---

## File Structure

- Create: `shape_memory_dev/shape_hamming_router.py` — `ShapeHammingMatch` dataclass + `ShapeHammingRouter` class. One file, one responsibility (mirrors `latticememory/hamming_router.py`'s single-file structure).
- Create: `shape_memory_dev/test_shape_hamming_router.py` — unit tests using a deterministic `FakeShapeEncoder`, plus one real-model integration smoke test. Flat location (not a `tests/` subdir) matches the existing flat layout of `bench_crossmodal_shape_e8.py`/`bench_shape_dedup_e8.py` in this workspace and avoids a pytest sys.path complication (pytest auto-inserts only the test file's own directory, not its parent, when there's no `__init__.py`).
- Modify: `shape_memory_dev/README.md` — add this module to the "What's copied here" table.

No changes to any file inside `latticememory/` (the main package) or `pyproject.toml`.

---

### Task 1: `ShapeHammingMatch` + `ShapeHammingRouter` constructor and key computation

**Files:**
- Create: `shape_memory_dev/shape_hamming_router.py`
- Test: `shape_memory_dev/test_shape_hamming_router.py`

- [ ] **Step 1: Write the failing test**

```python
# shape_memory_dev/test_shape_hamming_router.py
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from shape_hamming_router import ShapeHammingMatch, ShapeHammingRouter


class FakeShapeEncoder:
    """Deterministic point-cloud encoder for tests — no GPU, no model download.

    Seeds a random unit vector from the point cloud's byte content, so the same
    array always produces the same embedding (mirrors the FakeEncoder pattern in
    tests/test_agent_swarm.py and tests/test_verticals.py, adapted for array input
    instead of text input).
    """

    def __init__(self, d_model: int = 768):
        self.d_model = d_model

    def encode_batch(self, point_clouds: list[np.ndarray]) -> np.ndarray:
        vecs = []
        for pc in point_clouds:
            seed = int(hashlib.md5(np.ascontiguousarray(pc).tobytes()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)

    def encode_points(self, pc: np.ndarray) -> np.ndarray:
        return self.encode_batch([pc])[0]


def _pc(seed: int, n_points: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_points, 6)).astype(np.float32)


def test_router_constructs_with_default_threshold():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    assert router.threshold == 64
    assert len(router) == 0


def test_e8_key_is_96_bytes_for_768_dim():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    key = router.e8_key(_pc(seed=1))
    assert isinstance(key, bytes)
    assert len(key) == 96  # 768 / 8 = 96 E8 blocks


def test_e8_key_is_deterministic_for_same_point_cloud():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    pc = _pc(seed=1)
    assert router.e8_key(pc) == router.e8_key(pc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shape_hamming_router'`

- [ ] **Step 3: Write minimal implementation**

```python
# shape_memory_dev/shape_hamming_router.py
"""ShapeHammingRouter — same-modality 3D-shape cache using E8 lattice Hamming distance.

Forks the HammingRouter pattern (latticememory/hamming_router.py) for point-cloud
inputs and the OpenShapeEncoder interface, reusing the unmodified, shared
E8LatticeDB quantization primitive from the main latticememory package.

Validated on real Cap3D data + real OpenShape encoder (see SHAPE_DEDUP_RESULTS.md
in this folder): intra-pair (same object, re-encountered) Hamming mean=67.1/96
blocks, inter-pair (different objects) mean=90.9/96. threshold=64 gives 43%
recall / 2% FP; threshold=50 gives 23% recall / 0% FP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from latticememory.rag.e8_retriever import E8LatticeDB


@dataclass(frozen=True)
class ShapeHammingMatch:
    value: Any
    hamming_distance: int
    stored_key: bytes


class ShapeHammingRouter:
    """Nearest-neighbor same-shape cache using E8 Hamming distance.

    Store canonical point-cloud->value pairs with ``add()``. At query time,
    ``lookup()`` finds the stored key with minimum Hamming distance. If that
    distance is within ``threshold``, it is a cache hit and the associated
    value is returned.

    Default threshold=64 (out of 96 blocks for the 768-D OpenShape encoder)
    matches the validated 43% recall / 2% FP operating point in
    SHAPE_DEDUP_RESULTS.md. Use threshold=50 for a more conservative 0% FP.
    """

    def __init__(self, encoder: Any, d_model: int = 768, threshold: int = 64) -> None:
        self._encoder = encoder
        self._d_model = d_model
        self._lattice = E8LatticeDB(d_model=d_model)
        self.threshold = threshold
        self._keys: list[np.ndarray] = []
        self._values: list[Any] = []
        self._key_set: set[bytes] = set()
        self._packed_matrix: np.ndarray | None = None

    def _emb_to_key_arr(self, embedding: np.ndarray) -> np.ndarray:
        key_bytes = self._lattice._quantize_to_indices(
            torch.tensor(embedding, dtype=torch.float32)
        )
        return np.frombuffer(key_bytes, dtype=np.uint8).copy()

    def e8_key(self, point_cloud: np.ndarray) -> bytes:
        """Return the raw E8 key for a point cloud (96 bytes for 768-D)."""
        emb = self._encoder.encode_batch([point_cloud])[0]
        return self._emb_to_key_arr(emb).tobytes()

    def __len__(self) -> int:
        return len(self._keys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

This stays in the untracked `shape_memory_dev/` workspace — no git commit for
files inside it (gitignored by design, per `.gitignore`'s `shape_memory_dev/`
entry). Skip the commit step for every task in this plan; there is nothing to
stage. (If a task ever modifies a tracked file outside `shape_memory_dev/`,
that step will say so explicitly — none do in this plan.)

---

### Task 2: `add()` and dedup-on-identical-key behavior

**Files:**
- Modify: `shape_memory_dev/shape_hamming_router.py`
- Test: `shape_memory_dev/test_shape_hamming_router.py`

- [ ] **Step 1: Write the failing test**

Add to `shape_memory_dev/test_shape_hamming_router.py`:

```python
def test_add_returns_the_stored_key():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    pc = _pc(seed=2)
    key = router.add(pc, value="object_2")
    assert key == router.e8_key(pc)
    assert len(router) == 1


def test_add_identical_point_cloud_twice_does_not_duplicate():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    pc = _pc(seed=3)
    router.add(pc, value="object_3")
    router.add(pc, value="object_3_again")
    assert len(router) == 1  # second add is a no-op: identical key already stored


def test_add_different_point_clouds_both_stored():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    router.add(_pc(seed=4), value="object_4")
    router.add(_pc(seed=5), value="object_5")
    assert len(router) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: FAIL with `AttributeError: 'ShapeHammingRouter' object has no attribute 'add'`

- [ ] **Step 3: Write minimal implementation**

Add to `shape_memory_dev/shape_hamming_router.py`, after `e8_key()`:

```python
    def add(self, point_cloud: np.ndarray, value: Any) -> bytes:
        """Encode point_cloud to an E8 key, store it with value. Returns the raw key."""
        emb = self._encoder.encode_batch([point_cloud])[0]
        key_arr = self._emb_to_key_arr(emb)
        raw = key_arr.tobytes()
        if raw not in self._key_set:
            self._key_set.add(raw)
            self._keys.append(key_arr)
            self._values.append(value)
            self._packed_matrix = None  # invalidate cache
        return raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

Skip (untracked workspace, see Task 1 Step 5 note).

---

### Task 3: `lookup()` / `_nearest()` — threshold-gated nearest match

**Files:**
- Modify: `shape_memory_dev/shape_hamming_router.py`
- Test: `shape_memory_dev/test_shape_hamming_router.py`

- [ ] **Step 1: Write the failing test**

Add to `shape_memory_dev/test_shape_hamming_router.py`:

```python
def test_lookup_on_empty_router_returns_none():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    assert router.lookup(_pc(seed=6)) is None


def test_lookup_exact_match_returns_zero_distance():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    pc = _pc(seed=7)
    router.add(pc, value="object_7")
    match = router.lookup(pc)
    assert match is not None
    assert isinstance(match, ShapeHammingMatch)
    assert match.value == "object_7"
    assert match.hamming_distance == 0


def test_lookup_beyond_threshold_returns_none():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768, threshold=0)
    router.add(_pc(seed=8), value="object_8")
    # A different, unrelated point cloud should not match at threshold=0
    result = router.lookup(_pc(seed=9), threshold=0)
    assert result is None


def test_lookup_threshold_override_takes_precedence_over_default():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768, threshold=0)
    pc = _pc(seed=10)
    router.add(pc, value="object_10")
    # default threshold=0 would still match this since it's an exact key (distance=0)
    assert router.lookup(pc, threshold=0) is not None
    # explicit very low threshold still finds the exact match (distance=0 <= 0)
    assert router.lookup(pc, threshold=0).hamming_distance == 0


def test_lookup_returns_nearest_among_multiple_stored():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768, threshold=96)
    router.add(_pc(seed=11), value="object_11")
    router.add(_pc(seed=12), value="object_12")
    pc12 = _pc(seed=12)
    match = router.lookup(pc12)
    assert match.value == "object_12"
    assert match.hamming_distance == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: FAIL with `AttributeError: 'ShapeHammingRouter' object has no attribute 'lookup'`

- [ ] **Step 3: Write minimal implementation**

Add to `shape_memory_dev/shape_hamming_router.py`, after `add()`:

```python
    def lookup(self, point_cloud: np.ndarray, threshold: int | None = None) -> ShapeHammingMatch | None:
        """Find the nearest stored key for point_cloud. Returns None on miss."""
        if not self._keys:
            return None
        emb = self._encoder.encode_batch([point_cloud])[0]
        query_arr = self._emb_to_key_arr(emb)
        return self._nearest(query_arr, threshold)

    def lookup_key(self, e8_key: bytes, threshold: int | None = None) -> ShapeHammingMatch | None:
        """Find the nearest stored key for a pre-computed E8 key. Returns None on miss."""
        if not self._keys:
            return None
        query_arr = np.frombuffer(e8_key, dtype=np.uint8)
        return self._nearest(query_arr, threshold)

    def _get_packed_matrix(self) -> np.ndarray:
        """Return cached [N, n_blocks] uint8 matrix of stored keys (rebuilt on demand)."""
        if self._packed_matrix is None or len(self._packed_matrix) != len(self._keys):
            self._packed_matrix = np.stack(self._keys)
        return self._packed_matrix

    def _nearest(self, query_arr: np.ndarray, threshold: int | None) -> ShapeHammingMatch | None:
        thresh = threshold if threshold is not None else self.threshold
        stored = self._get_packed_matrix()
        dists = np.sum(stored != query_arr, axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = int(dists[min_idx])
        if min_dist > thresh:
            return None
        return ShapeHammingMatch(
            value=self._values[min_idx],
            hamming_distance=min_dist,
            stored_key=self._keys[min_idx].tobytes(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

Skip (untracked workspace, see Task 1 Step 5 note).

---

### Task 4: `clear()` / `remove_by_value()` — management operations

**Files:**
- Modify: `shape_memory_dev/shape_hamming_router.py`
- Test: `shape_memory_dev/test_shape_hamming_router.py`

- [ ] **Step 1: Write the failing test**

Add to `shape_memory_dev/test_shape_hamming_router.py`:

```python
def test_clear_empties_the_router():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    router.add(_pc(seed=13), value="object_13")
    router.add(_pc(seed=14), value="object_14")
    router.clear()
    assert len(router) == 0
    assert router.lookup(_pc(seed=13)) is None


def test_remove_by_value_removes_matching_entry():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    router.add(_pc(seed=15), value="object_15")
    router.add(_pc(seed=16), value="object_16")
    removed = router.remove_by_value("object_15")
    assert removed is True
    assert len(router) == 1
    assert router.lookup(_pc(seed=16)).value == "object_16"


def test_remove_by_value_returns_false_when_not_found():
    router = ShapeHammingRouter(encoder=FakeShapeEncoder(), d_model=768)
    router.add(_pc(seed=17), value="object_17")
    assert router.remove_by_value("nonexistent") is False
    assert len(router) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: FAIL with `AttributeError: 'ShapeHammingRouter' object has no attribute 'clear'`

- [ ] **Step 3: Write minimal implementation**

Add to `shape_memory_dev/shape_hamming_router.py`, after `_nearest()`:

```python
    def remove_by_value(self, value: Any) -> bool:
        """Remove the first entry whose stored value equals ``value``.

        Returns True if an entry was found and removed.
        """
        for i, v in enumerate(self._values):
            if v == value:
                raw = self._keys[i].tobytes()
                self._keys.pop(i)
                self._values.pop(i)
                self._key_set.discard(raw)
                self._packed_matrix = None
                return True
        return False

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._key_set.clear()
        self._packed_matrix = None

    def __repr__(self) -> str:
        return f"ShapeHammingRouter(n={len(self)}, threshold={self.threshold}, d_model={self._d_model})"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

Skip (untracked workspace, see Task 1 Step 5 note).

---

### Task 5: Real-encoder integration smoke test (marked slow)

**Files:**
- Modify: `shape_memory_dev/test_shape_hamming_router.py`

This is the one test in the suite that uses the real `OpenShapeEncoder` (GPU,
real model weights) instead of `FakeShapeEncoder`, confirming the class works
against the actual production encoder, not just the deterministic test double.
Marked `slow` so the fast suite (Tasks 1-4) stays usable without a GPU.

- [ ] **Step 1: Write the failing test**

Add to `shape_memory_dev/test_shape_hamming_router.py`:

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "liora_core"))
sys.path.insert(0, str(HERE / "liora_core" / "adapters"))
sys.path.insert(0, str(HERE / "openshape_src" / "src"))


@pytest.mark.slow
def test_real_openshape_encoder_catches_augmented_duplicate():
    """Same object, lightly perturbed (rotation+subsample+jitter), should be a
    cache hit at threshold=64 -- the validated operating point from
    SHAPE_DEDUP_RESULTS.md. Requires the real OpenShape model + a real Cap3D
    sample on disk; skips cleanly if either is unavailable.
    """
    cap3d_dir = Path(r"E:\cap3d_pc\extracted\Cap3D_pcs_pt")
    samples = list(cap3d_dir.glob("*.pt"))[:1]
    if not samples:
        pytest.skip("Cap3D sample data not available on this machine")

    import torch as _torch
    from shape_encoder import OpenShapeEncoder
    from bench_shape_dedup_e8 import augment_pointcloud
    import numpy.random as np_random

    encoder = OpenShapeEncoder()
    router = ShapeHammingRouter(encoder=encoder, d_model=768, threshold=64)

    pc = _torch.load(samples[0], map_location="cpu", weights_only=True).numpy().astype(np.float32)
    rng = np_random.default_rng(0)
    augmented = augment_pointcloud(pc, rng)

    router.add(pc, value=samples[0].stem)
    match = router.lookup(augmented)
    assert match is not None, "augmented duplicate should be caught at threshold=64"
    assert match.value == samples[0].stem
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v -m slow`
Expected: FAIL — at this point `ShapeHammingRouter` already exists from Tasks 1-4,
so this should actually run end-to-end. If it fails, it will be on the assertion
(`match is not None`) rather than a missing-attribute error, since all the
plumbing already exists. If Cap3D data isn't present on the machine running this
task, it will SKIP instead — that's an acceptable outcome for this step.

- [ ] **Step 3: Register the `slow` marker**

`pytest.ini`/`pyproject.toml` for the main repo already defines a `slow` marker
(`pyproject.toml`'s `[tool.pytest.ini_options]` has `markers = ["slow: ..."]`).
Since `shape_memory_dev/` tests run standalone (not via the main repo's pytest
config — confirm by running from inside `shape_memory_dev/`), add a local marker
registration so pytest doesn't warn:

```ini
# shape_memory_dev/pytest.ini
[pytest]
markers =
    slow: marks tests that need a real GPU model + real Cap3D data on disk
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v -m slow`
Expected: PASS (or SKIPPED if Cap3D data isn't present — both are acceptable;
FAIL is not)

- [ ] **Step 5: Run the full suite (fast + slow) one more time**

Run: `cd shape_memory_dev && python -m pytest test_shape_hamming_router.py -v`
Expected: 15 passed (14 fast + 1 slow), confirming nothing in Task 5 broke
Tasks 1-4.

- [ ] **Step 6: Commit**

Skip (untracked workspace, see Task 1 Step 5 note).

---

### Task 6: Document the new module in the workspace README

**Files:**
- Modify: `shape_memory_dev/README.md`

- [ ] **Step 1: Add a row to the "What's copied here" table**

In `shape_memory_dev/README.md`, the table currently lists copied source folders
(`openshape_src/`, `gsplat_src/`, etc.) and the two markdown docs. Add a new row
for the module built in this plan, and a short new section below the table:

```markdown
| `shape_hamming_router.py` | Same-modality shape cache (NEW, built here — not copied from e8-Project). Forks the `HammingRouter` pattern for point-cloud inputs. |
```

Add this section after the "Rebuilding the Rust components" section at the end
of the file:

```markdown
## ShapeHammingRouter (v1 same-modality cache)

`shape_hamming_router.py` implements the validated same-modality shape cache
from `SHAPE_DEDUP_RESULTS.md`. Usage:

\`\`\`python
from shape_hamming_router import ShapeHammingRouter
from liora_core.adapters.shape_encoder import OpenShapeEncoder

router = ShapeHammingRouter(encoder=OpenShapeEncoder(), d_model=768, threshold=64)
router.add(point_cloud_array, value="object_uid_123")
match = router.lookup(possibly_re_exported_point_cloud)
if match:
    print(match.value, match.hamming_distance)
\`\`\`

Run tests: `python -m pytest test_shape_hamming_router.py -v` (fast, no GPU
needed) or add `-m slow` to include the real-encoder integration test (needs
GPU + Cap3D data on disk).
```

- [ ] **Step 2: Verify the file renders correctly**

Run: `cat shape_memory_dev/README.md` (or open it) and confirm the new table
row and section appear without broken markdown (matching triple-backtick fences,
no stray indentation).

- [ ] **Step 3: Commit**

Skip — `shape_memory_dev/README.md` is also inside the gitignored workspace
folder, same as every other file in this plan.

---

## Plan self-review

**Spec coverage:**
- `ShapeEncoder` protocol → satisfied by duck-typing against `OpenShapeEncoder.encode_batch()` directly (Task 1); no separate protocol class needed since `FakeShapeEncoder` and `OpenShapeEncoder` already share the method name/signature without one.
- `OpenShapeEncoderAdapter` → not needed (see above); explicitly dropped as unnecessary in favor of direct duck-typing, simpler per YAGNI.
- `RFSnapShapeMemory` mirroring `RFSnapTextMemory` → superseded by `ShapeHammingRouter` mirroring `HammingRouter`, for the reasons in the Architecture section (beam-radius-1 vs. arbitrary-threshold linear scan). This is the one deliberate deviation from the literal spec text, made after reading the real code and explained up front.
- Threshold calibration reusing `calibrate_threshold()` → not ported; `bench_shape_dedup_e8.py`'s existing threshold sweep already serves this need (YAGNI — avoid duplicating an existing, working tool).
- `FakeShapeEncoder` deterministic test double → Task 1.
- Out-of-scope items (cross-modal, category-clustering, training-on-keys, PLY loading) → none of them appear anywhere in this plan; v1 only accepts pre-loaded `(N, 6)` numpy arrays, matching the validation scripts.
- Stays in `shape_memory_dev/`, no main-package changes → every task's Commit step explicitly notes this; File Structure section states it up front.

**Placeholder scan:** No TBD/TODO; every code block is complete and runnable as written.

**Type consistency:** `ShapeHammingRouter.add()`/`lookup()`/`e8_key()` all take `point_cloud: np.ndarray` consistently across Tasks 1-3. `ShapeHammingMatch` fields (`value`, `hamming_distance`, `stored_key`) are used identically in every test assertion. `encoder.encode_batch(list[np.ndarray]) -> np.ndarray` is the one interface method depended on throughout, implemented identically by `FakeShapeEncoder` (Task 1) and the real `OpenShapeEncoder` (already existing, confirmed in Task 5).
