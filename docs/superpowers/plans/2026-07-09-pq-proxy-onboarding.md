# PQ Proxy Onboarding & Design-Partner Quickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a design partner get the already-proven PQ+Redis proxy cache (99.17% hit rate, 0% FP on the real Bitext proof-pack) running against their own question/answer pairs, with the actually-validated PQ configuration as the default.

**Architecture:** A new small module, `latticememory/pq_seed.py`, owns the simple Q&A-pairs file format (extracted from the existing `--warm-path` loader) and a new function that builds a real-encoder, `PQLatticeDB`-backed `RFSnapSemanticCache` from that same format. `latticememory/proxy.py`'s existing `_warm_cache` is refactored to reuse the extracted loader instead of duplicating it. `latticememory/cli.py` gains a `--pq-mode` flag on `serve`; `latticememory/proxy_server.py`'s env-var defaults are corrected and its cache-construction branch is extended to call the new builder.

**Tech Stack:** Python 3.10+, existing LatticeMemory internals (`RFSnapLatticeMemory`, `RFSnapTextMemory`, `RFSnapSemanticCache`, `PQLatticeDB`), `sentence-transformers`, pytest with `FakeEncoder` fixtures (no real model download in tests).

## Global Constraints

- Validated PQ configuration is 8 blocks / 256-entry codebook (`LatticeIndex._DEFAULT_PQ_NUM_BLOCKS` / `_DEFAULT_PQ_CODEBOOK_SIZE`, described in `latticememory/index.py` as "the validated sweet spot from real-model testing"). Both new default constants must equal these exact values, not `PQLatticeDB`'s own raw class default (`num_blocks=16`), which is a different, unvalidated value.
- `--pq-proof-dataset` and its dataset schema are not modified, removed, or deprecated by this plan — they remain the reproducible-benchmark path.
- No change to `docker-compose.yml` — confirmed in the design spec that `LatticeRedisStore` never uses RediSearch, so plain `redis:7-alpine` is correct as shipped.
- No test may download the real `dfrokido/bge-large-e8-snap` model (~500MB). Use the existing `FakeEncoder` pattern (see `tests/test_lattice_index.py`) everywhere a `TextEncoder` is needed in a test.

---

## File Structure

- Create `latticememory/pq_seed.py`: `DEFAULT_PQ_NUM_BLOCKS`, `DEFAULT_PQ_CODEBOOK_SIZE` constants; `load_qa_pairs_file(path) -> list[dict]` (the Q&A-pairs file loader, extracted from `proxy.py`); `build_pq_cache_from_qa_pairs(qa_pairs, *, encoder, d_model, ...) -> RFSnapSemanticCache`; `build_pq_cache_from_qa_file(path, *, encoder_model, ...) -> RFSnapSemanticCache` (real-encoder convenience wrapper).
- Modify `latticememory/proxy.py`: `_warm_cache` delegates to `pq_seed.load_qa_pairs_file` instead of inlining CSV/JSON/JSONL parsing.
- Modify `latticememory/cli.py`: add `--pq-mode` to the `serve` subparser; fix `--pq-num-blocks`/`--pq-codebook-size` help text (currently says "default 4").
- Modify `latticememory/proxy_server.py`: import `pq_seed`'s default constants instead of hardcoding `"4"`; add the `--pq-mode` branch (fail-fast if `--warm-path` is absent; yields to `--pq-proof-dataset` with a logged warning if both are set).
- Create `docs/getting-started/design-partner-quickstart.md`: the three-stage walkthrough.
- Create `tests/test_pq_seed.py`.

---

### Task 1: Extract the Q&A-pairs file loader

**Files:**
- Create: `latticememory/pq_seed.py`
- Modify: `latticememory/proxy.py:344-399` (the `_warm_cache` method)
- Test: `tests/test_pq_seed.py`

**Interfaces:**
- Produces: `load_qa_pairs_file(path: str) -> list[dict]` — returns a list of dicts, one per row, with whatever keys the source file had (callers normalize `question`/`prompt` and `answer`/`value`/`response` themselves, matching the existing `_warm_cache` normalization). Returns `[]` and does not raise if the file is missing or the format is unsupported (matching `_warm_cache`'s current silent-skip-with-log behavior).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for latticememory.pq_seed's Q&A-pairs file loader."""
from __future__ import annotations

import json

from latticememory.pq_seed import load_qa_pairs_file


def test_load_qa_pairs_file_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"question": "What is the refund policy?", "answer": "30 days."}\n'
        '{"question": "How do I reset my password?", "answer": "Use the reset link."}\n',
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert len(rows) == 2
    assert rows[0]["question"] == "What is the refund policy?"
    assert rows[0]["answer"] == "30 days."


def test_load_qa_pairs_file_reads_json_list(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([{"prompt": "Hi", "value": "Hello!"}]),
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"prompt": "Hi", "value": "Hello!"}]


def test_load_qa_pairs_file_reads_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("question,answer,intent_id\nHi,Hello!,greeting\n", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"question": "Hi", "answer": "Hello!", "intent_id": "greeting"}]


def test_load_qa_pairs_file_missing_returns_empty(tmp_path):
    rows = load_qa_pairs_file(str(tmp_path / "does_not_exist.jsonl"))

    assert rows == []


def test_load_qa_pairs_file_unsupported_extension_returns_empty(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("not a supported format", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pq_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latticememory.pq_seed'`

- [ ] **Step 3: Create `latticememory/pq_seed.py` with the extracted loader**

```python
"""Q&A-pairs file format shared by --warm-path and PQ cache seeding.

This is the simple, general-purpose format: a list of {question, answer}
rows in CSV, JSON, or JSONL. It is deliberately NOT the proof-pack schema
(latticememory/proof_pack.py's cache_seed/calibration/evaluation/adversarial
splits) -- that schema is for reproducing the proof-pack benchmark; this
one is for a design partner's own data, which is just Q&A pairs.
"""
from __future__ import annotations

import csv as _csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PQ_NUM_BLOCKS = 8
DEFAULT_PQ_CODEBOOK_SIZE = 256


def load_qa_pairs_file(path: str) -> list[dict]:
    """Load Q&A pairs from a CSV, JSON, or JSONL file.

    CSV: columns become dict keys per row (DictReader). JSON: must be a
    list of dicts (a single dict is wrapped in a list). JSONL: one dict
    per line. Returns [] (with a logged warning) if the file is missing,
    unreadable, or has an unsupported extension -- callers decide what
    "no data" means for them, this function never raises for a bad input
    file.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("qa pairs file %s does not exist", path)
        return []

    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            with open(p, encoding="utf-8") as f:
                return [dict(row) for row in _csv.DictReader(f)]
        if suffix in (".json", ".jsonl"):
            text = p.read_text(encoding="utf-8")
            if suffix == ".jsonl":
                return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        logger.warning("qa pairs file %s: unsupported format %s", path, suffix)
        return []
    except Exception as exc:
        logger.warning("qa pairs file %s: failed to load: %s", path, exc)
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pq_seed.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Refactor `_warm_cache` to reuse the extracted loader**

In `latticememory/proxy.py`, replace the body of `_warm_cache` (currently lines 344-399, the CSV/JSON/JSONL parsing block) so it calls `load_qa_pairs_file` instead of inlining the same parsing logic:

```python
    def _warm_cache(self, path: str) -> int:
        """Load Q&A pairs from a file into the cache at startup.

        Supports CSV (columns: question, answer, intent_id), JSON (list of
        dicts), and JSONL (one dict per line).  Returns the number of entries
        added.  Missing answer/value is skipped with a warning.

        CSV columns ``question`` and ``answer`` are required; ``intent_id``,
        ``metadata``, and ``ttl_seconds`` are optional.
        """
        from latticememory.pq_seed import load_qa_pairs_file

        pairs = load_qa_pairs_file(path)

        added = 0
        for row in pairs:
            q = (row.get("question") or row.get("prompt") or "").strip()
            v = row.get("answer") or row.get("value") or row.get("response")
            if not q or v is None:
                continue
            meta: dict = {}
            if row.get("intent_id"):
                meta["intent_id"] = row["intent_id"]
            if row.get("metadata") and isinstance(row["metadata"], dict):
                meta.update(row["metadata"])
            ttl = row.get("ttl_seconds")
            self.cache.put(q, value=v, metadata=meta,
                           ttl_seconds=float(ttl) if ttl is not None else None)
            added += 1

        logger.info("warm_path: loaded %d entries from %s", added, path)
```

(The trailing `return added` line and any code after it in the original method body are unchanged -- only the parsing block above the `added = 0` line is replaced.)

- [ ] **Step 6: Run the full existing warm-path test coverage to confirm no regression**

Run: `python -m pytest tests/ -k warm -v`
Expected: PASS, same test count as before this change (the refactor must not change `_warm_cache`'s observable behavior)

- [ ] **Step 7: Commit**

```bash
git add latticememory/pq_seed.py latticememory/proxy.py tests/test_pq_seed.py
git commit -m "refactor: extract Q&A-pairs file loader into latticememory.pq_seed

Pulls the CSV/JSON/JSONL parsing out of LatticeLLMProxy._warm_cache into
a standalone, directly-testable load_qa_pairs_file() function. No
behavior change -- _warm_cache calls the extracted function instead of
inlining the same logic. This is the shared format the next task's PQ
cache builder reuses, instead of requiring the proof-pack's separate
adversarial-split schema."
```

---

### Task 2: PQ-backed cache builder from Q&A pairs

**Files:**
- Modify: `latticememory/pq_seed.py` (add to the file created in Task 1)
- Test: `tests/test_pq_seed.py` (extend)

**Interfaces:**
- Consumes: `load_qa_pairs_file` (Task 1, same file). `RFSnapLatticeMemory(d_model, sqlite_path=None, lattice=None)` and `RFSnapTextMemory(*, encoder, d_model, memory, model_id=None, batch_size=64)` from `latticememory.memory`/`latticememory.text_runtime`. `RFSnapSemanticCache(*, runtime, ...)` from `latticememory.semantic_cache`, whose `.put(prompt, *, value, metadata=None, ttl_seconds=None)` seeds one entry. `PQLatticeDB(d_model, num_blocks, codebook_size)` and its `.fit(embeddings: torch.Tensor)` from `latticememory.rag.pq_retriever`. `patch_cache_with_redis(cache, *, redis_url, namespace)` from `latticememory.redis_store`.
- Produces: `build_pq_cache_from_qa_pairs(qa_pairs, *, encoder, d_model=None, pq_num_blocks=DEFAULT_PQ_NUM_BLOCKS, pq_codebook_size=DEFAULT_PQ_CODEBOOK_SIZE, sqlite_path=None, redis_url=None, redis_namespace="lattice") -> RFSnapSemanticCache` and `build_pq_cache_from_qa_file(path, *, encoder_model="dfrokido/bge-large-e8-snap", **kwargs) -> RFSnapSemanticCache` (kwargs forwarded to the pairs-based builder). Task 3 calls `build_pq_cache_from_qa_file`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import hashlib


class FakeEncoder:
    """Deterministic fake encoder that hashes text to a fixed-dim vector.

    Same pattern as tests/test_lattice_index.py's FakeEncoder -- avoids
    downloading model weights in unit tests.
    """

    def __init__(self, d_model: int = 32):
        self.d_model = d_model

    def get_embedding_dimension(self):
        return self.d_model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        result = []
        for s in sentences:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            result.append(v)
        return np.stack(result)


def test_build_pq_cache_from_qa_pairs_seeds_entries():
    from latticememory.pq_seed import build_pq_cache_from_qa_pairs
    from latticememory.rag.pq_retriever import PQLatticeDB

    qa_pairs = [
        {"question": "What is the refund policy?", "answer": "30 days."},
        {"question": "How do I reset my password?", "answer": "Use the reset link."},
        {"question": "Where is my order?", "answer": "Check your email for tracking."},
        {"question": "Can I cancel my subscription?", "answer": "Yes, anytime."},
    ]

    cache = build_pq_cache_from_qa_pairs(
        qa_pairs, encoder=FakeEncoder(32), d_model=32, pq_num_blocks=4, pq_codebook_size=8,
    )

    assert isinstance(cache.runtime.memory.lattice, PQLatticeDB)
    result = cache.get("What is the refund policy?")
    assert result.hit is True
    assert result.value == "30 days."


def test_build_pq_cache_from_qa_pairs_uses_default_validated_pq_config():
    from latticememory.pq_seed import (
        DEFAULT_PQ_CODEBOOK_SIZE,
        DEFAULT_PQ_NUM_BLOCKS,
        build_pq_cache_from_qa_pairs,
    )

    qa_pairs = [{"question": "Hi", "answer": "Hello!"}]

    cache = build_pq_cache_from_qa_pairs(qa_pairs, encoder=FakeEncoder(32), d_model=32, pq_num_blocks=4, pq_codebook_size=8)
    lattice = cache.runtime.memory.lattice

    # Explicit args above override the defaults -- this test just proves
    # the DEFAULT_* constants exist and equal the validated sweet spot,
    # which is what Task 3's proxy_server.py wiring will actually rely on.
    assert DEFAULT_PQ_NUM_BLOCKS == 8
    assert DEFAULT_PQ_CODEBOOK_SIZE == 256
    assert lattice.num_blocks == 4  # explicit override was honored, not the default
    assert lattice.codebook_size == 8


def test_build_pq_cache_from_qa_pairs_empty_list_raises():
    from latticememory.pq_seed import build_pq_cache_from_qa_pairs

    import pytest
    with pytest.raises(ValueError, match="no Q&A pairs"):
        build_pq_cache_from_qa_pairs([], encoder=FakeEncoder(32), d_model=32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pq_seed.py -v -k build_pq_cache`
Expected: FAIL with `ImportError: cannot import name 'build_pq_cache_from_qa_pairs'`

- [ ] **Step 3: Add the builder functions to `latticememory/pq_seed.py`**

Append to the end of `latticememory/pq_seed.py` (the file created in Task 1):

```python
def build_pq_cache_from_qa_pairs(
    qa_pairs: list[dict],
    *,
    encoder,
    d_model: int | None = None,
    pq_num_blocks: int = DEFAULT_PQ_NUM_BLOCKS,
    pq_codebook_size: int = DEFAULT_PQ_CODEBOOK_SIZE,
    sqlite_path: str | None = None,
    redis_url: str | None = None,
    redis_namespace: str = "lattice",
    batch_size: int = 64,
):
    """Build a PQ-backed semantic cache from a list of {question, answer} rows.

    Unlike latticememory.proof_pack.build_seeded_pq_cache_from_support_jsonl
    (which requires the proof-pack's cache_seed/calibration/evaluation/
    adversarial schema and is explicitly documented as a proof/demo helper,
    not a general API), this takes the same simple shape --warm-path
    already accepts: a plain list of dicts with a question/prompt key and
    an answer/value/response key. Uses the real encoder passed in (a
    SentenceTransformer in production, a FakeEncoder in tests) -- never
    the proof-pack's synthetic _ProofPackEncoder.

    PQ codebooks are fit from the same rows being seeded: with no separate
    held-out calibration set, this is the entire point -- a design partner
    supplies one file of their own data, not a labeled four-way split.
    """
    from latticememory.memory import RFSnapLatticeMemory
    from latticememory.rag.pq_retriever import PQLatticeDB
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory

    prompts: list[str] = []
    answers: list[str] = []
    metadatas: list[dict] = []
    for row in qa_pairs:
        q = (row.get("question") or row.get("prompt") or "").strip()
        v = row.get("answer") or row.get("value") or row.get("response")
        if not q or v is None:
            continue
        meta: dict = {}
        if row.get("intent_id"):
            meta["intent_id"] = row["intent_id"]
        if row.get("metadata") and isinstance(row["metadata"], dict):
            meta.update(row["metadata"])
        prompts.append(q)
        answers.append(v)
        metadatas.append(meta)

    if not prompts:
        raise ValueError("no Q&A pairs to seed a PQ cache from -- got an empty or entirely-unusable list")

    runtime_dim = int(d_model or getattr(encoder, "get_embedding_dimension", lambda: 0)() or 0)
    if runtime_dim <= 0:
        probe = encoder.encode(["dimension probe"], batch_size=batch_size)
        runtime_dim = int(getattr(probe, "shape", [0, 0])[-1])

    pq = PQLatticeDB(d_model=runtime_dim, num_blocks=pq_num_blocks, codebook_size=pq_codebook_size)
    embeddings = encoder.encode(prompts, batch_size=batch_size)
    pq.fit(embeddings)

    memory = RFSnapLatticeMemory(d_model=runtime_dim, sqlite_path=sqlite_path, lattice=pq)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=runtime_dim, memory=memory, batch_size=batch_size)
    cache = RFSnapSemanticCache(runtime=runtime)

    for prompt, answer, meta in zip(prompts, answers, metadatas):
        cache.put(prompt, value=answer, metadata=meta)

    if redis_url:
        from latticememory.redis_store import patch_cache_with_redis
        patch_cache_with_redis(cache, redis_url=redis_url, namespace=redis_namespace)

    return cache


def build_pq_cache_from_qa_file(
    path: str,
    *,
    encoder_model: str = "dfrokido/bge-large-e8-snap",
    **kwargs,
):
    """Real-encoder convenience wrapper: load a Q&A-pairs file, build a PQ cache.

    This is what latticememory/proxy_server.py's --pq-mode branch calls.
    Loads the real SentenceTransformer named by encoder_model, exactly the
    same encoder-construction pattern as LatticeLLMProxy._build_runtime's
    default (non-PQ) path -- --pq-mode is meant to be a drop-in alternative
    cache backend for the same proxy, not a differently-encoded one.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_model)
    qa_pairs = load_qa_pairs_file(path)
    return build_pq_cache_from_qa_pairs(qa_pairs, encoder=encoder, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pq_seed.py -v`
Expected: PASS (8 tests total: 5 from Task 1, 3 new)

- [ ] **Step 5: Commit**

```bash
git add latticememory/pq_seed.py tests/test_pq_seed.py
git commit -m "feat: build a PQ-backed cache from plain Q&A pairs, not just the proof-pack schema

build_pq_cache_from_qa_pairs/build_pq_cache_from_qa_file give --pq-mode
(next task) a real, general-purpose way to get a PQLatticeDB-backed
RFSnapSemanticCache running: fit codebooks and seed entries from the
exact same simple {question, answer} shape --warm-path already accepts.
DEFAULT_PQ_NUM_BLOCKS=8 / DEFAULT_PQ_CODEBOOK_SIZE=256 match
LatticeIndex's own documented 'validated sweet spot from real-model
testing' -- not PQLatticeDB's own raw class default (num_blocks=16),
which was never validated for this use."
```

---

### Task 3: Wire `--pq-mode` into the CLI and proxy server

**Files:**
- Modify: `latticememory/cli.py:1054-1078` (the `serve` subparser)
- Modify: `latticememory/proxy_server.py:1-113`
- Test: `tests/test_cli_pq_mode_arg.py` (new)

**Interfaces:**
- Consumes: `pq_seed.DEFAULT_PQ_NUM_BLOCKS`, `pq_seed.DEFAULT_PQ_CODEBOOK_SIZE`, `pq_seed.build_pq_cache_from_qa_file` (Task 1 and 2).
- Produces: `lattice serve --pq-mode` as a documented, working flag; `proxy_server.py`'s `LATTICE_PQ_NUM_BLOCKS`/`LATTICE_PQ_CODEBOOK_SIZE` env-var defaults become `"8"`/`"256"`.

- [ ] **Step 1: Write the failing CLI argument-parsing test**

```python
"""Tests for the `lattice serve --pq-mode` CLI flag.

proxy_server.py itself is not unit tested (it does real network/model
work at module import time by design -- see the absence of any existing
test_proxy_server.py) -- this test covers argument parsing only, the
part that IS safely testable without starting a real server.
"""
from __future__ import annotations

from latticememory.cli import build_parser


def test_serve_accepts_pq_mode_flag():
    parser = build_parser()
    args = parser.parse_args(["serve", "--pq-mode", "--warm-path", "qa.jsonl"])

    assert args.pq_mode is True


def test_serve_pq_mode_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args(["serve"])

    assert args.pq_mode is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_pq_mode_arg.py -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'pq_mode'` (or a parser error if `build_parser` isn't the right entry point -- check `latticememory/cli.py` for how the parser is actually constructed and exposed if this import fails, and use that name instead; the parser-building function must already exist for `lattice serve --help` to work today)

- [ ] **Step 3: Add `--pq-mode` to the `serve` subparser**

In `latticememory/cli.py`, in the `serve` subparser block (around line 1071, right after the existing `--pq-proof-dataset`/`--pq-num-blocks`/`--pq-codebook-size` arguments), add:

```python
    p_srv.add_argument("--pq-mode", action="store_true", help="Build a PQ-backed cache from --warm-path's Q&A pairs (requires --warm-path). Uses the validated default (8 blocks / 256-entry codebook) unless --pq-num-blocks/--pq-codebook-size override it. Distinct from --pq-proof-dataset, which reproduces the proof-pack's own benchmark schema; --pq-proof-dataset takes precedence if both are given.")
```

Also fix the existing help text on the two lines directly above it (currently says "default 4" for both, which is the bug this plan exists to fix):

```python
    p_srv.add_argument("--pq-num-blocks", type=int, default=None, help="PQ num_blocks for --pq-proof-dataset or --pq-mode; default 8")
    p_srv.add_argument("--pq-codebook-size", type=int, default=None, help="PQ codebook_size for --pq-proof-dataset or --pq-mode; default 256")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_pq_mode_arg.py -v`
Expected: PASS

- [ ] **Step 5: Wire `--pq-mode` through to the environment variable, matching the existing `--pq-proof-dataset` pattern**

In `latticememory/cli.py`, find where `--pq-proof-dataset` is translated into `LATTICE_PQ_PROOF_DATASET` (line 539-540: `if getattr(args, "pq_proof_dataset", None): os.environ["LATTICE_PQ_PROOF_DATASET"] = args.pq_proof_dataset`). Add the same pattern immediately after it:

```python
    if getattr(args, "pq_mode", False):
        os.environ["LATTICE_PQ_MODE"] = "true"
```

- [ ] **Step 6: Fix the default mismatch and add the `--pq-mode` branch in `proxy_server.py`**

In `latticememory/proxy_server.py`, replace lines 38-40:

```python
pq_proof_dataset = os.getenv("LATTICE_PQ_PROOF_DATASET", None)
pq_num_blocks = int(os.getenv("LATTICE_PQ_NUM_BLOCKS", "4"))
pq_codebook_size = int(os.getenv("LATTICE_PQ_CODEBOOK_SIZE", "4"))
```

with:

```python
from latticememory.pq_seed import DEFAULT_PQ_CODEBOOK_SIZE, DEFAULT_PQ_NUM_BLOCKS

pq_proof_dataset = os.getenv("LATTICE_PQ_PROOF_DATASET", None)
pq_mode = os.getenv("LATTICE_PQ_MODE", "false").lower() == "true"
pq_num_blocks = int(os.getenv("LATTICE_PQ_NUM_BLOCKS", str(DEFAULT_PQ_NUM_BLOCKS)))
pq_codebook_size = int(os.getenv("LATTICE_PQ_CODEBOOK_SIZE", str(DEFAULT_PQ_CODEBOOK_SIZE)))
```

Then replace the existing `pq_proof_dataset` handling block (lines 55-79, the `semantic_cache = None` / `pq_proof = {"enabled": False}` block through the end of the `if pq_proof_dataset:` block) with:

```python
semantic_cache = None
pq_proof = {"enabled": False}
if pq_proof_dataset:
    dataset_path = Path(pq_proof_dataset)
    if not dataset_path.exists():
        raise RuntimeError(f"LATTICE_PQ_PROOF_DATASET does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise RuntimeError(f"LATTICE_PQ_PROOF_DATASET is not a file: {dataset_path}")
    from latticememory.proof_pack import build_seeded_pq_cache_from_support_jsonl

    if pq_mode:
        warnings.warn(
            "Both --pq-proof-dataset and --pq-mode were given -- "
            "--pq-proof-dataset takes precedence, --pq-mode is ignored.",
            RuntimeWarning,
            stacklevel=1,
        )

    semantic_cache = build_seeded_pq_cache_from_support_jsonl(
        pq_proof_dataset,
        redis_url=redis_url,
        redis_namespace=redis_namespace,
        pq_num_blocks=pq_num_blocks,
        pq_codebook_size=pq_codebook_size,
        flush_redis=True,
    )
    pq_proof = {
        "enabled": True,
        "dataset_path": str(dataset_path),
        "num_blocks": pq_num_blocks,
        "codebook_size": pq_codebook_size,
        "seeded_entries": semantic_cache.size,
        "mode": "proof_demo",
    }
elif pq_mode:
    if not warm_path:
        raise RuntimeError(
            "--pq-mode requires --warm-path: PQ codebooks are fit from the "
            "warm-start file's entries, and there's nothing to fit from "
            "without one."
        )
    from latticememory.pq_seed import build_pq_cache_from_qa_file

    semantic_cache = build_pq_cache_from_qa_file(
        warm_path,
        encoder_model=encoder_model,
        pq_num_blocks=pq_num_blocks,
        pq_codebook_size=pq_codebook_size,
        sqlite_path=sqlite_path,
        redis_url=redis_url,
        redis_namespace=redis_namespace,
    )
    pq_proof = {
        "enabled": True,
        "num_blocks": pq_num_blocks,
        "codebook_size": pq_codebook_size,
        "seeded_entries": semantic_cache.size,
        "mode": "pq_mode",
    }
```

Note the `warnings` import already exists earlier in this file (used for the `OPENAI_API_KEY` warning) -- no new import needed for the `warnings.warn` call above.

**Resolved, not left open:** `LatticeLLMProxy.__init__` runs `self._warm_cache(warm_path)` unconditionally whenever `warm_path is not None` (see `latticememory/proxy.py:340-342`) -- it does **not** check whether `semantic_cache` already came pre-seeded. Since `--pq-mode` builds `semantic_cache` from the same `warm_path` file, passing `warm_path` through unchanged to the `LatticeLLMProxy(...)` constructor call below would re-seed every entry a second time (harmless, since `RFSnapSemanticCache.put()` upserts by lattice key rather than appending, but a wasted second encode-and-fit pass at every startup for no benefit). Fix: find the existing `proxy = LatticeLLMProxy(...)` call further down in this file (the block that currently passes `warm_path=warm_path`) and change that one argument to pass `warm_path=(None if (pq_mode and semantic_cache is not None) else warm_path)` instead -- when `--pq-mode` already built and seeded the cache, don't ask `LatticeLLMProxy` to warm-start it again from the same file. Leave the `--pq-proof-dataset` case exactly as it behaves today (this plan does not change it, and it doesn't hit this same collision in practice since `--pq-proof-dataset` and `--warm-path` are independent flags nobody sets together).

- [ ] **Step 7: Manually verify the real command works end to end**

This step cannot be automated in this plan (it requires downloading the real encoder model and starting a real server) -- run it manually once during implementation, per the design spec's testing section:

```bash
cat > /tmp/design_partner_qa.jsonl <<'EOF'
{"question": "What is the refund policy?", "answer": "30-day returns, full refund."}
{"question": "How do I reset my password?", "answer": "Use the reset link on the login page."}
{"question": "Where is my order?", "answer": "Check your email for the tracking link."}
EOF
LATTICE_WARM_PATH=/tmp/design_partner_qa.jsonl LATTICE_PQ_MODE=true \
  python -m uvicorn latticememory.proxy_server:app --port 8000 &
sleep 5
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is the refund policy?"}]}' \
  | grep -o '"X-Lattice-Cache"'
```

Expected: server starts without error, `/health` returns 200, and the response headers show a cache hit (`X-Lattice-Cache: HIT`) for the exact-seeded prompt. Note whatever actually happens (including any bug found) in the task's implementation report -- do not mark this step done without actually running it.

- [ ] **Step 8: Run the full test suite to confirm no regression**

Run: `python -m pytest tests/ -q`
Expected: all tests pass except the same pre-existing unrelated failure seen throughout this project's history on this machine (a `transformers`/Windows environment `ModuleNotFoundError` unrelated to this change, if present -- confirm any failure is that specific pre-existing one, not something this task introduced)

- [ ] **Step 9: Commit**

```bash
git add latticememory/cli.py latticememory/proxy_server.py tests/test_cli_pq_mode_arg.py
git commit -m "feat: add lattice serve --pq-mode, fix PQ default mismatch

--pq-mode builds a PQ-backed cache from --warm-path's Q&A pairs via
pq_seed.build_pq_cache_from_qa_file, instead of requiring
--pq-proof-dataset's proof-pack-only schema. Fails fast with a clear
RuntimeError if --warm-path is absent, since there's nothing to fit PQ
codebooks from otherwise. --pq-proof-dataset still wins if both flags
are given (with a logged warning) -- it remains the reproducible-
benchmark path, unchanged.

Also fixes the actual bug motivating this plan: LATTICE_PQ_NUM_BLOCKS/
LATTICE_PQ_CODEBOOK_SIZE defaulted to 4/4, not the 8/256 documented
elsewhere as the validated configuration. Both now default from
pq_seed.DEFAULT_PQ_NUM_BLOCKS/DEFAULT_PQ_CODEBOOK_SIZE."
```

---

### Task 4: Design-partner quickstart doc

**Files:**
- Create: `docs/getting-started/design-partner-quickstart.md`

**Interfaces:**
- Consumes: `--pq-mode` and the corrected defaults (Task 3), `lattice calibrate` (already exists), `docker-compose up` (already correct, per the design spec).

- [ ] **Step 1: Write the quickstart doc**

```markdown
# Design Partner Quickstart

Three stages, in order of increasing setup cost and increasing hit rate.
Start at stage 1 -- it works with zero configuration -- and move to later
stages only once you want more cache hits than exact-repeat traffic gives
you. Every number below is from the real proof-pack run against a public
customer-support dataset (`docs/proxy_pq_redis_flywheel_proof_pack_2026-07-03.md`),
not a synthetic benchmark.

## Stage 1: Exact-repeat cache (zero config)

```bash
OPENAI_API_KEY=sk-... docker-compose up
```

Point your OpenAI client at `http://localhost:8000` instead of
`https://api.openai.com/v1`. Identical repeat prompts are served from
cache; everything else calls through to OpenAI as normal. This is safe
by construction -- there is no approximate matching in this stage, so
there is no false-positive risk. In the proof-pack's real workload, exact
matching alone caught 33-42% of traffic (see the `exact_string` rows in
the proof-pack doc).

## Stage 2: Calibrated paraphrase matching

Exact matching misses paraphrases ("What's your return policy?" vs. "What
is the refund policy?"). To catch those safely, calibrate a similarity
threshold from a small set of your own paraphrase and near-miss examples:

```bash
lattice calibrate \
  --paraphrases your_paraphrases.txt \
  --near-misses your_near_misses.txt \
  --metric cosine --fp-budget 0
```

Each file is `text_a|||text_b` per line -- `your_paraphrases.txt` holds
pairs that mean the same thing, `your_near_misses.txt` holds pairs that
are similar wording but a genuinely different question (these teach the
calibration where NOT to match). A few dozen pairs of each is enough to
start; more improves the calibrated threshold.

```bash
docker-compose run latticememory-proxy \
  lattice serve --hamming-mode serve --hamming-cosine-gate \
  --hamming-cosine-threshold <the threshold lattice calibrate printed>
```

## Stage 3: PQ + Redis (the fully-validated path)

This is the configuration the proof-pack measured directly: 99.17% hit
rate, 0% measured false positives, on a real 6,000-request customer-
support replay. It needs one file of your own question/answer pairs --
not a labeled four-way split, just what a support team already has:

```jsonl
{"question": "What is your refund policy?", "answer": "30-day returns, full refund to original payment method."}
{"question": "How do I reset my password?", "answer": "Use the 'Forgot password' link on the sign-in page."}
```

Save that as `qa_pairs.jsonl`, then:

```bash
docker-compose run \
  -e LATTICE_WARM_PATH=/data/qa_pairs.jsonl \
  -e LATTICE_PQ_MODE=true \
  -e LATTICE_REDIS_URL=redis://redis:6379/0 \
  --profile with-redis \
  latticememory-proxy
```

PQ codebooks are fit from the same file being seeded -- there's no
separate calibration step for this stage. The defaults (8 blocks, 256
codewords per block) are the configuration validated against real data;
you don't need to tune them to get the proof-pack's numbers. `--pq-mode`
requires `--warm-path` (equivalently, `LATTICE_WARM_PATH`) -- there's
nothing to fit codebooks from otherwise, and the proxy will fail to start
with a clear error if it's missing rather than silently serving an empty
cache.

## What to expect, honestly

The 99.17% figure is against a workload where most traffic really is
repeated or lightly-paraphrased customer-support questions -- if your
traffic is more open-ended, expect a lower number, and check
`/v1/analytics` on your running proxy for your own real hit rate rather
than assuming the proof-pack's number transfers directly. Asymmetric
workloads (a user question against a much longer document, not another
short question) are not what this stage is for -- see the main README's
"What it's for" table.
```

- [ ] **Step 2: Verify every command in the doc matches what Task 3 actually built**

Re-read the doc against `latticememory/cli.py` and `latticememory/proxy_server.py` as they now stand (post-Task-3) and confirm every flag name, env var name, and file format shown is exactly what the code accepts -- this is the one doc a copy-paste error in would actually hurt a design partner, so check it by hand, don't assume the draft above is already correct.

- [ ] **Step 3: Commit**

```bash
git add docs/getting-started/design-partner-quickstart.md
git commit -m "docs: design-partner quickstart for the proxy cache

Three stages (exact-only, calibrated Hamming, PQ+Redis) in order of
setup cost, each citing real proof-pack numbers instead of synthetic
ones. Closes the gap the design spec identified: the proven proxy+PQ+
Redis path (99.17% hit rate, 0% FP on real data) previously had no
onboarding doc a design partner could actually follow."
```

---

## Plan self-review notes

- **Spec coverage:** all three concrete deliverables from the design spec are covered — default fix (Task 3), general-purpose PQ seeding via `--warm-path`'s format (Tasks 1-2), quickstart doc (Task 4). The spec's stated precedence rule (`--pq-proof-dataset` wins if both flags given) and fail-fast requirement (`--pq-mode` without `--warm-path`) are both implemented in Task 3 Step 6, matching the spec's Architecture section verbatim.
- **Type consistency:** `build_pq_cache_from_qa_pairs`/`build_pq_cache_from_qa_file` names and signatures are identical everywhere they're referenced (Task 2's Interfaces block, Task 3's Step 6 code, Task 3's Interfaces block). `DEFAULT_PQ_NUM_BLOCKS`/`DEFAULT_PQ_CODEBOOK_SIZE` values (8/256) are consistent across the Global Constraints section, Task 2's test, and Task 3's `proxy_server.py` change.
- **Double-seeding risk checked against the real code, not left as a guess:** confirmed by reading `latticememory/proxy.py:340-342` directly that `LatticeLLMProxy.__init__` runs `_warm_cache(warm_path)` unconditionally whenever `warm_path is not None`, with no check for whether `semantic_cache` was already pre-seeded. Task 3 Step 6 has the concrete fix (pass `warm_path=None` to the constructor when `--pq-mode` already built the cache from that same file), not a placeholder asking the implementer to figure it out.
