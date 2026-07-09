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

    if len(prompts) < pq_codebook_size:
        raise ValueError(
            f"build_pq_cache_from_qa_pairs needs at least pq_codebook_size={pq_codebook_size} "
            f"usable Q&A pairs to fit {pq_codebook_size} centroids, got only "
            f"{len(prompts)}. Supply more Q&A pairs, or pass a smaller "
            f"pq_codebook_size (--pq-codebook-size on the CLI) sized to your data."
        )

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
