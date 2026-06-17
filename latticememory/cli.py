"""LatticeMemory CLI — batch operations and cache management.

Commands
--------
lattice populate   Load Q&A pairs from CSV/JSON into a cache file.
lattice inspect    Print cache statistics for an existing cache file.
lattice gaps       Show top miss clusters from a flywheel log.
lattice drift      Detect drifting intents from a flywheel miss log.
lattice serve      Start the proxy server.
lattice export     Export cache entries to a portable JSONL file.
lattice import     Import cache entries from a JSONL export.
lattice analytics  Fetch analytics from a running proxy.
lattice dedup      Deduplicate texts in a file using E8 lattice hashing.

Usage
-----
    lattice populate --input qa.csv --encoder dfrokido/bge-large-e8-snap \\
                     --output cache.db --domain helpdesk

    lattice inspect --cache cache.db

    lattice gaps --log misses.jsonl --top 10

    lattice drift --log misses.jsonl --window 604800 --min-delta 5

    lattice export --cache cache.db --output export.jsonl
    lattice import --input export.jsonl --cache cache.db

    lattice serve --key sk-... --cache cache.db --port 8000

    lattice analytics --host localhost --port 8000

    lattice dedup corpus.jsonl --text-col text --output corpus_deduped.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _detect_d_model(sqlite_path: str | Path, fallback: int = 1024) -> int:
    """Infer the embedding dimension from the first row of a SQLite cache.

    The SQLite store saves embeddings as raw float32 blobs, so
    ``len(blob) // 4`` gives the dimension.  Returns ``fallback`` when the
    database is empty or does not exist.
    """
    import sqlite3

    db = str(sqlite_path)
    if not Path(db).exists():
        return fallback
    try:
        conn = sqlite3.connect(db, check_same_thread=False)
        row = conn.execute("SELECT embedding FROM documents LIMIT 1").fetchone()
        conn.close()
        if row and row[0]:
            return len(row[0]) // 4
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# populate
# ---------------------------------------------------------------------------

def cmd_populate(args: argparse.Namespace) -> int:
    """Load Q&A pairs from CSV or JSON into a SQLite cache file."""
    import time

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is required. pip install sentence-transformers")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    output_path = Path(args.output)

    # --- Load input data --------------------------------------------------
    pairs: list[dict] = []
    if input_path.suffix.lower() == ".csv":
        import csv
        with open(input_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(dict(row))
    elif input_path.suffix.lower() in (".json", ".jsonl"):
        text = input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() == ".jsonl":
            pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            pairs = data if isinstance(data, list) else [data]
    else:
        print(f"ERROR: unsupported input format. Use .csv, .json, or .jsonl")
        return 1

    q_col  = args.question_col
    a_col  = args.answer_col
    i_col  = args.intent_col

    valid = [p for p in pairs if p.get(q_col) and p.get(a_col)]
    if not valid:
        print(f"ERROR: no valid rows found. Expected columns: {q_col!r}, {a_col!r}")
        return 1

    print(f"Found {len(valid)} Q&A pairs in {input_path.name}")

    # --- Build cache ------------------------------------------------------
    print(f"Loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)

    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache

    try:
        d = int(encoder.get_embedding_dimension())
    except AttributeError:
        import numpy as np
        probe = encoder.encode(["probe"])
        d = int(np.asarray(probe).shape[-1])

    from latticememory.memory import RFSnapLatticeMemory
    sq_path = str(output_path) if output_path.suffix == ".db" else None
    lm = RFSnapLatticeMemory(d_model=d, sqlite_path=sq_path)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)

    print(f"Encoding and indexing {len(valid)} pairs...")
    t0 = time.perf_counter()
    batch_size = 64
    added = 0

    # Batch encode all questions for efficiency
    questions = [p[q_col].strip() for p in valid]
    embeddings_iter = _batch_encode(encoder, questions, batch_size=batch_size)

    for p, emb in zip(valid, embeddings_iter):
        question = p[q_col].strip()
        answer   = p[a_col].strip()
        intent_id = p.get(i_col, "").strip() or None
        meta: dict = {"source": "lattice_populate"}
        if intent_id:
            meta["intent_id"] = intent_id
        if args.domain:
            meta["domain"] = args.domain
        cache.put(question, value=answer, metadata=meta)
        added += 1
        if added % 100 == 0:
            pct = added / len(valid) * 100
            print(f"  {added}/{len(valid)} ({pct:.0f}%)", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"  {added}/{added} (100%)                     ")

    # --- Save to SQLite if requested --------------------------------------
    if output_path.suffix == ".db":
        # RFSnapTextMemory already wrote to the SQLite path above
        pass
    elif output_path.suffix in (".json", ""):
        # Dump entries as JSON for portability
        entries = []
        for entry in cache._entries.values():
            entries.append({
                "question": entry.prompt,
                "answer": entry.value,
                "lattice_key": entry.lattice_key.hex() if entry.lattice_key else None,
                "metadata": entry.metadata,
            })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. {added} entries cached in {elapsed:.1f}s ({added/elapsed:.0f} pairs/sec)")
    print(f"Output: {output_path} ({cache.size} unique E8 cells)")
    return 0


def _batch_encode(encoder, texts: list[str], batch_size: int = 64):
    """Yield embeddings for texts in batches."""
    import numpy as np
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = encoder.encode(batch, normalize_embeddings=True)
        yield from embs


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    """Print cache statistics for a SQLite cache file."""
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        return 1

    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.memory import RFSnapLatticeMemory

    # Lightweight load — no encoder needed for stats
    d = _detect_d_model(cache_path)
    lm = RFSnapLatticeMemory(d_model=d, sqlite_path=str(cache_path))
    runtime = RFSnapTextMemory(encoder=None, d_model=d, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)

    entries = list(cache._entries.values())
    total = len(entries)

    intent_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for e in entries:
        intent_id = e.metadata.get("intent_id") if e.metadata else None
        domain    = e.metadata.get("domain") if e.metadata else None
        if intent_id:
            intent_counts[intent_id] = intent_counts.get(intent_id, 0) + 1
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"\nCache: {cache_path}")
    print(f"  Total entries    : {total}")
    print(f"  Unique intents   : {len(intent_counts)}")
    print(f"  Domains          : {', '.join(domain_counts) or 'unset'}")

    if intent_counts and args.verbose:
        print(f"\n  Entries per intent:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {count:>5}  {intent}")

    if args.sample and entries:
        import random
        sample = random.sample(entries, min(args.sample, total))
        print(f"\n  Sample entries ({len(sample)}):")
        for e in sample:
            q = e.prompt[:80]
            a = str(e.value)[:60]
            print(f"    Q: {q}")
            print(f"    A: {a}")
            print()
    return 0


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------

def cmd_gaps(args: argparse.Namespace) -> int:
    """Show top miss clusters from a flywheel JSONL log."""
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        return 1

    from latticememory.flywheel import LatticeFlywheel

    fw = LatticeFlywheel(log_path, cluster_threshold=args.threshold)
    stats = fw.stats()
    print(f"\nFlywheel log: {log_path}")
    print(f"  Total misses   : {stats['total_misses']}")
    print(f"  Unique queries : {stats['unique_questions']}")

    clusters = fw.top_gaps(n=args.top, min_cluster_size=args.min_size)
    if not clusters:
        print(f"  No clusters found (min_cluster_size={args.min_size}, threshold={args.threshold})")
        return 0

    print(f"\n  Top {len(clusters)} miss clusters (coverage gaps):\n")
    for c in clusters:
        print(f"  [{c.size:>4} misses] {c.representative!r}")
        if args.verbose:
            for m in c.members[1:4]:
                print(f"            {m.question!r}")
        print()

    if args.export:
        n = fw.export_review_queue(args.export, n=args.top, min_cluster_size=args.min_size)
        print(f"  Review queue exported: {args.export} ({n} items)")

    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    """Start the LatticeMemory proxy server."""
    import os
    if args.key:
        os.environ["OPENAI_API_KEY"] = args.key
    if args.upstream:
        os.environ["LATTICE_UPSTREAM_URL"] = args.upstream
    if args.cache:
        os.environ["LATTICE_SQLITE_PATH"] = args.cache
    if args.miss_log:
        os.environ["LATTICE_MISS_LOG_PATH"] = args.miss_log
    if args.hamming_mode:
        os.environ["LATTICE_HAMMING_MODE"] = args.hamming_mode
    if args.warm_path:
        os.environ["LATTICE_WARM_PATH"] = args.warm_path
    if args.admin_key:
        os.environ["LATTICE_ADMIN_KEY"] = args.admin_key

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. pip install 'latticememory[proxy]'")
        return 1

    print(f"Starting LatticeMemory proxy on port {args.port}...")
    uvicorn.run(
        "latticememory.proxy_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    """Export all cache entries to a portable JSONL file."""
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        return 1

    output_path = Path(args.output)

    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.memory import RFSnapLatticeMemory

    d = _detect_d_model(cache_path)
    lm = RFSnapLatticeMemory(d_model=d, sqlite_path=str(cache_path))
    runtime = RFSnapTextMemory(encoder=None, d_model=d, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)

    entries = list(cache._entries.values())
    if not entries:
        print(f"Cache is empty: {cache_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in entries:
            record = {
                "cache_id":    e.cache_id,
                "prompt":      e.prompt,
                "value":       e.value,
                "lattice_key": e.lattice_key.hex() if e.lattice_key else None,
                "metadata":    e.metadata,
                "created_at":  e.created_at,
                "updated_at":  e.updated_at,
                "ttl_seconds": e.ttl_seconds,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(entries)} entries -> {output_path}")
    return 0


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def cmd_import(args: argparse.Namespace) -> int:
    """Import cache entries from a JSONL export into a cache file.

    Uses the stored lattice_key directly (no re-encoding needed).  Entries
    with a ``ttl_seconds`` that would already be expired are skipped by
    default (use --include-expired to override).
    """
    import time as _time

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache, SemanticCacheEntry
    from latticememory.memory import RFSnapLatticeMemory

    output_path = Path(args.cache)
    sq_path = str(output_path) if output_path.suffix == ".db" else None
    # Detect d_model from existing DB, or infer from the first import record
    d = _detect_d_model(output_path) if sq_path and Path(output_path).exists() else 1024
    lm = RFSnapLatticeMemory(d_model=d, sqlite_path=sq_path)
    runtime = RFSnapTextMemory(encoder=None, d_model=d, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)

    lines = [l for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    added = skipped_expired = skipped_dup = 0
    now = _time.time()

    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        ttl = rec.get("ttl_seconds")
        created_at = rec.get("created_at", now)
        if ttl is not None and not args.include_expired and now > created_at + ttl:
            skipped_expired += 1
            continue

        cache_id = rec.get("cache_id")
        if cache_id and cache_id in cache._entries and not args.overwrite:
            skipped_dup += 1
            continue

        lattice_key_hex = rec.get("lattice_key")
        lattice_key = bytes.fromhex(lattice_key_hex) if lattice_key_hex else b""

        entry = SemanticCacheEntry(
            cache_id=cache_id or cache._cache_id_for(lattice_key),
            prompt=rec["prompt"],
            value=rec["value"],
            lattice_key=lattice_key,
            metadata=rec.get("metadata", {}),
            created_at=created_at,
            updated_at=rec.get("updated_at", now),
            ttl_seconds=ttl,
        )
        # restore_entry writes to _entries, Hamming router, AND SQLite
        cache.restore_entry(entry)
        added += 1

    print(f"Imported {added} entries into {output_path}")
    if skipped_expired:
        print(f"  Skipped {skipped_expired} expired entries (use --include-expired to import them)")
    if skipped_dup:
        print(f"  Skipped {skipped_dup} duplicate entries (use --overwrite to replace)")
    return 0


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def cmd_dedup(args: argparse.Namespace) -> int:
    """Deduplicate texts in a file using E8 lattice hashing (O(N), no pairwise comparisons)."""
    import time

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    suffix = input_path.suffix.lower()
    texts: list[str] = []

    if suffix == ".txt":
        texts = [l.strip() for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif suffix in (".json", ".jsonl"):
        raw = input_path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        else:
            rows = json.loads(raw)
        if rows and isinstance(rows[0], str):
            texts = rows
        else:
            col = args.text_col
            texts = [str(r.get(col, "")).strip() for r in rows if r.get(col, "")]
    elif suffix == ".csv":
        import csv
        with open(input_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            texts = [str(r.get(args.text_col, "")).strip() for r in reader if r.get(args.text_col, "")]
    else:
        print("ERROR: unsupported format — use .txt, .json, .jsonl, or .csv")
        return 1

    if not texts:
        print(f"ERROR: no texts found in {input_path} (text-col={args.text_col!r})")
        return 1

    print(f"Loaded {len(texts)} texts from {input_path.name}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is required — pip install sentence-transformers")
        return 1

    from latticememory.dedup import LatticeDedup

    print(f"Loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)
    deduper = LatticeDedup(d_model=int(encoder.get_sentence_embedding_dimension()))
    deduper.encoder = encoder

    t0 = time.perf_counter()
    result = deduper.deduplicate(texts)
    elapsed = time.perf_counter() - t0

    unique = result["unique_documents"]
    n_removed = result.get("duplicate_count", len(texts) - len(unique))
    pct = n_removed / len(texts) * 100 if texts else 0.0

    print(f"Done in {elapsed:.1f}s")
    print(f"  Input:   {len(texts)}")
    print(f"  Unique:  {len(unique)}")
    print(f"  Removed: {n_removed} ({pct:.1f}%)")

    out = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_deduped")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".txt":
        out.write_text("\n".join(unique) + "\n", encoding="utf-8")
    elif out.suffix.lower() == ".jsonl":
        out.write_text("\n".join(json.dumps(t) for t in unique), encoding="utf-8")
    else:
        out.write_text(json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Output:  {out}")
    return 0


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------

def cmd_drift(args: argparse.Namespace) -> int:
    """Show intents whose miss rate is increasing (query drift detection)."""
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        return 1

    from latticememory.flywheel import LatticeFlywheel

    fw = LatticeFlywheel(log_path, cluster_threshold=args.threshold)
    drifting = fw.detect_drift(
        window_seconds=args.window,
        min_delta=args.min_delta,
        min_cluster_size=args.min_cluster_size,
    )

    stats = fw.stats()
    print(f"\nFlywheel log : {log_path}")
    print(f"  Total misses   : {stats['total_misses']}")
    print(f"  Unique queries : {stats['unique_questions']}")
    print(f"  Window (days)  : {args.window / 86400:.1f}")

    if not drifting:
        print(f"\n  No drifting intents detected (min_delta={args.min_delta}, "
              f"min_cluster_size={args.min_cluster_size})")
        recommend = fw.should_finetune(
            min_drifting_clusters=args.min_clusters,
            min_delta=args.min_delta,
            window_seconds=args.window,
            min_cluster_size=args.min_cluster_size,
        )
        print(f"  Recommend fine-tune : {'YES' if recommend else 'no'}")
        return 0

    print(f"\n  Drifting intents ({len(drifting)}):\n")
    print(f"  {'delta':>6}  {'recent':>7}  {'prev':>7}  representative")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*40}")
    for row in drifting:
        rep = row["representative"][:60]
        print(f"  {row['delta']:>+6}  {row['recent']:>7}  {row['previous']:>7}  {rep!r}")

    recommend = fw.should_finetune(
        min_drifting_clusters=args.min_clusters,
        min_delta=args.min_delta,
        window_seconds=args.window,
        min_cluster_size=args.min_cluster_size,
    )
    print(f"\n  Recommend fine-tune : {'YES — run `lattice finetune`' if recommend else 'no (below threshold)'}")

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(drifting, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Drift report exported: {args.export}")

    return 0


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

def cmd_analytics(args: argparse.Namespace) -> int:
    """Fetch and print analytics from a running proxy."""
    import urllib.request

    url = f"http://{args.host}:{args.port}/v1/analytics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"ERROR: could not reach proxy at {url}: {exc}")
        return 1

    print(json.dumps(data, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lattice",
        description="LatticeMemory CLI — batch cache operations and proxy management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- populate ----
    p_pop = sub.add_parser("populate", help="Load Q&A pairs from CSV/JSON into a cache file")
    p_pop.add_argument("--input",        required=True, help="Input file (.csv, .json, .jsonl)")
    p_pop.add_argument("--output",       required=True, help="Output SQLite file (.db) or JSON (.json)")
    p_pop.add_argument("--encoder",      default="dfrokido/bge-large-e8-snap", help="Encoder model")
    p_pop.add_argument("--question-col", default="question",  help="Column name for questions")
    p_pop.add_argument("--answer-col",   default="answer",    help="Column name for answers")
    p_pop.add_argument("--intent-col",   default="intent_id", help="Column name for intent labels")
    p_pop.add_argument("--domain",       default=None,        help="Domain tag stored in metadata")
    p_pop.add_argument("--batch-size",   type=int, default=64, help="Encoding batch size")

    # ---- inspect ----
    p_ins = sub.add_parser("inspect", help="Print statistics for an existing cache file")
    p_ins.add_argument("--cache",    required=True, help="Cache file (.db)")
    p_ins.add_argument("--verbose",  action="store_true", help="Show per-intent breakdown")
    p_ins.add_argument("--sample",   type=int, default=0, help="Show N random sample entries")

    # ---- gaps ----
    p_gap = sub.add_parser("gaps", help="Show top miss clusters from a flywheel log")
    p_gap.add_argument("--log",       required=True,       help="Flywheel JSONL miss log")
    p_gap.add_argument("--top",       type=int, default=10, help="Number of top clusters to show")
    p_gap.add_argument("--min-size",  type=int, default=3,  help="Minimum cluster size")
    p_gap.add_argument("--threshold", type=int, default=25, help="Hamming cluster threshold (blocks)")
    p_gap.add_argument("--verbose",   action="store_true",  help="Show sample questions per cluster")
    p_gap.add_argument("--export",    default=None,         help="Export review queue to this JSON path")

    # ---- serve ----
    p_srv = sub.add_parser("serve", help="Start the proxy server")
    p_srv.add_argument("--key",          default=None,    help="OpenAI / LLM API key")
    p_srv.add_argument("--upstream",     default=None,    help="Upstream LLM URL")
    p_srv.add_argument("--cache",        default=None,    help="SQLite cache path")
    p_srv.add_argument("--miss-log",     default=None,    help="Flywheel miss log path")
    p_srv.add_argument("--hamming-mode", default=None,    help="serve | shadow | off")
    p_srv.add_argument("--warm-path",    default=None,    help="CSV/JSON/JSONL file to pre-warm cache at startup")
    p_srv.add_argument("--admin-key",    default=None,    help="Secret key required for /v1/cache mutations")
    p_srv.add_argument("--host",         default="0.0.0.0")
    p_srv.add_argument("--port",         type=int, default=8000)
    p_srv.add_argument("--workers",      type=int, default=1)

    # ---- export ----
    p_exp = sub.add_parser("export", help="Export cache entries to a portable JSONL file")
    p_exp.add_argument("--cache",   required=True, help="Source SQLite cache (.db)")
    p_exp.add_argument("--output",  required=True, help="Destination JSONL file")

    # ---- import ----
    p_imp = sub.add_parser("import", help="Import cache entries from a JSONL export")
    p_imp.add_argument("--input",   required=True, help="Source JSONL file (from lattice export)")
    p_imp.add_argument("--cache",   required=True, help="Destination cache file (.db or new)")
    p_imp.add_argument("--overwrite",       action="store_true", help="Overwrite duplicate entries")
    p_imp.add_argument("--include-expired", action="store_true", help="Import TTL-expired entries too")

    # ---- drift ----
    p_drift = sub.add_parser("drift", help="Detect drifting intents from a flywheel miss log")
    p_drift.add_argument("--log",              required=True,               help="Flywheel JSONL miss log")
    p_drift.add_argument("--window",           type=float, default=7*86400, help="Half-window in seconds (default: 7 days)")
    p_drift.add_argument("--min-delta",        type=int,   default=5,       help="Min count increase to flag a cluster as drifting")
    p_drift.add_argument("--min-cluster-size", type=int,   default=3,       help="Min total cluster size to consider")
    p_drift.add_argument("--min-clusters",     type=int,   default=3,       help="Min drifting clusters to recommend fine-tuning")
    p_drift.add_argument("--threshold",        type=int,   default=25,      help="Hamming cluster threshold (blocks)")
    p_drift.add_argument("--export",           default=None,                help="Export drift report to this JSON path")

    # ---- analytics ----
    p_ana = sub.add_parser("analytics", help="Fetch analytics from a running proxy")
    p_ana.add_argument("--host", default="localhost")
    p_ana.add_argument("--port", type=int, default=8000)

    # ---- dedup ----
    p_ded = sub.add_parser("dedup", help="Deduplicate texts in a file using E8 lattice hashing")
    p_ded.add_argument("input", help="Input file (.txt, .json, .jsonl, .csv)")
    p_ded.add_argument("--output",    default=None,  help="Output file (default: input_deduped.<ext>)")
    p_ded.add_argument("--encoder",   default="dfrokido/bge-large-e8-snap", help="Encoder model")
    p_ded.add_argument("--text-col",  default="text", help="Column/key name for text field (JSON/CSV)")

    args = parser.parse_args()
    dispatch = {
        "populate":  cmd_populate,
        "inspect":   cmd_inspect,
        "gaps":      cmd_gaps,
        "drift":     cmd_drift,
        "serve":     cmd_serve,
        "export":    cmd_export,
        "import":    cmd_import,
        "analytics": cmd_analytics,
        "dedup":     cmd_dedup,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
