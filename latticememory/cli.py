"""LatticeMemory CLI — batch operations and cache management.

Commands
--------
lattice populate   Load Q&A pairs from CSV/JSON into a cache file.
lattice inspect    Print cache statistics for an existing cache file.
lattice gaps       Show top miss clusters from a flywheel log.
lattice drift      Detect drifting intents from a flywheel miss log.
lattice calibrate  Calibrate HammingRouter threshold from labeled pairs.
lattice review     Export/import flywheel miss clusters for ops review.
lattice federated  Aggregate miss-key histograms across multiple flywheel logs.
lattice serve      Start the proxy server.
lattice export     Export cache entries to a portable JSONL file.
lattice import     Import cache entries from a JSONL export.
lattice analytics  Fetch analytics from a running proxy.
lattice dedup      Deduplicate texts in a file using E8 lattice hashing.
lattice ide        Open the LatticeMemory terminal IDE.

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

    lattice ide
    lattice ide chat "Summarize this cache"
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
# calibrate
# ---------------------------------------------------------------------------

def _load_pair_file(path: Path) -> list[tuple[str, str]]:
    """Load 'text_a|||text_b' pairs from a text file, one pair per line."""
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|||")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a and b:
            pairs.append((a, b))
    return pairs


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate a HammingRouter threshold from labeled paraphrase/near-miss pairs."""
    metric = getattr(args, "metric", "hamming") or "hamming"
    if metric not in ("hamming", "cosine"):
        print(f"ERROR: --metric must be 'hamming' or 'cosine', got {metric!r}")
        return 1

    holdout_paraphrases_arg = getattr(args, "holdout_paraphrases", None)
    holdout_near_misses_arg = getattr(args, "holdout_near_misses", None)
    if bool(holdout_paraphrases_arg) != bool(holdout_near_misses_arg):
        print("ERROR: --holdout-paraphrases and --holdout-near-misses must be given together")
        return 1

    paraphrases_path = Path(args.paraphrases)
    near_misses_path = Path(args.near_misses)
    if not paraphrases_path.exists():
        print(f"ERROR: paraphrases file not found: {paraphrases_path}")
        return 1
    if not near_misses_path.exists():
        print(f"ERROR: near-misses file not found: {near_misses_path}")
        return 1

    paraphrase_pairs = _load_pair_file(paraphrases_path)
    near_miss_pairs = _load_pair_file(near_misses_path)
    if not paraphrase_pairs:
        print(f"ERROR: no valid pairs found in {paraphrases_path}")
        return 1
    if not near_miss_pairs:
        print(f"ERROR: no valid pairs found in {near_misses_path}")
        return 1

    holdout_paraphrase_pairs: list[tuple[str, str]] | None = None
    holdout_near_miss_pairs: list[tuple[str, str]] | None = None
    if holdout_paraphrases_arg and holdout_near_misses_arg:
        holdout_paraphrases_path = Path(holdout_paraphrases_arg)
        holdout_near_misses_path = Path(holdout_near_misses_arg)
        if not holdout_paraphrases_path.exists():
            print(f"ERROR: holdout paraphrases file not found: {holdout_paraphrases_path}")
            return 1
        if not holdout_near_misses_path.exists():
            print(f"ERROR: holdout near-misses file not found: {holdout_near_misses_path}")
            return 1
        holdout_paraphrase_pairs = _load_pair_file(holdout_paraphrases_path)
        holdout_near_miss_pairs = _load_pair_file(holdout_near_misses_path)
        if not holdout_paraphrase_pairs:
            print(f"ERROR: no valid pairs found in {holdout_paraphrases_path}")
            return 1
        if not holdout_near_miss_pairs:
            print(f"ERROR: no valid pairs found in {holdout_near_misses_path}")
            return 1

    from latticememory.hamming_router import HammingRouter

    print(f"Loading encoder: {args.encoder}")
    router = HammingRouter.from_model(args.encoder)

    unit = "Cosine" if metric == "cosine" else "Hamming"
    if metric == "cosine":
        print("Computing cosine similarity statistics...")
        gap_results = router.cosine_gap_stats(paraphrase_pairs, near_miss_pairs)
        cal_results = router.calibrate_cosine_threshold(
            paraphrase_pairs, near_miss_pairs, fp_budget=args.fp_budget
        )
        gap_label = "Gap (paraphrase_p5 - near_miss_p95)"
    else:
        print("Computing Hamming distance statistics...")
        gap_results = router.gap_stats(paraphrase_pairs, near_miss_pairs)
        cal_results = router.calibrate_threshold(
            paraphrase_pairs, near_miss_pairs, fp_budget=args.fp_budget
        )
        gap_label = "Hamming Gap (near_miss_p5 - paraphrase_p95)"

    print("\n=======================================================")
    print(f"                {unit.upper()} SIMILARITY STATISTICS            " if metric == "cosine" else "                HAMMING DISTANCE STATISTICS            ")
    print("=======================================================")
    print(f"Paraphrase pairs: {gap_results['n_paraphrase_pairs']}")
    print(f"  Min {unit}: {gap_results['paraphrase']['min']}")
    print(f"  P5 {unit}:  {gap_results['paraphrase']['p5']}")
    print(f"  Mean {unit}: {gap_results['paraphrase']['mean']}")
    print(f"  P95 {unit}: {gap_results['paraphrase']['p95']}")
    print(f"  Max {unit}: {gap_results['paraphrase']['max']}")
    print()
    print(f"Near-miss pairs: {gap_results['n_near_miss_pairs']}")
    print(f"  Min {unit}: {gap_results['near_miss']['min']}")
    print(f"  P5 {unit}:  {gap_results['near_miss']['p5']}")
    print(f"  Mean {unit}: {gap_results['near_miss']['mean']}")
    print(f"  P95 {unit}: {gap_results['near_miss']['p95']}")
    print(f"  Max {unit}: {gap_results['near_miss']['max']}")
    print()
    print(f"{gap_label}: {gap_results['gap']}")
    if gap_results["gap"] <= 0:
        print("  WARNING: Gap <= 0. No threshold gives FP=0 at non-zero recall.")
        print("  Consider training a snap encoder or raising --fp-budget.")
    print("=======================================================")

    print("\nTHRESHOLD SWEEP:")
    print("-------------------------------------------------------")
    print(f"{'Threshold':10} | {'Recall (TP Rate)':17} | {'FP Rate':10}")
    print("-------------------------------------------------------")
    for row in gap_results["threshold_table"]:
        marker = " <-- selected" if row["threshold"] == cal_results["threshold"] else ""
        print(f"{row['threshold']:<10} | {row['recall']:<17.3f} | {row['fp_rate']:<10.3f}{marker}")
    print("-------------------------------------------------------")

    print("\nRECOMMENDATION:")
    print("-------------------------------------------------------")
    if cal_results["threshold"] == -1:
        print("WARNING: No valid threshold satisfies the requested FP budget.")
        print(f"FP Budget: {args.fp_budget}")
    else:
        reliable = len(paraphrase_pairs) >= 100 and len(near_miss_pairs) >= 100
        print(f"Optimal Threshold: {cal_results['threshold']}")
        print(f"Expected Recall:   {cal_results['recall']:.2%}")
        print(f"Expected FP Rate:  {cal_results['fp_rate']:.2%}")
        print(f"FP Budget:         {cal_results['fp_budget']:.2%}")
        print(f"Reliable:          {'Yes' if reliable else 'No (< 100 pairs per type — not production-ready)'}")
    print("-------------------------------------------------------")

    held_out_results: dict | None = None
    if holdout_paraphrase_pairs and holdout_near_miss_pairs and cal_results["threshold"] != -1:
        held_out_results = router.evaluate_threshold(
            holdout_paraphrase_pairs, holdout_near_miss_pairs, cal_results["threshold"], metric=metric
        )
        print("\nHELD-OUT EVALUATION (pairs not used for calibration):")
        print("-------------------------------------------------------")
        print(f"Held-out paraphrase pairs: {held_out_results['n_paraphrase_pairs']}")
        print(f"Held-out near-miss pairs:  {held_out_results['n_near_miss_pairs']}")
        print(f"False accepts: {held_out_results['false_accepts']} ({held_out_results['false_accept_rate']:.2%})")
        print(f"False rejects: {held_out_results['false_rejects']} ({held_out_results['false_reject_rate']:.2%})")
        print("-------------------------------------------------------")
    elif cal_results["threshold"] != -1:
        print(
            "\nWARNING: this is an in-sample calibration only -- the threshold above was "
            "evaluated on the same pairs used to choose it, which always looks better than "
            "it performs on unseen data. Pass --holdout-paraphrases and "
            "--holdout-near-misses (pairs not used above) for genuine held-out evidence "
            "before trusting this threshold in production."
        )

    if args.export:
        import datetime

        from latticememory.hamming_router import compute_calibration_data_sha256

        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sha = compute_calibration_data_sha256(
            {"paraphrases": paraphrase_pairs, "near_misses": near_miss_pairs}
        )
        out_data = {
            # Matches the schema latticememory.proxy.LatticeLLMProxy requires to load
            # a file via LATTICE_CALIBRATION_DATA_PATH / calibration_data_path without
            # re-running calibration -- see validate_precalibrated_artifact_schema().
            # Keep this in sync with calibrate_proxy.py's --export shape.
            # Only the hamming metric's artifact_type is recognized by that loader --
            # cosine calibration has no live-loading path yet, so it gets a distinct
            # artifact_type that will not validate as a pre-calibrated Hamming artifact.
            "artifact_type": (
                "latticememory_hamming_cosine_calibration"
                if metric == "cosine"
                else "latticememory_hamming_calibration"
            ),
            "artifact_version": 1,
            "metric": metric,
            "model": args.encoder,
            "d_model": router._d_model,
            "calibration_data_sha256": sha,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "fp_budget": args.fp_budget,
            "calibration": cal_results,
            "gap_stats": gap_results,
        }
        if held_out_results is not None:
            out_data["held_out_evaluation"] = held_out_results
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote calibration JSON to: {args.export}")

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
    if args.hamming_rerank:
        os.environ["LATTICE_HAMMING_RERANK"] = "true"
    if args.hamming_rerank_model:
        os.environ["LATTICE_HAMMING_RERANK_MODEL"] = args.hamming_rerank_model
    if getattr(args, "hamming_rerank_retries", None) is not None:
        os.environ["LATTICE_HAMMING_RERANK_RETRIES"] = str(args.hamming_rerank_retries)
    if getattr(args, "hamming_rerank_retry_delay", None) is not None:
        os.environ["LATTICE_HAMMING_RERANK_RETRY_DELAY"] = str(args.hamming_rerank_retry_delay)
    if getattr(args, "hamming_cosine_gate", False):
        os.environ["LATTICE_HAMMING_COSINE_GATE"] = "true"
    if getattr(args, "hamming_cosine_threshold", None) is not None:
        os.environ["LATTICE_HAMMING_COSINE_THRESHOLD"] = str(args.hamming_cosine_threshold)
    if getattr(args, "cache_cosine_gate", False):
        os.environ["LATTICE_CACHE_COSINE_GATE"] = "true"
    if getattr(args, "cache_cosine_threshold", None) is not None:
        os.environ["LATTICE_CACHE_COSINE_THRESHOLD"] = str(args.cache_cosine_threshold)
    if getattr(args, "redis_url", None):
        os.environ["LATTICE_REDIS_URL"] = args.redis_url
    if getattr(args, "redis_namespace", None):
        os.environ["LATTICE_REDIS_NAMESPACE"] = args.redis_namespace
    if getattr(args, "pq_proof_dataset", None):
        os.environ["LATTICE_PQ_PROOF_DATASET"] = args.pq_proof_dataset
    if getattr(args, "pq_mode", False):
        os.environ["LATTICE_PQ_MODE"] = "true"
    if getattr(args, "pq_num_blocks", None) is not None:
        os.environ["LATTICE_PQ_NUM_BLOCKS"] = str(args.pq_num_blocks)
    if getattr(args, "pq_codebook_size", None) is not None:
        os.environ["LATTICE_PQ_CODEBOOK_SIZE"] = str(args.pq_codebook_size)
    if args.warm_path:
        os.environ["LATTICE_WARM_PATH"] = args.warm_path
    if args.admin_key:
        os.environ["LATTICE_ADMIN_KEY"] = args.admin_key

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. pip install 'lattice-memory-e8[proxy]'")
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
# review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    """Drive the flywheel review workflow: export a review queue or import answered reviews."""
    if args.review_command == "export":
        return _cmd_review_export(args)
    return _cmd_review_import(args)


def _cmd_review_export(args: argparse.Namespace) -> int:
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        return 1

    from latticememory.flywheel import LatticeFlywheel

    fw = LatticeFlywheel(log_path, cluster_threshold=args.threshold)
    n = fw.export_review_queue(args.output, n=args.top, min_cluster_size=args.min_size)
    print(f"Review queue exported: {args.output} ({n} items)")
    return 0


def _cmd_review_import(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is required. pip install sentence-transformers")
        return 1

    print(f"Loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)
    try:
        d = int(encoder.get_embedding_dimension())
    except AttributeError:
        import numpy as np
        probe = encoder.encode(["probe"])
        d = int(np.asarray(probe).shape[-1])

    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.memory import RFSnapLatticeMemory
    from latticememory.flywheel import LatticeFlywheel

    cache_path = Path(args.cache)
    sq_path = str(cache_path) if cache_path.suffix == ".db" else None
    lm = RFSnapLatticeMemory(d_model=d, sqlite_path=sq_path)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d, memory=lm)
    cache = RFSnapSemanticCache(runtime=runtime)

    fw = LatticeFlywheel(args.log)
    added = fw.load_reviewed(input_path, cache)
    print(f"Imported {added} reviewed entries into {cache_path}")
    return 0


# ---------------------------------------------------------------------------
# federated
# ---------------------------------------------------------------------------

def cmd_federated(args: argparse.Namespace) -> int:
    """Aggregate miss-key histograms across multiple flywheel logs (multi-node deployments)."""
    from latticememory.flywheel import LatticeFlywheel

    per_log: dict[str, list[dict]] = {}
    for log_arg in args.logs:
        log_path = Path(log_arg)
        if not log_path.exists():
            print(f"ERROR: log file not found: {log_path}")
            return 1
        fw = LatticeFlywheel(log_path)
        per_log[str(log_path)] = fw.federated_key_histogram()

    combined: dict[str, dict] = {}
    for log_name, rows in per_log.items():
        for row in rows:
            key = row["centroid_key_hex"]
            entry = combined.setdefault(key, {
                "centroid_key_hex": key,
                "count": 0,
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "nodes": [],
            })
            entry["count"] += row["count"]
            entry["first_seen"] = min(entry["first_seen"], row["first_seen"])
            entry["last_seen"] = max(entry["last_seen"], row["last_seen"])
            entry["nodes"].append(log_name)

    combined_rows = sorted(combined.values(), key=lambda r: -r["count"])

    print(f"\nFederated key histogram across {len(per_log)} node(s):\n")
    for log_name, rows in per_log.items():
        print(f"  {log_name}: {len(rows)} key clusters")

    print(f"\nCombined ({len(combined_rows)} unique key clusters):\n")
    print(f"  {'count':>6}  {'nodes':>5}  centroid_key_hex")
    print(f"  {'-'*6}  {'-'*5}  {'-'*16}")
    for row in combined_rows[:20]:
        print(f"  {row['count']:>6}  {len(row['nodes']):>5}  {row['centroid_key_hex'][:16]}...")

    if args.export:
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {"per_log": per_log, "combined": combined_rows}
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFederated histogram exported: {args.export}")

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


def cmd_ide(args: argparse.Namespace) -> int:
    """Open the LatticeMemory terminal IDE or run a one-shot IDE command."""
    from latticememory.ide.cli import main as ide_main

    return ide_main(args.ide_args)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
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

    # ---- review ----
    p_rev = sub.add_parser("review", help="Flywheel review workflow: export/import miss clusters for ops review")
    rev_sub = p_rev.add_subparsers(dest="review_command", required=True)

    p_rev_exp = rev_sub.add_parser("export", help="Export top miss clusters as a review queue JSON")
    p_rev_exp.add_argument("--log",       required=True,        help="Flywheel JSONL miss log")
    p_rev_exp.add_argument("--output",    required=True,        help="Destination review queue JSON path")
    p_rev_exp.add_argument("--top",       type=int, default=10, help="Number of top clusters to export")
    p_rev_exp.add_argument("--min-size",  type=int, default=3,  help="Minimum cluster size")
    p_rev_exp.add_argument("--threshold", type=int, default=25, help="Hamming cluster threshold (blocks)")

    p_rev_imp = rev_sub.add_parser("import", help="Import answered review items back into a cache")
    p_rev_imp.add_argument("--input",   required=True, help="Reviewed queue JSON (from `lattice review export`)")
    p_rev_imp.add_argument("--log",     required=True, help="Flywheel JSONL miss log (for context)")
    p_rev_imp.add_argument("--cache",   required=True, help="Destination cache file (.db or new)")
    p_rev_imp.add_argument("--encoder", default="dfrokido/bge-large-e8-snap", help="Encoder model")

    # ---- federated ----
    p_fed = sub.add_parser("federated", help="Aggregate miss-key histograms across multiple flywheel logs (multi-node)")
    p_fed.add_argument("--logs",   required=True, nargs="+", help="Flywheel JSONL miss logs (one or more, e.g. from multiple proxy replicas)")
    p_fed.add_argument("--export", default=None,             help="Optional path to write the combined histogram as JSON")

    # ---- calibrate ----
    p_cal = sub.add_parser("calibrate", help="Calibrate HammingRouter threshold from labeled paraphrase/near-miss pairs")
    p_cal.add_argument("--paraphrases",  required=True, help="File of same-intent pairs ('text_a|||text_b' per line)")
    p_cal.add_argument("--near-misses",  required=True, help="File of different-intent-but-similar pairs (same format)")
    p_cal.add_argument("--encoder",      default="dfrokido/bge-large-e8-snap", help="Encoder model")
    p_cal.add_argument("--fp-budget",    type=float, default=0.0, help="Maximum false-positive rate budget (default: 0.0 = zero FP)")
    p_cal.add_argument("--export",       default=None, help="Optional path to write full sweep + recommendation as JSON")
    p_cal.add_argument("--metric",       default="hamming", choices=["hamming", "cosine"], help="Calibrate a Hamming-distance threshold (for --hamming-mode) or a raw-embedding cosine threshold (for --hamming-cosine-gate)")
    p_cal.add_argument("--holdout-paraphrases", default=None, help="Paraphrase pairs NOT used for calibration, to report a genuine held-out false-reject rate (must be given with --holdout-near-misses)")
    p_cal.add_argument("--holdout-near-misses", default=None, help="Near-miss pairs NOT used for calibration, to report a genuine held-out false-accept rate (must be given with --holdout-paraphrases)")

    # ---- serve ----
    p_srv = sub.add_parser("serve", help="Start the proxy server")
    p_srv.add_argument("--key",          default=None,    help="OpenAI / LLM API key")
    p_srv.add_argument("--upstream",     default=None,    help="Upstream LLM URL")
    p_srv.add_argument("--cache",        default=None,    help="SQLite cache path")
    p_srv.add_argument("--miss-log",     default=None,    help="Flywheel miss log path")
    p_srv.add_argument("--hamming-mode", default=None,    help="serve | shadow | off")
    p_srv.add_argument("--hamming-rerank", action="store_true", help="LLM second-pass check on hamming_nn candidates before serving (off by default)")
    p_srv.add_argument("--hamming-rerank-model", default=None, help="Dedicated judge model for --hamming-rerank, distinct from the chat model. Use a small, fast, non-reasoning model -- a reasoning model can spend its whole token budget reasoning and never produce a verdict.")
    p_srv.add_argument("--hamming-rerank-retries", type=int, default=None, help="Bounded retries for --hamming-rerank when the judge call fails with a transient Ollama saturation error (maximum pending requests exceeded); default 1 if unset. Non-saturation errors and real NO verdicts are never retried.")
    p_srv.add_argument("--hamming-rerank-retry-delay", type=float, default=None, help="Seconds to wait before each --hamming-rerank-retries attempt; default 0.25 if unset.")
    p_srv.add_argument("--hamming-cosine-gate", action="store_true", help="Raw-embedding cosine gate on hamming_nn candidates before serving (off by default)")
    p_srv.add_argument("--hamming-cosine-threshold", type=float, default=None, help="Cosine threshold for --hamming-cosine-gate; calibrate from paraphrase/near-miss pairs before deployment")
    p_srv.add_argument("--cache-cosine-gate", action="store_true", help="Raw-embedding cosine gate on every cache hit before serving; rejects unsafe compressed-key hits")
    p_srv.add_argument("--cache-cosine-threshold", type=float, default=None, help="Cosine threshold for --cache-cosine-gate; default 0.999")
    p_srv.add_argument("--redis-url", default=None, help="Redis URL for shared cache storage, e.g. redis://localhost:6382/0")
    p_srv.add_argument("--redis-namespace", default=None, help="Redis key namespace for shared cache entries")
    p_srv.add_argument("--pq-proof-dataset", default=None, help="Support proof-pack JSONL to fit and seed a PQ-backed cache before serving")
    p_srv.add_argument("--pq-num-blocks", type=int, default=None, help="PQ num_blocks for --pq-proof-dataset or --pq-mode; default 8")
    p_srv.add_argument("--pq-codebook-size", type=int, default=None, help="PQ codebook_size for --pq-proof-dataset or --pq-mode; default 256")
    p_srv.add_argument("--pq-mode", action="store_true", help="Build a PQ-backed cache from --warm-path's Q&A pairs (requires --warm-path). Uses the validated default (8 blocks / 256-entry codebook) unless --pq-num-blocks/--pq-codebook-size override it. Distinct from --pq-proof-dataset, which reproduces the proof-pack's own benchmark schema; --pq-proof-dataset takes precedence if both are given.")
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

    # ---- ide ----
    p_ide = sub.add_parser("ide", help="Open the LatticeMemory terminal IDE", add_help=False)
    p_ide.add_argument("ide_args", nargs=argparse.REMAINDER)

    # ---- dedup ----
    p_ded = sub.add_parser("dedup", help="Deduplicate texts in a file using E8 lattice hashing")
    p_ded.add_argument("input", help="Input file (.txt, .json, .jsonl, .csv)")
    p_ded.add_argument("--output",    default=None,  help="Output file (default: input_deduped.<ext>)")
    p_ded.add_argument("--encoder",   default="dfrokido/bge-large-e8-snap", help="Encoder model")
    p_ded.add_argument("--text-col",  default="text", help="Column/key name for text field (JSON/CSV)")

    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "ide":
        from latticememory.ide.cli import main as ide_main

        return ide_main(sys.argv[2:])

    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "populate":  cmd_populate,
        "inspect":   cmd_inspect,
        "gaps":      cmd_gaps,
        "drift":     cmd_drift,
        "calibrate": cmd_calibrate,
        "review":    cmd_review,
        "federated": cmd_federated,
        "serve":     cmd_serve,
        "export":    cmd_export,
        "import":    cmd_import,
        "analytics": cmd_analytics,
        "dedup":     cmd_dedup,
        "ide":       cmd_ide,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
