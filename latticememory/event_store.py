"""SQLite-backed durable event store for LatticeMemory observability."""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .observability import GeneratorTrace, RetrievalEvent

_BATCH_SIZE = 50
_BATCH_TIMEOUT = 0.1  # seconds


class LatticeEventStore:
    """Durable, queryable event store backed by SQLite.

    Writes are non-blocking: events are queued and committed in a background
    daemon thread.  WAL journal mode enables concurrent reads while the writer
    is active.  All DB access is serialised through a single Lock so the
    connection object can be shared between the writer thread and the read
    methods on the main thread (``check_same_thread=False``).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_schema(self._conn)
            self._conn.commit()

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_thread, daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                timestamp   REAL NOT NULL,
                request_id  TEXT,
                product     TEXT,
                dataset     TEXT,
                model_id    TEXT,
                index_id    TEXT,
                path        TEXT NOT NULL,
                candidate_count INTEGER,
                fallback_used   INTEGER NOT NULL DEFAULT 0,
                hit_count       INTEGER NOT NULL DEFAULT 0,
                top_doc_id      TEXT,
                latency_ms      REAL NOT NULL,
                query           TEXT,
                query_key       TEXT,
                stage_latency_ms TEXT,
                neighborhood    TEXT,
                quality_tags    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_timestamp  ON events (timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_path       ON events (path);
            CREATE INDEX IF NOT EXISTS idx_events_product    ON events (product);
            CREATE INDEX IF NOT EXISTS idx_events_model_id   ON events (model_id);
            CREATE INDEX IF NOT EXISTS idx_events_request_id ON events (request_id);

            CREATE TABLE IF NOT EXISTS generator_traces (
                trace_id        TEXT PRIMARY KEY,
                request_id      TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                prompt          TEXT,
                completion      TEXT,
                latency_ms      REAL,
                model_id        TEXT,
                quality_tags    TEXT,
                quality_score   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_traces_request_id ON generator_traces (request_id);
            CREATE INDEX IF NOT EXISTS idx_traces_timestamp  ON generator_traces (timestamp);
            """
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def insert(self, event: "RetrievalEvent") -> None:
        """Enqueue an event for background insertion (non-blocking)."""
        self._queue.put(("event", event))

    def insert_trace(self, trace: "GeneratorTrace") -> None:
        """Enqueue a generator trace for background insertion (non-blocking)."""
        self._queue.put(("trace", trace))

    def _writer_thread(self) -> None:
        """Background thread: drain queue, batch-write to SQLite."""
        while not self._stop.is_set():
            batch: list[Any] = []
            try:
                # Block waiting for first item
                item = self._queue.get(timeout=_BATCH_TIMEOUT)
                if item is None:
                    # Sentinel from close() or flush()
                    # Mark done and propagate stop signal
                    self._queue.task_done()
                    break
                batch.append(item)
            except queue.Empty:
                continue

            # Drain up to BATCH_SIZE more without blocking
            sentinel_seen = False
            while len(batch) < _BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                    if item is None:
                        # Sentinel encountered during batch drain.  Mark it done
                        # here (task_done for the sentinel itself) and set a flag
                        # so we break the outer loop after writing the current
                        # batch.  Re-queuing it would leave an unfinished task
                        # that queue.join() / flush() would wait on forever.
                        self._queue.task_done()
                        sentinel_seen = True
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            if batch:
                try:
                    event_rows = [self._event_to_row(p) for (tag, p) in batch if tag == "event"]
                    trace_rows = [self._trace_to_row(p) for (tag, p) in batch if tag == "trace"]
                    with self._lock:
                        if event_rows:
                            self._conn.executemany(
                                """
                                INSERT OR IGNORE INTO events (
                                    event_id, timestamp, request_id, product, dataset,
                                    model_id, index_id, path, candidate_count,
                                    fallback_used, hit_count, top_doc_id, latency_ms,
                                    query, query_key, stage_latency_ms, neighborhood,
                                    quality_tags
                                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """,
                                event_rows,
                            )
                        if trace_rows:
                            self._conn.executemany(
                                """
                                INSERT OR REPLACE INTO generator_traces (
                                    trace_id, request_id, timestamp, prompt, completion,
                                    latency_ms, model_id, quality_tags, quality_score
                                ) VALUES (?,?,?,?,?,?,?,?,?)
                                """,
                                trace_rows,
                            )
                        self._conn.commit()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "LatticeEventStore: batch write failed, %d items dropped", len(batch)
                    )
                finally:
                    for _ in batch:
                        self._queue.task_done()

            if sentinel_seen:
                break

    @staticmethod
    def _event_to_row(event: "RetrievalEvent") -> tuple:
        return (
            event.event_id,
            event.timestamp,
            event.request_id,
            event.product,
            event.dataset,
            event.model_id,
            event.index_id,
            event.path,
            event.candidate_count,
            1 if event.fallback_used else 0,
            event.hit_count,
            event.top_doc_id,
            event.latency_ms,
            event.query,
            json.dumps(event.query_key) if event.query_key is not None else None,
            json.dumps(event.stage_latency_ms),
            json.dumps(event.neighborhood),
            json.dumps(event.quality_tags),
        )

    @staticmethod
    def _trace_to_row(trace: "GeneratorTrace") -> tuple:
        return (
            trace.trace_id,
            trace.request_id,
            trace.timestamp,
            trace.prompt,
            trace.completion,
            trace.latency_ms,
            trace.model_id,
            json.dumps(trace.quality_tags),
            trace.quality_score,
        )

    # ------------------------------------------------------------------
    # Flush / close
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Block until all queued writes have been committed to SQLite."""
        self._queue.join()

    def close(self) -> None:
        """Signal the writer thread to stop, flush remaining events, close DB."""
        self._queue.put(None)  # sentinel
        self._writer.join(timeout=10)
        self._stop.set()
        if self._writer.is_alive():
            import logging
            logging.getLogger(__name__).warning(
                "LatticeEventStore: writer thread did not stop within 10 s; "
                "closing connection — pending writes may be lost"
            )
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Truncate
    # ------------------------------------------------------------------

    def truncate(self) -> None:
        """Delete all rows from events and generator_traces tables."""
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.execute("DELETE FROM generator_traces")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        path: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        model_id: str | None = None,
        index_id: str | None = None,
        request_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        quality_tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return events matching the supplied filters as a list of dicts."""
        clauses: list[str] = []
        params: list[Any] = []

        if path is not None:
            clauses.append("path = ?")
            params.append(path)
        if product is not None:
            clauses.append("product = ?")
            params.append(product)
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if index_id is not None:
            clauses.append("index_id = ?")
            params.append(index_id)
        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(request_id)
        if since_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(since_ts)
        if until_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(until_ts)
        if quality_tag is not None:
            # quality_tags is a JSON array; use a LIKE filter as fast approximation
            clauses.append("quality_tags LIKE ?")
            params.append(f'%"{quality_tag}"%')

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        result = []
        for row in rows:
            d = dict(row)
            # Deserialise JSON columns
            for col in ("query_key", "stage_latency_ms", "neighborhood", "quality_tags"):
                if d.get(col) is not None:
                    try:
                        d[col] = json.loads(d[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            d["fallback_used"] = bool(d["fallback_used"])
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def time_series(
        self,
        field: str,  # reserved placeholder for future metric selection; currently unused
        *,
        window_seconds: float = 3600,
        n_buckets: int = 60,
        product: str | None = None,
        model_id: str | None = None,
    ) -> list[dict]:
        """Bucket events into ``n_buckets`` equal slices over the last ``window_seconds``.

        Always returns exactly ``n_buckets`` dicts, with empty buckets
        having zero counts and 0.0 rates.
        """
        now = time.time()
        start = now - window_seconds
        bucket_size = window_seconds / n_buckets

        clauses = ["timestamp >= ?"]
        params: list[Any] = [start]
        if product is not None:
            clauses.append("product = ?")
            params.append(product)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"SELECT timestamp, fallback_used, hit_count, latency_ms, path FROM events {where}"

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        # Allocate bucket accumulators
        buckets: list[dict[str, Any]] = [
            {
                "timestamp_start": start + i * bucket_size,
                "event_count": 0,
                "fallback_count": 0,
                "hit_count": 0,
                "latencies": [],
                "path_counts": {},
            }
            for i in range(n_buckets)
        ]

        for row in rows:
            ts, fallback, hit, latency, path = (
                row["timestamp"],
                row["fallback_used"],
                row["hit_count"],
                row["latency_ms"],
                row["path"],
            )
            idx = int((ts - start) / bucket_size)
            idx = max(0, min(n_buckets - 1, idx))
            b = buckets[idx]
            b["event_count"] += 1
            b["fallback_count"] += fallback
            b["hit_count"] += (1 if hit > 0 else 0)
            b["latencies"].append(latency)
            b["path_counts"][path] = b["path_counts"].get(path, 0) + 1

        result = []
        for b in buckets:
            count = b["event_count"]
            latencies_sorted = sorted(b["latencies"])
            p50 = 0.0
            if latencies_sorted:
                mid = (len(latencies_sorted) - 1) // 2
                p50 = float(latencies_sorted[mid])
            result.append(
                {
                    "timestamp_start": b["timestamp_start"],
                    "event_count": count,
                    "fallback_rate": b["fallback_count"] / count if count else 0.0,
                    "hit_rate": b["hit_count"] / count if count else 0.0,
                    "p50_latency": p50,
                    "path_counts": b["path_counts"],
                }
            )
        return result

    def model_comparison(self, model_ids: list[str] | None = None) -> list[dict]:
        """Return per-model aggregate statistics sorted by event_count descending."""
        clauses: list[str] = []
        params: list[Any] = []
        if model_ids:
            placeholders = ",".join("?" * len(model_ids))
            clauses.append(f"model_id IN ({placeholders})")
            params.extend(model_ids)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # Single query: fetch all columns needed for grouping + p50 in one pass.
        # ORDER BY model_id so rows for each model are contiguous; latency_ms
        # ordering is done in Python after grouping.
        sql = f"""
            SELECT model_id, latency_ms, fallback_used, hit_count
            FROM events
            {where}
            ORDER BY model_id
        """

        with self._lock:
            cursor = self._conn.execute(sql, params)
            raw_rows = cursor.fetchall()

        # Group in Python: accumulate per-model stats in a single pass
        groups: dict[Any, dict[str, Any]] = {}
        for row in raw_rows:
            mid = row["model_id"]
            if mid not in groups:
                groups[mid] = {
                    "latencies": [],
                    "fallback_count": 0,
                    "total_hits": 0,
                }
            g = groups[mid]
            g["latencies"].append(row["latency_ms"])
            g["fallback_count"] += row["fallback_used"]
            g["total_hits"] += 1 if (row["hit_count"] or 0) > 0 else 0

        result = []
        for mid, g in groups.items():
            latencies_sorted = sorted(g["latencies"])
            count = len(latencies_sorted)
            avg_latency = sum(latencies_sorted) / count if count else 0.0
            p50 = 0.0
            if latencies_sorted:
                p50 = float(latencies_sorted[(count - 1) // 2])
            fallback_count = g["fallback_count"] or 0
            total_hits = g["total_hits"] or 0
            result.append(
                {
                    "model_id": mid,
                    "event_count": count,
                    "fallback_rate": fallback_count / count if count else 0.0,
                    "hit_rate": total_hits / count if count else 0.0,
                    "avg_latency": float(avg_latency),
                    "p50_latency": p50,
                }
            )

        # Sort by event_count descending to match original contract
        result.sort(key=lambda r: r["event_count"], reverse=True)
        return result

    def query_traces(
        self,
        *,
        request_id: str | None = None,
        model_id: str | None = None,
        quality_tag: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        since_ts: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Return generator traces matching the supplied filters as a list of dicts."""
        clauses: list[str] = []
        params: list[Any] = []

        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(request_id)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if quality_tag is not None:
            clauses.append("quality_tags LIKE ?")
            params.append(f'%"{quality_tag}"%')
        if min_score is not None:
            clauses.append("quality_score >= ?")
            params.append(min_score)
        if max_score is not None:
            clauses.append("quality_score <= ?")
            params.append(max_score)
        if since_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(since_ts)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM generator_traces {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        result = []
        for row in rows:
            d = dict(row)
            if d.get("quality_tags") is not None:
                try:
                    d["quality_tags"] = json.loads(d["quality_tags"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result

    def quality_correlation(
        self,
        *,
        model_id: str | None = None,
        since_ts: float | None = None,
    ) -> list[dict]:
        """Join events to generator_traces on request_id.

        Returns per-path breakdown: path, event_count, trace_count,
        avg_quality_score, fallback_rate.
        Only includes paths where at least one linked trace exists.
        Filters are applied to the events side (e.model_id, e.timestamp).
        """
        clauses: list[str] = []
        params: list[Any] = []

        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if since_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(since_ts)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            -- NOTE: trace attribution is per request_id, not per retrieval path.
            -- If request_id 'R' spans both lattice_exact and fallback paths,
            -- its traces are counted against both paths. trace_count/trace_rate
            -- should be interpreted as "traces linked to requests that touched this path."
            SELECT
                e.path,
                SUM(e.event_count)                                        AS event_count,
                SUM(gt.trace_count)                                       AS trace_count,
                CAST(SUM(gt.trace_count) AS REAL) / NULLIF(SUM(e.event_count), 0)
                                                                          AS trace_rate,
                SUM(e.fallback_sum * 1.0) / SUM(e.event_count)           AS fallback_rate,
                SUM(gt.score_sum) / NULLIF(SUM(gt.scored_count), 0)      AS avg_quality_score
            FROM (
                SELECT request_id, path,
                       COUNT(*)           AS event_count,
                       SUM(fallback_used) AS fallback_sum
                FROM events
                {where}
                GROUP BY request_id, path
            ) e
            INNER JOIN (
                SELECT request_id,
                       COUNT(*)               AS trace_count,
                       SUM(quality_score)     AS score_sum,
                       COUNT(quality_score)   AS scored_count
                FROM generator_traces
                GROUP BY request_id
            ) gt ON e.request_id = gt.request_id
            GROUP BY e.path
            ORDER BY event_count DESC
        """

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

