from __future__ import annotations

import json
import statistics
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, round((len(values) - 1) * p))
    return float(values[idx])


@dataclass(frozen=True)
class RetrievalEvent:
    query: str
    path: str
    latency_ms: float
    fallback_used: bool
    candidate_count: int
    hit_count: int
    top_doc_id: str | None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    request_id: str | None = None
    product: str | None = None
    dataset: str | None = None
    model_id: str | None = None
    index_id: str | None = None
    query_key: list[int] | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    neighborhood: list[dict] = field(default_factory=list)
    quality_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalEvent":
        return cls(
            query=data["query"],
            path=data["path"],
            latency_ms=float(data["latency_ms"]),
            fallback_used=bool(data["fallback_used"]),
            candidate_count=int(data["candidate_count"]),
            hit_count=int(data["hit_count"]),
            top_doc_id=data.get("top_doc_id"),
            event_id=data.get("event_id") or uuid4().hex,
            timestamp=float(data.get("timestamp", time.time())),
            request_id=data.get("request_id"),
            product=data.get("product"),
            dataset=data.get("dataset"),
            model_id=data.get("model_id"),
            index_id=data.get("index_id"),
            query_key=list(data["query_key"]) if data.get("query_key") is not None else None,
            stage_latency_ms={str(k): float(v) for k, v in data.get("stage_latency_ms", {}).items()},
            neighborhood=list(data.get("neighborhood", [])),
            quality_tags=list(data.get("quality_tags", [])),
        )


@dataclass(frozen=True)
class GeneratorTrace:
    request_id: str
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    prompt: str | None = None
    completion: str | None = None
    latency_ms: float | None = None
    model_id: str | None = None
    quality_tags: list[str] = field(default_factory=list)
    quality_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LatticeObservability:
    """In-memory metrics stream for RF-Snap LatticeMemory.

    This is intentionally API/dashboard-ready: callers can inspect raw recent
    events for debugging and summaries for health checks or monitoring.

    Optional ``db_path`` enables a SQLite-backed durable event store.  Both
    the in-memory deque and the SQLite store are populated simultaneously;
    the JSONL ``event_log_path`` can coexist with either or both.
    """

    def __init__(
        self,
        max_events: int = 10_000,
        event_log_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self.event_log_path = Path(event_log_path) if event_log_path is not None else None
        self._events: deque[RetrievalEvent] = deque(maxlen=max_events)
        self._recall_audits: list[float] = []
        self._drift_audits: list[float] = []

        if db_path is not None:
            from .event_store import LatticeEventStore  # lazy to avoid any circular risk
            self._store: "LatticeEventStore | None" = LatticeEventStore(db_path)
        else:
            self._store = None

    @property
    def events(self) -> list[RetrievalEvent]:
        return list(self._events)

    def record(self, event: RetrievalEvent) -> None:
        self._events.append(event)
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        if self._store is not None:
            self._store.insert(event)

    def record_recall(self, retrieved_ids: Iterable[str], expected_ids: Iterable[str], k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive")
        retrieved = list(retrieved_ids)[:k]
        expected_set = set(expected_ids)  # never truncate the ground-truth set
        if not expected_set:
            score = 0.0
        else:
            score = len(set(retrieved) & expected_set) / min(k, len(expected_set))
        self._recall_audits.append(float(score))
        return float(score)

    def record_drift(self, baseline_key: Iterable[int], current_key: Iterable[int]) -> float:
        baseline = list(baseline_key)
        current = list(current_key)
        if len(baseline) != len(current):
            raise ValueError("baseline_key and current_key must have the same length")
        if not baseline:
            drift = 0.0
        else:
            drift = sum(1 for a, b in zip(baseline, current) if a != b) / len(baseline)
        self._drift_audits.append(float(drift))
        return float(drift)

    def summary(self) -> dict:
        events = list(self._events)
        latencies = [event.latency_ms for event in events]
        paths = Counter(event.path for event in events)
        total = len(events)
        fallback_count = sum(1 for event in events if event.fallback_used)
        hit_count = sum(1 for event in events if event.hit_count > 0)
        avg_candidates = (
            sum(event.candidate_count for event in events) / total if total else 0.0
        )
        stage_names = sorted({stage for event in events for stage in event.stage_latency_ms})
        stage_latency_ms = {}
        for stage in stage_names:
            values = [event.stage_latency_ms[stage] for event in events if stage in event.stage_latency_ms]
            stage_latency_ms[stage] = {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values) if values else 0.0,
            }
        return {
            "events": total,
            "path_counts": dict(paths),
            "fallback_rate": fallback_count / total if total else 0.0,
            "hit_rate": hit_count / total if total else 0.0,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies) if latencies else 0.0,
            },
            "stage_latency_ms": stage_latency_ms,
            "avg_candidate_count": avg_candidates,
            "recall_audits": {
                "count": len(self._recall_audits),
                "mean": statistics.fmean(self._recall_audits) if self._recall_audits else 0.0,
            },
            "drift_audits": {
                "count": len(self._drift_audits),
                "mean_hamming_fraction": statistics.fmean(self._drift_audits) if self._drift_audits else 0.0,
            },
        }

    def reset(self) -> None:
        self._events.clear()
        self._recall_audits.clear()
        self._drift_audits.clear()
        # Truncate the durable log so a restart does not reload pre-reset events.
        if self.event_log_path is not None and self.event_log_path.exists():
            self.event_log_path.write_text("", encoding="utf-8")
        if self._store is not None:
            self._store.truncate()

    # ------------------------------------------------------------------
    # Store-backed analytics (require db_path to be configured)
    # ------------------------------------------------------------------

    def query_events(self, **kwargs) -> list[dict]:
        """Query durable event store.  Raises if no store is configured."""
        if self._store is None:
            raise RuntimeError("event store not configured")
        return self._store.query(**kwargs)

    def time_series(self, field: str, **kwargs) -> list[dict]:
        """Return time-bucketed metrics from the durable event store."""
        if self._store is None:
            raise RuntimeError("event store not configured")
        return self._store.time_series(field, **kwargs)

    def model_comparison(self, **kwargs) -> list[dict]:
        """Return per-model aggregate statistics from the durable event store."""
        if self._store is None:
            raise RuntimeError("event store not configured")
        return self._store.model_comparison(**kwargs)

    def record_generator_trace(self, trace: "GeneratorTrace") -> None:
        """Record a generator outcome. Persists to the event store if configured.

        The trace joins to retrieval events via trace.request_id.
        Raises RuntimeError if the event store is not configured (no db_path).
        """
        if self._store is None:
            raise RuntimeError(
                "event store not configured; pass db_path to enable generator trace recording"
            )
        self._store.insert_trace(trace)

    def query_traces(self, **kwargs) -> list[dict]:
        """Query generator traces from the durable event store."""
        if self._store is None:
            raise RuntimeError("event store not configured")
        return self._store.query_traces(**kwargs)

    def quality_correlation(self, **kwargs) -> list[dict]:
        """Return per-path quality breakdown joined from events and generator traces."""
        if self._store is None:
            raise RuntimeError("event store not configured")
        return self._store.quality_correlation(**kwargs)

    def export_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def load_jsonl(self, path: str | Path) -> int:
        """Load events from a JSONL file into the in-memory deque.

        Returns the number of events successfully parsed and appended.
        Malformed lines are silently skipped rather than aborting the load.
        If the file contains more events than max_events, the oldest are
        silently dropped by the deque — the return value reflects only
        successful parses, not guaranteed retention.
        """
        input_path = Path(path)
        loaded = 0
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._events.append(RetrievalEvent.from_dict(json.loads(line)))
                    loaded += 1
                except Exception:
                    pass  # skip malformed or unrecognised lines
        return loaded

