from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .memory import MemoryDocument, MemoryQuery, RFSnapLatticeMemory
from .observability import GeneratorTrace, LatticeObservability
from .semantic_cache import RFSnapSemanticCache
from .text_runtime import RFSnapTextMemory


def _latticememory_version() -> str:
    try:
        return package_version("latticememory")
    except PackageNotFoundError:
        return "0.0.0"


class DocumentInput(BaseModel):
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentsRequest(BaseModel):
    documents: list[DocumentInput]


class RetrieveRequest(BaseModel):
    text: str
    embedding: list[float]
    top_k: int = 5
    request_id: str | None = None
    product: str | None = None
    dataset: str | None = None
    model_id: str | None = None
    index_id: str | None = None
    quality_tags: list[str] = Field(default_factory=list)


class TextsRequest(BaseModel):
    texts: list[str]
    doc_ids: list[str] | None = None
    metadatas: list[dict[str, Any]] | None = None
    dataset: str | None = None
    index_id: str | None = None


class RetrieveTextRequest(BaseModel):
    text: str
    top_k: int = 5
    request_id: str | None = None
    product: str | None = None
    dataset: str | None = None
    model_id: str | None = None
    index_id: str | None = None
    quality_tags: list[str] = Field(default_factory=list)


class CachePutRequest(BaseModel):
    prompt: str
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheGetRequest(BaseModel):
    prompt: str
    request_id: str | None = None
    product: str | None = None
    dataset: str | None = None
    index_id: str | None = None


class RecallAuditRequest(BaseModel):
    retrieved_ids: list[str]
    expected_ids: list[str]
    k: int


class DriftAuditRequest(BaseModel):
    baseline_key: list[int]
    current_key: list[int]


class GeneratorTraceInput(BaseModel):
    request_id: str
    prompt: str | None = None
    completion: str | None = None
    latency_ms: float | None = None
    model_id: str | None = None
    quality_tags: list[str] = Field(default_factory=list)
    quality_score: float | None = None


def create_app(
    *,
    d_model: int = 1024,
    memory: RFSnapLatticeMemory | None = None,
    text_runtime: RFSnapTextMemory | None = None,
    semantic_cache: RFSnapSemanticCache | None = None,
    event_log_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """Create an isolated LatticeMemory API app.

    The lower-level RAG service remains in `liora_core.rag.e8_retrieval_server`.
    This service is the product-facing control plane for retrieval plus
    observability.

    ``event_log_path`` and ``db_path`` are only valid when ``memory`` is None
    (the app creates its own memory).  Passing either alongside a pre-built
    memory raises ValueError — the supplied memory already has its own
    observability instance and silently ignoring these paths would be a
    misconfiguration.
    """
    if memory is not None and text_runtime is not None and memory is not text_runtime.memory:
        raise ValueError("memory and text_runtime.memory must be the same object when both are supplied")
    if semantic_cache is not None:
        if text_runtime is not None and semantic_cache.runtime is not text_runtime:
            raise ValueError("semantic_cache.runtime must match text_runtime when both are supplied")
        text_runtime = semantic_cache.runtime
    if memory is not None and (event_log_path is not None or db_path is not None):
        raise ValueError(
            "event_log_path and db_path are ignored when memory is supplied; "
            "configure them on the LatticeObservability before "
            "constructing memory, then pass that memory here."
        )

    if text_runtime is not None:
        memory = text_runtime.memory
    if memory is None:
        observability = LatticeObservability(event_log_path=event_log_path, db_path=db_path)
        memory = RFSnapLatticeMemory(d_model=d_model, observability=observability)

    app = FastAPI(title="RF-Snap LatticeMemory Observability API", version=_latticememory_version())
    # All route handlers access memory via request.app.state.memory so that
    # callers can swap the memory at runtime by assigning app.state.memory.
    app.state.memory = memory
    app.state.text_runtime = text_runtime
    app.state.semantic_cache = semantic_cache

    @app.get("/health")
    async def health(request: Request) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        return {
            "status": "healthy",
            "service": "latticememory",
            "documents": mem.num_documents,
            "d_model": mem.d_model,
            "observability_events": mem.observability_summary()["events"],
        }

    @app.post("/documents")
    async def add_documents(request: Request, body: DocumentsRequest) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        if not body.documents:
            raise HTTPException(status_code=400, detail="documents must not be empty")
        try:
            mem.add_documents(
                [
                    MemoryDocument(
                        doc_id=doc.doc_id,
                        text=doc.text,
                        embedding=doc.embedding,
                        metadata=doc.metadata,
                    )
                    for doc in body.documents
                ]
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "success",
            "indexed": len(body.documents),
            "total_documents": mem.num_documents,
            "d_model": mem.d_model,
            "blocks": mem.lattice.num_blocks,
        }

    @app.post("/retrieve")
    async def retrieve(request: Request, body: RetrieveRequest) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        try:
            result = mem.retrieve(
                MemoryQuery(
                    text=body.text,
                    embedding=body.embedding,
                    top_k=body.top_k,
                    request_id=body.request_id,
                    product=body.product,
                    dataset=body.dataset,
                    model_id=body.model_id,
                    index_id=body.index_id,
                    quality_tags=body.quality_tags,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # retrieve() always records an event before returning; grab it directly
        # from the deque rather than copying the whole list.
        event = mem.observability._events[-1].to_dict()
        return {
            "status": "success",
            "matches": [
                {
                    "doc_id": hit.doc_id,
                    "text": hit.text,
                    "score": hit.score,
                    "metadata": hit.metadata,
                }
                for hit in result.hits
            ],
            "path": result.path,
            "fallback_used": result.fallback_used,
            "candidate_count": result.candidate_count,
            "latency_ms": result.latency_ms,
            "context_packet": result.to_context_packet(),
            "event": event,
        }

    @app.post("/texts")
    async def add_texts(request: Request, body: TextsRequest) -> dict:
        runtime: RFSnapTextMemory | None = request.app.state.text_runtime
        if runtime is None:
            raise HTTPException(status_code=400, detail="text runtime is not configured")
        if not body.texts:
            raise HTTPException(status_code=400, detail="texts must not be empty")
        try:
            result = runtime.add_texts(
                body.texts,
                doc_ids=body.doc_ids,
                metadatas=body.metadatas,
                dataset=body.dataset,
                index_id=body.index_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "success",
            "indexed": result.indexed,
            "total_documents": result.total_documents,
            "doc_ids": result.doc_ids,
        }

    @app.post("/retrieve-text")
    async def retrieve_text(request: Request, body: RetrieveTextRequest) -> dict:
        runtime: RFSnapTextMemory | None = request.app.state.text_runtime
        if runtime is None:
            raise HTTPException(status_code=400, detail="text runtime is not configured")
        try:
            result = runtime.retrieve_text(
                body.text,
                top_k=body.top_k,
                request_id=body.request_id,
                product=body.product,
                dataset=body.dataset,
                model_id=body.model_id,
                index_id=body.index_id,
                quality_tags=body.quality_tags,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        mem: RFSnapLatticeMemory = runtime.memory
        event = mem.observability._events[-1].to_dict()
        return {
            "status": "success",
            "matches": [
                {
                    "doc_id": hit.doc_id,
                    "text": hit.text,
                    "score": hit.score,
                    "metadata": hit.metadata,
                }
                for hit in result.hits
            ],
            "path": result.path,
            "fallback_used": result.fallback_used,
            "candidate_count": result.candidate_count,
            "latency_ms": result.latency_ms,
            "context_packet": result.to_context_packet(),
            "event": event,
        }

    @app.post("/cache/put")
    async def cache_put(request: Request, body: CachePutRequest) -> dict:
        cache: RFSnapSemanticCache | None = request.app.state.semantic_cache
        if cache is None:
            raise HTTPException(status_code=400, detail="semantic cache is not configured")
        entry = cache.put(body.prompt, value=body.value, metadata=body.metadata)
        return {
            "status": "success",
            "cache_id": entry.cache_id,
            "prompt": entry.prompt,
            "lattice_key": list(entry.lattice_key),
            "metadata": entry.metadata,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }

    @app.post("/cache/get")
    async def cache_get(request: Request, body: CacheGetRequest) -> dict:
        cache: RFSnapSemanticCache | None = request.app.state.semantic_cache
        if cache is None:
            raise HTTPException(status_code=400, detail="semantic cache is not configured")
        result = cache.get(
            body.prompt,
            request_id=body.request_id,
            product=body.product,
            dataset=body.dataset,
            index_id=body.index_id,
        )
        return {
            "hit": result.hit,
            "hit_type": result.hit_type,
            "value": result.value,
            "cache_id": result.cache_id,
            "lattice_key": list(result.lattice_key),
            "source_prompt": result.source_prompt,
            "metadata": result.metadata,
            "retrieval_path": result.retrieval_path,
            "latency_ms": result.latency_ms,
        }

    @app.get("/cache/stats")
    async def cache_stats(request: Request) -> dict:
        cache: RFSnapSemanticCache | None = request.app.state.semantic_cache
        if cache is None:
            raise HTTPException(status_code=400, detail="semantic cache is not configured")
        return cache.stats()

    @app.get("/observability/summary")
    async def observability_summary(request: Request) -> dict:
        return request.app.state.memory.observability_summary()

    @app.get("/observability/events")
    async def observability_events(
        request: Request,
        limit: int = Query(default=100, ge=1, le=10_000),
        path: str | None = None,
        product: str | None = None,
        dataset: str | None = None,
        model_id: str | None = None,
        index_id: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        events = mem.observability.events
        if path is not None:
            events = [e for e in events if e.path == path]
        if product is not None:
            events = [e for e in events if e.product == product]
        if dataset is not None:
            events = [e for e in events if e.dataset == dataset]
        if model_id is not None:
            events = [e for e in events if e.model_id == model_id]
        if index_id is not None:
            events = [e for e in events if e.index_id == index_id]
        if request_id is not None:
            events = [e for e in events if e.request_id == request_id]
        events = events[-limit:]
        return {"events": [e.to_dict() for e in events]}

    @app.post("/observability/recall-audit")
    async def recall_audit(request: Request, body: RecallAuditRequest) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        try:
            score = mem.observability.record_recall(body.retrieved_ids, body.expected_ids, body.k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"recall": score, "summary": mem.observability_summary()["recall_audits"]}

    @app.post("/observability/drift-audit")
    async def drift_audit(request: Request, body: DriftAuditRequest) -> dict:
        mem: RFSnapLatticeMemory = request.app.state.memory
        try:
            score = mem.observability.record_drift(body.baseline_key, body.current_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"hamming_fraction": score, "summary": mem.observability_summary()["drift_audits"]}

    @app.post("/observability/reset")
    async def reset_observability(request: Request) -> dict:
        request.app.state.memory.observability.reset()
        return {"status": "success", "events": 0}

    @app.get("/observability/time-series")
    async def observability_time_series(
        request: Request,
        window_seconds: float = Query(default=3600.0, gt=0),
        n_buckets: int = Query(default=60, ge=1),
        product: str | None = None,
        model_id: str | None = None,
    ):
        obs = request.app.state.memory.observability
        if obs._store is None:
            return JSONResponse(status_code=400, content={"error": "event store not configured"})
        buckets = obs.time_series(
            "fallback_rate",
            window_seconds=window_seconds,
            n_buckets=n_buckets,
            product=product,
            model_id=model_id,
        )
        return {"window_seconds": window_seconds, "n_buckets": n_buckets, "buckets": buckets}

    @app.get("/observability/model-comparison")
    async def observability_model_comparison(
        request: Request,
        model_ids: str | None = Query(default=None, description="Comma-separated model IDs to filter"),
    ):
        obs = request.app.state.memory.observability
        if obs._store is None:
            return JSONResponse(status_code=400, content={"error": "event store not configured"})
        ids = [m.strip() for m in model_ids.split(",") if m.strip()] if model_ids else None
        rows = obs.model_comparison(model_ids=ids) if ids is not None else obs.model_comparison()
        return {"models": rows}

    @app.post("/traces")
    async def record_trace(request: Request, body: GeneratorTraceInput) -> dict:
        obs = request.app.state.memory.observability
        if obs._store is None:
            return JSONResponse(status_code=400, content={"error": "event store not configured"})
        trace = GeneratorTrace(
            request_id=body.request_id,
            prompt=body.prompt,
            completion=body.completion,
            latency_ms=body.latency_ms,
            model_id=body.model_id,
            quality_tags=body.quality_tags,
            quality_score=body.quality_score,
        )
        obs.record_generator_trace(trace)
        return {"status": "success", "trace_id": trace.trace_id, "timestamp": trace.timestamp}

    @app.get("/traces/{request_id}")
    async def get_traces_for_request(request: Request, request_id: str) -> dict:
        obs = request.app.state.memory.observability
        if obs._store is None:
            return JSONResponse(status_code=400, content={"error": "event store not configured"})
        retrieval_events = obs.query_events(request_id=request_id, limit=200)
        generator_traces = obs.query_traces(request_id=request_id, limit=200)
        return {
            "request_id": request_id,
            "retrieval_events": retrieval_events,
            "generator_traces": generator_traces,
        }

    @app.get("/observability/quality-correlation")
    async def observability_quality_correlation(
        request: Request,
        model_id: str | None = None,
        since_ts: float | None = None,
    ):
        obs = request.app.state.memory.observability
        if obs._store is None:
            return JSONResponse(status_code=400, content={"error": "event store not configured"})
        rows = obs.quality_correlation(model_id=model_id, since_ts=since_ts)
        return {"paths": rows}

    @app.get("/stats")
    async def stats(request: Request) -> dict:
        return request.app.state.memory.stats()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lattice Observability</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f7f8fa; color: #1b1f24; }
    header { padding: 20px 32px; background: #111827; color: white; display: flex; align-items: center; justify-content: space-between; }
    header h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
    header .subtitle { color: #9ca3af; font-size: 13px; margin-top: 4px; }
    .refresh-badge { font-size: 12px; color: #6ee7b7; border: 1px solid #065f46; background: #022c22; border-radius: 20px; padding: 4px 12px; }
    main { padding: 24px 32px; display: grid; gap: 20px; max-width: 1600px; }
    /* Stat cards */
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; }
    .card .label { color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .card .value { font-size: 26px; font-weight: 700; margin-top: 6px; color: #0f172a; line-height: 1.1; }
    .card .value.good { color: #16a34a; }
    .card .value.warn { color: #d97706; }
    .card .value.bad  { color: #dc2626; }
    /* Chart grid */
    .chart-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
    .chart-grid-full { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .chart-wrap { position: relative; height: 260px; }
    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { padding: 8px 10px; border-bottom: 2px solid #e2e8f0; text-align: left; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
    td { padding: 8px 10px; border-bottom: 1px solid #f1f5f9; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f8fafc; }
    code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; font-size: 12px; }
    /* Path color badges */
    .path-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .path-lattice_exact  { background: #dcfce7; color: #166534; }
    .path-lattice_hamming,
    .path-lattice_hamming1 { background: #dbeafe; color: #1e40af; }
    .path-fallback       { background: #ffedd5; color: #9a3412; }
    .path-miss           { background: #fee2e2; color: #991b1b; }
    /* Quality tag badges */
    .tag { display: inline-block; margin: 1px 2px; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; background: #e0e7ff; color: #3730a3; }
    /* Filter bar */
    .filter-bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .filter-bar input, .filter-bar select { padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 13px; background: white; min-width: 130px; }
    .filter-bar button { padding: 6px 14px; border-radius: 6px; border: none; cursor: pointer; font-size: 13px; font-weight: 600; }
    .btn-apply { background: #3b82f6; color: white; }
    .btn-apply:hover { background: #2563eb; }
    .btn-clear  { background: #f1f5f9; color: #374151; }
    .btn-clear:hover  { background: #e2e8f0; }
    /* Placeholder */
    .placeholder { color: #94a3b8; font-style: italic; text-align: center; padding: 32px 0; font-size: 14px; }
    h2 { margin: 0 0 14px 0; font-size: 15px; font-weight: 700; color: #1e293b; }
    section.card h2 { margin-bottom: 14px; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Lattice Observability</h1>
      <div class="subtitle">RF-Snap LatticeMemory &mdash; live retrieval diagnostics</div>
    </div>
    <div class="refresh-badge" id="refresh-badge">Refreshing&hellip;</div>
  </header>

  <main>
    <!-- Stat cards -->
    <section class="stat-grid" id="stat-grid">
      <div class="card"><div class="label">Total Events</div><div class="value" id="stat-events">&mdash;</div></div>
      <div class="card"><div class="label">Fallback Rate</div><div class="value" id="stat-fallback">&mdash;</div></div>
      <div class="card"><div class="label">Hit Rate</div><div class="value" id="stat-hit">&mdash;</div></div>
      <div class="card"><div class="label">Latency p95</div><div class="value" id="stat-p95">&mdash;</div></div>
      <div class="card"><div class="label">Lattice Exact Rate</div><div class="value" id="stat-exact">&mdash;</div></div>
      <div class="card"><div class="label">Avg Candidates</div><div class="value" id="stat-candidates">&mdash;</div></div>
    </section>

    <!-- Path mix + Latency trend -->
    <section class="chart-grid">
      <div class="card">
        <h2>Path mix</h2>
        <div class="chart-wrap"><canvas id="pathMixChart"></canvas></div>
      </div>
      <div class="card">
        <h2>Latency trend (last hour)</h2>
        <div id="latency-chart-area">
          <div class="chart-wrap"><canvas id="latencyChart"></canvas></div>
        </div>
      </div>
    </section>

    <!-- Fallback trend -->
    <section class="chart-grid-full">
      <div class="card">
        <h2>Fallback rate trend (last hour)</h2>
        <div id="fallback-chart-area">
          <div class="chart-wrap"><canvas id="fallbackChart"></canvas></div>
        </div>
      </div>
      <div class="card" id="model-comparison-card">
        <h2>Per-model comparison</h2>
        <div id="model-table-area"><div class="placeholder">Loading&hellip;</div></div>
      </div>
    </section>

    <!-- Quality correlation -->
    <section class="card" id="quality-correlation-card">
      <h2>Quality correlation</h2>
      <div id="quality-correlation-area"><div class="placeholder">Loading&hellip;</div></div>
    </section>

    <!-- Recent events with filters -->
    <section class="card">
      <h2>Recent events</h2>
      <div class="filter-bar" style="margin-bottom:14px;">
        <input type="text" id="filter-product" placeholder="Product&hellip;">
        <input type="text" id="filter-model" placeholder="Model ID&hellip;">
        <select id="filter-path">
          <option value="">All paths</option>
          <option value="lattice_exact">lattice_exact</option>
          <option value="lattice_hamming1">lattice_hamming1</option>
          <option value="fallback">fallback</option>
          <option value="miss">miss</option>
        </select>
        <button class="btn-apply" onclick="applyFilters()">Apply</button>
        <button class="btn-clear" onclick="clearFilters()">Clear</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Product</th>
            <th>Path</th>
            <th>Top doc</th>
            <th>Latency (ms)</th>
            <th>Quality tags</th>
          </tr>
        </thead>
        <tbody id="recent-body"></tbody>
      </table>
    </section>
  </main>

  <script>
    // ---- Chart instances (created once, updated on each refresh) ----
    let pathMixChart = null;
    let latencyChart = null;
    let fallbackChart = null;

    // ---- Helpers ----
    function escapeHtml(s) {
      if (s == null) return '—';
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function fmt(v, decimals, suffix) {
      if (v == null || isNaN(v)) return '—';
      return Number(v).toFixed(decimals) + (suffix || '');
    }

    function fmtPct(rate) {
      if (rate == null || isNaN(rate)) return '—';
      return fmt(rate * 100, 1, '%');
    }

    function fmtTs(epochSeconds) {
      const d = new Date(epochSeconds * 1000);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function pathBadge(path) {
      const cls = 'path-' + (path || 'miss');
      return '<span class="path-badge ' + cls + '">' + escapeHtml(path || 'miss') + '</span>';
    }

    function tagBadges(tags) {
      if (!tags || !tags.length) return '';
      return tags.map(t => '<span class="tag">' + escapeHtml(t) + '</span>').join('');
    }

    function safeVal(x) { return (x == null) ? '—' : x; }

    // ---- Summary stat cards ----
    function updateStats(s) {
      document.getElementById('stat-events').textContent = safeVal(s.events);
      const fbPct = s.fallback_rate * 100;
      const fbEl = document.getElementById('stat-fallback');
      fbEl.textContent = fmt(s.fallback_rate * 100, 1, '%');
      fbEl.className = 'value' + (fbPct > 30 ? ' bad' : fbPct > 10 ? ' warn' : ' good');
      const hitPct = s.hit_rate * 100;
      const hitEl = document.getElementById('stat-hit');
      hitEl.textContent = fmt(s.hit_rate * 100, 1, '%');
      hitEl.className = 'value' + (hitPct > 70 ? ' good' : hitPct > 40 ? ' warn' : ' bad');
      document.getElementById('stat-p95').textContent = fmt((s.latency_ms || {}).p95, 2, ' ms');
      const total = s.events || 0;
      const exact = ((s.path_counts || {}).lattice_exact) || 0;
      const exactPct = total ? exact / total * 100 : 0;
      const exactEl = document.getElementById('stat-exact');
      exactEl.textContent = fmt(exactPct, 1, '%');
      exactEl.className = 'value' + (exactPct > 60 ? ' good' : exactPct > 30 ? ' warn' : ' bad');
      document.getElementById('stat-candidates').textContent = fmt(s.avg_candidate_count, 1);
    }

    // ---- Path mix doughnut ----
    function updatePathMix(pathCounts) {
      const labels = ['lattice_exact', 'lattice_hamming1', 'fallback', 'miss'];
      const colors = ['#16a34a', '#2563eb', '#d97706', '#dc2626'];
      const data = labels.map(l => (pathCounts || {})[l] || 0);
      const total = data.reduce((a, b) => a + b, 0);
      const legendLabels = labels.map((l, i) => {
        const pct = total ? (data[i] / total * 100).toFixed(1) : '0.0';
        return l + ' (' + pct + '%)';
      });

      if (!pathMixChart) {
        const ctx = document.getElementById('pathMixChart').getContext('2d');
        pathMixChart = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: legendLabels,
            datasets: [{ data: data, backgroundColor: colors, borderWidth: 2, borderColor: '#fff' }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'bottom', labels: { font: { size: 11 }, boxWidth: 12, padding: 10 } }
            }
          }
        });
      } else {
        pathMixChart.data.labels = legendLabels;
        pathMixChart.data.datasets[0].data = data;
        pathMixChart.update();
      }
    }

    // ---- Latency trend (p50 from time-series; p95 from summary as flat reference) ----
    function renderLatencyChartNoStore() {
      document.getElementById('latency-chart-area').innerHTML =
        '<div class="placeholder">Event store not configured &mdash; time-series unavailable</div>';
      if (latencyChart) { latencyChart.destroy(); latencyChart = null; }
    }

    function updateLatencyChart(buckets, p95Ref) {
      const labels = buckets.map(b => fmtTs(b.timestamp_start));
      const p50Data = buckets.map(b => b.p50_latency);
      const p95Data = buckets.map(() => p95Ref);

      if (!latencyChart) {
        const area = document.getElementById('latency-chart-area');
        area.innerHTML = '<div class="chart-wrap"><canvas id="latencyChart"></canvas></div>';
        const ctx = document.getElementById('latencyChart').getContext('2d');
        latencyChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: labels,
            datasets: [
              {
                label: 'p50 latency (ms)',
                data: p50Data,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.08)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.3
              },
              {
                label: 'p95 (overall, ms)',
                data: p95Data,
                borderColor: '#f59e0b',
                borderWidth: 1.5,
                borderDash: [4, 4],
                pointRadius: 0,
                fill: false,
                tension: 0
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { labels: { font: { size: 11 } } } },
            scales: {
              x: { ticks: { maxRotation: 0, font: { size: 10 }, maxTicksLimit: 12 } },
              y: { title: { display: true, text: 'ms', font: { size: 11 } }, beginAtZero: true }
            }
          }
        });
      } else {
        latencyChart.data.labels = labels;
        latencyChart.data.datasets[0].data = p50Data;
        latencyChart.data.datasets[1].data = p95Data;
        latencyChart.update();
      }
    }

    // ---- Fallback rate trend ----
    function renderFallbackChartNoStore() {
      document.getElementById('fallback-chart-area').innerHTML =
        '<div class="placeholder">Event store not configured &mdash; time-series unavailable</div>';
      if (fallbackChart) { fallbackChart.destroy(); fallbackChart = null; }
    }

    function updateFallbackChart(buckets) {
      const labels = buckets.map(b => fmtTs(b.timestamp_start));
      const rateData = buckets.map(b => (b.fallback_rate || 0) * 100);

      if (!fallbackChart) {
        const area = document.getElementById('fallback-chart-area');
        area.innerHTML = '<div class="chart-wrap"><canvas id="fallbackChart"></canvas></div>';
        const ctx = document.getElementById('fallbackChart').getContext('2d');
        fallbackChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: labels,
            datasets: [{
              label: 'Fallback rate (%)',
              data: rateData,
              borderColor: '#f97316',
              backgroundColor: 'rgba(249,115,22,0.08)',
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0.3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { font: { size: 11 } } } },
            scales: {
              x: { ticks: { maxRotation: 0, font: { size: 10 }, maxTicksLimit: 12 } },
              y: {
                title: { display: true, text: '%', font: { size: 11 } },
                min: 0, max: 100
              }
            }
          }
        });
      } else {
        fallbackChart.data.labels = labels;
        fallbackChart.data.datasets[0].data = rateData;
        fallbackChart.update();
      }
    }

    // ---- Model comparison table ----
    function renderModelTable(models) {
      if (!models || !models.length) {
        document.getElementById('model-table-area').innerHTML =
          '<div class="placeholder">No model data available</div>';
        return;
      }
      const rows = models.map(m =>
        '<tr>' +
        '<td><code>' + escapeHtml(m.model_id) + '</code></td>' +
        '<td>' + safeVal(m.event_count) + '</td>' +
        '<td>' + fmtPct(m.fallback_rate) + '</td>' +
        '<td>' + fmtPct(m.hit_rate) + '</td>' +
        '<td>' + fmt(m.avg_latency, 2, ' ms') + '</td>' +
        '<td>' + fmt(m.p50_latency, 2, ' ms') + '</td>' +
        '</tr>'
      ).join('');
      document.getElementById('model-table-area').innerHTML =
        '<table><thead><tr>' +
        '<th>Model</th><th>Events</th><th>Fallback rate</th><th>Hit rate</th><th>Avg latency</th><th>p50 latency</th>' +
        '</tr></thead><tbody>' + rows + '</tbody></table>';
    }

    function renderModelTableNoStore() {
      document.getElementById('model-table-area').innerHTML =
        '<div class="placeholder">Event store not configured</div>';
    }

    // ---- Recent events table ----
    function renderRecentEvents(evts) {
      if (!evts || !evts.length) {
        document.getElementById('recent-body').innerHTML =
          '<tr><td colspan="6" class="placeholder">No events recorded yet</td></tr>';
        return;
      }
      document.getElementById('recent-body').innerHTML = evts.map(e => {
        const t = e.timestamp ? fmtTs(e.timestamp) : '—';
        return '<tr>' +
          '<td>' + t + '</td>' +
          '<td>' + escapeHtml(e.product) + '</td>' +
          '<td>' + pathBadge(e.path) + '</td>' +
          '<td><code>' + escapeHtml(e.top_doc_id) + '</code></td>' +
          '<td>' + (e.latency_ms != null ? fmt(e.latency_ms, 2) : '—') + '</td>' +
          '<td>' + tagBadges(e.quality_tags) + '</td>' +
          '</tr>';
      }).join('');
    }

    // ---- Quality correlation table ----
    function renderQualityCorrelation(rows) {
      if (!rows || !rows.length) {
        document.getElementById('quality-correlation-area').innerHTML =
          '<div class="placeholder">No joined traces yet &mdash; record generator traces to populate</div>';
        return;
      }
      const tableRows = rows.map(r =>
        '<tr>' +
        '<td>' + pathBadge(r.path) + '</td>' +
        '<td>' + safeVal(r.event_count) + '</td>' +
        '<td>' + safeVal(r.trace_count) + '</td>' +
        '<td>' + (r.trace_rate != null ? fmt(r.trace_rate * 100, 1, '%') : '&mdash;') + '</td>' +
        '<td>' + (r.avg_quality_score != null ? fmt(r.avg_quality_score, 3) : '&mdash;') + '</td>' +
        '<td>' + fmtPct(r.fallback_rate) + '</td>' +
        '</tr>'
      ).join('');
      document.getElementById('quality-correlation-area').innerHTML =
        '<table><thead><tr>' +
        '<th>Path</th><th>Events</th><th>Traces</th><th>Trace rate</th><th>Avg score</th><th>Fallback rate</th>' +
        '</tr></thead><tbody>' + tableRows + '</tbody></table>';
    }

    function renderQualityCorrelationNoStore() {
      document.getElementById('quality-correlation-area').innerHTML =
        '<div class="placeholder">Event store not configured</div>';
    }

    // ---- Filter state ----
    let filterProduct = '';
    let filterModel = '';
    let filterPath = '';

    function applyFilters() {
      filterProduct = document.getElementById('filter-product').value.trim();
      filterModel   = document.getElementById('filter-model').value.trim();
      filterPath    = document.getElementById('filter-path').value;
      fetchRecentEvents();
    }

    function clearFilters() {
      filterProduct = ''; filterModel = ''; filterPath = '';
      document.getElementById('filter-product').value = '';
      document.getElementById('filter-model').value = '';
      document.getElementById('filter-path').value = '';
      fetchRecentEvents();
    }

    async function fetchRecentEvents() {
      try {
        let url = '/observability/events?limit=25';
        if (filterProduct) url += '&product=' + encodeURIComponent(filterProduct);
        if (filterModel)   url += '&model_id=' + encodeURIComponent(filterModel);
        if (filterPath)    url += '&path=' + encodeURIComponent(filterPath);
        const resp = await fetch(url);
        if (!resp.ok) { renderRecentEvents([]); return; }
        const data = await resp.json();
        renderRecentEvents(data.events || []);
      } catch (err) {
        renderRecentEvents([]);
      }
    }

    // ---- Main refresh loop ----
    async function refresh() {
      try {
        // Summary stats
        const summaryResp = await fetch('/observability/summary');
        if (summaryResp.ok) {
          const s = await summaryResp.json();
          updateStats(s);
          updatePathMix(s.path_counts || {});

          // Time-series (requires event store)
          try {
            const tsResp = await fetch('/observability/time-series?window_seconds=3600&n_buckets=60');
            if (tsResp.ok) {
              const tsData = await tsResp.json();
              const buckets = tsData.buckets || [];
              const p95Ref = (s.latency_ms || {}).p95 || 0;
              updateLatencyChart(buckets, p95Ref);
              updateFallbackChart(buckets);
            } else {
              renderLatencyChartNoStore();
              renderFallbackChartNoStore();
            }
          } catch (e) {
            renderLatencyChartNoStore();
            renderFallbackChartNoStore();
          }

          // Model comparison (requires event store)
          try {
            const mcResp = await fetch('/observability/model-comparison');
            if (mcResp.ok) {
              const mcData = await mcResp.json();
              renderModelTable(mcData.models || []);
            } else {
              renderModelTableNoStore();
            }
          } catch (e) {
            renderModelTableNoStore();
          }

          // Quality correlation (requires event store)
          try {
            const qcResp = await fetch('/observability/quality-correlation');
            if (qcResp.ok) {
              const qcData = await qcResp.json();
              renderQualityCorrelation(qcData.paths || []);
            } else {
              renderQualityCorrelationNoStore();
            }
          } catch (e) {
            renderQualityCorrelationNoStore();
          }

          // Update badge only on successful summary refresh
          document.getElementById('refresh-badge').textContent =
            'Updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
      } catch (err) {
        // Network error: leave existing values and badge unchanged
      }

      // Recent events (separate so filters work independently)
      await fetchRecentEvents();
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8081)

