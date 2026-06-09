# LatticeMemory LLM Cache Proxy — Docker image
#
# Build:  docker build -t latticememory/proxy .
# Run:    docker run -p 8000:8000 \
#           -e OPENAI_API_KEY=sk-... \
#           latticememory/proxy
#
# Full env-var reference:
#   OPENAI_API_KEY          Required. Forwarded to the upstream LLM API.
#   LATTICE_API_KEY         Alternative to OPENAI_API_KEY.
#   LATTICE_UPSTREAM_URL    Default: https://api.openai.com/v1/chat/completions
#   LATTICE_ENCODER_MODEL   Default: dfrokido/bge-large-e8-snap
#   LATTICE_HAMMING_THRESHOLD  Default: 70
#   LATTICE_HAMMING_MODE    serve | shadow | off  (Default: serve)
#   LATTICE_FP_BUDGET       Float 0.0–1.0 (Default: 0.0 = zero FP)
#   LATTICE_CALIBRATION_DATA_PATH  Path to calibration JSON (optional)
#   LATTICE_SQLITE_PATH     Path for SQLite persistence (optional)
#   LATTICE_COMPLIANCE_MODE true | false (Default: false)
#   LATTICE_VALIDATION_REQUIRED  true | false (Default: false)
#   LATTICE_AUDIT_LOG_PATH  Path for audit log JSONL file (optional)
#   LATTICE_DIVERGENCE_THRESHOLD  Float (optional, enables divergence check)

FROM python:3.11-slim

WORKDIR /app

# System build deps for numpy / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cache friendly)
COPY pyproject.toml .
COPY latticememory/ /app/latticememory/
RUN pip install --no-cache-dir -e ".[proxy]"

# Health check using the built-in /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose proxy port
EXPOSE 8000

# Run the ASGI server.
# Override with: docker run ... --workers 4 (for multi-worker)
ENTRYPOINT ["uvicorn", "latticememory.proxy_server:app", "--host", "0.0.0.0", "--port", "8000"]
CMD []
