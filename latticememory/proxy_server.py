"""LatticeMemory LLM Cache Proxy Server — ASGI entrypoint.

Run with: uvicorn latticememory.proxy_server:app --host 0.0.0.0 --port 8000

Or via Docker: docker run -p 8000:8000 -e OPENAI_API_KEY=... latticememory/proxy
"""
import os
from latticememory.proxy import LatticeLLMProxy

# Configuration from environment
upstream_url = os.getenv("LATTICE_UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")
upstream_api_key = os.getenv("OPENAI_API_KEY", os.getenv("LATTICE_API_KEY"))
encoder_model = os.getenv("LATTICE_ENCODER_MODEL", "dfrokido/bge-large-e8-snap")
hamming_threshold = int(os.getenv("LATTICE_HAMMING_THRESHOLD", "70"))
hamming_mode = os.getenv("LATTICE_HAMMING_MODE", "serve")
sqlite_path = os.getenv("LATTICE_SQLITE_PATH", None)
compliance_mode = os.getenv("LATTICE_COMPLIANCE_MODE", "false").lower() == "true"
validation_required = os.getenv("LATTICE_VALIDATION_REQUIRED", "false").lower() == "true"
audit_log_path = os.getenv("LATTICE_AUDIT_LOG_PATH", None)
divergence_threshold = float(os.getenv("LATTICE_DIVERGENCE_THRESHOLD", "0.1")) if os.getenv("LATTICE_DIVERGENCE_THRESHOLD") else None
fp_budget = float(os.getenv("LATTICE_FP_BUDGET", "0.0"))
calibration_data_path = os.getenv("LATTICE_CALIBRATION_DATA_PATH", None)
hamming_rerank = os.getenv("LATTICE_HAMMING_RERANK", "false").lower() == "true"
miss_log_path = os.getenv("LATTICE_MISS_LOG_PATH", None)
warm_path     = os.getenv("LATTICE_WARM_PATH", None)
admin_key     = os.getenv("LATTICE_ADMIN_KEY", None)
reviewer_key  = os.getenv("LATTICE_REVIEWER_KEY", None)

import warnings

if not upstream_api_key:
    warnings.warn(
        "OPENAI_API_KEY or LATTICE_API_KEY is not set. "
        "The proxy will start but upstream inference calls will fail with 401. "
        "Set the key before routing live traffic.",
        RuntimeWarning,
        stacklevel=1,
    )
    upstream_api_key = ""  # proxy handles missing key gracefully per-request

# Instantiate proxy
proxy = LatticeLLMProxy(
    upstream_url=upstream_url,
    upstream_api_key=upstream_api_key,
    encoder_model=encoder_model,
    d_model=None,  # Auto-detect
    sqlite_path=sqlite_path,
    hamming_router_mode=hamming_mode,
    hamming_threshold=hamming_threshold,
    compliance_mode=compliance_mode,
    validation_required=validation_required,
    audit_log_path=audit_log_path,
    divergence_threshold=divergence_threshold,
    fp_budget=fp_budget,
    calibration_data_path=calibration_data_path,
    hamming_rerank=hamming_rerank,
    miss_log_path=miss_log_path,
    warm_path=warm_path,
    admin_key=admin_key,
    reviewer_key=reviewer_key,
)

# Create FastAPI app
app = proxy.create_app()


def main() -> None:
    """Entry point for the ``latticememory-serve-proxy`` console script."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required to serve the proxy. "
            "Install with: pip install 'lattice-memory-e8[proxy]'"
        )
    port = int(os.getenv("LATTICE_PORT", "8000"))
    host = os.getenv("LATTICE_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

