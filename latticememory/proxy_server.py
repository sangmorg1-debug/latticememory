"""LatticeMemory LLM Cache Proxy Server — ASGI entrypoint.

Run with: uvicorn latticememory.proxy_server:app --host 0.0.0.0 --port 8000

Or via Docker: docker run -p 8000:8000 -e OPENAI_API_KEY=... latticememory/proxy
"""
import os
from pathlib import Path
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
hamming_rerank_model = os.getenv("LATTICE_HAMMING_RERANK_MODEL", None)
hamming_rerank_retries = int(os.getenv("LATTICE_HAMMING_RERANK_RETRIES", "1"))
hamming_rerank_retry_delay = float(os.getenv("LATTICE_HAMMING_RERANK_RETRY_DELAY", "0.25"))
hamming_cosine_gate = os.getenv("LATTICE_HAMMING_COSINE_GATE", "false").lower() == "true"
hamming_cosine_threshold = float(os.getenv("LATTICE_HAMMING_COSINE_THRESHOLD", "0.9"))
cache_cosine_gate = os.getenv("LATTICE_CACHE_COSINE_GATE", "false").lower() == "true"
cache_cosine_threshold = float(os.getenv("LATTICE_CACHE_COSINE_THRESHOLD", "0.999"))
miss_log_path = os.getenv("LATTICE_MISS_LOG_PATH", None)
warm_path     = os.getenv("LATTICE_WARM_PATH", None)
admin_key     = os.getenv("LATTICE_ADMIN_KEY", None)
reviewer_key  = os.getenv("LATTICE_REVIEWER_KEY", None)
redis_url     = os.getenv("LATTICE_REDIS_URL", None)
redis_namespace = os.getenv("LATTICE_REDIS_NAMESPACE", "lattice")
from latticememory.pq_seed import DEFAULT_PQ_CODEBOOK_SIZE, DEFAULT_PQ_NUM_BLOCKS

pq_proof_dataset = os.getenv("LATTICE_PQ_PROOF_DATASET", None)
pq_mode = os.getenv("LATTICE_PQ_MODE", "false").lower() == "true"
# Left unresolved (None) here on purpose when the user hasn't set them: the
# two branches below need different defaults when unset --
# --pq-proof-dataset's own historical 4/4 (build_seeded_pq_cache_from_support_jsonl's
# defaults in proof_pack.py) vs --pq-mode's validated 8/256
# (DEFAULT_PQ_NUM_BLOCKS/DEFAULT_PQ_CODEBOOK_SIZE) -- resolving to a single
# number here would silently retune whichever branch didn't ask for it.
pq_num_blocks = int(os.environ["LATTICE_PQ_NUM_BLOCKS"]) if os.getenv("LATTICE_PQ_NUM_BLOCKS") else None
pq_codebook_size = int(os.environ["LATTICE_PQ_CODEBOOK_SIZE"]) if os.getenv("LATTICE_PQ_CODEBOOK_SIZE") else None

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

    # Only pass pq_num_blocks/pq_codebook_size through when the user actually
    # set them -- otherwise let build_seeded_pq_cache_from_support_jsonl's
    # own historical 4/4 defaults apply, so --pq-proof-dataset stays
    # reproducible regardless of what --pq-mode's default is.
    proof_pq_kwargs = {}
    if pq_num_blocks is not None:
        proof_pq_kwargs["pq_num_blocks"] = pq_num_blocks
    if pq_codebook_size is not None:
        proof_pq_kwargs["pq_codebook_size"] = pq_codebook_size

    semantic_cache = build_seeded_pq_cache_from_support_jsonl(
        pq_proof_dataset,
        redis_url=redis_url,
        redis_namespace=redis_namespace,
        flush_redis=True,
        **proof_pq_kwargs,
    )
    lattice = semantic_cache.runtime.memory.lattice
    pq_proof = {
        "enabled": True,
        "dataset_path": str(dataset_path),
        "num_blocks": lattice.num_blocks,
        "codebook_size": lattice.codebook_size,
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

    # --pq-mode always resolves to the validated 8/256 default when the user
    # hasn't set LATTICE_PQ_NUM_BLOCKS/LATTICE_PQ_CODEBOOK_SIZE -- unlike
    # --pq-proof-dataset above, this branch is exactly what the corrected
    # default exists for.
    resolved_pq_num_blocks = pq_num_blocks if pq_num_blocks is not None else DEFAULT_PQ_NUM_BLOCKS
    resolved_pq_codebook_size = pq_codebook_size if pq_codebook_size is not None else DEFAULT_PQ_CODEBOOK_SIZE

    semantic_cache = build_pq_cache_from_qa_file(
        warm_path,
        encoder_model=encoder_model,
        pq_num_blocks=resolved_pq_num_blocks,
        pq_codebook_size=resolved_pq_codebook_size,
        sqlite_path=sqlite_path,
        redis_url=redis_url,
        redis_namespace=redis_namespace,
    )
    pq_proof = {
        "enabled": True,
        "num_blocks": resolved_pq_num_blocks,
        "codebook_size": resolved_pq_codebook_size,
        "seeded_entries": semantic_cache.size,
        "mode": "pq_mode",
    }


# Instantiate proxy
proxy = LatticeLLMProxy(
    upstream_url=upstream_url,
    upstream_api_key=upstream_api_key,
    encoder_model=encoder_model,
    d_model=None,  # Auto-detect
    sqlite_path=sqlite_path,
    semantic_cache=semantic_cache,
    hamming_router_mode=hamming_mode,
    hamming_threshold=hamming_threshold,
    compliance_mode=compliance_mode,
    validation_required=validation_required,
    audit_log_path=audit_log_path,
    divergence_threshold=divergence_threshold,
    fp_budget=fp_budget,
    calibration_data_path=calibration_data_path,
    hamming_rerank=hamming_rerank,
    hamming_rerank_model=hamming_rerank_model,
    hamming_rerank_retries=hamming_rerank_retries,
    hamming_rerank_retry_delay=hamming_rerank_retry_delay,
    hamming_cosine_gate=hamming_cosine_gate,
    hamming_cosine_threshold=hamming_cosine_threshold,
    cache_cosine_gate=cache_cosine_gate,
    cache_cosine_threshold=cache_cosine_threshold,
    miss_log_path=miss_log_path,
    warm_path=(None if (pq_mode and semantic_cache is not None) else warm_path),
    admin_key=admin_key,
    reviewer_key=reviewer_key,
    redis_url=redis_url,
    redis_namespace=redis_namespace,
)
proxy.pq_proof = pq_proof

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

