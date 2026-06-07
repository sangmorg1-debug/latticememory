import os
import time
import json
import warnings
import hashlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from latticememory.memory import RFSnapLatticeMemory
from latticememory.semantic_cache import RFSnapSemanticCache
from latticememory.text_runtime import RFSnapTextMemory
from latticememory.hamming_router import validate_calibration_data_schema

logger = logging.getLogger("latticememory.proxy")


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LatticeLLMProxy requires FastAPI. Install with: pip install 'latticememory[proxy]'"
        ) from exc


UpstreamClient = Callable[[dict[str, Any], dict[str, str]], Any]


class LatticeLLMProxy:
    """OpenAI-compatible chat-completions proxy with E8 semantic caching and compliance controls."""

    def __init__(
        self,
        *,
        upstream_url: str,
        upstream_api_key: str | None = None,
        encoder_model: str = "dfrokido/bge-large-e8-snap",
        encoder: Any | None = None,
        d_model: int | None = None,
        sqlite_path: str | None = None,
        semantic_cache: RFSnapSemanticCache | None = None,
        upstream_client: UpstreamClient | None = None,
        cost_per_1k_input_tokens_usd: float = 0.005,
        batch_size: int = 64,
        # Hamming-NN approximate cache (off by default — calibrate threshold before enabling)
        enable_hamming_router: bool | None = None,
        hamming_router_mode: str | None = None,
        hamming_threshold: int = 70,
        calibration_data_path: str | None = None,
        fp_budget: float = 0.0,
        require_calibration: bool = False,
        # Compliance & Canonicalization Extensions
        compliance_mode: bool = False,
        validation_required: bool = False,
        audit_log_path: str | None = None,
        divergence_threshold: float | None = None,
    ):
        self.upstream_url = upstream_url
        self.upstream_api_key = upstream_api_key
        self.encoder_model = encoder_model
        self.upstream_client = upstream_client
        self.cost_per_1k_input_tokens_usd = float(cost_per_1k_input_tokens_usd)
        self.batch_size = int(batch_size)
        
        # Compliance properties
        self.compliance_mode = compliance_mode
        self.validation_required = validation_required
        self.audit_log_path = audit_log_path
        self.divergence_threshold = divergence_threshold
        self.audit_events: list[dict[str, Any]] = []
        self._last_audit_hash: str = "0000000000000000000000000000000000000000000000000000000000000000"

        # Resolve compatibility between enable_hamming_router and hamming_router_mode
        if enable_hamming_router is not None and hamming_router_mode is not None:
            if enable_hamming_router is True and hamming_router_mode != "serve":
                raise ValueError(
                    f"Conflict: enable_hamming_router=True but hamming_router_mode={hamming_router_mode!r}. "
                    "Use only hamming_router_mode."
                )
            if enable_hamming_router is False and hamming_router_mode != "off":
                raise ValueError(
                    f"Conflict: enable_hamming_router=False but hamming_router_mode={hamming_router_mode!r}. "
                    "Use only hamming_router_mode."
                )

        if hamming_router_mode is not None:
            if hamming_router_mode not in ("off", "shadow", "serve"):
                raise ValueError(f"Invalid hamming_router_mode: {hamming_router_mode}")
            self.hamming_router_mode = hamming_router_mode
        elif enable_hamming_router is not None:
            self.hamming_router_mode = "serve" if enable_hamming_router else "off"
        else:
            self.hamming_router_mode = "off"

        if semantic_cache is not None:
            self.cache = semantic_cache
            self.cache.hamming_router_mode = self.hamming_router_mode
        else:
            runtime = self._build_runtime(
                encoder=encoder,
                encoder_model=encoder_model,
                d_model=d_model,
                sqlite_path=sqlite_path,
                batch_size=batch_size,
            )
            hamming_router = None
            if self.hamming_router_mode in ("serve", "shadow"):
                from latticememory.hamming_router import HammingRouter
                hamming_router = HammingRouter(
                    encoder=runtime.encoder,
                    d_model=runtime.d_model,
                    threshold=hamming_threshold,
                )
            self.cache = RFSnapSemanticCache(
                runtime=runtime,
                hamming_router=hamming_router,
                hamming_threshold=hamming_threshold,
                hamming_router_mode=self.hamming_router_mode,
            )

        # Calibration state telemetry
        self.hamming_router_calibrated = False
        self.hamming_router_recall = 0.0
        self.hamming_router_fp_rate = 0.0
        self.hamming_router_fp_budget = float(fp_budget)
        self.hamming_router_n_paraphrase_pairs = 0
        self.hamming_router_n_near_miss_pairs = 0
        self.hamming_router_encoder_model = getattr(self.cache.runtime, "model_id", None) or self.encoder_model
        self.hamming_router_encoder_dimension = getattr(self.cache.runtime, "d_model", None)
        self.hamming_router_calibration_file_hash = None
        self.hamming_router_calibration_timestamp = None

        # Handle calibration if calibration data is provided
        if calibration_data_path is not None:
            router = getattr(self.cache, "_hamming_router", None)
            if router is None:
                if require_calibration:
                    raise ValueError(
                        "Calibration requested (require_calibration=True) but no Hamming router "
                        "is configured on the semantic cache."
                    )
                # Fail closed: disable router and warn
                msg = "Optional Hamming router calibration requested but no Hamming router is configured. Disabling approximate cache."
                logger.warning(msg)
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                self.cache._hamming_router = None
            else:
                try:
                    if not os.path.exists(calibration_data_path):
                        raise ValueError(f"Calibration file does not exist: {calibration_data_path}")
                    with open(calibration_data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Determine file type and calibrate/load
                    if isinstance(data, dict) and data.get("artifact_type") == "latticememory_hamming_calibration":
                        from latticememory.hamming_router import (
                            validate_precalibrated_artifact_schema,
                        )
                        validate_precalibrated_artifact_schema(data)

                        # Validate version lock
                        cal_model = data["model"]
                        proxy_model = self.hamming_router_encoder_model
                        cal_dim = data["d_model"]
                        proxy_dim = self.hamming_router_encoder_dimension

                        if cal_model != proxy_model or cal_dim != proxy_dim:
                            raise ValueError(
                                f"Calibration version lock mismatch: proxy model/dimension is "
                                f"'{proxy_model}' (dim {proxy_dim}) but calibration file was built "
                                f"for model '{cal_model}' (dim {cal_dim})."
                            )

                        cal_threshold = data["calibration"]["threshold"]
                        if cal_threshold == -1:
                            raise ValueError("Calibration file contains an invalid threshold (-1).")

                        self.cache._hamming_threshold = cal_threshold
                        router.threshold = cal_threshold
                        self.hamming_router_calibrated = True
                        self.hamming_router_recall = data["calibration"]["recall"]
                        self.hamming_router_fp_rate = data["calibration"]["fp_rate"]
                        self.hamming_router_fp_budget = data["fp_budget"]
                        self.hamming_router_n_paraphrase_pairs = data["calibration"]["n_paraphrase_pairs"]
                        self.hamming_router_n_near_miss_pairs = data["calibration"]["n_near_miss_pairs"]
                        self.hamming_router_calibration_file_hash = data["calibration_data_sha256"]
                        # Convert ISO timestamp or retrieve timestamp
                        self.hamming_router_calibration_timestamp = data.get("created_at") or time.time()
                    else:
                        from latticememory.hamming_router import compute_calibration_data_sha256
                        validate_calibration_data_schema(data)
                        paraphrases = data["paraphrases"]
                        near_misses = data["near_misses"]

                        # Run calibration
                        cal = router.calibrate_threshold(
                            paraphrases, near_misses, fp_budget=self.hamming_router_fp_budget
                        )
                        cal_threshold = cal["threshold"]
                        if cal_threshold == -1:
                            raise ValueError("Calibration failed to find a valid threshold satisfying the FP budget.")

                        # Update threshold in cache and router
                        self.cache._hamming_threshold = cal_threshold
                        router.threshold = cal_threshold

                        # Record telemetry
                        self.hamming_router_calibrated = True
                        self.hamming_router_recall = cal["recall"]
                        self.hamming_router_fp_rate = cal["fp_rate"]
                        self.hamming_router_n_paraphrase_pairs = len(paraphrases)
                        self.hamming_router_n_near_miss_pairs = len(near_misses)
                        self.hamming_router_calibration_file_hash = compute_calibration_data_sha256(data)
                        self.hamming_router_calibration_timestamp = time.time()

                except Exception as exc:
                    if require_calibration:
                        raise ValueError(f"Hamming router calibration failed: {str(exc)}") from exc
                    # Fail closed: disable router, reset mode to off, and warn
                    msg = f"Optional Hamming router calibration failed: {exc}. Disabling Hamming router."
                    logger.warning(msg)
                    warnings.warn(msg, category=UserWarning, stacklevel=2)
                    self.hamming_router_mode = "off"
                    self.cache.hamming_router_mode = "off"
                    self.cache._hamming_router = None

        # Initialize/Restore Audit Trail
        if self.audit_log_path:
            if os.path.exists(self.audit_log_path):
                try:
                    with open(self.audit_log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                evt = json.loads(line)
                                self.audit_events.append(evt)
                                if "hash" in evt:
                                    self._last_audit_hash = evt["hash"]
                except Exception:
                    pass

    def _log_audit_event(
        self,
        event_type: str,
        prompt: str,
        key_hex: str,
        response_text: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evt = {
            "timestamp": time.time(),
            "event_type": event_type,
            "prompt": prompt,
            "e8_key": key_hex,
            "response": response_text,
            "metadata": extra or {},
            "previous_hash": self._last_audit_hash,
        }

        # Cryptographically chain entries to create a tamper-evident audit trail
        serialized = json.dumps(evt, sort_keys=True)
        evt_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        evt["hash"] = evt_hash
        self._last_audit_hash = evt_hash

        self.audit_events.append(evt)

        if self.audit_log_path:
            try:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(evt) + "\n")
            except Exception:
                pass

        return evt

    def _extract_response_text(self, body: Any) -> str:
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
            if "text" in body and isinstance(body["text"], str):
                return body["text"]
        return str(body)

    def create_app(self):
        _require_fastapi()
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="LatticeMemory LLM Cache Proxy", version="0.1.0")
        app.state.proxy = self

        @app.get("/health")
        async def health(request: Request) -> dict[str, Any]:
            proxy: LatticeLLMProxy = request.app.state.proxy
            router_active = getattr(proxy.cache, "_hamming_router", None) is not None
            is_reliable = (
                proxy.hamming_router_n_paraphrase_pairs >= 100
                and proxy.hamming_router_n_near_miss_pairs >= 100
            )
            return {
                "status": "healthy",
                "service": "latticememory-proxy",
                "cache_entries": proxy.cache.size,
                "compliance_mode": proxy.compliance_mode,
                "hamming_router": {
                    "enabled": router_active,
                    "mode": proxy.hamming_router_mode,
                    "threshold": int(proxy.cache._hamming_threshold) if router_active else None,
                    "calibrated": proxy.hamming_router_calibrated if router_active else False,
                    "recall": float(proxy.hamming_router_recall) if router_active else 0.0,
                    "fp_rate": float(proxy.hamming_router_fp_rate) if router_active else 0.0,
                    "fp_budget": float(proxy.hamming_router_fp_budget) if router_active else 0.0,
                    "n_paraphrase_pairs": int(proxy.hamming_router_n_paraphrase_pairs) if router_active else 0,
                    "n_near_miss_pairs": int(proxy.hamming_router_n_near_miss_pairs) if router_active else 0,
                    "reliable": is_reliable if router_active else False,
                    "encoder_model": proxy.hamming_router_encoder_model,
                    "encoder_dimension": proxy.hamming_router_encoder_dimension,
                    "calibration_file_hash": proxy.hamming_router_calibration_file_hash,
                    "calibration_timestamp": proxy.hamming_router_calibration_timestamp,
                }
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> JSONResponse:
            proxy: LatticeLLMProxy = request.app.state.proxy
            payload = await request.json()
            try:
                prompt = proxy._extract_prompt(payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            cached = proxy.cache.get(prompt)
            key_hex = cached.lattice_key.hex() if cached.lattice_key else "unknown"

            if cached.hit:
                # If compliance is active, check validation flag
                is_validated = (
                    cached.metadata.get("compliance_validated", False)
                    if proxy.compliance_mode
                    else True
                )

                if proxy.compliance_mode and proxy.validation_required and not is_validated:
                    # Target exists but requires validation/review first. Treat as compliance miss.
                    upstream_body = await proxy._call_upstream(payload)
                    response_text = proxy._extract_response_text(upstream_body)

                    proxy.cache.put(
                        prompt,
                        value=upstream_body,
                        metadata={
                            "model": payload.get("model"),
                            "compliance_validated": False,
                        },
                    )

                    evt = proxy._log_audit_event(
                        event_type="VAL_REQUIRED_MISS",
                        prompt=prompt,
                        key_hex=key_hex,
                        response_text=response_text,
                        extra={"reason": "Matched cache entry is unvalidated"},
                    )

                    return JSONResponse(
                        content=upstream_body,
                        headers={
                            "X-Lattice-Cache": "MISS",
                            "X-Lattice-Retrieval-Path": "compliance_unvalidated_miss",
                            "X-Lattice-Savings-USD": "0.000000",
                            "X-Lattice-Compliance": "PENDING_VALIDATION",
                            "X-Lattice-Audit-Hash": evt["hash"],
                        },
                    )

                # Output consistency (Divergence check)
                if proxy.compliance_mode and proxy.divergence_threshold is not None:
                    upstream_body = await proxy._call_upstream(payload)
                    new_text = proxy._extract_response_text(upstream_body)
                    cached_text = proxy._extract_response_text(cached.value)

                    import torch
                    import torch.nn.functional as F

                    emb_cached = proxy.cache.runtime._encode_texts([cached_text])[0]
                    emb_new = proxy.cache.runtime._encode_texts([new_text])[0]
                    cos_sim = float(
                        torch.dot(
                            F.normalize(emb_cached, p=2, dim=0),
                            F.normalize(emb_new, p=2, dim=0),
                        ).item()
                    )
                    cos_dist = 1.0 - cos_sim

                    if cos_dist > proxy.divergence_threshold:
                        # Divergence exceeds threshold. Mark for human review.
                        proxy.cache.put(
                            prompt,
                            value=cached.value,
                            metadata={
                                **cached.metadata,
                                "compliance_validated": False,
                                "divergence_distance": cos_dist,
                                "divergence_detected": True,
                            },
                        )

                        evt = proxy._log_audit_event(
                            event_type="DIVERGENCE_DETECTED",
                            prompt=prompt,
                            key_hex=key_hex,
                            response_text=cached_text,
                            extra={
                                "distance": cos_dist,
                                "threshold": proxy.divergence_threshold,
                                "new_response": new_text,
                            },
                        )

                        headers = {
                            "X-Lattice-Cache": "HIT_DIVERGED",
                            "X-Lattice-Retrieval-Path": cached.retrieval_path,
                            "X-Lattice-Savings-USD": proxy._format_usd(
                                proxy._estimate_savings(cached.value, prompt)
                            ),
                            "X-Lattice-Compliance": "DIVERGENCE_REVIEW_REQUIRED",
                            "X-Lattice-Audit-Hash": evt["hash"],
                            "X-Lattice-Hamming-Distance": str(cached.hamming_distance),
                        }
                        return JSONResponse(content=cached.value, headers=headers)

                # Validated/Allowed hit
                evt = proxy._log_audit_event(
                    event_type="HIT",
                    prompt=prompt,
                    key_hex=key_hex,
                    response_text=proxy._extract_response_text(cached.value),
                )
                headers = {
                    "X-Lattice-Cache": "HIT",
                    "X-Lattice-Retrieval-Path": cached.retrieval_path,
                    "X-Lattice-Savings-USD": proxy._format_usd(
                        proxy._estimate_savings(cached.value, prompt)
                    ),
                    "X-Lattice-Compliance": "APPROVED" if proxy.compliance_mode else "NONE",
                    "X-Lattice-Audit-Hash": evt["hash"],
                    "X-Lattice-Hamming-Distance": str(cached.hamming_distance),
                }
                return JSONResponse(content=cached.value, headers=headers)

            # Absolute Cache Miss or Shadow Hit
            if cached.shadow_hit:
                # Log the shadow hit event in the audit trail but do not serve it
                proxy._log_audit_event(
                    event_type="SHADOW_HIT",
                    prompt=prompt,
                    key_hex=key_hex,
                    response_text=proxy._extract_response_text(cached.shadow_value),
                    extra={
                        "hamming_distance": cached.shadow_hamming_distance,
                        "threshold": proxy.cache._hamming_threshold,
                    }
                )

            upstream_body = await proxy._call_upstream(payload)
            response_text = proxy._extract_response_text(upstream_body)

            metadata = {"model": payload.get("model")}
            if proxy.compliance_mode:
                metadata["compliance_validated"] = False

            proxy.cache.put(prompt, value=upstream_body, metadata=metadata)

            cached_refetch = proxy.cache.get(prompt)
            refetch_key_hex = (
                cached_refetch.lattice_key.hex() if cached_refetch.lattice_key else "unknown"
            )

            evt = proxy._log_audit_event(
                event_type="MISS",
                prompt=prompt,
                key_hex=refetch_key_hex,
                response_text=response_text,
            )

            headers = {
                "X-Lattice-Cache": "MISS",
                "X-Lattice-Retrieval-Path": "shadow_match_miss" if cached.shadow_hit else "miss",
                "X-Lattice-Savings-USD": "0.000000",
                "X-Lattice-Compliance": "PENDING_VALIDATION" if proxy.compliance_mode else "NONE",
                "X-Lattice-Audit-Hash": evt["hash"],
            }
            if cached.shadow_hit:
                headers["X-Lattice-Shadow-Hit"] = "true"
                headers["X-Lattice-Shadow-Distance"] = str(cached.shadow_shadow_hamming_distance if hasattr(cached, "shadow_shadow_hamming_distance") else cached.shadow_hamming_distance)
            return JSONResponse(content=upstream_body, headers=headers)

        @app.post("/v1/compliance/validate")
        async def validate_entry(request: Request) -> dict[str, Any]:
            proxy: LatticeLLMProxy = request.app.state.proxy
            if not proxy.compliance_mode:
                raise HTTPException(
                    status_code=400, detail="Compliance mode is disabled on this proxy"
                )

            payload = await request.json()
            prompt = payload.get("prompt")
            cache_id = payload.get("cache_id")

            target_entry = None
            if cache_id:
                target_entry = proxy.cache._entries.get(cache_id)
            elif prompt:
                cached = proxy.cache.get(prompt)
                if cached.hit:
                    for entry in proxy.cache._entries.values():
                        if entry.prompt == prompt or (entry.lattice_key == cached.lattice_key):
                            target_entry = entry
                            break

            if target_entry is None:
                raise HTTPException(status_code=404, detail="Cache entry not found")

            updated_metadata = {**target_entry.metadata, "compliance_validated": True}
            proxy.cache.put(
                target_entry.prompt, value=target_entry.value, metadata=updated_metadata
            )

            evt = proxy._log_audit_event(
                event_type="APPROVED",
                prompt=target_entry.prompt,
                key_hex=target_entry.lattice_key.hex() if target_entry.lattice_key else "unknown",
                response_text=proxy._extract_response_text(target_entry.value),
                extra={"validated_entry_id": target_entry.cache_id},
            )

            return {
                "status": "approved",
                "cache_id": target_entry.cache_id,
                "e8_key": target_entry.lattice_key.hex() if target_entry.lattice_key else None,
                "audit_hash": evt["hash"],
            }

        @app.get("/v1/compliance/audit-log")
        async def get_audit_log(request: Request) -> dict[str, Any]:
            proxy: LatticeLLMProxy = request.app.state.proxy
            if not proxy.compliance_mode:
                raise HTTPException(
                    status_code=400, detail="Compliance mode is disabled on this proxy"
                )

            is_valid_chain = True
            expected_prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"

            for evt in proxy.audit_events:
                raw_evt = {
                    "timestamp": evt["timestamp"],
                    "event_type": evt["event_type"],
                    "prompt": evt["prompt"],
                    "e8_key": evt["e8_key"],
                    "response": evt["response"],
                    "metadata": evt["metadata"],
                    "previous_hash": evt["previous_hash"],
                }
                serialized = json.dumps(raw_evt, sort_keys=True)
                computed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

                if computed != evt["hash"] or evt["previous_hash"] != expected_prev_hash:
                    is_valid_chain = False
                    break
                expected_prev_hash = evt["hash"]

            return {
                "events": proxy.audit_events,
                "chain_integrity_valid": is_valid_chain,
                "total_events": len(proxy.audit_events),
            }

        return app

    def _extract_prompt(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must include at least one user prompt")
        content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        raise ValueError("messages must include a non-empty final content field")

    async def _call_upstream(self, payload: dict[str, Any]) -> dict[str, Any]:
        from fastapi import HTTPException

        headers = {"Content-Type": "application/json"}
        if self.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"
        if self.upstream_client is not None:
            result = self.upstream_client(payload, headers)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise HTTPException(status_code=502, detail="upstream_client must return a JSON object")
            return result

        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.upstream_url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="upstream returned non-object JSON")
        return data

    def _estimate_savings(self, cached_value: Any, prompt: str) -> float:
        prompt_tokens = 0
        if isinstance(cached_value, dict):
            usage = cached_value.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(prompt) // 4)
        return (prompt_tokens / 1000.0) * self.cost_per_1k_input_tokens_usd

    @staticmethod
    def _format_usd(value: float) -> str:
        return f"{max(float(value), 0.0):.6f}"

    @staticmethod
    def _build_runtime(
        *,
        encoder: Any | None,
        encoder_model: str,
        d_model: int | None,
        sqlite_path: str | None,
        batch_size: int,
    ) -> RFSnapTextMemory:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(encoder_model)
        runtime_dim = int(d_model or getattr(encoder, "get_embedding_dimension", lambda: 0)() or 0)
        if runtime_dim <= 0:
            probe = encoder.encode(["dimension probe"], batch_size=batch_size)
            runtime_dim = int(getattr(probe, "shape", [0, 0])[-1])
        memory = RFSnapLatticeMemory(d_model=runtime_dim, sqlite_path=sqlite_path)
        return RFSnapTextMemory(
            encoder=encoder,
            d_model=runtime_dim,
            memory=memory,
            model_id=encoder_model,
            batch_size=batch_size,
        )
