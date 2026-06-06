import inspect
from collections.abc import Callable
from typing import Any

from latticememory.memory import RFSnapLatticeMemory
from latticememory.semantic_cache import RFSnapSemanticCache
from latticememory.text_runtime import RFSnapTextMemory


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LatticeLLMProxy requires FastAPI. Install with: pip install 'latticememory[proxy]'"
        ) from exc


UpstreamClient = Callable[[dict[str, Any], dict[str, str]], Any]


class LatticeLLMProxy:
    """OpenAI-compatible chat-completions proxy with E8 semantic caching."""

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
    ):
        self.upstream_url = upstream_url
        self.upstream_api_key = upstream_api_key
        self.encoder_model = encoder_model
        self.upstream_client = upstream_client
        self.cost_per_1k_input_tokens_usd = float(cost_per_1k_input_tokens_usd)
        self.batch_size = int(batch_size)
        if semantic_cache is not None:
            self.cache = semantic_cache
        else:
            runtime = self._build_runtime(
                encoder=encoder,
                encoder_model=encoder_model,
                d_model=d_model,
                sqlite_path=sqlite_path,
                batch_size=batch_size,
            )
            self.cache = RFSnapSemanticCache(runtime=runtime)

    def create_app(self):
        _require_fastapi()
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="LatticeMemory LLM Cache Proxy", version="0.1.0")
        app.state.proxy = self

        @app.get("/health")
        async def health(request: Request) -> dict[str, Any]:
            proxy: LatticeLLMProxy = request.app.state.proxy
            return {
                "status": "healthy",
                "service": "latticememory-proxy",
                "cache_entries": proxy.cache.size,
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
            if cached.hit:
                headers = {
                    "X-Lattice-Cache": "HIT",
                    "X-Lattice-Retrieval-Path": cached.retrieval_path,
                    "X-Lattice-Savings-USD": proxy._format_usd(proxy._estimate_savings(cached.value, prompt)),
                }
                return JSONResponse(content=cached.value, headers=headers)

            upstream_body = await proxy._call_upstream(payload)
            proxy.cache.put(prompt, value=upstream_body, metadata={"model": payload.get("model")})
            return JSONResponse(
                content=upstream_body,
                headers={
                    "X-Lattice-Cache": "MISS",
                    "X-Lattice-Retrieval-Path": "miss",
                    "X-Lattice-Savings-USD": "0.000000",
                },
            )

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
