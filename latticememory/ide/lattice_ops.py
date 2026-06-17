from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def list_verticals() -> list[dict[str, str]]:
    return [
        {"class": "LatticeSOCMonitor", "capability": "alert dedup and gap detection"},
        {"class": "LatticeTicketAnalyzer", "capability": "ticket routing and documentation gaps"},
        {"class": "LatticeContentModerator", "capability": "semantic content moderation"},
        {"class": "LatticeClauseCoder", "capability": "legal clause classification"},
        {"class": "LatticeEdgeMemory", "capability": "on-device key recognition"},
        {"class": "LatticePrivateSync", "capability": "privacy-preserving key sync"},
        {"class": "LatticePromptFirewall", "capability": "prompt injection detection"},
        {"class": "LatticeSemanticRateLimiter", "capability": "intent-aware rate limiting"},
        {"class": "LatticeTrainingCleaner", "capability": "training corpus deduplication"},
    ]


def proxy_doctor(*, host: str = "127.0.0.1", port: int = 8000, admin_key: str | None = None) -> dict[str, Any]:
    base = f"http://{host}:{port}"
    result: dict[str, Any] = {"base_url": base, "reachable": False}
    try:
        result["health"] = _get_json(f"{base}/health")
        result["openapi"] = _get_json(f"{base}/openapi.json").get("info", {})
        result["reachable"] = True
        if admin_key:
            result["cache"] = _get_json(f"{base}/v1/cache", headers={"X-Lattice-Admin-Key": admin_key})
    except Exception as exc:
        result["error"] = str(exc)
    return result


def proxy_analytics(*, host: str = "127.0.0.1", port: int = 8000) -> dict[str, Any]:
    return _get_json(f"http://{host}:{port}/v1/analytics")


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc
