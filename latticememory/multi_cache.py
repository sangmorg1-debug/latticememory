"""LatticeMultiCache — multi-tenant routing over isolated RFSnapSemanticCache instances.

Each tenant gets its own cache namespace so their entries never intermix.
Tenants are identified by a string tag (e.g. customer_id, team_id, environment).
Caches are created on first access via a user-supplied factory.

Quick-start::

    from latticememory.multi_cache import LatticeMultiCache
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("dfrokido/bge-large-e8-snap")

    def make_cache(tenant_id: str) -> RFSnapSemanticCache:
        runtime = RFSnapTextMemory(encoder=encoder, d_model=1024)
        return RFSnapSemanticCache(runtime=runtime)

    mc = LatticeMultiCache(cache_factory=make_cache)

    mc.put("acme", "How do I reset my password?", value="Visit /reset")
    mc.put("globex", "How do I reset my password?", value="Call IT helpdesk x123")

    print(mc.get("acme", "reset my password").value)    # "Visit /reset"
    print(mc.get("globex", "reset my password").value)  # "Call IT helpdesk x123"

Proxy integration::

    # In proxy_server.py, set LATTICE_TENANT_HEADER=X-Tenant-ID and the proxy
    # will call mc.get_cache(tenant_id) per request.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


class LatticeMultiCache:
    """A registry of per-tenant ``RFSnapSemanticCache`` instances.

    Parameters
    ----------
    cache_factory:
        Callable ``(tenant_id: str) -> RFSnapSemanticCache``.  Called once per
        new tenant; the returned cache is stored and reused for all future
        requests from that tenant.
    default_tenant:
        If set, ``get()`` / ``put()`` / ``get_cache()`` with no explicit
        tenant fall back to this tag instead of raising ``ValueError``.
    max_tenants:
        If set, raise ``RuntimeError`` when a new tenant would exceed this
        limit.  Use to guard against unbounded memory growth.
    """

    def __init__(
        self,
        cache_factory: Callable[[str], Any],
        *,
        default_tenant: str | None = None,
        max_tenants: int | None = None,
    ) -> None:
        self._factory = cache_factory
        self._default_tenant = default_tenant
        self._max_tenants = max_tenants
        self._caches: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Tenant management
    # ------------------------------------------------------------------

    def get_cache(self, tenant_id: str | None = None) -> Any:
        """Return (or lazily create) the cache for ``tenant_id``."""
        tid = tenant_id or self._default_tenant
        if tid is None:
            raise ValueError("tenant_id is required when no default_tenant is configured")
        with self._lock:
            if tid not in self._caches:
                if self._max_tenants is not None and len(self._caches) >= self._max_tenants:
                    raise RuntimeError(
                        f"LatticeMultiCache: max_tenants ({self._max_tenants}) reached. "
                        "Cannot create cache for new tenant."
                    )
                self._caches[tid] = self._factory(tid)
        return self._caches[tid]

    def tenant_ids(self) -> list[str]:
        """Return a sorted list of all registered tenant IDs."""
        with self._lock:
            return sorted(self._caches)

    def drop_tenant(self, tenant_id: str) -> bool:
        """Remove and discard a tenant's cache. Returns True if it existed."""
        with self._lock:
            return self._caches.pop(tenant_id, None) is not None

    def stats(self) -> dict[str, Any]:
        """Return per-tenant cache size summary."""
        with self._lock:
            return {
                "n_tenants": len(self._caches),
                "tenants": {
                    tid: {"size": getattr(cache, "size", len(getattr(cache, "_entries", {})))}
                    for tid, cache in self._caches.items()
                },
            }

    # ------------------------------------------------------------------
    # Pass-through helpers (so callers don't have to call get_cache() first)
    # ------------------------------------------------------------------

    def put(self, tenant_id: str | None, prompt: str, *, value: Any, **kwargs) -> Any:
        """Put an entry into the specified tenant's cache."""
        return self.get_cache(tenant_id).put(prompt, value=value, **kwargs)

    def get(self, tenant_id: str | None, prompt: str, **kwargs) -> Any:
        """Look up a prompt in the specified tenant's cache."""
        return self.get_cache(tenant_id).get(prompt, **kwargs)

    def delete(self, tenant_id: str | None, cache_id: str) -> bool:
        """Delete an entry from the specified tenant's cache."""
        return self.get_cache(tenant_id).delete(cache_id)

    def evict_expired_all(self) -> dict[str, int]:
        """Evict expired entries from all tenant caches. Returns {tenant_id: count}."""
        results: dict[str, int] = {}
        with self._lock:
            tenants = list(self._caches.items())
        for tid, cache in tenants:
            n = cache.evict_expired() if hasattr(cache, "evict_expired") else 0
            results[tid] = n
        return results
