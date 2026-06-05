"""LangChain BaseCache adapter for LatticeMemory."""
from __future__ import annotations

from typing import Any

try:
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation
except ImportError as e:
    raise ImportError(
        "langchain-core is required for LatticeMemoryCache. "
        "Install it with: pip install langchain-core"
    ) from e

from latticememory.index import LatticeIndex, DEFAULT_MODEL


class LatticeMemoryCache(BaseCache):
    """LangChain cache backed by the E8 lattice.

    Semantically equivalent prompts (same E8 address) map to the same cache
    entry. Lookup is O(1) — a hash table lookup on the lattice key.

    Args:
        model: SentenceTransformer model name. Defaults to dfrokido/bge-large-e8-snap.
        hamming_threshold: Must be 0 (exact match only). Near-miss matching is not
            yet implemented; passing a non-zero value raises NotImplementedError.
        device: "auto" | "cuda" | "cpu"
        index: Inject an existing LatticeIndex (for testing or sharing).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        hamming_threshold: int = 0,
        device: str = "auto",
        index: LatticeIndex | None = None,
    ):
        if hamming_threshold != 0:
            raise NotImplementedError(
                "hamming_threshold > 0 (near-miss matching) is not yet implemented. "
                "Use hamming_threshold=0 for exact matching."
            )
        self._index = index or LatticeIndex(model=model, device=device)
        self._hamming_threshold = hamming_threshold
        self._store: dict[tuple[str, str], list[Generation]] = {}

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        address = self._index.snap(prompt)
        return self._store.get((address, llm_string))

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        address = self._index.snap(prompt)
        self._store[(address, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        self._store.clear()
