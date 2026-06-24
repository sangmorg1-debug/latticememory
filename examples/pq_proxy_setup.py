"""LatticeMemory proxy with the Product Quantization (PQ) addressing backend.

LatticeLLMProxy has no dedicated PQ constructor flag - the production proxy
constructor is already large (compliance mode, calibration, hamming routing,
warm-start, etc.) and a live server has worse failure modes than a quick
script if codebook-fitting is handled wrong, so PQ is wired in here through
the proxy's existing semantic_cache= injection point instead of adding more
constructor flags. See docs/manual-results/2026-06-24-open-vocab-semantic-addressing-redesign.md
for why PQ exists: the default E8 lattice addressing has ~0% real-model hit
rate on open-vocabulary paraphrases (and even on a genuine closed-vocabulary
held-out test) - PQ's data-calibrated codebooks reach 31%+ Recall@1 on the
same real PAWS benchmark.

THE FOOTGUN, stated plainly: unlike LatticeIndex (which auto-fits codebooks
on the first add() call as a safety net), PQLatticeDB constructed this way
has NO automatic fitting. If you start serving traffic before calling
lattice.fit(...), every request will raise "Index not fitted. Run fit()
first." - fit the codebooks BEFORE calling proxy.app or starting the
server, not after.

Run:
    python examples/pq_proxy_setup.py
"""
from __future__ import annotations

from latticememory.memory import RFSnapLatticeMemory
from latticememory.proxy import LatticeLLMProxy
from latticememory.rag.pq_retriever import PQLatticeDB
from latticememory.semantic_cache import RFSnapSemanticCache
from latticememory.text_runtime import RFSnapTextMemory


def build_pq_backed_proxy(
    upstream_url: str,
    calibration_sample: list[str],
    *,
    num_blocks: int = 8,
    codebook_size: int = 256,
    device: str = "cpu",
    **proxy_kwargs,
) -> LatticeLLMProxy:
    """Build a LatticeLLMProxy backed by PQ instead of the default E8 lattice.

    calibration_sample should be representative of your real traffic - ideally
    1,000+ real queries/documents from your actual domain. Codebooks are fit
    HERE, before the proxy is returned - this function will raise if you pass
    fewer texts than codebook_size (a real bug this exact scenario caught and
    fixed: see tests/test_pq_real_encoder.py).
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("dfrokido/bge-large-e8-snap", device=device)
    d_model = encoder.get_embedding_dimension()

    lattice = PQLatticeDB(d_model=d_model, num_blocks=num_blocks, codebook_size=codebook_size, device=device)
    sample_embs = encoder.encode(calibration_sample, normalize_embeddings=True, convert_to_tensor=True, device=device)
    lattice.fit(sample_embs)  # <-- must happen before any request is served

    memory = RFSnapLatticeMemory(d_model=d_model, lattice=lattice)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, memory=memory)
    cache = RFSnapSemanticCache(runtime=runtime)

    return LatticeLLMProxy(upstream_url=upstream_url, semantic_cache=cache, **proxy_kwargs)


if __name__ == "__main__":
    # A real calibration sample should come from your own traffic/domain logs -
    # this is a tiny illustrative set, not a real production sample.
    calibration_sample = [
        "What is the refund policy?", "How do I reset my password?", "Where is my order?",
        "Can I cancel my subscription?", "How long does shipping take?",
        "What is the refund policy.", "How do I change my password?", "When will my order arrive?",
    ]
    proxy = build_pq_backed_proxy(
        upstream_url="https://api.openai.com/v1",
        calibration_sample=calibration_sample,
        num_blocks=4,
        codebook_size=4,  # tiny, illustrative only - production default is 8/256, needs 256+ calibration texts
    )
    print("PQ-backed proxy constructed and codebooks fit.")
    print(f"Backend: {type(proxy.cache.runtime.memory.lattice).__name__}")
    print()
    print("To actually serve traffic, run this proxy with uvicorn/the lattice CLI")
    print("the same way as any other LatticeLLMProxy instance - see quickstart_proxy.py.")
