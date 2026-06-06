from __future__ import annotations

import time
import pytest
import torch

from latticememory.stream import LatticeStreamDedup
from latticememory.text_runtime import RFSnapTextMemory
from latticememory.memory import RFSnapLatticeMemory
from tests.test_lattice_index import FakeEncoder


def test_streaming_exact_dedup():
    encoder = FakeEncoder(d_model=384)
    memory = RFSnapLatticeMemory(d_model=384)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=384, memory=memory)
    
    # Init stream dedup with 10s window
    stream = LatticeStreamDedup(runtime=runtime, time_window_seconds=10.0, allow_neighborhood=False)
    
    t0 = time.time()
    
    # Process first unique item
    res1 = stream.process("Hello streaming world", timestamp=t0)
    assert res1["is_duplicate"] is False
    assert res1["canonical_id"] is None
    assert res1["match_path"] == "miss"
    assert stream.size == 1
    
    # Process identical repeat within window
    res2 = stream.process("Hello streaming world", timestamp=t0 + 2.0)
    assert res2["is_duplicate"] is True
    assert res2["canonical_id"] == res1["doc_id"]
    assert res2["match_path"] == "exact"
    assert stream.size == 1  # repeats shouldn't grow the window size
    
    # Process identical repeat AFTER window expires
    res3 = stream.process("Hello streaming world", timestamp=t0 + 15.0)
    assert res3["is_duplicate"] is False
    assert res3["canonical_id"] is None
    assert res3["match_path"] == "miss"
    assert stream.size == 1  # oldest was pruned, so size remains 1 (the new miss)


def test_streaming_max_entries():
    encoder = FakeEncoder(d_model=384)
    memory = RFSnapLatticeMemory(d_model=384)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=384, memory=memory)
    
    # Max entries = 2
    stream = LatticeStreamDedup(runtime=runtime, time_window_seconds=100.0, max_entries=2, allow_neighborhood=False)
    
    res1 = stream.process("first item")
    res2 = stream.process("second item")
    assert stream.size == 2
    
    # This should evict "first item"
    res3 = stream.process("third item")
    assert stream.size == 2
    
    # Check that "first item" is no longer known (misses)
    res4 = stream.process("first item")
    assert res4["is_duplicate"] is False
    assert res4["match_path"] == "miss"


def test_streaming_neighborhood_dedup():
    encoder = FakeEncoder(d_model=384)
    # Configure beam radius 1 to scan Hamming-1 neighborhood
    memory = RFSnapLatticeMemory(d_model=384, beam_radius=1)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=384, memory=memory)
    
    stream = LatticeStreamDedup(runtime=runtime, time_window_seconds=10.0, allow_neighborhood=True)
    
    # Mock behavior of FakeEncoder:
    # We want two different texts to have keys that are Hamming-1 distance apart.
    # FakeEncoder hashes sentences and returns deterministic embeddings.
    # To construct a Hamming-1 relation, let's use texts that we know will have similar embeddings,
    # or let's mock the encoder to return a custom embedding where we control the Hamming distance!
    # Or, we can just use the same text with a minor whitespace to check if FakeEncoder snaps them,
    # but wait, FakeEncoder uses MD5 of the text. MD5 hashes of different strings are completely unrelated.
    # So different text will produce completely different random keys (Hamming distance > 100).
    # To test neighborhood matching deterministically, let's mock the runtime._encode_texts method or E8 snapping!
    
    # Let's mock runtime._encode_texts to return a custom coordinate we specify.
    original_encode = runtime._encode_texts
    
    # Let's generate a base coordinate
    base_emb = torch.randn(384)
    # Snapped base key
    base_key = memory.lattice_key_for(base_emb)
    
    # Let's construct a neighbor embedding by perturbing one block of the base embedding
    # so E8 snapping maps it to a Hamming-1 key.
    # We can just mock lattice_key_for to return:
    # - base_key for "text A"
    # - a mutated key (Hamming-1) for "text B"
    neighbor_key = bytearray(base_key)
    neighbor_key[0] = (neighbor_key[0] + 1) % 240
    neighbor_key = bytes(neighbor_key)
    
    # Double check distance is exactly 1
    assert sum(a != b for a, b in zip(base_key, neighbor_key)) == 1
    
    # Mock the memory.lattice_key_for method!
    original_key_for = memory.lattice_key_for
    def mocked_key_for(embedding):
        # We can inspect the text by checking which query text it matches
        # but lattice_key_for only gets the embedding tensor.
        # So we can mock runtime._encode_texts to return:
        # - base_emb for "concept A"
        # - neighbor_emb for "concept B" (which snaps to neighbor_key)
        return original_key_for(embedding)
        
    # Let's construct the actual neighbor embedding.
    # E8LatticeDB._quantize_to_indices snaps 8-dim blocks to nearest E8 point.
    # To change exactly 1 block's snapped index, we can just edit the embedding.
    # But even simpler, we can mock `lattice_key_for` to return:
    # - base_key if text starts with "concept A"
    # - neighbor_key if text starts with "concept B"
    # Wait, `lattice_key_for` is called inside `stream.process` like:
    # `lattice_key = self.runtime.memory.lattice_key_for(embedding)`
    # Since `process` gets `text`, and calls `runtime._encode_texts([text])`,
    # we can mock `runtime._encode_texts` to return:
    # - base_emb for "concept A"
    # - a perturbed vector for "concept B"
    # Or, we can just mock `lattice_key_for` dynamically based on some state.
    # Let's mock both:
    def mock_encode_texts(texts):
        # Return a tensor of shape [len(texts), 384]
        res_embs = []
        for t in texts:
            if t == "concept A":
                res_embs.append(base_emb)
            elif t == "concept B":
                # Let's construct a vector that snaps to neighbor_key!
                # We can do this by taking the reconstructed vector for neighbor_key:
                # E8LatticeDB has a _codebook of size 240x8.
                # Let's reconstruct it:
                reconstructed = torch.cat([memory.lattice._codebook[idx] for idx in neighbor_key])
                res_embs.append(reconstructed)
            else:
                res_embs.append(base_emb)
        return torch.stack(res_embs)
        
    runtime._encode_texts = mock_encode_texts
    
    t0 = time.time()
    resA = stream.process("concept A", timestamp=t0)
    assert resA["is_duplicate"] is False
    
    resB = stream.process("concept B", timestamp=t0 + 1.0)
    # Concept B is a Hamming-1 neighbor of Concept A, so it should be a duplicate!
    assert resB["is_duplicate"] is True
    assert resB["canonical_id"] == resA["doc_id"]
    assert resB["match_path"] == "neighborhood"
