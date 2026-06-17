"""Gate 4: Two AI agents sharing semantic memory over AgentMemorySync.

Demonstrates:
  - Agent A learns facts; Agent B starts empty
  - B syncs from A (pull): receives all keys and documents
  - A learns a new fact and broadcasts its key (push): B receives automatically
  - Final diff confirms both agents hold identical semantic memory
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.agent_sync import AgentMemorySync
from latticememory.memory import MemoryDocument, RFSnapLatticeMemory
from latticememory.text_runtime import RFSnapTextMemory


class FakeEncoder:
    """Deterministic mock encoder — no model download required."""

    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, **kwargs):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        vecs = []
        for s in texts:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return vecs[0] if single else np.stack(vecs)


def _make_agent(d_model: int = 384) -> AgentMemorySync:
    encoder = FakeEncoder(d_model)
    lm = RFSnapLatticeMemory(d_model=d_model)
    rt = RFSnapTextMemory(encoder=encoder, d_model=d_model, memory=lm)
    return AgentMemorySync(runtime=rt)


def _index(sync: AgentMemorySync, text: str) -> bytes:
    """Index one text into an agent's memory and return its E8 lattice key."""
    emb = sync.runtime._encode_texts([text])[0]
    doc_id = "doc-" + hashlib.sha1(text.encode()).hexdigest()[:16]
    doc = MemoryDocument(doc_id=doc_id, text=text, embedding=emb)
    sync.runtime.memory.add_documents([doc])
    return sync.runtime.memory.lattice._keys[doc_id]


def main() -> None:
    D = 384
    sep = "=" * 68

    print(sep)
    print("  LatticeMemory - Gate 4: Agent Swarm Memory Sync Demo")
    print(sep)

    agent_a = _make_agent(D)
    agent_b = _make_agent(D)

    # ── Phase 1: Agent A learns facts ──────────────────────────────────
    facts = [
        "The capital of France is Paris.",
        "Python was created by Guido van Rossum in 1991.",
        "The speed of light is approximately 299,792 km/s.",
        "Water freezes at 0 degrees Celsius at standard pressure.",
    ]
    print(f"\n[Phase 1] Agent-A learns {len(facts)} facts")
    for fact in facts:
        key = _index(agent_a, fact)
        print(f"  E8:{key.hex()[:20]}...  {fact[:55]!r}")

    print(f"\n  Agent-A known keys : {len(agent_a.get_known_keys())}")
    print(f"  Agent-B known keys : {len(agent_b.get_known_keys())}  (empty)")

    # ── Phase 2: Register peers and pull-sync B from A ─────────────────
    print(f"\n[Phase 2] Register peers then pull-sync Agent-B <- Agent-A")
    agent_a.register_peer(agent_b)
    agent_b.sync_from_peer(agent_a)

    b_after_pull = agent_b.get_known_keys()
    assert b_after_pull == agent_a.get_known_keys(), "Pull sync failed"
    print(f"  Agent-B known keys after pull : {len(b_after_pull)}")
    print("  [PASS] Agent-B now holds all of Agent-A's keys and documents")

    # ── Phase 3: A learns a new fact; push-broadcast key to B ──────────
    new_fact = "The Eiffel Tower stands 330 metres tall in Paris."
    print(f"\n[Phase 3] Agent-A learns new fact and broadcasts key (push)")
    print(f"  Fact: {new_fact!r}")
    count_before = len(agent_b.get_known_keys())
    new_key = _index(agent_a, new_fact)
    agent_a.share(new_key)

    assert new_key in agent_b.get_known_keys(), "Push broadcast failed"
    count_after = len(agent_b.get_known_keys())
    print(f"  Broadcast key : E8:{new_key.hex()[:20]}...")
    print(f"  Agent-B keys  : {count_before} -> {count_after}  (received via push)")

    # ── Phase 4: B retrieves the document it received via broadcast ────
    print(f"\n[Phase 4] Agent-B retrieves document received via broadcast")
    docs = agent_b.get_documents_for_key(new_key)
    for doc in docs:
        print(f"  doc_id : {doc.doc_id}")
        print(f"  text   : {doc.text!r}")

    # ── Final diff ─────────────────────────────────────────────────────
    diff = agent_a.diff(agent_b.get_known_keys())
    print(f"\n[Summary]")
    print(f"  Agent-A keys  : {len(agent_a.get_known_keys())}")
    print(f"  Agent-B keys  : {len(agent_b.get_known_keys())}")
    print(f"  Extra in A    : {len(diff['extra'])}   (keys A has that B is missing)")
    print(f"  Missing in B  : {len(diff['missing'])}  (keys B has that A is missing)")
    assert diff["extra"] == set(), "Agent-B is missing keys from A"
    assert diff["missing"] == set(), "Agent-A is missing keys from B"
    print(f"\n  PASS - both agents share identical semantic memory.")
    print(sep)


if __name__ == "__main__":
    main()
