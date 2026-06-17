"""Tests for the agent swarm memory sync demo (Gate 4)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from latticememory.agent_sync import AgentMemorySync
from latticememory.memory import MemoryDocument, RFSnapLatticeMemory
from latticememory.text_runtime import RFSnapTextMemory


class FakeEncoder:
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
    emb = sync.runtime._encode_texts([text])[0]
    doc_id = "doc-" + hashlib.sha1(text.encode()).hexdigest()[:16]
    doc = MemoryDocument(doc_id=doc_id, text=text, embedding=emb)
    sync.runtime.memory.add_documents([doc])
    return sync.runtime.memory.lattice._keys[doc_id]


# ── pull sync (sync_from_peer) ────────────────────────────────────────────────

def test_pull_sync_transfers_all_keys():
    agent_a = _make_agent()
    agent_b = _make_agent()

    facts = ["Paris is the capital of France.", "Water boils at 100C.", "Einstein developed relativity."]
    for fact in facts:
        _index(agent_a, fact)

    assert len(agent_b.get_known_keys()) == 0
    agent_a.register_peer(agent_b)
    agent_b.sync_from_peer(agent_a)

    assert agent_b.get_known_keys() == agent_a.get_known_keys()


def test_pull_sync_transfers_document_text():
    agent_a = _make_agent()
    agent_b = _make_agent()

    text = "The speed of light is 299,792 km/s."
    _index(agent_a, text)

    agent_a.register_peer(agent_b)
    agent_b.sync_from_peer(agent_a)

    # Agent B should be able to retrieve the document text
    key = list(agent_a.get_known_keys())[0]
    docs = agent_b.get_documents_for_key(key)
    assert len(docs) == 1
    assert docs[0].text == text


def test_pull_sync_empty_source():
    agent_a = _make_agent()
    agent_b = _make_agent()
    _index(agent_b, "some fact")

    agent_a.register_peer(agent_b)
    # Syncing from empty agent_a should not remove B's keys
    agent_b.sync_from_peer(agent_a)
    assert len(agent_b.get_known_keys()) == 1


# ── push broadcast (share) ────────────────────────────────────────────────────

def test_push_broadcast_delivers_key_to_peer():
    agent_a = _make_agent()
    agent_b = _make_agent()

    agent_a.register_peer(agent_b)
    text = "New discovery: quantum entanglement teleportation."
    key = _index(agent_a, text)
    agent_a.share(key)

    assert key in agent_b.get_known_keys()


def test_push_broadcast_delivers_document_text():
    agent_a = _make_agent()
    agent_b = _make_agent()

    agent_a.register_peer(agent_b)
    text = "Mount Everest is 8,849 metres tall."
    key = _index(agent_a, text)
    agent_a.share(key)

    docs = agent_b.get_documents_for_key(key)
    assert len(docs) == 1
    assert docs[0].text == text


def test_push_broadcast_does_not_duplicate_known_key():
    agent_a = _make_agent()
    agent_b = _make_agent()

    text = "Gravity pulls objects downward."
    key_a = _index(agent_a, text)
    key_b = _index(agent_b, text)  # B already has the same key

    agent_a.register_peer(agent_b)
    agent_a.share(key_a)

    # B's key count should still be 1, not 2
    assert len(agent_b.get_known_keys()) == 1


# ── diff ──────────────────────────────────────────────────────────────────────

def test_diff_identical_agents_has_empty_sets():
    agent_a = _make_agent()
    agent_b = _make_agent()

    _index(agent_a, "shared fact A")
    _index(agent_b, "shared fact A")  # same text → same key

    diff = agent_a.diff(agent_b.get_known_keys())
    assert diff["extra"] == set()
    assert diff["missing"] == set()


def test_diff_detects_missing_and_extra():
    agent_a = _make_agent()
    agent_b = _make_agent()

    _index(agent_a, "only in A")
    _index(agent_b, "only in B and this is a very different topic about robots")

    diff = agent_a.diff(agent_b.get_known_keys())
    assert len(diff["missing"]) >= 1  # B has something A doesn't
    assert len(diff["extra"]) >= 1    # A has something B doesn't


# ── full swarm cycle (matches the demo exactly) ───────────────────────────────

def test_full_swarm_sync_cycle():
    """End-to-end: Phase 1 learn, Phase 2 pull-sync, Phase 3 push-broadcast."""
    D = 384
    agent_a = _make_agent(D)
    agent_b = _make_agent(D)

    # Phase 1: A learns 4 facts
    facts = [
        "The capital of France is Paris.",
        "Python was created by Guido van Rossum in 1991.",
        "The speed of light is approximately 299,792 km/s.",
        "Water freezes at 0 degrees Celsius at standard pressure.",
    ]
    for fact in facts:
        _index(agent_a, fact)

    assert len(agent_a.get_known_keys()) == len(facts)
    assert len(agent_b.get_known_keys()) == 0

    # Phase 2: register peers and pull-sync B from A
    agent_a.register_peer(agent_b)
    agent_b.sync_from_peer(agent_a)
    assert agent_b.get_known_keys() == agent_a.get_known_keys()

    # Phase 3: A learns a 5th fact and broadcasts
    new_fact = "The Eiffel Tower stands 330 metres tall in Paris."
    new_key = _index(agent_a, new_fact)
    agent_a.share(new_key)
    assert new_key in agent_b.get_known_keys()

    # Final: both agents share identical memory
    diff = agent_a.diff(agent_b.get_known_keys())
    assert diff["extra"] == set()
    assert diff["missing"] == set()
