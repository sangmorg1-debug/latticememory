from __future__ import annotations

import pytest
import torch

from latticememory.text_runtime import RFSnapTextMemory
from latticememory.memory import RFSnapLatticeMemory, MemoryDocument
from latticememory.agent_sync import (
    AgentMemorySync,
    make_autogen_sync_tools,
    LangGraphLatticeAdapter,
)
from tests.test_lattice_index import FakeEncoder


def _create_agent_sync(d_model: int = 384) -> AgentMemorySync:
    encoder = FakeEncoder(d_model=d_model)
    memory = RFSnapLatticeMemory(d_model=d_model)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, memory=memory)
    return AgentMemorySync(runtime)


def test_peer_registration():
    sync1 = _create_agent_sync()
    sync2 = _create_agent_sync()

    sync1.register_peer(sync2)

    assert sync2 in sync1._peers
    assert sync1 in sync2._peers


def test_get_known_keys_and_diff():
    sync1 = _create_agent_sync()
    sync2 = _create_agent_sync()

    # Index some doc in sync1
    text1 = "artificial general intelligence"
    emb1 = sync1.runtime._encode_texts([text1])[0]
    doc1 = MemoryDocument(doc_id="doc1", text=text1, embedding=emb1)
    sync1.receive_document(doc1)

    # Index different doc in sync2
    text2 = "quantum computing breakthroughs"
    emb2 = sync2.runtime._encode_texts([text2])[0]
    doc2 = MemoryDocument(doc_id="doc2", text=text2, embedding=emb2)
    sync2.receive_document(doc2)

    keys1 = sync1.get_known_keys()
    keys2 = sync2.get_known_keys()

    assert len(keys1) == 1
    assert len(keys2) == 1
    assert keys1 != keys2

    # Compare keys1 from perspective of sync2
    diffs2 = sync2.diff(keys1)
    # sync2 is missing keys1, and sync2 has extra keys2
    assert keys1.issubset(diffs2["missing"])
    assert keys2.issubset(diffs2["extra"])


def test_sync_from_peer():
    sync1 = _create_agent_sync()
    sync2 = _create_agent_sync()

    # Seed sync1 with 2 documents
    texts = ["deep learning", "gradient descent"]
    for i, t in enumerate(texts):
        emb = sync1.runtime._encode_texts([t])[0]
        doc = MemoryDocument(doc_id=f"sync1-doc-{i}", text=t, embedding=emb)
        sync1.receive_document(doc)

    assert len(sync2.get_known_keys()) == 0

    # Sync sync2 from sync1
    sync2.sync_from_peer(sync1)

    assert sync2.get_known_keys() == sync1.get_known_keys()
    assert sync2.runtime.memory.get_document_text("sync1-doc-0") == "deep learning"
    assert sync2.runtime.memory.get_document_text("sync1-doc-1") == "gradient descent"


def test_broadcast_share():
    sync1 = _create_agent_sync()
    sync2 = _create_agent_sync()

    sync1.register_peer(sync2)

    # Seed sync1
    text = "reinforcement learning from human feedback"
    emb = sync1.runtime._encode_texts([text])[0]
    doc = MemoryDocument(doc_id="rlhf", text=text, embedding=emb)
    sync1.receive_document(doc)

    key = list(sync1.get_known_keys())[0]

    # share the key with peer swarm
    sync1.share(key)

    # sync2 should automatically receive it via broadcast callback
    assert key in sync2.get_known_keys()
    assert sync2.runtime.memory.get_document_text("rlhf") == text


def test_autogen_adapters():
    sync = _create_agent_sync()
    peer_sync = _create_agent_sync()
    sync.register_peer(peer_sync)
    
    text = "adapter test doc"
    emb = peer_sync.runtime._encode_texts([text])[0]
    doc = MemoryDocument(doc_id="autogen-1", text=text, embedding=emb)
    peer_sync.receive_document(doc)

    tools = make_autogen_sync_tools(sync)
    
    # get_keys from peer_sync to know which E8 key to query
    peer_keys = [k.hex() for k in peer_sync.get_known_keys()]
    assert len(peer_keys) == 1
    key_hex = peer_keys[0]

    # 1. get_keys (local sync has no keys yet)
    keys = tools["get_keys"]()
    assert len(keys) == 0

    # 2. request_key_docs
    docs = tools["request_key_docs"](key_hex)
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "autogen-1"
    assert docs[0]["text"] == text

    # 3. share_key from the peer
    peer_tools = make_autogen_sync_tools(peer_sync)
    res = peer_tools["share_key"](key_hex)
    assert "Successfully shared key" in res
    assert key_hex in [k.hex() for k in sync.get_known_keys()]


def test_langgraph_adapter():
    sync = _create_agent_sync()
    adapter = LangGraphLatticeAdapter(sync)

    # We want to test langgraph node logic:
    # 1. Add input_texts to register locally and produce keys
    state = {
        "shared_keys": [],
        "input_texts": ["hello from langgraph", "another node state item"]
    }
    
    output = adapter.sync_node(state)
    assert len(output["generated_keys"]) == 2
    assert len(sync.get_known_keys()) == 2

    # 2. Test receiving shared_keys and syncing them (we mock a peer and request returning some docs)
    peer_sync = _create_agent_sync()
    text = "concept from peer"
    emb = peer_sync.runtime._encode_texts([text])[0]
    doc = MemoryDocument(doc_id="peer-doc-1", text=text, embedding=emb)
    peer_sync.receive_document(doc)
    peer_key = list(peer_sync.get_known_keys())[0].hex()

    # Register peer so requesting from peer_sync works
    sync.register_peer(peer_sync)

    state_with_peer = {
        "shared_keys": [peer_key],
        "input_texts": []
    }
    
    output_sync = adapter.sync_node(state_with_peer)
    assert "peer-doc-1" in output_sync["indexed_doc_ids"]
    assert sync.runtime.memory.get_document_text("peer-doc-1") == text
