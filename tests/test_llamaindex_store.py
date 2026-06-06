from __future__ import annotations

import pytest
import torch
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from latticememory.integrations.llamaindex import LatticeVectorStore
from latticememory.memory import RFSnapLatticeMemory
from tests.test_lattice_index import FakeEncoder


@pytest.fixture()
def memory():
    # Keep it in-memory with a fake 384-dimensional encoder
    mem = RFSnapLatticeMemory(d_model=384)
    return mem


def test_llamaindex_vector_store_add_and_query(memory):
    store = LatticeVectorStore(memory=memory)
    
    # Reconstruct some nodes
    node1 = TextNode(
        id_="node-1",
        text="The capital of France is Paris.",
        embedding=list(torch.randn(384).numpy().astype(float)),
        metadata={"country": "France"},
    )
    node2 = TextNode(
        id_="node-2",
        text="London is the capital of the United Kingdom.",
        embedding=list(torch.randn(384).numpy().astype(float)),
        metadata={"country": "UK"},
    )
    
    # Add nodes
    added_ids = store.add([node1, node2])
    assert added_ids == ["node-1", "node-2"]
    assert store.client.num_documents == 2
    
    # Query node-1 using its own embedding (should be exact match or closest neighbor)
    query = VectorStoreQuery(
        query_embedding=node1.embedding,
        similarity_top_k=1,
    )
    res = store.query(query)
    
    assert len(res.nodes) == 1
    assert res.ids == ["node-1"]
    assert res.nodes[0].text == "The capital of France is Paris."
    assert res.nodes[0].metadata == {"country": "France"}
    assert res.similarities[0] > 0.9  # Should have high cosine similarity


def test_llamaindex_vector_store_delete(memory):
    store = LatticeVectorStore(memory=memory)
    
    node1 = TextNode(
        id_="node-1",
        text="The capital of France is Paris.",
        embedding=list(torch.randn(384).numpy().astype(float)),
    )
    node2 = TextNode(
        id_="node-2",
        text="London is the capital of the United Kingdom.",
        embedding=list(torch.randn(384).numpy().astype(float)),
    )
    
    store.add([node1, node2])
    assert store.client.num_documents == 2
    
    # Delete node-1
    store.delete("node-1")
    assert store.client.num_documents == 1
    assert "node-1" not in store.client._doc_id_set
    
    # Querying node-1 should return nothing since it was deleted
    query1 = VectorStoreQuery(
        query_embedding=node1.embedding,
        similarity_top_k=1,
    )
    res1 = store.query(query1)
    assert "node-1" not in res1.ids
    
    # Querying node-2 should still succeed
    query2 = VectorStoreQuery(
        query_embedding=node2.embedding,
        similarity_top_k=1,
    )
    res2 = store.query(query2)
    assert res2.ids == ["node-2"]
