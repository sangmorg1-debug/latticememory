from __future__ import annotations

import os
import pytest
import torch
import numpy as np

from latticememory.memory import RFSnapLatticeMemory, MemoryDocument, DenseVectorFallback
from latticememory.sqlite_store import LatticeSqliteStore
from tests.test_lattice_index import FakeEncoder


@pytest.fixture()
def temp_db_path(tmp_path):
    return tmp_path / "test_index.db"


def test_sqlite_store_adds_and_retrieves(temp_db_path):
    store = LatticeSqliteStore(temp_db_path)
    doc = MemoryDocument(
        doc_id="doc-1",
        text="Hello world",
        embedding=torch.randn(384),
        metadata={"category": "test"},
    )
    lattice_keys = {"doc-1": b"dummy-key-for-test-384"}
    
    store.add_documents([doc], lattice_keys)
    assert store.count() == 1
    
    retrieved = store.get_document("doc-1")
    assert retrieved is not None
    assert retrieved.doc_id == "doc-1"
    assert retrieved.text == "Hello world"
    assert retrieved.metadata == {"category": "test"}
    assert torch.allclose(torch.as_tensor(retrieved.embedding), torch.as_tensor(doc.embedding))
    
    store.close()


def test_memory_sqlite_sync(temp_db_path):
    # Setup RFSnapLatticeMemory with SQLite persistence
    memory = RFSnapLatticeMemory(d_model=384, sqlite_path=temp_db_path)
    doc = MemoryDocument(
        doc_id="doc-1",
        text="Test text",
        embedding=torch.randn(384),
        metadata={"foo": "bar"},
    )
    memory.add_documents([doc])
    
    # Check that it synced to DB
    assert memory.sqlite_store.count() == 1
    db_doc = memory.sqlite_store.get_document("doc-1")
    assert db_doc is not None
    assert db_doc.text == "Test text"
    
    # Create a new memory instance loading from the same DB
    loaded_memory = RFSnapLatticeMemory(d_model=384, sqlite_path=temp_db_path)
    assert loaded_memory.num_documents == 1
    assert loaded_memory._texts["doc-1"] == "Test text"
    assert loaded_memory._metadata["doc-1"] == {"foo": "bar"}
    
    # Check exact key matches match
    assert loaded_memory.lattice._keys["doc-1"] == memory.lattice._keys["doc-1"]


def test_memory_save_and_load_sqlite(temp_db_path):
    memory = RFSnapLatticeMemory(d_model=384)
    doc = MemoryDocument(
        doc_id="doc-1",
        text="Save me",
        embedding=torch.randn(384),
        metadata={"prio": 1},
    )
    memory.add_documents([doc])
    
    # Save manually
    memory.save_to_sqlite(str(temp_db_path))
    assert os.path.exists(temp_db_path)
    
    # Load manually
    loaded = RFSnapLatticeMemory.load_from_sqlite(str(temp_db_path))
    assert loaded.num_documents == 1
    assert loaded._texts["doc-1"] == "Save me"
    assert loaded._metadata["doc-1"] == {"prio": 1}


def test_memory_save_and_load_sqlite_with_fallback(temp_db_path):
    fallback = DenseVectorFallback(d_model=384, quantization_bits=8)
    memory = RFSnapLatticeMemory(d_model=384, fallback=fallback)
    doc = MemoryDocument(
        doc_id="doc-1",
        text="Save fallback text",
        embedding=torch.randn(384),
        metadata={"prio": 2},
    )
    memory.add_documents([doc])
    
    memory.save_to_sqlite(str(temp_db_path))
    
    new_fallback = DenseVectorFallback(d_model=384, quantization_bits=8)
    loaded = RFSnapLatticeMemory.load_from_sqlite(str(temp_db_path), fallback=new_fallback)
    
    assert loaded.num_documents == 1
    assert loaded.fallback is not None
    assert loaded.fallback.num_documents == 1
    assert loaded.fallback.text_for("doc-1") == "Save fallback text"


def test_semantic_cache_sqlite_persistence(temp_db_path):
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.text_runtime import RFSnapTextMemory
    
    encoder = FakeEncoder(d_model=384)
    # Build first instance
    memory = RFSnapLatticeMemory(d_model=384, sqlite_path=temp_db_path)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=384, memory=memory)
    cache = RFSnapSemanticCache(runtime=runtime)
    
    # Store some values
    val = {"choices": [{"message": {"content": "Hello response"}}]}
    cache.put("What is E8?", value=val, metadata={"user": "alice"})
    assert cache.size == 1
    
    # Verify hit on first instance
    res = cache.get("What is E8?")
    assert res.hit is True
    assert res.value == val
    assert res.metadata == {"user": "alice"}
    
    # Close first store connection
    memory.sqlite_store.close()
    
    # Recreate the memory and cache using same SQLite DB path
    memory2 = RFSnapLatticeMemory(d_model=384, sqlite_path=temp_db_path)
    runtime2 = RFSnapTextMemory(encoder=encoder, d_model=384, memory=memory2)
    cache2 = RFSnapSemanticCache(runtime=runtime2)
    
    # Check that it reloaded the entries on startup
    assert cache2.size == 1
    res2 = cache2.get("What is E8?")
    assert res2.hit is True
    assert res2.value == val
    assert res2.metadata == {"user": "alice"}

