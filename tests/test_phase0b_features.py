import os
import pytest
from latticememory.agent_memory import AgentEpisodicMemory
from latticememory.dedup import LatticeDedup
from tests.test_lattice_index import FakeEncoder

def test_agent_episodic_memory():
    db_path = "test_agent_mem.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    try:
        mem = AgentEpisodicMemory(d_model=384, sqlite_path=db_path)
        mem.encoder = FakeEncoder(d_model=384)
        
        # Add a memory
        r1 = mem.add_episode("User prefers green tea.")
        assert r1["version"] == 1
        assert r1["is_duplicate"] is False
        
        # Add duplicate memory
        r2 = mem.add_episode("User prefers green tea.")
        assert r2["version"] == 2
        assert r2["is_duplicate"] is True
        
        # Retrieve history
        history = mem.get_episode_history("User prefers green tea.")
        assert len(history) == 2
        assert history[0]["metadata"]["version"] == 1
        assert history[1]["metadata"]["version"] == 2
        
        # Audit trail
        audit = mem.get_audit_trail()
        assert len(audit) >= 3
        assert audit[0]["operation"] == "write"
        assert audit[2]["operation"] == "version_read"
    finally:
        if os.path.exists(db_path):
            mem.memory.sqlite_store.close()
            os.remove(db_path)

def test_lattice_dedup():
    dedup = LatticeDedup(d_model=384)
    dedup.encoder = FakeEncoder(d_model=384)
    
    docs = [
        "First unique document",
        "First unique document",
        "Second unique document"
    ]
    
    result = dedup.deduplicate(docs)
    assert len(result["unique_documents"]) == 2
    assert result["duplicate_count"] == 1
    assert result["compression_ratio"] == 1.0 / 3.0
