"""
Phase 0b Flagship Product Demo: AI Agent Episodic Memory

Demonstrates exact E8 address matching for memory versioning, semantic deduplication,
and the memory audit log for agents.
"""
from __future__ import annotations

import os
import sys
import hashlib
import numpy as np

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.agent_memory import AgentEpisodicMemory

class FakeEncoder:
    """Deterministic mock encoder that hashes text to a unit vector."""
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, **kwargs):
        single = isinstance(sentences, str)
        s_list = [sentences] if single else sentences
        
        result = []
        for s in s_list:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            result.append(v)
            
        return result[0] if single else np.stack(result)


def run_demo():
    print("=========================================================================")
    print("      LatticeMemory Phase 0b: AI Agent Episodic Memory Demo              ")
    print("=========================================================================")
    
    d_model = 384
    
    # 1. Initialize memory with SQLite durability
    db_file = "agent_memory_demo.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    print(f"Initializing AgentEpisodicMemory with SQLite database: {db_file}...")
    agent_mem = AgentEpisodicMemory(d_model=d_model, sqlite_path=db_file)
    
    # Inject our fast mock encoder
    agent_mem.encoder = FakeEncoder(d_model=d_model)
    
    # 2. Add sequential agent episodes
    episodes = [
        "Observation: User likes dark roast coffee at 8 AM.",
        "Observation: User likes dark roast coffee at 8 AM.", # Exact duplicate
        "Observation: User prefers espresso at 8 AM.",       # Semantic overlap (snaps to same E8 cell)
        "Observation: User dislikes cold brew coffee.",       # Distinct semantic point
    ]
    
    print("\n[Agent] Recording episodes...")
    for ep in episodes:
        res = agent_mem.add_episode(ep, metadata={"source": "sensor_observation"})
        print(f"Recorded: '{res['content']}'")
        print(f"  -> Address: E8_{res['address'][:16]}...")
        print(f"  -> Version: {res['version']} (Duplicate? {res['is_duplicate']})")
        print()
        
    # 3. Retrieve version history for the coffee memory cell
    target_content = "Observation: User likes dark roast coffee at 8 AM."
    print("=========================================================================")
    print(f"Retrieving Version History for E8 Address of:")
    print(f"  '{target_content}'")
    print("=========================================================================")
    history = agent_mem.get_episode_history(target_content)
    for h in history:
        print(f"  [Version {h['metadata']['version']}]")
        print(f"    Content: '{h['text']}'")
        print(f"    Doc ID:  {h['doc_id']}")
        print(f"    Timestamp: {h['metadata']['timestamp']}")
        print()
        
    # 4. Search agent memory using semantic query
    search_query = "morning coffee preference"
    print("=========================================================================")
    print(f"Semantic Search Query: '{search_query}'")
    print("=========================================================================")
    hits = agent_mem.retrieve_episodes(search_query, limit=2)
    for idx, hit in enumerate(hits):
        print(f"  Hit {idx + 1} (Score: {hit['score']:.4f}):")
        print(f"    Content: '{hit['text']}'")
        print(f"    Address: E8_{hit['address'][:16]}...")
        print(f"    Version: {hit['metadata']['version']}")
        print()
        
    # 5. Review memory audit trail (crucial for agent hallucination prevention)
    print("=========================================================================")
    print("                Memory Access Audit Trail (Telemetry)                    ")
    print("=========================================================================")
    trail = agent_mem.get_audit_trail()
    for entry in trail:
        print(f"  [{entry['operation'].upper()}] Address: E8_{entry['address'][:16]}...")
        print(f"    Content:   '{entry['content']}'")
        print(f"    Result IDs: {entry['result_ids']}")
        print()
        
    # Clean up DB file
    if os.path.exists(db_file):
        agent_mem.memory.sqlite_store.close()
        os.remove(db_file)

if __name__ == "__main__":
    run_demo()
