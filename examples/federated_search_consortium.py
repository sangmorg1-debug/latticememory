"""Example script demonstrating Federated Semantic Search using LatticeMemory.

Simulates two independent nodes (Hospital A and Hospital B) querying each other's
databases securely by transmitting only E8 lattice address keys.
"""
from __future__ import annotations

import os
import sys

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.index import LatticeIndex
from tests.test_lattice_index import FakeEncoder

def run_demo():
    print("--- Phase 2: Federated Semantic Search Consortium Demo ---")
    
    # Both nodes must use the same underlying model family (d_model = 384)
    d_model = 384
    encoder_a = FakeEncoder(d_model=d_model)
    encoder_b = FakeEncoder(d_model=d_model)
    
    # 1. Initialize Hospital A's private local index
    index_a = LatticeIndex.__new__(LatticeIndex)
    index_a._init_with_encoder(encoder_a, d_model=d_model)
    
    hospital_a_records = [
        "Patient diagnosed with Type-2 Diabetes Mellitus and hyperglycemia.",
        "Clinical observation: acute bronchitis, severe cough, breathing difficulty.",
        "Patient recovering from cardiovascular surgery, blood pressure stable.",
    ]
    doc_ids_a = ["pat-001", "pat-002", "pat-003"]
    index_a.add(hospital_a_records, doc_ids=doc_ids_a)
    print(f"[Hospital A] Private index populated with {index_a.stats().docs} records.")
    
    # 2. Initialize Hospital B's private local index
    index_b = LatticeIndex.__new__(LatticeIndex)
    index_b._init_with_encoder(encoder_b, d_model=d_model)
    
    hospital_b_records = [
        "Diagnosis: insulin-dependent Type-1 Diabetes.",
        "Patient under observation for chronic asthma and wheezing.",
    ]
    doc_ids_b = ["pat-101", "pat-102"]
    index_b.add(hospital_b_records, doc_ids=doc_ids_b)
    print(f"[Hospital B] Private index populated with {index_b.stats().docs} records.")

    # 3. Simulation: Hospital B wants to query Hospital A for diabetes patients
    query_text = "Patient diagnosed with Type-2 Diabetes Mellitus and hyperglycemia."
    print(f"\n[Hospital B] Local search query: '{query_text}'")
    
    # Hospital B snaps the query locally to get the E8 key
    query_key = index_b.snap(query_text)
    print(f"[Hospital B] Locally snapped E8 key: {query_key[:16]}... (Length: {len(query_key)} hex chars)")
    
    print("\n[Network] Hospital B transmits ONLY the E8 key to Hospital A...")
    
    # 4. Hospital A receives the key and runs an O(1) hash search locally
    # It does not receive the raw query text or query embedding
    print("[Hospital A] Received E8 key. Resolving exact match in E8 hash store...")
    
    # Query key is hex, E8LatticeDB stores keys as bytes. Convert back to bytes:
    query_key_bytes = bytes.fromhex(query_key)
    matched_doc_ids = list(index_a._runtime.memory.lattice.hash_store.get(query_key_bytes, []))
    
    print(f"[Hospital A] O(1) Search resolved. Found matching Document IDs: {matched_doc_ids}")
    
    if matched_doc_ids:
        print("[Hospital A] Returning records matching the E8 address:")
        for doc_id in matched_doc_ids:
            doc_text = index_a._runtime.memory._text_for(doc_id)
            print(f"  -> ID: {doc_id} | Record: '{doc_text}'")
    else:
        print("[Hospital A] No matches found.")
        
    print("\nFederated search completed. Zero raw text or float vectors were shared over the network.")

if __name__ == "__main__":
    run_demo()
