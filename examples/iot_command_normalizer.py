"""Example script demonstrating IoT command normalization using LatticeMemory.

Trains a query-side contrastive adapter to snap variable user commands to
exact E8 addresses representing canonical smart home actions.
"""
from __future__ import annotations

import os
import sys

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.dual_encoder import train_lattice_contrastive_encoder, RFSnapDualTextMemory
from tests.test_lattice_index import FakeEncoder

def run_demo():
    print("--- Phase 1b: IoT Smart Home Semantic Cache Demo ---")
    
    # 1. Create a deterministic fake encoder (384-dimensional)
    d_model = 384
    base_encoder = FakeEncoder(d_model=d_model)
    
    # 2. Define training pairs: (user_variation, canonical_doc_representation)
    training_pairs = [
        ("make the kitchen brighter", "command:turn_on_kitchen_lights"),
        ("light up the cooking area", "command:turn_on_kitchen_lights"),
        ("switch on kitchen lamps", "command:turn_on_kitchen_lights"),
        
        ("turn off the kitchen lights", "command:turn_off_kitchen_lights"),
        ("kill the kitchen lamps", "command:turn_off_kitchen_lights"),
        ("dim the cooking area fully", "command:turn_off_kitchen_lights"),
        
        ("set living room temperature to 72", "command:set_temp_72"),
        ("cool down the living room to 72", "command:set_temp_72"),
        ("thermostat living room 72", "command:set_temp_72"),
    ]
    
    # 3. Train the query-side contrastive adapter
    print(f"Training query adapter on {len(training_pairs)} command variations...")
    result = train_lattice_contrastive_encoder(
        base_encoder=base_encoder,
        pairs=training_pairs,
        d_model=d_model,
        epochs=15,
        lr=0.01,
        temperature=0.05,
        lambda_address=1.0,
    )
    
    print(f"Training complete. Final E8 Key Match Accuracy: {result.final_train_accuracy * 100:.1f}%")
    
    # 4. Instantiate the dual encoder memory runtime
    runtime = RFSnapDualTextMemory(
        document_encoder=result.dual_encoder.document_encoder,
        query_encoder=result.dual_encoder.query_encoder,
        d_model=d_model,
    )
    
    # 5. Index the canonical target documents
    canonical_documents = [
        "command:turn_on_kitchen_lights",
        "command:turn_off_kitchen_lights",
        "command:set_temp_72",
    ]
    doc_ids = ["cmd-on-lights", "cmd-off-lights", "cmd-temp"]
    runtime.add_texts(canonical_documents, doc_ids=doc_ids)
    
    # 6. Test retrieval and snapping
    test_queries = [
        "light up the cooking area",      # exact training sample
        "kill the kitchen lamps",         # exact training sample
        "thermostat living room 72",      # exact training sample
    ]
    
    print("\nRetrieving commands on local E8 index:")
    for query in test_queries:
        res = runtime.retrieve_text(query, top_k=1)
        hit = res.hits[0] if res.hits else None
        print(f"Query: '{query}'")
        if hit:
            print(f"  -> Match: '{hit.text}' (ID: {hit.doc_id})")
            print(f"  -> Path: {res.path} (Exact O(1) hash hit? {res.path == 'lattice_exact'})")
            print(f"  -> Cosine score: {hit.score:.4f}")
        else:
            print("  -> Miss")
        print()

if __name__ == "__main__":
    run_demo()
