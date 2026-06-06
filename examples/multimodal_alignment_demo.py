"""Example script demonstrating Multimodal Snapping Alignment using LatticeMemory.

CLIP-style models align text and images into the same neighborhood, but not
the exact same coordinate. This script shows how to train a query-side adapter
to project image embeddings onto the exact E8 addresses of their corresponding
text representations.
"""
from __future__ import annotations

import os
import sys

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.dual_encoder import fit_lattice_dual_encoder, RFSnapDualTextMemory
from tests.test_lattice_index import FakeEncoder

def run_demo():
    print("--- Phase 2: Multimodal Snapping Alignment Demo ---")
    
    # 1. Initialize a deterministic mock encoder (384-dimensional)
    d_model = 384
    base_encoder = FakeEncoder(d_model=d_model)
    
    # 2. Define unaligned multimodal pairs: (image_query, text_doc)
    multimodal_pairs = [
        ("image:a fluffy cat sleeping", "text:a fluffy cat sleeping"),
        ("image:a red sports car on highway", "text:a red sports car on highway"),
        ("image:a golden retriever catching a frisbee", "text:a golden retriever catching a frisbee"),
        ("image:a steaming mug of coffee", "text:a steaming mug of coffee"),
    ]
    
    # 3. Before alignment: check if they snap to the same E8 key
    print("\nChecking snapping BEFORE query alignment:")
    unaligned_runtime = RFSnapDualTextMemory(
        document_encoder=base_encoder,
        query_encoder=base_encoder, # Using base_encoder for query (unaligned)
        d_model=d_model,
    )
    
    unaligned_matches = 0
    for img_query, txt_doc in multimodal_pairs:
        img_key = unaligned_runtime.memory.lattice_key_for(unaligned_runtime._encode_queries([img_query])[0]).hex()
        txt_key = unaligned_runtime.memory.lattice_key_for(unaligned_runtime._encode_documents([txt_doc])[0]).hex()
        matched = (img_key == txt_key)
        if matched:
            unaligned_matches += 1
        print(f"Page: '{txt_doc[5:]}'")
        print(f"  -> Image E8 key: {img_key[:16]}...")
        print(f"  -> Text E8 key:  {txt_key[:16]}...")
        print(f"  -> Exact key match? {matched}")
        
    print(f"Unaligned E8 Key Match Rate: {unaligned_matches / len(multimodal_pairs) * 100:.1f}%")

    # 4. Train the query-side linear adapter to map images onto text spaces
    print(f"\nTraining query adapter on {len(multimodal_pairs)} multimodal pairs...")
    dual = fit_lattice_dual_encoder(
        base_encoder=base_encoder,
        pairs=multimodal_pairs,
        d_model=d_model,
        ridge=1e-4,
    )
    
    # 5. Instantiate the aligned dual encoder memory runtime
    aligned_runtime = RFSnapDualTextMemory(
        document_encoder=dual.document_encoder,
        query_encoder=dual.query_encoder, # Using trained query adapter
        d_model=d_model,
    )
    
    # 6. Index the text documents
    texts = [txt for _img, txt in multimodal_pairs]
    doc_ids = [f"txt-{i}" for i in range(len(texts))]
    aligned_runtime.add_texts(texts, doc_ids=doc_ids)
    
    # 7. Check snapping AFTER query alignment
    print("\nChecking snapping AFTER query alignment:")
    aligned_matches = 0
    for img_query, txt_doc in multimodal_pairs:
        img_key = aligned_runtime.memory.lattice_key_for(aligned_runtime._encode_queries([img_query])[0]).hex()
        txt_key = aligned_runtime.memory.lattice_key_for(aligned_runtime._encode_documents([txt_doc])[0]).hex()
        matched = (img_key == txt_key)
        if matched:
            aligned_matches += 1
            
    print(f"Aligned E8 Key Match Rate: {aligned_matches / len(multimodal_pairs) * 100:.1f}%")
    
    # 8. Test O(1) cross-modal retrieval
    print("\nRetrieving text documents using image queries:")
    for img_query, _txt_doc in multimodal_pairs:
        res = aligned_runtime.retrieve_text(img_query, top_k=1)
        hit = res.hits[0] if res.hits else None
        print(f"Query Image: '{img_query}'")
        if hit:
            print(f"  -> Retrieved Text: '{hit.text}' (ID: {hit.doc_id})")
            print(f"  -> Path: {res.path} (Exact O(1) cross-modal hit? {res.path == 'lattice_exact'})")
        else:
            print("  -> Miss")
        print()

if __name__ == "__main__":
    run_demo()
