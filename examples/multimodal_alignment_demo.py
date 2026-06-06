"""Example script demonstrating Multimodal Snapping Alignment using LatticeMemory with a real CLIP model.

WARNING: This demo is a simplified demonstration using a small number of pairs.
In production with large-scale unseen image/text distributions, cross-modal alignment
accuracy will decrease due to E8 block-level coordinate mismatches (structural limits).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.dual_encoder import fit_lattice_dual_encoder, RFSnapDualTextMemory
from sentence_transformers import SentenceTransformer

class MultimodalCLIPWrapper:
    """Wrapper around CLIP SentenceTransformer to support both text and dynamically generated PIL images."""
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        outputs = []
        for item in sentences:
            if isinstance(item, str) and item.startswith("image:"):
                concept = item[len("image:"):]
                # Dynamically construct a unique image using PIL representing this concept
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), concept, fill=(255, 255, 255))
                
                # Encode the image
                emb = self.model.encode(img, batch_size=1)
                outputs.append(emb)
            elif isinstance(item, str) and item.startswith("text:"):
                concept = item[len("text:"):]
                emb = self.model.encode(concept, batch_size=1)
                outputs.append(emb)
            else:
                emb = self.model.encode(item, batch_size=1)
                outputs.append(emb)
        return np.vstack(outputs)


def run_demo():
    print("=========================================================================")
    print("      LatticeMemory Phase 2: Multimodal Snapping Alignment Demo          ")
    print("=========================================================================")
    print("WARNING: This demo is a simplified representation of alignment capabilities.")
    print("In real-world applications with high-entropy distributions, E8 matching drops.")
    print("-------------------------------------------------------------------------")
    
    # 1. Initialize the real cached sentence-transformers CLIP model (512-dimensional)
    d_model = 512
    print("Loading cached 'sentence-transformers/clip-ViT-B-32' model...")
    base_clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    base_encoder = MultimodalCLIPWrapper(base_clip)
    
    # 2. Define multimodal pairs: (image_query, text_doc)
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
        query_encoder=base_encoder,  # Using base CLIP for both query and doc
        d_model=d_model,
    )
    
    unaligned_matches = 0
    for img_query, txt_doc in multimodal_pairs:
        img_key = unaligned_runtime.memory.lattice_key_for(unaligned_runtime._encode_queries([img_query])[0]).hex()
        txt_key = unaligned_runtime.memory.lattice_key_for(unaligned_runtime._encode_documents([txt_doc])[0]).hex()
        matched = (img_key == txt_key)
        if matched:
            unaligned_matches += 1
        print(f"Concept: '{txt_doc[5:]}'")
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
        query_encoder=dual.query_encoder,  # Using trained query adapter
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
        print(f"Concept: '{txt_doc[5:]}'")
        print(f"  -> Image E8 key: {img_key[:16]}...")
        print(f"  -> Text E8 key:  {txt_key[:16]}...")
        print(f"  -> Exact key match? {matched}")
            
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
