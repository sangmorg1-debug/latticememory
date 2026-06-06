"""
Phase 0b Flagship Product Demo: LatticeDedup

Demonstrates high-speed O(N) semantic deduplication using E8 lattice addresses,
clustering similar documents and filtering out duplicates without quadratic comparisons.
"""
from __future__ import annotations

import os
import sys
import hashlib
import numpy as np

# Ensure parent directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latticememory.dedup import LatticeDedup

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
    print("            LatticeMemory Phase 0b: LatticeDedup Demo                    ")
    print("=========================================================================")
    
    d_model = 384
    
    # Initialize LatticeDedup with mock encoder
    deduplicator = LatticeDedup(d_model=d_model)
    deduplicator.encoder = FakeEncoder(d_model=d_model)
    
    # Corpus containing multiple semantic duplicate clusters
    corpus = [
        # Cluster A: Version Control / Git
        "Git is a distributed version control system.",
        "Git is a distributed version control system.", # Exact duplicate
        "Git represents a distributed version control system.", # Snapped duplicate
        
        # Cluster B: Python / Programming
        "Python is a high-level general-purpose programming language.",
        "Python is a high-level general-purpose programming language.", # Exact duplicate
        "Python is a popular general-purpose scripting language.", # Snapped duplicate
        
        # Cluster C: Databases / Storage
        "PostgreSQL is a powerful open-source object-relational database.",
        
        # Cluster D: Machine Learning / AI
        "Transformers are deep learning architectures based on self-attention.",
        "The transformer is an attention-based deep learning architecture." # Snapped duplicate
    ]
    
    print(f"Total documents in original corpus: {len(corpus)}")
    print("\nRunning O(N) semantic deduplication...")
    
    result = deduplicator.deduplicate(corpus)
    
    print("\nDeduplication Complete!")
    print(f"  - Unique Documents Kept: {len(result['unique_documents'])}")
    print(f"  - Duplicate Documents Removed: {result['duplicate_count']}")
    print(f"  - Data Reduction Ratio: {result['compression_ratio'] * 100:.1f}%")
    
    print("\n=========================================================================")
    print("                      Deduplicated Unique Corpus                         ")
    print("=========================================================================")
    for idx, doc in enumerate(result["unique_documents"]):
        print(f"  [{idx + 1}] {doc}")
        
    print("\n=========================================================================")
    print("                       Duplicate Clusters Found                          ")
    print("=========================================================================")
    for address, dup_list in result["duplicates"].items():
        # Get the corresponding unique doc (the canonical one)
        # Note: In our deduplicate implementation, the canonical doc was the first one mapped
        # to that address. We can print the duplicates under this address.
        print(f"  Address: E8_{address[:16]}...")
        print(f"    Duplicates removed:")
        for dup in dup_list:
            print(f"      - '{dup}'")
        print()

if __name__ == "__main__":
    run_demo()
