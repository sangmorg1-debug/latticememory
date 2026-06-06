from typing import List, Dict, Any, Tuple
from collections import defaultdict

from latticememory.memory import RFSnapLatticeMemory

class LatticeDedup:
    """
    LatticeDedup: O(N) Semantic Deduplication.
    
    Groups and deduplicates collections of documents by mapping their embeddings
    to discrete E8 lattice addresses. Documents that map to the same E8 address
    are clustered as semantic duplicates, eliminating expensive O(N^2) pairwise comparisons.
    """
    def __init__(
        self,
        d_model: int = 1024,
        sentence_transformer_model: str = "dfrokido/bge-large-e8-snap"
    ):
        """
        Args:
            d_model: Hidden dimension of the embedding space.
            sentence_transformer_model: Encoder model family name.
        """
        self.d_model = d_model
        self.encoder_model_name = sentence_transformer_model
        self.encoder = None

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.encoder_model_name)
        return self.encoder

    def deduplicate(self, documents: List[str]) -> Dict[str, Any]:
        """
        Deduplicates a corpus of texts by snapping their embeddings to E8 cells.
        
        Args:
            documents: A list of text documents to deduplicate.
        Returns:
            Dict containing:
              - 'unique_documents': list of unique (canonical) documents.
              - 'duplicates': dict mapping E8 address hex strings to lists of duplicate documents.
              - 'compression_ratio': fraction of documents removed.
        """
        if not documents:
            return {
                "unique_documents": [],
                "duplicates": {},
                "compression_ratio": 0.0
            }
            
        encoder = self._get_encoder()
        
        # We temporarily spin up an in-memory LatticeMemory instance to use its quantizer
        lattice_mem = RFSnapLatticeMemory(d_model=self.d_model)
        
        # Group documents by E8 address key
        grouped_docs = defaultdict(list)
        
        # 1. Compute embeddings in batch for speed
        embeddings = encoder.encode(documents).tolist()
        
        # 2. Map each doc to its E8 address
        for doc_text, emb in zip(documents, embeddings):
            lat_key = lattice_mem.lattice_key_for(emb)
            address = lat_key.hex()
            grouped_docs[address].append(doc_text)
            
        # 3. Construct unique list and duplicate clusters
        unique_documents = []
        duplicates = {}
        
        for address, doc_list in grouped_docs.items():
            # The first document added to the E8 cell is treated as the canonical version
            unique_documents.append(doc_list[0])
            if len(doc_list) > 1:
                # Store the other duplicates
                duplicates[address] = doc_list[1:]
                
        total_docs = len(documents)
        unique_count = len(unique_documents)
        removed_count = total_docs - unique_count
        compression_ratio = removed_count / total_docs if total_docs > 0 else 0.0
        
        return {
            "unique_documents": unique_documents,
            "duplicates": duplicates,
            "compression_ratio": compression_ratio,
            "duplicate_count": removed_count
        }
