import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from latticememory.memory import RFSnapLatticeMemory, MemoryDocument, MemoryQuery

class AgentEpisodicMemory:
    """
    AI Agent Episodic Memory using LatticeMemory.
    
    Provides agents with O(1) exact memory lookup, automatic semantic deduplication
    via E8 snapping, version history for overlapping memory addresses,
    and a read/write audit log to prevent hallucinations.
    """
    def __init__(
        self,
        d_model: int = 1024,
        sqlite_path: Optional[str | Path] = None,
        sentence_transformer_model: str = "dfrokido/bge-large-e8-snap"
    ):
        """
        Args:
            d_model: Hidden dimension of the embedding space.
            sqlite_path: Path to SQLite DB for durable persistence.
            sentence_transformer_model: Encoder model family name.
        """
        self.d_model = d_model
        self.sqlite_path = sqlite_path
        self.encoder_model_name = sentence_transformer_model
        
        # Initialize the core E8 LatticeMemory
        self.memory = RFSnapLatticeMemory(d_model=d_model, sqlite_path=sqlite_path)
        self.encoder = None
        self.audit_log: List[Dict[str, Any]] = []

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.encoder_model_name)
        return self.encoder

    def _encode_text(self, text: str) -> List[float]:
        """Robustly encode a single string to a list of float embedding values."""
        encoder = self._get_encoder()
        try:
            emb_res = encoder.encode([text])
            # Handle batch vs single returns
            if hasattr(emb_res, "shape") and len(emb_res.shape) > 1:
                val = emb_res[0]
            elif isinstance(emb_res, list) or isinstance(emb_res, np.ndarray):
                val = emb_res[0]
            else:
                val = emb_res
        except Exception:
            val = encoder.encode(text)
            
        # Convert to native python list of floats
        if hasattr(val, "tolist"):
            return val.tolist()
        elif isinstance(val, list):
            return val
        else:
            return list(val)

    def _log_audit(self, operation: str, address: str, content: str, result_ids: List[str]):
        """Record a memory access event to the audit trail."""
        self.audit_log.append({
            "timestamp": time.time(),
            "operation": operation,
            "address": address,
            "content": content,
            "result_ids": result_ids
        })

    def add_episode(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adds a new episode/memory to the agent's memory.
        If another episode already maps to the same E8 address, it is versioned.
        
        Args:
            content: The text representation of the episode/memory.
            metadata: Custom metadata to associate with the episode.
        Returns:
            Dict containing the added episode details (address, version, doc_id).
        """
        metadata = dict(metadata) if metadata is not None else {}
        
        # 1. Compute embedding using the encoder
        embedding = self._encode_text(content)
        
        # Ensure correct dimension
        if len(embedding) != self.d_model:
            # Pad or truncate if needed for mock/test vectors, otherwise raise error
            raise ValueError(f"Encoder dimension {len(embedding)} does not match d_model {self.d_model}")
            
        # 2. Compute the E8 key
        lat_key = self.memory.lattice_key_for(embedding)
        address = lat_key.hex()
        
        # 3. Check for existing episodes in the same E8 address cell
        existing_ids = self.memory.get_document_ids_by_lattice_key(lat_key)
        
        version = len(existing_ids) + 1
        doc_id = str(uuid.uuid4())
        
        # Enrich metadata with versioning info
        metadata["version"] = version
        metadata["timestamp"] = time.time()
        metadata["is_duplicate"] = len(existing_ids) > 0
        metadata["previous_versions"] = existing_ids
        
        # 4. Create and index the document
        doc = MemoryDocument(
            doc_id=doc_id,
            text=content,
            embedding=embedding,
            metadata=metadata
        )
        self.memory.add_documents([doc])
        
        # 5. Log write to audit trail
        self._log_audit("write", address, content, [doc_id])
        
        return {
            "doc_id": doc_id,
            "address": address,
            "version": version,
            "is_duplicate": len(existing_ids) > 0,
            "content": content,
            "metadata": metadata
        }

    def retrieve_episodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar episodes using E8 snapping and reranking.
        """
        query_emb = self._encode_text(query)
        
        # Run E8-accelerated retrieval
        mem_query = MemoryQuery(embedding=query_emb, top_k=limit, text=query)
        result = self.memory.retrieve(mem_query)
        
        # Extract E8 address of the query
        lat_key = self.memory.lattice_key_for(query_emb)
        query_address = lat_key.hex()
        
        hits = []
        result_ids = []
        for hit in result.hits:
            doc_id = hit.doc_id
            text = self.memory.get_document_text(doc_id)
            meta = self.memory.get_document_metadata(doc_id)
            
            # Find the actual E8 key stored for this doc
            doc_lat_key = self.memory.get_document_lattice_key(doc_id)
            doc_address = doc_lat_key.hex() if doc_lat_key is not None else ""
            
            hits.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": meta,
                "address": doc_address,
                "score": hit.score
            })
            result_ids.append(doc_id)
            
        # Log read to audit trail
        self._log_audit("read", query_address, query, result_ids)
        
        return hits

    def get_episode_history(self, content: str) -> List[Dict[str, Any]]:
        """
        Retrieves all versioned episodes stored in the exact same E8 cell
        as the provided text content.
        """
        emb = self._encode_text(content)
        
        lat_key = self.memory.lattice_key_for(emb)
        address = lat_key.hex()
        
        doc_ids = self.memory.get_document_ids_by_lattice_key(lat_key)
        
        history = []
        for doc_id in doc_ids:
            text = self.memory.get_document_text(doc_id)
            meta = self.memory.get_document_metadata(doc_id)
            history.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": meta,
                "address": address
            })
            
        # Sort by version ascending
        history.sort(key=lambda x: x["metadata"].get("version", 1))
        
        # Log audit trail
        self._log_audit("version_read", address, content, doc_ids)
        
        return history

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the complete audit trail of read/write operations."""
        return self.audit_log
