from typing import Any, List, Optional
import torch

try:
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode, BaseNode
except ImportError as e:
    raise ImportError(
        "llama-index-core is required for LatticeVectorStore. "
        "Install it with: pip install llama-index-core"
    ) from e

from latticememory.memory import RFSnapLatticeMemory, MemoryDocument, MemoryQuery


class LatticeVectorStore(BasePydanticVectorStore):
    """LlamaIndex Vector Store implementation backed by LatticeMemory E8 lattice.

    Enables O(1) semantic lookups, low memory footprint, and optional SQLite persistence.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    _memory: RFSnapLatticeMemory

    def __init__(
        self,
        memory: Optional[RFSnapLatticeMemory] = None,
        d_model: int = 1024,
        sqlite_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if memory is not None:
            self._memory = memory
        else:
            self._memory = RFSnapLatticeMemory(d_model=d_model, sqlite_path=sqlite_path)

    @property
    def client(self) -> Any:
        return self._memory

    def add(self, nodes: List[BaseNode]) -> List[str]:
        docs = []
        for node in nodes:
            emb = node.get_embedding()
            if emb is None:
                raise ValueError(f"Node {node.node_id} is missing an embedding.")
            docs.append(
                MemoryDocument(
                    doc_id=node.node_id,
                    text=node.get_content(),
                    embedding=emb,
                    metadata=node.metadata or {},
                )
            )
        self._memory.add_documents(docs)
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document by ref_doc_id."""
        if ref_doc_id in self._memory._doc_id_set:
            self._memory._doc_id_set.remove(ref_doc_id)
            self._memory._doc_ids = [d for d in self._memory._doc_ids if d != ref_doc_id]
            if ref_doc_id in self._memory._texts:
                del self._memory._texts[ref_doc_id]
            if ref_doc_id in self._memory._metadata:
                del self._memory._metadata[ref_doc_id]
            if ref_doc_id in self._memory._doc_index:
                del self._memory._doc_index[ref_doc_id]
            # Rebuild index mapping
            self._memory._doc_index = {doc_id: i for i, doc_id in enumerate(self._memory._doc_ids)}
            
            # Delete from internal E8LatticeDB
            if ref_doc_id in self._memory.lattice._embeddings:
                del self._memory.lattice._embeddings[ref_doc_id]
            if ref_doc_id in self._memory.lattice._metadata:
                del self._memory.lattice._metadata[ref_doc_id]
            if ref_doc_id in self._memory.lattice._keys:
                old_key = self._memory.lattice._keys[ref_doc_id]
                del self._memory.lattice._keys[ref_doc_id]
                self._memory.lattice.hash_store[old_key] = [
                    d for d in self._memory.lattice.hash_store[old_key] if d != ref_doc_id
                ]
                
            # Re-stack ANN embeddings
            ann_embeddings = [self._memory.lattice._embeddings[d].detach().cpu() for d in self._memory._doc_ids]
            self._memory._ann_embeddings = torch.stack(ann_embeddings) if ann_embeddings else None
            
            # Sync to SQLite if configured
            if self._memory.sqlite_store is not None:
                self._memory.sqlite_store.delete_documents([ref_doc_id])

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("Query is missing query_embedding.")
        
        mem_query = MemoryQuery(
            text="",
            embedding=query.query_embedding,
            top_k=query.similarity_top_k,
        )
        res = self._memory.retrieve(mem_query)
        
        nodes = []
        similarities = []
        ids = []
        
        for hit in res.hits:
            node = TextNode(
                id_=hit.doc_id,
                text=hit.text,
                metadata=hit.metadata,
            )
            nodes.append(node)
            similarities.append(hit.score)
            ids.append(hit.doc_id)
            
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
