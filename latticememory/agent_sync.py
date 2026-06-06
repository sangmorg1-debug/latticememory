from __future__ import annotations

import hashlib
from typing import Any, Iterable

import torch

from latticememory.text_runtime import RFSnapTextMemory
from latticememory.memory import MemoryDocument


class AgentMemorySync:
    """Multi-Agent Shared Semantic Memory synchronization layer using E8 lattice keys.

    Allows agent swarms (e.g., in AutoGen, CrewAI, or LangGraph) to sync their memories
    efficiently by comparing O(1) E8 keys rather than transferring large embedding vectors
    or raw texts. Agents only pull full documents when they identify a missing key.
    """

    def __init__(self, runtime: RFSnapTextMemory):
        """Initialize the agent sync instance.

        Args:
            runtime: The local RFSnapTextMemory runtime of the agent.
        """
        self.runtime = runtime
        self._peers: list[AgentMemorySync] = []

    def register_peer(self, peer: AgentMemorySync) -> None:
        """Register a peer agent for bi-directional key broadcasting.

        Args:
            peer: Another AgentMemorySync instance.
        """
        if peer is self:
            return
        if peer not in self._peers:
            self._peers.append(peer)
            # Ensure bi-directional registration
            if self not in peer._peers:
                peer.register_peer(self)

    def get_known_keys(self) -> set[bytes]:
        """Return the set of E8 lattice keys (in bytes) currently stored in memory."""
        return set(self.runtime.memory.lattice.hash_store.keys())

    def get_document_by_id(self, doc_id: str) -> MemoryDocument | None:
        """Retrieve a MemoryDocument by its document ID.

        Args:
            doc_id: The unique document identifier.
        """
        text = self.runtime.memory.get_document_text(doc_id)
        if text is None:
            return None
        emb = self.runtime.memory.lattice._embeddings.get(doc_id)
        if emb is None:
            return None
        meta = self.runtime.memory.get_document_metadata(doc_id) or {}
        return MemoryDocument(doc_id=doc_id, text=text, embedding=emb, metadata=meta)

    def get_documents_for_key(self, key: bytes) -> list[MemoryDocument]:
        """Retrieve all MemoryDocuments associated with the given E8 lattice key.

        Args:
            key: The E8 lattice key in bytes.
        """
        doc_ids = self.runtime.memory.get_document_ids_by_lattice_key(key)
        docs = []
        for doc_id in doc_ids:
            doc = self.get_document_by_id(doc_id)
            if doc is not None:
                docs.append(doc)
        return docs

    def diff(self, peer_keys: set[bytes]) -> dict[str, set[bytes]]:
        """Compare this agent's keys with a peer's keys to find missing/extra keys.

        Args:
            peer_keys: Set of E8 keys (bytes) from the peer.

        Returns:
            A dict with:
                - "missing": Keys the peer has but we are missing.
                - "extra": Keys we have but the peer is missing.
        """
        our_keys = self.get_known_keys()
        return {
            "missing": peer_keys - our_keys,
            "extra": our_keys - peer_keys,
        }

    def share(self, key: bytes) -> None:
        """Broadcast an E8 key to all registered peer agents.

        If a peer is missing the key, they will automatically request the documents.

        Args:
            key: The E8 key in bytes.
        """
        for peer in list(self._peers):
            peer.handle_shared_key(self, key)

    def handle_shared_key(self, sender: AgentMemorySync, key: bytes) -> None:
        """Callback triggered when a peer broadcasts a key.

        Args:
            sender: The peer agent broadcasting the key.
            key: The E8 key in bytes.
        """
        our_keys = self.get_known_keys()
        if key not in our_keys:
            docs = sender.get_documents_for_key(key)
            for doc in docs:
                self.receive_document(doc)

    def request(self, key: bytes, peer: AgentMemorySync | None = None) -> list[MemoryDocument]:
        """Request all documents matching a key from a peer (or all peers if None).

        Args:
            key: The E8 key in bytes.
            peer: The specific peer to query, or None to query all.
        """
        if peer is not None:
            return peer.get_documents_for_key(key)

        docs = []
        seen_ids = set()
        for p in self._peers:
            for doc in p.get_documents_for_key(key):
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    docs.append(doc)
        return docs

    def receive_document(self, doc: MemoryDocument) -> None:
        """Add a received document to the local memory index.

        Args:
            doc: The MemoryDocument instance.
        """
        self.runtime.memory.add_documents([doc])

    def sync_from_peer(self, peer: AgentMemorySync) -> None:
        """Synchronize missing documents from a peer.

        Args:
            peer: The peer AgentMemorySync instance.
        """
        peer_keys = peer.get_known_keys()
        diffs = self.diff(peer_keys)
        for key in diffs["missing"]:
            docs = peer.get_documents_for_key(key)
            for doc in docs:
                self.receive_document(doc)


# ----------------------------------------------------------------------
# Orchestration Adapters
# ----------------------------------------------------------------------

def make_autogen_sync_tools(sync: AgentMemorySync) -> dict[str, Any]:
    """Exposes the AgentMemorySync functions as AutoGen-compatible tool definitions.

    Args:
        sync: The local AgentMemorySync instance.

    Returns:
        A dict mapping tool name to tool function.
    """
    def get_keys() -> list[str]:
        """Get all known E8 keys from the agent's memory as hex strings."""
        return [k.hex() for k in sync.get_known_keys()]

    def request_key_docs(key_hex: str) -> list[dict[str, Any]]:
        """Request full documents from the swarm for a given E8 key hex string."""
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError:
            return []
        docs = sync.request(key_bytes)
        return [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    def share_key(key_hex: str) -> str:
        """Broadcast an E8 key hex string to the peer swarm."""
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError:
            return "Error: Invalid hex key format."
        sync.share(key_bytes)
        return f"Successfully shared key {key_hex} with peers."

    return {
        "get_keys": get_keys,
        "request_key_docs": request_key_docs,
        "share_key": share_key,
    }


class LangGraphLatticeAdapter:
    """Orchestration adapter to integrate AgentMemorySync with LangGraph workflow states."""

    def __init__(self, sync: AgentMemorySync):
        """Initialize the LangGraph adapter.

        Args:
            sync: The local AgentMemorySync instance.
        """
        self.sync = sync

    def sync_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """A LangGraph node function that processes keys/texts to sync memory state.

        Reads:
            - state["shared_keys"]: list of E8 hex keys to retrieve and index.
            - state["input_texts"]: list of raw texts to embed and register locally.

        Returns:
            Updates to merge into the graph state:
                - "indexed_doc_ids": list of new document IDs written to memory.
                - "generated_keys": list of new E8 hex keys registered from input_texts.
        """
        new_docs_indexed = []
        shared_keys = state.get("shared_keys", [])
        for key_hex in shared_keys:
            try:
                key_bytes = bytes.fromhex(key_hex)
            except ValueError:
                continue
            
            docs = self.sync.request(key_bytes)
            for doc in docs:
                if doc.doc_id not in self.sync.runtime.memory._doc_id_set:
                    self.sync.receive_document(doc)
                    new_docs_indexed.append(doc.doc_id)

        added_keys = []
        input_texts = state.get("input_texts", [])
        for text in input_texts:
            embedding = self.sync.runtime._encode_texts([text])[0]
            doc_id = f"lg-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
            doc = MemoryDocument(doc_id=doc_id, text=text, embedding=embedding)
            self.sync.receive_document(doc)
            key = self.sync.runtime.memory.get_document_lattice_key(doc_id)
            if key:
                added_keys.append(key.hex())

        return {
            "indexed_doc_ids": new_docs_indexed,
            "generated_keys": added_keys,
        }
