from __future__ import annotations

import json
import sqlite3
import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Generator

from latticememory.memory import MemoryDocument


class LatticeSqliteStore:
    """SQLite-backed persistent document store for LatticeMemory.

    Stores document ID, text, metadata, raw embeddings (as float32 BLOB), and
    quantized E8 lattice keys. Enables persistent indices without companion files.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id      TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    embedding   BLOB NOT NULL,
                    metadata    TEXT NOT NULL,
                    lattice_key BLOB NOT NULL
                )
                """
            )
            # Index for fast key-based retrieval if needed
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_lattice_key ON documents (lattice_key)"
            )

    def add_documents(
        self,
        documents: Iterable[MemoryDocument],
        lattice_keys: dict[str, bytes],
    ) -> None:
        """Upsert a batch of documents and their corresponding E8 lattice keys."""
        rows = []
        for doc in documents:
            vector = torch.as_tensor(doc.embedding, dtype=torch.float32).cpu().numpy()
            emb_blob = vector.tobytes()
            meta_json = json.dumps(doc.metadata)
            lat_key = lattice_keys.get(doc.doc_id)
            if lat_key is None:
                raise ValueError(f"Missing E8 lattice key for document {doc.doc_id!r}")
            rows.append((doc.doc_id, doc.text, emb_blob, meta_json, lat_key))

        if not rows:
            return

        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO documents (doc_id, text, embedding, metadata, lattice_key)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_documents(self, doc_ids: Iterable[str]) -> None:
        """Delete a batch of documents by ID."""
        rows = [(doc_id,) for doc_id in doc_ids]
        if not rows:
            return
        with self._conn:
            self._conn.executemany("DELETE FROM documents WHERE doc_id = ?", rows)

    def get_document(self, doc_id: str) -> MemoryDocument | None:
        """Retrieve a single document by ID."""
        cursor = self._conn.execute(
            "SELECT text, embedding, metadata FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_doc(doc_id, row)

    def get_all_documents(self) -> Generator[MemoryDocument, None, None]:
        """Yield all documents stored in the database."""
        cursor = self._conn.execute("SELECT doc_id, text, embedding, metadata FROM documents")
        for row in cursor:
            yield self._row_to_doc(row["doc_id"], row)

    def get_all_rows(self) -> list[sqlite3.Row]:
        """Get all raw rows in the database (for index reconstruction)."""
        cursor = self._conn.execute("SELECT doc_id, text, embedding, metadata, lattice_key FROM documents")
        return cursor.fetchall()

    def count(self) -> int:
        """Return total number of documents stored."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the SQLite database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_doc(doc_id: str, row: sqlite3.Row | dict) -> MemoryDocument:
        emb_bytes = row["embedding"]
        vector = torch.from_numpy(np.frombuffer(emb_bytes, dtype=np.float32).copy())
        metadata = json.loads(row["metadata"])
        return MemoryDocument(
            doc_id=doc_id,
            text=row["text"],
            embedding=vector,
            metadata=metadata,
        )
