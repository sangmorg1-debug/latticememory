from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from latticememory.memory import MemoryDocument, RFSnapLatticeMemory
from latticememory.sqlite_store import LatticeSqliteStore

from benchmarks.common import generate_unit_embeddings, write_json_result


def run_benchmark(
    *,
    n_docs: int = 100_000,
    d_model: int = 1024,
    output_path: str | Path | None = None,
) -> dict:
    if n_docs < 1:
        raise ValueError("n_docs must be positive")
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")

    embeddings = generate_unit_embeddings(n_docs, d_model)
    memory = RFSnapLatticeMemory(d_model=d_model)
    docs = []
    lattice_keys = {}
    for idx, embedding in enumerate(embeddings):
        doc_id = f"doc-{idx}"
        key = memory.lattice_key_for(embedding)
        docs.append(MemoryDocument(doc_id=doc_id, text=f"synthetic document {idx}", embedding=embedding, metadata={}))
        lattice_keys[doc_id] = key

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "compression.db"
        store = LatticeSqliteStore(db_path)
        store.add_documents(docs, lattice_keys)
        sqlite_document_count = store.count()
        store.close()

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT SUM(length(embedding)), SUM(length(lattice_key)) FROM documents"
            ).fetchone()
        finally:
            conn.close()
        db_file_size_bytes = db_path.stat().st_size

    float32_embedding_bytes = int(row[0] or 0)
    stored_lattice_key_bytes = int(row[1] or 0)
    theoretical_lattice_payload_bytes = int(n_docs * (d_model // 8) * 3)
    result = {
        "benchmark": "compression",
        "n_docs": int(n_docs),
        "d_model": int(d_model),
        "blocks": int(d_model // 8),
        "sqlite_document_count": int(sqlite_document_count),
        "float32_embedding_bytes": float32_embedding_bytes,
        "stored_lattice_key_bytes": stored_lattice_key_bytes,
        "theoretical_lattice_payload_bytes": theoretical_lattice_payload_bytes,
        "stored_key_compression_vs_float32": round(float32_embedding_bytes / stored_lattice_key_bytes, 1),
        "theoretical_compression_vs_float32": round(float32_embedding_bytes / theoretical_lattice_payload_bytes, 1),
        "sqlite_file_size_bytes": int(db_file_size_bytes),
    }
    return write_json_result(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LatticeMemory compression bytes.")
    parser.add_argument("--n-docs", type=int, default=100_000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/compression.json"))
    args = parser.parse_args()
    result = run_benchmark(n_docs=args.n_docs, d_model=args.d_model, output_path=args.output)
    print(f"Wrote {result['benchmark']} benchmark to {args.output}")


if __name__ == "__main__":
    main()
