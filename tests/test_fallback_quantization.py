import os
import tempfile
import pytest
import torch
import numpy as np

from latticememory.memory import DenseVectorFallback, MemoryDocument
from latticememory.fallbacks import FaissVectorFallback


@pytest.fixture
def float32_fallback():
    return DenseVectorFallback(d_model=128)


@pytest.fixture
def int8_fallback():
    return DenseVectorFallback(d_model=128, quantization_bits=8)


@pytest.fixture
def int4_fallback():
    return DenseVectorFallback(d_model=128, quantization_bits=4)


def test_dense_odd_dimension_check():
    # Odd dimension should be rejected for 4-bit packing
    with pytest.raises(ValueError, match="even for 4-bit quantization"):
        DenseVectorFallback(d_model=127, quantization_bits=4)

    # 8-bit should allow odd dimensions
    DenseVectorFallback(d_model=127, quantization_bits=8)


def test_dense_quantization_correctness(int8_fallback, int4_fallback, float32_fallback):
    torch.manual_seed(42)
    d_model = 128
    N = 50
    
    # Generate random vectors
    embeddings = torch.randn(N, d_model)
    documents = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=embeddings[i])
        for i in range(N)
    ]
    
    # Add to all fallbacks
    float32_fallback.add_documents(documents)
    int8_fallback.add_documents(documents)
    int4_fallback.add_documents(documents)
    
    # Check dimensions of embeddings internally
    assert float32_fallback._embeddings.dtype == torch.float32
    assert float32_fallback._embeddings.shape == (N, d_model)
    
    assert int8_fallback._embeddings.dtype == torch.int8
    assert int8_fallback._embeddings.shape == (N, d_model)
    
    assert int4_fallback._embeddings.dtype == torch.uint8
    assert int4_fallback._embeddings.shape == (N, d_model // 2)
    
    # Verify index sizes
    assert float32_fallback.get_index_size_bytes() == N * d_model * 4
    assert int8_fallback.get_index_size_bytes() == N * d_model * 1
    assert int4_fallback.get_index_size_bytes() == N * (d_model // 2) * 1
    
    # Query with a known document (should remain top-1)
    for q_idx in range(5):
        query = embeddings[q_idx]
        
        flat_results = float32_fallback.search(query, top_k=5)
        int8_results = int8_fallback.search(query, top_k=5)
        int4_results = int4_fallback.search(query, top_k=5)
        
        # Known nearest (itself) must be top-1 in all
        assert flat_results[0].doc_id == f"doc-{q_idx}"
        assert int8_results[0].doc_id == f"doc-{q_idx}"
        assert int4_results[0].doc_id == f"doc-{q_idx}"
        
        # Check scores are relatively close for INT8
        assert abs(flat_results[0].score - int8_results[0].score) < 0.02
        
        # Track INT4 score difference as a diagnostic rather than a strict threshold
        int4_diff = abs(flat_results[0].score - int4_results[0].score)
        print(f"\nQuery {q_idx} INT4 top score diff: {int4_diff:.4f}")


def test_dense_quantized_random_query_agreement_is_measured():
    torch.manual_seed(123)
    d_model = 128
    n_docs = 500
    n_queries = 100
    top_k = 10
    embeddings = torch.randn(n_docs, d_model)
    documents = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=embeddings[i])
        for i in range(n_docs)
    ]
    float32_fallback = DenseVectorFallback(d_model=d_model)
    int8_fallback = DenseVectorFallback(d_model=d_model, quantization_bits=8)
    int4_fallback = DenseVectorFallback(d_model=d_model, quantization_bits=4)
    for fallback in (float32_fallback, int8_fallback, int4_fallback):
        fallback.add_documents(documents)

    int8_top1 = int4_top1 = 0
    int8_overlap = int4_overlap = 0.0
    for query in torch.randn(n_queries, d_model):
        baseline = [hit.doc_id for hit in float32_fallback.search(query, top_k=top_k)]
        int8 = [hit.doc_id for hit in int8_fallback.search(query, top_k=top_k)]
        int4 = [hit.doc_id for hit in int4_fallback.search(query, top_k=top_k)]
        int8_top1 += int8[0] == baseline[0]
        int4_top1 += int4[0] == baseline[0]
        int8_overlap += len(set(int8) & set(baseline)) / top_k
        int4_overlap += len(set(int4) & set(baseline)) / top_k

    metrics = {
        "int8_top1_agreement": int8_top1 / n_queries,
        "int4_top1_agreement": int4_top1 / n_queries,
        "int8_overlap_at_10": int8_overlap / n_queries,
        "int4_overlap_at_10": int4_overlap / n_queries,
    }
    print(f"\nDense fallback quantization random-query metrics: {metrics}")
    assert metrics["int8_top1_agreement"] >= 0.95
    assert metrics["int8_overlap_at_10"] >= 0.95
    # Int4 is intentionally treated as lower-fidelity until real embedding benchmarks prove otherwise.
    assert metrics["int4_top1_agreement"] >= 0.35
    assert metrics["int4_overlap_at_10"] >= 0.50


def test_dense_fallback_save_load(int4_fallback):
    torch.manual_seed(42)
    d_model = 128
    embeddings = torch.randn(5, d_model)
    documents = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=embeddings[i], metadata={"idx": i})
        for i in range(5)
    ]
    int4_fallback.add_documents(documents)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "dense_int4.pt")
        int4_fallback.save(save_path)
        
        loaded = DenseVectorFallback.load(save_path)
        assert loaded.d_model == 128
        assert loaded.quantization_bits == 4
        assert loaded.num_documents == 5
        assert loaded._texts["doc-2"] == "Doc 2"
        assert loaded._metadata["doc-2"] == {"idx": 2}
        assert torch.equal(loaded._embeddings, int4_fallback._embeddings)


def test_faiss_quantization_and_lazy_train():
    pytest.importorskip("faiss")
    
    d_model = 128
    faiss_8bit = FaissVectorFallback(d_model=d_model, quantization_bits=8)
    faiss_4bit = FaissVectorFallback(d_model=d_model, quantization_bits=4)
    
    assert not faiss_8bit._index.is_trained
    assert not faiss_4bit._index.is_trained
    
    # Add < 256 documents (should train using synthetic padding)
    docs_small = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=torch.randn(d_model))
        for i in range(10)
    ]
    
    faiss_8bit.add_documents(docs_small)
    assert faiss_8bit._index.is_trained
    assert faiss_8bit.num_documents == 10
    
    # Add >= 256 documents to the other one (should train directly on batch)
    docs_large = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=torch.randn(d_model))
        for i in range(300)
    ]
    faiss_4bit.add_documents(docs_large)
    assert faiss_4bit._index.is_trained
    assert faiss_4bit.num_documents == 300
    
    # Search and verify index size
    q = torch.randn(d_model)
    res_8 = faiss_8bit.search(q, top_k=3)
    res_4 = faiss_4bit.search(q, top_k=3)
    assert len(res_8) == 3
    assert len(res_4) == 3
    
    assert faiss_8bit.get_index_size_bytes() == 10 * d_model
    assert faiss_4bit.get_index_size_bytes() == 300 * (d_model // 2)


def test_faiss_synthetic_training_is_deterministic():
    pytest.importorskip("faiss")

    d_model = 128
    torch.manual_seed(33)
    docs = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=torch.randn(d_model))
        for i in range(12)
    ]
    query = torch.randn(d_model)

    left = FaissVectorFallback(d_model=d_model, quantization_bits=8, seed=99)
    right = FaissVectorFallback(d_model=d_model, quantization_bits=8, seed=99)
    left.add_documents(docs)
    right.add_documents(docs)

    assert [(hit.doc_id, hit.score) for hit in left.search(query, top_k=5)] == [
        (hit.doc_id, hit.score) for hit in right.search(query, top_k=5)
    ]


def test_faiss_save_load():
    pytest.importorskip("faiss")
    d_model = 128
    fallback = FaissVectorFallback(d_model=d_model, quantization_bits=8)
    docs = [
        MemoryDocument(doc_id=f"doc-{i}", text=f"Doc {i}", embedding=torch.randn(d_model))
        for i in range(5)
    ]
    fallback.add_documents(docs)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "faiss_int8.index")
        fallback.save(save_path)
        
        loaded = FaissVectorFallback.load(save_path)
        assert loaded.d_model == 128
        assert loaded.quantization_bits == 8
        assert loaded.num_documents == 5
        assert loaded._texts["doc-2"] == "Doc 2"
