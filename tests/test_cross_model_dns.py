import pytest
import torch
from latticememory.dns import CrossModelAligner

def test_cross_model_aligner_api():
    d_model_a = 384
    d_model_b = 512
    d_shared = 384
    
    aligner = CrossModelAligner(
        d_model_a=d_model_a,
        d_model_b=d_model_b,
        d_shared=d_shared,
        device="cpu"
    )
    
    # Generate dummy embeddings
    torch.manual_seed(42)
    dummy_a = torch.randn(10, d_model_a)
    dummy_b = torch.randn(10, d_model_b)
    
    # Test project_and_snap APIs
    projected_a = aligner.project_and_snap_a(dummy_a, rescale=True)
    assert projected_a.shape == (10, d_shared)
    
    projected_b = aligner.project_and_snap_b(dummy_b, rescale=False)
    assert projected_b.shape == (10, d_shared)
    
    # Test snap_a / snap_b canonical 48-byte E8 key resolution
    key_a = aligner.snap_a(dummy_a[0])
    assert isinstance(key_a, bytes)
    assert len(key_a) == 48
    
    key_b = aligner.snap_b(dummy_b[0])
    assert isinstance(key_b, bytes)
    assert len(key_b) == 48

def test_cross_model_aligner_fit():
    d_model_a = 384
    d_model_b = 512
    d_shared = 384
    
    aligner = CrossModelAligner(
        d_model_a=d_model_a,
        d_model_b=d_model_b,
        d_shared=d_shared,
        device="cpu"
    )
    
    # Generate simple clean parallel dataset
    torch.manual_seed(42)
    latent = torch.randn(50, 4)
    W_A = torch.randn(4, d_model_a)
    W_B = torch.randn(4, d_model_b)
    
    emb_a = latent @ W_A
    emb_b = latent @ W_B
    
    # Before alignment: check exact matches
    matches_before = sum(1 for i in range(len(emb_a)) if aligner.snap_a(emb_a[i]) == aligner.snap_b(emb_b[i]))
    
    # Run alignment training (5 epochs)
    losses = aligner.train_alignment(
        emb_a=emb_a,
        emb_b=emb_b,
        epochs=5,
        lr=0.0001,
        pre_fit_ridge=True
    )
    
    # After alignment: check exact matches
    matches_after = sum(1 for i in range(len(emb_a)) if aligner.snap_a(emb_a[i]) == aligner.snap_b(emb_b[i]))
    
    assert len(losses) == 5
    assert matches_after > matches_before
    assert matches_after > 0

def test_cross_model_aligner_fit_ridge_correctness():
    d_model_a = 384
    d_model_b = 512
    d_shared = 384
    
    aligner = CrossModelAligner(
        d_model_a=d_model_a,
        d_model_b=d_model_b,
        d_shared=d_shared,
        device="cpu"
    )
    
    # Generate clean parallel dataset where B is exactly a linear projection of A
    torch.manual_seed(42)
    emb_a = torch.randn(50, d_model_a)
    # Mapping matrix from A to B
    M = torch.randn(d_model_a, d_model_b)
    emb_b = emb_a @ M
    
    # Fit ridge
    aligner.fit_ridge(emb_a, emb_b)
    
    # Projected embeddings should align
    with torch.no_grad():
        proj_a = aligner.proj_a(emb_a)
        proj_b = aligner.proj_b(emb_b)
        
    diff = torch.mean(torch.norm(proj_a - proj_b, dim=-1))
    # Check that difference is very small (it should resolve the linear transformation perfectly)
    assert diff < 0.1
