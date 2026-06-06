import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Union

from latticememory.rag.e8_retriever import E8LatticeDB, _e8_nearest
from latticememory.moe import E8SnapSTE

class CrossModelAligner(nn.Module):
    """
    Cross-Model Semantic DNS Resolver.
    
    Aligns and projects embeddings from two different model spaces (e.g. Model A and Model B)
    into a shared multi-block E8 lattice space. Uses Straight-Through Estimation (STE) and
    a joint contrastive-alignment loss to map identical concepts to identical E8 coordinates.
    """
    def __init__(self, d_model_a: int, d_model_b: int, d_shared: int = 384, device: str = "cpu"):
        """
        Args:
            d_model_a: Dimension of Model A space.
            d_model_b: Dimension of Model B space.
            d_shared: Dimension of the shared projection space (must be divisible by 8).
        """
        super().__init__()
        if d_shared % 8 != 0:
            raise ValueError(f"d_shared={d_shared} must be divisible by 8")
            
        self.d_model_a = d_model_a
        self.d_model_b = d_model_b
        self.d_shared = d_shared
        self.device = torch.device(device)
        
        # Learnable projections to the shared space
        self.proj_a = nn.Linear(d_model_a, d_shared)
        self.proj_b = nn.Linear(d_model_b, d_shared)
        
        # Internal E8 LatticeDB to reuse codebook and indexing logic
        self.lattice_db = E8LatticeDB(d_model=d_shared, device=self.device)
        
        self.to(self.device)

    def fit_ridge(self, emb_a: torch.Tensor, emb_b: torch.Tensor, ridge: float = 1e-4) -> None:
        """
        Initializes projection layers using Ridge Regression.
        Sets proj_a to a truncated identity projection and solves for proj_b to map emb_b to emb_a.
        """
        import numpy as np
        emb_a_np = torch.as_tensor(emb_a, device=self.device).detach().cpu().numpy()
        emb_b_np = torch.as_tensor(emb_b, device=self.device).detach().cpu().numpy()
        
        # Add bias column to emb_b
        X = emb_b_np
        Y = emb_a_np
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Solve least-squares: (X_bias.T @ X_bias + ridge * I) @ W = X_bias.T @ Y
        gram = X_bias.T @ X_bias
        gram += ridge * np.eye(gram.shape[0])
        rhs = X_bias.T @ Y
        W_sol = np.linalg.pinv(gram) @ rhs
        
        weight_b = W_sol[:-1, :].T
        bias_b = W_sol[-1, :]
        
        with torch.no_grad():
            self.proj_a.weight.copy_(torch.eye(self.d_shared, self.d_model_a, device=self.device))
            self.proj_a.bias.zero_()
            self.proj_b.weight.copy_(torch.from_numpy(weight_b).to(self.device))
            self.proj_b.bias.copy_(torch.from_numpy(bias_b).to(self.device))

    def project_and_snap_a(self, emb_a: torch.Tensor, rescale: bool = True) -> torch.Tensor:
        """Projects Model A embedding and snaps it using STE."""
        projected = self.proj_a(emb_a)
        return self._block_snap(projected, rescale=rescale)

    def project_and_snap_b(self, emb_b: torch.Tensor, rescale: bool = True) -> torch.Tensor:
        """Projects Model B embedding and snaps it using STE."""
        projected = self.proj_b(emb_b)
        return self._block_snap(projected, rescale=rescale)

    def _block_snap(self, projected: torch.Tensor, rescale: bool = True) -> torch.Tensor:
        """Splits into 8D blocks, scales by beta, snaps using STE, and optionally reconstructs."""
        # projected shape: (..., d_shared)
        orig_shape = projected.shape
        flat = projected.view(-1, 8)
        
        # Scale each block by its norm divided by sqrt(2) (matching E8LatticeDB._quantize_to_indices)
        beta = flat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8) / math.sqrt(2.0)
        scaled = flat / beta
        
        # Snap with STE
        snapped = E8SnapSTE.apply(scaled)
        
        if rescale:
            rescaled = snapped * beta
            return rescaled.view(orig_shape)
        else:
            return snapped.view(orig_shape)

    @torch.no_grad()
    def snap_a(self, emb_a: torch.Tensor) -> bytes:
        """Resolves Model A embedding to a canonical 48-byte E8 key."""
        emb_tensor = torch.as_tensor(emb_a, dtype=torch.float32, device=self.device)
        projected = self.proj_a(emb_tensor)
        return self.lattice_db._quantize_to_indices(projected)

    @torch.no_grad()
    def snap_b(self, emb_b: torch.Tensor) -> bytes:
        """Resolves Model B embedding to a canonical 48-byte E8 key."""
        emb_tensor = torch.as_tensor(emb_b, dtype=torch.float32, device=self.device)
        projected = self.proj_b(emb_tensor)
        return self.lattice_db._quantize_to_indices(projected)

    def train_alignment(
        self, 
        emb_a: torch.Tensor, 
        emb_b: torch.Tensor, 
        epochs: int = 100, 
        lr: float = 0.002,
        temperature: float = 0.07,
        lambda_align: float = 1.0,
        pre_fit_ridge: bool = True
    ) -> List[float]:
        """
        Trains the projection layers to align Model A and Model B embeddings.
        
        Minimizes a joint contrastive (InfoNCE) and coordinate-alignment (MSE) loss
        on the STE-snapped vectors.
        """
        if pre_fit_ridge:
            self.fit_ridge(emb_a, emb_b)
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Project and snap both embeddings using STE (unrescaled for alignment loss)
            snapped_a = self.project_and_snap_a(emb_a, rescale=False)
            snapped_b = self.project_and_snap_b(emb_b, rescale=False)
            
            # 1. InfoNCE Contrastive Loss (forces distinct concepts to separate)
            a_flat = snapped_a.view(snapped_a.shape[0], -1)
            b_flat = snapped_b.view(snapped_b.shape[0], -1)
            
            a_norm = F.normalize(a_flat, p=2, dim=-1)
            b_norm = F.normalize(b_flat, p=2, dim=-1)
            
            sim_matrix = torch.matmul(a_norm, b_norm.T) / temperature
            targets = torch.arange(sim_matrix.shape[0], device=self.device)
            
            loss_contrastive = 0.5 * (
                F.cross_entropy(sim_matrix, targets) + 
                F.cross_entropy(sim_matrix.T, targets)
            )
            
            # 2. Coordinate Alignment Loss (drives positive snapped coordinates to be identical)
            loss_align = F.mse_loss(snapped_a, snapped_b)
            
            # Joint Loss
            loss = loss_contrastive + lambda_align * loss_align
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        self.eval()
        return losses
