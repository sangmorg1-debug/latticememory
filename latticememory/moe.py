import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Any

from latticememory.rag.e8_retriever import _e8_nearest

class E8SnapSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for E8 Lattice Snapping.
    During the forward pass, projects the input to the nearest E8 lattice point.
    During the backward pass, passes the gradient directly through.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _e8_nearest(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class LatticeMoERouter(nn.Module):
    """
    E8 Sub-Lattice Mixture of Experts (MoE) Router.
    
    Routes token activation representations to specialized experts using
    discrete E8 lattice and sub-lattice (D8, E7, E6) coordinate math.
    Supports Straight-Through Estimation (STE) for load-balance training.
    """
    def __init__(
        self, 
        d_model: int, 
        num_experts: int = 8, 
        routing_type: str = "coset_modulo", 
        beta: float = 1.0
    ):
        """
        Args:
            d_model: Hidden dimension of input tokens.
            num_experts: Number of experts to route to. For 'd8' routing, num_experts is forced to 2.
            routing_type: One of 'd8', 'e7_e6', 'coset_modulo'.
            beta: Scaling factor for E8 snapping (controls cell size/resolution).
        """
        super().__init__()
        self.d_model = d_model
        self.routing_type = routing_type
        
        if routing_type == "d8":
            self.num_experts = 2
        else:
            self.num_experts = num_experts
            
        self.beta = beta
        
        # Learnable projection from hidden dimension to 8D lattice space
        self.projection = nn.Linear(d_model, 8)
        nn.init.normal_(self.projection.weight, mean=0.0, std=2.0)
        nn.init.normal_(self.projection.bias, mean=0.0, std=2.0)
        
        # Learnable soft gating logits layer (for auxiliary load-balancing loss)
        self.gating = nn.Linear(8, self.num_experts)
        
        # Fixed integer coefficients for modulo routing (coprime vectors to disperse hashes)
        self.register_buffer(
            "coset_coefficients", 
            torch.tensor([1, 3, 5, 7, 11, 13, 17, 19], dtype=torch.long)
        )

    def route_d8(self, snapped: torch.Tensor) -> torch.Tensor:
        """
        Routes based on D8 membership.
        D8 is the sub-lattice of E8 containing only integer vectors with an even sum.
        Since E8 coordinates are either all integers or all half-integers:
        - If integer coordinates: routed to Expert 0.
        - If half-integer coordinates: routed to Expert 1 (coset E8 \ D8).
        """
        # Determine if coordinates are integers
        # We can check if snapped % 1 == 0. (For half-integers, snapped % 1 == 0.5)
        # To avoid floating point issues, round and compare
        is_integer = torch.isclose(snapped % 1.0, torch.zeros_like(snapped), atol=1e-5).all(dim=-1)
        expert_indices = torch.where(is_integer, torch.zeros_like(is_integer, dtype=torch.long), torch.ones_like(is_integer, dtype=torch.long))
        return expert_indices

    def route_e7_e6(self, snapped: torch.Tensor) -> torch.Tensor:
        """
        Routes using E7 and E6 orthogonal subspace projections.
        
        E7 is defined by orthogonality to root r1 = (0, 0, 0, 0, 0, 0, 1, -1) (i.e. x_7 = x_8).
        E6 is defined by orthogonality to r1 and r2 = (0, 0, 0, 0, 0, 1, -1, 0) (i.e. x_6 = x_7 = x_8).
        
        We compute coordinate differences along these root directions:
        - d_e7 = x_7 - x_8 (always an integer since coordinates are either both int or both half-int)
        - d_e6 = x_6 - x_7
        
        Then route to experts using these discrete differences:
        index = (d_e7 + d_e6 * 3) % num_experts
        """
        x_6 = snapped[..., 5]
        x_7 = snapped[..., 6]
        x_8 = snapped[..., 7]
        
        # Differences are guaranteed to be integers
        d_e7 = torch.round(x_7 - x_8).long()
        d_e6 = torch.round(x_6 - x_7).long()
        
        expert_indices = (d_e7 + d_e6 * 3) % self.num_experts
        return expert_indices

    def route_coset_modulo(self, snapped: torch.Tensor) -> torch.Tensor:
        """
        Routes to N experts using a linear combination of E8 coordinates modulo N.
        Since E8 coordinates can be half-integers, we multiply by 2 to obtain integers,
        then take the dot product with coset_coefficients.
        Because of E8 lattice parity properties, this dot product is mathematically
        guaranteed to be a multiple of 4. We divide by 4 to get a dense, balanced index.
        """
        # Multiply by 2 and round to get exact integers
        ints = torch.round(snapped * 2.0).long()
        
        # Dot product along last dimension with coefficients
        dot_product = (ints * self.coset_coefficients).sum(dim=-1)
        
        # Divide by 4 (since dot_product is always a multiple of 4 in E8)
        expert_indices = (dot_product // 4) % self.num_experts
        return expert_indices

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            H: Input activations of shape (..., d_model)
        Returns:
            expert_indices: Chosen expert IDs of shape (...)
            snapped: The snapped E8 lattice coordinate (..., 8)
            metrics: Dict containing soft gating logits and load metrics
        """
        orig_shape = H.shape[:-1]
        H_flat = H.view(-1, self.d_model)
        
        # 1. Project tokens to 8D lattice space
        projected = self.projection(H_flat)
        
        # 2. Scale by beta (controls grid size)
        scaled = projected / self.beta
        
        # 3. Snap to E8 using the Straight-Through Estimator (STE)
        snapped = E8SnapSTE.apply(scaled)
        
        # 4. Route using selected mathematical gating scheme
        if self.routing_type == "d8":
            expert_indices = self.route_d8(snapped)
        elif self.routing_type == "e7_e6":
            expert_indices = self.route_e7_e6(snapped)
        elif self.routing_type == "coset_modulo":
            expert_indices = self.route_coset_modulo(snapped)
        else:
            raise ValueError(f"Unknown routing type: {self.routing_type}")
            
        # 5. Compute soft gating logits (for load-balancing loss)
        soft_logits = self.gating(scaled)
        
        # Reshape outputs to original batch/seq layout
        expert_indices = expert_indices.view(orig_shape)
        snapped = snapped.view(orig_shape + (8,))
        soft_logits = soft_logits.view(orig_shape + (self.num_experts,))
        
        # Compute routing metrics
        metrics = self._compute_routing_metrics(expert_indices, soft_logits)
        
        return expert_indices, snapped, metrics

    def _compute_routing_metrics(self, expert_indices: torch.Tensor, soft_logits: torch.Tensor) -> Dict[str, Any]:
        """
        Compute entropy, expert distribution, and coefficient of variation (CV) for load balancing.
        """
        flat_indices = expert_indices.view(-1)
        num_tokens = flat_indices.numel()
        
        # Count occurrences per expert
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        fractions = counts / num_tokens
        
        # Entropy of discrete routing
        entropy = -torch.sum(fractions * torch.log(fractions + 1e-10))
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else torch.tensor(1.0)
        
        # Coefficient of variation (standard deviation / mean)
        std = torch.std(fractions)
        mean = torch.mean(fractions)
        cv = std / (mean + 1e-10)
        
        return {
            "soft_logits": soft_logits,
            "expert_fractions": fractions,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "coefficient_of_variation": cv,
            "counts": counts
        }

    def compute_load_balancing_loss(self, soft_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes standard MoE auxiliary load-balancing loss.
        L = N * sum_{i=1}^N (f_i * P_i)
        where f_i is the fraction of tokens routed to expert i,
        and P_i is the mean soft probability allocated to expert i.
        
        Minimizing this loss drives both f_i and P_i towards 1/N.
        """
        flat_indices = expert_indices.view(-1)
        flat_logits = soft_logits.view(-1, self.num_experts)
        num_tokens = flat_indices.numel()
        
        # 1. Differentiable routing probabilities (P_i)
        probs = F.softmax(flat_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        # 2. Discrete fractions (f_i) - we treat this as a constant target for gradient safety
        # or propagate gradients using one-hot indicators.
        # Here we use a differentiable soft approximation or a gradient-stopped one-hot vector.
        one_hot = F.one_hot(flat_indices, num_classes=self.num_experts).float()
        fractions = one_hot.mean(dim=0).detach()  # Stop gradient for target fraction to stabilize
        
        # Auxiliary loss formula
        loss = self.num_experts * torch.sum(fractions * mean_probs)
        return loss
