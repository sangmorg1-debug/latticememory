# Implementation Plan: Phase 3 (MoE Router Research)

This plan outlines the steps to build **Phase 3** of the LatticeMemory roadmap: implementing and simulating an E8 sub-lattice Mixture of Experts (MoE) router, addressing token geometry stability, and formulating the academic research proof.

---

## Goal Description

Mixture of Experts (MoE) models typically rely on soft gating (e.g. top-k softmax over expert weights), which suffers from:
1. **Routing Drift**: Token representations drift, causing routing to shift dynamically.
2. **Computational Overhead**: Computing softmax across hundreds of experts is expensive.
3. **Load Imbalance**: Gating often collapses to a subset of experts, requiring complex auxiliary loss functions.

By using the discrete, permanent geometry of the E8 lattice and its sub-lattices ($D_8$, $E_7$, $E_6$), we can route tokens directly to expert models based on the mathematical region (shell, coset, or sub-lattice subspace) they snap to. This converts routing into an $O(1)$ geometric lookup.

---

## User Review Required

### 1. Token Geometry Stability Caveat
> [!WARNING]
> Snapping works exceptionally well for fully pooled sentence embeddings because they represent stable, static semantic points.
> However, **intermediate token representations in a transformer's residual stream do not share this stability**. They are highly dynamic, shift layer-by-layer, and their geometric distributions differ from sentence-level vectors. 
> To make E8 routing stable:
> - We must validate that token vectors do not collapse to a single lattice point (routing collapse).
> - We must measure the "routing stability" (how often a token switches experts between consecutive layers).
> - We must use a projection adapter that maps token representations to E8 coordinates with a load-balancing constraint.

### 2. Mathematical Gating Schemes
We propose three routing formulations:
1. **$D_8$ Coset Gating (2 Experts)**: Routes based on whether the snapped $E_8$ point is an integer vector (in $D_8$) or a half-integer vector (in the coset $E_8 \setminus D_8$).
2. **Subspace Orthogonal Projection ($E_7$ and $E_6$ Gating)**: Projects the snapped $E_8$ point onto the $E_7$ subspace (where $x_7 = x_8$) or the $E_6$ subspace (where $x_6 = x_7 = x_8$), and routes based on the projection residual or sub-lattice coordinates.
3. **Coset Hash Gating ($N$ Experts)**: Routes to $N$ experts using a deterministic hash/modulo of the $E_8$ snapped coordinates (e.g., mapping to the quotient group $E_8 / M E_8$).

---

## Proposed Changes

### [latticememory](file:///e:/latticememory/latticememory)

#### [NEW] [moe.py](file:///e:/latticememory/latticememory/moe.py)
* Create `LatticeMoERouter` as a PyTorch/NumPy compatible module:
  * Learnable projection layer to map tokens from hidden dimension $D$ to $8$-dimensional space.
  * Nearest-point $E_8$ snapping.
  * Router functions for:
    * `route_d8` (binary routing based on $D_8$ membership).
    * `route_e7_e6` (routing based on projection onto $E_7$/$E_6$ subspaces).
    * `route_coset_modulo` (routing to arbitrary $N$ experts using modulo/coset arithmetic).
  * Compute load-balancing metrics: entropy of routing distribution, coefficient of variation (CV).

---

### [examples](file:///e:/latticememory/examples)

#### [NEW] [moe_routing_simulation.py](file:///e:/latticememory/examples/moe_routing_simulation.py)
* Simulate a transformer's residual stream activations for a batch of sequences across multiple layers.
* Apply the `LatticeMoERouter` to analyze:
  * **Routing Balance**: Verify that tokens are distributed evenly across experts.
  * **Routing Stability**: Measure the token-expert persistence rate across consecutive layers.
  * **Representation Entropy**: Compare routing under random projections vs. optimized/load-balanced projections.
* Output detailed statistics demonstrating the mathematical feasibility of E8-based token routing.

---

## Verification Plan

### Automated Run
* Run the simulation: `python examples/moe_routing_simulation.py`
* Verify that:
  1. $D_8$ binary gating splits tokens into two distinct expert groups.
  2. Coset modulo gating routes to $N$ experts with high entropy (balanced load).
  3. Token routing stability metrics are calculated and analyzed across simulated layers.
