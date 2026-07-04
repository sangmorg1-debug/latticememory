# Experimental Shape Runtime

`latticememory.shape_runtime` is an experimental adapter for precomputed shape,
CAD, product, or geometry feature vectors.

What it does:

- indexes fixed-size shape vectors with `RFSnapLatticeMemory`
- retrieves nearest shape vectors through the existing lattice memory path
- exposes a `ShapeHammingRouter` for E8 Hamming lookup over vector keys

What it does not do:

- encode raw meshes, point clouds, images, or CAD files
- claim benchmarked 3D retrieval quality
- replace a domain-specific geometry encoder

The intended use is narrow: if an external model already produces stable shape
embeddings, this runtime lets those embeddings use the same lattice indexing,
Hamming-routing, and metadata hooks as text/document memory.
