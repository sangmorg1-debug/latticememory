# Implementation Plan: Phase 1 (Rust/WASM Edge Kernel)

This plan outlines the steps to build **Phase 1** of the LatticeMemory roadmap: re-implementing the core E8 lattice snapping math in Rust and compiling it to WebAssembly (WASM) for edge, mobile, and browser environments.

---

## User Review Required

> [!WARNING]
> **Rust Toolchain Missing**: The local environment does not have `cargo` installed. I will write the complete Rust crate and build configuration files (`Cargo.toml` and source code) directly into the repository so that they are ready for compilation once the Rust toolchain is installed (via `rustup` and `wasm-pack`). 

---

## Open Questions

1. **JS/WASM API Structure**: Should the WASM library expose a simple flat function `snap_embedding(float32_array, d_model)` returning a byte array, or a full `LatticeIndex` class directly in WebAssembly? *(Recommended: Expose a flat utility function `snap_embeddings` first to act as a lightweight, low-level snapping kernel for edge caches, keeping the WASM binary under 100KB).*

---

## Proposed Changes

We will create a new `rust/` directory containing the complete Cargo package structure:

### 1. Cargo Configuration

#### [NEW] [Cargo.toml](file:///e:/latticememory/rust/Cargo.toml)
* Configure Rust package `latticememory_kernel`.
* Add `wasm-bindgen` dependency.
* Configure crate type as `cdylib` and `rlib` for compilation to both static library and WASM.

---

### 2. Rust Core Quantizer Implementation

#### [NEW] [src/lib.rs](file:///e:/latticememory/rust/src/lib.rs)
* Implement E8 Shell-1 codebook construction (240 coordinate vectors of length 8).
* Implement D8 decoder: finding the closest point in $D_8$ (integer coordinates with even sum) to a given 8D float slice.
* Implement E8 nearest point snapping: comparing $D_8$ and shifted $D_8 + 0.5$ candidates.
* Implement snapping function:
  ```rust
  #[wasm_bindgen]
  pub fn snap_embeddings(embeddings: &[f32], d_model: usize) -> Vec<u8>
  ```
  Splits float array into blocks of 8, finds the nearest E8 coordinate, and returns the byte array of indices.

---

### 3. Verification Plan

#### Manual Verification
* The Rust files will be visually reviewed for mathematical correctness and matched line-for-line against the Python `e8_retriever.py` reference.
* Once the user installs `wasm-pack` (`cargo install wasm-pack`), they can run:
  ```bash
  cd rust
  wasm-pack build --target web
  ```
  to compile it to a browser-ready package.
