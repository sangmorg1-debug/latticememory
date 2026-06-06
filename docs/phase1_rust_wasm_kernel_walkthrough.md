# Walkthrough - Phase 1 Rust/WASM Edge Kernel

We have successfully implemented **Phase 1** of the LatticeMemory roadmap, building a high-performance Rust library for E8 lattice snapping math with web-bindgen bindings.

---

## Files Created

### 1. Cargo Configuration
* **[Cargo.toml](file:///e:/latticememory/rust/Cargo.toml)**: Defines the `latticememory_kernel` package configured to compile into both a standard Rust library (`rlib`) and a WebAssembly shared library (`cdylib`). It integrates size optimization parameters (e.g., LTO, `opt-level = "s"`) to keep the WASM build lightweight.

### 2. Snapping Kernel Source
* **[src/lib.rs](file:///e:/latticememory/rust/src/lib.rs)**: Contains the pure Rust port of the E8 snapping math.
  - Generates the 240 Shell-1 codebook coordinate vectors of length 8.
  - Implements the closest-point $D_8$ grid decoding function (`decode_d8`).
  - Implements the closest-point $E_8$ snapping function (`e8_nearest`).
  - Implements the batch vector snapping method (`snap_embeddings`) annotated with `#[wasm_bindgen]` for compilation into Javascript.

---

## Verification & Math Check

Since `cargo` is not locally available, we verified correctness by doing a line-by-line mathematical review against the Python reference:

1. **Permutation Codebook Generation**: Rust loops exactly match `itertools.combinations(range(8), 2)` and sign permutation loops in `_build_shell1_codebook`.
2. **D8 Parity Adjustment**: The `decode_d8` function implements the same argmax rounding error correction and sign projection logic.
3. **E8 Comparison**: Distance computations exactly compare $D_8$ and shifted $D_8 + 0.5$ centers.
4. **Unit Tests**: Embedded a `tests` module directly inside `lib.rs` verifying codebook coordinate count (240) and snapping determinism (384-dimensional test vectors).

To compile the library to WASM when the Rust toolchain is set up, run:
```bash
cd rust
wasm-pack build --target web
```
This generates the JS/WASM package inside `rust/pkg/` for use in browser extensions or IoT runtimes.
