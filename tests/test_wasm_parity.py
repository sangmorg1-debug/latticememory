from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from latticememory.memory import RFSnapLatticeMemory


@pytest.mark.skipif(shutil.which("node") is None, reason="requires node")
def test_rust_wasm_snap_matches_python_e8_quantizer():
    """Extension WASM and Python produce identical E8 keys for the same vector."""
    vector = np.linspace(-1.0, 1.0, 384, dtype=np.float32)
    vector /= np.linalg.norm(vector) + 1e-9
    python_key = list(RFSnapLatticeMemory(d_model=384).lattice_key_for(vector))
    js_values = ", ".join(f"{float(value):.9g}" for value in vector)
    js = f"""
import {{ initSync, snap_embeddings }} from "./browser_extension/wasm/latticememory_kernel.js";
import {{ readFileSync }} from "node:fs";

const wasmBytes = readFileSync("./browser_extension/wasm/latticememory_kernel_bg.wasm");
initSync({{ module: wasmBytes }});
const embedding = Float32Array.from([{js_values}]);
const key = snap_embeddings(embedding, 384);
console.log(JSON.stringify(Array.from(key)));
"""

    import os
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.resolve()
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", js],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == python_key
