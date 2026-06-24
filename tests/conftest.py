from __future__ import annotations

import os

# transformers 5.6.2's threaded weight-materialization path (core_model_loading.py,
# GLOBAL_WORKERS = min(4, cpu_count())) crashes with a Windows access violation when
# many tests in one pytest process repeatedly load a real SentenceTransformer model
# (confirmed: passes in isolation, crashes only as part of the full suite - cumulative
# thread-pool/resource issue, not a logic bug in any test). Forcing single-threaded
# weight loading avoids it without changing the pinned transformers version.
try:
    import transformers.core_model_loading as _core_model_loading

    _core_model_loading.GLOBAL_WORKERS = 1
except ImportError:
    pass


_LEGACY_OR_OPTIONAL_GPU_IMPORT_BREAKS = [
    "test_17d.py",
    "test_app_flagship_tdd.py",
    "test_flagship_v15_tdd.py",
    "test_parity.py",
]

if os.environ.get("E8_INCLUDE_LEGACY_TESTS") != "1":
    collect_ignore = list(_LEGACY_OR_OPTIONAL_GPU_IMPORT_BREAKS)
