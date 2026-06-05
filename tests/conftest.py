from __future__ import annotations

import os


_LEGACY_OR_OPTIONAL_GPU_IMPORT_BREAKS = [
    "test_17d.py",
    "test_app_flagship_tdd.py",
    "test_flagship_v15_tdd.py",
    "test_parity.py",
]

if os.environ.get("E8_INCLUDE_LEGACY_TESTS") != "1":
    collect_ignore = list(_LEGACY_OR_OPTIONAL_GPU_IMPORT_BREAKS)
