from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "browser_extension"


def test_manifest_v3_declares_history_storage_wasm_and_module_worker():
    manifest = json.loads((EXT / "manifest.json").read_text())

    assert manifest["manifest_version"] == 3
    assert manifest["name"] == "LatticeMemory Local Search"
    assert manifest["minimum_chrome_version"] == "91"
    assert set(manifest["permissions"]) >= {"history", "storage"}
    assert manifest["background"] == {
        "service_worker": "background.js",
        "type": "module",
    }
    assert manifest["action"]["default_popup"] == "popup.html"
    assert manifest["content_scripts"][0]["js"] == ["content.js"]
    resources = manifest["web_accessible_resources"][0]["resources"]
    assert "wasm/latticememory_kernel_bg.wasm" in resources
    assert "wasm/latticememory_kernel.js" in resources


def test_browser_extension_contains_required_runtime_files():
    expected = [
        "background.js",
        "content.js",
        "popup.html",
        "popup.js",
        "wasm/latticememory_kernel_bg.wasm",
        "wasm/latticememory_kernel.js",
    ]

    for relative in expected:
        assert (EXT / relative).is_file(), relative
    assert (EXT / "wasm/latticememory_kernel_bg.wasm").stat().st_size > 10_000


def test_background_service_worker_indexes_history_and_uses_wasm_snapper():
    source = (EXT / "background.js").read_text()

    assert "chrome.history.onVisited.addListener" in source
    assert "indexedDB.open" in source
    assert "snap_embeddings" in source
    assert "latticememory_kernel.js" in source
    assert "handleSearch" in source
    assert "handleClearIndex" in source


def test_background_uses_canonical_index_text_and_hamming_search():
    source = (EXT / "background.js").read_text()

    assert "buildIndexText" in source
    assert "indexText" in source
    assert "hammingDistance" in source
    assert "MAX_HAMMING_DISTANCE" in source
    assert "store.getAll()" in source


def test_popup_exposes_search_stats_clear_and_export_controls():
    html = (EXT / "popup.html").read_text()
    script = (EXT / "popup.js").read_text()

    assert 'id="query"' in html
    assert 'id="search"' in html
    assert 'id="clear"' in html
    assert 'id="export"' in html
    assert "chrome.runtime.sendMessage" in script
    assert "searchHistory" in script
    assert "clearIndex" in script
    assert "exportIndex" in script
