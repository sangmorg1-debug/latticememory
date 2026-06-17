from __future__ import annotations

import subprocess

import pytest

from latticememory.ide.vscode import VSCodeUnavailable, list_extensions, open_path, status
from latticememory.ide.workspace import resolve_workspace_path


def test_resolve_workspace_path_rejects_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    readme = root / "README.md"
    readme.write_text("hi", encoding="utf-8")

    assert resolve_workspace_path(root, "README.md") == readme.resolve()
    with pytest.raises(ValueError, match="outside workspace"):
        resolve_workspace_path(root, "..")


def test_vscode_status_uses_code_version(monkeypatch):
    def fake_run(args, capture_output, text, check):
        assert args == ["code", "--version"]
        return subprocess.CompletedProcess(args, 0, stdout="1.90.0\nhash\nx64\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert status()["available"] is True
    assert status()["version"] == "1.90.0"


def test_vscode_unavailable_when_code_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(VSCodeUnavailable):
        status()


def test_open_path_and_list_extensions(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, capture_output=True, text=True, check=False):
        calls.append(args)
        if "--list-extensions" in args:
            return subprocess.CompletedProcess(args, 0, stdout="ms-python.python\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    file_path = tmp_path / "file.py"
    file_path.write_text("print('x')", encoding="utf-8")

    open_path(file_path)
    assert calls[-1] == ["code", str(file_path)]
    assert list_extensions() == ["ms-python.python"]
