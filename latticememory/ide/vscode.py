from __future__ import annotations

import subprocess
from pathlib import Path


class VSCodeUnavailable(RuntimeError):
    pass


def _run_code(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(["code", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise VSCodeUnavailable("VS Code CLI 'code' was not found on PATH.") from exc
    if result.returncode != 0:
        raise VSCodeUnavailable(result.stderr.strip() or "VS Code CLI command failed.")
    return result


def status() -> dict[str, str | bool]:
    result = _run_code(["--version"])
    version = result.stdout.splitlines()[0] if result.stdout.splitlines() else "unknown"
    return {"available": True, "version": version}


def open_path(path: str | Path) -> None:
    _run_code([str(path)])


def list_extensions() -> list[str]:
    result = _run_code(["--list-extensions"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def install_extension(extension_id: str) -> None:
    _run_code(["--install-extension", extension_id])
