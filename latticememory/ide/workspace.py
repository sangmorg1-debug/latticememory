from __future__ import annotations

from pathlib import Path


def resolve_workspace(root: str | Path | None = None) -> Path:
    return Path(root or ".").resolve()


def resolve_workspace_path(root: str | Path, target: str | Path) -> Path:
    workspace = resolve_workspace(root)
    candidate = (workspace / target).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"Path is outside workspace: {target}")
    return candidate
