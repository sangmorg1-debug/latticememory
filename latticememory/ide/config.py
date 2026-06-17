from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IdeConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = ""

    def redacted(self) -> dict[str, str]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": redact_secret(self.api_key),
        }


def config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "latticememory" / "ide_config.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "latticememory" / "ide_config.json"
    return Path.home() / ".config" / "latticememory" / "ide_config.json"


def load_config(path: Path | None = None) -> IdeConfig:
    cfg_path = path or config_path()
    if not cfg_path.exists():
        return IdeConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return IdeConfig(
        base_url=str(data.get("base_url", "")),
        model=str(data.get("model", "")),
        api_key=str(data.get("api_key", "")),
    )


def save_config(config: IdeConfig, path: Path | None = None) -> Path:
    cfg_path = path or config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {"base_url": config.base_url, "model": config.model, "api_key": config.api_key},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


def provider_from_env(config: IdeConfig) -> IdeConfig:
    return IdeConfig(
        base_url=os.environ.get("LATTICE_IDE_BASE_URL", config.base_url),
        model=os.environ.get("LATTICE_IDE_MODEL", config.model),
        api_key=os.environ.get("LATTICE_IDE_API_KEY", config.api_key),
    )


def redact_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
