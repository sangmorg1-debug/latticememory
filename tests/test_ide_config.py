from __future__ import annotations

import json

from latticememory.ide.config import (
    IdeConfig,
    config_path,
    load_config,
    provider_from_env,
    redact_secret,
    save_config,
)


def test_redact_secret_hides_middle():
    assert redact_secret("sk-abcdefghijklmnopqrstuvwxyz") == "sk-a...wxyz"
    assert redact_secret("") == ""
    assert redact_secret(None) == ""


def test_save_and_load_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = config_path()
    cfg = IdeConfig(base_url="https://api.example.com/v1", model="demo-model", api_key="secret")

    save_config(cfg)
    loaded = load_config()

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "demo-model"
    assert loaded == cfg


def test_environment_overrides_file_values(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    save_config(IdeConfig(base_url="https://file.example/v1", model="file-model", api_key="file-key"))
    monkeypatch.setenv("LATTICE_IDE_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("LATTICE_IDE_MODEL", "env-model")
    monkeypatch.setenv("LATTICE_IDE_API_KEY", "env-key")

    cfg = provider_from_env(load_config())

    assert cfg.base_url == "https://env.example/v1"
    assert cfg.model == "env-model"
    assert cfg.api_key == "env-key"
