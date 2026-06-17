from __future__ import annotations

import pytest

from latticememory import cli as lattice_cli
from latticememory.ide import cli as ide_cli
from latticememory.ide.lattice_ops import list_verticals, proxy_doctor


def test_list_verticals_includes_prompt_firewall():
    rows = list_verticals()

    assert any(row["class"] == "LatticePromptFirewall" for row in rows)
    assert any(row["class"] == "LatticeTrainingCleaner" for row in rows)


def test_proxy_doctor_reports_unreachable():
    result = proxy_doctor(host="127.0.0.1", port=9)

    assert result["reachable"] is False
    assert "error" in result


def test_provider_show_prints_redacted_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("LATTICE_IDE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LATTICE_IDE_MODEL", "demo")
    monkeypatch.setenv("LATTICE_IDE_API_KEY", "sk-abcdefghijklmnopqrstuvwxyz")

    assert ide_cli.main(["provider", "show"]) == 0
    out = capsys.readouterr().out

    assert "https://api.example.com/v1" in out
    assert "demo" in out
    assert "sk-a...wxyz" in out
    assert "abcdefghijklmnopqrstuvwxyz" not in out


def test_chat_uses_provider(monkeypatch, capsys):
    monkeypatch.setenv("LATTICE_IDE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LATTICE_IDE_MODEL", "demo")
    monkeypatch.setenv("LATTICE_IDE_API_KEY", "secret")
    monkeypatch.setattr("latticememory.ide.cli.chat_completion", lambda config, prompt: f"answer: {prompt}")

    assert ide_cli.main(["chat", "hello"]) == 0

    assert "answer: hello" in capsys.readouterr().out


def test_cli_verticals_list(capsys):
    assert ide_cli.main(["verticals", "list"]) == 0

    assert "LatticePromptFirewall" in capsys.readouterr().out


def test_cache_inspect_delegates_to_existing_cli(monkeypatch):
    captured = {}

    def fake_inspect(args):
        captured["cache"] = args.cache
        captured["verbose"] = args.verbose
        captured["sample"] = args.sample
        return 0

    monkeypatch.setattr("latticememory.ide.cli.cmd_inspect", fake_inspect)

    assert ide_cli.main(["cache", "inspect", "--cache", "helpdesk.db", "--verbose", "--sample", "2"]) == 0
    assert captured == {"cache": "helpdesk.db", "verbose": True, "sample": 2}


def test_top_level_ide_help_delegates_to_ide_parser(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["lattice", "ide", "--help"])

    with pytest.raises(SystemExit) as exc:
        lattice_cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Configure BYOK provider" in out
    assert "VS Code bridge" in out


def test_interactive_loop_dispatches_single_command(monkeypatch, capsys):
    commands = iter(["verticals list", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))

    assert ide_cli.main([]) == 0

    assert "LatticeTrainingCleaner" in capsys.readouterr().out
