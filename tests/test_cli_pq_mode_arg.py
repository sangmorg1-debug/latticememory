"""Tests for the `lattice serve --pq-mode` CLI flag.

proxy_server.py itself is not unit tested (it does real network/model
work at module import time by design -- see the absence of any existing
test_proxy_server.py) -- this test covers argument parsing only, the
part that IS safely testable without starting a real server.
"""
from __future__ import annotations

from latticememory.cli import build_parser


def test_serve_accepts_pq_mode_flag():
    parser = build_parser()
    args = parser.parse_args(["serve", "--pq-mode", "--warm-path", "qa.jsonl"])

    assert args.pq_mode is True


def test_serve_pq_mode_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args(["serve"])

    assert args.pq_mode is False
