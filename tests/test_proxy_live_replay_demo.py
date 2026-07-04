from __future__ import annotations

import json
import html

from examples.proxy_live_replay_demo import (
    _content_from_chat_body,
    _render_html_report,
    _render_markdown_report,
    _select_replay_rows,
)
from latticememory.proof_pack import build_support_dataset


def test_select_replay_rows_uses_evaluation_then_adversarial_limit():
    dataset = build_support_dataset(
        seed_count=4,
        calibration_count=4,
        evaluation_count=6,
        adversarial_count=3,
    )

    rows = _select_replay_rows(dataset, limit=7)

    assert len(rows) == 7
    assert [row["split"] for row in rows[:6]] == ["evaluation"] * 6
    assert rows[6]["split"] == "adversarial"


def test_content_from_chat_body_reads_openai_shape():
    body = {
        "choices": [
            {"message": {"content": "The canonical answer."}},
        ]
    }

    assert _content_from_chat_body(body) == "The canonical answer."


def test_live_replay_reports_include_analytics_and_claim_caveat():
    summary = {
        "total_requests": 10,
        "hits": 9,
        "misses": 1,
        "upstream_calls": 1,
        "hit_rate": 0.9,
        "upstream_call_rate": 0.1,
        "false_positive_rate": 0.0,
        "adversarial_false_positive_rate": 0.0,
        "latency_ms_avg": 1.25,
        "analytics": {"hit_rate": 0.9, "estimated_savings_usd": 0.0123},
    }

    markdown = _render_markdown_report(summary)
    html_report = _render_html_report(summary)

    assert "LatticeMemory Live Proxy Replay Demo" in markdown
    assert "0.9000" in markdown
    assert "does not prove RedisVL superiority" in markdown
    assert "<html" in html_report
    assert html.escape(json.dumps(summary["analytics"], indent=2, sort_keys=True)) in html_report
