"""Tests for latticememory.pq_seed's Q&A-pairs file loader."""
from __future__ import annotations

import json

from latticememory.pq_seed import load_qa_pairs_file


def test_load_qa_pairs_file_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"question": "What is the refund policy?", "answer": "30 days."}\n'
        '{"question": "How do I reset my password?", "answer": "Use the reset link."}\n',
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert len(rows) == 2
    assert rows[0]["question"] == "What is the refund policy?"
    assert rows[0]["answer"] == "30 days."


def test_load_qa_pairs_file_reads_json_list(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([{"prompt": "Hi", "value": "Hello!"}]),
        encoding="utf-8",
    )

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"prompt": "Hi", "value": "Hello!"}]


def test_load_qa_pairs_file_reads_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("question,answer,intent_id\nHi,Hello!,greeting\n", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == [{"question": "Hi", "answer": "Hello!", "intent_id": "greeting"}]


def test_load_qa_pairs_file_missing_returns_empty(tmp_path):
    rows = load_qa_pairs_file(str(tmp_path / "does_not_exist.jsonl"))

    assert rows == []


def test_load_qa_pairs_file_unsupported_extension_returns_empty(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("not a supported format", encoding="utf-8")

    rows = load_qa_pairs_file(str(path))

    assert rows == []
