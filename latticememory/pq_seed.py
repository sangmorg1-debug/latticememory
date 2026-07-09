"""Q&A-pairs file format shared by --warm-path and PQ cache seeding.

This is the simple, general-purpose format: a list of {question, answer}
rows in CSV, JSON, or JSONL. It is deliberately NOT the proof-pack schema
(latticememory/proof_pack.py's cache_seed/calibration/evaluation/adversarial
splits) -- that schema is for reproducing the proof-pack benchmark; this
one is for a design partner's own data, which is just Q&A pairs.
"""
from __future__ import annotations

import csv as _csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PQ_NUM_BLOCKS = 8
DEFAULT_PQ_CODEBOOK_SIZE = 256


def load_qa_pairs_file(path: str) -> list[dict]:
    """Load Q&A pairs from a CSV, JSON, or JSONL file.

    CSV: columns become dict keys per row (DictReader). JSON: must be a
    list of dicts (a single dict is wrapped in a list). JSONL: one dict
    per line. Returns [] (with a logged warning) if the file is missing,
    unreadable, or has an unsupported extension -- callers decide what
    "no data" means for them, this function never raises for a bad input
    file.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("qa pairs file %s does not exist", path)
        return []

    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            with open(p, encoding="utf-8") as f:
                return [dict(row) for row in _csv.DictReader(f)]
        if suffix in (".json", ".jsonl"):
            text = p.read_text(encoding="utf-8")
            if suffix == ".jsonl":
                return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        logger.warning("qa pairs file %s: unsupported format %s", path, suffix)
        return []
    except Exception as exc:
        logger.warning("qa pairs file %s: failed to load: %s", path, exc)
        return []
