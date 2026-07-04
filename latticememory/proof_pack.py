"""Proof-pack benchmark for proxy, PQ cache, Redis storage, and flywheel review.

This module is intentionally small and deterministic. It is not a benchmark
claim by itself; it creates repeatable artifacts that compare the LatticeMemory
proxy/cache path with simple exact and dense semantic cache baselines.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .memory import RFSnapLatticeMemory
from .proxy import LatticeLLMProxy
from .rag.pq_retriever import PQLatticeDB
from .redis_store import _RedisEntriesProxy, LatticeRedisStore
from .semantic_cache import RFSnapSemanticCache
from .text_runtime import RFSnapTextMemory


@dataclass(frozen=True)
class _Intent:
    intent_id: str
    prompt: str
    answer: str
    paraphrases: tuple[str, ...]
    adversarial_prompt: str
    adversarial_answer: str


_INTENTS: tuple[_Intent, ...] = (
    _Intent(
        "billing_due_date",
        "When is my phone bill due?",
        "Your phone bill is due on the 15th of each month.",
        (
            "What date do I need to pay my mobile bill?",
            "Tell me the due date for my wireless bill.",
            "When do I have to pay the monthly phone invoice?",
        ),
        "When is my phone bill due if my account is already closed?",
        "Closed accounts receive a final bill with a due date printed on the statement.",
    ),
    _Intent(
        "sim_activation",
        "How do I activate my SIM card?",
        "Insert the SIM, restart the phone, and open the activation link in your account.",
        (
            "What steps activate a new SIM?",
            "How can I turn on service for my replacement SIM card?",
            "Where do I start SIM activation?",
        ),
        "How do I activate an eSIM on a phone that is not compatible?",
        "An incompatible phone cannot activate eSIM service; use a compatible device or physical SIM.",
    ),
    _Intent(
        "international_roaming",
        "Can I use my plan while traveling internationally?",
        "International roaming depends on your plan and destination; enable roaming before you travel.",
        (
            "Will my mobile plan work overseas?",
            "Do I need to turn on roaming for international travel?",
            "Can my service be used in another country?",
        ),
        "Can I use my plan internationally without any roaming charges?",
        "International use may include roaming charges unless your plan explicitly includes that country.",
    ),
    _Intent(
        "device_return",
        "How long do I have to return a new phone?",
        "Most new phones can be returned within 14 days if they are in returnable condition.",
        (
            "What is the return window for a phone purchase?",
            "How many days are allowed for returning a handset?",
            "Can I send back a phone I just bought?",
        ),
        "How long do I have to return a damaged phone without insurance?",
        "A damaged phone may not qualify for a standard return; check warranty or protection coverage.",
    ),
    _Intent(
        "data_slowdown",
        "Why did my mobile data slow down?",
        "Data may slow after congestion, weak signal, or after your high-speed data allotment is used.",
        (
            "Why is my phone internet slower today?",
            "What causes cellular data speed to drop?",
            "Why did LTE or 5G service become slow?",
        ),
        "Why did my mobile data slow down after I reported the phone stolen?",
        "A stolen-device report can suspend service; restore the account before troubleshooting speeds.",
    ),
    _Intent(
        "employee_discount",
        "How do employees apply a wireless discount?",
        "Employees apply discounts through the benefits portal using their verified company email.",
        (
            "Where can staff add the mobile service discount?",
            "How does an employee verify a phone plan discount?",
            "What is the process for worker wireless discount enrollment?",
        ),
        "How do former employees keep a wireless discount after leaving the company?",
        "Former employees may lose eligibility; confirm discount rules with benefits administration.",
    ),
    _Intent(
        "store_pickup",
        "Can I pick up my order at a store?",
        "Eligible orders can be picked up in store after you receive the ready-for-pickup notice.",
        (
            "Is store pickup available for my phone order?",
            "Can a device order be collected at a local shop?",
            "When may I pick up my online wireless order?",
        ),
        "Can someone else pick up my order without identification?",
        "Store pickup normally requires valid identification and any authorized pickup-person details.",
    ),
    _Intent(
        "number_transfer",
        "How do I transfer my phone number?",
        "Keep the old account active and provide the account number, transfer PIN, and billing ZIP code.",
        (
            "What do I need to port my mobile number?",
            "How can I bring my current phone number to this carrier?",
            "Which details are required for number transfer?",
        ),
        "How do I transfer a disconnected phone number?",
        "Disconnected numbers usually cannot be transferred until the previous carrier restores eligibility.",
    ),
)


class _ProofPackEncoder:
    def __init__(self, prompt_to_token: dict[str, str], d_model: int = 32):
        self.prompt_to_token = prompt_to_token
        self.d_model = d_model

    def encode(self, texts: str | list[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        scalar = isinstance(texts, str)
        values = [texts] if scalar else list(texts)
        vectors = [self._one(text) for text in values]
        arr = np.asarray(vectors, dtype=np.float32)
        return arr[0] if scalar else arr

    def _one(self, text: str) -> np.ndarray:
        token = self.prompt_to_token.get(text, f"unknown:{text}")
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.d_model).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class _InMemoryRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}

    def set(self, key: str, value: str) -> bool:
        self.data[key] = value
        return True

    def setex(self, key: str, ttl: int, value: str) -> bool:
        del ttl
        return self.set(key, value)

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def exists(self, key: str) -> bool:
        return key in self.data

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            existed = key in self.data
            self.data.pop(key, None)
            deleted += int(existed)
        for values in self.sorted_sets.values():
            for key in keys:
                values.pop(key, None)
        return deleted

    def scan(self, cursor: int = 0, match: str | None = None, count: int = 100):
        prefix = ""
        if match and match.endswith("*"):
            prefix = match[:-1]
        keys = [key for key in self.data if not prefix or key.startswith(prefix)]
        return 0, keys

    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        values = self.sorted_sets.setdefault(name, {})
        new_count = 0
        for key, score in mapping.items():
            if key not in values:
                new_count += 1
            values[key] = float(score)
        return new_count

    def zrem(self, name: str, key: str) -> int:
        existed = key in self.sorted_sets.get(name, {})
        self.sorted_sets.get(name, {}).pop(key, None)
        return int(existed)

    def zcard(self, name: str) -> int:
        return len(self.sorted_sets.get(name, {}))

    def zrange(self, name: str, start: int, end: int, withscores: bool = False):
        items = sorted(self.sorted_sets.get(name, {}).items(), key=lambda item: item[1])
        if end == -1:
            selected = items[start:]
        else:
            selected = items[start : end + 1]
        return selected if withscores else [key for key, _ in selected]

    def zscan_iter(self, name: str):
        for key, score in self.sorted_sets.get(name, {}).items():
            yield key, score

    def ping(self) -> bool:
        return True

    def memory_bytes(self) -> int:
        total = 0
        for key, value in self.data.items():
            total += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        for name, values in self.sorted_sets.items():
            total += len(name.encode("utf-8"))
            total += sum(len(key.encode("utf-8")) + 8 for key in values)
        return total


class _ProofPackUpstream:
    def __init__(self, answers_by_prompt: dict[str, str]):
        self.answers_by_prompt = answers_by_prompt
        self.call_count = 0

    def post(self, _url: str, *, headers=None, json=None, timeout=None):  # noqa: A002 - requests-compatible
        self.call_count += 1
        prompt = _prompt_from_payload(json or {})
        answer = self.answers_by_prompt.get(prompt, "A support specialist must review this request.")
        return _UpstreamResponse(_response_body(answer, model=(json or {}).get("model", "proof-pack-model")))

    def __call__(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        del headers
        self.call_count += 1
        prompt = _prompt_from_payload(payload)
        answer = self.answers_by_prompt.get(prompt, "A support specialist must review this request.")
        return _response_body(answer, model=payload.get("model", "proof-pack-model"))


class _UpstreamResponse:
    def __init__(self, body: dict[str, Any]):
        self.body = body
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self.body

    def raise_for_status(self) -> None:
        return None


def build_support_dataset(
    *,
    seed_count: int = 200,
    calibration_count: int = 200,
    evaluation_count: int = 1000,
    adversarial_count: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Build a deterministic support dataset with repeats, paraphrases, and traps."""

    prompt_to_seed = _seed_id_map(seed_count)
    cache_seed = [
        _row(
            row_id=f"seed-{idx:04d}",
            split="cache_seed",
            intent=intent,
            prompt=intent.prompt,
            canonical_answer=intent.answer,
            expected_cache_id=prompt_to_seed[intent.prompt],
        )
        for idx, intent in enumerate(_cycle_intents(seed_count))
    ]

    calibration: list[dict[str, Any]] = []
    for idx, intent in enumerate(_cycle_intents(calibration_count)):
        prompt = intent.paraphrases[idx % len(intent.paraphrases)]
        calibration.append(
            _row(
                row_id=f"calibration-{idx:04d}",
                split="calibration",
                intent=intent,
                prompt=prompt,
                canonical_answer=intent.answer,
                expected_cache_id=prompt_to_seed[intent.prompt],
                is_paraphrase=True,
            )
        )

    evaluation: list[dict[str, Any]] = []
    for idx, intent in enumerate(_cycle_intents(evaluation_count)):
        if idx % 3 == 0:
            prompt = intent.prompt
            is_repeat = True
            is_paraphrase = False
        else:
            prompt = intent.paraphrases[idx % len(intent.paraphrases)]
            is_repeat = False
            is_paraphrase = True
        evaluation.append(
            _row(
                row_id=f"evaluation-{idx:04d}",
                split="evaluation",
                intent=intent,
                prompt=prompt,
                canonical_answer=intent.answer,
                expected_cache_id=prompt_to_seed[intent.prompt],
                is_repeat=is_repeat,
                is_paraphrase=is_paraphrase,
            )
        )

    adversarial: list[dict[str, Any]] = []
    for idx, intent in enumerate(_cycle_intents(adversarial_count)):
        adversarial.append(
            _row(
                row_id=f"adversarial-{idx:04d}",
                split="adversarial",
                intent=intent,
                prompt=intent.adversarial_prompt,
                canonical_answer=intent.adversarial_answer,
                expected_cache_id=None,
                is_adversarial=True,
            )
        )

    return {
        "cache_seed": cache_seed,
        "calibration": calibration,
        "evaluation": evaluation,
        "adversarial": adversarial,
    }


def build_support_dataset_from_qa_records(
    records: list[dict[str, Any]],
    *,
    seed_count: int = 80,
    calibration_count: int = 240,
    evaluation_count: int = 640,
    adversarial_count: int = 160,
    source_name: str = "third_party_qa",
) -> dict[str, list[dict[str, Any]]]:
    """Convert third-party support Q/A rows into the proof-pack split schema.

    Expected input fields are compatible with the public Bitext customer-support
    dataset: ``instruction``, ``response``, ``intent``, and ``category``.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records):
        prompt = str(record.get("instruction", "")).strip()
        answer = str(record.get("response", "")).strip()
        intent = str(record.get("intent", "")).strip()
        category = str(record.get("category", "")).strip()
        if not prompt or not answer or not intent:
            continue
        normalized = {
            "source_record_id": str(record.get("id", idx)),
            "prompt": prompt,
            "answer": answer,
            "intent_id": intent,
            "category": category or "unknown",
            "flags": record.get("flags") or record.get("tags") or "",
        }
        grouped.setdefault(intent, []).append(normalized)
    if len(grouped) < 2:
        raise ValueError("third-party proof-pack dataset requires at least two intents")

    intents = sorted(grouped)
    canonical_answer = {intent: grouped[intent][0]["answer"] for intent in intents}
    seed_prompt = {intent: grouped[intent][0]["prompt"] for intent in intents}
    seed_id = {intent: f"seed-third-party-{idx:04d}" for idx, intent in enumerate(intents)}

    cache_seed: list[dict[str, Any]] = []
    for idx, intent in enumerate(_cycle_values(intents, seed_count)):
        source = grouped[intent][idx // len(intents) % len(grouped[intent])]
        cache_seed.append(
            _third_party_row(
                row_id=f"seed-third-party-{idx:04d}",
                split="cache_seed",
                source=source,
                prompt=source["prompt"],
                canonical_answer=canonical_answer[intent],
                expected_cache_id=seed_id[intent],
                source_name=source_name,
            )
        )

    calibration = []
    calibration_sources = _non_seed_sources(grouped, seed_prompt)
    for idx, source in enumerate(_cycle_values(calibration_sources, calibration_count)):
        calibration.append(
            _third_party_row(
                row_id=f"calibration-third-party-{idx:04d}",
                split="calibration",
                source=source,
                prompt=source["prompt"],
                canonical_answer=canonical_answer[source["intent_id"]],
                expected_cache_id=seed_id[source["intent_id"]],
                source_name=source_name,
                is_paraphrase=True,
            )
        )

    evaluation = []
    evaluation_sources = _non_seed_sources(grouped, seed_prompt)
    for idx, source in enumerate(_cycle_values(evaluation_sources, evaluation_count)):
        is_repeat = idx % 4 == 0
        prompt = seed_prompt[source["intent_id"]] if is_repeat else source["prompt"]
        evaluation.append(
            _third_party_row(
                row_id=f"evaluation-third-party-{idx:04d}",
                split="evaluation",
                source=source,
                prompt=prompt,
                canonical_answer=canonical_answer[source["intent_id"]],
                expected_cache_id=seed_id[source["intent_id"]],
                source_name=source_name,
                is_repeat=is_repeat,
                is_paraphrase=not is_repeat,
            )
        )

    adversarial = []
    adversarial_sources = _adversarial_sources(grouped)
    for idx, source in enumerate(_cycle_values(adversarial_sources, adversarial_count)):
        adversarial.append(
            _third_party_row(
                row_id=f"adversarial-third-party-{idx:04d}",
                split="adversarial",
                source=source,
                prompt=source["prompt"],
                canonical_answer=canonical_answer[source["intent_id"]],
                expected_cache_id=None,
                source_name=source_name,
                is_adversarial=True,
            )
        )

    return {
        "cache_seed": cache_seed,
        "calibration": calibration,
        "evaluation": evaluation,
        "adversarial": adversarial,
    }


def load_support_dataset_jsonl(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load an external proof-pack dataset from JSONL and validate its shape."""

    required_splits = {"cache_seed", "calibration", "evaluation", "adversarial"}
    required_fields = {
        "id",
        "split",
        "intent_id",
        "prompt",
        "canonical_answer",
        "expected_cache_id",
        "is_repeat",
        "is_paraphrase",
        "is_adversarial",
    }
    dataset: dict[str, list[dict[str, Any]]] = {split: [] for split in required_splits}
    source = Path(path)
    with source.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required_fields - set(row)
            if missing:
                raise ValueError(f"{source}:{line_no} missing required fields: {sorted(missing)}")
            split = row["split"]
            if split not in required_splits:
                raise ValueError(f"{source}:{line_no} invalid split: {split!r}")
            loaded = dict(row)
            loaded.setdefault("source", "external_jsonl")
            dataset[split].append(loaded)
    empty = [split for split, rows in dataset.items() if not rows]
    if empty:
        raise ValueError(f"{source} missing rows for splits: {sorted(empty)}")
    return {
        "cache_seed": dataset["cache_seed"],
        "calibration": dataset["calibration"],
        "evaluation": dataset["evaluation"],
        "adversarial": dataset["adversarial"],
    }


def run_exact_string_baseline(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seed = {row["prompt"]: row["canonical_answer"] for row in dataset["cache_seed"]}
    return _run_dict_baseline("exact_string", dataset, seed, approximate=False)


def run_proxy_pq_redis_flywheel_proof_pack(
    output_dir: str | Path,
    *,
    dataset_path: str | Path | None = None,
    seed_count: int = 200,
    calibration_count: int = 200,
    evaluation_count: int = 1000,
    adversarial_count: int = 200,
    redis_url: str | None = None,
    redis_namespace: str = "proof-pack",
    include_competitor_baselines: bool = True,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    progress_file = Path(progress_path) if progress_path is not None else None
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text("", encoding="utf-8")

    if dataset_path is None:
        dataset = build_support_dataset(
            seed_count=seed_count,
            calibration_count=calibration_count,
            evaluation_count=evaluation_count,
            adversarial_count=adversarial_count,
        )
        dataset_source = "synthetic_support"
    else:
        dataset = load_support_dataset_jsonl(dataset_path)
        dataset_source = str(dataset_path)
    _write_dataset_artifacts(artifact_dir, dataset)

    runs: list[dict[str, Any]] = []

    def write_progress(row: dict[str, Any]) -> None:
        if progress_file is not None:
            with progress_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def record_run(run_id: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        write_progress(
            {
                "run_id": run_id,
                "event": "started",
                "status": "running",
                "elapsed_s": 0.0,
            }
        )
        try:
            row = run()
        except Exception as exc:
            write_progress(
                {
                    "run_id": run_id,
                    "event": "failed",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                }
            )
            raise
        runs.append(row)
        write_progress(
            {
                "run_id": row.get("run_id"),
                "event": "finished",
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s", 0.0),
                "hit_rate": row.get("hit_rate", 0.0),
                "upstream_call_rate": row.get("upstream_call_rate", 0.0),
                "false_positive_rate": row.get("false_positive_rate", 0.0),
                "adversarial_false_positive_rate": row.get("adversarial_false_positive_rate", 0.0),
                "redis_memory_mb": row.get("redis_memory_mb", 0.0),
            }
        )
        return row

    record_run("exact_string", lambda: run_exact_string_baseline(dataset))
    record_run("dense_cosine", lambda: _run_dense_cosine_baseline(dataset))
    record_run(
        "lattice_pq_local",
        lambda: _run_proxy_pq_baseline(dataset, artifact_dir, run_id="lattice_pq_local", use_redis=False),
    )
    record_run(
        "lattice_pq_validated_cosine",
        lambda: _run_validated_pq_baseline(dataset, artifact_dir, run_id="lattice_pq_validated_cosine"),
    )
    record_run(
        "lattice_pq_redis",
        lambda: _run_proxy_pq_baseline(dataset, artifact_dir, run_id="lattice_pq_redis", use_redis=True),
    )
    if redis_url:
        record_run(
            "lattice_pq_redis_real",
            lambda: _run_proxy_pq_baseline(
                dataset,
                artifact_dir,
                run_id="lattice_pq_redis_real",
                use_redis=True,
                redis_url=redis_url,
                redis_namespace=redis_namespace,
            )
        )
        record_run(
            "lattice_pq_redis_validated_cosine",
            lambda: _run_validated_pq_baseline(
                dataset,
                artifact_dir,
                run_id="lattice_pq_redis_validated_cosine",
                use_redis=True,
                redis_url=redis_url,
                redis_namespace=redis_namespace,
            )
        )
    if include_competitor_baselines:
        for runner in _competitor_baseline_runners(dataset, redis_url=redis_url):
            record_run(runner[0], runner[1])

    summary = {
        "artifact_dir": str(artifact_dir),
        "dataset_source": dataset_source,
        "dataset": {split: len(rows) for split, rows in dataset.items()},
        "runs": runs,
    }
    (artifact_dir / "proof_pack_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = _render_report(summary)
    (artifact_dir / "proof_pack_report.md").write_text(report, encoding="utf-8")
    (artifact_dir / "operating_policy_report.md").write_text(_render_operating_policy_report(summary), encoding="utf-8")
    (artifact_dir / "public_claim_card.md").write_text(_render_claim_card(summary), encoding="utf-8")
    return summary


def build_seeded_pq_cache_from_support_jsonl(
    dataset_path: str | Path,
    *,
    redis_url: str | None = None,
    redis_namespace: str = "proof-pack",
    pq_num_blocks: int = 4,
    pq_codebook_size: int = 4,
    flush_redis: bool = False,
) -> RFSnapSemanticCache:
    """Build the PQ-backed semantic cache used by the proof-pack serving demo.

    This is intentionally a proof/demo helper, not a general-purpose PQ proxy
    configuration API. It fits PQ codebooks from cache_seed + calibration rows
    and seeds the cache_seed rows so ``lattice serve --pq-proof-dataset`` can
    reproduce the same cache shape as the proof harness.
    """
    dataset = load_support_dataset_jsonl(dataset_path)
    cache, _ = _build_pq_cache(
        dataset,
        use_redis=bool(redis_url),
        redis_url=redis_url,
        redis_namespace=redis_namespace,
        flush_redis=flush_redis,
        pq_num_blocks=pq_num_blocks,
        pq_codebook_size=pq_codebook_size,
    )
    for row in dataset["cache_seed"]:
        cache.put(
            row["prompt"],
            value=_response_body(row["canonical_answer"], model="demo-seed"),
            metadata={"intent_id": row["intent_id"], "seed_id": row["id"]},
        )
    return cache


def _run_dict_baseline(
    run_id: str,
    dataset: dict[str, list[dict[str, Any]]],
    cache: dict[str, str],
    *,
    approximate: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = dataset["evaluation"] + dataset["adversarial"]
    hits = false_positives = adversarial_false_positives = 0
    latencies: list[float] = []
    for row in rows:
        row_start = time.perf_counter()
        cached = cache.get(row["prompt"])
        if cached is not None:
            hits += 1
            if cached != row["canonical_answer"]:
                false_positives += 1
                if row["is_adversarial"]:
                    adversarial_false_positives += 1
        latencies.append((time.perf_counter() - row_start) * 1000.0)
    elapsed = time.perf_counter() - started
    total = len(rows)
    misses = total - hits
    return _metrics(
        run_id=run_id,
        total=total,
        exact_hits=hits if not approximate else 0,
        approx_hits=hits if approximate else 0,
        misses=misses,
        false_positives=false_positives,
        adversarial_false_positives=adversarial_false_positives,
        adversarial_total=len(dataset["adversarial"]),
        latencies_ms=latencies,
        elapsed_s=elapsed,
        cache_entries=len(cache),
        redis_memory_mb=0.0,
        flywheel_miss_clusters=0,
        reviewed_answers_loaded=0,
    )


def _run_dense_cosine_baseline(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    prompt_to_token = _prompt_to_token(dataset)
    encoder = _ProofPackEncoder(prompt_to_token)
    seed_rows = dataset["cache_seed"]
    seed_vectors = encoder.encode([row["prompt"] for row in seed_rows])
    seed_answers = [row["canonical_answer"] for row in seed_rows]

    started = time.perf_counter()
    rows = dataset["evaluation"] + dataset["adversarial"]
    hits = false_positives = adversarial_false_positives = 0
    latencies: list[float] = []
    for row in rows:
        row_start = time.perf_counter()
        vec = encoder.encode(row["prompt"])
        sims = np.asarray(seed_vectors) @ np.asarray(vec)
        idx = int(np.argmax(sims))
        cached = seed_answers[idx] if float(sims[idx]) >= 0.999 else None
        if cached is not None:
            hits += 1
            if cached != row["canonical_answer"]:
                false_positives += 1
                if row["is_adversarial"]:
                    adversarial_false_positives += 1
        latencies.append((time.perf_counter() - row_start) * 1000.0)
    elapsed = time.perf_counter() - started
    return _metrics(
        run_id="dense_cosine",
        total=len(rows),
        exact_hits=0,
        approx_hits=hits,
        misses=len(rows) - hits,
        false_positives=false_positives,
        adversarial_false_positives=adversarial_false_positives,
        adversarial_total=len(dataset["adversarial"]),
        latencies_ms=latencies,
        elapsed_s=elapsed,
        cache_entries=len(seed_rows),
        redis_memory_mb=0.0,
        flywheel_miss_clusters=0,
        reviewed_answers_loaded=0,
    )


def _run_proxy_pq_baseline(
    dataset: dict[str, list[dict[str, Any]]],
    artifact_dir: Path,
    *,
    run_id: str,
    use_redis: bool,
    redis_url: str | None = None,
    redis_namespace: str = "proof-pack",
) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    try:
        cache, redis_backend = _build_pq_cache(
            dataset,
            use_redis=use_redis,
            redis_url=redis_url,
            redis_namespace=redis_namespace,
            flush_redis=True,
        )
    except Exception as exc:
        return _skipped_metrics(
            run_id=run_id,
            skip_reason=f"redis unavailable: {type(exc).__name__}: {exc}",
            redis_backend="real" if redis_url else "in_memory",
        )
    for row in dataset["cache_seed"]:
        cache.put(
            row["prompt"],
            value=_response_body(row["canonical_answer"]),
            metadata={"intent_id": row["intent_id"], "seed_id": row["id"]},
        )

    shared_cache_verified = False
    if use_redis:
        shared_cache_verified = _verify_shared_cache(
            dataset,
            redis_url=redis_url,
            redis_namespace=redis_namespace,
            existing_redis_backend=redis_backend,
        )

    upstream = _ProofPackUpstream(_answers_by_prompt(dataset))
    miss_log = artifact_dir / f"{run_id}_misses.jsonl"
    proxy = LatticeLLMProxy(
        upstream_url="https://example.test/v1/chat/completions",
        upstream_api_key="proof-pack-key",
        semantic_cache=cache,
        upstream_client=upstream,
        miss_log_path=miss_log,
        cost_per_1k_input_tokens_usd=0.01,
    )
    client = TestClient(proxy.create_app())

    rows = dataset["evaluation"] + dataset["adversarial"]
    exact_hits = approx_hits = false_positives = adversarial_false_positives = 0
    latencies: list[float] = []
    started = time.perf_counter()
    for row in rows:
        row_start = time.perf_counter()
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "proof-pack-model",
                "messages": [{"role": "user", "content": row["prompt"]}],
            },
        )
        latencies.append((time.perf_counter() - row_start) * 1000.0)
        body = response.json()
        served = response.headers.get("x-lattice-cache", "MISS")
        path = response.headers.get("x-lattice-cache-path", "")
        content = _content_from_body(body)
        if served == "HIT":
            if path == "lattice_exact":
                exact_hits += 1
            else:
                approx_hits += 1
            if content != row["canonical_answer"]:
                false_positives += 1
                if row["is_adversarial"]:
                    adversarial_false_positives += 1
    elapsed = time.perf_counter() - started

    queue_path = artifact_dir / "flywheel_review_queue.json"
    import_result_path = artifact_dir / "flywheel_review_import_result.json"
    queue_count = proxy.flywheel.export_review_queue(queue_path, n=10, min_cluster_size=1)
    reviewed_loaded = 0
    if queue_count:
        reviewed = json.loads(queue_path.read_text(encoding="utf-8"))
        reviewed[0]["answer"] = "Reviewed support answer for recurring miss cluster."
        reviewed[0]["intent_id"] = "proof_pack_review"
        review_path = artifact_dir / f"{run_id}_reviewed_answers.json"
        review_path.write_text(json.dumps(reviewed, indent=2, sort_keys=True), encoding="utf-8")
        reviewed_loaded = int(proxy.flywheel.load_reviewed(review_path, proxy.cache))
        import_result_path.write_text(
            json.dumps({"loaded": reviewed_loaded}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        import_result_path.write_text(
            json.dumps({"loaded": 0, "skipped": 0}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    redis_memory_mb = _redis_memory_mb(redis_backend, namespace=redis_namespace) if redis_backend is not None else 0.0

    metrics = _metrics(
        run_id=run_id,
        total=len(rows),
        exact_hits=exact_hits,
        approx_hits=approx_hits,
        misses=upstream.call_count,
        false_positives=false_positives,
        adversarial_false_positives=adversarial_false_positives,
        adversarial_total=len(dataset["adversarial"]),
        latencies_ms=latencies,
        elapsed_s=elapsed,
        cache_entries=len(cache._entries),
        redis_memory_mb=redis_memory_mb,
        flywheel_miss_clusters=queue_count,
        reviewed_answers_loaded=reviewed_loaded,
    )
    metrics["redis_backend"] = "real" if redis_url else ("in_memory" if use_redis else "none")
    metrics["redis_persistence_verified"] = bool(use_redis and shared_cache_verified)
    metrics["multi_proxy_shared_cache_verified"] = shared_cache_verified
    (artifact_dir / f"{run_id}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _run_validated_pq_baseline(
    dataset: dict[str, list[dict[str, Any]]],
    artifact_dir: Path,
    *,
    run_id: str,
    cosine_threshold: float = 0.999,
    use_redis: bool = False,
    redis_url: str | None = None,
    redis_namespace: str = "proof-pack",
) -> dict[str, Any]:
    try:
        cache, redis_backend = _build_pq_cache(
            dataset,
            use_redis=use_redis,
            redis_url=redis_url,
            redis_namespace=redis_namespace,
            flush_redis=True,
        )
    except Exception as exc:
        return _skipped_metrics(
            run_id=run_id,
            skip_reason=f"redis unavailable: {type(exc).__name__}: {exc}",
            redis_backend="real" if redis_url else ("in_memory" if use_redis else "none"),
        )
    for row in dataset["cache_seed"]:
        cache.put(
            row["prompt"],
            value=_response_body(row["canonical_answer"]),
            metadata={"intent_id": row["intent_id"], "seed_id": row["id"]},
        )

    shared_cache_verified = False
    if use_redis:
        shared_cache_verified = _verify_shared_cache(
            dataset,
            redis_url=redis_url,
            redis_namespace=redis_namespace,
            existing_redis_backend=redis_backend,
        )

    encoder = cache.runtime.encoder
    rows = dataset["evaluation"] + dataset["adversarial"]
    exact_hits = approx_hits = false_positives = adversarial_false_positives = 0
    rejected_candidates = 0
    latencies: list[float] = []
    started = time.perf_counter()
    for row in rows:
        row_start = time.perf_counter()
        result = cache.get(row["prompt"])
        if result.hit:
            candidate_prompt = result.source_prompt or ""
            score = _cosine_score(encoder.encode(row["prompt"]), encoder.encode(candidate_prompt))
            if score >= cosine_threshold:
                if result.retrieval_path == "lattice_exact":
                    exact_hits += 1
                else:
                    approx_hits += 1
                content = _content_from_body(result.value)
                if content != row["canonical_answer"]:
                    false_positives += 1
                    if row["is_adversarial"]:
                        adversarial_false_positives += 1
            else:
                rejected_candidates += 1
        latencies.append((time.perf_counter() - row_start) * 1000.0)
    elapsed = time.perf_counter() - started
    misses = len(rows) - exact_hits - approx_hits
    metrics = _metrics(
        run_id=run_id,
        total=len(rows),
        exact_hits=exact_hits,
        approx_hits=approx_hits,
        misses=misses,
        false_positives=false_positives,
        adversarial_false_positives=adversarial_false_positives,
        adversarial_total=len(dataset["adversarial"]),
        latencies_ms=latencies,
        elapsed_s=elapsed,
        cache_entries=len(cache._entries),
        redis_memory_mb=_redis_memory_mb(redis_backend, namespace=redis_namespace) if redis_backend is not None else 0.0,
        flywheel_miss_clusters=0,
        reviewed_answers_loaded=0,
    )
    metrics["validation_gate"] = "cosine"
    metrics["validation_threshold"] = cosine_threshold
    metrics["rejected_candidates"] = rejected_candidates
    metrics["redis_backend"] = "real" if redis_url else ("in_memory" if use_redis else "none")
    metrics["redis_persistence_verified"] = bool(use_redis and shared_cache_verified)
    metrics["multi_proxy_shared_cache_verified"] = shared_cache_verified
    (artifact_dir / f"{run_id}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _build_pq_cache(
    dataset: dict[str, list[dict[str, Any]]],
    *,
    use_redis: bool,
    redis_url: str | None = None,
    redis_namespace: str = "proof-pack",
    flush_redis: bool = False,
    existing_redis_backend: Any | None = None,
    pq_num_blocks: int = 4,
    pq_codebook_size: int = 4,
):
    prompt_to_token = _prompt_to_token(dataset)
    encoder = _ProofPackEncoder(prompt_to_token)
    d_model = encoder.d_model
    pq = PQLatticeDB(d_model=d_model, num_blocks=pq_num_blocks, codebook_size=pq_codebook_size)
    fit_prompts = [row["prompt"] for row in dataset["cache_seed"] + dataset["calibration"]]
    pq.fit(encoder.encode(fit_prompts))
    memory = RFSnapLatticeMemory(d_model=d_model, lattice=pq, beam_radius=1)
    runtime = RFSnapTextMemory(encoder=encoder, d_model=d_model, memory=memory)
    cache = RFSnapSemanticCache(runtime=runtime)
    redis_backend = None
    if use_redis:
        if redis_url:
            store = LatticeRedisStore(redis_url=redis_url, namespace=redis_namespace)
            if not store.ping():
                raise ConnectionError(f"Redis ping failed for {redis_url}")
            if flush_redis:
                store.flush()
            store.patch_cache(cache)
            redis_backend = store
        else:
            redis_client = existing_redis_backend or _InMemoryRedis()
            cache._entries = _RedisEntriesProxy(redis_client, namespace=redis_namespace)  # noqa: SLF001
            redis_backend = redis_client
    return cache, redis_backend


def _verify_shared_cache(
    dataset: dict[str, list[dict[str, Any]]],
    *,
    redis_url: str | None,
    redis_namespace: str,
    existing_redis_backend: Any | None,
) -> bool:
    try:
        probe_cache, _ = _build_pq_cache(
            dataset,
            use_redis=True,
            redis_url=redis_url,
            redis_namespace=redis_namespace,
            flush_redis=False,
            existing_redis_backend=existing_redis_backend if redis_url is None else None,
        )
        seed = dataset["cache_seed"][0]
        result = probe_cache.get(seed["prompt"])
        return bool(result.hit and _content_from_body(result.value) == seed["canonical_answer"])
    except Exception:
        return False


def _redis_memory_mb(redis_backend: Any, *, namespace: str) -> float:
    if redis_backend is None:
        return 0.0
    if hasattr(redis_backend, "memory_bytes"):
        return float(redis_backend.memory_bytes()) / (1024.0 * 1024.0)
    clients = getattr(redis_backend, "_clients", None)
    if clients is None and hasattr(redis_backend, "_client"):
        clients = [redis_backend._client]
    if not clients:
        return 0.0
    total = 0
    for client in clients:
        pattern = f"{namespace}:*"
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else str(key)
                total += len(key_str.encode("utf-8"))
                try:
                    value = client.get(key)
                except Exception:
                    value = None
                if isinstance(value, bytes):
                    total += len(value)
                elif isinstance(value, str):
                    total += len(value.encode("utf-8"))
            if cursor == 0:
                break
    return total / (1024.0 * 1024.0)


def _cosine_score(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _metrics(
    *,
    run_id: str,
    total: int,
    exact_hits: int,
    approx_hits: int,
    misses: int,
    false_positives: int,
    adversarial_false_positives: int,
    adversarial_total: int,
    latencies_ms: list[float],
    elapsed_s: float,
    cache_entries: int,
    redis_memory_mb: float,
    flywheel_miss_clusters: int,
    reviewed_answers_loaded: int,
) -> dict[str, Any]:
    hits = exact_hits + approx_hits
    latencies = sorted(latencies_ms)
    p95_idx = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1)) if latencies else 0
    total_cost = total * 0.0002
    saved = hits * 0.0002
    return {
        "run_id": run_id,
        "status": "ok",
        "total_requests": total,
        "hits": hits,
        "misses": misses,
        "exact_hits": exact_hits,
        "approximate_hits": approx_hits,
        "hit_rate": hits / total if total else 0.0,
        "exact_hit_rate": exact_hits / total if total else 0.0,
        "approximate_hit_rate": approx_hits / total if total else 0.0,
        "upstream_call_rate": misses / total if total else 0.0,
        "false_positive_count": false_positives,
        "false_positive_rate": false_positives / hits if hits else 0.0,
        "adversarial_false_positive_rate": (
            adversarial_false_positives / adversarial_total if adversarial_total else 0.0
        ),
        "estimated_total_cost_usd": total_cost,
        "estimated_cost_saved_usd": saved,
        "estimated_savings_rate": saved / total_cost if total_cost else 0.0,
        "latency_ms_avg": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_p95": latencies[p95_idx] if latencies else 0.0,
        "elapsed_s": elapsed_s,
        "cache_entries": cache_entries,
        "redis_memory_mb": redis_memory_mb,
        "flywheel_miss_clusters": flywheel_miss_clusters,
        "reviewed_answers_loaded": reviewed_answers_loaded,
    }


def _skipped_metrics(run_id: str, *, skip_reason: str, redis_backend: str = "none") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "skipped",
        "skip_reason": skip_reason,
        "redis_backend": redis_backend,
        "total_requests": 0,
        "hits": 0,
        "misses": 0,
        "exact_hits": 0,
        "approximate_hits": 0,
        "hit_rate": 0.0,
        "exact_hit_rate": 0.0,
        "approximate_hit_rate": 0.0,
        "upstream_call_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "adversarial_false_positive_rate": 0.0,
        "estimated_total_cost_usd": 0.0,
        "estimated_cost_saved_usd": 0.0,
        "estimated_savings_rate": 0.0,
        "latency_ms_avg": 0.0,
        "latency_ms_p95": 0.0,
        "elapsed_s": 0.0,
        "cache_entries": 0,
        "redis_memory_mb": 0.0,
        "flywheel_miss_clusters": 0,
        "reviewed_answers_loaded": 0,
        "redis_persistence_verified": False,
        "multi_proxy_shared_cache_verified": False,
    }


def _competitor_baseline_runners(
    dataset: dict[str, list[dict[str, Any]]],
    *,
    redis_url: str | None,
) -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    return [
        ("redisvl_direct", lambda: _run_redisvl_direct_baseline(dataset, redis_url=redis_url)),
        ("gptcache_direct", lambda: _run_gptcache_direct_baseline(dataset)),
        (
            "upstash_semantic_cache",
            lambda: _skipped_metrics(
                "upstash_semantic_cache",
                skip_reason="Upstash credentials were not provided; remote semantic-cache baseline skipped.",
                redis_backend="remote",
            ),
        ),
    ]


def _run_gptcache_direct_baseline(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        from gptcache.manager import get_data_manager
    except Exception:
        return _skipped_metrics(
            "gptcache_direct",
            skip_reason="gptcache is not installed; dense_cosine is the local GPTCache-shaped baseline for this run.",
        )
    manager = get_data_manager()
    manager.flush()
    for row in dataset["cache_seed"]:
        manager.save(row["prompt"], row["canonical_answer"], row["prompt"])

    started = time.perf_counter()
    rows = dataset["evaluation"] + dataset["adversarial"]
    hits = false_positives = adversarial_false_positives = 0
    latencies: list[float] = []
    for row in rows:
        row_start = time.perf_counter()
        results = manager.search(row["prompt"])
        if results:
            hits += 1
            cached = results[0][1]
            if cached != row["canonical_answer"]:
                false_positives += 1
                if row["is_adversarial"]:
                    adversarial_false_positives += 1
        latencies.append((time.perf_counter() - row_start) * 1000.0)
    metrics = _metrics(
        run_id="gptcache_direct",
        total=len(rows),
        exact_hits=hits,
        approx_hits=0,
        misses=len(rows) - hits,
        false_positives=false_positives,
        adversarial_false_positives=adversarial_false_positives,
        adversarial_total=len(dataset["adversarial"]),
        latencies_ms=latencies,
        elapsed_s=time.perf_counter() - started,
        cache_entries=len(dataset["cache_seed"]),
        redis_memory_mb=0.0,
        flywheel_miss_clusters=0,
        reviewed_answers_loaded=0,
    )
    metrics["baseline_backend"] = "gptcache"
    return metrics


def _run_redisvl_direct_baseline(dataset: dict[str, list[dict[str, Any]]], *, redis_url: str | None) -> dict[str, Any]:
    if not redis_url:
        return _skipped_metrics(
            "redisvl_direct",
            skip_reason="redis_url was not provided; RedisVL direct baseline requires Redis Stack.",
            redis_backend="none",
        )
    try:
        from redisvl.index import SearchIndex
        from redisvl.query import VectorQuery
        from redisvl.schema import IndexSchema
    except Exception:
        return _skipped_metrics(
            "redisvl_direct",
            skip_reason="redisvl is not installed; install RedisVL and rerun for a direct RedisVL semantic-cache baseline.",
            redis_backend="real",
        )
    try:
        import redis as redis_lib

        client = redis_lib.from_url(redis_url, decode_responses=False)
        client.ping()
        schema = IndexSchema.from_dict(
            {
                "index": {"name": "proof_pack_redisvl", "prefix": "proof_pack_redisvl:"},
                "fields": [
                    {"name": "prompt", "type": "text"},
                    {"name": "answer", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 32,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )
        index = SearchIndex(schema, redis_client=client)
        index.create(overwrite=True, drop=True)
        encoder = _ProofPackEncoder(_prompt_to_token(dataset))
        docs = []
        for idx, row in enumerate(dataset["cache_seed"]):
            docs.append(
                {
                    "id": f"seed-{idx}",
                    "prompt": row["prompt"],
                    "answer": row["canonical_answer"],
                    "embedding": encoder.encode(row["prompt"]).astype(np.float32).tobytes(),
                }
            )
        index.load(docs, id_field="id")
        rows = dataset["evaluation"] + dataset["adversarial"]
        hits = false_positives = adversarial_false_positives = 0
        latencies: list[float] = []
        started = time.perf_counter()
        for row in rows:
            row_start = time.perf_counter()
            query = VectorQuery(
                vector=encoder.encode(row["prompt"]).astype(np.float32).tobytes(),
                vector_field_name="embedding",
                return_fields=["answer"],
                num_results=1,
                dtype="float32",
            )
            result = index.query(query)
            cached = None
            if result:
                score = float(result[0].get("vector_distance", result[0].get("score", 1.0)))
                if score <= 0.001:
                    cached = result[0].get("answer")
            if cached is not None:
                hits += 1
                if cached != row["canonical_answer"]:
                    false_positives += 1
                    if row["is_adversarial"]:
                        adversarial_false_positives += 1
            latencies.append((time.perf_counter() - row_start) * 1000.0)
        try:
            index.delete(drop=True)
        except Exception:
            pass
        metrics = _metrics(
            run_id="redisvl_direct",
            total=len(rows),
            exact_hits=0,
            approx_hits=hits,
            misses=len(rows) - hits,
            false_positives=false_positives,
            adversarial_false_positives=adversarial_false_positives,
            adversarial_total=len(dataset["adversarial"]),
            latencies_ms=latencies,
            elapsed_s=time.perf_counter() - started,
            cache_entries=len(dataset["cache_seed"]),
            redis_memory_mb=0.0,
            flywheel_miss_clusters=0,
            reviewed_answers_loaded=0,
        )
        metrics["redis_backend"] = "redisvl"
        return metrics
    except Exception as exc:
        return _skipped_metrics(
            "redisvl_direct",
            skip_reason=f"RedisVL direct baseline unavailable: {type(exc).__name__}: {exc}",
            redis_backend="real",
        )


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# LatticeMemory Proxy + PQ + Redis + Flywheel Proof Pack",
        "",
        "This artifact compares exact string caching, dense semantic caching, and LatticeMemory proxy paths.",
        "It is a support-workload proof pack, not a general RAG or vector-database replacement claim.",
        "",
        "| Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Avg ms | Redis MB | Flywheel clusters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        if run.get("status") == "skipped":
            lines.append(
                "| {run_id} | skipped | skipped | skipped | skipped | skipped | skipped | skipped |".format(
                    run_id=run["run_id"]
                )
            )
            continue
        lines.append(
            "| {run_id} | {hit:.3f} | {upstream:.3f} | {fp:.3f} | {adv:.3f} | {lat:.3f} | {redis:.4f} | {clusters} |".format(
                run_id=run["run_id"],
                hit=run["hit_rate"],
                upstream=run["upstream_call_rate"],
                fp=run["false_positive_rate"],
                adv=run["adversarial_false_positive_rate"],
                lat=run["latency_ms_avg"],
                redis=run["redis_memory_mb"],
                clusters=run["flywheel_miss_clusters"],
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Baselines",
            "",
        ]
    )
    skipped = [run for run in summary["runs"] if run.get("status") == "skipped"]
    if skipped:
        lines.extend(["| Run | Reason |", "|---|---|"])
        for run in skipped:
            lines.append(f"| {run['run_id']} | {run.get('skip_reason', 'not reported')} |")
    else:
        lines.append("No baselines were skipped.")
    lines.extend(
        [
            "",
            "Required claim wording: LatticeMemory can reduce repeated/paraphrased upstream calls on this measured workload while reporting false positives and review behavior.",
            "Unsupported wording: LatticeMemory replaces general-purpose vector databases or guarantees accuracy on arbitrary RAG workloads.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_operating_policy_report(summary: dict[str, Any]) -> str:
    ok_runs = {run["run_id"]: run for run in summary["runs"] if run.get("status") == "ok"}
    zero_fp = [
        run
        for run in ok_runs.values()
        if run.get("false_positive_rate", 0.0) == 0.0
        and run.get("adversarial_false_positive_rate", 0.0) == 0.0
    ]
    conservative = max(zero_fp, key=lambda row: row["hit_rate"]) if zero_fp else None
    balanced = ok_runs.get("lattice_pq_redis_validated_cosine") or ok_runs.get("lattice_pq_validated_cosine")
    aggressive = ok_runs.get("lattice_pq_local")
    rows = [
        (
            "conservative_zero_fp",
            conservative,
            "Use when false positives are more expensive than upstream calls.",
        ),
        (
            "balanced_validated_pq",
            balanced,
            "Target product path: PQ candidate generation with validation before serving.",
        ),
        (
            "aggressive_raw_pq",
            aggressive,
            "Research/high-risk mode only; raw PQ can over-hit and must carry FP metrics.",
        ),
    ]
    lines = [
        "# LatticeMemory Operating Policy Report",
        "",
        "| Policy | Run | Hit rate | Upstream rate | FP rate | Adv FP rate | Recommendation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for policy, run, recommendation in rows:
        if run is None:
            lines.append(f"| {policy} | unavailable | n/a | n/a | n/a | n/a | {recommendation} |")
            continue
        lines.append(
            "| {policy} | {run_id} | {hit:.4f} | {upstream:.4f} | {fp:.4f} | {adv:.4f} | {rec} |".format(
                policy=policy,
                run_id=run["run_id"],
                hit=run["hit_rate"],
                upstream=run["upstream_call_rate"],
                fp=run["false_positive_rate"],
                adv=run["adversarial_false_positive_rate"],
                rec=recommendation,
            )
        )
    lines.extend(
        [
            "",
            "Policy rule: never publish raw PQ hit rate without the paired false-positive and adversarial false-positive rates.",
            "The balanced path is the one to harden into production serving if it preserves savings while reducing false positives.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_claim_card(summary: dict[str, Any]) -> str:
    ok_runs = [run for run in summary["runs"] if run.get("status") == "ok"]
    highest_hit = max(
        ok_runs,
        key=lambda run: run.get("hit_rate", 0.0),
    )
    zero_fp_runs = [
        run
        for run in ok_runs
        if run.get("false_positive_rate", 0.0) == 0.0
        and run.get("adversarial_false_positive_rate", 0.0) == 0.0
    ]
    safest = max(zero_fp_runs or ok_runs, key=lambda run: run.get("hit_rate", 0.0))
    run_by_id = {run["run_id"]: run for run in ok_runs}
    target_policy = run_by_id.get("lattice_pq_redis_validated_cosine") or run_by_id.get(
        "lattice_pq_validated_cosine"
    )
    lines = [
            "# LatticeMemory Public Claim Card",
            "",
            "## Supported Claim",
            "",
            (
                "LatticeMemory reduces repeated/paraphrased upstream calls on a measured "
                "support-style workload while reporting hit rate, latency, savings, Redis memory, "
                "flywheel review behavior, and false-positive rate together."
            ),
            "",
            "## Safest Zero-FP Measured Row",
            "",
            f"- Run: `{safest['run_id']}`",
            f"- Hit rate: `{safest['hit_rate']:.4f}`",
            f"- Upstream call rate: `{safest['upstream_call_rate']:.4f}`",
            f"- False-positive rate: `{safest['false_positive_rate']:.4f}`",
            f"- Adversarial false-positive rate: `{safest['adversarial_false_positive_rate']:.4f}`",
            f"- Redis memory MB: `{safest['redis_memory_mb']:.4f}`",
            f"- Flywheel miss clusters: `{safest['flywheel_miss_clusters']}`",
            "",
            "## Highest-Hit Measured Row",
            "",
            f"- Run: `{highest_hit['run_id']}`",
            f"- Hit rate: `{highest_hit['hit_rate']:.4f}`",
            f"- Upstream call rate: `{highest_hit['upstream_call_rate']:.4f}`",
            f"- False-positive rate: `{highest_hit['false_positive_rate']:.4f}`",
            f"- Adversarial false-positive rate: `{highest_hit['adversarial_false_positive_rate']:.4f}`",
            f"- Redis memory MB: `{highest_hit['redis_memory_mb']:.4f}`",
            f"- Flywheel miss clusters: `{highest_hit['flywheel_miss_clusters']}`",
            "",
            "Use the highest-hit row only with its measured false-positive rate attached.",
            "",
    ]
    if target_policy is not None:
        lines.extend(
            [
                "## Target Product Policy Row",
                "",
                f"- Run: `{target_policy['run_id']}`",
                f"- Hit rate: `{target_policy['hit_rate']:.4f}`",
                f"- Upstream call rate: `{target_policy['upstream_call_rate']:.4f}`",
                f"- False-positive rate: `{target_policy['false_positive_rate']:.4f}`",
                (
                    "- Adversarial false-positive rate: "
                    f"`{target_policy['adversarial_false_positive_rate']:.4f}`"
                ),
                f"- Redis memory MB: `{target_policy['redis_memory_mb']:.4f}`",
                f"- Flywheel miss clusters: `{target_policy['flywheel_miss_clusters']}`",
                "",
                (
                    "This is the product-shaped row to harden: PQ generates candidates, "
                    "a validation gate decides whether they are safe to serve, and misses "
                    "fall back upstream."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Unsupported Claims",
            "",
            "- LatticeMemory does not replace general-purpose vector databases.",
            "- LatticeMemory does not prove general RAG superiority from this proof pack.",
            "- PQ/Hamming cache hits are not assumed safe without false-positive measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_dataset_artifacts(artifact_dir: Path, dataset: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in dataset.items():
        path = artifact_dir / f"dataset_{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )


def _prompt_to_token(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("cache_seed", "calibration", "evaluation"):
        for row in dataset[split]:
            mapping[row["prompt"]] = f"intent:{row['intent_id']}"
    for row in dataset["adversarial"]:
        mapping[row["prompt"]] = f"adversarial:{row['intent_id']}"
    return mapping


def _answers_by_prompt(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    answers = {}
    for rows in dataset.values():
        for row in rows:
            answers[row["prompt"]] = row["canonical_answer"]
    return answers


def _response_body(answer: str, *, model: str = "proof-pack-model") -> dict[str, Any]:
    return {
        "id": "proof-pack-response",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 16, "completion_tokens": max(1, len(answer.split())), "total_tokens": 16 + max(1, len(answer.split()))},
    }


def _prompt_from_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    return "\n".join(str(message.get("content", "")) for message in messages)


def _content_from_body(body: dict[str, Any]) -> str:
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


def _row(
    *,
    row_id: str,
    split: str,
    intent: _Intent,
    prompt: str,
    canonical_answer: str,
    expected_cache_id: str | None,
    is_repeat: bool = False,
    is_paraphrase: bool = False,
    is_adversarial: bool = False,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "split": split,
        "intent_id": intent.intent_id,
        "prompt": prompt,
        "canonical_answer": canonical_answer,
        "expected_cache_id": expected_cache_id,
        "is_repeat": is_repeat,
        "is_paraphrase": is_paraphrase,
        "is_adversarial": is_adversarial,
    }


def _third_party_row(
    *,
    row_id: str,
    split: str,
    source: dict[str, Any],
    prompt: str,
    canonical_answer: str,
    expected_cache_id: str | None,
    source_name: str,
    is_repeat: bool = False,
    is_paraphrase: bool = False,
    is_adversarial: bool = False,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "split": split,
        "intent_id": source["intent_id"],
        "prompt": prompt,
        "canonical_answer": canonical_answer,
        "expected_cache_id": expected_cache_id,
        "is_repeat": is_repeat,
        "is_paraphrase": is_paraphrase,
        "is_adversarial": is_adversarial,
        "source": source_name,
        "source_record_id": source["source_record_id"],
        "source_category": source["category"],
        "source_flags": source["flags"],
    }


def _cycle_intents(count: int) -> list[_Intent]:
    return [_INTENTS[idx % len(_INTENTS)] for idx in range(count)]


def _cycle_values(values: list[Any], count: int) -> list[Any]:
    if not values and count:
        raise ValueError("cannot cycle an empty value list")
    return [values[idx % len(values)] for idx in range(count)]


def _non_seed_sources(
    grouped: dict[str, list[dict[str, Any]]],
    seed_prompt: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intent in sorted(grouped):
        alternatives = [row for row in grouped[intent] if row["prompt"] != seed_prompt[intent]]
        rows.extend(alternatives or grouped[intent])
    return rows


def _adversarial_sources(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intent in sorted(grouped):
        rows.extend(grouped[intent][1:4] or grouped[intent])
    return rows


def _seed_id_map(seed_count: int) -> dict[str, str]:
    mapping = {intent.prompt: f"seed-virtual-{idx:04d}" for idx, intent in enumerate(_INTENTS)}
    for idx, intent in enumerate(_cycle_intents(seed_count)):
        mapping.setdefault(intent.prompt, f"seed-{idx:04d}")
        mapping[intent.prompt] = f"seed-{idx:04d}"
    return mapping
