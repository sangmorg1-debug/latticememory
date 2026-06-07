"""Generate a product demo dataset with NVIDIA NIM.

The generated source dataset is split into:
- calibration_data.json for threshold calibration
- heldout_paraphrases.json for held-out recall
- heldout_near_misses.json for held-out false-positive checks
- prompts_responses.json for cache-hit simulation

Requires one of NVIDIA_API_KEY, NVCF_API_KEY, NGC_API_KEY, NIM_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _api_key(extra_env: str | None = None) -> str | None:
    if extra_env:
        value = os.environ.get(extra_env)
        if value:
            return value
    for name in ("NVIDIA_API_KEY", "NVCF_API_KEY", "NGC_API_KEY", "NIM_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _dataset_prompt(n_intents: int, paraphrases_per_intent: int, near_miss_pairs: int) -> str:
    return f"""
Create a realistic product-demo dataset for testing a safe semantic cache for an
ecommerce/SaaS customer support chatbot.

Return ONLY valid JSON, no markdown.

Schema:
{{
  "domain": "customer_support_ecommerce_saas",
  "intents": [
    {{
      "intent_id": "snake_case_unique",
      "category": "billing|account|shipping|returns|subscription|security|product|support",
      "canonical_prompt": "short user question",
      "safe_answer": "concise support answer, 1-2 sentences, safe and generic",
      "paraphrases": ["natural user phrasings with the same answer intent"]
    }}
  ],
  "near_miss_pairs": [
    {{"a_intent": "intent_id", "b_intent": "different but easily confused intent_id", "reason": "why these are near misses"}}
  ]
}}

Requirements:
- Exactly {n_intents} intents.
- Exactly {paraphrases_per_intent} paraphrases per intent.
- At least {near_miss_pairs} near_miss_pairs.
- Include close-but-different pairs like cancel vs pause subscription, refund vs
  return status, reset password vs change email, invoice copy vs update payment
  method, shipping status vs delivery address change.
- Paraphrases must be realistic short user prompts, not labels.
- near_miss_pairs must only use intent ids present in intents.
- Keep all text non-sensitive and generic.
""".strip()


def call_nvidia_chat(
    *,
    api_key: str,
    model: str,
    base_url: str,
    prompt: str,
    timeout: int = 180,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate strict valid JSON datasets. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.35,
        "max_tokens": 12000,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API request failed with HTTP {exc.code}: {detail}") from exc
    return payload["choices"][0]["message"]["content"]


def parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("NVIDIA response JSON must be an object")
    return data


def validate_source_dataset(
    data: dict[str, Any],
    *,
    min_intents: int = 12,
    min_paraphrases_per_intent: int = 8,
    min_near_miss_pairs: int = 20,
) -> None:
    intents = data.get("intents")
    if not isinstance(intents, list) or len(intents) < min_intents:
        raise ValueError(f"expected at least {min_intents} intents")
    intent_ids = set()
    for idx, intent in enumerate(intents):
        if not isinstance(intent, dict):
            raise ValueError(f"intent {idx} must be an object")
        for key in ("intent_id", "category", "canonical_prompt", "safe_answer", "paraphrases"):
            if key not in intent:
                raise ValueError(f"intent {idx} missing {key!r}")
        if not isinstance(intent["intent_id"], str) or not intent["intent_id"]:
            raise ValueError(f"intent {idx} has invalid intent_id")
        if intent["intent_id"] in intent_ids:
            raise ValueError(f"duplicate intent_id {intent['intent_id']!r}")
        intent_ids.add(intent["intent_id"])
        paraphrases = intent["paraphrases"]
        if not isinstance(paraphrases, list) or len(paraphrases) < min_paraphrases_per_intent:
            raise ValueError(f"intent {intent['intent_id']} needs at least {min_paraphrases_per_intent} paraphrases")
        if not all(isinstance(p, str) and p for p in paraphrases):
            raise ValueError(f"intent {intent['intent_id']} has invalid paraphrases")

    pairs = data.get("near_miss_pairs")
    if not isinstance(pairs, list) or len(pairs) < min_near_miss_pairs:
        raise ValueError(f"expected at least {min_near_miss_pairs} near_miss_pairs")
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"near_miss_pair {idx} must be an object")
        if pair.get("a_intent") not in intent_ids or pair.get("b_intent") not in intent_ids:
            raise ValueError(f"near_miss_pair {idx} references an unknown intent")
        if pair.get("a_intent") == pair.get("b_intent"):
            raise ValueError(f"near_miss_pair {idx} references the same intent twice")


def split_demo_dataset(data: dict[str, Any]) -> dict[str, Any]:
    validate_source_dataset(data)
    intents = data["intents"]
    by_id = {intent["intent_id"]: intent for intent in intents}

    calibration_paraphrases: list[list[str]] = []
    heldout_paraphrases: list[list[str]] = []
    for intent in intents:
        canonical = intent["canonical_prompt"]
        paraphrases = intent["paraphrases"]
        for paraphrase in paraphrases[:8]:
            calibration_paraphrases.append([canonical, paraphrase])
        for paraphrase in paraphrases[8:12]:
            heldout_paraphrases.append([canonical, paraphrase])

    calibration_near_misses: list[list[str]] = []
    heldout_near_misses: list[list[str]] = []
    for pair in data["near_miss_pairs"]:
        a = by_id[pair["a_intent"]]
        b = by_id[pair["b_intent"]]
        a_prompts = [a["canonical_prompt"], *a["paraphrases"]]
        b_prompts = [b["canonical_prompt"], *b["paraphrases"]]
        for offset in range(4):
            calibration_near_misses.append([
                a_prompts[offset % len(a_prompts)],
                b_prompts[(offset + 1) % len(b_prompts)],
            ])
        for offset in range(4, 6):
            heldout_near_misses.append([
                a_prompts[offset % len(a_prompts)],
                b_prompts[(offset + 1) % len(b_prompts)],
            ])

    prompts_responses: list[dict[str, str]] = []
    for intent in intents:
        answer = intent["safe_answer"]
        intent_id = intent["intent_id"]
        for prompt in [intent["canonical_prompt"], *intent["paraphrases"][:6]]:
            prompts_responses.append({
                "intent": intent_id,
                "prompt": prompt,
                "response": answer,
            })

    return {
        "calibration_data": {
            "domain": data.get("domain", "customer_support_ecommerce_saas"),
            "source": "nvidia_nim_generated",
            "paraphrases": calibration_paraphrases,
            "near_misses": calibration_near_misses,
        },
        "heldout_paraphrases": {"paraphrases": heldout_paraphrases},
        "heldout_near_misses": {"near_misses": heldout_near_misses},
        "prompts_responses": prompts_responses,
        "counts": {
            "intents": len(intents),
            "calibration_paraphrases": len(calibration_paraphrases),
            "calibration_near_misses": len(calibration_near_misses),
            "heldout_paraphrases": len(heldout_paraphrases),
            "heldout_near_misses": len(heldout_near_misses),
            "prompts_responses": len(prompts_responses),
        },
    }


def write_demo_files(data: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_demo_dataset(data)
    paths = {
        "source": output_dir / "nvidia_source_dataset.json",
        "calibration": output_dir / "calibration_data.json",
        "heldout_paraphrases": output_dir / "heldout_paraphrases.json",
        "heldout_near_misses": output_dir / "heldout_near_misses.json",
        "prompts_responses": output_dir / "prompts_responses.json",
        "manifest": output_dir / "manifest.json",
    }
    paths["source"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    paths["calibration"].write_text(json.dumps(splits["calibration_data"], indent=2), encoding="utf-8")
    paths["heldout_paraphrases"].write_text(json.dumps(splits["heldout_paraphrases"], indent=2), encoding="utf-8")
    paths["heldout_near_misses"].write_text(json.dumps(splits["heldout_near_misses"], indent=2), encoding="utf-8")
    paths["prompts_responses"].write_text(json.dumps(splits["prompts_responses"], indent=2), encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(
            {
                "artifact_type": "latticememory_nvidia_product_demo_dataset",
                "artifact_version": 1,
                "counts": splits["counts"],
                "files": {name: str(path.name) for name, path in paths.items() if name != "manifest"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a LatticeMemory product demo dataset with NVIDIA NIM")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", default="benchmarks/demo_data/customer_support_nvidia")
    parser.add_argument("--intents", type=int, default=18)
    parser.add_argument("--paraphrases-per-intent", type=int, default=12)
    parser.add_argument("--near-miss-pairs", type=int, default=28)
    parser.add_argument(
        "--api-key-env",
        help="Optional environment variable name containing the NVIDIA API key",
    )
    args = parser.parse_args()

    api_key = _api_key(args.api_key_env)
    if not api_key:
        print("Missing NVIDIA API key. Set NVIDIA_API_KEY or NVCF_API_KEY.", file=sys.stderr)
        return 2

    prompt = _dataset_prompt(args.intents, args.paraphrases_per_intent, args.near_miss_pairs)
    content = call_nvidia_chat(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        prompt=prompt,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nvidia_raw_response.txt").write_text(content, encoding="utf-8")
    data = parse_json_response(content)
    validate_source_dataset(
        data,
        min_intents=args.intents,
        min_paraphrases_per_intent=args.paraphrases_per_intent,
        min_near_miss_pairs=args.near_miss_pairs,
    )
    paths = write_demo_files(data, output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
