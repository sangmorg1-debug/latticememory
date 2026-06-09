"""LatticeQABot — Q&A bot powered by the LatticeMemory intent cache.

Two routing modes:

  centroid (default for closed-set)
    Builds one cosine-centroid per intent from all stored examples.
    At query time: dot-product to all centroids, return highest-scoring answer
    above confidence_threshold. Matches the 94% recall / 0 FP benchmark result.
    Requires intent_id labels on stored Q&A pairs.

  exact
    E8 lattice exact-cell lookup — zero-computation at inference.
    Ideal once exact snapping is solved; today has lower recall and FP risk on
    adjacent intents. Use when you have many exact paraphrase variants stored.

Source tiers for every ask():
  1. cache  — score >= confidence_threshold → instant cached answer
  2. llm    — escalation_threshold <= score < confidence_threshold → upstream LLM
  3. escalate — score < escalation_threshold (or no LLM configured) → static reply

Every miss is logged to the flywheel (if configured) so the review loop can
surface coverage gaps and generate training pairs automatically.

Quick-start:
    from latticememory.qa_bot import LatticeQABot

    bot = LatticeQABot(encoder_model="dfrokido/bge-large-e8-snap")
    bot.load_qa([
        {"question": "How do I reset my password?", "answer": "Visit /reset.", "intent_id": "reset_pw"},
        {"question": "I forgot my password",         "answer": "Visit /reset.", "intent_id": "reset_pw"},
    ])
    resp = bot.ask("I lost access to my account")
    print(resp.answer, resp.source, f"{resp.confidence:.2f}")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from latticememory.flywheel import LatticeFlywheel
    from latticememory.semantic_cache import RFSnapSemanticCache


@dataclass
class QAResponse:
    answer: str
    source: str          # "cache", "llm", "escalate"
    confidence: float    # 0.0 – 1.0
    latency_ms: float
    cache_id: str | None = None
    intent_id: str | None = None
    hamming_distance: int = -1


class LatticeQABot:
    """Confidence-routed Q&A bot backed by LatticeMemory.

    Parameters
    ----------
    routing_mode:
        "centroid" (default) — cosine similarity to intent centroids, requires
        intent_id labels. Matches benchmark recall (94% / ~1.67% FP).
        "exact" — E8 lattice exact-cell matching. Zero cost, but recall depends
        on how well paraphrases snap to the same cell.
    encoder_model:
        HuggingFace model name or local path for the E8 snap encoder.
    encoder:
        Pre-built SentenceTransformer. Overrides encoder_model.
    cache:
        Pre-built RFSnapSemanticCache. Used only in "exact" mode.
    upstream_url:
        OpenAI-compatible completions endpoint for the LLM tier.
    upstream_api_key:
        API key for upstream LLM.
    upstream_model:
        Model name sent to the upstream API.
    confidence_threshold:
        Min score to serve from cache. Default 0.85.
    escalation_threshold:
        Below this score: no LLM call, return escalation_response. Default 0.3.
    escalation_response:
        Static reply for low-confidence questions.
    flywheel:
        LatticeFlywheel for automatic miss logging.
    hamming_mode:
        "off" | "shadow" | "serve" — Hamming-NN approximate matching (exact mode only).
    hamming_threshold:
        Block-level Hamming threshold for NN matching (exact mode only).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the user's question concisely and accurately."
    )

    def __init__(
        self,
        *,
        routing_mode: str = "centroid",
        encoder_model: str = "dfrokido/bge-large-e8-snap",
        encoder: Any | None = None,
        cache: "RFSnapSemanticCache | None" = None,
        upstream_url: str | None = None,
        upstream_api_key: str | None = None,
        upstream_model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        confidence_threshold: float = 0.85,
        escalation_threshold: float = 0.30,
        escalation_response: str = (
            "I'm not confident in my answer for this question. "
            "Please contact support for assistance."
        ),
        flywheel: "LatticeFlywheel | None" = None,
        hamming_mode: str = "off",
        hamming_threshold: int = 70,
    ) -> None:
        if routing_mode not in ("centroid", "exact"):
            raise ValueError(f"routing_mode must be 'centroid' or 'exact', got {routing_mode!r}")
        self.routing_mode = routing_mode
        self.encoder_model = encoder_model
        self._encoder = encoder
        self.upstream_url = upstream_url
        self.upstream_api_key = upstream_api_key
        self.upstream_model = upstream_model
        self.system_prompt = system_prompt
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        self.escalation_response = escalation_response
        self.flywheel = flywheel
        self.hamming_mode = hamming_mode
        self.hamming_threshold = hamming_threshold

        # Centroid routing state
        self._intent_embeddings: dict[str, list[np.ndarray]] = {}  # intent_id -> list of embs
        self._intent_answers: dict[str, str] = {}                  # intent_id -> canonical answer
        self._centroid_matrix: np.ndarray | None = None            # (n_intents, d) stacked
        self._centroid_intents: list[str] = []                     # parallel to rows of matrix

        # Exact routing state
        self._cache: "RFSnapSemanticCache | None" = cache
        if routing_mode == "exact" and cache is None:
            self._cache = self._build_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_qa(
        self,
        qa_pairs: list[dict] | str | Path,
        *,
        question_key: str = "question",
        answer_key: str = "answer",
        intent_key: str = "intent_id",
    ) -> int:
        """Ingest Q&A pairs.

        Accepts a list of dicts or a path to a JSON file. Each dict must have
        question and answer keys. For centroid routing, intent_id is required;
        pairs without it are silently skipped in centroid mode.

        Returns the number of entries stored.
        """
        if isinstance(qa_pairs, (str, Path)):
            import json
            qa_pairs = json.loads(Path(qa_pairs).read_text(encoding="utf-8"))

        encoder = self._get_encoder()
        questions = [item.get(question_key, "").strip() for item in qa_pairs]
        questions = [q for q in questions if q]
        if not questions:
            return 0

        # Batch encode all questions at once for efficiency
        embeddings = encoder.encode(questions, normalize_embeddings=True)

        added = 0
        emb_iter = iter(embeddings)
        for item in qa_pairs:
            question = item.get(question_key, "").strip()
            answer = item.get(answer_key, "").strip()
            if not question or not answer:
                continue
            emb = next(emb_iter)
            intent_id = item.get(intent_key)

            if self.routing_mode == "centroid":
                if not intent_id:
                    continue  # centroid mode requires intent label
                self._intent_embeddings.setdefault(intent_id, []).append(emb)
                self._intent_answers[intent_id] = answer  # last write wins
                self._centroid_matrix = None  # invalidate cache
                added += 1
            else:
                # Exact mode: store in E8 cache
                meta: dict[str, Any] = {"source": "load_qa"}
                if intent_id:
                    meta["intent_id"] = intent_id
                assert self._cache is not None
                self._cache.put(question, value=answer, metadata=meta)
                added += 1

        return added

    def ask(self, question: str) -> QAResponse:
        """Route a question through cache → LLM → escalation tiers."""
        t0 = time.perf_counter()

        if self.routing_mode == "centroid":
            return self._ask_centroid(question, t0)
        return self._ask_exact(question, t0)

    def put(self, question: str, answer: str, *, intent_id: str | None = None) -> None:
        """Store a single Q&A pair."""
        if self.routing_mode == "centroid":
            if not intent_id:
                raise ValueError("intent_id required for centroid routing mode")
            enc = self._get_encoder()
            emb = enc.encode([question], normalize_embeddings=True)[0]
            self._intent_embeddings.setdefault(intent_id, []).append(emb)
            self._intent_answers[intent_id] = answer
            self._centroid_matrix = None
        else:
            meta: dict[str, Any] = {"source": "manual"}
            if intent_id:
                meta["intent_id"] = intent_id
            assert self._cache is not None
            self._cache.put(question, value=answer, metadata=meta)

    @property
    def cache_size(self) -> int:
        if self.routing_mode == "centroid":
            return sum(len(v) for v in self._intent_embeddings.values())
        return self._cache.size if self._cache else 0

    @property
    def n_intents(self) -> int:
        if self.routing_mode == "centroid":
            return len(self._intent_answers)
        return 0

    # ------------------------------------------------------------------
    # Centroid routing
    # ------------------------------------------------------------------

    def _ask_centroid(self, question: str, t0: float) -> QAResponse:
        if not self._intent_answers:
            latency_ms = (time.perf_counter() - t0) * 1000
            return QAResponse(
                answer=self.escalation_response,
                source="escalate",
                confidence=0.0,
                latency_ms=latency_ms,
            )

        encoder = self._get_encoder()
        q_emb = encoder.encode([question], normalize_embeddings=True)[0]
        confidence, intent_id = self._centroid_lookup(q_emb)

        # Tier 1: Cache hit
        if confidence >= self.confidence_threshold:
            answer = self._intent_answers[intent_id]
            latency_ms = (time.perf_counter() - t0) * 1000
            return QAResponse(
                answer=answer,
                source="cache",
                confidence=float(confidence),
                latency_ms=latency_ms,
                intent_id=intent_id,
            )

        # Log miss to flywheel
        if self.flywheel is not None:
            self.flywheel.log_miss(question, metadata={"routing_mode": "centroid", "top_score": float(confidence)})

        # Tier 3: Below escalation → no LLM
        if confidence < self.escalation_threshold or self.upstream_url is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return QAResponse(
                answer=self.escalation_response,
                source="escalate",
                confidence=float(confidence),
                latency_ms=latency_ms,
            )

        # Tier 2: LLM fallback
        llm_answer = self._call_llm(question)
        latency_ms = (time.perf_counter() - t0) * 1000
        return QAResponse(
            answer=llm_answer or self.escalation_response,
            source="llm" if llm_answer else "escalate",
            confidence=float(confidence),
            latency_ms=latency_ms,
        )

    def _centroid_lookup(self, q_emb: np.ndarray) -> tuple[float, str]:
        """Return (best_score, best_intent_id) via cosine similarity to centroids."""
        matrix = self._get_centroid_matrix()
        scores = matrix @ q_emb  # (n_intents,) — cosine sim (embeddings are L2-normed)
        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), self._centroid_intents[best_idx]

    def _get_centroid_matrix(self) -> np.ndarray:
        if self._centroid_matrix is not None:
            return self._centroid_matrix
        centroids = []
        intent_ids = []
        for intent_id, embs in self._intent_embeddings.items():
            c = np.stack(embs).mean(axis=0)
            norm = float(np.linalg.norm(c))
            centroids.append(c / max(norm, 1e-8))
            intent_ids.append(intent_id)
        self._centroid_intents = intent_ids
        self._centroid_matrix = np.stack(centroids).astype(np.float32)
        return self._centroid_matrix

    # ------------------------------------------------------------------
    # Exact E8 routing
    # ------------------------------------------------------------------

    def _ask_exact(self, question: str, t0: float) -> QAResponse:
        assert self._cache is not None
        result = self._cache.get(question)
        lookup_ms = (time.perf_counter() - t0) * 1000
        confidence = _result_confidence(result)

        if result.hit and confidence >= self.confidence_threshold:
            return QAResponse(
                answer=str(result.value),
                source="cache",
                confidence=confidence,
                latency_ms=lookup_ms,
                cache_id=result.cache_id,
                intent_id=result.metadata.get("intent_id") if result.metadata else None,
                hamming_distance=result.hamming_distance,
            )

        if self.flywheel is not None:
            key_hex = result.lattice_key.hex() if result.lattice_key else None
            self.flywheel.log_miss(
                question,
                e8_key_hex=key_hex,
                nearest_cache_prompt=result.shadow_source_prompt if result.shadow_hit else None,
                nearest_cache_distance=result.shadow_hamming_distance if result.shadow_hit else -1,
            )

        if confidence < self.escalation_threshold or self.upstream_url is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return QAResponse(
                answer=self.escalation_response,
                source="escalate",
                confidence=confidence,
                latency_ms=latency_ms,
            )

        llm_answer = self._call_llm(question)
        latency_ms = (time.perf_counter() - t0) * 1000
        if llm_answer:
            self._cache.put(question, value=llm_answer, metadata={"source": "llm"})
        return QAResponse(
            answer=llm_answer or self.escalation_response,
            source="llm" if llm_answer else "escalate",
            confidence=confidence,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_cache(self) -> "RFSnapSemanticCache":
        from latticememory.text_runtime import RFSnapTextMemory
        from latticememory.semantic_cache import RFSnapSemanticCache
        from latticememory.hamming_router import HammingRouter

        encoder = self._get_encoder()
        try:
            d = int(encoder.get_embedding_dimension())
        except AttributeError:
            try:
                d = int(encoder.get_sentence_embedding_dimension())
            except Exception:
                probe = encoder.encode(["probe"])
                d = int(np.asarray(probe).shape[-1])

        runtime = RFSnapTextMemory(encoder=encoder, d_model=d)
        hamming_router: HammingRouter | None = None
        if self.hamming_mode != "off":
            hamming_router = HammingRouter(encoder=encoder, d_model=d, threshold=self.hamming_threshold)

        return RFSnapSemanticCache(
            runtime=runtime,
            hamming_router=hamming_router,
            hamming_threshold=self.hamming_threshold,
            hamming_router_mode=self.hamming_mode,
        )

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.encoder_model)
        return self._encoder

    def _call_llm(self, question: str) -> str | None:
        if not self.upstream_url or not self.upstream_api_key:
            return None
        try:
            import json
            import urllib.request

            payload = json.dumps({
                "model": self.upstream_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 512,
            }).encode("utf-8")

            req = urllib.request.Request(
                self.upstream_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.upstream_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except Exception:
            return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _result_confidence(result: Any) -> float:
    if not result.hit:
        return 0.0
    if result.hamming_distance == -1:
        return 1.0
    return max(0.0, 1.0 - result.hamming_distance / 128.0)
