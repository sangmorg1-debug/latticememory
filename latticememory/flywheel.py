"""LatticeFlywheel — active learning loop that grows cache hit rate from miss signals.

Workflow:
  1. log_miss()         — persist every cache miss (question + E8 key)
  2. cluster_gaps()     — group near-identical misses → surface coverage gaps
  3. export_review_queue() — write a JSON file for human review (fill in answers)
  4. load_reviewed()    — ingest approved Q&As back into the cache
  5. generate_training_pairs() — build contrastive pairs from confirmed clusters
  6. finetune()         — run SnapTrainer to tighten snap on confirmed intents

Wire into the proxy via miss_log_path= on LatticeLLMProxy. Every MISS event is
automatically logged with its E8 key so no encoder is needed at review time.

Quick-start (standalone):
    flywheel = LatticeFlywheel("misses.jsonl")
    flywheel.log_miss("How do I reset my password?", e8_key_hex=key_hex)
    clusters = flywheel.top_gaps(n=5)
    n = flywheel.export_review_queue("review.json")
    # ... human fills in answers in review.json ...
    added = flywheel.load_reviewed("review.json", cache)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from latticememory.semantic_cache import RFSnapSemanticCache
    from latticememory.snap_trainer import SnapTrainingConfig, SnapTrainResult


@dataclass
class MissRecord:
    question: str
    e8_key_hex: str  # 128-byte E8 key as hex; empty string if unavailable
    timestamp: float = field(default_factory=time.time)
    nearest_cache_prompt: str | None = None
    nearest_cache_distance: int = -1
    metadata: dict = field(default_factory=dict)


@dataclass
class MissCluster:
    """A group of semantically similar missed questions."""
    cluster_id: int
    representative: str         # Most central question (closest to centroid)
    members: list               # list[MissRecord]
    size: int
    centroid_key_hex: str       # Majority-vote E8 key across all member keys


class LatticeFlywheel:
    """Active learning flywheel: log misses → cluster → review → retrain.

    The loop closes automatically: as reviewed Q&As enter the cache the hit
    rate rises; new misses represent genuinely new intent gaps, not old ones.
    After enough new clusters accumulate, finetune() sharpens the encoder so
    existing paraphrases snap more reliably to the right cache entry.
    """

    DEFAULT_CLUSTER_THRESHOLD = 25  # blocks out of 128 — same-intent boundary

    def __init__(
        self,
        log_path: str | Path,
        *,
        encoder: Any | None = None,
        encoder_model: str = "dfrokido/bge-large-e8-snap",
        cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD,
    ) -> None:
        self.log_path = Path(log_path)
        self.encoder_model = encoder_model
        self._encoder = encoder
        self._router: Any | None = None  # lazily-built HammingRouter for key computation
        self.cluster_threshold = cluster_threshold

    # ------------------------------------------------------------------
    # Miss logging
    # ------------------------------------------------------------------

    def log_miss(
        self,
        question: str,
        e8_key_hex: str | None = None,
        *,
        nearest_cache_prompt: str | None = None,
        nearest_cache_distance: int = -1,
        metadata: dict | None = None,
    ) -> None:
        """Append one miss record to the JSONL log.

        e8_key_hex is the 256-char hex string of the 128-byte E8 key. When the
        caller is the proxy the key is already computed; pass it directly. If
        omitted and an encoder is configured the key is computed on the fly.
        """
        if not e8_key_hex:
            e8_key_hex = self._compute_key(question) or ""

        record = MissRecord(
            question=question,
            e8_key_hex=e8_key_hex,
            nearest_cache_prompt=nearest_cache_prompt,
            nearest_cache_distance=nearest_cache_distance,
            metadata=metadata or {},
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_log(self) -> list[MissRecord]:
        """Load all records from the JSONL log."""
        if not self.log_path.exists():
            return []
        records: list[MissRecord] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    records.append(MissRecord(**d))
        return records

    def clear_log(self) -> None:
        """Truncate the miss log (after a review cycle, for example)."""
        if self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_gaps(self, min_cluster_size: int = 3) -> list[MissCluster]:
        """Greedy single-linkage cluster by E8 block-level Hamming distance.

        Records without a key are dropped. Two records join the same cluster
        when the distance from the new record's key to the cluster centroid is
        <= self.cluster_threshold. Returns clusters sorted by size, descending.
        """
        raw = self.load_log()
        records = self._fill_missing_keys(raw)
        records = [r for r in records if r.e8_key_hex]
        if not records:
            return []

        keys = [bytes.fromhex(r.e8_key_hex) for r in records]
        # cluster_members: list of index-lists; cluster_keys: running centroid
        cluster_members: list[list[int]] = []
        cluster_keys: list[bytes] = []

        for idx, key in enumerate(keys):
            best_cid: int | None = None
            best_dist = self.cluster_threshold + 1
            for cid, ckey in enumerate(cluster_keys):
                dist = _hamming(key, ckey)
                if dist < best_dist:
                    best_dist = dist
                    best_cid = cid
            if best_cid is not None and best_dist <= self.cluster_threshold:
                cluster_members[best_cid].append(idx)
                cluster_keys[best_cid] = _majority_key(
                    [keys[i] for i in cluster_members[best_cid]]
                )
            else:
                cluster_members.append([idx])
                cluster_keys.append(key)

        clusters: list[MissCluster] = []
        for cid, (member_indices, ckey) in enumerate(zip(cluster_members, cluster_keys)):
            if len(member_indices) < min_cluster_size:
                continue
            members = [records[i] for i in member_indices]
            rep = min(members, key=lambda r: _hamming(bytes.fromhex(r.e8_key_hex), ckey))
            clusters.append(
                MissCluster(
                    cluster_id=cid,
                    representative=rep.question,
                    members=members,
                    size=len(members),
                    centroid_key_hex=ckey.hex(),
                )
            )

        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def top_gaps(self, n: int = 10, min_cluster_size: int = 3) -> list[MissCluster]:
        """Return the n largest miss clusters (coverage gaps to fill)."""
        return self.cluster_gaps(min_cluster_size=min_cluster_size)[:n]

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------

    def export_review_queue(
        self,
        path: str | Path,
        n: int = 10,
        min_cluster_size: int = 3,
    ) -> int:
        """Write a JSON review queue from the top miss clusters.

        Format::

            [
              {
                "cluster_id": 0,
                "representative_question": "How do I reset my password?",
                "sample_questions": ["...", ...],  // up to 5 examples
                "size": 12,
                "answer": null,       // reviewer fills this in
                "intent_id": null,    // optional canonical name
                "centroid_key_hex": "..."
              },
              ...
            ]

        Returns the number of items written.
        """
        clusters = self.top_gaps(n=n, min_cluster_size=min_cluster_size)
        queue = [
            {
                "cluster_id": c.cluster_id,
                "representative_question": c.representative,
                "sample_questions": [m.question for m in c.members[:5]],
                "size": c.size,
                "answer": None,
                "intent_id": None,
                "centroid_key_hex": c.centroid_key_hex,
            }
            for c in clusters
        ]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
        return len(queue)

    def load_reviewed(
        self,
        path: str | Path,
        cache: "RFSnapSemanticCache",
    ) -> int:
        """Ingest answered review items back into cache.

        Only items with a non-empty answer string are loaded. The representative
        question is stored as the canonical cache key; sample questions are also
        stored so the cache recognises paraphrases immediately.

        Returns the number of Q&A entries added.
        """
        items = json.loads(Path(path).read_text(encoding="utf-8"))
        added = 0
        for item in items:
            answer = item.get("answer")
            if not answer or not isinstance(answer, str) or not answer.strip():
                continue
            question = item["representative_question"]
            meta = {
                "source": "flywheel_review",
                "cluster_id": item.get("cluster_id"),
                "intent_id": item.get("intent_id"),
                "cluster_size": item.get("size"),
            }
            cache.put(question, value=answer, metadata=meta)
            for alt in item.get("sample_questions", []):
                if alt and alt != question:
                    cache.put(alt, value=answer, metadata=meta)
            added += 1
        return added

    # ------------------------------------------------------------------
    # Training pair generation
    # ------------------------------------------------------------------

    def generate_training_pairs(
        self,
        cache: "RFSnapSemanticCache | None" = None,
        min_examples_per_cluster: int = 3,
    ) -> list:
        """Build RoutingTrainingExample objects from confirmed miss clusters.

        Strategy:
          - Within each cluster: all (query, positive) pairs — same intent
          - Cross-cluster: other cluster representatives → hard negatives
          - Nearest cache neighbor of cluster centroid (if cache provided) → hard negative

        Returns a list of ``RoutingTrainingExample(query, positive, negatives)``.
        """
        from latticememory.training import RoutingTrainingExample

        clusters = self.cluster_gaps(min_cluster_size=min_examples_per_cluster)
        if not clusters:
            return []

        other_reps = {c.cluster_id: c.representative for c in clusters}

        examples: list[RoutingTrainingExample] = []
        for cluster in clusters:
            questions = [m.question for m in cluster.members]
            cross_negatives = [
                rep for cid, rep in other_reps.items() if cid != cluster.cluster_id
            ][:8]

            # Hard negative: nearest cache entry to this cluster's centroid
            cache_neighbor: str | None = None
            if cache is not None and cluster.centroid_key_hex:
                centroid = bytes.fromhex(cluster.centroid_key_hex)
                best_dist = 999
                for entry in cache._entries.values():
                    if entry.lattice_key:
                        d = _hamming(centroid, entry.lattice_key)
                        if 0 < d < best_dist:
                            best_dist = d
                            cache_neighbor = entry.prompt
                if best_dist > 50:   # too far to be a useful hard negative
                    cache_neighbor = None

            for i, query in enumerate(questions):
                positives = [q for j, q in enumerate(questions) if j != i]
                if not positives:
                    continue
                negatives = list(cross_negatives)
                if cache_neighbor and cache_neighbor not in negatives:
                    negatives.insert(0, cache_neighbor)
                if negatives:
                    examples.append(
                        RoutingTrainingExample(
                            query=query,
                            positive=positives[0],
                            negatives=negatives,
                        )
                    )
        return examples

    # ------------------------------------------------------------------
    # Automatic fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        base_model: str,
        output_dir: str | Path,
        *,
        cache: "RFSnapSemanticCache | None" = None,
        config: "SnapTrainingConfig | None" = None,
        min_examples_per_cluster: int = 3,
        add_paws: bool = True,
        paws_limit: int = 10_000,
    ) -> "SnapTrainResult":
        """Generate training pairs from miss clusters and run SnapTrainer.

        The fine-tuned model is saved to ``output_dir/best_snap_encoder``. Pass
        that path to ``LatticeQABot(encoder_model=...)`` or the proxy env var
        ``LATTICE_ENCODER_MODEL`` to deploy it.

        Parameters
        ----------
        base_model:
            HuggingFace model name or local path to start from.
        output_dir:
            Directory for the trained checkpoint.
        cache:
            Optional cache — enables hard negatives from nearest cache neighbors.
        config:
            Override training hyperparameters. Defaults are collapse-safe.
        min_examples_per_cluster:
            Minimum cluster size to include in training.
        add_paws:
            Whether to supplement with PAWS general-language pairs for stability.
        paws_limit:
            How many PAWS examples to add (kept small to not drown domain signal).

        Returns
        -------
        SnapTrainResult with best checkpoint metrics.
        """
        from latticememory.snap_trainer import SnapTrainer, SnapTrainingConfig

        domain_examples = self.generate_training_pairs(
            cache=cache, min_examples_per_cluster=min_examples_per_cluster
        )
        if not domain_examples:
            raise ValueError(
                "No training examples generated. Need at least "
                f"{min_examples_per_cluster} clustered misses per cluster. "
                "Check flywheel.stats() and cluster_gaps()."
            )

        # Build val_clusters from miss clusters so Observatory can track per-intent progress
        clusters = self.cluster_gaps(min_cluster_size=min_examples_per_cluster)
        val_clusters = {
            f"flywheel_cluster_{c.cluster_id}": [m.question for m in c.members[:10]]
            for c in clusters
        }

        if config is None:
            import dataclasses
            config = SnapTrainingConfig(
                epochs=3,
                batch_size=4,
                gradient_accumulation_steps=8,
                freeze_layers=20,
                output_dir=str(output_dir),
            )
        else:
            if config.output_dir is None:
                import dataclasses
                config = dataclasses.replace(config, output_dir=str(output_dir))

        trainer = SnapTrainer(base_model=base_model, val_clusters=val_clusters)

        examples = list(domain_examples)
        if add_paws:
            try:
                paws = trainer.load_paws(limit=paws_limit)
                examples = examples + paws
            except Exception:
                pass  # PAWS unavailable — train on domain pairs only

        return trainer.train(examples, config=config)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Summary stats for the miss log."""
        records = self.load_log()
        return {
            "total_misses": len(records),
            "unique_questions": len({r.question for r in records}),
            "log_path": str(self.log_path),
            "log_exists": self.log_path.exists(),
            "cluster_threshold": self.cluster_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_router(self) -> Any:
        """Lazily build a HammingRouter for key computation."""
        if self._router is not None:
            return self._router
        if self._encoder is not None:
            from latticememory.hamming_router import HammingRouter
            try:
                d = int(self._encoder.get_sentence_embedding_dimension())
            except Exception:
                probe = self._encoder.encode(["probe"])
                d = int(np.asarray(probe).shape[-1])
            self._router = HammingRouter(encoder=self._encoder, d_model=d)
            return self._router
        if self.encoder_model:
            try:
                from latticememory.hamming_router import HammingRouter
                self._router = HammingRouter.from_model(self.encoder_model)
                return self._router
            except Exception:
                pass
        return None

    def _compute_key(self, text: str) -> str:
        router = self._get_router()
        if router is None:
            return ""
        try:
            return router.e8_key(text).hex()
        except Exception:
            return ""

    def _fill_missing_keys(self, records: list[MissRecord]) -> list[MissRecord]:
        """Compute E8 keys for records that were logged without one."""
        missing = [r for r in records if not r.e8_key_hex]
        if not missing:
            return records
        router = self._get_router()
        if router is None:
            return [r for r in records if r.e8_key_hex]
        filled = []
        for r in records:
            if r.e8_key_hex:
                filled.append(r)
            else:
                try:
                    key_hex = router.e8_key(r.question).hex()
                    filled.append(
                        MissRecord(
                            question=r.question,
                            e8_key_hex=key_hex,
                            timestamp=r.timestamp,
                            nearest_cache_prompt=r.nearest_cache_prompt,
                            nearest_cache_distance=r.nearest_cache_distance,
                            metadata=r.metadata,
                        )
                    )
                except Exception:
                    pass  # drop record if key computation fails
        return filled


# ------------------------------------------------------------------
# Module-level helpers (pure functions, no class state)
# ------------------------------------------------------------------

def _hamming(a: bytes, b: bytes) -> int:
    """Block-level Hamming distance between two E8 keys."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def _majority_key(keys: list[bytes]) -> bytes:
    """Per-block majority vote — approximate centroid in E8 space."""
    if not keys:
        return b""
    length = len(keys[0])
    result = bytearray(length)
    for i in range(length):
        counts: dict[int, int] = {}
        for key in keys:
            val = key[i] if i < len(key) else 0
            counts[val] = counts.get(val, 0) + 1
        result[i] = max(counts, key=lambda v: counts[v])
    return bytes(result)
