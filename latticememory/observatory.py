"""LatticeObservatory — block-level interpretability for the E8 lattice.

Every E8 key is structured: one byte per 8D block, 0-239 (the Shell-1 address).
This module makes that structure fully observable so an AI can reason over it
symbolically — not just "similarity = 0.73" but "blocks 12, 47, 91 differ;
those correspond to embedding dims 96-103, 376-383, 728-735."

Primary use: feed export_for_llm() into a reasoning model to get targeted
encoder-tuning recommendations, collision diagnostics, and routing failure maps.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from latticememory.index import LatticeIndex


class LatticeObservatory:
    """Read-only interpretability layer over a LatticeIndex.

    Exposes the E8 lattice block-by-block so an AI can:
    - identify which 8D subspaces are stable vs. noisy for a semantic cluster
    - detect key collisions (unrelated concepts sharing one cell)
    - trace exactly which blocks cause a routing mismatch between two texts
    - export a self-describing JSON snapshot for LLM-guided optimization

    All methods are non-mutating. Pass a LatticeIndex; nothing is written.
    """

    def __init__(self, index: LatticeIndex) -> None:
        self._idx = index
        self._runtime = index._runtime
        self._mem = self._runtime.memory
        self._d_model = index._d_model
        self._n_blocks = self._d_model // 8

    # ------------------------------------------------------------------
    # block_stability
    # ------------------------------------------------------------------

    def block_stability(self, texts: list[str]) -> dict:
        """Per-block entropy across a cluster of texts.

        Entropy 0.0 = every text maps to the same E8 address for this block
        (perfectly stable — this block is part of the concept fingerprint).
        High entropy = the block address varies across semantically similar
        texts (noisy — a source of routing misses).

        Returns a dict with per-block stats plus the concept_fingerprint_blocks
        list (blocks with stability == 1.0) and noisiest_blocks (sorted worst-first).
        """
        if not texts:
            return {"error": "no texts provided"}

        keys = [self._key_for(t) for t in texts]
        n = len(keys)

        blocks: list[dict] = []
        for i in range(self._n_blocks):
            addrs = [k[i] for k in keys]
            counter: dict[int, int] = {}
            for a in addrs:
                counter[a] = counter.get(a, 0) + 1
            total_entropy = -sum(
                (c / n) * math.log2(c / n) for c in counter.values()
            )
            most_common = max(counter, key=counter.__getitem__)
            stability = counter[most_common] / n
            blocks.append({
                "block": i,
                "dim_range": f"{i * 8}-{i * 8 + 7}",
                "entropy": round(total_entropy, 4),
                "stability": round(stability, 4),
                "unique_addresses": len(counter),
                "most_common_address": most_common,
            })

        mean_entropy = sum(b["entropy"] for b in blocks) / self._n_blocks
        stable = [b for b in blocks if b["stability"] == 1.0]
        unstable = sorted(
            [b for b in blocks if b["stability"] < 1.0],
            key=lambda b: b["stability"],
        )

        return {
            "n_texts": n,
            "n_blocks": self._n_blocks,
            "mean_block_entropy": round(mean_entropy, 4),
            "fully_stable_block_count": len(stable),
            "unstable_block_count": len(unstable),
            "concept_fingerprint_blocks": [b["block"] for b in stable],
            "noisiest_blocks": unstable[:10],
            "all_blocks": blocks,
        }

    # ------------------------------------------------------------------
    # cell_coherence
    # ------------------------------------------------------------------

    def cell_coherence(self, key: str | bytes) -> dict:
        """Semantic coherence of all documents sharing an E8 key.

        Computes pairwise cosine similarities for every document in the cell.

        coherence_label:
          'tight'     — mean cosine >= 0.95  (cell captured a real concept)
          'loose'     — mean cosine 0.80-0.95 (fuzzy concept boundary)
          'collision' — mean cosine < 0.80   (unrelated concepts share the key)
          'singleton' — only one document    (no pairwise comparison possible)
        """
        key_bytes = bytes.fromhex(key) if isinstance(key, str) else key
        doc_ids = self._mem.get_document_ids_by_lattice_key(key_bytes)
        if not doc_ids:
            return {"key": key_bytes.hex(), "doc_count": 0, "status": "empty_cell"}

        embs: list[np.ndarray] = []
        texts: list[str] = []
        for doc_id in doc_ids:
            emb_t = self._mem.lattice._embeddings.get(doc_id)
            if emb_t is not None:
                embs.append(
                    F.normalize(emb_t.float().cpu(), p=2, dim=0).numpy()
                )
                texts.append((self._mem.get_document_text(doc_id) or "")[:120])

        if len(embs) < 2:
            return {
                "key": key_bytes.hex(),
                "doc_count": len(embs),
                "mean_cosine": 1.0,
                "min_cosine": 1.0,
                "coherence_label": "singleton",
                "texts": texts,
            }

        mat = np.stack(embs)
        sims = mat @ mat.T
        n = len(embs)
        pairs = [float(sims[i, j]) for i in range(n) for j in range(i + 1, n)]
        mean_cos = float(np.mean(pairs))
        min_cos = float(np.min(pairs))

        label = (
            "tight" if mean_cos >= 0.95
            else "loose" if mean_cos >= 0.80
            else "collision"
        )

        return {
            "key": key_bytes.hex(),
            "doc_count": n,
            "mean_cosine": round(mean_cos, 4),
            "min_cosine": round(min_cos, 4),
            "coherence_label": label,
            "texts": texts,
        }

    # ------------------------------------------------------------------
    # trace_mismatch
    # ------------------------------------------------------------------

    def trace_mismatch(self, query: str, doc: str) -> dict:
        """Block-by-block diff between a query's and a document's E8 keys.

        Shows exactly which 8D subspaces (and which embedding dimensions)
        cause the routing mismatch, and delivers a routing verdict:
          lattice_exact    — hamming == 0, O(1) hit guaranteed
          lattice_hamming1 — hamming == 1, O(1) neighborhood hit
          fallback_required — hamming > 1, dense fallback needed
        """
        q_emb = self._encode(query)
        d_emb = self._encode(doc)
        q_key = self._mem.lattice_key_for(q_emb)
        d_key = self._mem.lattice_key_for(d_emb)

        q_norm = F.normalize(q_emb, p=2, dim=0).numpy()
        d_norm = F.normalize(d_emb, p=2, dim=0).numpy()
        cosine = float(np.dot(q_norm, d_norm))

        hamming = sum(1 for i in range(self._n_blocks) if q_key[i] != d_key[i])

        differing = [
            {
                "block": i,
                "dim_range": f"{i * 8}-{i * 8 + 7}",
                "query_address": int(q_key[i]),
                "doc_address": int(d_key[i]),
            }
            for i in range(self._n_blocks)
            if q_key[i] != d_key[i]
        ]
        matching = [
            {
                "block": i,
                "dim_range": f"{i * 8}-{i * 8 + 7}",
                "address": int(q_key[i]),
            }
            for i in range(self._n_blocks)
            if q_key[i] == d_key[i]
        ]

        if hamming == 0:
            verdict = "lattice_exact"
        elif hamming == 1:
            verdict = "lattice_hamming1"
        else:
            verdict = "fallback_required"

        return {
            "query": query,
            "doc": (doc[:120] + "...") if len(doc) > 120 else doc,
            "cosine_similarity": round(cosine, 4),
            "hamming_distance": hamming,
            "match_rate": round(1.0 - hamming / self._n_blocks, 4),
            "routing_verdict": verdict,
            "n_differing_blocks": len(differing),
            "n_matching_blocks": len(matching),
            "differing_blocks": differing,
            "matching_blocks": matching,
        }

    # ------------------------------------------------------------------
    # routing_profile
    # ------------------------------------------------------------------

    def routing_profile(self, queries: list[str], docs: list[str]) -> dict:
        """Aggregate routing analysis over a set of known (query, doc) pairs.

        For each pair computes the Hamming distance and routing verdict, then
        identifies which blocks are responsible for the most routing failures.
        The top_mismatching_blocks list is the direct input to block-targeted
        encoder fine-tuning.
        """
        if len(queries) != len(docs):
            raise ValueError("queries and docs must have the same length")
        if not queries:
            return {"error": "no pairs provided"}

        n = len(queries)
        exact = hamming1 = fallback = 0
        hamming_vals: list[int] = []
        miss_freq = [0] * self._n_blocks

        for q, d in zip(queries, docs):
            q_key = self._mem.lattice_key_for(self._encode(q))
            d_key = self._mem.lattice_key_for(self._encode(d))
            h = sum(1 for i in range(self._n_blocks) if q_key[i] != d_key[i])
            hamming_vals.append(h)
            for i in range(self._n_blocks):
                if q_key[i] != d_key[i]:
                    miss_freq[i] += 1
            if h == 0:
                exact += 1
            elif h == 1:
                hamming1 += 1
            else:
                fallback += 1

        top_blocks = sorted(
            [
                {
                    "block": i,
                    "dim_range": f"{i * 8}-{i * 8 + 7}",
                    "miss_count": miss_freq[i],
                    "miss_rate": round(miss_freq[i] / n, 4),
                }
                for i in range(self._n_blocks)
                if miss_freq[i] > 0
            ],
            key=lambda b: b["miss_rate"],
            reverse=True,
        )

        target_dims = (
            ", ".join(b["dim_range"] for b in top_blocks[:5])
            if top_blocks else "none"
        )

        return {
            "n_pairs": n,
            "routing": {
                "lattice_exact": exact,
                "lattice_hamming1": hamming1,
                "fallback": fallback,
                "exact_rate": round(exact / n, 4),
                "hamming1_rate": round(hamming1 / n, 4),
                "fallback_rate": round(fallback / n, 4),
            },
            "hamming": {
                "mean": round(float(np.mean(hamming_vals)), 2),
                "median": round(float(np.median(hamming_vals)), 2),
                "p90": round(float(np.percentile(hamming_vals, 90)), 2),
                "max": int(max(hamming_vals)),
            },
            "top_mismatching_blocks": top_blocks[:20],
            "optimization_target": (
                f"Tune encoder dims [{target_dims}] — highest per-pair miss rates."
                if top_blocks else "No mismatches. Index is exact-routing for all pairs."
            ),
        }

    # ------------------------------------------------------------------
    # export_for_llm
    # ------------------------------------------------------------------

    def export_for_llm(self, *, n_sample_cells: int = 5) -> dict:
        """Export a complete, structured lattice snapshot for LLM analysis.

        This is the primary interface for AI-guided optimization. Feed the
        output directly into a reasoning model. The JSON is self-describing:
        every section has an 'interpretation' or guidance field so the model
        has context, not just numbers.

        Sections:
          index_summary     — doc count, key count, bucket distribution
          block_analysis    — per-block entropy, most stable / most variable
          sample_cells      — coherence analysis of the largest cells
          recommendations   — auto-generated actionable findings
          interpretation_guide — vocabulary for the receiving model
        """
        hash_store = self._mem.lattice.hash_store
        if not hash_store:
            return {"status": "empty_index", "docs": 0}

        bucket_sizes = [len(ids) for ids in hash_store.values()]

        # Per-block entropy across every document in the index
        all_keys = [
            self._mem.lattice._keys[doc_id]
            for doc_id in self._mem._doc_ids
            if doc_id in self._mem.lattice._keys
        ]
        block_entropy: list[float] = []
        if all_keys:
            for i in range(self._n_blocks):
                addrs = [k[i] for k in all_keys]
                counter: dict[int, int] = {}
                for a in addrs:
                    counter[a] = counter.get(a, 0) + 1
                total = len(addrs)
                h = -sum((c / total) * math.log2(c / total) for c in counter.values())
                block_entropy.append(round(h, 4))

        most_stable = sorted(range(len(block_entropy)), key=lambda i: block_entropy[i])[:10]
        most_variable = sorted(range(len(block_entropy)), key=lambda i: block_entropy[i], reverse=True)[:10]

        # Coherence of the largest cells
        sorted_cells = sorted(hash_store.items(), key=lambda x: len(x[1]), reverse=True)
        sample_cells = [self.cell_coherence(k) for k, _ in sorted_cells[:n_sample_cells]]

        return {
            "schema_version": 1,
            "index_summary": {
                "total_docs": self._mem.num_documents,
                "unique_keys": len(hash_store),
                "d_model": self._d_model,
                "n_blocks": self._n_blocks,
                "mean_bucket_size": round(float(np.mean(bucket_sizes)), 3),
                "max_bucket_size": int(max(bucket_sizes)),
                "singleton_keys": sum(1 for s in bucket_sizes if s == 1),
                "multi_doc_keys": sum(1 for s in bucket_sizes if s > 1),
            },
            "block_analysis": {
                "entropy_per_block": block_entropy,
                "mean_entropy": round(float(np.mean(block_entropy)), 4) if block_entropy else None,
                "most_stable_blocks": most_stable,
                "most_variable_blocks": most_variable,
                "interpretation": (
                    "Block entropy: 0.0 = all docs use the same address (semantic fingerprint). "
                    f"~7.9 = log2(240) = fully random. "
                    f"Blocks {most_stable[:3]} are the most stable — strongest semantic signal. "
                    f"Blocks {most_variable[:3]} are noisiest — routing failures originate here."
                ),
            },
            "sample_cells": sample_cells,
            "recommendations": self._recommendations(block_entropy, sample_cells, bucket_sizes),
            "interpretation_guide": {
                "coherence_label": {
                    "tight": "mean cosine >= 0.95. Cell holds a well-defined concept.",
                    "loose": "mean cosine 0.80-0.95. Concept boundary is fuzzy.",
                    "collision": "mean cosine < 0.80. Unrelated concepts share a key — false cache hits possible.",
                    "singleton": "One document. No pairwise comparison available.",
                },
                "routing_verdict": {
                    "lattice_exact": "Hamming=0. O(1) cache hit guaranteed.",
                    "lattice_hamming1": "Hamming=1. O(1) neighborhood hit.",
                    "fallback_required": "Hamming>1. Dense index required.",
                },
                "optimization_approach": (
                    "Run routing_profile(queries, docs) on your target workload. "
                    "The top_mismatching_blocks list names the exact embedding dimensions "
                    "responsible for routing failures. Fine-tune only those subspaces — "
                    "block-targeted optimization avoids the global collapse that full-encoder "
                    "training causes."
                ),
            },
        }

    # ------------------------------------------------------------------
    # collision_audit
    # ------------------------------------------------------------------

    def collision_audit(self) -> dict:
        """Full scan of every cell, ranked by collision risk (worst first).

        Unlike export_for_llm which samples top-N cells, this audits the
        entire index. Use this as the first diagnostic step before any
        optimization — it gives you the complete picture of where false
        cache hits can occur.

        coherence_label distribution:
          collision — mean cosine < 0.80  (false hits guaranteed)
          loose     — mean cosine 0.80-0.95 (fuzzy, monitor)
          tight     — mean cosine >= 0.95  (safe)
          singleton — one doc, no risk
        """
        hash_store = self._mem.lattice.hash_store
        if not hash_store:
            return {"status": "empty_index", "cells_audited": 0}

        all_results: list[dict] = []
        for key in hash_store:
            all_results.append(self.cell_coherence(key))

        singletons = [r for r in all_results if r.get("coherence_label") == "singleton"]
        multi = [r for r in all_results if r.get("coherence_label") != "singleton"]
        multi.sort(key=lambda r: r.get("mean_cosine", 1.0))

        collision = [r for r in multi if r.get("coherence_label") == "collision"]
        loose = [r for r in multi if r.get("coherence_label") == "loose"]
        tight = [r for r in multi if r.get("coherence_label") == "tight"]

        return {
            "total_cells": len(hash_store),
            "singleton_cells": len(singletons),
            "multi_doc_cells": len(multi),
            "collision_cells": len(collision),
            "loose_cells": len(loose),
            "tight_cells": len(tight),
            "collision_rate": round(len(collision) / max(len(multi), 1), 4),
            "ranked_by_risk": multi,
            "summary": (
                f"{len(collision)} collision cells out of {len(multi)} multi-doc cells. "
                f"False cache hits possible — run trace_mismatch() on the top entries to isolate which blocks to fix."
                if collision else
                "No collision cells. Cache coherence is high across all multi-doc cells."
            ),
        }

    # ------------------------------------------------------------------
    # fragmentation_score
    # ------------------------------------------------------------------

    def fragmentation_score(self, texts: list[str]) -> dict:
        """How unified is a semantic cluster in E8 space?

        1.0 = all texts share one E8 key (perfect for caching — any query in
              this cluster gets an O(1) hit).
        0.0 = every text lands in its own cell (concept is completely
              fragmented — caching is impossible, every query is a miss).

        Use this before adding a concept cluster to your index to predict
        whether E8 routing will capture it, or whether you need the dense
        fallback for that topic.
        """
        if not texts:
            return {"error": "no texts provided"}

        keys = [self._key_for(t) for t in texts]
        unique_keys = list(dict.fromkeys(keys))  # preserve order, deduplicate
        n = len(texts)
        n_unique = len(unique_keys)

        score = 1.0 if n == 1 else 1.0 - (n_unique - 1) / (n - 1)

        label = (
            "unified" if score == 1.0
            else "cohesive" if score >= 0.75
            else "fragmented" if score >= 0.25
            else "scattered"
        )

        key_distribution = {}
        for k in keys:
            h = k.hex()
            key_distribution[h] = key_distribution.get(h, 0) + 1

        return {
            "n_texts": n,
            "n_unique_keys": n_unique,
            "score": round(score, 4),
            "label": label,
            "cacheable": n_unique == 1,
            "interpretation": (
                f"{n} texts map to {n_unique} E8 key(s). "
                + (
                    "All texts share one cell — O(1) cache hit guaranteed for this cluster."
                    if n_unique == 1
                    else f"Concept fragments across {n_unique} cells — {n - n_unique} texts "
                    f"would be cache misses for a query landing in the dominant cell."
                )
            ),
            "dominant_key": max(key_distribution, key=key_distribution.__getitem__),
            "key_distribution": key_distribution,
        }

    # ------------------------------------------------------------------
    # suggest_training_pairs
    # ------------------------------------------------------------------

    def suggest_training_pairs(self, pairs: list[tuple[str, str]]) -> dict:
        """Given (query, doc) pairs that should route to the same key but don't,
        generate targeted fine-tuning training data.

        Each pair is analyzed block by block. The output is a list of
        (anchor, positive) training pairs structured to maximize gradient
        signal on the specific blocks responsible for routing failures —
        block-targeted contrastive training rather than full-encoder training.

        The returned training_recipe tells you exactly which dims to constrain,
        which loss to use, and how many steps to target.

        Args:
            pairs: list of (query, doc) tuples that are semantically equivalent
                   but produce different E8 keys (routing mismatches).
        """
        if not pairs:
            return {"error": "no pairs provided"}

        miss_freq: dict[int, int] = {}
        training_pairs: list[dict] = []
        mismatch_count = 0

        for query, doc in pairs:
            q_key = self._key_for(query)
            d_key = self._key_for(doc)
            differing = [i for i in range(self._n_blocks) if q_key[i] != d_key[i]]
            h = len(differing)

            if h > 0:
                mismatch_count += 1
                for i in differing:
                    miss_freq[i] = miss_freq.get(i, 0) + 1
                training_pairs.append({
                    "anchor": query,
                    "positive": doc,
                    "hamming_distance": h,
                    "target_blocks": differing[:5],
                    "target_dims": [f"{i * 8}-{i * 8 + 7}" for i in differing[:5]],
                })

        if not training_pairs:
            return {
                "mismatch_count": 0,
                "message": "All pairs already route to the same E8 key. No training needed.",
                "training_pairs": [],
            }

        top_blocks = sorted(miss_freq, key=miss_freq.__getitem__, reverse=True)[:10]
        top_dim_ranges = [f"{i * 8}-{i * 8 + 7}" for i in top_blocks]

        return {
            "n_pairs": len(pairs),
            "mismatch_count": mismatch_count,
            "mismatch_rate": round(mismatch_count / len(pairs), 4),
            "training_pairs": training_pairs,
            "top_target_blocks": top_blocks,
            "top_target_dims": top_dim_ranges,
            "training_recipe": {
                "loss": "E8RoutingLoss (latticememory.training)",
                "freeze_strategy": (
                    f"Freeze all encoder layers. Unfreeze only projection dims "
                    f"[{', '.join(top_dim_ranges[:5])}]. "
                    "Block-targeted fine-tuning preserves the stable blocks."
                ),
                "n_pairs_needed": max(len(training_pairs) * 10, 100),
                "suggested_epochs": 3,
                "note": (
                    "Augment training_pairs with hard negatives: texts from different "
                    "semantic clusters that happen to share a key (collision cells). "
                    "Run collision_audit() to find them."
                ),
            },
        }

    # ------------------------------------------------------------------
    # compare_snapshots
    # ------------------------------------------------------------------

    def compare_snapshots(self, before: dict, after: dict) -> dict:
        """Diff two export_for_llm() snapshots to measure optimization impact.

        Call export_for_llm() before and after encoder fine-tuning, then pass
        both results here to get a structured delta report. This is the
        verification step that closes the optimization loop.

        Returns deltas for: block entropy, collision rate, routing distribution,
        and overall health assessment.
        """
        for label, snap in (("before", before), ("after", after)):
            if "index_summary" not in snap or "block_analysis" not in snap:
                return {"error": f"'{label}' snapshot is not a valid export_for_llm() output"}

        b_entropy = before["block_analysis"].get("entropy_per_block", [])
        a_entropy = after["block_analysis"].get("entropy_per_block", [])

        block_deltas: list[dict] = []
        if b_entropy and a_entropy and len(b_entropy) == len(a_entropy):
            for i, (b, a) in enumerate(zip(b_entropy, a_entropy)):
                delta = round(a - b, 4)
                block_deltas.append({
                    "block": i,
                    "dim_range": f"{i * 8}-{i * 8 + 7}",
                    "before": b,
                    "after": a,
                    "delta": delta,
                    "direction": "improved" if delta < 0 else "degraded" if delta > 0 else "unchanged",
                })

        improved_blocks = sum(1 for d in block_deltas if d["direction"] == "improved")
        degraded_blocks = sum(1 for d in block_deltas if d["direction"] == "degraded")

        b_mean = before["block_analysis"].get("mean_entropy")
        a_mean = after["block_analysis"].get("mean_entropy")
        entropy_delta = round(a_mean - b_mean, 4) if (b_mean is not None and a_mean is not None) else None

        b_sum = before["index_summary"]
        a_sum = after["index_summary"]

        b_cells = b_sum.get("multi_doc_keys", 0)
        a_cells = a_sum.get("multi_doc_keys", 0)

        b_collisions = sum(
            1 for c in before.get("sample_cells", [])
            if c.get("coherence_label") == "collision"
        )
        a_collisions = sum(
            1 for c in after.get("sample_cells", [])
            if c.get("coherence_label") == "collision"
        )

        verdict = "unknown"
        if entropy_delta is not None:
            if entropy_delta < -0.1 and a_collisions <= b_collisions:
                verdict = "improved"
            elif entropy_delta > 0.1 or a_collisions > b_collisions:
                verdict = "degraded"
            else:
                verdict = "neutral"

        most_improved = sorted(block_deltas, key=lambda d: d["delta"])[:5]
        most_degraded = sorted(block_deltas, key=lambda d: d["delta"], reverse=True)[:5]

        return {
            "verdict": verdict,
            "entropy": {
                "before_mean": b_mean,
                "after_mean": a_mean,
                "delta": entropy_delta,
                "improved_blocks": improved_blocks,
                "degraded_blocks": degraded_blocks,
            },
            "collisions": {
                "before_sampled": b_collisions,
                "after_sampled": a_collisions,
                "delta": a_collisions - b_collisions,
            },
            "index": {
                "before_docs": b_sum.get("total_docs"),
                "after_docs": a_sum.get("total_docs"),
                "before_keys": b_sum.get("unique_keys"),
                "after_keys": a_sum.get("unique_keys"),
            },
            "most_improved_blocks": most_improved,
            "most_degraded_blocks": most_degraded,
            "block_deltas": block_deltas,
            "summary": (
                f"Verdict: {verdict.upper()}. "
                f"Mean block entropy: {b_mean} -> {a_mean} (delta {entropy_delta:+.4f}). "
                f"{improved_blocks} blocks improved, {degraded_blocks} degraded. "
                f"Sampled collision cells: {b_collisions} -> {a_collisions}."
            ) if entropy_delta is not None else "Insufficient data to compare snapshots.",
        }

    # ------------------------------------------------------------------
    # neighbor_density
    # ------------------------------------------------------------------

    def neighbor_density(self, key: str | bytes) -> dict:
        """Map the Hamming-1 neighborhood of an E8 key.

        For each stored key at Hamming distance exactly 1, reports which
        block differs, how many docs live there, and a sample text.

        Use this to answer: "If I expand from lattice_exact to lattice_hamming1
        for this key, how many additional documents do I get, and are they
        relevant or noise?"

        High neighbor density + high cell coherence in neighbors = safe to
        expand. High neighbor density + collision neighbors = noisy recall.
        """
        key_bytes = bytes.fromhex(key) if isinstance(key, str) else key
        hash_store = self._mem.lattice.hash_store

        own_doc_ids = hash_store.get(key_bytes, [])
        own_docs = len(own_doc_ids)

        neighbors: list[dict] = []
        for stored_key, doc_ids in hash_store.items():
            if stored_key == key_bytes:
                continue
            diffs = [i for i in range(self._n_blocks) if stored_key[i] != key_bytes[i]]
            if len(diffs) != 1:
                continue
            block_i = diffs[0]
            sample_text = None
            if doc_ids:
                raw = self._mem.get_document_text(doc_ids[0])
                sample_text = (raw[:80] + "...") if raw and len(raw) > 80 else raw
            coherence = self.cell_coherence(stored_key)
            neighbors.append({
                "key": stored_key.hex(),
                "differing_block": block_i,
                "dim_range": f"{block_i * 8}-{block_i * 8 + 7}",
                "doc_count": len(doc_ids),
                "coherence_label": coherence.get("coherence_label", "unknown"),
                "mean_cosine": coherence.get("mean_cosine"),
                "sample_text": sample_text,
            })

        neighbors.sort(key=lambda n: n["doc_count"], reverse=True)
        total_neighbor_docs = sum(n["doc_count"] for n in neighbors)
        safe_neighbors = [n for n in neighbors if n["coherence_label"] in {"tight", "singleton"}]
        noisy_neighbors = [n for n in neighbors if n["coherence_label"] == "collision"]

        return {
            "key": key_bytes.hex(),
            "own_doc_count": own_docs,
            "hamming1_neighbor_cells": len(neighbors),
            "total_neighbor_docs": total_neighbor_docs,
            "safe_neighbor_cells": len(safe_neighbors),
            "noisy_neighbor_cells": len(noisy_neighbors),
            "neighbors": neighbors,
            "expansion_verdict": (
                "safe" if not noisy_neighbors and total_neighbor_docs > 0
                else "noisy" if noisy_neighbors
                else "empty"
            ),
            "recall_estimate": (
                f"lattice_hamming1 expansion adds {total_neighbor_docs} docs "
                f"across {len(neighbors)} neighbor cell(s). "
                + (f"{len(noisy_neighbors)} noisy (collision) neighbor(s) — "
                   "post-retrieval cosine filter recommended."
                   if noisy_neighbors else "All neighbors are coherent.")
            ),
        }

    # ------------------------------------------------------------------
    # semantic_probe  (JEPA-style: what does each block encode?)
    # ------------------------------------------------------------------

    def semantic_probe(self, labeled_texts: dict[str, list[str]]) -> dict:
        """Map E8 blocks to semantic dimensions via mutual information.

        JEPA framing: which 8D subspaces carry predictive signal for each
        label? A block with high information gain for a label is encoding
        that semantic dimension. A block with near-zero gain is noise for
        that label.

        Use this to answer: "Which dims should I fine-tune to make the
        encoder reliably separate TOPIC vs. SENTIMENT vs. STYLE?"

        Args:
            labeled_texts: dict mapping label name to list of texts.
                           e.g. {"sports": [...], "science": [...]}

        Returns block-level information gain and separability scores,
        ranked so the top entries are the prime fine-tuning targets.
        """
        if not labeled_texts:
            return {"error": "no labeled_texts provided"}
        if len(labeled_texts) < 2:
            return {"error": "need at least 2 labels to compute separability"}

        label_list = sorted(labeled_texts.keys())
        all_keys: list[bytes] = []
        all_labels: list[str] = []
        for label in label_list:
            for text in labeled_texts[label]:
                all_keys.append(self._key_for(text))
                all_labels.append(label)

        n = len(all_labels)
        if n < 2:
            return {"error": "need at least 2 total texts"}

        label_counts = Counter(all_labels)
        h_label = -sum((c / n) * math.log2(c / n) for c in label_counts.values() if c > 0)

        block_results: list[dict] = []
        for i in range(self._n_blocks):
            addr_for_label: dict[str, list[int]] = {lab: [] for lab in label_list}
            for key, lab in zip(all_keys, all_labels):
                addr_for_label[lab].append(key[i])

            addrs_all = [k[i] for k in all_keys]
            addr_counter = Counter(addrs_all)

            joint = Counter((k[i], lab) for k, lab in zip(all_keys, all_labels))

            h_label_given_block = 0.0
            for addr, addr_count in addr_counter.items():
                p_addr = addr_count / n
                h_cond = 0.0
                for lab in label_list:
                    p_lab_given_addr = joint.get((addr, lab), 0) / addr_count
                    if p_lab_given_addr > 0:
                        h_cond -= p_lab_given_addr * math.log2(p_lab_given_addr)
                h_label_given_block += p_addr * h_cond

            info_gain = max(0.0, h_label - h_label_given_block)
            separability = info_gain / h_label if h_label > 0 else 0.0

            # Per-label dominant address (which E8 cell does each class prefer?)
            label_dominant: dict[str, int] = {}
            for lab in label_list:
                if addr_for_label[lab]:
                    c = Counter(addr_for_label[lab])
                    label_dominant[lab] = c.most_common(1)[0][0]

            block_results.append({
                "block": i,
                "dim_range": f"{i * 8}-{i * 8 + 7}",
                "info_gain": round(info_gain, 4),
                "separability": round(separability, 4),
                "label_dominant_address": label_dominant,
            })

        top = sorted(block_results, key=lambda b: b["info_gain"], reverse=True)
        bottom = sorted(block_results, key=lambda b: b["info_gain"])

        return {
            "n_classes": len(label_list),
            "n_texts": n,
            "labels": label_list,
            "label_entropy": round(h_label, 4),
            "top_separating_blocks": top[:10],
            "bottom_separating_blocks": bottom[:5],
            "all_blocks": block_results,
            "fine_tune_targets": [b["dim_range"] for b in top[:5]],
            "freeze_candidates": [b["dim_range"] for b in bottom[:5]],
            "interpretation": (
                f"Top separating blocks: {[b['block'] for b in top[:3]]} "
                f"(dims {', '.join(b['dim_range'] for b in top[:3])}). "
                f"These encode the distinction between {label_list}. "
                f"Bottom blocks {[b['block'] for b in bottom[:3]]} carry no label signal — "
                "freeze them during targeted training to preserve unrelated structure."
            ),
        }

    # ------------------------------------------------------------------
    # block_correlation  (JEPA-style: which blocks are redundant?)
    # ------------------------------------------------------------------

    def block_correlation(self, texts: list[str]) -> dict:
        """Compute normalized mutual information between every pair of blocks.

        NMI = 0.0 → blocks are statistically independent (each carries unique signal).
        NMI = 1.0 → blocks are perfectly correlated (one is redundant).

        JEPA framing: if block i can predict block j with high accuracy, then
        block j is redundant given block i — they occupy the same semantic
        subspace. Fine-tuning both wastes gradient and risks entanglement.

        Returns the top correlated pairs, redundancy candidates (NMI > 0.9),
        and per-block average correlation so you can identify "hub" blocks
        that are correlated with many others.
        """
        if len(texts) < 2:
            return {"error": "need at least 2 texts to compute block correlation"}

        keys = [self._key_for(t) for t in texts]
        n = len(texts)

        block_addrs = [[k[i] for k in keys] for i in range(self._n_blocks)]

        def _entropy(seq: list[int]) -> float:
            cnt = Counter(seq)
            total = len(seq)
            return -sum((c / total) * math.log2(c / total) for c in cnt.values() if c > 0)

        def _mi(a: list[int], b: list[int]) -> float:
            joint = Counter(zip(a, b))
            ca, cb = Counter(a), Counter(b)
            total = len(a)
            mi = 0.0
            for (ai, bi), count in joint.items():
                p_ab = count / total
                p_a = ca[ai] / total
                p_b = cb[bi] / total
                if p_a > 0 and p_b > 0 and p_ab > 0:
                    mi += p_ab * math.log2(p_ab / (p_a * p_b))
            return max(0.0, mi)

        entropies = [_entropy(block_addrs[i]) for i in range(self._n_blocks)]

        nmi_by_block: dict[int, list[float]] = {i: [] for i in range(self._n_blocks)}
        high_pairs: list[dict] = []

        for i in range(self._n_blocks):
            for j in range(i + 1, self._n_blocks):
                h_i, h_j = entropies[i], entropies[j]
                if h_i == 0 and h_j == 0:
                    nmi = 1.0
                elif h_i == 0 or h_j == 0:
                    nmi = 0.0
                else:
                    mi = _mi(block_addrs[i], block_addrs[j])
                    nmi = min(1.0, 2 * mi / (h_i + h_j))
                nmi = round(nmi, 4)
                nmi_by_block[i].append(nmi)
                nmi_by_block[j].append(nmi)
                if nmi > 0.5:
                    high_pairs.append({
                        "block_i": i,
                        "block_j": j,
                        "dim_range_i": f"{i * 8}-{i * 8 + 7}",
                        "dim_range_j": f"{j * 8}-{j * 8 + 7}",
                        "nmi": nmi,
                        "label": "redundant" if nmi > 0.9 else "correlated",
                    })

        high_pairs.sort(key=lambda p: p["nmi"], reverse=True)

        # Per-block mean NMI — "hub" blocks correlate with many others
        hub_blocks = sorted(
            [
                {
                    "block": i,
                    "dim_range": f"{i * 8}-{i * 8 + 7}",
                    "mean_nmi_with_others": round(float(np.mean(nmi_by_block[i])), 4) if nmi_by_block[i] else 0.0,
                }
                for i in range(self._n_blocks)
            ],
            key=lambda b: b["mean_nmi_with_others"],
            reverse=True,
        )

        redundant = [p for p in high_pairs if p["nmi"] > 0.9]
        all_nmis = [p["nmi"] for p in high_pairs] + [0.0] * (
            self._n_blocks * (self._n_blocks - 1) // 2 - len(high_pairs)
        )

        return {
            "n_texts": n,
            "n_blocks": self._n_blocks,
            "mean_inter_block_nmi": round(float(np.mean(all_nmis)) if all_nmis else 0.0, 4),
            "high_correlation_pairs": high_pairs[:20],
            "redundant_pairs": redundant,
            "hub_blocks": hub_blocks[:10],
            "interpretation": (
                f"{len(redundant)} redundant block pairs (NMI > 0.9). "
                "Each redundant pair wastes 8 encoder dims — fine-tune only one from each pair, "
                "freeze the other. Hub blocks (high mean NMI) are the most inter-correlated; "
                "they likely encode a dominant surface feature rather than semantic content."
                if redundant else
                "No highly redundant blocks detected. Each block carries statistically independent signal."
            ),
        }

    # ------------------------------------------------------------------
    # address_trajectory  (JEPA-style: encoder continuity through E8 space)
    # ------------------------------------------------------------------

    def address_trajectory(self, text_sequence: list[str]) -> dict:
        """Track how the E8 key evolves step-by-step through a text sequence.

        JEPA framing: a well-trained encoder should represent semantic
        progressions as smooth paths through representation space. Large
        Hamming jumps between semantically adjacent texts reveal where the
        encoder is discontinuous — the blocks that jump most frequently are
        the fine-tuning targets for continuity training.

        Use this with paraphrase chains, topic progressions, or sentence
        transformations to map the encoder's semantic geometry.

        Args:
            text_sequence: ordered list of texts (e.g. a paraphrase chain,
                           a concept progression, before/after edits).
        """
        if len(text_sequence) < 2:
            return {"error": "need at least 2 texts for a trajectory"}

        keys = [self._key_for(t) for t in text_sequence]
        steps: list[dict] = []
        all_changed: list[int] = []

        for i in range(1, len(keys)):
            changed = [b for b in range(self._n_blocks) if keys[i][b] != keys[i - 1][b]]
            all_changed.extend(changed)
            h = len(changed)
            steps.append({
                "step": i,
                "from_text": (text_sequence[i - 1][:60] + "...") if len(text_sequence[i - 1]) > 60 else text_sequence[i - 1],
                "to_text": (text_sequence[i][:60] + "...") if len(text_sequence[i]) > 60 else text_sequence[i],
                "hamming_delta": h,
                "changed_blocks": changed[:10],
                "changed_dim_ranges": [f"{b * 8}-{b * 8 + 7}" for b in changed[:10]],
                "routing_continuity": (
                    "exact" if h == 0
                    else "hamming1" if h == 1
                    else "discontinuous"
                ),
            })

        n_steps = len(steps)
        total_hamming = sum(s["hamming_delta"] for s in steps)
        mean_hamming = total_hamming / n_steps
        smooth = sum(1 for s in steps if s["routing_continuity"] in {"exact", "hamming1"})

        change_freq = Counter(all_changed)
        volatile_blocks = sorted(
            [
                {
                    "block": b,
                    "dim_range": f"{b * 8}-{b * 8 + 7}",
                    "change_count": c,
                    "change_rate": round(c / n_steps, 4),
                }
                for b, c in change_freq.items()
            ],
            key=lambda x: x["change_count"],
            reverse=True,
        )

        return {
            "n_texts": len(text_sequence),
            "n_steps": n_steps,
            "total_hamming": total_hamming,
            "mean_hamming_per_step": round(mean_hamming, 2),
            "smooth_steps": smooth,
            "discontinuous_steps": n_steps - smooth,
            "trajectory_continuity": round(smooth / n_steps, 4),
            "most_volatile_blocks": volatile_blocks[:10],
            "steps": steps,
            "training_recommendation": (
                "Trajectory is smooth — encoder represents this progression continuously."
                if mean_hamming <= 1.0 else
                f"Mean Hamming {mean_hamming:.1f}/step — encoder is discontinuous. "
                f"Run triplet loss training with intermediate paraphrases as hard positives. "
                f"Target volatile blocks: dims {', '.join(b['dim_range'] for b in volatile_blocks[:3])}."
            ),
        }

    # ------------------------------------------------------------------
    # generate_training_curriculum  (the prescription tool)
    # ------------------------------------------------------------------

    def generate_training_curriculum(self, clusters: dict[str, list[str]]) -> dict:
        """Generate a complete, block-weighted training curriculum from a cluster spec.

        This is the "mold the encoder" tool. Given how you want the E8 lattice
        to look (which texts should share a key, which should be separated), it
        produces everything needed to get there:

          - Positive pairs: within-cluster texts that need to unify
          - Hard negatives: cross-cluster pairs that are currently too close
          - Block training weights: which 8D subspaces need the most gradient
          - Curriculum steps: warm-up → core → separation ordering

        JEPA framing: treat each cluster as a "target concept cell." The
        curriculum pushes the encoder to make every text in a cluster
        predictable from the cluster's canonical E8 address.

        Args:
            clusters: dict of {cluster_name: [texts]}. Each cluster defines
                      a set of texts that SHOULD route to the same E8 key.
        """
        if not clusters:
            return {"error": "no clusters provided"}

        cluster_names = list(clusters.keys())

        # Analyze fragmentation per cluster
        cluster_analysis: list[dict] = []
        all_cluster_keys: dict[str, list[bytes]] = {}
        for name, texts in clusters.items():
            keys = [self._key_for(t) for t in texts]
            all_cluster_keys[name] = keys
            frag = self.fragmentation_score(texts) if len(texts) > 1 else {"score": 1.0, "n_unique_keys": 1, "label": "unified"}
            cluster_analysis.append({
                "name": name,
                "n_texts": len(texts),
                "fragmentation_score": frag.get("score", 1.0),
                "n_unique_keys": frag.get("n_unique_keys", 1),
                "label": frag.get("label", "unified"),
                "needs_unification": frag.get("score", 1.0) < 0.75,
            })

        # Positive pairs: within each cluster, all (anchor, positive) combos
        positive_pairs: list[dict] = []
        for name, texts in clusters.items():
            cluster_keys = all_cluster_keys[name]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    h = sum(1 for b in range(self._n_blocks) if cluster_keys[i][b] != cluster_keys[j][b])
                    positive_pairs.append({
                        "type": "positive",
                        "cluster": name,
                        "anchor": texts[i],
                        "positive": texts[j],
                        "current_hamming": h,
                        "target_hamming": 0,
                        "difficulty": "easy" if h <= 2 else "medium" if h <= 10 else "hard",
                    })

        # Hard negatives: for each cluster pair, find the closest texts (lowest Hamming)
        hard_negatives: list[dict] = []
        for i, name_a in enumerate(cluster_names):
            for name_b in cluster_names[i + 1:]:
                texts_a, texts_b = clusters[name_a], clusters[name_b]
                keys_a, keys_b = all_cluster_keys[name_a], all_cluster_keys[name_b]
                min_h = self._n_blocks + 1
                hardest: tuple[str, str] | None = None
                for ia, ka in enumerate(keys_a):
                    for ib, kb in enumerate(keys_b):
                        h = sum(1 for b in range(self._n_blocks) if ka[b] != kb[b])
                        if h < min_h:
                            min_h = h
                            hardest = (texts_a[ia], texts_b[ib])
                if hardest:
                    hard_negatives.append({
                        "type": "hard_negative",
                        "cluster_a": name_a,
                        "cluster_b": name_b,
                        "anchor": hardest[0],
                        "negative": hardest[1],
                        "current_hamming": min_h,
                        "difficulty": "hard" if min_h < self._n_blocks // 4 else "medium",
                    })

        # Block training weights: within-cluster address entropy per block
        block_weights: list[dict] = []
        for block_i in range(self._n_blocks):
            total_h = 0.0
            total_n = 0
            for name, keys in all_cluster_keys.items():
                if not keys:
                    continue
                addrs = [k[block_i] for k in keys]
                cnt = Counter(addrs)
                n = len(addrs)
                h = -sum((c / n) * math.log2(c / n) for c in cnt.values() if c > 0)
                total_h += h * n
                total_n += n
            avg_h = total_h / max(total_n, 1)
            block_weights.append({
                "block": block_i,
                "dim_range": f"{block_i * 8}-{block_i * 8 + 7}",
                "training_weight": round(avg_h, 4),
            })

        block_weights.sort(key=lambda b: b["training_weight"], reverse=True)
        top_weights = block_weights[:20]

        n_needing_unification = sum(1 for a in cluster_analysis if a["needs_unification"])
        positive_pairs.sort(key=lambda p: p["current_hamming"], reverse=True)
        hard_negatives.sort(key=lambda n: n["current_hamming"])

        return {
            "n_clusters": len(clusters),
            "cluster_analysis": cluster_analysis,
            "clusters_needing_unification": n_needing_unification,
            "positive_pairs": positive_pairs,
            "hard_negative_pairs": hard_negatives,
            "block_training_weights": top_weights,
            "curriculum_steps": [
                {
                    "phase": 1,
                    "name": "warm_up",
                    "description": "Easy positives only — establish cluster centers without collapse",
                    "n_pairs": sum(1 for p in positive_pairs if p["difficulty"] == "easy"),
                    "lr_multiplier": 0.1,
                    "target_pairs": [p for p in positive_pairs if p["difficulty"] == "easy"],
                },
                {
                    "phase": 2,
                    "name": "cluster_unification",
                    "description": "Medium/hard positives — pull fragmented cluster members together",
                    "n_pairs": sum(1 for p in positive_pairs if p["difficulty"] in {"medium", "hard"}),
                    "lr_multiplier": 1.0,
                    "target_pairs": [p for p in positive_pairs if p["difficulty"] in {"medium", "hard"}],
                },
                {
                    "phase": 3,
                    "name": "cluster_separation",
                    "description": "Hard negatives — push cluster boundaries apart",
                    "n_pairs": len(hard_negatives),
                    "lr_multiplier": 0.5,
                    "target_pairs": hard_negatives,
                },
            ],
            "loss_config": {
                "loss": "E8RoutingLoss (latticememory.training)",
                "block_weights": {b["dim_range"]: b["training_weight"] for b in top_weights[:10]},
                "margin": 0.2,
                "temperature": 0.07,
                "freeze_strategy": (
                    f"Unfreeze dims [{', '.join(b['dim_range'] for b in top_weights[:5])}]. "
                    f"Freeze dims [{', '.join(b['dim_range'] for b in block_weights[-5:])}]. "
                    "Frozen dims have near-zero within-cluster entropy — they already encode stable signal."
                ),
            },
            "summary": (
                f"{len(positive_pairs)} positive pairs, {len(hard_negatives)} hard negatives "
                f"across {len(clusters)} clusters. "
                f"{n_needing_unification} cluster(s) need unification. "
                f"Top 3 training targets: {', '.join(b['dim_range'] for b in top_weights[:3])}."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        arr = self._runtime._encode_texts([text])[0]
        return torch.as_tensor(arr, dtype=torch.float32)

    def _key_for(self, text: str) -> bytes:
        return self._mem.lattice_key_for(self._encode(text))

    def _recommendations(
        self,
        block_entropy: list[float],
        sample_cells: list[dict],
        bucket_sizes: list[int],
    ) -> list[str]:
        recs: list[str] = []

        if block_entropy:
            mean_h = float(np.mean(block_entropy))
            high_count = sum(1 for h in block_entropy if h > 3.0)
            if high_count > self._n_blocks * 0.3:
                recs.append(
                    f"{high_count}/{self._n_blocks} blocks have entropy > 3.0 (>30% near-random). "
                    "Domain-specific encoder fine-tuning on those block ranges would most improve "
                    "cache hit rate. Run routing_profile() to identify exact target dims."
                )
            if mean_h < 0.5 and self._mem.num_documents > 10:
                recs.append(
                    "Very low mean block entropy across a non-trivial corpus. "
                    "This may indicate key collapse — all content mapping to the same region. "
                    "Inspect whether the encoder was trained with a cell-alignment objective "
                    "(address loss or neighborhood loss) that caused mode collapse."
                )

        collision_cells = [c for c in sample_cells if c.get("coherence_label") == "collision"]
        if collision_cells:
            recs.append(
                f"{len(collision_cells)}/{len(sample_cells)} sampled cells have coherence_label='collision' "
                "(mean cosine < 0.80). False cache hits are possible — different concepts routed "
                "to the same key. Add a post-retrieval cosine threshold guard, or increase d_model "
                "to raise key resolution."
            )

        max_bucket = max(bucket_sizes, default=0)
        if max_bucket > 50:
            recs.append(
                f"Largest cell holds {max_bucket} documents — O(1) becomes O(N) within the bucket. "
                "Run cell_coherence() on that key. If coherence_label is 'collision', "
                "this is a hot-spot collision, not a legitimate large cluster."
            )

        if not recs:
            recs.append(
                "No critical issues detected. "
                "Index appears healthy for cache and dedup workloads."
            )

        return recs
