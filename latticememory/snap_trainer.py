"""SnapTrainer — train a symmetric E8 snap encoder with live Observatory evaluation.

Key lesson from v1 collapse:
  lambda_address=50 overwhelmed contrastive (1:50 ratio) → model mapped everything to one cell.
  lambda_negative=0 removed the only force separating different concepts → no resistance to collapse.

Correct recipe (v2):
  - lambda_address: 3.0   (gentle alignment — not overwhelming)
  - lambda_negative: 1.0  (PAWS label=0 hard negatives push concepts apart)
  - freeze_layers: 20      (freeze first 20/24 transformer layers — preserve base semantics)
  - separation_score > 0.8 is a required success gate alongside fragmentation_score

The Observatory now measures BOTH:
  - fragmentation_score: do paraphrases land in the SAME cell? (intra-cluster)
  - separation_score:    do different concepts land in DIFFERENT cells? (inter-cluster)
Both must pass before training is considered successful.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from latticememory.index import LatticeIndex
from latticememory.observatory import LatticeObservatory
from latticememory.training import (
    E8RoutingLoss,
    RoutingTrainingExample,
    _enable_gradient_checkpointing,
    _encode_trainable_texts,
    _encoder_parameters,
    _move_encoder_to_device,
    _select_batch_negatives,
    _set_encoder_eval,
    _set_encoder_train,
)


@dataclass
class SnapTrainingConfig:
    """Hyperparameters for SnapTrainer.train().

    Critical parameters (v2 defaults fix the collapse seen in v1):
      lambda_address:  3.0   — was 50.0; 50:1 ratio to contrastive caused collapse
      lambda_negative: 1.0   — was 0.0; PAWS negatives are required to prevent collapse
      freeze_layers:   20    — freeze first 20/24 BGE-large layers; preserve base semantics
      zero_fp_recall_target: 0.8056 — product gate for safe HammingRouter recall
      separation_target: 0.8 — inter-cluster separation gate (checks for collapse)
    """
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 1e-4
    temperature: float = 0.05
    lambda_address: float = 0.0
    lambda_hamming: float = 0.3
    lambda_negative: float = 1.0
    lambda_near_miss: float = 1.0
    near_miss_margin: float = 80.0
    lambda_address_hinge: float = 3.0
    address_hinge_margin: float = 0.2
    lambda_address_mse: float = 3.0
    lambda_soft_hard: float = 0.0
    soft_hard_temperature_start: float = 1.0
    soft_hard_temperature_end: float = 0.05
    soft_hard_straight_through: bool = False
    soft_hard_focal_gamma: float = 0.0
    soft_hard_top_k_blocks: int = 0
    fp16: bool = True
    gradient_checkpointing: bool = True
    # Freeze the first N transformer layers. Prevents address pressure from destroying the
    # base semantic representation. For BGE-large (24 layers), 20 is the right number.
    freeze_layers: int = 20
    log_every_batches: int = 50
    seed: int = 42
    device: str = "auto"
    output_dir: str | Path | None = None
    obs_eval_every_steps: int = 50
    # Product success is calibrated safe recall. Fragmentation remains a research metric.
    fragmentation_target: float = 0.75
    zero_fp_recall_target: float = 0.8056
    separation_target: float = 0.80


@dataclass
class SnapObsCheckpoint:
    """Observatory metrics captured at a specific gradient step during training."""
    epoch: int
    global_step: int
    batch_in_epoch: int
    examples_seen: int
    mean_fragmentation_score: float
    separation_score: float
    hamming_gap: float
    zero_fp_recall: float
    zero_fp_threshold: int
    cluster_scores: dict[str, float]
    mean_hamming_per_step: float
    trajectory_continuity: float
    mean_inter_block_nmi: float
    address_loss_recent: float
    train_loss_recent: float
    is_best: bool = False
    is_collapsed: bool = False
    elapsed_sec: float = 0.0
    address_hinge_loss_recent: float = 0.0
    address_mse_loss_recent: float = 0.0
    soft_hard_loss_recent: float = 0.0
    target_cell_probability_recent: float = 0.0
    soft_hard_temperature: float = 0.0
    paraphrase_hamming_mean: float = 0.0
    near_miss_hamming_mean: float = 0.0


@dataclass
class SnapEpochMetrics:
    epoch: int
    train_loss: float
    address_loss: float
    contrastive_loss: float
    mean_fragmentation_score: float
    separation_score: float
    hamming_gap: float
    zero_fp_recall: float
    zero_fp_threshold: int
    mean_hamming_per_step: float
    trajectory_continuity: float
    mean_inter_block_nmi: float
    cluster_scores: dict[str, float]
    elapsed_sec: float
    is_best: bool = False
    address_hinge_loss: float = 0.0
    address_mse_loss: float = 0.0
    soft_hard_loss: float = 0.0
    target_cell_probability: float = 0.0
    soft_hard_temperature: float = 0.0
    paraphrase_hamming_mean: float = 0.0
    near_miss_hamming_mean: float = 0.0


@dataclass
class SnapTrainResult:
    epoch_metrics: list[SnapEpochMetrics]
    obs_checkpoints: list[SnapObsCheckpoint]
    best_global_step: int
    best_fragmentation_score: float
    best_separation_score: float
    reached_target: bool
    output_dir: Path | None
    base_model: str
    d_model: int
    examples_trained: int
    best_hamming_gap: float = 0.0
    best_zero_fp_recall: float = 0.0
    best_zero_fp_threshold: int = 0


def _freeze_encoder_layers(encoder, n: int) -> int:
    """Freeze the first n transformer layers of a SentenceTransformer (BERT-family).

    Returns the count of frozen parameters.
    """
    if n <= 0:
        return 0
    bert = None
    # SentenceTransformer wraps: encoder[0].auto_model = BertModel/RobertaModel
    m0 = getattr(encoder, "__getitem__", None)
    if m0 is not None:
        try:
            bert = encoder[0].auto_model
        except Exception:
            pass
    if bert is None:
        return 0

    frozen = 0
    # Freeze embedding layer
    emb = getattr(bert, "embeddings", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad = False
            frozen += p.numel()

    # Freeze first n transformer layers
    enc = getattr(bert, "encoder", None)
    layers = getattr(enc, "layer", None) if enc else None
    if layers is not None:
        for layer in list(layers)[:n]:
            for p in layer.parameters():
                p.requires_grad = False
                frozen += p.numel()
    return frozen


def _is_better_snap_checkpoint(
    *,
    mean_fragmentation: float,
    hamming_gap: float,
    zero_fp_recall: float,
    best_fragmentation: float,
    best_hamming_gap: float,
    best_zero_fp_recall: float,
) -> bool:
    if zero_fp_recall > best_zero_fp_recall:
        return True
    if zero_fp_recall < best_zero_fp_recall:
        return False
    if hamming_gap > best_hamming_gap:
        return True
    if hamming_gap < best_hamming_gap:
        return False
    return mean_fragmentation > best_fragmentation


def _product_gate_passed(*, ckpt: SnapObsCheckpoint, config: SnapTrainingConfig) -> bool:
    return (
        ckpt.zero_fp_recall >= config.zero_fp_recall_target
        and ckpt.separation_score >= config.separation_target
        and not ckpt.is_collapsed
    )


class SnapTrainer:
    """Train a symmetric E8 snap encoder using live Observatory-guided evaluation.

    The Observatory runs every obs_eval_every_steps gradient updates and measures:
      - zero_fp_recall:      product gate for safe calibrated HammingRouter recall
      - separation_score:    do different concepts land in different E8 cells? (anti-collapse gate)
      - fragmentation_score: research metric for exact same-cell snapping

    A model that collapses to constant output scores fragmentation=1.0 but separation=0.0.
    Product success requires zero-FP recall and separation; exact fragmentation is not required.

    Collapse prevention (v2):
      - freeze_layers=20 preserves the base semantic representation in frozen layers
      - lambda_negative=1.0 requires PAWS label=0 hard negatives in RoutingTrainingExample
      - lambda_address=3.0 gives gentle alignment signal without dominating the contrastive loss

    Usage:
        trainer = SnapTrainer(
            base_model="BAAI/bge-large-en-v1.5",
            val_clusters={
                "refund_policy":   ["I want a refund", "Can I get my money back?", ...],
                "order_status":    ["Where is my order?", "Track my package", ...],
            },
        )
        examples = trainer.load_paws(limit=50_000)
        result = trainer.train(examples)
    """

    def __init__(
        self,
        *,
        base_model: str = "BAAI/bge-large-en-v1.5",
        val_clusters: dict[str, list[str]],
        d_model: int | None = None,
        device: str = "auto",
    ) -> None:
        if not val_clusters:
            raise ValueError("val_clusters must contain at least 1 cluster")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_model = base_model
        self.val_clusters = val_clusters
        self.device = device

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(base_model, device=device)

        if d_model is None:
            embed_fn = getattr(self.encoder, "get_embedding_dimension", None)
            d_model = int(embed_fn() or 0) if embed_fn else 0
            if d_model <= 0:
                probe = self.encoder.encode(["probe"])
                d_model = int(np.asarray(probe).shape[-1])

        self.d_model = d_model

    @classmethod
    def _from_encoder(
        cls,
        encoder,
        *,
        val_clusters: dict[str, list[str]],
        d_model: int,
        device: str = "cpu",
        base_model: str = "custom",
    ) -> "SnapTrainer":
        """Build a SnapTrainer from a pre-built encoder (for testing, no HF download)."""
        if not val_clusters:
            raise ValueError("val_clusters must contain at least 1 cluster")
        obj = object.__new__(cls)
        obj.encoder = encoder
        obj.val_clusters = val_clusters
        obj.d_model = d_model
        obj.device = device
        obj.base_model = base_model
        return obj

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def load_paws(self, *, limit: int | None = None, split: str = "train") -> list[RoutingTrainingExample]:
        """Load PAWS labeled_final paraphrase pairs.

        label=1 pairs → positives (same concept, should share E8 cell)
        label=0 pairs → hard negatives (look similar but different concept, different cell)
        Both are required: positives define the snap target, negatives prevent collapse.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("pip install 'datasets'") from exc

        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split, streaming=True)
        positives_by_s1: dict[str, list[str]] = {}
        hard_negs_by_s1: dict[str, list[str]] = {}

        for row in ds:
            s1, s2, label = str(row["sentence1"]), str(row["sentence2"]), int(row["label"])
            if label == 1:
                positives_by_s1.setdefault(s1, []).append(s2)
            else:
                hard_negs_by_s1.setdefault(s1, []).append(s2)

        examples: list[RoutingTrainingExample] = []
        for anchor, positives in positives_by_s1.items():
            negs = hard_negs_by_s1.get(anchor, [])[:3]
            for pos in positives:
                examples.append(RoutingTrainingExample(query=anchor, positive=pos, negatives=negs))
                if limit and len(examples) >= limit:
                    return examples
        return examples

    def load_qqp(self, *, limit: int | None = None, split: str = "train") -> list[RoutingTrainingExample]:
        """Load QQP (Quora Question Pairs) paraphrase pairs (no hard negatives)."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("pip install 'datasets'") from exc

        ds = load_dataset("glue", "qqp", split=split, streaming=True)
        examples: list[RoutingTrainingExample] = []
        for row in ds:
            if int(row["label"]) != 1:
                continue
            examples.append(RoutingTrainingExample(
                query=str(row["question1"]),
                positive=str(row["question2"]),
                negatives=[],
            ))
            if limit and len(examples) >= limit:
                break
        return examples

    def load_nli_entailment(self, *, limit: int | None = None) -> list[RoutingTrainingExample]:
        """Load MultiNLI entailment pairs (label=0) as symmetric positives."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("pip install 'datasets'") from exc

        ds = load_dataset("multi_nli", split="train", streaming=True)
        examples: list[RoutingTrainingExample] = []
        for row in ds:
            if int(row["label"]) != 0:
                continue
            examples.append(RoutingTrainingExample(
                query=str(row["premise"]),
                positive=str(row["hypothesis"]),
                negatives=[],
            ))
            if limit and len(examples) >= limit:
                break
        return examples

    def load_banking77(self, *, limit: int | None = None, split: str = "train") -> list[RoutingTrainingExample]:
        """Load Banking77 intent classification dataset as symmetric paraphrase clusters.

        Queries are mapped to the first text of their corresponding intent label
        as the canonical representative. Deterministic hard negatives are sampled from
        other intent labels using a seeded random generator.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("pip install 'datasets'") from exc

        # Use the parquet-compatible version of banking77 from mteb
        ds = load_dataset("mteb/banking77", split=split)
        
        from collections import defaultdict
        import random

        by_label = defaultdict(list)
        for row in ds:
            by_label[int(row["label"])].append(str(row["text"]))

        examples: list[RoutingTrainingExample] = []
        labels = sorted(list(by_label.keys()))
        rng = random.Random(42)  # Seeded for determinism

        for label in labels:
            texts = by_label[label]
            if not texts:
                continue
            canonical = texts[0]
            for text in texts:
                # Select 3 deterministic negatives from other labels
                negs = []
                while len(negs) < min(3, len(labels) - 1):
                    other_label = rng.choice(labels)
                    if other_label != label and by_label[other_label]:
                        neg_text = rng.choice(by_label[other_label])
                        if neg_text not in negs:
                            negs.append(neg_text)
                examples.append(RoutingTrainingExample(
                    query=text,
                    positive=canonical,
                    negatives=negs,
                ))
                if limit and len(examples) >= limit:
                    return examples
        return examples

    def load_clinc150(self, *, limit: int | None = None, split: str = "train") -> list[RoutingTrainingExample]:
        """Load CLINC150 intent classification dataset as symmetric paraphrase clusters.

        Queries are mapped to the first text of their corresponding intent label
        as the canonical representative. Deterministic hard negatives are sampled from
        other intent labels using a seeded random generator.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("pip install 'datasets'") from exc

        # config="plus" is the standard setup with out-of-scope examples
        ds = load_dataset("clinc_oos", "plus", split=split)
        
        from collections import defaultdict
        import random

        by_intent = defaultdict(list)
        for row in ds:
            by_intent[int(row["intent"])].append(str(row["text"]))

        examples: list[RoutingTrainingExample] = []
        intents = sorted(list(by_intent.keys()))
        rng = random.Random(42)  # Seeded for determinism

        for intent in intents:
            texts = by_intent[intent]
            if not texts:
                continue
            canonical = texts[0]
            for text in texts:
                # Select 3 deterministic negatives from other intents
                negs = []
                while len(negs) < min(3, len(intents) - 1):
                    other_intent = rng.choice(intents)
                    if other_intent != intent and by_intent[other_intent]:
                        neg_text = rng.choice(by_intent[other_intent])
                        if neg_text not in negs:
                            negs.append(neg_text)
                examples.append(RoutingTrainingExample(
                    query=text,
                    positive=canonical,
                    negatives=negs,
                ))
                if limit and len(examples) >= limit:
                    return examples
        return examples

    def load_val_clusters(self) -> list[RoutingTrainingExample]:
        """Convert the validation clusters (self.val_clusters) into training examples.

        Each sentence in a validation cluster is mapped to the first sentence in that
        cluster as the canonical representative. Deterministic hard negatives are sampled
        from other validation clusters using a seeded random generator.
        """
        import random
        rng = random.Random(42)  # Seeded for determinism

        examples: list[RoutingTrainingExample] = []
        labels = sorted(list(self.val_clusters.keys()))

        for label in labels:
            texts = self.val_clusters[label]
            if not texts:
                continue
            canonical = texts[0]
            for text in texts:
                # Select 3 deterministic negatives from other validation clusters
                negs = []
                while len(negs) < min(3, len(labels) - 1):
                    other_label = rng.choice(labels)
                    if other_label != label and self.val_clusters[other_label]:
                        neg_text = rng.choice(self.val_clusters[other_label])
                        if neg_text not in negs:
                            negs.append(neg_text)
                examples.append(RoutingTrainingExample(
                    query=text,
                    positive=canonical,
                    negatives=negs,
                ))
        return examples

    # ------------------------------------------------------------------
    # Observatory evaluation
    # ------------------------------------------------------------------

    def eval_with_observatory(self) -> dict:
        """Run a full Observatory audit on val_clusters with the current encoder.

        Measures BOTH fragmentation (intra-cluster) AND separation (inter-cluster).
        A collapsed encoder (everything → same cell) gets fragmentation=1.0 but separation=0.0.
        Both scores must be reported — do not use fragmentation alone as a success gate.
        """
        _set_encoder_eval(self.encoder)
        all_texts = [t for texts in self.val_clusters.values() for t in texts]

        temp_idx = LatticeIndex.__new__(LatticeIndex)
        temp_idx._mode = "cache"
        temp_idx._init_with_encoder(self.encoder, d_model=self.d_model)
        temp_idx.add(all_texts)
        obs = LatticeObservatory(temp_idx)

        # Intra-cluster: do paraphrases land in the same cell?
        cluster_scores: dict[str, float] = {}
        for name, texts in self.val_clusters.items():
            frag = obs.fragmentation_score(texts)
            cluster_scores[name] = frag["score"]
        mean_frag = float(np.mean(list(cluster_scores.values())))

        # Inter-cluster: do DIFFERENT concepts land in DIFFERENT cells?
        # Pick one text per cluster, compute pairwise E8 key comparisons.
        cluster_anchors = [texts[0] for texts in self.val_clusters.values()]
        anchor_embs = self.encoder.encode(cluster_anchors, normalize_embeddings=True)
        from latticememory.rag.e8_retriever import E8LatticeDB
        lattice = E8LatticeDB(d_model=self.d_model)
        anchor_keys = [
            lattice._quantize_to_indices(torch.tensor(e, dtype=torch.float32))
            for e in anchor_embs
        ]
        n_anchors = len(anchor_keys)
        cross_pairs = 0
        different_pairs = 0
        for i in range(n_anchors):
            for j in range(i + 1, n_anchors):
                cross_pairs += 1
                if anchor_keys[i] != anchor_keys[j]:
                    different_pairs += 1
        separation = different_pairs / cross_pairs if cross_pairs > 0 else 0.0

        # Trajectory on the largest cluster
        traj_texts = max(self.val_clusters.values(), key=len)
        traj = obs.address_trajectory(traj_texts)

        # Block correlation
        sample = all_texts[:min(30, len(all_texts))]
        corr = obs.block_correlation(sample)

        # Cache safety: true paraphrases should be below threshold, near-misses above it.
        # Use within-cluster pairs as positives and cross-cluster anchors as near-misses.
        paraphrase_pairs: list[tuple[str, str]] = []
        for texts in self.val_clusters.values():
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    paraphrase_pairs.append((texts[i], texts[j]))

        cluster_items = list(self.val_clusters.items())
        near_miss_pairs: list[tuple[str, str]] = []
        for i, (_name_a, texts_a) in enumerate(cluster_items):
            for _name_b, texts_b in cluster_items[i + 1:]:
                if texts_a and texts_b:
                    near_miss_pairs.append((texts_a[0], texts_b[0]))

        if paraphrase_pairs and near_miss_pairs:
            from latticememory.hamming_router import HammingRouter
            router = HammingRouter(encoder=self.encoder, d_model=self.d_model)
            gap = router.gap_stats(paraphrase_pairs, near_miss_pairs)
            calibrated = router.calibrate_threshold(
                paraphrase_pairs, near_miss_pairs, fp_budget=0.0
            )
            hamming_gap = float(gap["gap"])
            zero_fp_recall = float(calibrated["recall"])
            zero_fp_threshold = int(calibrated["threshold"])
            paraphrase_hamming_mean = float(gap["paraphrase"]["mean"])
            near_miss_hamming_mean = float(gap["near_miss"]["mean"])
        else:
            hamming_gap = 0.0
            zero_fp_recall = 0.0
            zero_fp_threshold = 0
            paraphrase_hamming_mean = 0.0
            near_miss_hamming_mean = 0.0

        is_collapsed = mean_frag > 0.5 and separation < 0.3

        return {
            "mean_fragmentation_score": round(mean_frag, 4),
            "separation_score": round(separation, 4),
            "hamming_gap": round(hamming_gap, 4),
            "zero_fp_recall": round(zero_fp_recall, 4),
            "zero_fp_threshold": zero_fp_threshold,
            "paraphrase_hamming_mean": round(paraphrase_hamming_mean, 4),
            "near_miss_hamming_mean": round(near_miss_hamming_mean, 4),
            "is_collapsed": is_collapsed,
            "cluster_scores": {k: round(v, 4) for k, v in cluster_scores.items()},
            "mean_hamming_per_step": round(traj["mean_hamming_per_step"], 2),
            "trajectory_continuity": round(traj["trajectory_continuity"], 4),
            "mean_inter_block_nmi": round(corr.get("mean_inter_block_nmi", 0.0), 4),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        examples: list[RoutingTrainingExample],
        config: SnapTrainingConfig | None = None,
    ) -> SnapTrainResult:
        """Train with live Observatory eval (fragmentation + separation) every N steps.

        Layer freezing (config.freeze_layers) prevents address pressure from destroying
        the base semantic representation. With 20/24 layers frozen, the model fine-tunes
        only the last 4 layers + pooler — enough plasticity to learn E8 alignment, not
        enough to collapse the entire representation space.
        """
        if not examples:
            raise ValueError("examples must not be empty")
        if config is None:
            config = SnapTrainingConfig()

        device_str = config.device if config.device != "auto" else self.device
        train_device = torch.device(device_str)
        output_path = Path(config.output_dir) if config.output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        _move_encoder_to_device(self.encoder, train_device)

        # Freeze first N layers before enabling gradient checkpointing
        n_frozen = _freeze_encoder_layers(self.encoder, config.freeze_layers)

        if config.gradient_checkpointing:
            _enable_gradient_checkpointing(self.encoder)

        loss_fn = E8RoutingLoss(
            d_model=self.d_model,
            temperature=config.temperature,
            lambda_address=config.lambda_address,
            lambda_hamming=config.lambda_hamming,
            lambda_negative=config.lambda_negative,
            lambda_near_miss=config.lambda_near_miss,
            near_miss_margin=config.near_miss_margin,
            lambda_address_hinge=config.lambda_address_hinge,
            address_hinge_margin=config.address_hinge_margin,
            lambda_address_mse=config.lambda_address_mse,
            lambda_soft_hard=config.lambda_soft_hard,
            soft_hard_temperature=config.soft_hard_temperature_start,
            soft_hard_straight_through=config.soft_hard_straight_through,
            soft_hard_focal_gamma=config.soft_hard_focal_gamma,
            soft_hard_top_k_blocks=config.soft_hard_top_k_blocks,
        ).to(train_device)

        params = [p for p in _encoder_parameters(self.encoder) if p.requires_grad]
        if not params:
            raise ValueError("encoder has no trainable parameters — all layers frozen?")

        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        n = len(examples)
        total_batches_per_epoch = (n + config.batch_size - 1) // config.batch_size
        steps_per_epoch = max(
            1,
            (total_batches_per_epoch + config.gradient_accumulation_steps - 1)
            // config.gradient_accumulation_steps,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=config.epochs * steps_per_epoch,
            pct_start=0.1,
        )
        use_amp = bool(config.fp16 and train_device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        rng = torch.Generator()
        rng.manual_seed(config.seed)

        epoch_metrics_all: list[SnapEpochMetrics] = []
        obs_checkpoints: list[SnapObsCheckpoint] = []
        best_fragmentation = -1.0
        best_separation = 0.0
        best_hamming_gap = 0.0
        best_zero_fp_recall = 0.0
        best_zero_fp_threshold = 0
        best_global_step = 0
        global_step = 0
        examples_seen = 0
        training_start = time.perf_counter()

        progress_path = output_path / "snap_progress.jsonl" if output_path else None
        if progress_path:
            progress_path.write_text("", encoding="utf-8")

        def _log(row: dict) -> None:
            print(json.dumps(row, sort_keys=True), flush=True)
            if progress_path:
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

        _log({
            "metric": "training_init",
            "n_examples": n,
            "n_trainable_params": sum(p.numel() for p in params),
            "n_frozen_params": n_frozen,
            "freeze_layers": config.freeze_layers,
            "lambda_address": config.lambda_address,
            "lambda_negative": config.lambda_negative,
            "lambda_near_miss": config.lambda_near_miss,
            "near_miss_margin": config.near_miss_margin,
            "lambda_hamming": config.lambda_hamming,
            "lambda_soft_hard": config.lambda_soft_hard,
            "soft_hard_temperature_start": config.soft_hard_temperature_start,
            "soft_hard_temperature_end": config.soft_hard_temperature_end,
            "soft_hard_straight_through": config.soft_hard_straight_through,
            "soft_hard_focal_gamma": config.soft_hard_focal_gamma,
            "soft_hard_top_k_blocks": config.soft_hard_top_k_blocks,
        })

        total_optimizer_steps = max(1, config.epochs * steps_per_epoch)

        def _soft_hard_temperature(step: int) -> float:
            if total_optimizer_steps <= 1:
                return float(config.soft_hard_temperature_end)
            progress = min(1.0, max(0.0, step / float(total_optimizer_steps - 1)))
            start = float(config.soft_hard_temperature_start)
            end = float(config.soft_hard_temperature_end)
            return start + ((end - start) * progress)

        def _run_obs_eval(
            epoch: int,
            step_in_epoch: int,
            recent_losses: list,
            recent_addr: list,
            recent_hinge: list,
            recent_mse: list,
            recent_soft_hard: list,
            recent_target_prob: list,
        ) -> SnapObsCheckpoint | None:
            nonlocal best_fragmentation, best_separation, best_hamming_gap, best_zero_fp_recall, best_zero_fp_threshold, best_global_step
            t0 = time.perf_counter()
            with torch.no_grad():
                obs = self.eval_with_observatory()
            _set_encoder_train(self.encoder)
            obs_sec = round(time.perf_counter() - t0, 2)

            mean_frag = obs["mean_fragmentation_score"]
            separation = obs["separation_score"]
            is_collapsed = obs["is_collapsed"]

            # Best = fragmentation first, then cache-safety tie-breakers.
            is_best = (not is_collapsed) and _is_better_snap_checkpoint(
                mean_fragmentation=mean_frag,
                hamming_gap=obs["hamming_gap"],
                zero_fp_recall=obs["zero_fp_recall"],
                best_fragmentation=best_fragmentation,
                best_hamming_gap=best_hamming_gap,
                best_zero_fp_recall=best_zero_fp_recall,
            )
            if is_best:
                best_fragmentation = mean_frag
                best_separation = separation
                best_hamming_gap = obs["hamming_gap"]
                best_zero_fp_recall = obs["zero_fp_recall"]
                best_zero_fp_threshold = obs["zero_fp_threshold"]
                best_global_step = global_step
                if output_path:
                    best_path = output_path / "best_snap_encoder"
                    save = getattr(self.encoder, "save", None)
                    if callable(save):
                        save(str(best_path))

            recent_frags = [c.mean_fragmentation_score for c in obs_checkpoints[-4:]]
            trend = (
                "improving" if (recent_frags and mean_frag > recent_frags[-1]) else
                "stable" if (recent_frags and abs(mean_frag - recent_frags[-1]) < 0.005) else
                "declining" if recent_frags else "first_check"
            )

            ckpt = SnapObsCheckpoint(
                epoch=epoch,
                global_step=global_step,
                batch_in_epoch=step_in_epoch,
                examples_seen=examples_seen,
                mean_fragmentation_score=mean_frag,
                separation_score=separation,
                hamming_gap=obs["hamming_gap"],
                zero_fp_recall=obs["zero_fp_recall"],
                zero_fp_threshold=obs["zero_fp_threshold"],
                cluster_scores=obs["cluster_scores"],
                mean_hamming_per_step=obs["mean_hamming_per_step"],
                trajectory_continuity=obs["trajectory_continuity"],
                mean_inter_block_nmi=obs["mean_inter_block_nmi"],
                address_loss_recent=round(float(np.mean(recent_addr)) if recent_addr else 0.0, 4),
                train_loss_recent=round(float(np.mean(recent_losses)) if recent_losses else 0.0, 4),
                is_best=is_best,
                is_collapsed=is_collapsed,
                elapsed_sec=round(time.perf_counter() - training_start, 1),
                address_hinge_loss_recent=round(float(np.mean(recent_hinge)) if recent_hinge else 0.0, 4),
                address_mse_loss_recent=round(float(np.mean(recent_mse)) if recent_mse else 0.0, 4),
                soft_hard_loss_recent=round(float(np.mean(recent_soft_hard)) if recent_soft_hard else 0.0, 4),
                target_cell_probability_recent=round(float(np.mean(recent_target_prob)) if recent_target_prob else 0.0, 4),
                soft_hard_temperature=round(float(loss_fn.soft_hard_temperature), 6),
                paraphrase_hamming_mean=obs["paraphrase_hamming_mean"],
                near_miss_hamming_mean=obs["near_miss_hamming_mean"],
            )

            _log({
                "metric": "obs_checkpoint",
                "epoch": epoch,
                "global_step": global_step,
                "examples_seen": examples_seen,
                "mean_fragmentation_score": mean_frag,
                "separation_score": separation,
                "hamming_gap": obs["hamming_gap"],
                "zero_fp_recall": obs["zero_fp_recall"],
                "zero_fp_threshold": obs["zero_fp_threshold"],
                "paraphrase_hamming_mean": obs["paraphrase_hamming_mean"],
                "near_miss_hamming_mean": obs["near_miss_hamming_mean"],
                "is_collapsed": is_collapsed,
                "cluster_scores": obs["cluster_scores"],
                "mean_hamming_per_step": obs["mean_hamming_per_step"],
                "mean_inter_block_nmi": obs["mean_inter_block_nmi"],
                "address_loss_recent": ckpt.address_loss_recent,
                "train_loss_recent": ckpt.train_loss_recent,
                "address_hinge_loss_recent": ckpt.address_hinge_loss_recent,
                "address_mse_loss_recent": ckpt.address_mse_loss_recent,
                "soft_hard_loss_recent": ckpt.soft_hard_loss_recent,
                "target_cell_probability_recent": ckpt.target_cell_probability_recent,
                "soft_hard_temperature": ckpt.soft_hard_temperature,
                "trend": trend,
                "is_best": is_best,
                "obs_eval_sec": obs_sec,
            })
            return ckpt

        reached_target = False

        for epoch in range(1, config.epochs + 1):
            _set_encoder_train(self.encoder)
            optimizer.zero_grad(set_to_none=True)
            perm = torch.randperm(n, generator=rng)
            epoch_started = time.perf_counter()
            epoch_losses: list[float] = []
            epoch_address: list[float] = []
            epoch_contrastive: list[float] = []
            epoch_address_hinge: list[float] = []
            epoch_address_mse: list[float] = []
            epoch_soft_hard: list[float] = []
            epoch_target_prob: list[float] = []
            recent_losses: list[float] = []
            recent_addr: list[float] = []
            recent_hinge: list[float] = []
            recent_mse: list[float] = []
            recent_soft_hard: list[float] = []
            recent_target_prob: list[float] = []
            total_batches = max(1, (n + config.batch_size - 1) // config.batch_size)
            step_in_epoch = 0

            for batch_num, start in enumerate(range(0, n, config.batch_size), start=1):
                batch_idx = perm[start: start + config.batch_size]
                batch = [examples[int(i)] for i in batch_idx.tolist()]
                queries = [e.query for e in batch]
                positives = [e.positive for e in batch]
                neg_texts = _select_batch_negatives(batch, epoch=epoch) if config.lambda_negative > 0 else []

                with torch.autocast(device_type=train_device.type, dtype=torch.float16, enabled=use_amp):
                    loss_fn.soft_hard_temperature = _soft_hard_temperature(global_step)
                    q_emb = _encode_trainable_texts(self.encoder, queries, device=train_device, d_model=self.d_model)
                    p_emb = _encode_trainable_texts(self.encoder, positives, device=train_device, d_model=self.d_model)
                    n_emb = (
                        _encode_trainable_texts(self.encoder, neg_texts, device=train_device, d_model=self.d_model).unsqueeze(1)
                        if neg_texts else None
                    )
                    out = loss_fn(q_emb, p_emb, n_emb)
                    scaled = out.total / config.gradient_accumulation_steps

                scaler.scale(scaled).backward()

                should_step = (batch_num % config.gradient_accumulation_steps == 0) or (start + config.batch_size >= n)
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    step_in_epoch += 1
                    examples_seen += len(batch)

                    epoch_losses.append(float(out.total.detach()))
                    epoch_address.append(float(out.address.detach()))
                    epoch_contrastive.append(float(out.contrastive.detach()))
                    epoch_address_hinge.append(float(out.address_hinge.detach()))
                    epoch_address_mse.append(float(out.address_mse.detach()))
                    epoch_soft_hard.append(float(out.soft_hard.detach()))
                    epoch_target_prob.append(float(out.target_cell_probability.detach()))
                    recent_losses.append(float(out.total.detach()))
                    recent_addr.append(float(out.address.detach()))
                    recent_hinge.append(float(out.address_hinge.detach()))
                    recent_mse.append(float(out.address_mse.detach()))
                    recent_soft_hard.append(float(out.soft_hard.detach()))
                    recent_target_prob.append(float(out.target_cell_probability.detach()))
                    if len(recent_losses) > config.obs_eval_every_steps:
                        recent_losses.pop(0)
                        recent_addr.pop(0)
                        recent_hinge.pop(0)
                        recent_mse.pop(0)
                        recent_soft_hard.pop(0)
                        recent_target_prob.pop(0)

                    if config.log_every_batches and (batch_num % (config.log_every_batches * config.gradient_accumulation_steps) == 0):
                        _log({
                            "metric": "train_progress",
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch": batch_num,
                            "total_batches": total_batches,
                            "loss": round(float(out.total.detach()), 4),
                            "address_loss": round(float(out.address.detach()), 4),
                            "address_hinge_loss": round(float(out.address_hinge.detach()), 4),
                            "address_mse_loss": round(float(out.address_mse.detach()), 4),
                            "soft_hard_loss": round(float(out.soft_hard.detach()), 4),
                            "target_cell_probability": round(float(out.target_cell_probability.detach()), 4),
                            "soft_hard_temperature": round(float(loss_fn.soft_hard_temperature), 6),
                            "elapsed_sec": round(time.perf_counter() - epoch_started, 1),
                        })

                    if global_step % config.obs_eval_every_steps == 0:
                        ckpt = _run_obs_eval(
                            epoch,
                            step_in_epoch,
                            recent_losses,
                            recent_addr,
                            recent_hinge,
                            recent_mse,
                            recent_soft_hard,
                            recent_target_prob,
                        )
                        if ckpt is not None:
                            obs_checkpoints.append(ckpt)
                            if _product_gate_passed(ckpt=ckpt, config=config):
                                _log({
                                    "metric": "target_reached",
                                    "gate": "product_zero_fp_recall",
                                    "global_step": global_step,
                                    "mean_fragmentation_score": ckpt.mean_fragmentation_score,
                                    "zero_fp_recall": ckpt.zero_fp_recall,
                                    "zero_fp_recall_target": config.zero_fp_recall_target,
                                    "separation_score": ckpt.separation_score,
                                    "epoch": epoch,
                                })
                                reached_target = True
                                break

                else:
                    epoch_losses.append(float(out.total.detach()))
                    epoch_address.append(float(out.address.detach()))
                    epoch_contrastive.append(float(out.contrastive.detach()))
                    epoch_address_hinge.append(float(out.address_hinge.detach()))
                    epoch_address_mse.append(float(out.address_mse.detach()))
                    epoch_soft_hard.append(float(out.soft_hard.detach()))
                    epoch_target_prob.append(float(out.target_cell_probability.detach()))

                if reached_target:
                    break

            # End-of-epoch eval
            ckpt = _run_obs_eval(
                epoch,
                step_in_epoch,
                recent_losses,
                recent_addr,
                recent_hinge,
                recent_mse,
                recent_soft_hard,
                recent_target_prob,
            )
            if ckpt is not None:
                obs_checkpoints.append(ckpt)
                if not reached_target:
                    if _product_gate_passed(ckpt=ckpt, config=config):
                        _log({
                            "metric": "target_reached",
                            "gate": "product_zero_fp_recall",
                            "global_step": global_step,
                            "mean_fragmentation_score": ckpt.mean_fragmentation_score,
                            "zero_fp_recall": ckpt.zero_fp_recall,
                            "zero_fp_recall_target": config.zero_fp_recall_target,
                            "separation_score": ckpt.separation_score,
                            "trigger": "end_of_epoch",
                        })
                        reached_target = True

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_addr = float(np.mean(epoch_address)) if epoch_address else 0.0
            mean_cont = float(np.mean(epoch_contrastive)) if epoch_contrastive else 0.0
            last_obs = obs_checkpoints[-1] if obs_checkpoints else None

            epoch_m = SnapEpochMetrics(
                epoch=epoch,
                train_loss=round(mean_loss, 6),
                address_loss=round(mean_addr, 6),
                contrastive_loss=round(mean_cont, 6),
                mean_fragmentation_score=last_obs.mean_fragmentation_score if last_obs else 0.0,
                separation_score=last_obs.separation_score if last_obs else 0.0,
                hamming_gap=last_obs.hamming_gap if last_obs else 0.0,
                zero_fp_recall=last_obs.zero_fp_recall if last_obs else 0.0,
                zero_fp_threshold=last_obs.zero_fp_threshold if last_obs else 0,
                mean_hamming_per_step=last_obs.mean_hamming_per_step if last_obs else 0.0,
                trajectory_continuity=last_obs.trajectory_continuity if last_obs else 0.0,
                mean_inter_block_nmi=last_obs.mean_inter_block_nmi if last_obs else 0.0,
                cluster_scores=last_obs.cluster_scores if last_obs else {},
                elapsed_sec=round(time.perf_counter() - epoch_started, 1),
                is_best=(last_obs.is_best if last_obs else False),
                address_hinge_loss=round(float(np.mean(epoch_address_hinge)) if epoch_address_hinge else 0.0, 6),
                address_mse_loss=round(float(np.mean(epoch_address_mse)) if epoch_address_mse else 0.0, 6),
                soft_hard_loss=round(float(np.mean(epoch_soft_hard)) if epoch_soft_hard else 0.0, 6),
                target_cell_probability=round(float(np.mean(epoch_target_prob)) if epoch_target_prob else 0.0, 6),
                soft_hard_temperature=round(float(loss_fn.soft_hard_temperature), 6),
                paraphrase_hamming_mean=last_obs.paraphrase_hamming_mean if last_obs else 0.0,
                near_miss_hamming_mean=last_obs.near_miss_hamming_mean if last_obs else 0.0,
            )
            epoch_metrics_all.append(epoch_m)

            _log({
                "metric": "epoch_summary",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_m.train_loss,
                "address_loss": epoch_m.address_loss,
                "address_hinge_loss": epoch_m.address_hinge_loss,
                "address_mse_loss": epoch_m.address_mse_loss,
                "soft_hard_loss": epoch_m.soft_hard_loss,
                "target_cell_probability": epoch_m.target_cell_probability,
                "soft_hard_temperature": epoch_m.soft_hard_temperature,
                "mean_fragmentation_score": epoch_m.mean_fragmentation_score,
                "separation_score": epoch_m.separation_score,
                "hamming_gap": epoch_m.hamming_gap,
                "zero_fp_recall": epoch_m.zero_fp_recall,
                "zero_fp_threshold": epoch_m.zero_fp_threshold,
                "paraphrase_hamming_mean": epoch_m.paraphrase_hamming_mean,
                "near_miss_hamming_mean": epoch_m.near_miss_hamming_mean,
                "mean_hamming_per_step": epoch_m.mean_hamming_per_step,
                "mean_inter_block_nmi": epoch_m.mean_inter_block_nmi,
                "elapsed_sec": epoch_m.elapsed_sec,
            })

            if reached_target:
                break

        _set_encoder_eval(self.encoder)
        if output_path:
            final_path = output_path / "final_snap_encoder"
            save = getattr(self.encoder, "save", None)
            if callable(save):
                save(str(final_path))

            summary = {
                "base_model": self.base_model,
                "d_model": self.d_model,
                "freeze_layers": config.freeze_layers,
                "lambda_address": config.lambda_address,
                "lambda_negative": config.lambda_negative,
                "lambda_near_miss": config.lambda_near_miss,
                "near_miss_margin": config.near_miss_margin,
                "lambda_address_hinge": config.lambda_address_hinge,
                "address_hinge_margin": config.address_hinge_margin,
                "lambda_address_mse": config.lambda_address_mse,
                "lambda_soft_hard": config.lambda_soft_hard,
                "soft_hard_temperature_start": config.soft_hard_temperature_start,
                "soft_hard_temperature_end": config.soft_hard_temperature_end,
                "soft_hard_straight_through": config.soft_hard_straight_through,
                "soft_hard_focal_gamma": config.soft_hard_focal_gamma,
                "soft_hard_top_k_blocks": config.soft_hard_top_k_blocks,
                "epochs_trained": len(epoch_metrics_all),
                "best_global_step": best_global_step,
                "best_fragmentation_score": round(best_fragmentation, 4),
                "best_separation_score": round(best_separation, 4),
                "best_hamming_gap": round(best_hamming_gap, 4),
                "best_zero_fp_recall": round(best_zero_fp_recall, 4),
                "best_zero_fp_threshold": best_zero_fp_threshold,
                "fragmentation_target": config.fragmentation_target,
                "fragmentation_metric_role": "research_exact_snap",
                "product_gate": "zero_fp_recall",
                "zero_fp_recall_target": config.zero_fp_recall_target,
                "separation_target": config.separation_target,
                "reached_target": reached_target,
                "val_cluster_names": list(self.val_clusters.keys()),
                "total_obs_checkpoints": len(obs_checkpoints),
                "epoch_metrics": [
                    {
                        "epoch": m.epoch,
                        "train_loss": m.train_loss,
                        "address_loss": m.address_loss,
                        "address_hinge_loss": m.address_hinge_loss,
                        "address_mse_loss": m.address_mse_loss,
                        "soft_hard_loss": m.soft_hard_loss,
                        "target_cell_probability": m.target_cell_probability,
                        "soft_hard_temperature": m.soft_hard_temperature,
                        "mean_fragmentation_score": m.mean_fragmentation_score,
                        "separation_score": m.separation_score,
                        "hamming_gap": m.hamming_gap,
                        "zero_fp_recall": m.zero_fp_recall,
                        "zero_fp_threshold": m.zero_fp_threshold,
                        "paraphrase_hamming_mean": m.paraphrase_hamming_mean,
                        "near_miss_hamming_mean": m.near_miss_hamming_mean,
                        "mean_hamming_per_step": m.mean_hamming_per_step,
                        "cluster_scores": m.cluster_scores,
                    }
                    for m in epoch_metrics_all
                ],
                "obs_checkpoints": [
                    {
                        "epoch": c.epoch,
                        "global_step": c.global_step,
                        "examples_seen": c.examples_seen,
                        "mean_fragmentation_score": c.mean_fragmentation_score,
                        "separation_score": c.separation_score,
                        "hamming_gap": c.hamming_gap,
                        "zero_fp_recall": c.zero_fp_recall,
                        "zero_fp_threshold": c.zero_fp_threshold,
                        "paraphrase_hamming_mean": c.paraphrase_hamming_mean,
                        "near_miss_hamming_mean": c.near_miss_hamming_mean,
                        "cluster_scores": c.cluster_scores,
                        "is_collapsed": c.is_collapsed,
                        "is_best": c.is_best,
                        "address_hinge_loss_recent": c.address_hinge_loss_recent,
                        "address_mse_loss_recent": c.address_mse_loss_recent,
                        "soft_hard_loss_recent": c.soft_hard_loss_recent,
                        "target_cell_probability_recent": c.target_cell_probability_recent,
                        "soft_hard_temperature": c.soft_hard_temperature,
                    }
                    for c in obs_checkpoints
                ],
            }
            (output_path / "snap_training_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )

        return SnapTrainResult(
            epoch_metrics=epoch_metrics_all,
            obs_checkpoints=obs_checkpoints,
            best_global_step=best_global_step,
            best_fragmentation_score=round(best_fragmentation, 4),
            best_separation_score=round(best_separation, 4),
            reached_target=reached_target,
            output_dir=output_path,
            base_model=self.base_model,
            d_model=self.d_model,
            examples_trained=examples_seen,
            best_hamming_gap=round(best_hamming_gap, 4),
            best_zero_fp_recall=round(best_zero_fp_recall, 4),
            best_zero_fp_threshold=best_zero_fp_threshold,
        )
