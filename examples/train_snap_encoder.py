"""Train a symmetric E8 snap encoder from scratch using Observatory-guided eval.

This script trains a new snap encoder starting from BAAI/bge-large-en-v1.5 using
PAWS-wiki paraphrase pairs as training data. The LatticeObservatory fragmentation_score
is used as the primary eval metric instead of STS or recall@1.

Why from scratch:
  dfrokido/bge-large-e8-snap produced fragmentation_score=0.0 and mean_hamming=108.25
  on paraphrase chains — evidence it was trained with asymmetric MS MARCO data only.
  This script fixes that by training on symmetric paraphrase pairs.

Usage:
  python examples/train_snap_encoder.py                         # defaults
  python examples/train_snap_encoder.py --epochs 5 --limit 50000
  python examples/train_snap_encoder.py --data qqp --epochs 3 --output snap_v2/
  python examples/train_snap_encoder.py --data paws+qqp --epochs 5 --limit 100000

After training, evaluate with the observatory:
  python examples/evaluate_snap_encoder.py --model snap_v2/best_snap_encoder

Push to HuggingFace:
  huggingface-cli upload dfrokido/bge-large-e8-snap-v2 snap_v2/best_snap_encoder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a symmetric E8 snap encoder using Observatory-guided evaluation."
    )
    p.add_argument(
        "--base-model",
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model ID to fine-tune (default: BAAI/bge-large-en-v1.5)",
    )
    p.add_argument(
        "--data",
        default="paws",
        choices=["paws", "qqp", "nli", "paws+qqp", "paws+nli", "paws+qqp+nli", "banking77", "clinc150", "banking77+clinc150", "val", "banking77+val"],
        help="Training data source(s) (default: paws)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="Max training examples per source (default: 50000)",
    )
    p.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4)")
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8, effective batch=32)",
    )
    p.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate (default: 2e-5)")
    p.add_argument(
        "--lambda-address",
        type=float,
        default=0.0,
        help="E8 cell address cross-entropy weight (default: 0.0, disabled in favor of hinge/MSE losses)",
    )
    p.add_argument(
        "--lambda-address-hinge",
        type=float,
        default=3.0,
        help="E8 address block-level margin hinge loss weight (default: 3.0)",
    )
    p.add_argument(
        "--address-hinge-margin",
        type=float,
        default=0.2,
        help="Margin for block-level hinge loss (default: 0.2)",
    )
    p.add_argument(
        "--lambda-address-mse",
        type=float,
        default=3.0,
        help="E8 address block coordinate-level MSE loss weight (default: 3.0)",
    )
    p.add_argument(
        "--lambda-hamming",
        type=float,
        default=0.3,
        help="Hamming distance loss weight (default: 0.3)",
    )
    p.add_argument(
        "--lambda-negative",
        type=float,
        default=1.0,
        help="Hard negative loss weight — requires PAWS negatives in training data (default: 1.0)",
    )
    p.add_argument(
        "--lambda-near-miss",
        type=float,
        default=1.0,
        help="E8 near-miss hinge loss weight for pushing hard negatives beyond margin (default: 1.0)",
    )
    p.add_argument(
        "--near-miss-margin",
        type=float,
        default=80.0,
        help="Minimum expected E8 Hamming distance for hard negatives (default: 80)",
    )
    p.add_argument(
        "--freeze-layers",
        type=int,
        default=20,
        help="Freeze first N transformer layers (default: 20 for BGE-large 24-layer). Prevents collapse.",
    )
    p.add_argument(
        "--fragmentation-target",
        type=float,
        default=0.75,
        help="Fragmentation gate: stop early when mean_fragmentation_score >= this (default: 0.75)",
    )
    p.add_argument(
        "--separation-target",
        type=float,
        default=0.80,
        help="Separation gate: both must pass to declare success (default: 0.80)",
    )
    p.add_argument(
        "--output",
        default="snap_trained/",
        help="Output directory for checkpoints and summary (default: snap_trained/)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device (default: auto)",
    )
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed-precision training")
    p.add_argument(
        "--no-cluster-training",
        action="store_true",
        help="Disable connected-component canonical target mapping for paraphrase pairs",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation clusters: hand-picked concept groups that represent typical
# symmetric caching workloads. These are NOT in any training set.
# Add / remove clusters to match your target domain.
# ---------------------------------------------------------------------------
VAL_CLUSTERS: dict[str, list[str]] = {
    "refund_policy": [
        "I want a refund on my order.",
        "Can I get my money back?",
        "How do I return a product for a refund?",
        "I need to get reimbursed for a purchase.",
        "Is it possible to reverse my payment?",
    ],
    "order_status": [
        "Where is my order?",
        "Track my package.",
        "What is the status of my shipment?",
        "I want to know when my delivery will arrive.",
        "Has my order been dispatched?",
    ],
    "reset_password": [
        "How do I reset my password?",
        "I forgot my login credentials.",
        "Can you help me recover my account?",
        "I can't log in, need a password reset.",
        "Send me a link to change my password.",
    ],
    "opening_hours": [
        "What are your business hours?",
        "When do you open?",
        "What time does the store close?",
        "Are you open on weekends?",
        "What are your hours of operation?",
    ],
    "cancel_subscription": [
        "How do I cancel my subscription?",
        "I want to unsubscribe.",
        "Please cancel my membership.",
        "Stop billing me for the plan.",
        "End my recurring payment.",
    ],
}


def main() -> None:
    args = parse_args()

    print(json.dumps({"step": "init", "base_model": args.base_model, "data": args.data}))

    from latticememory.snap_trainer import SnapTrainer, SnapTrainingConfig
    from latticememory.training import RoutingTrainingExample, build_canonical_cluster_examples

    trainer = SnapTrainer(
        base_model=args.base_model,
        val_clusters=VAL_CLUSTERS,
        device=args.device,
    )

    print(json.dumps({"step": "encoder_loaded", "d_model": trainer.d_model}))

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    sources = args.data.split("+")
    all_examples: list[RoutingTrainingExample] = []

    for source in sources:
        print(json.dumps({"step": "loading_data", "source": source, "limit": args.limit}))
        if source == "paws":
            examples = trainer.load_paws(limit=args.limit)
        elif source == "qqp":
            examples = trainer.load_qqp(limit=args.limit)
        elif source == "nli":
            examples = trainer.load_nli_entailment(limit=args.limit)
        elif source == "banking77":
            examples = trainer.load_banking77(limit=args.limit)
        elif source == "clinc150":
            examples = trainer.load_clinc150(limit=args.limit)
        elif source == "val":
            examples = trainer.load_val_clusters()
        else:
            print(f"Unknown data source: {source}", file=sys.stderr)
            sys.exit(1)
        all_examples.extend(examples)
        print(json.dumps({"step": "data_loaded", "source": source, "n_examples": len(examples)}))

    print(json.dumps({"step": "total_examples", "n": len(all_examples)}))

    if not all_examples:
        print("ERROR: no training examples loaded", file=sys.stderr)
        sys.exit(1)

    if not args.no_cluster_training:
        before_unique_positives = len({example.positive for example in all_examples})
        all_examples = build_canonical_cluster_examples(all_examples)
        after_unique_positives = len({example.positive for example in all_examples})
        print(json.dumps({
            "step": "cluster_training_canonicalized",
            "n_examples": len(all_examples),
            "unique_positives_before": before_unique_positives,
            "unique_positives_after": after_unique_positives,
        }))
    else:
        print(json.dumps({
            "step": "cluster_training_disabled",
            "n_examples": len(all_examples),
        }))

    # ------------------------------------------------------------------
    # Run baseline Observatory eval before training
    # ------------------------------------------------------------------
    print(json.dumps({"step": "baseline_eval"}))
    baseline = trainer.eval_with_observatory()
    print(json.dumps({"step": "baseline_result", **baseline}))

    if baseline["mean_fragmentation_score"] >= args.fragmentation_target:
        print(json.dumps({
            "step": "skip_training",
            "reason": f"baseline fragmentation_score {baseline['mean_fragmentation_score']:.4f} "
                      f">= target {args.fragmentation_target}. Model already meets target.",
        }))
        return

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    config = SnapTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=1e-4,
        temperature=0.05,
        lambda_address=args.lambda_address,
        lambda_hamming=args.lambda_hamming,
        lambda_negative=args.lambda_negative,
        lambda_near_miss=args.lambda_near_miss,
        near_miss_margin=args.near_miss_margin,
        lambda_address_hinge=args.lambda_address_hinge,
        address_hinge_margin=args.address_hinge_margin,
        lambda_address_mse=args.lambda_address_mse,
        freeze_layers=args.freeze_layers,
        fp16=not args.no_fp16,
        gradient_checkpointing=True,
        log_every_batches=50,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
        fragmentation_target=args.fragmentation_target,
        separation_target=args.separation_target,
    )

    print(json.dumps({
        "step": "training_start",
        "epochs": args.epochs,
        "n_examples": len(all_examples),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lambda_address": args.lambda_address,
        "lambda_address_hinge": args.lambda_address_hinge,
        "address_hinge_margin": args.address_hinge_margin,
        "lambda_address_mse": args.lambda_address_mse,
        "lambda_negative": args.lambda_negative,
        "lambda_near_miss": args.lambda_near_miss,
        "near_miss_margin": args.near_miss_margin,
        "freeze_layers": args.freeze_layers,
        "fragmentation_target": args.fragmentation_target,
        "separation_target": args.separation_target,
        "output": args.output,
    }))

    result = trainer.train(all_examples, config)

    # ------------------------------------------------------------------
    # Final Observatory eval on the best checkpoint
    # ------------------------------------------------------------------
    best_path = Path(args.output) / "best_snap_encoder"
    if best_path.exists():
        print(json.dumps({"step": "loading_best_checkpoint", "path": str(best_path)}))
        from sentence_transformers import SentenceTransformer
        best_enc = SentenceTransformer(str(best_path))
        best_trainer = SnapTrainer._from_encoder(
            best_enc, val_clusters=VAL_CLUSTERS, d_model=trainer.d_model
        )
        final_eval = best_trainer.eval_with_observatory()
    else:
        final_eval = trainer.eval_with_observatory()

    print(json.dumps({"step": "final_eval", **final_eval}))
    print(json.dumps({
        "step": "done",
        "best_global_step": result.best_global_step,
        "best_fragmentation_score": result.best_fragmentation_score,
        "best_separation_score": result.best_separation_score,
        "reached_target": result.reached_target,
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }))

    if not result.reached_target:
        print(
            f"\nWARNING: Training completed but fragmentation_score "
            f"{result.best_fragmentation_score:.4f} < target {args.fragmentation_target}.\n"
            f"Consider: --lambda-address {args.lambda_address * 2:.0f}  (double address weight)\n"
            f"       or: --data paws+qqp  (more diverse paraphrase data)\n"
            f"       or: --epochs {args.epochs + 5}  (more training)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
