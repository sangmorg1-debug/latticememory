from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Ensure parent directory is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from latticememory.dual_encoder import LatticeDualEncoder, LinearAdapterEncoder, ResidualMLPAdapterEncoder
from latticememory.training import load_msmarco_examples, evaluate_routing_examples
from sentence_transformers import SentenceTransformer


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and diagnose E8 key alignment on MS MARCO.")
    parser.add_argument("--adapter", required=True, help="Path to query_adapter.pt")
    parser.add_argument("--model", default="dfrokido/bge-large-e8-snap", help="Base model name")
    parser.add_argument("--dataset-name", default="microsoft/ms_marco")
    parser.add_argument("--dataset-config", default="v1.1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200, help="Number of examples to evaluate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu is recommended to avoid GPU conflicts)")
    args = parser.parse_args()

    from benchmarks.common import load_text_encoder
    print(f"Loading base model: {args.model} on {args.device}...")
    encoder, d_model = load_text_encoder(args.model, d_model=None, device=args.device)

    print(f"Loading query adapter from: {args.adapter}...")
    # Attempt to load as residual MLP or linear
    try:
        query_encoder = ResidualMLPAdapterEncoder.load_adapter(
            base_encoder=encoder,
            path=args.adapter
        )
        print("Successfully loaded adapter as ResidualMLPAdapterEncoder")
    except Exception:
        try:
            query_encoder = LinearAdapterEncoder.load_adapter(
                base_encoder=encoder,
                path=args.adapter
            )
            print("Successfully loaded adapter as LinearAdapterEncoder")
        except Exception as e:
            print(f"Failed to load query adapter: {e}")
            return 1

    dual_encoder = LatticeDualEncoder(
        document_encoder=encoder,
        query_encoder=query_encoder,
        d_model=d_model,
        training_pairs=0,
        ridge=0.0
    )

    print(f"Loading {args.limit} MS MARCO examples ({args.split} split)...")
    examples = load_msmarco_examples(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        streaming=True,
        min_negatives=1,
        limit=args.limit
    )
    print(f"Loaded {len(examples)} examples.")

    print("\nEvaluating and calculating Hamming distance distribution...")
    metrics = evaluate_routing_examples(dual_encoder, examples, top_k=1)

    print("\n=========================================================================")
    print("                    E8 Routing Diagnostic Metrics                        ")
    print("=========================================================================")
    print(f"Total Evaluated:             {metrics['total']}")
    print(f"Lattice Routed:              {metrics['lattice_routed']} ({metrics['lattice_route_rate'] * 100:.2f}%)")
    print(f"Recall@1:                    {metrics['recall_at_1'] * 100:.2f}%")
    print(f"Path Breakdown:              {metrics['path_counts']}")
    print("-------------------------------------------------------------------------")
    print(f"Mean Hamming Distance:       {metrics['mean_hamming_distance']:.4f}")
    print(f"Min Hamming Distance:        {metrics['min_hamming_distance']}")
    print(f"Max Hamming Distance:        {metrics['max_hamming_distance']}")
    print(f"p50 Hamming Distance:        {metrics['p50_hamming_distance']:.2f}")
    print(f"p95 Hamming Distance:        {metrics['p95_hamming_distance']:.2f}")
    print(f"p99 Hamming Distance:        {metrics['p99_hamming_distance']:.2f}")
    print(f"Hamming Distance Histogram:  {metrics['hamming_distance_histogram']}")
    print("=========================================================================")

    print("\nSample Mismatches (First 5):")
    mismatches = [row for row in metrics["rows"] if not row["correct_at_1"] or row["hamming_distance_to_positive"] > 0]
    for idx, row in enumerate(mismatches[:5]):
        print(f"\nSample #{idx + 1}:")
        print(f"  Query:            {row['query']}")
        print(f"  Expected Positive: {row['expected']}")
        print(f"  Retrieval Path:   {row['path']}")
        print(f"  Hamming Distance:  {row['hamming_distance_to_positive']}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
