"""Run gap_stats on the hard near-miss helpdesk calibration dataset.

Measures intra-intent Hamming distances (paraphrase pairs) vs inter-intent
Hamming distances (confusable near-miss pairs) for the helpdesk domain.

Output tells us whether there is a threshold that separates same-intent
paraphrases from cross-intent near-misses on this domain.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

sys.path.insert(0, str(Path(__file__).parent.parent))

CAL_PATH = Path(__file__).parent / "demo_data" / "hard_near_miss_helpdesk" / "calibration_data.json"
BASE_MODEL = "dfrokido/bge-large-e8-snap"


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from latticememory import HammingRouter

    print("=" * 65)
    print("Helpdesk domain: Hamming gap_stats")
    print(f"  model: {BASE_MODEL}")
    print("=" * 65)

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    para_pairs  = [tuple(p) for p in cal["paraphrases"]]
    nm_pairs    = [tuple(p) for p in cal["near_misses"]]
    print(f"  {len(para_pairs)} paraphrase pairs  {len(nm_pairs)} near-miss pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading encoder (device={device}) ...")
    encoder = SentenceTransformer(BASE_MODEL, device=device)
    d_model = encoder.get_embedding_dimension()
    router = HammingRouter(encoder=encoder, d_model=d_model, threshold=70)

    print("Computing gap_stats ...")
    stats = router.gap_stats(para_pairs, nm_pairs)

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    p = stats["paraphrase"]
    n = stats["near_miss"]
    print(f"  Paraphrase:  n={p['n']}  min={p['min']}  p5={p['p5']:.1f}  "
          f"mean={p['mean']}  p95={p['p95']:.1f}  max={p['max']}")
    print(f"  Near-miss:   n={n['n']}  min={n['min']}  p5={n['p5']:.1f}  "
          f"mean={n['mean']}  p95={n['p95']:.1f}  max={n['max']}")
    print(f"  Gap (near_p5 - para_p95): {stats['gap']:+.1f}")
    print()

    print("  Threshold sweep:")
    print(f"  {'threshold':>10}  {'recall':>8}  {'fp_rate':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for row in stats["threshold_table"]:
        t = row["threshold"]
        marker = " <-- proxy default" if t == 70 else ""
        print(f"  {t:>10}  {row['recall']:>7.1%}  {row['fp_rate']:>7.1%}{marker}")

    print()
    if stats["gap"] > 0:
        result = router.calibrate_threshold(para_pairs, nm_pairs, fp_budget=0.0)
        print(f"  PASS: gap > 0  =>  calibrated threshold = {result['threshold']}")
        print(f"    recall={result['recall']:.1%}  fp_rate={result['fp_rate']:.1%}")
    else:
        print("  WARNING: gap <= 0 — distributions overlap on this domain.")
        print("  No zero-FP threshold with useful recall exists.")

    out = Path(__file__).parent / "results" / "helpdesk_gap_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
