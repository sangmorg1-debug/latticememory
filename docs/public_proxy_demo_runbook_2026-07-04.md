# Public Proxy Demo Runbook

**Date:** 2026-07-04
**Purpose:** reproduce the LatticeMemory Redis-backed cache proof and run a
live proxy replay without making unsupported RedisVL or general RAG claims.

## 1. Start Redis Stack

RedisVL requires Redis Stack, not plain Redis.

```powershell
docker compose -f docker-compose.redis-stack.yml up -d
```

If Docker is available only inside WSL on Windows:

```powershell
wsl docker compose -f /mnt/e/latticememory/docker-compose.redis-stack.yml up -d
```

The runbook assumes Redis is reachable from Windows Python at:

```text
redis://localhost:6382/0
```

## 2. Reproduce The Proof Pack

This is the deterministic proof path. It uses the checked-in Bitext JSONL and
writes a fresh artifact directory.

```powershell
python - <<'PY'
from pathlib import Path
import json
from latticememory.proof_pack import run_proxy_pq_redis_flywheel_proof_pack

artifact_dir = Path("artifacts/proxy_demo_repro_2026-07-04")
dataset_path = Path("artifacts/proxy_pq_redis_flywheel_bitext_large_2026-07-04/bitext_support_dataset_large.jsonl")
summary = run_proxy_pq_redis_flywheel_proof_pack(
    artifact_dir,
    dataset_path=dataset_path,
    redis_url="redis://localhost:6382/0",
    redis_namespace="proof-pack-public-demo",
    include_competitor_baselines=True,
    progress_path=artifact_dir / "proof_pack_progress.jsonl",
)
(artifact_dir / "proof_pack_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
```

PowerShell-safe form:

```powershell
@'
from pathlib import Path
import json
from latticememory.proof_pack import run_proxy_pq_redis_flywheel_proof_pack

artifact_dir = Path("artifacts/proxy_demo_repro_2026-07-04")
dataset_path = Path("artifacts/proxy_pq_redis_flywheel_bitext_large_2026-07-04/bitext_support_dataset_large.jsonl")
summary = run_proxy_pq_redis_flywheel_proof_pack(
    artifact_dir,
    dataset_path=dataset_path,
    redis_url="redis://localhost:6382/0",
    redis_namespace="proof-pack-public-demo",
    include_competitor_baselines=True,
    progress_path=artifact_dir / "proof_pack_progress.jsonl",
)
(artifact_dir / "proof_pack_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
'@ | python -
```

Expected product-shaped row:

```text
lattice_pq_redis_validated_cosine
hit_rate ~= 0.9812
upstream_call_rate ~= 0.0188
false_positive_rate = 0.0000
adversarial_false_positive_rate = 0.0000
```

## 3. Run A Live Proxy

For a real upstream-backed proxy:

```powershell
lattice serve `
  --redis-url redis://localhost:6382/0 `
  --redis-namespace public-demo-live `
  --cache-cosine-gate `
  --cache-cosine-threshold 0.999 `
  --key $env:OPENAI_API_KEY `
  --port 8000
```

This validates the production HTTP path, Redis-backed cache storage, cache
cosine validation, and `/v1/analytics`.

Important distinction: `lattice serve` uses the standard proxy cache runtime. The
PQ proof pack uses a PQ-fitted cache constructed by the proof harness. Do not
claim the live `lattice serve` command alone reproduces the PQ row unless the
server is explicitly wired to a PQ-fitted `semantic_cache`.

## 4. Replay Through The Live Proxy

```powershell
python examples\proxy_live_replay_demo.py `
  --dataset-jsonl artifacts\proxy_pq_redis_flywheel_bitext_large_2026-07-04\bitext_support_dataset_large.jsonl `
  --base-url http://127.0.0.1:8000 `
  --model gpt-4o-mini `
  --output-json artifacts\proxy_live_replay_demo.json `
  --output-md artifacts\proxy_live_replay_demo.md `
  --output-html artifacts\proxy_live_replay_demo.html
```

If the proxy has `--admin-key`, add:

```powershell
--admin-key YOUR_ADMIN_KEY
```

Fetch analytics directly:

```powershell
lattice analytics --host 127.0.0.1 --port 8000
```

## 5. Public Claim Wording

Supported:

> On a 6,000-request public Bitext support workload, LatticeMemory's Redis-backed
> validated PQ cache row served 98.12% of requests with zero measured false
> positives, while reporting adversarial false positives, latency, Redis memory,
> and upstream-call rate beside the hit rate.

Not supported:

- LatticeMemory replaces RedisVL.
- LatticeMemory replaces vector databases.
- Raw PQ hits are safe without validation.
- This proves general RAG superiority.

## 6. Where RedisVL Fits

RedisVL direct is included as a baseline when RedisVL and Redis Stack are
available. In the current Bitext proof, RedisVL performs strongly. The product
claim should focus on LatticeMemory's auditability, explicit operating-policy
matrix, false-positive accounting, flywheel-ready proxy shape, and PQ +
validation mode rather than claiming blanket superiority.
