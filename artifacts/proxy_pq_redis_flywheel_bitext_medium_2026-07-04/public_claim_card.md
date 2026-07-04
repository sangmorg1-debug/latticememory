# LatticeMemory Public Claim Card

## Supported Claim

LatticeMemory reduces repeated/paraphrased upstream calls on a measured support-style workload while reporting hit rate, latency, savings, Redis memory, flywheel review behavior, and false-positive rate together.

## Safest Zero-FP Measured Row

- Run: `dense_cosine`
- Hit rate: `1.0000`
- Upstream call rate: `0.0000`
- False-positive rate: `0.0000`
- Adversarial false-positive rate: `0.0000`
- Redis memory MB: `0.0000`
- Flywheel miss clusters: `0`

## Highest-Hit Measured Row

- Run: `dense_cosine`
- Hit rate: `1.0000`
- Upstream call rate: `0.0000`
- False-positive rate: `0.0000`
- Adversarial false-positive rate: `0.0000`
- Redis memory MB: `0.0000`
- Flywheel miss clusters: `0`

Use the highest-hit row only with its measured false-positive rate attached.

## Target Product Policy Row

- Run: `lattice_pq_redis_validated_cosine`
- Hit rate: `0.9600`
- Upstream call rate: `0.0400`
- False-positive rate: `0.0000`
- Adversarial false-positive rate: `0.0000`
- Redis memory MB: `0.0511`
- Flywheel miss clusters: `0`

This is the product-shaped row to harden: PQ generates candidates, a validation gate decides whether they are safe to serve, and misses fall back upstream.

## Unsupported Claims

- LatticeMemory does not replace general-purpose vector databases.
- LatticeMemory does not prove general RAG superiority from this proof pack.
- PQ/Hamming cache hits are not assumed safe without false-positive measurement.
