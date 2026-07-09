# Design Partner Quickstart

Three stages, in order of increasing setup cost. Start at stage 1 -- it works
with zero configuration -- and move to later stages once you want a
calibrated, false-positive-safe cache instead of the uncalibrated default
that stage 1 ships with. Every number below is from the real proof-pack run
against a public customer-support dataset
(`docs/proxy_pq_redis_flywheel_proof_pack_2026-07-03.md`), not a synthetic
benchmark, and each is attributed to the specific configuration that produced
it -- not assumed to transfer to a different configuration.

These commands were checked against this release's actual
`latticememory/cli.py`, `latticememory/proxy_server.py`, `Dockerfile`, and
`docker-compose.yml`, flag by flag and env-var by env-var. If something
doesn't behave as described, check `/v1/analytics` on your running proxy
first -- it reports your real hit rate and false-positive rate, which is
more useful than any number in this doc.

## Stage 1: Start the proxy (zero config)

```bash
OPENAI_API_KEY=sk-... docker-compose up
```

Point your OpenAI client at `http://localhost:8000` instead of
`https://api.openai.com/v1`. Everything else calls through to OpenAI as
normal.

Two matching mechanisms are live here by default, straight from
`docker-compose.yml`'s environment block and `proxy_server.py`'s own
defaults:

- **The E8 exact/near-exact key cache** -- always on, deterministic, reliable
  for literal repeat prompts.
- **The Hamming approximate router**, in `serve` mode
  (`LATTICE_HAMMING_MODE` defaults to `serve`, not `off`, when unset) at a
  distance threshold of 70 (`LATTICE_HAMMING_THRESHOLD=70` in
  `docker-compose.yml`) -- with no cosine gate.

That means stage 1 is **not** an exact-only, zero-false-positive-risk
configuration, even though it needs no setup -- the Hamming router is
already doing approximate matching on paraphrase-shaped queries out of the
box. The project's own guidance (see the main README's "Running
`--hamming-mode serve` in production" section) is not to trust that router
without a calibrated cosine gate first: distance alone can't fully separate
genuine paraphrases from same-template/different-topic queries. Treat stage
1 as "fastest way to see cache hits," not "safe to leave running against
real traffic" -- stage 2 makes the approximate matching that's already
active here actually safe.

For context on repeat-traffic volume: the proof-pack's `exact_string`
baseline (a plain Python dict keyed by normalized prompt string, used there
purely as a baseline for comparison, not what ships) caught 27-42% of a real
customer-support workload on literal repeats alone across the proof-pack's
runs (see the `exact_string` rows in the proof-pack doc). That's a reasonable
floor for what identical-repeat traffic alone gets you; the E8 key cache
behaves similarly for verbatim/near-verbatim text.

## Stage 2: Calibrate the Hamming router

Calibrate a similarity threshold from a small set of your own paraphrase and
near-miss examples, then turn on the cosine gate so the approximate matching
already active in stage 1 stops guessing blind:

```bash
lattice calibrate \
  --paraphrases your_paraphrases.txt \
  --near-misses your_near_misses.txt \
  --metric cosine --fp-budget 0
```

Each file is `text_a|||text_b` per line -- `your_paraphrases.txt` holds pairs
that mean the same thing, `your_near_misses.txt` holds pairs that are similar
wording but a genuinely different question (these teach the calibration
where NOT to match). A few dozen pairs of each is enough to start; more
improves the calibrated threshold. `lattice calibrate` runs locally against
your files and an encoder -- it doesn't need a running proxy -- and prints a
recommended threshold.

Apply that threshold to the Docker proxy through environment variables, not
CLI flags. The Docker image's `ENTRYPOINT` always runs the ASGI server
directly (`uvicorn latticememory.proxy_server:app ...`); passing
`lattice serve --hamming-mode ...` as extra arguments to `docker-compose run`
does **not** work the way it might look -- those arguments get appended
after the fixed entrypoint's own arguments rather than replacing it with a
new command, so `lattice` is never actually invoked inside the container.
Configuration for the Docker path is env-var only:

```bash
docker-compose run --rm --service-ports \
  -e LATTICE_HAMMING_MODE=serve \
  -e LATTICE_HAMMING_COSINE_GATE=true \
  -e LATTICE_HAMMING_COSINE_THRESHOLD=<the threshold lattice calibrate printed> \
  latticememory-proxy
```

`--service-ports` is required here: unlike `docker-compose up`,
`docker-compose run` does not publish the service's `ports:` mapping by
default -- without it your OpenAI client can't reach `localhost:8000`.

(If you're running the proxy directly with `pip install
'lattice-memory-e8[proxy]'` instead of Docker, the equivalent is
`lattice serve --hamming-mode serve --hamming-cosine-gate
--hamming-cosine-threshold <threshold> --key sk-...` -- see `lattice serve
--help` for the full flag list.)

## Stage 3: PQ + Redis

This sets up the same architecture the proof-pack's validated configuration
uses -- PQ-backed candidate generation plus a cosine gate before serving --
applied to your own data. It needs one file of your own question/answer
pairs, not a labeled four-way split, just what a support team already has:

```jsonl
{"question": "What is your refund policy?", "answer": "30-day returns, full refund to original payment method."}
{"question": "How do I reset my password?", "answer": "Use the 'Forgot password' link on the sign-in page."}
```

**Your file needs at least as many Q&A pairs as `--pq-codebook-size`
(`LATTICE_PQ_CODEBOOK_SIZE`), which defaults to 256.** PQ codebooks are fit
by clustering your own rows into that many centroids, and fitting more
centroids than you have data points is not possible -- `--pq-mode` fails
fast at startup with a clear error naming your actual count if your file is
smaller than the codebook size. The two-pair example above is illustrative
of the *shape* of the file, not something you can run as-is with the
default codebook size. If you have a few dozen real pairs rather than 256+,
pass a smaller codebook explicitly, sized to roughly match your data volume
-- e.g. `--pq-codebook-size 16` (or `-e LATTICE_PQ_CODEBOOK_SIZE=16` for the
Docker path below). Smaller codebooks trade match quality for working at
all with less data; grow the codebook size as your Q&A file grows.

Save your real file as `qa_pairs.jsonl` in this repo's working directory.
Redis is behind a Compose profile, so it doesn't start with plain
`docker-compose up` -- start it explicitly first, then run the proxy
against it:

```bash
docker-compose --profile with-redis up -d redis
```

```bash
docker-compose run --rm --service-ports \
  -v "$(pwd)/qa_pairs.jsonl:/data/qa_pairs.jsonl" \
  -e LATTICE_WARM_PATH=/data/qa_pairs.jsonl \
  -e LATTICE_PQ_MODE=true \
  -e LATTICE_CACHE_COSINE_GATE=true \
  -e LATTICE_REDIS_URL=redis://redis:6379/0 \
  latticememory-proxy
```

What each piece is doing, and why it's there:

- **`docker-compose --profile with-redis up -d redis` is a separate step**
  because `latticememory-proxy` has no `depends_on: redis` in
  `docker-compose.yml`. Running `docker-compose run latticememory-proxy`
  alone -- even with `--profile with-redis` tacked onto that same command --
  will not start Redis for you; `run` only starts the target service and its
  declared dependencies, and this service declares none.
- **The `-v ".../qa_pairs.jsonl:/data/qa_pairs.jsonl"` mount is required**
  because `/data` inside the container is a Docker-managed named volume
  (`lattice-data`), not a bind mount of your working directory -- your file
  is not visible inside the container until you mount it explicitly.
- **`LATTICE_PQ_MODE=true`** fits PQ codebooks (8 blocks, 256 codewords --
  the corrected, validated default as of this release) from the same file
  being seeded, and builds the semantic cache from it directly.
  `--pq-mode`/`LATTICE_PQ_MODE` requires `LATTICE_WARM_PATH` -- there's
  nothing to fit codebooks from otherwise, and the proxy fails to start with
  a clear error if it's missing rather than silently serving an empty cache.
- **`LATTICE_CACHE_COSINE_GATE=true` is what makes this the *validated* path**
  instead of raw PQ. Without it, PQ candidate matching is real but has a
  measured false-positive rate -- roughly 2-10% across the proof-pack's
  runs, depending on dataset size, never zero. The proof-pack's headline
  "99.17% hit rate, 0% measured false positives, 6,000-request replay"
  number came from a cosine-gated configuration specifically (see
  `docs/public_proxy_demo_runbook_2026-07-04.md`, step 3), and that
  particular run used `--pq-proof-dataset` with different PQ parameters (4
  blocks / 4 codewords) rather than `--pq-mode` with this release's
  corrected 8/256 default. Treat this stage as "the same validated
  architecture, applied to your own data," not "guaranteed to reproduce
  99.17% exactly" -- check your own `/v1/analytics` for your real numbers.
- The default cosine threshold (`LATTICE_CACHE_COSINE_THRESHOLD`, 0.999)
  already matches what the proof-pack's validated run used, so you don't
  need to set it explicitly unless you want a different tradeoff.

## What to expect, honestly

The 99.17% figure is against a workload where most traffic really is
repeated or lightly-paraphrased customer-support questions, under a
cosine-validated PQ+Redis configuration -- if your traffic is more
open-ended, or you skip the cosine gate, expect a different number. Check
`/v1/analytics` on your running proxy for your own real hit rate and
false-positive rate rather than assuming the proof-pack's numbers transfer
directly. Asymmetric workloads (a user question against a much longer
document, not another short question) are not what this stage is for -- see
the main README's "What it's for" table.
