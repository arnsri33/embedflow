# Migration planner

`embedflow plan` is a bounded, advisory analysis for moving from an existing
source vector index to a target embedding model. It composes the existing
backend audit, model contracts, retained registry evidence, probe retrieval,
frozen T2-v1 diagnostic, cache behavior, and economics helpers. It never
routes traffic and does not create, update, or delete source-index data.

## Quickstart

Install EmbedFlow with the optional dependency for the configured backend, then
provide representative query probes as JSONL:

```json
{"id":"q1","query":"how do I reset my password?"}
{"id":"q2","query":"what is the vacation policy?"}
```

Run a terminal plan or save a machine-readable artifact:

```bash
embedflow plan --config embedflow.yaml --queries probes.jsonl
embedflow plan --config embedflow.yaml --queries probes.jsonl \
  --k-grid 20,50,100,200,500 --format json --output migration-plan.json
```

For a deterministic, network-free walkthrough, run `examples/planner/run_demo.sh`.
The demo uses the same planner API and tiny hash encoders; it is not a quality
benchmark.

## What the result means

The result is a versioned (`schema_version: 1`) object with source/target
contracts, preflight checks, evidence, candidate-depth diagnostics, cache and
rollout guidance, economics, and structured warnings. The Python API is:

```python
from embedflow.planner import MigrationPlanner

result = MigrationPlanner("embedflow.yaml").plan("probes.jsonl")
artifact = result.to_dict()
```

Recommendation states are deliberately small:

- `PROCEED`: probe evidence and preflight are strong enough for an operator-controlled shadow/canary sequence.
- `PROCEED_WITH_CAUTION`: probes are acceptable, but evidence coverage or contract/ANN provenance is limited.
- `EXPAND_PROBE`: T2-v1 asks for a deeper pool or the evidence is too small.
- `DEFER`: finite-tail behavior is `UNSAFE_OR_UNCERTAIN`; do not canary based on this plan.
- `BLOCKED`: structural/backend preflight failed; no candidate K is certified.

These are recommendations, not automatic deployment actions. The suggested
rollout is Validate → Shadow → small canary → expanded canary → target-primary
path, with the source remaining authoritative until the operator approves each
step. Stop conditions include source/target errors, p95 latency, queue depth,
cache hit rate, and any native/qrels evaluation regressions.

## Evidence and T2-v1

Registry matching preserves the existing hierarchy: `EXACT REGISTRY MATCH`,
`PRIOR EVIDENCE AVAILABLE` (same contracts, different/unknown corpus),
`RELATED EVIDENCE ONLY`, and `NO REGISTRY MATCH`. Prior or related evidence can
seed context but cannot override current probe results or certify a new corpus.

The planner calls the existing frozen T2-v1 implementation unchanged. T2-v1 is
a no-qrels finite-tail diagnostic of candidate behavior. `SAFE` does **not**
prove a candidate gap, nDCG, recall, or zero quality loss. Without actual qrels
or native-target evaluation, `candidate_gap` is `null`/`UNKNOWN`. ANN fidelity
is also `UNKNOWN` unless an exact/reference comparison is supplied by the
backend audit workflow.

No probes is a valid preflight/economics mode, but T2 is `NOT_RUN`, confidence
is `INSUFFICIENT`, and no K is presented as safe. Probe JSONL rows may use
`query` (or the existing `text` alias) and an optional `id`; empty, malformed,
or duplicate IDs are rejected. `--max-probes` and `--seed` provide deterministic
bounded sampling. Candidate documents are deduplicated by canonical ID and
encoded in bounded batches; `--max-target-encodes` can impose a hard cap.

## Cache, performance, and economics

Planning uses an ephemeral target-cache/state directory when it opens a runtime,
so the normal serving cache is not polluted. It recommends conservative sync
miss and background batch settings and defaults to traffic-driven progressive
warming when no access trace is supplied. An access trace can be supplied with
`--access-trace` as rows such as `{"document_id":"abc","count":192}` to
model hot-document coverage.

Performance values retain provenance (`measured`, `user_supplied`, `modeled`,
`registry`, or `unknown`). `--profile` performs only a small local encode/search
sample with one unmeasured warmup pass excluded; it is diagnostic and is not a
formal benchmark. Economics accepts `--target-docs-per-second` and
`--gpu-hourly-cost`; absent inputs remain `UNKNOWN`. Raw vector storage is
`documents × target dimension × dtype bytes` and excludes ANN overhead,
metadata, replicas, backups, and database overhead. Progressive percentages
are materialization scenarios, not claims that a percentage is sufficient.

## Configuration and privacy

An optional `planner:` section mirrors the CLI options (`max_probes`, `seed`,
`k_grid`, work caps, cache hints, throughput/cost inputs, and corpus identity).
CLI values take precedence. `EMBEDFLOW_PLANNER_*` environment overrides follow
the normal config mechanism. Probe text is never copied into plan artifacts or
telemetry by default; only counts, IDs used for diagnostics, and aggregates are
reported. Backend credentials are handled by the existing backend adapters and
are redacted from errors and structured output.

The planner does not provide qrel evaluation, traffic routing, rollback
automation, distributed scheduling, ANN tuning, integrated-vectorizer contract
verification, or production cost guarantees. Use `embedflow evaluate` with
qrels/native target rankings when empirical retrieval-quality claims are
required.
