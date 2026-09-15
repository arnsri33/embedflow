# Traffic-aware prewarming

Traffic-aware prewarming uses privacy-safe Shadow Mode aggregates to select the
uncached target documents that occurred most often in the source candidate
pool. It is a bounded operational warming tool in the existing migration
workflow:

```text
PLAN -> SHADOW -> PREWARM -> CANARY -> MIGRATE
```

It never writes the source index. `prewarm plan` only reads Shadow telemetry
and the target cache. `prewarm run` delegates document lookup, target encoding,
and cache writes to the existing materialization worker.

## Plan and run

```bash
embedflow prewarm plan \
  --config embedflow.yaml --since 24h --max-docs 50000 \
  --output prewarm-plan.json
embedflow prewarm run --config embedflow.yaml --plan prewarm-plan.json
embedflow prewarm status --config embedflow.yaml
```

Plans are schema-versioned (`schema_version: 1`), deterministically ranked by
`candidate_occurrences DESC, document_id ASC`, and fingerprinted with the
source/target contracts, source-index identity, candidate K, and Shadow
configuration. Execution refuses a stale or tampered plan. A plan is always
bounded by `max_docs`; there is no implicit full-corpus operation.

Optional configuration defaults are available under `prewarm`:

```yaml
prewarm:
  max_docs: 1000
  strategy: traffic_hotset
  batch_size: 32
  max_retries: 3
```

`--target-observed-coverage`, `--max-storage-gb`, and `--max-runtime` (when a
real measured/user-supplied `--docs-per-second` is supplied) further reduce
the hard selection cap. `--gpu-hourly-cost` (or the existing planner/economics
configuration value) adds a user-supplied modeled cost. Runtime, cost, and
storage values are explicitly marked as modeled/unknown; no cloud price or
throughput is guessed.

## Coverage and privacy

The plan reports **observed candidate-occurrence coverage**:

```text
occurrences with a valid target vector / all source candidate occurrences
```

This is not retrieval coverage, recall, nDCG, or a guarantee of migration
quality. It only describes the selected Shadow window and candidate K. Ties are
stable, and current target-cache entries count as warm only when their target
model and (when document text is available) content fingerprints match. Legacy
cache rows created before content bindings were introduced remain usable but
are treated as content-unknown.

Shadow aggregation stores document IDs and counters, not raw query text,
document text, vectors, or credentials. Treat the generated plan/ID list as
application data because selected document IDs are needed to execute it.

## Resume and idempotence

The runner stores a plan-specific queue/state beside the target cache and uses
the target cache as the execution source of truth. Re-running a completed plan
skips already-warm vectors. Interrupts leave completed vectors valid and a
subsequent run can continue the remaining IDs. Documents that disappear or
change are resolved again at run time; stale content is not silently reused.

Two concurrent runners use the existing durable queue/cache deduplication. The
queue is at-least-once under process crashes, so operators should inspect
`prewarm status` and the run report for failures.

The source FAISS/Qdrant/pgvector/Pinecone/Milvus/Weaviate index is never
created, altered, rebuilt, upserted, or deleted by this feature. The API for
prewarming is currently Python/core plus CLI; dashboard execution controls are
intentionally not added to avoid turning an advisory plan into autonomous
traffic routing.
