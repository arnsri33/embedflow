# Shadow Mode

Shadow Mode runs the configured target migration path beside real traffic while
the existing source result remains authoritative. It is an observation and
cache-warming tool, not a traffic router:

```yaml
runtime:
  mode: shadow

shadow:
  enabled: true
  sample_rate: 0.10
  sample_seed: 42
  candidate_k: 100
  materialize: true
  max_inflight: 32
  queue_capacity: 1000
  timeout_ms: 10000
  telemetry:
    enabled: true
    path: ./.embedflow/shadow.sqlite3
    retain_query_records: false
    retain_query_text: false
```

Install the optional backend/model extras required by the selected
configuration, then start the ordinary service:

```bash
pip install "embedflow[dashboard]"
embedflow serve --config embedflow.yaml
# or override the file for one process:
embedflow serve --config embedflow.yaml --mode shadow
```

The request path performs source encoding, source ANN retrieval, and document
resolution exactly as source-only serving. It returns that result without
waiting for target encoding, cache misses, reranking, telemetry, or the
materialization worker. Shadow jobs use a bounded queue and worker pool. A full
queue drops only shadow work; a target/model/cache/telemetry failure or timeout
is recorded and cannot change the primary response.

If the target model cannot be loaded during startup, source/shadow serving
still opens with target work marked unavailable; sampled requests record the
isolated target-encoding failure. Normal migration mode remains fail-fast for
the same startup error.

`candidate_k` is explicit. The planner may suggest a value, but Shadow Mode
does not silently select one. `materialize: false` reads already cached target
vectors without changing cache or queue state. With `materialize: true`, cold
candidate IDs are deduplicated in the existing persistent materialization queue
and warmed asynchronously; synchronous target misses remain zero in Shadow
Mode. A comparison with incomplete target vectors is reported as `partial`,
with target coverage shown separately.

## Reports

Reports use the privacy-preserving SQLite telemetry store (by default beside
the configured cache):

```bash
embedflow shadow report --config embedflow.yaml --since 24h
embedflow shadow report --config embedflow.yaml --since 24h --format json --output shadow-report.json
```

Use `/shadow/report` or the `shadow` section of `/status` and `/metrics` in the
FastAPI service. JSON/YAML stdout contains only the requested artifact; progress
and errors belong on stderr. `--since` accepts seconds or `s`, `m`, `h`, `d`,
and `w` suffixes.

Reports contain sampled/completed/partial/failed/timed-out/dropped counts,
cache and materialization accounting, source and shadow latency distributions,
target coverage, top-1 agreement, and top-k overlap. Shadow latency is not
user-facing latency because it is off the primary critical path. Ranking
disagreement and overlap are diagnostics, not recall, nDCG, or quality loss;
without qrels/native-target evaluation no quality guarantee is made.

T2-v1 remains the frozen window-level diagnostic. EmbedFlow does not label an
individual production query `T2 SAFE`. A report recommendation is operational
guidance only:

- `CONTINUE_SHADOW` means evidence or coverage is still limited.
- `EXPAND_K` means the accumulated T2 window requests a larger candidate pool.
- `INVESTIGATE` means failures, timeouts, or uncertain T2 behavior need review.
- `READY_FOR_CANARY_EVALUATION` means an operator may consider a separately
  designed canary; it never routes traffic automatically.

Raw query text, candidate text, source vectors, target vectors, and credentials
are not persisted in telemetry. Query IDs may be retained only when explicitly
enabled, and retention is bounded by `max_records`. The telemetry database is
segmented by source/target contract and candidate-K fingerprint so unrelated
migrations are not mixed.

On shutdown EmbedFlow stops accepting new shadow work, drops queued jobs when
necessary, and waits only the configured bounded grace period. A corrupted or
unwritable telemetry file degrades observability; it does not take source
serving down. Source indexes are read-only during Shadow Mode. The planner and
Shadow Mode are separate: generate a plan first, then copy its recommended K
explicitly into a reviewed Shadow configuration.

Shadow Mode is advisory and experimental operational instrumentation. It does
not provide autonomous rollout, rollback, qrel evaluation, or ANN-fidelity
proof.
