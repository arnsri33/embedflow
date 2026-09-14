# Configuration

EmbedFlow accepts YAML through `--config`. Relative paths are resolved against
the directory containing that YAML file, so the same configuration can move
between machines without editing absolute paths.

```yaml
source:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cuda

target:
  model: Qwen/Qwen3-Embedding-0.6B
  device: cuda

index:
  backend: faiss                 # faiss, qdrant, pgvector, pinecone, milvus, or weaviate
  path: ./legacy.index
  ids: ./legacy.index.ids.json   # FAISS sidecar
  metric: cosine
  nprobe: 64
  # Qdrant fields: url, collection, vector_name, api_key_env
  # pgvector fields: dsn_env, schema, table, id_column, vector_column, text_column
  #                 hnsw_ef_search, ivfflat_probes
  # Pinecone fields: host (preferred) or index_name, api_key_env, namespace,
  #                  text_metadata_field
  # Milvus fields: uri, token_env, database, collection, id_field, vector_field,
  #                text_field, partition_names, search_params, auto_load
  # Weaviate fields: uri (or http_host/http_port), grpc_host/grpc_port,
  #                  secure, collection, vector_name, text_property,
  #                  tenant, api_key_env

documents:
  path: ./documents.jsonl
  id_field: id
  text_field: text

probe:
  queries: ./probe_queries.jsonl
  k_values: [10, 20, 50, 100, 200, 500]
  kmax: 500
  epsilon: 0.01

migration:
  candidate_depth: auto
  max_sync_misses: 4
  background_batch_size: 32
  max_retries: 3
  worker_count: 1

cache:
  path: ./embedflow_cache

telemetry:
  latency_log: ./logs/latency.jsonl

economics:
  gpu_price_per_hour: null
  target_docs_per_second: null

# Optional advisory planner settings. CLI flags override these values.
planner:
  max_probes: 250
  seed: 42
  k_grid: [20, 50, 100, 200, 500]
  max_candidates: null
  max_target_encodes: null
  max_sync_misses: null
  background_batch_size: null
  gpu_hourly_cost: null
  target_docs_per_second: null
  queries_per_second: null
  daily_queries: null
  cache_hit_rate: null
  latency_budget_ms: null
  access_trace: null
  corpus_name: null
  corpus_fingerprint: null

# Source-authoritative observation mode.  ``runtime.mode: migration`` (the
# default) leaves Shadow Mode inactive.  ``mode: shadow`` is an explicit opt-in.
runtime:
  mode: migration             # migration, normal, source, or shadow

shadow:
  enabled: true
  sample_rate: 0.10
  sample_seed: 42
  candidate_k: 100
  materialize: true
  max_inflight: 32
  queue_capacity: 1000
  timeout_ms: 10000
  shutdown_grace_ms: 1000
  telemetry:
    enabled: true
    path: ./.embedflow/shadow.sqlite3
    retain_query_records: false
    retain_query_text: false
    max_records: 10000
    retention_days: null
    report_k: 10
    min_target_coverage_for_ranking: 1.0

state_path: ./embedflow_state.json
```

Shadow Mode always returns the source-authoritative result before target work
finishes. See [`shadow-mode.md`](shadow-mode.md) for queue, timeout, privacy,
materialization, and report semantics.

## Model contracts

Model configuration can include `revision`, `dimension`, `max_length`,
`pooling`, `padding_side`, `truncation_side`, `query_instruction`,
`document_instruction`, `normalization`, and `dtype`. These fields describe
the embedding contract and are included in the model fingerprint. Local paths
and devices are deployment details and do not change the fingerprint.

The target cache uses `(document_id, target_model_fingerprint)`. A change to a
revision, prompt, pooling method, length, normalization, or dimension creates a
different semantic cache namespace.

## Environment overrides

The following variables override matching YAML fields when set:

```text
EMBEDFLOW_SOURCE_MODEL
EMBEDFLOW_TARGET_MODEL
EMBEDFLOW_SOURCE_DEVICE
EMBEDFLOW_TARGET_DEVICE
EMBEDFLOW_INDEX_PATH
EMBEDFLOW_INDEX_URL
EMBEDFLOW_INDEX_COLLECTION
EMBEDFLOW_INDEX_VECTOR_NAME
EMBEDFLOW_QDRANT_API_KEY_ENV
EMBEDFLOW_PGVECTOR_DSN_ENV
EMBEDFLOW_PGVECTOR_SCHEMA
EMBEDFLOW_PGVECTOR_TABLE
EMBEDFLOW_PGVECTOR_ID_COLUMN
EMBEDFLOW_PGVECTOR_VECTOR_COLUMN
EMBEDFLOW_PGVECTOR_TEXT_COLUMN
EMBEDFLOW_PGVECTOR_HNSW_EF_SEARCH
EMBEDFLOW_PGVECTOR_IVFFLAT_PROBES
EMBEDFLOW_PINECONE_HOST
EMBEDFLOW_PINECONE_INDEX_NAME
EMBEDFLOW_PINECONE_NAMESPACE
EMBEDFLOW_PINECONE_TEXT_METADATA_FIELD
EMBEDFLOW_PINECONE_API_KEY_ENV
EMBEDFLOW_MILVUS_URI
EMBEDFLOW_MILVUS_TOKEN_ENV
EMBEDFLOW_MILVUS_DATABASE
EMBEDFLOW_MILVUS_COLLECTION
EMBEDFLOW_MILVUS_ID_FIELD
EMBEDFLOW_MILVUS_VECTOR_FIELD
EMBEDFLOW_MILVUS_TEXT_FIELD
EMBEDFLOW_MILVUS_PARTITIONS
EMBEDFLOW_MILVUS_AUTO_LOAD
EMBEDFLOW_WEAVIATE_URI
EMBEDFLOW_WEAVIATE_HTTP_HOST
EMBEDFLOW_WEAVIATE_HTTP_PORT
EMBEDFLOW_WEAVIATE_GRPC_HOST
EMBEDFLOW_WEAVIATE_GRPC_PORT
EMBEDFLOW_WEAVIATE_SECURE
EMBEDFLOW_WEAVIATE_GRPC_SECURE
EMBEDFLOW_WEAVIATE_API_KEY_ENV
EMBEDFLOW_WEAVIATE_COLLECTION
EMBEDFLOW_WEAVIATE_VECTOR_NAME
EMBEDFLOW_WEAVIATE_TEXT_PROPERTY
EMBEDFLOW_WEAVIATE_TENANT
EMBEDFLOW_INDEX_NPROBE
EMBEDFLOW_DOCUMENTS_PATH
EMBEDFLOW_CACHE_PATH
EMBEDFLOW_STATE_PATH
EMBEDFLOW_LATENCY_LOG
EMBEDFLOW_CANDIDATE_DEPTH
EMBEDFLOW_MAX_SYNC_MISSES
EMBEDFLOW_BACKGROUND_BATCH_SIZE
EMBEDFLOW_PROBE_KMAX
EMBEDFLOW_PLANNER_MAX_PROBES
EMBEDFLOW_PLANNER_SEED
EMBEDFLOW_PLANNER_MAX_CANDIDATES
EMBEDFLOW_PLANNER_MAX_TARGET_ENCODINGS
EMBEDFLOW_PLANNER_MAX_SYNC_MISSES
EMBEDFLOW_PLANNER_BACKGROUND_BATCH_SIZE
EMBEDFLOW_PLANNER_GPU_HOURLY_COST
EMBEDFLOW_PLANNER_TARGET_DOCS_PER_SECOND
EMBEDFLOW_PLANNER_QPS
EMBEDFLOW_PLANNER_DAILY_QUERIES
EMBEDFLOW_PLANNER_CACHE_HIT_RATE
EMBEDFLOW_PLANNER_LATENCY_BUDGET_MS
EMBEDFLOW_PLANNER_ACCESS_TRACE
EMBEDFLOW_PLANNER_CORPUS_NAME
EMBEDFLOW_PLANNER_CORPUS_FINGERPRINT
EMBEDFLOW_RUNTIME_MODE
EMBEDFLOW_SHADOW_ENABLED
EMBEDFLOW_SHADOW_SAMPLE_RATE
EMBEDFLOW_SHADOW_SAMPLE_SEED
EMBEDFLOW_SHADOW_CANDIDATE_K
EMBEDFLOW_SHADOW_MATERIALIZE
EMBEDFLOW_SHADOW_MAX_INFLIGHT
EMBEDFLOW_SHADOW_QUEUE_CAPACITY
EMBEDFLOW_SHADOW_TIMEOUT_MS
EMBEDFLOW_SHADOW_SHUTDOWN_GRACE_MS
EMBEDFLOW_SHADOW_TELEMETRY_ENABLED
EMBEDFLOW_SHADOW_TELEMETRY_PATH
EMBEDFLOW_SHADOW_RETAIN_QUERY_RECORDS
EMBEDFLOW_SHADOW_RETAIN_QUERY_TEXT
EMBEDFLOW_SHADOW_MAX_RECORDS
EMBEDFLOW_SHADOW_REPORT_K
EMBEDFLOW_SHADOW_RETENTION_DAYS
EMBEDFLOW_SHADOW_MIN_TARGET_COVERAGE
```

## Input files

Document JSONL rows must contain the configured ID and text fields. Probe
queries are JSONL rows with a `query` field (or a configured text field where
the command supports it). FAISS indexes use a JSON ID sidecar unless IDs are
embedded in the selected index format. Qdrant reads document IDs and payloads
from the configured collection.
