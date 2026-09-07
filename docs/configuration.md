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
  backend: faiss                 # faiss or qdrant
  path: ./legacy.index
  ids: ./legacy.index.ids.json   # FAISS sidecar
  metric: cosine
  nprobe: 64
  # Qdrant fields: url, collection, vector_name, api_key_env

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

state_path: ./embedflow_state.json
```

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
EMBEDFLOW_INDEX_NPROBE
EMBEDFLOW_DOCUMENTS_PATH
EMBEDFLOW_CACHE_PATH
EMBEDFLOW_STATE_PATH
EMBEDFLOW_LATENCY_LOG
EMBEDFLOW_CANDIDATE_DEPTH
EMBEDFLOW_MAX_SYNC_MISSES
EMBEDFLOW_BACKGROUND_BATCH_SIZE
EMBEDFLOW_PROBE_KMAX
```

## Input files

Document JSONL rows must contain the configured ID and text fields. Probe
queries are JSONL rows with a `query` field (or a configured text field where
the command supports it). FAISS indexes use a JSON ID sidecar unless IDs are
embedded in the selected index format. Qdrant reads document IDs and payloads
from the configured collection.
