# Milvus

EmbedFlow can use an existing Milvus collection as a read-only source
candidate index. A source query is executed by Milvus, the returned documents
are reranked by the target model, and target vectors are stored in EmbedFlow's
local cache. The adapter does not copy or rebuild the source collection.

## Install and connect

Install the optional official SDK (the adapter is tested with pymilvus 2.5.5
through 3.0.1):

```bash
python -m pip install "embedflow[milvus]"
```

For a local standalone server, the URI is normally
`http://127.0.0.1:19530`. Cloud/Milvus-compatible endpoints can use the same
`MilvusClient` URI and token semantics. Keep credentials out of YAML:

```bash
export EMBEDFLOW_MILVUS_TOKEN='user:password-or-cloud-token'
```

An unauthenticated local server may omit the variable. EmbedFlow never puts a
token in status, telemetry, generated configuration, or error messages.

Example configuration for a collection that stores its own text:

```yaml
source:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  device: cuda
target:
  model: Qwen/Qwen3-Embedding-0.6B
  device: cuda
index:
  backend: milvus
  uri: http://127.0.0.1:19530
  token_env: EMBEDFLOW_MILVUS_TOKEN
  database: default
  collection: documents
  id_field: id
  vector_field: embedding
  text_field: content
  metric: cosine
  partition_names: []
  auto_load: false
documents:
  path: ./documents.jsonl
  id_field: id
  text_field: text
```

`uri`, `database`, `collection`, `id_field`, `vector_field`, and
`text_field` are quoted as ordinary SDK arguments; they are not interpolated
into SQL. `vector_field` should identify a dense `FLOAT_VECTOR` field. The
first release does not claim support for `SPARSE_FLOAT_VECTOR`, `BINARY_VECTOR`,
`FLOAT16_VECTOR`, or `BFLOAT16_VECTOR`. If a collection has multiple dense
vector fields, configure `vector_field` explicitly.

## Text and IDs

There are two supported text layouts:

* Set `text_field` when the Milvus collection has `id`, `embedding`, and a
  string text field. Search requests ask Milvus for only that field.
* Keep `text_field` unset and provide the normal EmbedFlow JSONL document
  store. Candidate retrieval then asks Milvus for IDs only and resolves text
  through the shared document-store contract.

Milvus `INT64` and `VARCHAR` primary keys are supported. IDs are exposed to
EmbedFlow as strings, so an INT64 value `42` is represented as `"42"` at the
application boundary while a VARCHAR value `"42"` remains a string. No numeric
coercion is performed for VARCHAR IDs.

## Metrics and indexes

Canonical metric names are `cosine`, `inner_product`/`dot`, and
`l2`/`euclidean`; they map to Milvus `COSINE`, `IP`, and `L2`. Milvus returns
COSINE/IP similarities (larger is better) and L2 distances (smaller is
better). EmbedFlow negates L2 distances only, preserving the shared
higher-is-better `SearchHit.score` convention.

The adapter does not create indexes. Existing HNSW and IVF_FLAT indexes are
queried when present. Optional native search parameters can be supplied as:

```yaml
index:
  search_params:
    ef: 64       # HNSW; raised to at least top-k when required by Milvus
    # nprobe: 16 # IVF_FLAT
```

Native Milvus form (`metric_type` plus `params`) is also accepted. Unknown
keys are rejected. `ef` is never sent below the requested top-k because HNSW
servers reject that request.

## Loading, databases, and partitions

EmbedFlow checks collection load state. `auto_load: false` (the conservative
default) returns an actionable error if the collection is not loaded. Set
`auto_load: true` only when explicitly permitting EmbedFlow to load this
collection; EmbedFlow never releases or unloads a collection.

`database` selects the Milvus database. `partition_names` restricts every
search and text lookup to the listed partitions. A missing partition fails at
connection time when the server exposes partition metadata. No arbitrary
Milvus filter expression is exposed because the shared `VectorIndex` contract
has no portable filter field.

## Commands and audit

```bash
embedflow doctor --config embedflow.yaml
embedflow audit-index --config embedflow.yaml
embedflow analyze --config embedflow.yaml --queries ./probe_queries.jsonl
embedflow serve --config embedflow.yaml --device cuda
embedflow search --config embedflow.yaml "your query"
embedflow status --config embedflow.yaml
```

`audit-index` reports URI (redacted), database, collection, fields, dense
vector type, dimension, metric, index type, load state, partitions, row
count, and a small retrieval/text probe. It intentionally keeps ANN fidelity,
T2-v1 compatibility, and candidate quality as separate `UNKNOWN`/empirical
questions; a reachable collection is not proof of migration compatibility.

Normal `doctor`, `audit-index`, `analyze`, `serve`, `search`, `status`, and
`prewarm` operations are read-only against the configured collection. Fixture
creation and inserts are confined to `scripts/milvus_fixture.py`, tests, and
the example setup.

## Local example

The repository includes a small standalone deployment and deterministic
fixture:

```bash
cd examples/milvus
docker compose up -d
python -m pip install "embedflow[milvus,dashboard]"
./run_demo.sh
```

The fixture utility is deliberately explicit and refuses to overwrite an
existing collection. It can create HNSW, IVF_FLAT, or FLAT test indexes. No
model weights or credentials are committed.

## Troubleshooting and limitations

* `Milvus support requires ...`: install the `[milvus]` extra in the active
  environment.
* `collection ... was not found`: check `uri`, `database`, and collection name.
* `collection ... is not loaded`: load it administratively or opt in to
  `auto_load: true`.
* Dimension/metric errors mean the configured source model contract does not
  match the existing vector field/index; EmbedFlow does not rewrite it.
* The adapter uses one synchronous `MilvusClient`; concurrent calls are
  serialized by a lock. A pool is intentionally not introduced for this
  release.
* Milvus-compatible URI/token semantics should work with Zilliz Cloud where
  supported by the SDK, but Zilliz Cloud has not been independently validated
  for this release.
