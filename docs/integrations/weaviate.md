# Weaviate

EmbedFlow's Weaviate backend (tested with `weaviate-client` 4.23.1 and
Weaviate 1.30.6) reads an existing, externally vectorized dense collection.
It retrieves candidates with the v4 `near_vector` API, reranks them with the
target model, and progressively materializes target vectors in EmbedFlow's
local cache. It does not create, update, delete, batch-import, or re-vectorize
objects in the source collection.

## Install and connect

```bash
python -m pip install "embedflow[weaviate]"
```

The local v4 client needs both HTTP and gRPC. A standard Docker deployment
exposes HTTP 8080 and gRPC 50051:

```bash
cd examples/weaviate
docker compose up -d
```

Configuration uses a URI or explicit host/port fields:

```yaml
index:
  backend: weaviate
  uri: http://127.0.0.1:8080
  http_port: 8080
  grpc_host: 127.0.0.1
  grpc_port: 50051
  collection: Documents
  vector_name: default
  text_property: content
  metric: cosine
  api_key_env: WEAVIATE_API_KEY
```

For a custom or cloud-compatible endpoint set `http_host`, `http_port`,
`grpc_host`, `grpc_port`, `secure`, and `grpc_secure` as appropriate. Keep the
credential out of YAML:

```bash
export WEAVIATE_API_KEY='your-key'
embedflow doctor --config ./embedflow.yaml
```

Cloud connections use the same v4 URI/token conventions. This release has
validated local standalone Weaviate, not an independently operated Weaviate
Cloud cluster.

## Collection contract

The selected collection must contain a dense, externally supplied vector. Set
`vector_name` for named-vector collections; if more than one vector exists and
the name is omitted, EmbedFlow fails rather than guessing. `cosine`, `dot` (or
`inner_product`), and `l2`/`euclidean` are supported. Weaviate's distance is
converted to EmbedFlow's larger-is-better score convention (`-distance`), so
rank order remains the server's rank order.

Weaviate objects use UUIDs. EmbedFlow preserves their canonical string form;
it does not generate IDs during reads. The source model dimension must match
the existing vector index. Weaviate versions that do not expose dimensionality
in collection configuration report the value as unverified and the server
still validates each query vector.

For text, choose one of these modes:

* `text_property: content` requests only that property with each candidate and
  validates that it is a string.
* Provide the normal EmbedFlow JSONL `documents` store. Candidate queries then
  request no Weaviate properties and the shared document store resolves text.

The adapter requests no stored vector payload. `tenant` can bind a configured
multi-tenant collection to one tenant; an omitted tenant never selects one
implicitly.

## Audit and operations

```bash
embedflow audit-index --config ./embedflow.yaml
embedflow analyze --config ./embedflow.yaml
embedflow serve --config ./embedflow.yaml --device cuda
```

`doctor` and `audit-index` report readiness, collection/vector configuration,
metric, object count, tenant, and text resolution. Reachability is reported
separately from candidate compatibility and ANN fidelity; a healthy server is
not evidence that a model migration is scientifically safe. `client.close()`
is called when a one-shot command or serving session shuts down.

## Read-only and troubleshooting

The production adapter uses only collection lookup/config, `near_vector`,
batched property lookup, iteration, and aggregate counts. Fixture creation and
batch writes live only in `scripts/weaviate_fixture.py` and tests. Do not point
fixture setup at a user collection.

* `Weaviate support requires ...`: install the optional extra in the active
  environment.
* HTTP works but startup says gRPC is unavailable: expose 50051 and set
  `grpc_host`/`grpc_port` correctly; v4 search uses gRPC.
* `vector_name ... was not found` or multiple-vector errors: inspect the
  collection schema and configure the exact name.
* Dimension or metric mismatch: use the source model contract and the
  collection's configured distance metric.
* Missing/invalid text: fix `text_property` or use an external document store.

The adapter does not expose arbitrary Weaviate filters or index-tuning writes.
HNSW configuration is introspected read-only. Integrated Weaviate vectorizers
are not treated as the EmbedFlow source model unless their contract is
independently verified; this release focuses on externally vectorized dense
collections.
