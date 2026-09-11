# Pinecone

EmbedFlow can use an existing Pinecone dense index as its source candidate
layer. It queries that index, reranks the returned documents with the target
embedding model, and materializes target vectors into EmbedFlow's local cache.
The source index is read-only during normal operation.

## Install

```bash
python -m pip install "embedflow[pinecone]"
```

The extra installs the official `pinecone` Python SDK. The base package does
not import or require it.

## Configure an existing index

Set the API key in the environment, never in YAML:

```bash
export PINECONE_API_KEY='...'
```

Use the data-plane host copied from the Pinecone console:

```yaml
source:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  device: cuda

target:
  model: Qwen/Qwen3-Embedding-0.6B
  device: cuda

index:
  backend: pinecone
  api_key_env: PINECONE_API_KEY
  host: my-index-xxxxx.svc.aped-xxxx.pinecone.io
  namespace: production
  metric: cosine
  text_metadata_field: text

documents:
  # Omit this file when text_metadata_field is present in Pinecone metadata.
  # If present, it is used as the external text resolver instead.
  path: ./documents.jsonl
  id_field: id
  text_field: text
```

`host` takes precedence when both `host` and `index_name` are supplied. If a
host is unavailable, `index_name` can resolve through the control plane; host
targeting is recommended for production data-plane traffic. The configured
namespace is passed on every query and fetch. An empty namespace selects
Pinecone's default namespace.

The source index must be a dense vector index whose dimension and metric match
the source model contract. Pinecone IDs remain strings, including numeric-
looking IDs such as `"42"`, UUID-looking values, Unicode, and punctuation.

## Document text

EmbedFlow needs candidate text for target-model encoding. There are two modes:

* Set `text_metadata_field` to read text from Pinecone metadata. Candidate
  queries request metadata but never request vector values. Missing or
  non-string text is reported with the candidate ID and field name.
* Provide the normal JSONL `documents` store when Pinecone contains IDs and
  vectors only. The external store remains the source of text and avoids
  duplicating a corpus in Pinecone.

## Commands

```bash
embedflow doctor --config embedflow.yaml
embedflow audit-index --config embedflow.yaml
embedflow analyze --config embedflow.yaml
embedflow serve --config embedflow.yaml
embedflow search --config embedflow.yaml "what causes auroras?"
embedflow status --config embedflow.yaml
```

`audit-index` separates backend health from retrieval fidelity. It checks SDK
and credentials, reachability, namespace stats, dimension, metric, dense
vector type, a small candidate query, and text resolution where possible. A
successful API call is not an ANN recall or T2-v1 compatibility guarantee.

## Read-only behavior

`doctor`, `audit-index`, `analyze`, `serve`, `search`, `status`, and `prewarm`
only query the configured index. EmbedFlow does not create or delete indexes,
upsert or delete vectors, alter metadata, or change namespaces. Test fixtures
may create temporary indexes explicitly, but production initialization never
does.

## Metrics and limits

`cosine`, `dot`/`inner_product` (Pinecone `dotproduct`), and
`l2`/`euclidean` are accepted. Cosine and dot-product scores already rank
higher-is-better. Pinecone's Euclidean score is squared distance, so EmbedFlow
negates it to preserve the shared higher-is-better convention. Query `top_k`
is validated in the range 1–10,000, matching the Pinecone API limit.

## Integrated embedding indexes

Pinecone indexes created with hosted/integrated embedding manage the source
embedding contract outside EmbedFlow. Unless the source model, dimension, and
metric can be verified, EmbedFlow reports that contract as unknown; it does not
assume Pinecone's hosted model is the configured source encoder. The first
adapter release is intended for standard dense-vector indexes.

## Eventual consistency and tests

Pinecone is eventually consistent after writes. Any test setup that creates a
temporary fixture must poll `describe_index_stats` with a bounded timeout before
querying it. EmbedFlow's production adapter performs no writes and therefore
does not need a write-read delay.

## Troubleshooting

* `Pinecone support requires ...`: install `embedflow[pinecone]` in the active
  environment.
* `Environment variable PINECONE_API_KEY is not set.`: export the variable
  named by `api_key_env`.
* `Unable to reach configured Pinecone index`: check the host, project, and
  network policy. Credentials are not printed in this error.
* Dimension or metric mismatch: compare the source model contract with the
  existing index configuration; EmbedFlow will not rewrite the index.
* Missing metadata text: set the correct `text_metadata_field` or provide an
  external JSONL document store.

## Local smoke fixture

The repository includes a non-destructive smoke helper for an existing index:

```bash
export PINECONE_API_KEY='...'
python scripts/pinecone_smoke.py \
  --host "$PINECONE_INDEX_HOST" \
  --namespace production
```

It only describes stats and, when given a query vector, performs a query. It
never upserts or deletes records. A remote create/query/delete integration is
opt-in and requires both `EMBEDFLOW_PINECONE_INTEGRATION_TEST=1` and
`EMBEDFLOW_PINECONE_CREATE_TEST_INDEX=1`; it uses a unique disposable index and
deletes it in teardown.

## Limitations

The adapter uses one synchronous SDK client. The SDK call itself is safe for
the modest concurrent serving loads tested by EmbedFlow, but no connection pool
is introduced. Metadata filters are not exposed because the shared
`VectorIndex` contract has no portable filter field. Pinecone index provisioning
and namespace administration remain outside EmbedFlow.
