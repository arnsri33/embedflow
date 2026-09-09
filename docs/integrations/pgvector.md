# PostgreSQL / pgvector

EmbedFlow can read candidates from an existing PostgreSQL table that uses the
[pgvector](https://github.com/pgvector/pgvector) extension. The adapter uses
the shared `VectorIndex` interface, so analysis, serving, cache warming, and
the API follow the same path as FAISS and Qdrant.

## Install

```bash
python -m pip install "embedflow[pgvector]"
```

The extra installs Psycopg 3 with its binary distribution. The base
`embedflow` install does not import or require PostgreSQL dependencies.

## Connection and table layout

Keep the DSN in an environment variable:

```bash
export EMBEDFLOW_PGVECTOR_DSN='postgresql://user:password@localhost:5432/app'
```

Example configuration:

```yaml
source:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu

target:
  model: Qwen/Qwen3-Embedding-0.6B
  device: cpu

index:
  backend: pgvector
  dsn_env: EMBEDFLOW_PGVECTOR_DSN
  schema: public
  table: documents
  id_column: id
  vector_column: embedding
  text_column: content
  metric: cosine
  # hnsw_ef_search: 100
  # ivfflat_probes: 20

documents:
  # Omit this file when text_column is present in the table. A separate
  # JSONL document store can be supplied when the table stores IDs only.
  path: ./documents.jsonl
  id_field: id
  text_field: text
```

When `documents.path` does not exist, EmbedFlow resolves candidate text from
`text_column` in the configured table. If a JSONL file exists, it remains the
document store and the pgvector table is used for vectors and IDs. The adapter
only reads the table during ordinary `analyze`, `serve`, `search`, `status`,
and `audit-index` operations; it does not create extensions, indexes, tables,
or modify rows.

The ID column may be an integer, bigint, UUID, or text type. IDs are normalized
to EmbedFlow's canonical string representation without numeric coercion.
The configured vector dimension is checked against `vector(n)` where available
and against a bounded non-null row for unbounded `vector` columns.

## Metrics and score direction

Supported metrics are `cosine`, `l2`/`euclidean`, and
`inner_product`/`dot`. pgvector's distance operators are converted to the
EmbedFlow convention where larger scores rank first:

| EmbedFlow metric | pgvector operator | Internal score |
| --- | --- | --- |
| cosine | `<=>` | `1 - distance` |
| l2 / euclidean | `<->` | `-distance` |
| inner_product / dot | `<#>` | `-distance` |

The last conversion accounts for pgvector's negative inner-product operator.

## ANN settings

EmbedFlow never creates or changes an ANN index. If configured, `hnsw_ef_search`
and `ivfflat_probes` are applied with transaction-local `set_config` calls for
the retrieval query. Omit them to use the database/index defaults. The
`metadata()` and `audit-index` output reports configured settings and detected
valid HNSW/IVFFlat indexes when PostgreSQL exposes them.

## Commands

```bash
embedflow analyze --config embedflow.yaml
embedflow serve --config embedflow.yaml
embedflow search --config embedflow.yaml "what causes auroras?"
embedflow status --config embedflow.yaml
embedflow audit-index --config embedflow.yaml
```

`audit-index` checks table and column presence, pgvector type and dimension,
NULL vectors, duplicate canonical IDs, a retrieval probe, and available ANN
metadata. A successful connection is not an ANN recall audit; use an exact
reference comparison when fidelity matters.

## Local example

The repository includes a small Docker example that creates a disposable
pgvector table and index:

```bash
cd examples/pgvector
docker compose up -d
export EMBEDFLOW_PGVECTOR_DSN='postgresql://embedflow:embedflow@localhost:5432/embedflow'
./run_demo.sh
```

The example is for local testing only. It contains no credentials for a real
service and does not represent a production database configuration.

## Troubleshooting

- `pgvector support requires psycopg`: install `embedflow[pgvector]` in the
  active environment.
- `DSN not found`: export the variable named by `index.dsn_env`.
- `table ... was not found`: check the schema/table spelling and database.
- `is not a pgvector column`: install/enable the extension in the database and
  point `vector_column` at a `vector` column.
- Dimension mismatch: set `source.dimension` to the encoder's dimension and
  verify the table's vector type.
- Authentication errors are redacted in EmbedFlow output; inspect PostgreSQL
  server logs without pasting passwords into issue reports.

## Limitations

The adapter currently exposes the common table layout and no arbitrary SQL
filters. Metadata filtering will follow the shared backend interface when that
interface gains a portable filter contract. Connection operations on one
adapter are serialized; applications needing a larger pool can create several
sessions behind their own pool. ANN health is reported as `UNKNOWN` unless a
reference comparison is run.
