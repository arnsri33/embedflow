# EmbedFlow + Milvus

This example connects EmbedFlow to an existing local Milvus collection. The
adapter is read-only during `doctor`, `audit-index`, `analyze`, `search`, and
serving; the fixture setup is the only code that creates a collection or
inserts vectors.

## Start Milvus

```bash
docker compose up -d
python -m pip install "embedflow[milvus,dashboard]"
```

Wait until the service is healthy, then create the deterministic fixture:

```bash
./run_demo.sh
```

The script creates a small disposable collection (`embedflow_demo`) with
`id`, `content`, and a 64-dimensional `embedding` field, builds an HNSW index,
loads it, and runs a read-only EmbedFlow smoke test. It never changes an
existing collection with another name.

For a real deployment, copy `embedflow.yaml.example`, point `uri` and
`collection` at the collection you already own, and set `vector_field`/`id_field`
to the actual schema. For authenticated deployments:

```bash
export EMBEDFLOW_MILVUS_TOKEN='user:password-or-cloud-token'
```

No token is written to YAML or status output. See
[`docs/integrations/milvus.md`](../../docs/integrations/milvus.md) for schema,
metrics, partitions, loading, and external document-store guidance.

Stop the example with `docker compose down -v` when finished. The `-v` flag
removes only the example's named volumes.
