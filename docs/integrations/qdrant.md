# Qdrant

Qdrant is an optional source-index backend. It works with a local server,
local Qdrant storage where supported by the installed client, or a user-owned
remote collection.

## Configuration

```yaml
index:
  backend: qdrant
  url: http://localhost:6333
  collection: my_legacy_collection
  metric: cosine
  # vector_name: text       # named-vector collections
  # api_key_env: QDRANT_API_KEY
```

Install the adapter with:

```bash
python -m pip install -e ".[qdrant,dashboard]"
```

For a remote service, export the key named by `api_key_env` before starting the
server. Credentials stay outside YAML and Git. Set `vector_name` for a named
vector collection; leave it unset for a single unnamed vector.

## Bundled example

```bash
cd examples/qdrant
PYTHONPATH=../.. python build_index.py
PYTHONPATH=../.. embedflow analyze --config embedflow.yaml \
  --queries queries.jsonl --demo --output-dir analysis
PYTHONPATH=../.. embedflow serve --config embedflow.yaml --demo
```

The example uses a deterministic model and a local user-owned collection. It
does not require a cloud account. The adapter preserves application document
IDs through its deterministic collection mapping.

## Local model smoke test

Maintainers with staged model snapshots can run:

```bash
python scripts/real_qdrant_smoke.py --model-root ./models
```

This exercises the MiniLM → Qwen3-0.6B lifecycle against a local collection;
it is a functional smoke test, not a latency benchmark.
