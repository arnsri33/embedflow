# Qdrant migration example

This is a user-owned local Qdrant example. It uses the same document store,
model contract, target cache, and migration engine as the FAISS example; only
the source index adapter changes.

Install the optional dependency:

```bash
python -m pip install -e ".[qdrant,dashboard]"
```

Build a local Qdrant collection using the deterministic model:

```bash
PYTHONPATH=../.. python build_index.py
```

Then run the probe and serve:

```bash
PYTHONPATH=../.. embedflow analyze --config embedflow.yaml --queries queries.jsonl --demo --output-dir analysis
PYTHONPATH=../.. embedflow serve --config embedflow.yaml --demo
```

For a server-backed collection, change `index.url` to your Qdrant URL, set
`QDRANT_API_KEY` (or the environment variable named by `index.api_key_env`),
and keep credentials out of YAML and Git. For named vectors, set
`index.vector_name` to the collection's vector key. Never commit credentials.
