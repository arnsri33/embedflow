# FAISS

FAISS is the default source-index backend. EmbedFlow supports cosine and inner
product-style configurations through the stable `VectorIndex` interface.

## Inputs

Provide an index and an ID sidecar:

```yaml
index:
  backend: faiss
  path: ./legacy.index
  ids: ./legacy.index.ids.json
  metric: cosine
  nprobe: 64
```

The sidecar maps vector rows to document IDs. Every ID returned by FAISS must
exist in the configured document store.

## Build the example

The bundled example uses deterministic hash embeddings and is fully offline:

```bash
cd examples/faiss
PYTHONPATH=../.. embedflow init --config embedflow.yaml --build-index \
  --queries queries.jsonl --demo
PYTHONPATH=../.. embedflow analyze --config embedflow.yaml \
  --queries queries.jsonl --demo --output-dir analysis
PYTHONPATH=../.. embedflow serve --config embedflow.yaml --demo
```

Use `embedflow audit-index` with an exact reference index when ANN fidelity is
part of the evaluation.
