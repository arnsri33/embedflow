# FAISS migration example

This example uses the deterministic hash models so it is fully offline.

```bash
cd examples/faiss
PYTHONPATH=../.. embedflow init --config embedflow.yaml --build-index \
  --queries queries.jsonl --demo
PYTHONPATH=../.. embedflow analyze --config embedflow.yaml --queries queries.jsonl --demo \
  --output-dir analysis
PYTHONPATH=../.. embedflow serve --config embedflow.yaml --demo
```

Open <http://127.0.0.1:8000/> and search the same question twice. The source
FAISS index remains online while target vectors are progressively cached.
