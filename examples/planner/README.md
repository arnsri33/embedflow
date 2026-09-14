# Offline planner demo

This tiny demo requires no network, model download, or optional backend. It
builds the existing deterministic FAISS/NumPy fixture and runs the same
`MigrationPlanner` used by the production CLI.

```bash
./run_demo.sh
```

The generated plan is advisory. Its `SAFE` result, if shown, is a finite-tail
T2-v1 signal over synthetic hash embeddings, not a retrieval-quality claim.
