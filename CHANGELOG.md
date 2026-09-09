# Changelog

## v0.2.0 — pgvector backend

- Added a read-only pgvector backend for existing PostgreSQL vector tables.
- Added cosine, Euclidean/L2, and inner-product retrieval with safe SQL
  identifier composition.
- Added pgvector table auditing, optional transaction-local HNSW/IVFFlat
  settings, Docker example, and numerical/SQL-safety tests.

## v0.1.0 — initial public release

- Candidate-compatibility analysis and Mode A evaluation workflows.
- Frozen T2-v1 finite-tail diagnostic with explicit empirical-warning language.
- FAISS and Qdrant index adapters.
- Persistent target-vector cache and progressive materialization queue.
- FastAPI service, search/status/prewarm endpoints, and dashboard.
- Latency telemetry, economics projections, report generation, and demos.
- Public packaging, examples, tests, CI configuration, and contributor docs.

This is an early open-source release. See the limitations in the README and
`docs/concepts.md` before using it for a production migration.
