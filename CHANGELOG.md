# Changelog

## v0.6.0 — Migration planner

- Added an advisory `embedflow plan` command and Python API that combine
  source-index preflight, registry evidence, representative probes, frozen
  T2-v1 diagnostics, candidate-depth selection, cache planning, economics, and
  staged rollout guidance without routing traffic or mutating the source.
- Added structured JSON/YAML plan artifacts with explicit warnings and
  measured/user-supplied/modeled/unknown provenance.

## v0.5.0 — Weaviate backend

- Added a read-only Weaviate v4 backend for existing externally-vectorized
  dense collections, including named-vector selection and UUID IDs.
- Added HTTP/gRPC connection settings, property-backed or external text
  resolution, metric normalization, collection auditing, and safe client
  shutdown.
- Added the optional `weaviate-client` extra, Docker fixture, example, and
  live 10,000-object validation harness.

## v0.4.0 — Milvus backend

- Added a read-only Milvus backend for existing dense `FLOAT_VECTOR`
  collections.
- Added collection/database/partition selection, HNSW/IVF search parameters,
  metric and schema auditing, and Milvus-backed document text resolution.
- Added deterministic standalone Docker fixtures, examples, and optional
  `pymilvus` packaging.

## v0.3.0 — Pinecone backend

- Added a read-only Pinecone backend for existing dense indexes.
- Added host targeting, index-name resolution, namespace-aware retrieval, and
  metadata or external-document text resolution.
- Added Pinecone status/audit integration, optional dependency packaging, and
  unit/integration smoke fixtures.

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
