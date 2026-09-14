# EmbedFlow

**Progressive embedding-model migration over existing vector indexes.**

EmbedFlow lets a new embedding model serve over candidates from an existing
vector index while target document vectors are materialized progressively. It
supports migration analysis, persistent caching, background work, FAISS,
Qdrant, pgvector, Pinecone, Milvus, Weaviate, a CLI, and FastAPI.

The full project README and architecture diagram are on
<https://github.com/arnsri33/embedflow>.

## Install

```bash
python -m pip install embedflow
```

For FAISS and the dashboard:

```bash
python -m pip install "embedflow[faiss,dashboard]"
```

For an existing PostgreSQL/pgvector table:

```bash
python -m pip install "embedflow[pgvector]"
```

For an existing Pinecone dense index:

```bash
python -m pip install "embedflow[pinecone]"
```

For an existing Milvus collection:

```bash
python -m pip install "embedflow[milvus]"
```

For an existing Weaviate v4 collection:

```bash
python -m pip install "embedflow[weaviate]"
```

Qdrant and model-runtime extras are documented in the
[installation guide](https://github.com/arnsri33/embedflow/blob/main/docs/installation.md).
For model-backed analysis, install `embedflow[faiss,models,dashboard]`.

## Why EmbedFlow?

Embedding-model upgrades usually mean re-embedding the corpus and building a
second index before the new model can serve. EmbedFlow tests whether the
existing retriever can remain useful during that transition.

Different representation spaces can still preserve useful retrieval
neighborhoods. EmbedFlow retrieves a bounded source candidate set, scores those
candidates with the target model, and fills a persistent target-vector cache in
the background.

## Try it

The deterministic demo needs no paid service or model download:
install `embedflow[faiss,dashboard]` to use its browser UI.

```bash
embedflow demo
```

Open <http://127.0.0.1:8000/>. The first search can be `COLD` or `PARTIAL`;
repeated traffic becomes `WARM` as target vectors are materialized.

## Analyze a migration

```bash
embedflow analyze \
  --documents ./documents.jsonl \
  --index ./legacy.index \
  --source-model sentence-transformers/all-MiniLM-L6-v2 \
  --target-model Qwen/Qwen3-Embedding-0.6B \
  --probe-queries ./probe_queries.jsonl \
  --device cuda \
  --output-dir ./analysis
```

The report includes the frozen T2-v1 diagnostic (`SAFE`, `EXPAND`, or
`UNSAFE_OR_UNCERTAIN`), a recommended initial candidate depth, and ANN health.
`SAFE` is an empirical deployment signal; validate important migrations on the
target corpus.

Start progressive serving with:

```bash
embedflow serve --config ./analysis/embedflow.analysis.yaml --device cuda
```

## Known migration evidence

EmbedFlow ships a versioned registry of measured results from the research
study. Matching model contracts provide useful starting depths and show what
was observed on earlier corpora; a new corpus still receives its own analysis.

```bash
embedflow registry list
embedflow registry show \
  --source Qwen/Qwen3-Embedding-4B \
  --target Qwen/Qwen3-Embedding-8B
```

Selected core records (nDCG@10, `G(50)`):

| Source | Target | Evaluation | `G(50)` | Observed depth |
| --- | --- | --- | ---: | ---: |
| MiniLM-L6-v2 | Qwen3-8B | BRIGHT, 413K | 0.03665 | — |
| Qwen3-0.6B | Qwen3-8B | BRIGHT, 413K | 0.01465 | 200 |
| Qwen3-4B | Qwen3-8B | BRIGHT, 413K | 0.00347 | 20 |
| MiniLM-L6-v2 | Qwen3-8B | Natural Questions, 1M | 0.03255 | 500 |
| Qwen3-4B | Qwen3-8B | Natural Questions, 1M | -0.00043 | 20 |

See the [registry documentation](https://github.com/arnsri33/embedflow/blob/main/docs/registry.md)
for matching levels, contract fingerprints, and provenance.

## Advisory migration plans

Use representative query probes to produce a bounded migration recommendation
before serving:

```bash
embedflow plan --config ./embedflow.yaml --queries ./probe_queries.jsonl
embedflow plan --config ./embedflow.yaml --queries ./probe_queries.jsonl --format json --output migration-plan.json
```

The planner reports preflight, registry evidence, frozen T2-v1 finite-tail
behavior, candidate K, cache/economics projections, and staged rollout
guidance. It never routes traffic or mutates the source index; `SAFE` is not a
qrels-based retrieval-quality guarantee. See the [planner guide](https://github.com/arnsri33/embedflow/blob/main/docs/planner.md).

### Source-authoritative Shadow Mode

Run the target migration path beside sampled traffic while always returning the
source result:

```yaml
runtime: {mode: shadow}
shadow: {enabled: true, sample_rate: 0.10, candidate_k: 100, materialize: true}
```

```bash
embedflow serve --config ./embedflow.yaml
embedflow shadow report --config ./embedflow.yaml --since 24h --format json
```

Shadow work is asynchronous, bounded, privacy-conscious, and failure isolated.
Reports contain operational and ranking-disagreement diagnostics—not qrel-based
quality guarantees—and recommendations never route canary traffic. See the
[Shadow Mode guide](https://github.com/arnsri33/embedflow/blob/main/docs/shadow-mode.md).

## Research

For candidate depth `K`, EmbedFlow measures:

```text
G(K) = M_T - M_{T|S_K}
```

Lower `G(K)` means the source candidate pool recovers more of native target
retrieval quality. Containment reports neighborhood overlap separately. The
study includes 63 development settings, frozen BRIGHT validation, and Natural
Questions scale experiments through 1M documents.

Read the [concepts](https://github.com/arnsri33/embedflow/blob/main/docs/concepts.md)
and [methodology](https://github.com/arnsri33/embedflow/blob/main/docs/methodology.md)
for definitions and reproduction details.

## Integrations

| Backend | Status |
| --- | --- |
| FAISS | Supported |
| Qdrant | Supported |
| pgvector | Supported |
| Pinecone | Supported |
| Milvus | Supported |
| Weaviate | Supported |

See the [FAISS guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/faiss.md),
[Qdrant guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/qdrant.md),
and [pgvector guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/pgvector.md),
and [Pinecone guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/pinecone.md),
and [Milvus guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/milvus.md).
See the [Weaviate guide](https://github.com/arnsri33/embedflow/blob/main/docs/integrations/weaviate.md).

## CLI

```bash
embedflow --help
embedflow analyze --help
embedflow plan --help
embedflow serve --config ./embedflow.yaml
embedflow status --config ./embedflow.yaml
embedflow registry list
embedflow economics --corpus-size 1000000000 --docs-per-second 100 --gpu-price 3.29
embedflow doctor --config ./embedflow.yaml
```

The [CLI reference](https://github.com/arnsri33/embedflow/blob/main/docs/cli.md)
and [API reference](https://github.com/arnsri33/embedflow/blob/main/docs/api.md)
cover the remaining commands and endpoints.

## Status

EmbedFlow v0.7.0 is a pre-1.0 release for research and early real-world
testing. T2-v1 is an empirical finite-tail diagnostic, partial rankings can
differ from fully warm target reranking, and ANN fidelity needs a reference
comparison to audit.

## Links

- [GitHub repository](https://github.com/arnsri33/embedflow)
- [Documentation](https://github.com/arnsri33/embedflow#readme)
- [License](https://github.com/arnsri33/embedflow/blob/main/LICENSE)
