# Quickstart

This page covers the two supported workflows: a no-target-index migration and
a labelled evaluation with native target evidence.

## Offline demo

From the repository root:

```bash
./scripts/run_demo.sh
```

The script creates a tiny corpus and source index, runs the finite-tail probe,
starts the local API, and serves deterministic hash embeddings. Open
<http://127.0.0.1:8000/> and submit the same query twice. The first response
can be `COLD` or `PARTIAL`; queued target vectors make later responses `WARM`.

Use `./scripts/run_demo.sh --no-serve` for a setup-only smoke test.

## Analyze without a target index

Prepare a JSONL document store and unlabeled probe queries, then run:

```bash
embedflow analyze \
  --documents ./documents.jsonl \
  --index ./legacy.index \
  --source-model sentence-transformers/all-MiniLM-L6-v2 \
  --target-model Qwen/Qwen3-Embedding-0.6B \
  --probe-queries ./probe_queries.jsonl \
  --model-root ./models \
  --device cuda \
  --output-dir ./analysis
```

Use `--device cpu` on a CPU-only machine. The command writes a report and a
reusable YAML configuration. It also checks the built-in registry and shows
prior evidence when the model contracts have been studied before.

## Serve progressively

```bash
embedflow serve --config ./analysis/embedflow.analysis.yaml --device cuda
```

The service keeps the source index online, bounds synchronous target encoding,
and sends remaining misses to a durable materialization queue. Inspect the
state with:

```bash
embedflow status --config ./analysis/embedflow.analysis.yaml
```

## Evaluate with qrels

For a research evaluation with qrels and native target evidence:

```bash
embedflow evaluate \
  --config ./experiment.yaml \
  --queries ./queries.jsonl \
  --qrels ./qrels.json \
  --native-target-index ./target.index \
  --reference-index ./exact-source.index \
  --k-values 10,20,50,100,200,500 \
  --epsilon 0.01 \
  --output-dir ./results
```

See [`methodology.md`](methodology.md) for the inputs and metrics.
