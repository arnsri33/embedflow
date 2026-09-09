# CLI reference

Run `embedflow --help` or `embedflow <command> --help` for the live option
list. The commands below are the main public entry points.

## Project and analysis

```bash
embedflow init --config ./embedflow.yaml
embedflow analyze --config ./embedflow.yaml --output-dir ./analysis
embedflow evaluate --config ./experiment.yaml --output-dir ./results
```

`analyze` is the no-target-index workflow. It uses probe queries and frozen
T2-v1 logic. `evaluate` is the labelled workflow; it computes source quality,
native target quality, target-within-source-candidates quality, candidate gap,
containment, and migration depth when the required inputs are available.

## Serving and operations

```bash
embedflow serve --config ./embedflow.yaml
embedflow search --config ./embedflow.yaml "what causes auroras?"
embedflow status --config ./embedflow.yaml
embedflow prewarm --config ./embedflow.yaml --documents 10000 --async
embedflow audit-index --config ./embedflow.yaml
embedflow export-target --config ./embedflow.yaml --output-index ./target.index
```

`status` reports cache coverage, hit/miss counters, queue depth, and
materialization throughput. `audit-index` checks the source index against an
exact/reference configuration where supported. `prewarm` schedules target
document work; it does not change source-index results.

## Registry and profiles

```bash
embedflow registry list
embedflow registry show \
  --source Qwen/Qwen3-Embedding-4B \
  --target Qwen/Qwen3-Embedding-8B
embedflow registry match --config ./embedflow.yaml
embedflow registry verify
embedflow benchmark-profiles list
```

Registry matching uses model contracts and corpus identity. See
[`registry.md`](registry.md).

## Economics and diagnostics

```bash
embedflow economics \
  --corpus-size 1000000000 \
  --docs-per-second 100 \
  --gpu-price 3.29
embedflow doctor --config ./embedflow.yaml
embedflow demo
```

Economics is a projection from the throughput and price supplied on the
command line. `doctor` checks Python, optional dependencies, paths,
dimensions, cache integrity, and contract fingerprints.

## Output files

Analysis writes a human-readable report and `migration_report.json`. Research
evaluation writes `results.json`, `results.csv`,
`candidate_gap_curve.csv`, `containment_curve.csv`, and `report.md` under the
selected output directory.
