# Research evaluation example

This tiny fixture demonstrates Mode A with qrels. It is not a benchmark. Build
the source index, then evaluate the candidate gap curve:

```bash
cd examples/research_analysis
PYTHONPATH=../.. embedflow init --config embedflow.yaml --build-index --queries queries.jsonl --demo
PYTHONPATH=../.. embedflow evaluate --config embedflow.yaml --queries queries.jsonl \
  --qrels qrels.json --k-values 2,4,8 --demo --output-dir results
```

Because no native target index is supplied, the target model encodes this tiny
corpus in-process to establish `M_T`. For a serious experiment, provide a
native target index or saved native rankings and record the exact manifest and
model contracts.
