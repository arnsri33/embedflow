# T2-v1 frozen specification for Qwen3-8B target-shift validation

**Frozen before any Qwen3-8B target-shift feature/result inspection:**
2026-08-23

This is an exact copy of the previously frozen representative T2 rule.  It
must not be altered during this experiment.

## Probe-only inputs and normalization

The probe may use source-ranked candidate IDs, source ranks and scores, exact
target scores for documents in the finite source candidate pool, and stable
target rerankings of source prefixes.  Embeddings are treated as the existing
normalized vectors and candidate scores use the audited target/source dot
product convention in the saved candidate tables.  No cross-model score
calibration is performed inside the probe.

The finite reference is `R_500`: target reranking restricted to the source
top-500 candidates.  `R_500` is not native target retrieval.

The probe must not use qrels, native full-corpus target rankings, target nDCG,
candidate gaps, confidence intervals, or any retrospective compatibility
label.

## Frozen K values

`K = {10, 20, 50, 100, 200, 500}`.  T2-v1 is evaluated at K=50 and uses
the source-top-500 finite reference.

## Frozen feature definitions

For each query and source prefix:

* `probe_residual_tail(K) = 1 - |Top10(R_K) ∩ Top10(R_500)| / 10`.
* `stability_to_500(K) = |Top10(R_K) ∩ Top10(R_500)| / 10`.
* `deepest_source_rank_top10` is the maximum original source rank in
  `Top10(R_500)`; `deepest_p90` is its query-level 90th percentile.
* For each adjacent source-shell expansion `K→K'`, `p_any_entrant` is the
  fraction of queries in which at least one newly exposed document enters
  `Top10(R_K')`.
* `late_tail_area` is the prior finite-shell weighted summary of entrant
  probabilities over expansions 50→100, 100→200, and 200→500.
* `fraction_margin_nonpositive` and `last_shell_any_rate` are computed with
  the prior implementation and are used only by the fixed high-risk guard.

## Exact T2-v1 decision logic

At K=50, output `SAFE` iff all of the following are true:

1. mean `probe_residual_tail(50) <= 0.05`;
2. `deepest_p90 <= 200`;
3. `late_tail_area <= 0.20`.

If SAFE is false, output `UNSAFE_OR_UNCERTAIN` iff at least one of these
fixed guards is true:

* mean `probe_residual_tail(50) > 0.10`;
* mean `stability_to_500(50) < 0.90`;
* `last_shell_any_rate > 0.25`;
* `fraction_margin_nonpositive(50) > 0.50`.

Otherwise output `EXPAND`.

## Evaluation-only labels

Only after blind predictions are persisted may the evaluation join the
authoritative `G(50) = native target nDCG@10 − exact target rerank nDCG@10
within source top-50` labels.  Practical compatibility is `G(50) <= 0.01`
using unrounded values.  CI-NI fields are secondary evaluation labels.

**T2-v1 was frozen before any Qwen3-8B target-shift results were inspected.**
