# Public methodology

This document describes the method intended for public reproduction. It does
not include private reviewer notes, unpublished operational data, or large
experiment archives.

## Mode A: evaluated migration

Inputs:

- source rankings or a source index and source model;
- target model and either a native target index or saved native rankings;
- document text lookup for target scoring of source candidates;
- query set and qrels;
- tested candidate depths `K`.

For each query and each `K`, EmbedFlow reranks source top-`K` candidates with
the target query/document contract. The current evaluator computes nDCG@k for
the native target ranking and the restricted ranking, then reports their
difference as `G(K)`. It also reports target-top-10 containment in the source
candidate list. Bootstrap intervals are paired by query when enabled.

If an exact/reference source ranking is supplied, ANN fidelity is measured as
overlap with that reference at a declared `k`. Without it, ANN status remains
`UNKNOWN`.

## Mode B: no target index

Inputs:

- source model and existing source index;
- target model;
- document lookup;
- unlabeled probe queries.

The frozen T2-v1 implementation computes finite-tail features from target
scores on the source candidate pool and applies the exact rule in
`src/t2_v1.py`. The public report calls its K output a recommended initial K.
No qrels, native target rankings, native target nDCG, or retrospective labels
are used by the diagnostic.

## Model contracts and cache safety

The model fingerprint is a SHA-256 digest over semantic contract fields:

- model ID and revision;
- dimension;
- max length;
- pooling;
- query/document instructions;
- padding and truncation sides;
- normalization;
- dtype.

Local model paths and devices are excluded because they are deployment details.
The target-vector cache key is `(document_id, target_model_fingerprint)` and
stores dimension, dtype, checksum, and timestamps. A cache entry from a
different model revision or prompt contract cannot be silently reused.

## Reproducibility

Record the following with every experiment:

- exact model revisions and contracts;
- corpus and query manifest hashes;
- qrels provenance;
- source index type, metric, and ANN settings;
- candidate depth list and quality cutoff;
- bootstrap seed/resample count;
- hardware and software versions.

The deterministic demo uses a tiny hash embedding model to exercise the
serving state machine without downloading weights. Real-model examples should
be described as functional demonstrations, not as benchmark results.
