# EmbedFlow concepts

## Candidate compatibility

Let `S` be the deployed source model, `T` the desired target model, and `D`
the corpus. The source index returns `C^S_K(q)`, the top-`K` candidates for
query `q`. The target model scores only those candidates. Candidate
compatibility asks whether the target's useful results are present in that
source candidate pool.

Representation spaces may be incompatible while retrieval neighborhoods remain
compatible enough for a bounded candidate set. That is the product opportunity;
it is not an assumption that every model pair will work.

## Candidate gap and containment

For a retrieval metric `M`:

```text
G(K) = M_T - M_{T|S_K}
```

`M_T` is native target retrieval over the full corpus. `M_{T|S_K}` is target
retrieval restricted to source top-`K`. Smaller `G(K)` is better.

Containment is different: it is the fraction of native target top-`k` documents
that appear in the source candidate list. High containment does not imply a
small candidate gap because score ordering and graded relevance still matter.

## Observed migration depth

When qrels and native target retrieval are available, EmbedFlow reports:

```text
K*_epsilon = min { K : G(K) <= epsilon }
```

`epsilon=0.01` is a stringent reference setting, not a universal production
safety threshold. The curve is empirical and should be accompanied by query
counts and uncertainty intervals.

Without a native target index, there is no observed `M_T`, so EmbedFlow uses
the phrase **recommended initial candidate depth** for the finite-tail/T2-v1
recommendation. It does not call that value `K*`.

## T2-v1

T2-v1 is a frozen leakage-safe finite-tail diagnostic used before a native
target index exists. It examines probe behavior as the source candidate depth
grows and returns exactly:

```text
SAFE
EXPAND
UNSAFE_OR_UNCERTAIN
```

`SAFE` means the observed finite-tail features satisfy the frozen rule. It is
not a guarantee, proof, retrospective compatibility label, or replacement for
qrels/native target evaluation. The implementation is reused from
`src/t2_v1.py` and `src/probe_features.py`; it is hash-checked against the
frozen specification.

## ANN decomposition

For ANN configuration `A`:

```text
G_A(K) = G(K) + P_A(K)
```

`G(K)` is intrinsic source-target candidate compatibility. `P_A(K)` is the
additional penalty caused by approximate source retrieval. A T2-v1 `SAFE`
diagnostic says nothing about ANN health. EmbedFlow reports ANN health as
`UNKNOWN` until an exact/reference source comparison is supplied; a supplied
reference comparison is reported separately.

## Progressive serving

Serving starts from the existing source index. At request time EmbedFlow:

1. encodes the query with the source and target contracts;
2. retrieves source top-`K` candidates;
3. reads target document vectors from the persistent cache;
4. encodes at most `max_sync_misses` missing documents synchronously;
5. enqueues remaining misses for the durable background worker;
6. target-reranks whatever target vectors are available.

The response explicitly identifies `COLD`, `PARTIAL`, or `WARM` based on the
target vectors available for that response, including any bounded synchronous
encodes. The separate `cache_hits`, `cache_misses`, `sync_encoded`, and
`async_queued` fields show how that state was reached. A partial ranking is not
claimed to be identical to a fully warm target rerank. Once all candidate
vectors are warm, target scoring over that same candidate set is deterministic.
