# Limitations and release scope

EmbedFlow v0.1.0 is an alpha release for research and early real-world
testing. The serving path is designed to make migration experiments concrete;
production rollout still requires application-specific validation.

## Compatibility

Candidate gap, containment, and T2-v1 describe different evidence. Candidate
gap uses native target retrieval and qrels. Containment measures neighborhood
overlap. T2-v1 uses finite-tail probe behavior before a native target index
exists and returns `SAFE`, `EXPAND`, or `UNSAFE_OR_UNCERTAIN`.

The `epsilon=0.01` reference is a stringent study setting. It is not a
universal production threshold. A recommended initial depth in no-target-index
mode is a deployment starting point, not an observed `K*`.

## Serving

`PARTIAL` responses rank only candidates whose target vectors are available at
that moment. The bounded synchronous budget and background queue make this
state explicit. Once the same candidate set is warm, target reranking is
deterministic.

ANN fidelity is independent of T2-v1. EmbedFlow reports `UNKNOWN` until an
exact/reference source comparison is supplied.

## Measurements

The registry contains retained research results, including studies through 1M
documents and a separate workload-specific serving profile. These records are
useful prior evidence with documented contracts and provenance. They do not
replace a new corpus evaluation. Latency and economics vary with hardware,
model runtime, corpus, batch size, and workload.

The historical Qwen3-0.6B source rows use a max-length-512 contract and the
retained trailing-space query behavior. The Qwen3-4B and 8B rows use the
verified max-length-8192 contract; parameter count alone is not an explanation
for the observed differences.
