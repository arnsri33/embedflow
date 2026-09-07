# Known migration evidence registry

EmbedFlow's packaged registry is a versioned set of small JSON/JSONL files
under `embedflow/data/registry/`. It contains only the research values that
can be traced to retained tables. Model weights, vector indexes, raw corpora,
and runtime caches are intentionally excluded.

## What is in core

The v0.1.0 `core` namespace contains:

- three pooled BRIGHT four-domain source → Qwen3-8B candidate-gap curves;
- twelve Natural Questions canonical-prefix scale rows (100K, 250K, 500K,
  and 1M for MiniLM, Qwen3-0.6B, and Qwen3-4B → Qwen3-8B);
- the retained 63-cell development count audit and 28-cell frozen T2-v1
  holdout summary;

The development summary reports **26/63 CI-certified cells** (25/45
point-estimate-compatible and 11/45 CI-certified after excluding NanoQuora and
NanoMSMARCO). These are aggregate counts, not a migration depth or a model-pair
guarantee.
- measured, workload-specific 25K serving latency and Qwen3-8B model-profile
  records.

Every migration row stores the model contract, candidate depths, `G(K)`,
containment, epsilon, observed/CI-certified depths when available, and a
provenance artifact plus SHA-256. A null field means that the retained
artifact did not establish that value.

## Match levels

`embedflow registry match --config config.yaml` compares full contract
fingerprints, not display names:

1. **EXACT REGISTRY MATCH** — source and target contracts plus the canonical
   corpus construction/fingerprint match. `--use-registry` may reuse those
   canonical values, and the report records exactly what was reused.
2. **PRIOR EVIDENCE AVAILABLE** — both contracts match, but the corpus is new
   or its fingerprint is unavailable. Prior curves can prioritize a K grid;
   they cannot produce `SAFE` for the new corpus.
3. **RELATED EVIDENCE ONLY** — a model family/name is related, but a revision,
   prompt, pooling, max length, padding, truncation, normalization, or other
   contract field differs. Run a new analysis.

Matching never treats a model parameter count as a causal explanation. In
particular, the retained Qwen3-0.6B scale rows use a historical max-length-512
source contract and its historical trailing-space query-template behavior;
the Qwen3-4B/8B rows use the verified max-length-8192 contract.

## Verification

Run:

```bash
embedflow registry verify
```

Verification checks schema and registry versions, JSONL validity, duplicate
IDs, contract fingerprints, finite metric values, candidate-depth consistency,
observed-depth consistency, provenance digest shape, and packaged file
checksums. External provenance paths are informational; they need not exist on
an installed machine. A checksum mismatch is a release failure.

## Scientific interpretation

`G(K) = M_T - M_{T|S_K}` is a candidate-gap measurement from a labelled/native
target evaluation. Containment is reported separately. A negative gap is a
valid finite-sample outcome and is not silently clipped. `K*` is an observed
point estimate only when native target evidence/qrels exist; a CI-certified
depth is a separate conservative certificate. The epsilon `0.01` reference is
not a universal production safety threshold.

T2-v1 rows are leakage-safe finite-tail diagnostics and use only source
candidates and target scores within that candidate pool. `SAFE` is empirical,
not a proof or guarantee. ANN fidelity is independent: without an exact source
reference, status remains `UNKNOWN` even when T2 is `SAFE`.

## Extending the registry

Core rows are reviewed and tied to retained artifacts. New reproducible rows
should start in the community namespace and include the fields described in
[`contributing-benchmarks.md`](contributing-benchmarks.md). Do not commit
embeddings, indexes, credentials, proprietary datasets, or unverifiable
headline numbers.
