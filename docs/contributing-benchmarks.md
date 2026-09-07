# Contributing benchmark evidence

Registry additions must be reproducible and explicit about scope. A pull
request should include a JSONL row in a reviewed namespace (community rows are
not trusted as core evidence) with:

- source and target model IDs, revisions, dimensions, pooling, prompts,
  padding/truncation, normalization, dtype, and contract fingerprints;
- corpus and query-set identity, row counts, split, and a reproducible
  fingerprint or canonical construction description;
- metric definition, epsilon, tested K grid, raw `G(K)` values, containment,
  and observed/CI-certified depths only where actually computed;
- frozen T2-v1 version/status and probe size when a finite-tail diagnostic is
  included; never include qrels or native target ranks in T2 features;
- ANN index type/configuration and an exact-source reference when reporting
  ANN fidelity;
- EmbedFlow version, hardware/software metadata for latency/throughput,
  reproduction commands, artifact checksums, and limitations.

Maintainers verify the provenance link, contract hash, finite values, and
internal arithmetic before considering promotion to `core`. A community row
does not become a compatibility decision merely by being present in the
registry.
