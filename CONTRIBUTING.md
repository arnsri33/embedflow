# Contributing to EmbedFlow

Thanks for helping make progressive embedding migration useful and rigorous.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional integrations can be installed with `.[faiss]`, `.[qdrant]`,
`.[models]`, or `.[dashboard]`.

## Checks before opening a pull request

```bash
ruff check .
pytest -q
python -m build
```

Keep tests CPU-safe. Mark GPU-only tests explicitly and make them skip cleanly
when CUDA or model weights are unavailable.

## Adding a vector database backend

Implement the `VectorIndex` contract in `embedflow/indexes/base.py`, including
`search`, `size`, `metadata`, and persistence/connection behavior. Add the
backend behind an explicit optional dependency, validate dimensions and metric
semantics, document credentials through environment variables, and add an
integration test that skips when the service is unavailable. Do not report ANN
health as PASS unless an exact/reference comparison has actually been run.

## Adding a model adapter

Preserve the `EmbeddingModel` contract: query/document encoding, dimension,
normalization, pooling, prompts, and a semantic fingerprint. Add tests for
the fingerprint and for query/document shape and normalization behavior.

## Research and reproducibility

Do not modify the frozen T2-v1 implementation or its hash specification as a
side effect of product changes. Changes to metrics or evaluation semantics
must include a regression test and an explanation in the pull request.

## Pull requests

Describe the user-visible behavior, compatibility implications, test commands,
and any limitations. Keep changes focused, avoid committing generated caches,
model weights, private data, or credentials, and update documentation for new
CLI flags or configuration fields.
