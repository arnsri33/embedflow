# Installation

EmbedFlow supports Python 3.10–3.12. The base package contains configuration,
registry, metrics, and the deterministic demo. Optional extras add integrations
and model runtimes.

## Local checkout

```bash
git clone https://github.com/arnsri33/embedflow.git
cd embedflow
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[faiss,dashboard]"
```

Use `requirements.txt` for the portable runtime set and
`requirements-dev.txt` for linting, tests, and packaging:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Optional extras

| Extra | Adds |
| --- | --- |
| `faiss` | FAISS source-index adapter |
| `qdrant` | Qdrant client and adapter |
| `pgvector` | Psycopg 3 binary driver and pgvector adapter |
| `pinecone` | Official Pinecone Python SDK and adapter |
| `models` | PyTorch, Transformers, Sentence Transformers, and Hub client |
| `dashboard` | FastAPI, Uvicorn, and Pydantic |
| `dev` | Pytest, Ruff, and build tooling |
| `all` | All runtime integrations |

For a real model-backed local install:

```bash
python -m pip install -e ".[faiss,models,dashboard]"
```

Model weights are downloaded by the selected runtime or loaded from paths you
provide. They are not part of the repository. Set `HF_HOME` or the runtime's
normal cache setting if you want to control the download location.

## Qdrant

Install the client with `.[qdrant]`. A local Qdrant server can be configured
with `index.url`; a user-owned cloud or remote server can use an API key named
by `index.api_key_env`. Keep the key in the environment. See
[`integrations/qdrant.md`](integrations/qdrant.md).

## PostgreSQL / pgvector

Install the optional adapter with `python -m pip install "embedflow[pgvector]"`.
The adapter connects to an existing table and reads the DSN from the
environment; see [`integrations/pgvector.md`](integrations/pgvector.md).

## Pinecone

Install the optional adapter with `python -m pip install "embedflow[pinecone]"`.
Set `PINECONE_API_KEY` in the environment and configure an existing dense
index host; see [`integrations/pinecone.md`](integrations/pinecone.md).

## CPU and GPU

The deterministic demo runs on CPU. Real model serving accepts `--device cpu`
or `--device cuda`; the model runtime must support the selected device. CUDA
availability is checked by `embedflow doctor`.

## Installation checks

```bash
embedflow --help
embedflow doctor
embedflow registry verify
embedflow demo
```

`doctor` reports missing optional dependencies while the base package keeps its
startup path lightweight.
