"""Small public facade for starting a progressive model migration.

The research implementation deliberately exposes the lower-level config and
engine objects.  This module adds the short path a user needs in an
application: point EmbedFlow at an existing FAISS/Qdrant index, name the old
and new embedding models, and receive a serving session.  All retrieval,
cache, worker, and probe behavior is delegated to the existing implementation;
this is not a second query path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..cache import SQLiteVectorCache
from ..config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    hydrate_research_contract,
    save_config,
)
from ..indexes import PgVectorDocumentStore, PgVectorIndex
from ..indexes.base import VectorIndex
from ..migration.compatibility import run_probe, save_probe
from ..migration.state import DocumentStore
from ..models import EmbeddingModel, load_embedding_model
from ..runtime import load_index
from ..serving.engine import MigrationEngine

_MODEL_REGISTRY = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b",
    "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
    "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
}


def _infer_backend(index_value: str, explicit: str | None = None) -> str:
    """Infer a backend from an index value when the caller omits ``backend``."""
    if explicit:
        return str(explicit).lower()
    lowered = str(index_value).strip().lower()
    if lowered.startswith(("postgresql://", "postgres://")):
        return "pgvector"
    if "://" in lowered:
        return "qdrant"
    return "faiss"


def _is_model(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("encode_queries", "encode_documents", "dimension", "fingerprint"))


def _model_config(value: str | EmbeddingModel, model_root: str | Path | None = None) -> ModelConfig:
    """Build a contract for a model identifier or an already-loaded model."""
    if _is_model(value):
        # HuggingFaceEmbeddingModel retains its exact hydrated config.  Reuse
        # it when available so fingerprints and prompt contracts remain exact.
        retained = getattr(value, "cfg", None)
        if isinstance(retained, ModelConfig):
            return retained
        return ModelConfig(model=str(value.model_id), dimension=int(value.dimension))
    if not isinstance(value, (str, Path)):
        raise TypeError("old_model/new_model must be a model ID/path or EmbeddingModel instance")
    identifier = str(value)
    cfg = hydrate_research_contract(ModelConfig(identifier), project_root=Path(__file__).resolve().parents[2])
    candidate = Path(identifier)
    if candidate.exists():
        cfg.local_path = str(candidate.resolve())
    elif model_root is not None and identifier in _MODEL_REGISTRY:
        staged = Path(model_root).expanduser().resolve() / _MODEL_REGISTRY[identifier]
        if staged.exists():
            cfg.local_path = str(staged)
    return cfg


def _queries(value: str | Path | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[tuple[str, str]] = []
        with path.open() as handle:
            seen: set[str] = set()
            for i, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid query JSON at {path}:{i}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"query row {i} in {path} must be a JSON object")
                query_id = str(row.get("id", row.get("query_id", i)))
                text = row.get("text", row.get("query"))
                if query_id in seen:
                    raise ValueError(f"duplicate query ID {query_id!r} in {path}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"query {query_id!r} has no text")
                seen.add(query_id)
                rows.append((query_id, text))
        if not rows:
            raise ValueError(f"query file is empty: {value}")
        return rows
    try:
        rows = [(str(query_id), text) for query_id, text in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("probe_queries must be an iterable of (query_id, text) pairs") from exc
    if len({query_id for query_id, _ in rows}) != len(rows):
        raise ValueError("probe_queries must not contain duplicate query IDs")
    if not rows or any(not isinstance(text, str) or not text.strip() for _, text in rows):
        raise ValueError("probe_queries must contain non-empty query text")
    return rows


class MigrationSession:
    """Handle returned by :func:`migrate`.

    ``search`` and ``status`` are synchronous convenience methods for an
    application.  ``serve`` mounts the same FastAPI dashboard/API used by the
    CLI.  Use ``close`` or a context manager to release models and the cache.
    """

    def __init__(self, engine: MigrationEngine, config: EmbedFlowConfig,
                 config_path: Path | None, owns_source: bool, owns_target: bool, owns_index: bool):
        self.engine = engine
        self.config = config
        self.config_path = config_path
        self._owns_source = owns_source
        self._owns_target = owns_target
        self._owns_index = owns_index
        self._closed = False

    @property
    def plan(self):
        return self.engine.plan

    def search(self, query: str, top_k: int = 10, candidate_depth: int | None = None,
               max_sync_misses: int | None = None) -> dict[str, Any]:
        return self.engine.search(query, top_k, candidate_depth, max_sync_misses)

    def status(self) -> dict[str, Any]:
        return self.engine.status()

    def prewarm(self, document_ids: list[str], asynchronous: bool = True) -> dict[str, Any]:
        return self.engine.prewarm(document_ids, asynchronous=asynchronous)

    def app(self):
        from ..serving.api import create_app
        return create_app(self.engine)

    def serve(self, host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
        """Run the API/dashboard until interrupted."""
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("serving requires uvicorn; install `embedflow[dashboard]`") from exc
        uvicorn.run(self.app(), host=host, port=int(port), log_level=log_level)

    def close(self) -> None:
        if self._closed:
            return
        self.engine.close(close_models=False, close_indexes=self._owns_index)
        if self._owns_source:
            self.engine.source_model.close()
        if self._owns_target:
            self.engine.target_model.close()
        self._closed = True

    def __enter__(self) -> MigrationSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def migrate(
    *,
    index: str | Path | VectorIndex,
    old_model: str | Path | EmbeddingModel,
    new_model: str | Path | EmbeddingModel,
    documents: str | Path | DocumentStore | PgVectorDocumentStore | None = None,
    backend: str | None = None,
    index_url: str | None = None,
    collection: str = "embedflow",
    vector_name: str | None = None,
    api_key_env: str | None = "QDRANT_API_KEY",
    dsn_env: str | None = "EMBEDFLOW_PGVECTOR_DSN",
    schema: str = "public",
    table: str = "documents",
    id_column: str = "id",
    vector_column: str = "embedding",
    text_column: str | None = "content",
    hnsw_ef_search: int | None = None,
    ivfflat_probes: int | None = None,
    metric: str = "cosine",
    model_root: str | Path | None = None,
    device: str | None = None,
    cache_path: str | Path = "./embedflow_cache",
    state_path: str | Path = "./embedflow_state.json",
    config_path: str | Path | None = None,
    candidate_depth: int = 50,
    kmax_probe: int = 500,
    max_sync_misses: int = 4,
    background_batch_size: int = 32,
    probe_queries: str | Path | Iterable[tuple[str, str]] | None = None,
    probe_limit: int | None = None,
    start_worker: bool = True,
) -> MigrationSession:
    """Start progressive migration over an existing index.

    ``index`` may be an existing EmbedFlow ``VectorIndex`` instance or a path
    to a FAISS index, Qdrant path/URL, or pgvector DSN.  Model
    strings are revision-hydrated when they are registered by the research
    contract; model objects can be supplied by an application directly.
    ``probe_queries`` is optional so serving can start immediately.  When
    supplied, the existing frozen T2-v1 implementation is run before the
    session is returned.
    """
    if isinstance(documents, (DocumentStore, PgVectorDocumentStore)):
        document_store = documents
    elif documents is None:
        document_store = None
    else:
        document_store = DocumentStore(str(documents))
    source_cfg = _model_config(old_model, model_root)
    target_cfg = _model_config(new_model, model_root)
    selected_device = device or target_cfg.device or source_cfg.device or "cpu"

    owns_source = not _is_model(old_model)
    owns_target = not _is_model(new_model)
    owns_index = not (isinstance(index, VectorIndex) or all(hasattr(index, name) for name in ("search", "size", "dimension", "metadata")))
    source_model = None
    target_model = None
    source_index = None
    cache = None
    engine = None
    try:
        source_model = old_model if not owns_source else load_embedding_model(source_cfg, model_root=model_root, device=selected_device)
        target_model = new_model if not owns_target else load_embedding_model(target_cfg, model_root=model_root, device=selected_device)
        source_cfg.dimension = int(source_model.dimension)
        target_cfg.dimension = int(target_model.dimension)

        if not owns_index:
            source_index = index
            index_backend = backend or str(source_index.metadata().get("backend", "faiss")).lower()
            index_path = str(getattr(source_index, "path", "./legacy.index") or "./legacy.index")
        else:
            index_path = str(index_url or index)
            index_backend = _infer_backend(index_path, backend)
            if index_backend not in {"faiss", "qdrant", "pgvector"}:
                raise ValueError("backend must be faiss, qdrant, or pgvector")

        explicit_url = index_url
        if index_backend == "pgvector" and explicit_url is None and "://" in index_path:
            explicit_url = index_path
        if index_backend == "pgvector" and explicit_url:
            # A direct DSN is accepted for convenience, but generated config
            # files must remain safe to commit. Resolve it through the named
            # environment variable for this process and persist only a local
            # placeholder path plus the variable name.
            if not dsn_env or not str(dsn_env).strip():
                raise ValueError("pgvector dsn_env is required when an explicit DSN is supplied")
            os.environ[str(dsn_env).strip()] = explicit_url
            explicit_url = None
            index_path = "./legacy.index"
        if document_store is None and index_backend != "pgvector":
            raise ValueError("documents is required for FAISS and Qdrant; pgvector can resolve text from its text_column")
        cfg = EmbedFlowConfig(
            source=source_cfg,
            target=target_cfg,
            index=IndexConfig(backend=index_backend, path=index_path, collection=collection,
                              url=explicit_url, metric=str(metric), vector_name=vector_name,
                              api_key_env=api_key_env, dsn_env=dsn_env, schema=schema, table=table,
                              id_column=id_column, vector_column=vector_column, text_column=text_column,
                              hnsw_ef_search=hnsw_ef_search, ivfflat_probes=ivfflat_probes),
            documents=DocumentsConfig(path=str(document_store.path) if document_store is not None else "./documents.jsonl"),
            migration=MigrationConfig(candidate_depth=int(candidate_depth), kmax_probe=max(int(kmax_probe), int(candidate_depth)),
                                      max_sync_misses=int(max_sync_misses), background_batch_size=int(background_batch_size)),
            cache=CacheConfig(path=str(cache_path)),
            state_path=str(state_path),
            dashboard_title=f"EmbedFlow — {source_cfg.model} → {target_cfg.model}",
        )
        base = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
        cfg.resolve_paths(base)
        cfg.validate()

        if owns_index:
            if document_store is None:
                source_index = PgVectorIndex.from_config(cfg)
                document_store = PgVectorDocumentStore(source_index, text_field=cfg.index.text_column or "content", owns_index=False)
            else:
                source_index = load_index(cfg, document_store)
        elif document_store is None:
            if not isinstance(source_index, PgVectorIndex):
                raise ValueError("documents is required unless the supplied index is a pgvector backend")
            document_store = PgVectorDocumentStore(source_index, text_field=cfg.index.text_column or "content", owns_index=False)
        if int(source_index.dimension) != int(source_model.dimension):
            raise ValueError(f"source model dimension {source_model.dimension} != existing index dimension {source_index.dimension}")
        stored_fingerprint = source_index.metadata().get("model_fingerprint")
        if stored_fingerprint and stored_fingerprint != source_model.fingerprint:
            raise ValueError("source model fingerprint does not match the existing index contract")

        cache = SQLiteVectorCache(cfg.cache.path, target_model.fingerprint, target_model.dimension)
        probe: dict[str, Any] = {}
        if probe_queries is not None:
            probe = run_probe(source_model, target_model, source_index, document_store, _queries(probe_queries),
                              kmax=cfg.migration.kmax_probe, seed=42, limit=probe_limit)
            if config_path:
                save_probe(probe, Path(config_path).expanduser().resolve().with_name("probe_result.json"))
        if config_path:
            config_file = Path(config_path).expanduser().resolve()
            save_config(cfg, config_file)
        else:
            config_file = None
        engine = MigrationEngine(cfg, source_model, target_model, source_index, cache, document_store,
                                 probe=probe, start_worker=start_worker)
        return MigrationSession(engine, cfg, config_file, owns_source, owns_target, owns_index)
    except Exception:
        # A failed setup must not leak a loaded model, SQLite connection, or a
        # local Qdrant file lock. Only close resources created by this call;
        # caller-owned model/index objects remain their caller's responsibility.
        if engine is not None:
            try:
                engine.close(close_models=False, close_indexes=owns_index)
            except Exception:
                pass
        elif cache is not None:
            try:
                cache.close()
            except Exception:
                pass
        if owns_index and source_index is not None:
            try:
                close_index = getattr(source_index, "close", None)
                if callable(close_index):
                    close_index()
            except Exception:
                pass
        if owns_source and source_model is not None:
            try:
                source_model.close()
            except Exception:
                pass
        if owns_target and target_model is not None:
            try:
                target_model.close()
            except Exception:
                pass
        raise


__all__ = ["MigrationSession", "migrate"]
