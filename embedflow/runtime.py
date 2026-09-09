from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache import SQLiteVectorCache
from .config import EmbedFlowConfig, load_config
from .indexes import FaissIndex, NumpyIndex, PgVectorDocumentStore, PgVectorIndex, QdrantIndex
from .migration.state import DocumentStore
from .models import load_embedding_model
from .serving.engine import MigrationEngine


def load_documents(cfg: EmbedFlowConfig, index: Any | None = None) -> DocumentStore | PgVectorDocumentStore:
    """Load the configured document resolver.

    A pgvector table commonly stores both the legacy vector and document text.
    When no JSONL file is present, use a lazy resolver over the same read-only
    connection instead of requiring users to copy their corpus into a second
    store.  A JSONL path still takes precedence for deployments that keep text
    elsewhere.
    """
    if cfg.index.backend.lower() == "pgvector" and not Path(cfg.documents.path).expanduser().exists():
        pg_index = index if isinstance(index, PgVectorIndex) else PgVectorIndex.from_config(cfg)
        return PgVectorDocumentStore(
            pg_index,
            id_field=cfg.documents.id_field,
            text_field=cfg.index.text_column or cfg.documents.text_field,
            owns_index=index is None,
        )
    return DocumentStore(cfg.documents.path, cfg.documents.id_field, cfg.documents.text_field)


def load_index(cfg: EmbedFlowConfig, documents: DocumentStore | PgVectorDocumentStore):
    if cfg.index.backend.lower() == "pgvector":
        if isinstance(documents, PgVectorDocumentStore):
            return documents.index
        return PgVectorIndex.from_config(cfg, documents=documents.documents)
    metadata = documents.documents
    if cfg.index.backend.lower() == "faiss":
        try:
            return FaissIndex.load(cfg.index.path, ids_path=cfg.index.ids, metric=cfg.index.metric,
                                   documents=metadata, nprobe=cfg.index.nprobe)
        except (RuntimeError, ValueError) as exc:
            # A NumPy fallback index may be opened on a machine that happens
            # to have FAISS installed.  Try the fallback for any FAISS read
            # error, but preserve the original exception when the file is not
            # a valid NumPy artifact either.
            try:
                return NumpyIndex.load(cfg.index.path, metric=cfg.index.metric, documents=metadata)
            except Exception as fallback_exc:
                raise exc from fallback_exc
    return QdrantIndex.connect(cfg.index.path if not cfg.index.url else cfg.index.url,
                               cfg.index.collection, int(cfg.source.dimension or 0), documents=metadata,
                               metric=cfg.index.metric, api_key_env=cfg.index.api_key_env,
                               vector_name=cfg.index.vector_name)


def open_engine(config_path: str | Path, device: str | None = None, demo: bool = False,
                start_worker: bool = True, documents: DocumentStore | PgVectorDocumentStore | None = None,
                allow_empty_index: bool = False) -> MigrationEngine:
    """Load models, a source index, cache, and the shared migration engine.

    Serving and analysis require at least one source vector.  ``audit-index``
    can opt into ``allow_empty_index`` so it can report an empty table as an
    audit finding instead of failing during engine setup.
    """
    cfg = load_config(config_path)
    if cfg.index.metric.lower() == "cosine" and (cfg.source.normalization.lower() != "l2" or cfg.target.normalization.lower() != "l2"):
        raise ValueError("cosine index/reranking requires l2-normalized source and target vectors")
    documents_created = documents is None
    documents = load_documents(cfg) if documents is None else documents
    override_device = device
    source_device = override_device or cfg.source.device or cfg.target.device or "cpu"
    target_device = override_device or cfg.target.device or cfg.source.device or "cpu"
    source_model = None
    target_model = None
    source_index = None
    cache = None
    try:
        source_model = load_embedding_model(cfg.source, model_root=Path(config_path).parent / "models", device=source_device, demo=demo)
        if cfg.source.dimension and int(source_model.dimension) != int(cfg.source.dimension):
            raise ValueError(f"source model dimension {source_model.dimension} != configured {cfg.source.dimension}")
        source_index = load_index(cfg, documents)
        if source_index.size() <= 0 and not allow_empty_index:
            raise ValueError("legacy index is empty or its configured Qdrant collection is unavailable")
        if int(source_index.dimension) != int(source_model.dimension):
            raise ValueError(f"source model dimension {source_model.dimension} != existing index dimension {source_index.dimension}")
        index_ids = getattr(source_index, "ids", None)
        if index_ids is not None:
            missing_text = sorted(set(map(str, index_ids)) - set(documents.documents))
            if missing_text:
                raise ValueError(f"legacy index references {len(missing_text)} documents missing from the document store (e.g. {missing_text[:3]})")
        stored_fingerprint = source_index.metadata().get("model_fingerprint")
        if stored_fingerprint and stored_fingerprint != source_model.fingerprint:
            raise ValueError("source model fingerprint does not match the existing index contract")
        target_model = load_embedding_model(cfg.target, model_root=Path(config_path).parent / "models", device=target_device, demo=demo)
        if cfg.target.dimension and int(target_model.dimension) != int(cfg.target.dimension):
            raise ValueError(f"target model dimension {target_model.dimension} != configured {cfg.target.dimension}")
        cache = SQLiteVectorCache(cfg.cache.path, target_model.fingerprint, target_model.dimension)
        probe = {}
        probe_path = Path(cfg.state_path).with_name("probe_result.json")
        if probe_path.exists():
            try:
                loaded_probe = json.loads(probe_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid probe result {probe_path}") from exc
            if not isinstance(loaded_probe, dict):
                raise ValueError(f"probe result {probe_path} must contain a JSON object")
            probe = loaded_probe
        return MigrationEngine(cfg, source_model, target_model, source_index, cache, documents,
                               probe=probe, ann_status="UNKNOWN", start_worker=start_worker)
    except Exception:
        if cache is not None:
            cache.close()
        if target_model is not None:
            target_model.close()
        if source_model is not None:
            source_model.close()
        if source_index is not None:
            try:
                close_index = getattr(source_index, "close", None)
                if callable(close_index):
                    close_index()
            except Exception:
                pass
        if documents_created:
            try:
                close_documents = getattr(documents, "close", None)
                if callable(close_documents):
                    close_documents()
            except Exception:
                pass
        raise


def build_faiss_from_documents(cfg: EmbedFlowConfig, model: Any, documents: DocumentStore,
                               path: str | Path | None = None, nlist: int | None = None):
    vectors = model.encode_documents(list(documents.documents.values()), batch_size=32)
    target = path or cfg.index.path
    try:
        return FaissIndex.build(vectors, list(documents.documents), path=target,
                                metric=cfg.index.metric, nlist=nlist, ids_path=cfg.index.ids, documents=documents.documents,
                                metadata={"model_fingerprint": model.fingerprint, "model_id": model.model_id})
    except RuntimeError as exc:
        if "FAISS" not in str(exc): raise
        return NumpyIndex.build(vectors, list(documents.documents), path=target, metric=cfg.index.metric, ids_path=cfg.index.ids,
                                documents=documents.documents,
                                metadata={"model_fingerprint": model.fingerprint, "model_id": model.model_id})
