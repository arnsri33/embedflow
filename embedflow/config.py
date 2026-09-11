from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_UNSET = object()


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    """Validate integer-valued YAML fields without lossy coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or (minimum is not None and parsed < minimum):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be an integer{qualifier}")
    return parsed


@dataclass(init=False)
class ModelConfig:
    model: str
    revision: str | None = None
    dimension: int | None = None
    max_length: int = 512
    pooling: str = "mean_tokens"
    padding_side: str = "right"
    truncation_side: str = "right"
    query_instruction: str = ""
    document_instruction: str = ""
    normalization: str = "l2"
    dtype: str = "float32"
    local_path: str | None = None
    device: str | None = None

    def __init__(
        self,
        model: str,
        revision: str | None | object = _UNSET,
        dimension: int | None | object = _UNSET,
        max_length: int | object = _UNSET,
        pooling: str | object = _UNSET,
        padding_side: str | object = _UNSET,
        truncation_side: str | object = _UNSET,
        query_instruction: str | object = _UNSET,
        document_instruction: str | object = _UNSET,
        normalization: str | object = _UNSET,
        dtype: str | object = _UNSET,
        local_path: str | None | object = _UNSET,
        device: str | None | object = _UNSET,
    ) -> None:
        """Construct a model contract while remembering explicit overrides.

        The research contracts have non-generic defaults (for example,
        Qwen3-8B uses length 8192 while the product's generic default is
        512).  A normal dataclass cannot distinguish an omitted value from a
        caller explicitly requesting that generic default.  The sentinel
        arguments let ``hydrate_research_contract`` fill only omitted fields,
        preserving deliberate prompt/pooling/max-length changes for matching
        and cache safety.
        """
        defaults = {
            "revision": None,
            "dimension": None,
            "max_length": 512,
            "pooling": "mean_tokens",
            "padding_side": "right",
            "truncation_side": "right",
            "query_instruction": "",
            "document_instruction": "",
            "normalization": "l2",
            "dtype": "float32",
            "local_path": None,
            "device": None,
        }
        values = {
            "revision": revision,
            "dimension": dimension,
            "max_length": max_length,
            "pooling": pooling,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "query_instruction": query_instruction,
            "document_instruction": document_instruction,
            "normalization": normalization,
            "dtype": dtype,
            "local_path": local_path,
            "device": device,
        }
        self.model = str(model)
        self._explicit_fields = {key for key, value in values.items() if value is not _UNSET}
        for key, value in values.items():
            setattr(self, key, defaults[key] if value is _UNSET else value)

    def contract(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        # Paths and execution devices are deployment details, not embedding
        # semantics.  Excluding them keeps a cache/index portable across
        # machines while retaining model revision, prompts, pooling, padding,
        # truncation, normalization, dtype, and dimension in the contract.
        semantic = {key: value for key, value in self.contract().items() if key not in {"local_path", "device"}}
        payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class IndexConfig:
    backend: str = "faiss"
    path: str = "./legacy.index"
    metric: str = "cosine"
    nprobe: int = 64
    collection: str = "embedflow"
    url: str | None = None
    ids: str | None = None
    vector_name: str | None = None
    api_key_env: str | None = "QDRANT_API_KEY"
    # pgvector settings. ``url`` is accepted as an explicit DSN for callers
    # that already manage secrets; YAML deployments should use ``dsn_env``.
    dsn_env: str | None = "EMBEDFLOW_PGVECTOR_DSN"
    schema: str = "public"
    table: str = "documents"
    id_column: str = "id"
    vector_column: str = "embedding"
    text_column: str | None = "content"
    hnsw_ef_search: int | None = None
    ivfflat_probes: int | None = None
    # Pinecone settings. Host is preferred for data-plane operations; an
    # index_name is accepted for controlled/test environments and is resolved
    # through the Pinecone control plane.
    host: str | None = None
    index_name: str | None = None
    namespace: str = ""
    text_metadata_field: str | None = None


@dataclass
class DocumentsConfig:
    path: str = "./documents.jsonl"
    id_field: str = "id"
    text_field: str = "text"


@dataclass
class MigrationConfig:
    candidate_depth: int | str = 50
    kmax_probe: int = 500
    probe_queries: int = 100
    max_sync_misses: int = 4
    background_batch_size: int = 32
    max_retries: int = 3
    worker_count: int = 1


@dataclass
class CacheConfig:
    path: str = "./embedflow_cache"


@dataclass
class EconomicsConfig:
    gpu_price_per_hour: float | None = None
    target_docs_per_second: float | None = None


@dataclass
class ProbeConfig:
    """Settings for the leakage-safe finite-pool compatibility probe."""

    queries: str | None = None
    k_values: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    kmax: int = 500
    epsilon: float = 0.01
    seed: int = 42
    limit: int | None = None


@dataclass
class TelemetryConfig:
    latency_log: str = "./logs/latency.jsonl"


@dataclass
class EmbedFlowConfig:
    source: ModelConfig
    target: ModelConfig
    index: IndexConfig = field(default_factory=IndexConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    state_path: str = "./embedflow_state.json"
    dashboard_title: str = "EmbedFlow — Progressive Embedding Migration"

    def resolve_paths(self, base: Path) -> EmbedFlowConfig:
        """Resolve relative paths against the configuration file directory."""
        for obj, attr in ((self.index, "path"), (self.documents, "path"),
                          (self.cache, "path"), (self, "state_path"),
                          (self.telemetry, "latency_log")):
            raw_value = str(getattr(obj, attr))
            # A remote endpoint/DSN is a connection endpoint, not a filesystem path.
            # Leave it untouched so ``index: {backend: qdrant, path: https://…}``
            # works even when no separate ``url`` field is supplied.
            if "://" in raw_value:
                continue
            value = Path(raw_value)
            if not value.is_absolute():
                setattr(obj, attr, str((base / value).resolve()))
        if self.source.local_path:
            p = Path(self.source.local_path)
            self.source.local_path = str((base / p).resolve()) if not p.is_absolute() else str(p)
        if self.target.local_path:
            p = Path(self.target.local_path)
            self.target.local_path = str((base / p).resolve()) if not p.is_absolute() else str(p)
        if self.probe.queries:
            p = Path(self.probe.queries)
            self.probe.queries = str((base / p).resolve()) if not p.is_absolute() else str(p)
        if self.index.ids:
            p = Path(self.index.ids)
            self.index.ids = str((base / p).resolve()) if not p.is_absolute() else str(p)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for label, model in (("source", self.source), ("target", self.target)):
            if not model.model.strip():
                raise ValueError(f"{label}.model is required")
            if model.dimension is not None:
                model.dimension = _integer(model.dimension, f"{label}.dimension", minimum=1)
            model.max_length = _integer(model.max_length, f"{label}.max_length", minimum=1)
            if not isinstance(model.normalization, str) or model.normalization.lower() not in {"l2", "none", "identity"}:
                raise ValueError(f"{label}.normalization must be l2, none, or identity")
        if self.source.fingerprint == self.target.fingerprint:
            raise ValueError("source and target embedding contracts must differ for migration")
        backend = self.index.backend.lower() if isinstance(self.index.backend, str) else ""
        if backend not in {"faiss", "qdrant", "pgvector", "pinecone"}:
            raise ValueError("index.backend must be faiss, qdrant, pgvector, or pinecone")
        allowed_metrics = {"cosine", "dot", "inner_product"}
        if backend == "pgvector":
            allowed_metrics |= {"l2", "euclidean"}
        if backend == "pinecone":
            allowed_metrics |= {"dotproduct", "l2", "euclidean"}
        if not isinstance(self.index.metric, str) or self.index.metric.lower() not in allowed_metrics:
            names = "cosine, dot, inner_product" + (", dotproduct, l2, or euclidean" if backend == "pinecone" else ", l2, or euclidean" if backend == "pgvector" else "")
            raise ValueError(f"index.metric must be {names}")
        if backend == "pgvector":
            for label, value in (("index.schema", self.index.schema), ("index.table", self.index.table),
                                 ("index.id_column", self.index.id_column), ("index.vector_column", self.index.vector_column)):
                if not isinstance(value, str) or not value.strip() or "\x00" in value:
                    raise ValueError(f"{label} must be a non-empty string without NUL bytes")
            if self.index.text_column is not None and (not isinstance(self.index.text_column, str) or
                                                        not self.index.text_column.strip() or "\x00" in self.index.text_column):
                raise ValueError("index.text_column must be null or a non-empty string without NUL bytes")
            if self.index.dsn_env is not None and (not isinstance(self.index.dsn_env, str) or not self.index.dsn_env.strip()):
                raise ValueError("index.dsn_env must be a non-empty environment-variable name")
            self.index.hnsw_ef_search = None if self.index.hnsw_ef_search is None else _integer(self.index.hnsw_ef_search, "index.hnsw_ef_search", minimum=1)
            self.index.ivfflat_probes = None if self.index.ivfflat_probes is None else _integer(self.index.ivfflat_probes, "index.ivfflat_probes", minimum=1)
        if backend == "pinecone":
            if not any(isinstance(value, str) and value.strip() for value in (self.index.host, self.index.index_name, self.index.url)):
                # ``path`` is accepted as a host by the adapter for old
                # programmatic configs, but the explicit field is preferred.
                path = str(self.index.path or "")
                if ".pinecone.io" not in path and not path.startswith("https://"):
                    raise ValueError("index.backend=pinecone requires a host or index_name")
            for label, value in (("index.host", self.index.host), ("index.index_name", self.index.index_name),
                                 ("index.api_key_env", self.index.api_key_env), ("index.text_metadata_field", self.index.text_metadata_field)):
                if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                    raise ValueError(f"{label} must be null or a non-empty string without NUL bytes")
            if not isinstance(self.index.namespace, str) or "\x00" in self.index.namespace:
                raise ValueError("index.namespace must be a string without NUL bytes")
        candidate_depth = self.migration.candidate_depth
        if isinstance(candidate_depth, str) and candidate_depth.strip().lower() == "auto":
            candidate_depth = "auto"
            self.migration.candidate_depth = "auto"
        if candidate_depth != "auto":
            candidate_depth = _integer(candidate_depth, "migration.candidate_depth", minimum=1)
            self.migration.candidate_depth = candidate_depth
        self.migration.kmax_probe = _integer(self.migration.kmax_probe, "migration.kmax_probe", minimum=1)
        self.migration.probe_queries = _integer(self.migration.probe_queries, "migration.probe_queries", minimum=1)
        if candidate_depth != "auto" and self.migration.kmax_probe < int(candidate_depth):
            raise ValueError("migration.kmax_probe must be >= candidate_depth")
        if backend == "pinecone":
            if candidate_depth != "auto" and int(candidate_depth) > 10_000:
                raise ValueError("Pinecone migration.candidate_depth must be <= 10000")
            if self.migration.kmax_probe > 10_000:
                raise ValueError("Pinecone migration.kmax_probe must be <= 10000")
        self.migration.max_sync_misses = _integer(self.migration.max_sync_misses, "migration.max_sync_misses", minimum=0)
        self.migration.background_batch_size = _integer(self.migration.background_batch_size, "migration.background_batch_size", minimum=1)
        self.migration.max_retries = _integer(self.migration.max_retries, "migration.max_retries", minimum=1)
        self.migration.worker_count = _integer(self.migration.worker_count, "migration.worker_count", minimum=1)
        self.index.nprobe = _integer(self.index.nprobe, "index.nprobe", minimum=1)
        self.probe.kmax = _integer(self.probe.kmax, "probe.kmax", minimum=10)
        if self.probe.kmax < 10:
            raise ValueError("probe.kmax must be at least 10")
        if backend == "pinecone" and self.probe.kmax > 10_000:
            raise ValueError("Pinecone probe.kmax must be <= 10000")
        if not self.probe.k_values:
            raise ValueError("probe.k_values must contain positive integers")
        try:
            self.probe.k_values = [_integer(k, "probe.k_values item", minimum=1) for k in self.probe.k_values]
        except ValueError as exc:
            raise ValueError("probe.k_values must contain positive integers") from exc
        if len(set(self.probe.k_values)) != len(self.probe.k_values):
            raise ValueError("probe.k_values must not contain duplicates")
        if any(k > int(self.probe.kmax) for k in self.probe.k_values):
            raise ValueError("probe.k_values cannot exceed probe.kmax")
        self.probe.epsilon = float(self.probe.epsilon)
        if not math.isfinite(self.probe.epsilon) or self.probe.epsilon < 0:
            raise ValueError("probe.epsilon must be finite and non-negative")


def _model(raw: dict[str, Any], fallback: str) -> ModelConfig:
    raw = dict(raw or {})
    model = raw.pop("model", raw.pop("model_id", fallback))
    # Accept the research project's names and hydrate their exact contracts.
    return ModelConfig(model=str(model), **raw)


def from_dict(raw: dict[str, Any], *, validate: bool = True) -> EmbedFlowConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a YAML object")
    raw = dict(raw or {})
    source = hydrate_research_contract(_model(raw.get("source", {}), ""))
    target = hydrate_research_contract(_model(raw.get("target", {}), ""))
    migration_raw = dict(raw.get("migration", {}))
    if isinstance(migration_raw.get("candidate_depth"), str) and migration_raw["candidate_depth"].strip().lower() == "auto":
        migration_raw["candidate_depth"] = "auto"
    index_raw = dict(raw.get("index", {}))
    if str(index_raw.get("backend", "faiss")).lower() == "pinecone" and "api_key_env" not in index_raw:
        index_raw["api_key_env"] = "PINECONE_API_KEY"
    cfg = EmbedFlowConfig(
        source=source,
        target=target,
        index=IndexConfig(**index_raw),
        documents=DocumentsConfig(**dict(raw.get("documents", {}))),
        migration=MigrationConfig(**migration_raw),
        cache=CacheConfig(**dict(raw.get("cache", {}))),
        economics=EconomicsConfig(**dict(raw.get("economics", {}))),
        probe=ProbeConfig(**dict(raw.get("probe", {}))),
        telemetry=TelemetryConfig(**dict(raw.get("telemetry", {}))),
        state_path=str(raw.get("state_path", "./embedflow_state.json")),
        dashboard_title=str(raw.get("dashboard_title", EmbedFlowConfig.__dataclass_fields__["dashboard_title"].default)),
    )
    if validate:
        cfg.validate()
    return cfg


def hydrate_research_contract(model: ModelConfig, project_root: Path | None = None) -> ModelConfig:
    """Use the frozen research contract for one of its registered model keys."""
    # Keep the public package self-contained.  The research checkout also has
    # these contracts in ``config/nq_5090_1m.yaml``; the built-ins ensure that
    # installing EmbedFlow from a clean Git clone still preserves the exact
    # pooling, prompt, padding, and revision semantics for known checkpoints.
    builtin = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "dimension": 384, "max_length": 512, "pooling": "mean_tokens",
            "padding_side": "right", "truncation_side": "right",
            "query_instruction": "", "document_instruction": "", "normalization": "l2",
        },
        "Qwen/Qwen3-Embedding-0.6B": {
            "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "dimension": 1024, "max_length": 512, "pooling": "last_non_padding_token",
            "padding_side": "left", "truncation_side": "right",
            "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}",
            "document_instruction": "", "normalization": "l2",
        },
        "Qwen/Qwen3-Embedding-4B": {
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
            "dimension": 2560, "max_length": 8192, "pooling": "last_non_padding_token",
            "padding_side": "left", "truncation_side": "right",
            "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}",
            "document_instruction": "", "normalization": "l2",
        },
        "Qwen/Qwen3-Embedding-8B": {
            "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
            "dimension": 4096, "max_length": 8192, "pooling": "last_non_padding_token",
            "padding_side": "left", "truncation_side": "right",
            "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}",
            "document_instruction": "", "normalization": "l2",
        },
    }
    if model.model in builtin:
        for field_name, value in builtin[model.model].items():
            # An explicit user value wins, while dataclass defaults are filled
            # from the frozen contract.  ``revision``/dimension are especially
            # important for cache and index safety.
            current = getattr(model, field_name)
            default = ModelConfig.__dataclass_fields__[field_name].default
            explicit = field_name in getattr(model, "_explicit_fields", set())
            if not explicit and (current is None or current == default or field_name in {"query_instruction", "document_instruction"} and not current):
                setattr(model, field_name, value)
    try:
        import yaml
        root = project_root or Path(__file__).resolve().parents[1]
        config_path = root / "config" / "nq_5090_1m.yaml"
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
            for key, item in (raw.get("models") or {}).items():
                if item.get("model_id") == model.model or key == model.model:
                    known = dict(item)
                    known.pop("min_vram_gib", None)
                    known.pop("dtype_preference", None)
                    known["model"] = known.pop("model_id", model.model)
                    # The research YAML calls Qwen pooling `last_token`; the
                    # product encoder accepts that as last_non_padding_token.
                    if known.get("pooling") == "last_token":
                        known["pooling"] = "last_non_padding_token"
                    for field_name in ModelConfig.__dataclass_fields__:
                        if field_name in known and field_name not in getattr(model, "_explicit_fields", set()):
                            setattr(model, field_name, known[field_name])
                    return model
    except Exception:
        # Generic Hugging Face models remain usable when the research config is
        # not present. The caller can still inspect the resulting fingerprint.
        pass
    return model


def load_config(path: str | Path) -> EmbedFlowConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    import yaml
    try:
        parsed = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {path}: {exc}") from exc
    # Apply deployment overrides before validation so a secrets-free YAML file
    # can supply connection fields such as the Pinecone host through the
    # documented environment mechanism.
    cfg = from_dict(parsed or {}, validate=False)
    _apply_environment_overrides(cfg)
    cfg.validate()
    return cfg.resolve_paths(path.parent.resolve())


def _apply_environment_overrides(cfg: EmbedFlowConfig) -> None:
    """Apply deliberately small, documented deployment overrides.

    YAML remains the source of truth for experiments. These variables are
    useful for containerized serving and secrets-free deployment manifests;
    unset variables leave the file untouched.
    """
    paths: dict[str, tuple[Any, str]] = {
        "EMBEDFLOW_SOURCE_MODEL": (cfg.source, "model"),
        "EMBEDFLOW_TARGET_MODEL": (cfg.target, "model"),
        "EMBEDFLOW_SOURCE_DEVICE": (cfg.source, "device"),
        "EMBEDFLOW_TARGET_DEVICE": (cfg.target, "device"),
        "EMBEDFLOW_INDEX_BACKEND": (cfg.index, "backend"),
        "EMBEDFLOW_INDEX_PATH": (cfg.index, "path"),
        "EMBEDFLOW_INDEX_URL": (cfg.index, "url"),
        "EMBEDFLOW_INDEX_COLLECTION": (cfg.index, "collection"),
        "EMBEDFLOW_INDEX_VECTOR_NAME": (cfg.index, "vector_name"),
        "EMBEDFLOW_QDRANT_API_KEY_ENV": (cfg.index, "api_key_env"),
        "EMBEDFLOW_PGVECTOR_DSN_ENV": (cfg.index, "dsn_env"),
        "EMBEDFLOW_PGVECTOR_SCHEMA": (cfg.index, "schema"),
        "EMBEDFLOW_PGVECTOR_TABLE": (cfg.index, "table"),
        "EMBEDFLOW_PGVECTOR_ID_COLUMN": (cfg.index, "id_column"),
        "EMBEDFLOW_PGVECTOR_VECTOR_COLUMN": (cfg.index, "vector_column"),
        "EMBEDFLOW_PGVECTOR_TEXT_COLUMN": (cfg.index, "text_column"),
        "EMBEDFLOW_PINECONE_HOST": (cfg.index, "host"),
        "EMBEDFLOW_PINECONE_INDEX_NAME": (cfg.index, "index_name"),
        "EMBEDFLOW_PINECONE_NAMESPACE": (cfg.index, "namespace"),
        "EMBEDFLOW_PINECONE_TEXT_METADATA_FIELD": (cfg.index, "text_metadata_field"),
        "EMBEDFLOW_PINECONE_API_KEY_ENV": (cfg.index, "api_key_env"),
        "EMBEDFLOW_DOCUMENTS_PATH": (cfg.documents, "path"),
        "EMBEDFLOW_CACHE_PATH": (cfg.cache, "path"),
        "EMBEDFLOW_STATE_PATH": (cfg, "state_path"),
        "EMBEDFLOW_LATENCY_LOG": (cfg.telemetry, "latency_log"),
    }
    for variable, (target, field_name) in paths.items():
        value = os.environ.get(variable)
        if value is not None and value.strip():
            setattr(target, field_name, value.strip())
    candidate = os.environ.get("EMBEDFLOW_CANDIDATE_DEPTH")
    if candidate:
        if candidate.strip().lower() == "auto":
            cfg.migration.candidate_depth = "auto"
        else:
            try:
                parsed = int(candidate)
                if float(candidate) != parsed:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("EMBEDFLOW_CANDIDATE_DEPTH must be auto or an integer") from exc
            cfg.migration.candidate_depth = parsed
    for variable, target, field_name in (
        ("EMBEDFLOW_MAX_SYNC_MISSES", cfg.migration, "max_sync_misses"),
        ("EMBEDFLOW_BACKGROUND_BATCH_SIZE", cfg.migration, "background_batch_size"),
        ("EMBEDFLOW_PROBE_KMAX", cfg.probe, "kmax"),
        ("EMBEDFLOW_INDEX_NPROBE", cfg.index, "nprobe"),
        ("EMBEDFLOW_PGVECTOR_HNSW_EF_SEARCH", cfg.index, "hnsw_ef_search"),
        ("EMBEDFLOW_PGVECTOR_IVFFLAT_PROBES", cfg.index, "ivfflat_probes"),
    ):
        value = os.environ.get(variable)
        if value is not None and value.strip():
            try:
                parsed = int(value)
                if float(value) != parsed:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{variable} must be an integer") from exc
            setattr(target, field_name, parsed)


def save_config(cfg: EmbedFlowConfig, path: str | Path) -> None:
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
