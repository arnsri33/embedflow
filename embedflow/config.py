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
    # Milvus settings. ``uri`` is the endpoint accepted by MilvusClient;
    # ``path`` remains a backwards-compatible endpoint escape hatch.
    uri: str | None = None
    token_env: str | None = "EMBEDFLOW_MILVUS_TOKEN"
    database: str = "default"
    id_field: str = "id"
    vector_field: str | None = "embedding"
    text_field: str | None = "content"
    partition_names: list[str] = field(default_factory=list)
    search_params: dict[str, Any] = field(default_factory=dict)
    auto_load: bool = False
    # Weaviate v4 settings. ``uri`` may be a local HTTP endpoint or a cloud
    # cluster URL; explicit host/port fields are useful for local/custom
    # deployments and keep gRPC connectivity visible in configuration.
    http_host: str = "localhost"
    http_port: int = 8080
    grpc_host: str | None = None
    grpc_port: int = 50051
    secure: bool = False
    grpc_secure: bool | None = None
    tenant: str | None = None
    text_property: str | None = None


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
class PlannerConfig:
    """Bounded, reproducible settings for the advisory migration planner.

    Planner settings are deliberately separate from ``MigrationConfig``.  A
    plan is analysis state and must not silently change serving defaults.
    Values supplied by the CLI override these defaults for one invocation.
    """

    max_probes: int = 250
    seed: int = 42
    k_grid: list[int] = field(default_factory=lambda: [20, 50, 100, 200, 500])
    max_candidates: int | None = None
    max_target_encodes: int | None = None
    max_sync_misses: int | None = None
    background_batch_size: int | None = None
    gpu_hourly_cost: float | None = None
    target_docs_per_second: float | None = None
    queries_per_second: float | None = None
    daily_queries: float | None = None
    cache_hit_rate: float | None = None
    latency_budget_ms: float | None = None
    access_trace: str | None = None
    corpus_name: str | None = None
    corpus_fingerprint: str | None = None


@dataclass
class RuntimeConfig:
    """Serving mode selection.

    ``migration`` is the historical source-candidate/target-reranking path.
    ``shadow`` keeps the source result authoritative and runs the target path
    off the request's critical path.
    """

    mode: str = "migration"


@dataclass
class ShadowTelemetryConfig:
    """Privacy-conscious persistence settings for shadow observations."""

    enabled: bool = True
    path: str | None = None
    retain_query_records: bool = False
    retain_query_text: bool = False
    max_records: int = 10_000
    retention_days: int | None = None
    report_k: int = 10
    min_target_coverage_for_ranking: float = 1.0


@dataclass
class ShadowConfig:
    """Bounded, failure-isolated shadow execution settings."""

    # A runtime mode of ``shadow`` is itself an explicit opt-in.  Keeping this
    # default true means a minimal ``runtime: {mode: shadow}`` configuration
    # behaves as users expect; ``shadow.enabled: false`` remains an explicit
    # kill switch.
    enabled: bool = True
    sample_rate: float = 0.10
    sample_seed: int = 42
    candidate_k: int | None = None
    materialize: bool = True
    max_inflight: int = 32
    queue_capacity: int = 1000
    timeout_ms: int = 10_000
    shutdown_grace_ms: int = 1_000
    telemetry: ShadowTelemetryConfig = field(default_factory=ShadowTelemetryConfig)


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
    # Appended after the historical fields so positional construction of
    # existing configurations remains source-compatible.
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

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
        if self.planner.access_trace:
            p = Path(self.planner.access_trace)
            self.planner.access_trace = str((base / p).resolve()) if not p.is_absolute() else str(p)
        if self.shadow.telemetry.path:
            p = Path(self.shadow.telemetry.path)
            self.shadow.telemetry.path = str((base / p).resolve()) if not p.is_absolute() else str(p)
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
        if backend not in {"faiss", "qdrant", "pgvector", "pinecone", "milvus", "weaviate"}:
            raise ValueError("index.backend must be faiss, qdrant, pgvector, pinecone, milvus, or weaviate")
        if not isinstance(self.runtime.mode, str) or self.runtime.mode.strip().lower() not in {"migration", "normal", "source", "shadow"}:
            raise ValueError("runtime.mode must be migration, normal, source, or shadow")
        self.runtime.mode = self.runtime.mode.strip().lower()
        if not isinstance(self.shadow.enabled, bool):
            raise ValueError("shadow.enabled must be a boolean")
        for label, value in (("shadow.sample_rate", self.shadow.sample_rate),
                             ("shadow.timeout_ms", self.shadow.timeout_ms),
                             ("shadow.max_inflight", self.shadow.max_inflight),
                             ("shadow.queue_capacity", self.shadow.queue_capacity),
                             ("shadow.shutdown_grace_ms", self.shadow.shutdown_grace_ms)):
            if label == "shadow.sample_rate":
                if isinstance(value, bool):
                    raise ValueError("shadow.sample_rate must be between 0 and 1")
                try:
                    parsed = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("shadow.sample_rate must be between 0 and 1") from exc
                if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                    raise ValueError("shadow.sample_rate must be between 0 and 1")
                self.shadow.sample_rate = parsed
            else:
                parsed = _integer(value, label, minimum=1)
                setattr(self.shadow, label.split(".")[-1], parsed)
        limits = {"max_inflight": 1024, "queue_capacity": 1_000_000,
                  "timeout_ms": 3_600_000, "shutdown_grace_ms": 300_000}
        for field_name, limit in limits.items():
            if int(getattr(self.shadow, field_name)) > limit:
                raise ValueError(f"shadow.{field_name} exceeds the safe limit of {limit}")
        self.shadow.sample_seed = _integer(self.shadow.sample_seed, "shadow.sample_seed")
        if self.shadow.candidate_k is not None:
            self.shadow.candidate_k = _integer(self.shadow.candidate_k, "shadow.candidate_k", minimum=1)
        if self.shadow.candidate_k is not None and self.shadow.candidate_k < 1:
            raise ValueError("shadow.candidate_k must be positive")
        if self.shadow.candidate_k is not None:
            shadow_k = int(self.shadow.candidate_k)
            backend_limits = {"pinecone": 10_000, "milvus": 16_384, "weaviate": 10_000}
            limit = backend_limits.get(backend)
            if limit is not None and shadow_k > limit:
                raise ValueError(f"{backend} shadow.candidate_k must be <= {limit}")
        if not isinstance(self.shadow.materialize, bool):
            raise ValueError("shadow.materialize must be a boolean")
        telemetry = self.shadow.telemetry
        if not isinstance(telemetry.enabled, bool):
            raise ValueError("shadow.telemetry.enabled must be a boolean")
        for label, value in (("shadow.telemetry.retain_query_records", telemetry.retain_query_records),
                             ("shadow.telemetry.retain_query_text", telemetry.retain_query_text)):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        telemetry.max_records = _integer(telemetry.max_records, "shadow.telemetry.max_records", minimum=1)
        if telemetry.retention_days is not None:
            telemetry.retention_days = _integer(telemetry.retention_days, "shadow.telemetry.retention_days", minimum=1)
        telemetry.report_k = _integer(telemetry.report_k, "shadow.telemetry.report_k", minimum=1)
        try:
            coverage = float(telemetry.min_target_coverage_for_ranking)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("shadow.telemetry.min_target_coverage_for_ranking must be between 0 and 1") from exc
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("shadow.telemetry.min_target_coverage_for_ranking must be between 0 and 1")
        telemetry.min_target_coverage_for_ranking = coverage
        if telemetry.path is not None and (not isinstance(telemetry.path, str) or not telemetry.path.strip() or "\x00" in telemetry.path):
            raise ValueError("shadow.telemetry.path must be null or a non-empty path without NUL bytes")
        allowed_metrics = {"cosine", "dot", "inner_product"}
        if backend in {"pgvector", "milvus"}:
            allowed_metrics |= {"l2", "euclidean"}
        if backend in {"pinecone", "milvus"}:
            allowed_metrics |= {"dotproduct", "l2", "euclidean"}
        if backend == "weaviate":
            allowed_metrics |= {"dotproduct", "l2", "euclidean"}
        if not isinstance(self.index.metric, str) or self.index.metric.lower() not in allowed_metrics:
            names = "cosine, dot, inner_product" + (", dotproduct, l2, or euclidean" if backend in {"pinecone", "milvus", "weaviate"} else ", l2, or euclidean" if backend == "pgvector" else "")
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
        if backend == "milvus":
            # ``uri`` is the documented field.  ``path`` remains a
            # backwards-compatible endpoint escape hatch for programmatic
            # callers, so permit it when it contains a URI rather than
            # rejecting a valid legacy configuration at the first check.
            uri_configured = isinstance(self.index.uri, str) and bool(self.index.uri.strip())
            path_uri = isinstance(self.index.path, str) and "://" in self.index.path
            for label, value in (("index.collection", self.index.collection),
                                 ("index.database", self.index.database), ("index.id_field", self.index.id_field)):
                if not isinstance(value, str) or not value.strip() or "\x00" in value:
                    raise ValueError(f"{label} must be a non-empty string without NUL bytes")
            for label, value in (("index.vector_field", self.index.vector_field), ("index.text_field", self.index.text_field),
                                 ("index.token_env", self.index.token_env)):
                if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                    raise ValueError(f"{label} must be null or a non-empty string without NUL bytes")
            if not uri_configured and not path_uri:
                raise ValueError("index.backend=milvus requires index.uri (for example http://localhost:19530)")
            if not isinstance(self.index.partition_names, list) or any(not isinstance(item, str) or not item.strip() for item in self.index.partition_names):
                raise ValueError("index.partition_names must be a list of non-empty strings")
            if len(set(self.index.partition_names)) != len(self.index.partition_names):
                raise ValueError("index.partition_names must not contain duplicates")
            if not isinstance(self.index.search_params, Mapping):
                raise ValueError("index.search_params must be a mapping")
            self.index.search_params = dict(self.index.search_params)
            # Keep the public configuration surface deliberately small.  The
            # adapter accepts the common HNSW/IVF/range controls in either a
            # flat mapping or Milvus' native {metric_type, params} shape;
            # reject typos before any client is initialized.
            search_keys = {"ef", "nprobe", "radius", "range_filter"}
            if "params" in self.index.search_params or "metric_type" in self.index.search_params:
                unknown = set(self.index.search_params) - {"metric_type", "params"}
                if unknown:
                    raise ValueError(f"index.search_params has unsupported keys: {sorted(unknown)!r}")
                nested = self.index.search_params.get("params", {})
                if not isinstance(nested, Mapping):
                    raise ValueError("index.search_params.params must be a mapping")
                unknown_nested = set(nested) - search_keys
                if unknown_nested:
                    raise ValueError(f"index.search_params.params has unsupported keys: {sorted(unknown_nested)!r}")
            else:
                unknown = set(self.index.search_params) - search_keys
                if unknown:
                    raise ValueError(f"index.search_params has unsupported keys: {sorted(unknown)!r}")
            params_to_check = self.index.search_params.get("params", self.index.search_params)
            if isinstance(params_to_check, Mapping):
                for name in ("ef", "nprobe"):
                    if name in params_to_check:
                        _integer(params_to_check[name], f"index.search_params.{name}", minimum=1)
                for name in ("radius", "range_filter"):
                    if name in params_to_check:
                        raw = params_to_check[name]
                        if isinstance(raw, bool):
                            raise ValueError(f"index.search_params.{name} must be a finite number")
                        try:
                            if not math.isfinite(float(raw)):
                                raise ValueError
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(f"index.search_params.{name} must be a finite number") from exc
            if not isinstance(self.index.auto_load, bool):
                raise ValueError("index.auto_load must be a boolean")
        if backend == "weaviate":
            for label, value in (("index.collection", self.index.collection),
                                 ("index.vector_name", self.index.vector_name),
                                 ("index.text_property", self.index.text_property),
                                 ("index.api_key_env", self.index.api_key_env),
                                 ("index.tenant", self.index.tenant),
                                 ("index.http_host", self.index.http_host),
                                 ("index.grpc_host", self.index.grpc_host),
                                 ("index.uri", self.index.uri)):
                if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                    raise ValueError(f"{label} must be null or a non-empty string without NUL bytes")
            for label, value in (("index.http_port", self.index.http_port), ("index.grpc_port", self.index.grpc_port)):
                setattr(self.index, label.split(".")[-1], _integer(value, label, minimum=1))
            if self.index.http_port > 65535 or self.index.grpc_port > 65535:
                raise ValueError("Weaviate ports must be between 1 and 65535")
            if not isinstance(self.index.secure, bool):
                raise ValueError("index.secure must be a boolean")
            if self.index.grpc_secure is not None and not isinstance(self.index.grpc_secure, bool):
                raise ValueError("index.grpc_secure must be null or a boolean")
            if not isinstance(self.index.namespace, str):
                # ``namespace`` is not a Weaviate setting, but rejecting an
                # accidental non-string value here gives a clearer config
                # error than allowing it to leak into a client call.
                raise ValueError("index.namespace must be a string")
            if not any(isinstance(value, str) and value.strip() for value in (self.index.uri, self.index.http_host, self.index.path)):
                raise ValueError("index.backend=weaviate requires index.uri or index.http_host")
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
        if backend == "milvus":
            if candidate_depth != "auto" and int(candidate_depth) > 16_384:
                raise ValueError("Milvus migration.candidate_depth must be <= 16384")
            if self.migration.kmax_probe > 16_384:
                raise ValueError("Milvus migration.kmax_probe must be <= 16384")
        if backend == "weaviate":
            if candidate_depth != "auto" and int(candidate_depth) > 10_000:
                raise ValueError("Weaviate migration.candidate_depth must be <= 10000")
            if self.migration.kmax_probe > 10_000:
                raise ValueError("Weaviate migration.kmax_probe must be <= 10000")
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
        if backend == "weaviate" and self.probe.kmax > 10_000:
            raise ValueError("Weaviate probe.kmax must be <= 10000")
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
        planner = self.planner
        planner.max_probes = _integer(planner.max_probes, "planner.max_probes", minimum=1)
        planner.seed = _integer(planner.seed, "planner.seed")
        if not isinstance(planner.k_grid, list) or not planner.k_grid:
            raise ValueError("planner.k_grid must contain at least one positive integer")
        planner.k_grid = [_integer(k, "planner.k_grid item", minimum=1) for k in planner.k_grid]
        if len(set(planner.k_grid)) != len(planner.k_grid):
            raise ValueError("planner.k_grid must not contain duplicates")
        planner.k_grid = sorted(planner.k_grid)
        for label, value in (("planner.max_candidates", planner.max_candidates),
                             ("planner.max_target_encodes", planner.max_target_encodes),
                             ("planner.max_sync_misses", planner.max_sync_misses),
                             ("planner.background_batch_size", planner.background_batch_size)):
            if value is not None:
                setattr(planner, label.split(".")[-1], _integer(value, label, minimum=0 if label.endswith("max_sync_misses") else 1))
        for label, value in (("planner.gpu_hourly_cost", planner.gpu_hourly_cost),
                             ("planner.target_docs_per_second", planner.target_docs_per_second),
                             ("planner.queries_per_second", planner.queries_per_second),
                             ("planner.daily_queries", planner.daily_queries),
                             ("planner.cache_hit_rate", planner.cache_hit_rate),
                             ("planner.latency_budget_ms", planner.latency_budget_ms)):
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{label} must be a finite non-negative number")
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{label} must be a finite non-negative number") from exc
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(f"{label} must be a finite non-negative number")
            if label.endswith("cache_hit_rate") and parsed > 1:
                raise ValueError("planner.cache_hit_rate must be between 0 and 1")
            setattr(planner, label.split(".")[-1], parsed)
        for label, value in (("planner.access_trace", planner.access_trace),
                             ("planner.corpus_name", planner.corpus_name),
                             ("planner.corpus_fingerprint", planner.corpus_fingerprint)):
            if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                raise ValueError(f"{label} must be null or a non-empty string without NUL bytes")


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
    runtime_raw = dict(raw.get("runtime", {}))
    shadow_raw = dict(raw.get("shadow", {}))
    shadow_telemetry_raw = shadow_raw.get("telemetry", {})
    if shadow_telemetry_raw is None:
        shadow_telemetry_raw = {}
    if not isinstance(shadow_telemetry_raw, Mapping):
        raise ValueError("shadow.telemetry must be a YAML object")
    shadow_raw["telemetry"] = ShadowTelemetryConfig(**dict(shadow_telemetry_raw))
    if isinstance(migration_raw.get("candidate_depth"), str) and migration_raw["candidate_depth"].strip().lower() == "auto":
        migration_raw["candidate_depth"] = "auto"
    index_raw = dict(raw.get("index", {}))
    backend_name = str(index_raw.get("backend", "faiss")).lower()
    if backend_name == "pinecone" and "api_key_env" not in index_raw:
        index_raw["api_key_env"] = "PINECONE_API_KEY"
    if backend_name == "weaviate" and "api_key_env" not in index_raw:
        index_raw["api_key_env"] = "WEAVIATE_API_KEY"
    if backend_name == "milvus":
        if "token_env" not in index_raw:
            index_raw["token_env"] = "EMBEDFLOW_MILVUS_TOKEN"
        # Unlike the legacy FAISS/pgvector defaults, Milvus collections may
        # contain several dense fields.  Preserve omission so the adapter can
        # auto-select exactly one compatible field or reject an ambiguity;
        # generated examples/CLI configs still write ``embedding`` explicitly.
        if "vector_field" not in index_raw:
            index_raw["vector_field"] = None
    cfg = EmbedFlowConfig(
        source=source,
        target=target,
        index=IndexConfig(**index_raw),
        documents=DocumentsConfig(**dict(raw.get("documents", {}))),
        migration=MigrationConfig(**migration_raw),
        cache=CacheConfig(**dict(raw.get("cache", {}))),
        economics=EconomicsConfig(**dict(raw.get("economics", {}))),
        probe=ProbeConfig(**dict(raw.get("probe", {}))),
        planner=PlannerConfig(**_planner_raw(raw.get("planner", {}))),
        runtime=RuntimeConfig(**runtime_raw),
        shadow=ShadowConfig(**shadow_raw),
        telemetry=TelemetryConfig(**dict(raw.get("telemetry", {}))),
        state_path=str(raw.get("state_path", "./embedflow_state.json")),
        dashboard_title=str(raw.get("dashboard_title", EmbedFlowConfig.__dataclass_fields__["dashboard_title"].default)),
    )
    if validate:
        cfg.validate()
    return cfg


def _planner_raw(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize planner aliases while keeping the public YAML surface small."""
    if raw is None:
        value: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise ValueError("planner must be a YAML object")
    if "gpu_hourly_cost" not in value and "gpu_price_per_hour" in value:
        value["gpu_hourly_cost"] = value.pop("gpu_price_per_hour")
    if "queries_per_second" not in value and "qps" in value:
        value["queries_per_second"] = value.pop("qps")
    if "target_docs_per_second" not in value and "docs_per_second" in value:
        value["target_docs_per_second"] = value.pop("docs_per_second")
    if "max_candidates" not in value and "max_work" in value:
        value["max_candidates"] = value.pop("max_work")
    return value


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
        "EMBEDFLOW_RUNTIME_MODE": (cfg.runtime, "mode"),
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
        "EMBEDFLOW_MILVUS_URI": (cfg.index, "uri"),
        "EMBEDFLOW_MILVUS_TOKEN_ENV": (cfg.index, "token_env"),
        "EMBEDFLOW_MILVUS_DATABASE": (cfg.index, "database"),
        "EMBEDFLOW_MILVUS_COLLECTION": (cfg.index, "collection"),
        "EMBEDFLOW_MILVUS_ID_FIELD": (cfg.index, "id_field"),
        "EMBEDFLOW_MILVUS_VECTOR_FIELD": (cfg.index, "vector_field"),
        "EMBEDFLOW_MILVUS_TEXT_FIELD": (cfg.index, "text_field"),
        "EMBEDFLOW_MILVUS_PARTITIONS": (cfg.index, "partition_names"),
        "EMBEDFLOW_WEAVIATE_URI": (cfg.index, "uri"),
        "EMBEDFLOW_WEAVIATE_HTTP_HOST": (cfg.index, "http_host"),
        "EMBEDFLOW_WEAVIATE_HTTP_PORT": (cfg.index, "http_port"),
        "EMBEDFLOW_WEAVIATE_GRPC_HOST": (cfg.index, "grpc_host"),
        "EMBEDFLOW_WEAVIATE_GRPC_PORT": (cfg.index, "grpc_port"),
        "EMBEDFLOW_WEAVIATE_SECURE": (cfg.index, "secure"),
        "EMBEDFLOW_WEAVIATE_GRPC_SECURE": (cfg.index, "grpc_secure"),
        "EMBEDFLOW_WEAVIATE_API_KEY_ENV": (cfg.index, "api_key_env"),
        "EMBEDFLOW_WEAVIATE_COLLECTION": (cfg.index, "collection"),
        "EMBEDFLOW_WEAVIATE_VECTOR_NAME": (cfg.index, "vector_name"),
        "EMBEDFLOW_WEAVIATE_TEXT_PROPERTY": (cfg.index, "text_property"),
        "EMBEDFLOW_WEAVIATE_TENANT": (cfg.index, "tenant"),
        "EMBEDFLOW_DOCUMENTS_PATH": (cfg.documents, "path"),
        "EMBEDFLOW_CACHE_PATH": (cfg.cache, "path"),
        "EMBEDFLOW_STATE_PATH": (cfg, "state_path"),
        "EMBEDFLOW_LATENCY_LOG": (cfg.telemetry, "latency_log"),
        "EMBEDFLOW_PLANNER_ACCESS_TRACE": (cfg.planner, "access_trace"),
        "EMBEDFLOW_PLANNER_CORPUS_NAME": (cfg.planner, "corpus_name"),
        "EMBEDFLOW_PLANNER_CORPUS_FINGERPRINT": (cfg.planner, "corpus_fingerprint"),
        "EMBEDFLOW_SHADOW_TELEMETRY_PATH": (cfg.shadow.telemetry, "path"),
    }
    for variable, (target, field_name) in paths.items():
        value = os.environ.get(variable)
        if value is not None and value.strip():
            setattr(target, field_name, value.strip())
    for variable, target, field_name in (
        ("EMBEDFLOW_SHADOW_ENABLED", cfg.shadow, "enabled"),
        ("EMBEDFLOW_SHADOW_MATERIALIZE", cfg.shadow, "materialize"),
        ("EMBEDFLOW_SHADOW_TELEMETRY_ENABLED", cfg.shadow.telemetry, "enabled"),
        ("EMBEDFLOW_SHADOW_RETAIN_QUERY_RECORDS", cfg.shadow.telemetry, "retain_query_records"),
        ("EMBEDFLOW_SHADOW_RETAIN_QUERY_TEXT", cfg.shadow.telemetry, "retain_query_text"),
    ):
        value = os.environ.get(variable)
        if value is not None and value.strip():
            normalized = value.strip().lower()
            if normalized not in {"0", "1", "true", "false", "yes", "no"}:
                raise ValueError(f"{variable} must be true or false")
            setattr(target, field_name, normalized in {"1", "true", "yes"})
    shadow_ints = (
        ("EMBEDFLOW_SHADOW_SAMPLE_SEED", cfg.shadow, "sample_seed"),
        ("EMBEDFLOW_SHADOW_CANDIDATE_K", cfg.shadow, "candidate_k"),
        ("EMBEDFLOW_SHADOW_MAX_INFLIGHT", cfg.shadow, "max_inflight"),
        ("EMBEDFLOW_SHADOW_QUEUE_CAPACITY", cfg.shadow, "queue_capacity"),
        ("EMBEDFLOW_SHADOW_TIMEOUT_MS", cfg.shadow, "timeout_ms"),
        ("EMBEDFLOW_SHADOW_SHUTDOWN_GRACE_MS", cfg.shadow, "shutdown_grace_ms"),
        ("EMBEDFLOW_SHADOW_MAX_RECORDS", cfg.shadow.telemetry, "max_records"),
        ("EMBEDFLOW_SHADOW_REPORT_K", cfg.shadow.telemetry, "report_k"),
        ("EMBEDFLOW_SHADOW_RETENTION_DAYS", cfg.shadow.telemetry, "retention_days"),
    )
    for variable, target, field_name in shadow_ints:
        value = os.environ.get(variable)
        if value is not None and value.strip():
            try:
                parsed = int(value)
                if float(value) != parsed:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{variable} must be an integer") from exc
            setattr(target, field_name, parsed)
    shadow_floats = (
        ("EMBEDFLOW_SHADOW_SAMPLE_RATE", cfg.shadow, "sample_rate"),
        ("EMBEDFLOW_SHADOW_MIN_TARGET_COVERAGE", cfg.shadow.telemetry, "min_target_coverage_for_ranking"),
    )
    for variable, target, field_name in shadow_floats:
        value = os.environ.get(variable)
        if value is not None and value.strip():
            try:
                setattr(target, field_name, float(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{variable} must be a finite number") from exc
    for variable, field_name in (("EMBEDFLOW_WEAVIATE_SECURE", "secure"),
                                 ("EMBEDFLOW_WEAVIATE_GRPC_SECURE", "grpc_secure")):
        value = os.environ.get(variable)
        if value is not None and value.strip():
            normalized = value.strip().lower()
            if normalized not in {"0", "1", "true", "false", "yes", "no"}:
                raise ValueError(f"{variable} must be true or false")
            setattr(cfg.index, field_name, normalized in {"1", "true", "yes"})
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
    planner_ints = (
        ("EMBEDFLOW_PLANNER_MAX_PROBES", cfg.planner, "max_probes"),
        ("EMBEDFLOW_PLANNER_SEED", cfg.planner, "seed"),
        ("EMBEDFLOW_PLANNER_MAX_CANDIDATES", cfg.planner, "max_candidates"),
        ("EMBEDFLOW_PLANNER_MAX_TARGET_ENCODINGS", cfg.planner, "max_target_encodes"),
        ("EMBEDFLOW_PLANNER_MAX_SYNC_MISSES", cfg.planner, "max_sync_misses"),
        ("EMBEDFLOW_PLANNER_BACKGROUND_BATCH_SIZE", cfg.planner, "background_batch_size"),
    )
    for variable, target, field_name in planner_ints:
        value = os.environ.get(variable)
        if value is not None and value.strip():
            try:
                parsed = int(value)
                if float(value) != parsed:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{variable} must be an integer") from exc
            setattr(target, field_name, parsed)
    planner_floats = (
        ("EMBEDFLOW_PLANNER_GPU_HOURLY_COST", "gpu_hourly_cost"),
        ("EMBEDFLOW_PLANNER_TARGET_DOCS_PER_SECOND", "target_docs_per_second"),
        ("EMBEDFLOW_PLANNER_QPS", "queries_per_second"),
        ("EMBEDFLOW_PLANNER_DAILY_QUERIES", "daily_queries"),
        ("EMBEDFLOW_PLANNER_CACHE_HIT_RATE", "cache_hit_rate"),
        ("EMBEDFLOW_PLANNER_LATENCY_BUDGET_MS", "latency_budget_ms"),
    )
    for variable, field_name in planner_floats:
        value = os.environ.get(variable)
        if value is not None and value.strip():
            try:
                setattr(cfg.planner, field_name, float(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{variable} must be a finite number") from exc
    partitions = os.environ.get("EMBEDFLOW_MILVUS_PARTITION_NAMES") or os.environ.get("EMBEDFLOW_MILVUS_PARTITIONS")
    if partitions is not None and partitions.strip():
        cfg.index.partition_names = [item.strip() for item in partitions.split(",") if item.strip()]
    auto_load = os.environ.get("EMBEDFLOW_MILVUS_AUTO_LOAD")
    if auto_load is not None and auto_load.strip():
        value = auto_load.strip().lower()
        if value not in {"0", "1", "true", "false", "yes", "no"}:
            raise ValueError("EMBEDFLOW_MILVUS_AUTO_LOAD must be true or false")
        cfg.index.auto_load = value in {"1", "true", "yes"}


def save_config(cfg: EmbedFlowConfig, path: str | Path) -> None:
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
