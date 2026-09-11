"""Read-only Pinecone adapter for existing dense vector indexes.

Pinecone is deliberately kept behind the small :class:`VectorIndex` contract
used by FAISS, Qdrant, and pgvector.  Normal EmbedFlow operations only issue
data-plane reads (``query``, ``fetch``, ``describe_index_stats``); the adapter
never creates, updates, or deletes an index or its records.

The Pinecone SDK is imported lazily so the base ``embedflow`` installation
remains lightweight.  API keys are read from an environment variable and are
never included in metadata or error messages.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector

_METRIC_ALIASES = {
    "cosine": "cosine",
    "dot": "dotproduct",
    "dotproduct": "dotproduct",
    "inner_product": "dotproduct",
    "euclidean": "euclidean",
    "l2": "euclidean",
}
_MAX_TOP_K = 10_000


def normalize_pinecone_metric(metric: str) -> str:
    """Return Pinecone's canonical metric name.

    EmbedFlow accepts ``dot``/``inner_product`` and ``l2``/``euclidean`` as
    aliases shared with the other backends.  Pinecone calls those metrics
    ``dotproduct`` and ``euclidean`` respectively.
    """

    if not isinstance(metric, str) or metric.strip().lower() not in _METRIC_ALIASES:
        raise ValueError("Pinecone metric must be cosine, dot/inner_product, or l2/euclidean")
    return _METRIC_ALIASES[metric.strip().lower()]


def _lookup(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from SDK response objects and mapping-shaped test doubles."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        # A few API response variants use camelCase field names.
        camel = key.split("_")
        camel_key = camel[0] + "".join(part.title() for part in camel[1:])
        if camel_key in value:
            return value[camel_key]
        return default
    result = getattr(value, key, default)
    if result is not default:
        return result
    camel = key.split("_")
    return getattr(value, camel[0] + "".join(part.title() for part in camel[1:]), default)


def _enum_value(value: Any) -> Any:
    """Return the scalar carried by SDK enum fields, when present."""

    return getattr(value, "value", value)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            return None
        return converted if isinstance(converted, Mapping) else None
    return None


def _index_detail(value: Any, key: str, default: Any = None) -> Any:
    """Read index metadata across Pinecone SDK response generations.

    Older SDK responses exposed ``dimension``/``metric`` directly on the
    index description. Current SDKs put dense-vector fields under
    ``description.schema.fields`` and expose some fields on stats instead.
    Keep this compatibility shim local to the adapter rather than coupling
    the migration engine to SDK model classes.
    """

    direct = _lookup(value, key, None)
    if direct is not None:
        return direct
    schema = _lookup(value, "schema", None)
    fields = _lookup(schema, "fields", None)
    if isinstance(fields, Mapping):
        for field in fields.values():
            found = _lookup(field, key, None)
            if found is not None:
                return found
    return default


def _redact_pinecone_error(exc: BaseException, secret: str | None = None) -> str:
    """Keep SDK errors useful while removing credentials and bearer material."""

    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    # Keep this conservative: only credential-shaped material is removed.
    message = re.sub(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(token\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    return message or exc.__class__.__name__


# Keep the familiar private helper name available to callers/tests that use
# the other backend modules' redaction helpers.
_redact_error = _redact_pinecone_error


def _pinecone_module():
    try:
        import pinecone
    except ImportError as exc:  # pragma: no cover - exercised in clean installs
        raise RuntimeError('Pinecone support requires: pip install "embedflow[pinecone]"') from exc
    return pinecone


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pinecone {label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"Pinecone {label} contains a NUL byte")
    return value.strip()


def _dimension(value: Any, label: str = "dimension") -> int:
    if isinstance(value, bool):
        raise ValueError(f"Pinecone {label} must be a positive integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Pinecone {label} must be a positive integer") from exc
    if not exact or parsed < 1:
        raise ValueError(f"Pinecone {label} must be a positive integer")
    return parsed


class PineconeIndex(VectorIndex):
    """Read-only ``VectorIndex`` backed by an existing Pinecone dense index.

    ``index`` is the data-plane client returned by ``Pinecone.Index``.  The
    constructor also accepts ``client=`` as a descriptive alias, which is
    convenient for small deterministic test doubles.  ``connect`` is the
    normal application entry point and targets an index by host whenever one
    is supplied.
    """

    def __init__(
        self,
        index: Any | None = None,
        *,
        client: Any | None = None,
        dimension: int,
        metric: str = "cosine",
        host: str | None = None,
        index_name: str | None = None,
        namespace: str = "",
        text_metadata_field: str | None = None,
        documents: Mapping[str, Any] | None = None,
        api_key_env: str | None = "PINECONE_API_KEY",
        api_key: str | None = None,
        index_description: Any | None = None,
    ) -> None:
        data_index = index if index is not None else client
        if data_index is None:
            raise ValueError("Pinecone index client is required")
        self.index = data_index
        # ``client`` is retained as an alias for code that treats every remote
        # backend as a client, while ``index`` is the Pinecone data-plane name.
        self.client = data_index
        self.dimension = _dimension(dimension)
        self.metric = normalize_pinecone_metric(metric)
        self.host = None if host is None else _non_empty(host, "host").rstrip("/")
        self.index_name = None if index_name is None else _non_empty(index_name, "index_name")
        if not isinstance(namespace, str):
            raise ValueError("Pinecone namespace must be a string")
        self.namespace = namespace
        self.text_metadata_field = None if text_metadata_field is None else _non_empty(text_metadata_field, "text_metadata_field")
        self.documents = documents
        self.api_key_env = api_key_env
        self._api_key = api_key
        self._index_description = index_description
        self._closed = False
        self._lock = threading.RLock()
        self._text_cache: dict[str, str] = {}
        self._stats_cache: Any | None = None

    @classmethod
    def connect(
        cls,
        host: str | None = None,
        *,
        api_key_env: str | None = "PINECONE_API_KEY",
        index_name: str | None = None,
        namespace: str = "",
        dimension: int | None = None,
        metric: str = "cosine",
        text_metadata_field: str | None = None,
        documents: Mapping[str, Any] | None = None,
    ) -> PineconeIndex:
        """Connect to an existing index without performing any writes.

        Host targeting is preferred.  If only ``index_name`` is supplied, the
        control-plane ``describe_index`` call resolves its host before the
        data-plane client is created.  A configured host always wins when both
        values are present.
        """

        if host is None and index_name is None:
            raise ValueError("Pinecone host or index_name is required")
        if host is not None:
            host = _non_empty(host, "host").rstrip("/")
        if index_name is not None:
            index_name = _non_empty(index_name, "index_name")
        env_name = api_key_env or "PINECONE_API_KEY"
        if not isinstance(env_name, str) or not env_name.strip() or "\x00" in env_name:
            raise ValueError("Pinecone api_key_env must be a non-empty environment-variable name")
        env_name = env_name.strip()
        api_key = os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_name} is not set.")
        pinecone = _pinecone_module()
        try:
            pc = pinecone.Pinecone(api_key=api_key)
            description = None
            if host is None:
                description = pc.describe_index(name=index_name)
                host = _lookup(description, "host")
                if not host:
                    raise RuntimeError("Pinecone describe_index did not return an index host")
                host = _non_empty(host, "host").rstrip("/")
            data_index = pc.Index(host=host)
            requested_dimension = None if dimension is None else _dimension(dimension)
            described_dimension = _index_detail(description, "dimension")
            described_metric = _index_detail(description, "metric")
            stats = None
            # Host-targeted data-plane connections do not have a control-plane
            # description. Current Pinecone stats expose dimension/metric, so
            # use one read-only stats call to validate the source contract when
            # the description did not provide it.
            if described_dimension is None or described_metric is None:
                stats_method = getattr(data_index, "describe_index_stats", None)
                if callable(stats_method):
                    stats = stats_method()
                if described_dimension is None:
                    described_dimension = _lookup(stats, "dimension")
                if described_metric is None:
                    described_metric = _lookup(stats, "metric")
            actual_dimension = None if described_dimension is None else _dimension(described_dimension)
            if requested_dimension is not None and actual_dimension is not None and requested_dimension != actual_dimension:
                raise ValueError(
                    f"source encoder dimension {requested_dimension} does not match Pinecone index dimension {actual_dimension}"
                )
            dimension = requested_dimension or actual_dimension
            if dimension is None:
                if stats is None:
                    stats = data_index.describe_index_stats()
                stats_dimension = _lookup(stats, "dimension")
                if stats_dimension is not None:
                    dimension = _dimension(stats_dimension)
            if requested_dimension is not None and dimension is not None and requested_dimension != int(dimension):
                raise ValueError(
                    f"source encoder dimension {requested_dimension} does not match Pinecone index dimension {int(dimension)}"
                )
            if dimension is None:
                raise ValueError("Pinecone index dimension is unavailable; set source.dimension")
            canonical_metric = normalize_pinecone_metric(metric)
            if described_metric is not None:
                actual_metric = normalize_pinecone_metric(str(_enum_value(described_metric)))
                if actual_metric != canonical_metric:
                    raise ValueError(
                        f"configured Pinecone metric {canonical_metric} does not match index metric {actual_metric}"
                    )
            return cls(
                data_index,
                dimension=dimension,
                metric=canonical_metric,
                host=host,
                index_name=index_name,
                namespace=namespace,
                text_metadata_field=text_metadata_field,
                documents=documents,
                api_key_env=env_name,
                api_key=api_key,
                index_description=description,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unable to reach configured Pinecone index: {_redact_pinecone_error(exc, api_key)}") from exc

    @classmethod
    def from_config(cls, cfg: Any, documents: Mapping[str, Any] | None = None) -> PineconeIndex:
        index_cfg = cfg.index
        host = getattr(index_cfg, "host", None)
        index_name = getattr(index_cfg, "index_name", None)
        # ``url`` is accepted as a backwards-compatible endpoint escape hatch,
        # but the explicit host field is preferred and documented.
        if host is None and isinstance(getattr(index_cfg, "url", None), str) and index_cfg.url.strip():
            host = index_cfg.url
        if host is None and isinstance(getattr(index_cfg, "path", None), str):
            path = index_cfg.path.strip()
            if path and (".pinecone.io" in path or path.startswith("https://")):
                host = path
        text_field = getattr(index_cfg, "text_metadata_field", None)
        if documents is None and not text_field:
            raise ValueError("Pinecone text_metadata_field is required when no separate document store is configured")
        api_key_env = getattr(index_cfg, "api_key_env", None)
        if not api_key_env or api_key_env == "QDRANT_API_KEY":
            api_key_env = "PINECONE_API_KEY"
        return cls.connect(
            host=host,
            index_name=index_name,
            api_key_env=api_key_env,
            namespace=getattr(index_cfg, "namespace", ""),
            dimension=getattr(cfg.source, "dimension", None),
            metric=getattr(index_cfg, "metric", "cosine"),
            text_metadata_field=text_field,
            documents=documents,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Pinecone index is closed")

    def _call(self, operation: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Pinecone {operation} failed: {_redact_pinecone_error(exc, self._api_key)}") from exc

    def _stats(self, *, refresh: bool = False) -> Any:
        self._ensure_open()
        if self._stats_cache is None or refresh:
            method = getattr(self.index, "describe_index_stats", None)
            if not callable(method):
                raise RuntimeError("Pinecone data-plane client does not provide describe_index_stats")
            self._stats_cache = self._call("describe_index_stats", method)
        return self._stats_cache

    def _namespace_count(self, stats: Any) -> int:
        namespaces = _lookup(stats, "namespaces", None)
        if self.namespace:
            entry = namespaces.get(self.namespace) if isinstance(namespaces, Mapping) else None
            return int(_lookup(entry, "vector_count", 0) or 0)
        if isinstance(namespaces, Mapping):
            # Pinecone's stats contract includes the default namespace under
            # the empty-string key.  When other namespaces exist but that key
            # is absent, the default namespace has zero vectors; using the
            # all-namespace total here would make corpus size and migration
            # status silently wrong.
            if "" in namespaces:
                return int(_lookup(namespaces[""], "vector_count", 0) or 0)
            if namespaces:
                return 0
        total = _lookup(stats, "total_vector_count")
        if total is not None:
            return int(total)
        return 0

    @staticmethod
    def _namespace_metadata(stats: Any) -> dict[str, Any]:
        """Convert SDK namespace models into JSON-safe status metadata."""

        namespaces = _lookup(stats, "namespaces", {}) or {}
        if not isinstance(namespaces, Mapping):
            return {}
        output: dict[str, Any] = {}
        for name, summary in namespaces.items():
            count = _lookup(summary, "vector_count", 0)
            try:
                count = int(count or 0)
            except (TypeError, ValueError, OverflowError):
                count = "UNKNOWN"
            output[str(name)] = {"vector_count": count}
        return output

    def _parse_matches(self, response: Any) -> list[Any]:
        matches = _lookup(response, "matches", None)
        if matches is None:
            # A malformed response must not silently look like an empty index.
            raise ValueError("Pinecone query response did not contain matches")
        if not isinstance(matches, (list, tuple)):
            try:
                matches = list(matches)
            except TypeError as exc:
                raise ValueError("Pinecone query response matches must be a sequence") from exc
        return list(matches)

    def _remember_metadata_text(self, match: Any, document_id: str) -> None:
        if not self.text_metadata_field:
            return
        metadata = _lookup(match, "metadata", None) or {}
        value = _lookup(metadata, self.text_metadata_field, None)
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError(
                f"Pinecone candidate {document_id!r} metadata field {self.text_metadata_field!r} must be text"
            )
        with self._lock:
            self._text_cache[document_id] = value

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        if k > _MAX_TOP_K:
            raise ValueError(f"Pinecone top_k must be <= {_MAX_TOP_K}")
        value = validate_query_vector(query_vector, self.dimension)
        query = getattr(self.index, "query", None)
        if not callable(query):
            raise RuntimeError("Pinecone data-plane client does not provide query")
        response = self._call(
            "query",
            query,
            namespace=self.namespace,
            vector=value.tolist(),
            top_k=k,
            include_values=False,
            include_metadata=bool(self.text_metadata_field and self.documents is None),
        )
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for rank, match in enumerate(self._parse_matches(response)):
            document_id_raw = _lookup(match, "id", None)
            if document_id_raw is None:
                raise ValueError("Pinecone query response contained a match without an ID")
            document_id = str(document_id_raw)
            if not document_id or len(document_id) > 512:
                raise ValueError("Pinecone query response contained an ID outside the supported 1-512 character range")
            if document_id in seen:
                raise ValueError(f"Pinecone query returned duplicate document ID {document_id!r}")
            score_raw = _lookup(match, "score", None)
            if score_raw is None:
                raise ValueError(f"Pinecone match {document_id!r} did not contain a score")
            if isinstance(score_raw, bool):
                raise ValueError(f"Pinecone match {document_id!r} contained an invalid score")
            try:
                raw_score = float(score_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Pinecone match {document_id!r} contained an invalid score") from exc
            if not np.isfinite(raw_score):
                raise ValueError(f"Pinecone match {document_id!r} contained a non-finite score")
            # Pinecone's cosine and dotproduct scores are higher-is-better.
            # Its Euclidean score is squared distance, where lower is better.
            score = -raw_score if self.metric == "euclidean" else raw_score
            self._remember_metadata_text(match, document_id)
            seen.add(document_id)
            hits.append(SearchHit(document_id=document_id, score=float(score), source_rank=rank))
        return hits[:k]

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        self._ensure_open()
        normalized = list(dict.fromkeys(str(value) for value in ids))
        if not normalized:
            return {}
        if self.documents is not None:
            output: dict[str, Any] = {}
            for document_id in normalized:
                if document_id in self.documents:
                    output[document_id] = self.documents[document_id]
            return output
        if not self.text_metadata_field:
            raise RuntimeError("Pinecone text_metadata_field is unset and no separate document store was configured")
        with self._lock:
            output = {document_id: self._text_cache[document_id] for document_id in normalized if document_id in self._text_cache}
        missing = [document_id for document_id in normalized if document_id not in output]
        if not missing:
            return output
        fetch = getattr(self.index, "fetch", None)
        if not callable(fetch):
            raise RuntimeError("Pinecone data-plane client does not provide fetch for metadata-backed text")
        response = self._call("fetch", fetch, ids=missing, namespace=self.namespace)
        vectors = _lookup(response, "vectors", None)
        if vectors is None:
            raise ValueError("Pinecone fetch response did not contain vectors")
        if not isinstance(vectors, Mapping):
            try:
                vectors = dict(vectors)
            except (TypeError, ValueError) as exc:
                raise ValueError("Pinecone fetch response vectors must be an object") from exc
        for raw_id, record in vectors.items():
            document_id = str(_lookup(record, "id", raw_id))
            metadata = _lookup(record, "metadata", None) or {}
            value = _lookup(metadata, self.text_metadata_field, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Pinecone candidate {document_id!r} metadata field {self.text_metadata_field!r} must be text"
                )
            output[document_id] = value
            with self._lock:
                self._text_cache[document_id] = value
        return {document_id: output[document_id] for document_id in normalized if document_id in output}

    def iter_ids(self) -> Iterator[str]:
        """Yield IDs for explicit prewarm operations when the SDK supports listing."""

        list_method = getattr(self.index, "list", None)
        if not callable(list_method):
            raise RuntimeError("Pinecone client does not provide list; pass explicit document IDs to prewarm")
        result = self._call("list", list_method, namespace=self.namespace)
        for page in result:
            # Modern SDKs yield ListResponse pages whose ``vectors`` field
            # contains Vector records. Older/test clients may yield IDs or
            # records directly; support both shapes.
            entries = _lookup(page, "vectors", None)
            if entries is None:
                entries = _lookup(page, "ids", None)
            if entries is None:
                entries = page if isinstance(page, (list, tuple, set)) else [page]
            for item in entries:
                if isinstance(item, str):
                    yield item
                else:
                    value = _lookup(item, "id", item)
                    if value is not None:
                        yield str(value)

    def size(self) -> int:
        return self._namespace_count(self._stats())

    def _description(self) -> Any | None:
        if self._index_description is not None:
            return self._index_description
        return None

    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        stats = self._stats()
        description = self._description()
        namespaces = self._namespace_metadata(stats)
        metric_value = _lookup(stats, "metric", None) or _index_detail(description, "metric", None)
        if metric_value is None:
            metric = "UNKNOWN"
        else:
            try:
                metric = normalize_pinecone_metric(str(_enum_value(metric_value)))
            except ValueError:
                metric = "UNKNOWN"
        vector_type = _lookup(stats, "vector_type", None) or _index_detail(description, "vector_type", None)
        vector_type = _enum_value(vector_type) if vector_type is not None else None
        spec = _lookup(description, "spec", None)
        index_kind = "UNKNOWN"
        if isinstance(spec, Mapping):
            if "serverless" in spec:
                index_kind = "serverless"
            elif "pod" in spec:
                index_kind = "pod"
        deployment = _lookup(description, "deployment", None)
        deployment_name = type(deployment).__name__.lower() if deployment is not None else ""
        if "managed" in deployment_name or "serverless" in deployment_name:
            index_kind = "serverless"
        elif "pod" in deployment_name:
            index_kind = "pod"
        return {
            "backend": "pinecone",
            "host": self.host or "UNKNOWN",
            "index_name": self.index_name or "UNKNOWN",
            "namespace": self.namespace,
            "dimension": self.dimension,
            "metric": metric,
            "vector_type": vector_type or "UNKNOWN",
            "index_type": index_kind,
            "size": self._namespace_count(stats),
            "total_vector_count": _lookup(stats, "total_vector_count", "UNKNOWN"),
            "namespaces": namespaces,
            "api_key_env": self.api_key_env,
            "text_metadata_field": self.text_metadata_field,
        }

    def health_check(self) -> dict[str, Any]:
        try:
            stats = self._stats(refresh=True)
            return {
                "ok": True,
                "backend": "pinecone",
                "host": self.host or "UNKNOWN",
                "namespace": self.namespace,
                "vector_count": self._namespace_count(stats),
            }
        except Exception as exc:
            return {
                "ok": False,
                "backend": "pinecone",
                "host": self.host or "UNKNOWN",
                "namespace": self.namespace,
                "error": _redact_pinecone_error(exc, self._api_key),
            }

    def audit(self, source_dimension: int | None = None) -> dict[str, Any]:
        """Return safe read-only health and compatibility checks."""

        checks: dict[str, Any] = {
            "sdk": {"ok": True, "detail": "pinecone SDK loaded"},
            # An instance constructed around an injected data-plane client has
            # already delegated credential ownership to its caller.  ``connect``
            # verifies the environment variable before constructing an index.
            "credentials": {"ok": True, "detail": "available" if self._api_key is not None else "managed by injected client"},
            "connection": self.health_check(),
        }
        if source_dimension is not None:
            checks["dimension"] = {
                "configured": int(source_dimension),
                "index": self.dimension,
                "ok": int(source_dimension) == self.dimension,
            }
        else:
            checks["dimension"] = {"configured": self.dimension, "index": self.dimension, "ok": True}
        checks["metric"] = {
            "configured": self.metric,
            "index": "UNKNOWN",
            "verified": False,
            "ok": self.metric in {"cosine", "dotproduct", "euclidean"},
        }
        stats = None
        try:
            stats = self._stats()
            count = self._namespace_count(stats)
            metric_value = _lookup(stats, "metric", None) or _index_detail(self._description(), "metric", None)
            if metric_value is not None:
                try:
                    actual_metric = normalize_pinecone_metric(str(_enum_value(metric_value)))
                except ValueError:
                    checks["metric"] = {
                        "configured": self.metric,
                        "index": "UNKNOWN",
                        "verified": True,
                        "ok": False,
                    }
                else:
                    checks["metric"] = {
                        "configured": self.metric,
                        "index": actual_metric,
                        "verified": True,
                        "ok": actual_metric == self.metric,
                    }
            namespaces = _lookup(stats, "namespaces", None)
            if isinstance(namespaces, Mapping):
                namespace_known = self.namespace in namespaces
            else:
                namespace_known = False
            # An entirely empty/absent stats map cannot distinguish an empty
            # namespace from a namespace that has never been created. Treat
            # that as an audit finding rather than claiming an unverified
            # namespace is healthy. The candidate query below still reports
            # whether the data plane itself is reachable.
            namespace_ok = namespace_known
            checks["namespace"] = {"name": self.namespace, "vector_count": count,
                                    "known": namespace_known, "ok": namespace_ok}
            checks["stats"] = {"ok": True, "total_vector_count": _lookup(stats, "total_vector_count", "UNKNOWN")}
        except Exception as exc:
            checks["stats"] = {"ok": False, "error": _redact_pinecone_error(exc, self._api_key)}
            count = 0
        # Do not perform an unguarded second stats call after the guarded
        # block above.  An unavailable data plane should produce a structured
        # failed audit, not a raw exception from this reporting path.
        vector_type = _lookup(stats, "vector_type", None) or _index_detail(self._description(), "vector_type", "UNKNOWN")
        vector_type = _enum_value(vector_type)
        vector_type_normalized = str(vector_type).lower() if vector_type is not None else "unknown"
        checks["vector_type"] = {"value": vector_type, "ok": vector_type_normalized in {"dense", "unknown"}}
        # A small read-only candidate query proves the data-plane operation. An
        # empty namespace is still a successful query; it simply has no matches.
        try:
            probe = np.zeros(self.dimension, dtype="float32")
            probe[0] = 1.0
            matches = self.search(probe, 1)
            text_ok = True
            if matches and self.documents is None and self.text_metadata_field:
                resolved = self.fetch_documents([matches[0].document_id])
                text_ok = matches[0].document_id in resolved
            checks["candidate_query"] = {"ok": True, "matches": len(matches)}
            checks["document_text"] = {"ok": text_ok, "mode": "external" if self.documents is not None else "metadata"}
        except Exception as exc:
            checks["candidate_query"] = {"ok": False, "error": _redact_pinecone_error(exc, self._api_key)}
            checks["document_text"] = {"ok": False, "mode": "external" if self.documents is not None else "metadata"}
        ok = all(bool(value.get("ok", True)) if isinstance(value, Mapping) else bool(value) for value in checks.values())
        return {"backend": "pinecone", "namespace": self.namespace, "ok": ok, "checks": checks}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.index, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class _PineconeDocumentMapping(Mapping[str, str]):
    def __init__(self, store: PineconeDocumentStore):
        self.store = store

    def __getitem__(self, key: str) -> str:
        values = self.store.get([str(key)])
        if str(key) not in values:
            raise KeyError(str(key))
        return values[str(key)]

    def __iter__(self) -> Iterator[str]:
        return self.store.index.iter_ids()

    def __len__(self) -> int:
        return self.store.size()


class PineconeDocumentStore:
    """Lazy ``DocumentStore``-compatible text resolver for Pinecone metadata."""

    def __init__(self, index: PineconeIndex, *, text_field: str | None = None, owns_index: bool = True):
        self.index = index
        self.id_field = "id"
        self.text_field = text_field or index.text_metadata_field or "text"
        self.path = None
        self.documents: Mapping[str, str] = _PineconeDocumentMapping(self)
        self._owns_index = owns_index

    def get(self, document_ids: Any) -> dict[str, str]:
        ids = [str(value) for value in document_ids]
        values = self.index.fetch_documents(ids)
        missing = [value for value in ids if value not in values]
        if missing:
            field = self.index.text_metadata_field or self.text_field
            raise KeyError(f"Pinecone document text missing for IDs {missing[:5]} (metadata field {field!r})")
        return {value: str(values[value]) for value in ids}

    def size(self) -> int:
        return self.index.size()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()


__all__ = [
    "PineconeIndex",
    "PineconeDocumentStore",
    "normalize_pinecone_metric",
    "_redact_pinecone_error",
]
