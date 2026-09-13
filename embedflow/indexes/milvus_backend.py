"""Read-only Milvus adapter for existing dense-vector collections.

The adapter intentionally implements only the small :class:`VectorIndex`
contract used by EmbedFlow.  Normal application operations perform reads
(``search``, ``get``, and collection metadata calls); collection creation,
insertion, deletion, index management, and unloading are deliberately not
part of this class.  Fixture setup belongs in tests/examples.

``pymilvus`` is imported lazily so a base EmbedFlow installation remains free
of Milvus dependencies.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector

_METRIC_ALIASES = {
    "cosine": "cosine",
    "ip": "ip",
    "dot": "ip",
    "dotproduct": "ip",
    "inner_product": "ip",
    "inner-product": "ip",
    "l2": "l2",
    "euclidean": "l2",
}
_MILVUS_METRICS = {"cosine": "COSINE", "ip": "IP", "l2": "L2"}
# Milvus documents a 16,384 top-k limit for ANN search.  Keeping the bound
# here also prevents accidental oversized requests to remote deployments.
_MAX_TOP_K = 16_384
# FLOAT16/BFLOAT16 are represented differently by some Milvus server/client
# combinations.  Until the adapter has a portable representation for those
# fields, only the ordinary FLOAT_VECTOR contract is advertised.
_SUPPORTED_DENSE_TYPES = {"float_vector", "floatvector"}
_UNSUPPORTED_TYPES = {"sparse_float_vector", "sparsefloatvector", "binary_vector", "binaryvector"}
_SEARCH_PARAM_KEYS = {"ef", "nprobe", "radius", "range_filter"}
_NUMERIC_DENSE_TYPES = {
    2: "int8", 3: "int16", 4: "int32", 5: "int64", 20: "string", 21: "varchar",
    100: "binary_vector", 101: "float_vector", 102: "float16_vector", 103: "bfloat16_vector",
    104: "sparse_float_vector",
}
_SDK_LOGGER_NAMES = (
    "pymilvus.decorators",
    "pymilvus.client.grpc_handler",
    "pymilvus.milvus_client.milvus_client",
)
_SDK_LOG_LOCK = threading.RLock()


@contextmanager
def _quiet_sdk_logs() -> Iterator[None]:
    """Suppress verbose SDK RPC tracebacks while an adapter call is running.

    pymilvus logs a complete gRPC traceback at ERROR before raising its
    exception.  That is useful when debugging the SDK directly, but it makes
    ordinary EmbedFlow CLI errors noisy and can echo server diagnostics that
    include connection details.  Adapter errors are redacted and re-raised
    below, so temporarily disabling only the known pymilvus loggers gives a
    concise, stable user-facing error.  Logger state is restored immediately.
    """

    with _SDK_LOG_LOCK:
        loggers = [logging.getLogger(name) for name in _SDK_LOGGER_NAMES]
        previous = [(logger, logger.disabled) for logger in loggers]
        try:
            for logger in loggers:
                logger.disabled = True
            yield
        finally:
            for logger, disabled in previous:
                logger.disabled = disabled


def normalize_milvus_metric(metric: str) -> str:
    """Return the internal Milvus metric (``cosine``, ``ip``, or ``l2``)."""

    if not isinstance(metric, str) or metric.strip().lower() not in _METRIC_ALIASES:
        raise ValueError("Milvus metric must be cosine, inner_product/dot, or l2/euclidean")
    return _METRIC_ALIASES[metric.strip().lower()]


def _lookup(value: Any, key: str, default: Any = None) -> Any:
    """Read mapping, model, and camelCase SDK response fields."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        parts = key.split("_")
        camel = parts[0] + "".join(part.title() for part in parts[1:])
        if camel in value:
            return value[camel]
        # pymilvus Hit is dict-like but also exposes compatibility properties
        # such as ``id``, ``pk``, ``distance``, and ``entity``.  Prefer the
        # mapping payload, then honor those object properties when the key is
        # only available through the SDK model API.
        return getattr(value, key, getattr(value, camel, default))
    result = getattr(value, key, default)
    if result is not default:
        return result
    parts = key.split("_")
    camel = parts[0] + "".join(part.title() for part in parts[1:])
    return getattr(value, camel, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _redact_milvus_error(exc: BaseException, secret: str | None = None) -> str:
    """Keep SDK errors useful while removing token/password material."""

    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    message = re.sub(r"(?i)(token\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(password\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", message)
    # URI userinfo can contain a token/password in Zilliz-style endpoints.
    # Cover HTTP, Milvus gRPC, and generic URI forms used by hosted
    # deployments.  Credentials are supplied separately to MilvusClient in
    # normal operation, but a proxy/SDK may echo a credential-bearing URI.
    message = re.sub(r"(?i)((?:https?|milvus(?:\+grpc)?|grpc)://[^:/\s]+:)[^@\s]+(@)", r"\1<redacted>\2", message)
    # Also cover token-only userinfo (``scheme://token@host``), which some
    # hosted gateways accept without a username/password pair.
    message = re.sub(r"(?i)((?:https?|milvus(?:\+grpc)?|grpc)://)[^/@\s]+(@)", r"\1<redacted>\2", message)
    return message or exc.__class__.__name__


_redact_error = _redact_milvus_error


def _redact_uri(uri: str | None) -> str:
    if not uri:
        return "UNKNOWN"
    value = re.sub(r"(?i)^((?:https?|milvus(?:\+grpc)?|grpc)://[^:/\s]+:)[^@\s]+(@)", r"\1<redacted>\2", str(uri))
    value = re.sub(r"(?i)^((?:https?|milvus(?:\+grpc)?|grpc)://)[^/@\s]+(@)", r"\1<redacted>\2", value)
    value = re.sub(r"(?i)([?&](?:token|password|passwd|api_key)=)[^&\s]+", r"\1<redacted>", value)
    return value


def _pymilvus_module():
    try:
        import pymilvus
    except ImportError as exc:  # pragma: no cover - exercised in clean installs
        raise RuntimeError('Milvus support requires: pip install "embedflow[milvus]"') from exc
    return pymilvus


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Milvus {label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"Milvus {label} contains a NUL byte")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Milvus {label} must be a positive integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Milvus {label} must be a positive integer") from exc
    if not exact or parsed < 1:
        raise ValueError(f"Milvus {label} must be a positive integer")
    return parsed


def _validate_search_params(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small, explicit Milvus search-parameter surface.

    Native Milvus accepts a large index-specific parameter dictionary.  The
    adapter intentionally exposes only the common HNSW/IVF controls and the
    radius range fields used by the supported dense metrics.  Rejecting an
    unknown key at construction time avoids silently ignoring a typo.
    """

    params = dict(value)
    if "params" in params or "metric_type" in params:
        unknown = set(params) - {"metric_type", "params"}
        if unknown:
            raise ValueError(f"Milvus search_params has unsupported keys: {sorted(unknown)!r}")
        metric_type = params.get("metric_type")
        if metric_type is not None and (not isinstance(metric_type, str) or not metric_type.strip()):
            raise ValueError("Milvus search_params.metric_type must be a non-empty string")
        nested = params.get("params", {})
        if not isinstance(nested, Mapping):
            raise ValueError("Milvus search_params.params must be a mapping")
        unknown_nested = set(nested) - _SEARCH_PARAM_KEYS
        if unknown_nested:
            raise ValueError(f"Milvus search_params.params has unsupported keys: {sorted(unknown_nested)!r}")
        normalized = {**params, "params": dict(nested)}
        check = normalized["params"]
    else:
        unknown = set(params) - _SEARCH_PARAM_KEYS
        if unknown:
            raise ValueError(f"Milvus search_params has unsupported keys: {sorted(unknown)!r}")
        normalized = params
        check = normalized
    for key, raw in check.items():
        if key in {"ef", "nprobe"}:
            _positive_int(raw, f"search_params.{key}")
        else:
            if isinstance(raw, bool):
                raise ValueError(f"Milvus search_params.{key} must be a finite number")
            try:
                numeric = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Milvus search_params.{key} must be a finite number") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"Milvus search_params.{key} must be a finite number")
    return normalized


def _field_type_name(field: Any) -> str:
    value = _lookup(field, "data_type", _lookup(field, "type", ""))
    value = _enum_value(value)
    if isinstance(value, int):
        return _NUMERIC_DENSE_TYPES.get(value, str(value)).lower()
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    # ``DataType.FLOAT_VECTOR`` stringifies as ``DataType.FLOAT_VECTOR``.
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _field_params(field: Any) -> Mapping[str, Any]:
    params = _lookup(field, "params", None)
    if isinstance(params, Mapping):
        return params
    type_params = _lookup(field, "type_params", None)
    return type_params if isinstance(type_params, Mapping) else {}


def _field_dimension(field: Any) -> int | None:
    params = _field_params(field)
    for key in ("dim", "dimension"):
        value = _lookup(params, key, None)
        if value is not None:
            return _positive_int(value, "vector dimension")
    return None


def _normalise_rows(value: Any) -> list[Mapping[str, Any]]:
    """Normalize ``get`` responses from current and older SDKs."""

    if value is None:
        return []
    data = _lookup(value, "data", None)
    if data is not None and data is not value:
        value = data
    if isinstance(value, Mapping):
        # A few proxies wrap rows in ``results``.
        results = _lookup(value, "results", None)
        if results is not None:
            value = results
        else:
            value = [value]
    if not isinstance(value, (list, tuple)):
        try:
            value = list(value)
        except TypeError as exc:
            raise ValueError("Milvus response rows must be a sequence") from exc
    rows: list[Mapping[str, Any]] = []
    for row in value:
        if isinstance(row, Mapping):
            rows.append(row)
        else:
            converted = getattr(row, "to_dict", None)
            if callable(converted):
                row = converted()
            if not isinstance(row, Mapping):
                raise ValueError("Milvus response row is not an object")
            rows.append(row)
    return rows


class MilvusIndex(VectorIndex):
    """Read-only ``VectorIndex`` backed by an existing Milvus collection."""

    def __init__(
        self,
        client: Any,
        *,
        collection: str,
        dimension: int,
        metric: str = "cosine",
        database: str = "default",
        id_field: str = "id",
        vector_field: str | None = "embedding",
        text_field: str | None = "content",
        partition_names: Sequence[str] | None = None,
        documents: Mapping[str, Any] | None = None,
        uri: str | None = None,
        token_env: str | None = "EMBEDFLOW_MILVUS_TOKEN",
        token: str | None = None,
        search_params: Mapping[str, Any] | None = None,
        auto_load: bool = False,
        schema_description: Any | None = None,
        index_description: Any | None = None,
        loaded_state: Any | None = None,
    ) -> None:
        if client is None:
            raise ValueError("Milvus client is required")
        self.client = client
        self.collection = _non_empty(collection, "collection")
        self.database = _non_empty(database or "default", "database")
        self.id_field = _non_empty(id_field, "id_field")
        self.vector_field = None if vector_field is None else _non_empty(vector_field, "vector_field")
        self.text_field = None if text_field is None else _non_empty(text_field, "text_field")
        self.dimension = _positive_int(dimension, "dimension")
        self.metric = normalize_milvus_metric(metric)
        if partition_names is None:
            self.partition_names: tuple[str, ...] = ()
        else:
            if isinstance(partition_names, (str, bytes)):
                raise ValueError("Milvus partition_names must be a list of strings")
            values = tuple(_non_empty(item, "partition name") for item in partition_names)
            if len(set(values)) != len(values):
                raise ValueError("Milvus partition_names must not contain duplicates")
            self.partition_names = values
        self.documents = documents
        self.uri = uri.strip() if isinstance(uri, str) else uri
        self.token_env = token_env
        self._token = token
        if search_params is not None and not isinstance(search_params, Mapping):
            raise ValueError("Milvus search_params must be a mapping")
        self.search_params = _validate_search_params(search_params or {})
        self.auto_load = bool(auto_load)
        self._schema = schema_description
        self._index_description = index_description
        self._loaded_state = loaded_state
        self._closed = False
        self._lock = threading.RLock()
        self._size_cache: int | None = None
        self._index_type: str | None = None
        self._metric_verified: str | None = None
        self._id_kind: str = "unknown"
        self._text_cache: dict[str, str] = {}
        self._partitions_available: tuple[str, ...] | None = None

    @classmethod
    def connect(
        cls,
        uri: str | None = None,
        *,
        token_env: str | None = "EMBEDFLOW_MILVUS_TOKEN",
        token: str | None = None,
        database: str = "default",
        collection: str,
        id_field: str = "id",
        vector_field: str | None = "embedding",
        text_field: str | None = "content",
        partition_names: Sequence[str] | None = None,
        dimension: int | None = None,
        metric: str = "cosine",
        documents: Mapping[str, Any] | None = None,
        search_params: Mapping[str, Any] | None = None,
        auto_load: bool = False,
    ) -> MilvusIndex:
        # ``None`` means use the local Milvus default; an explicitly blank
        # URI is a configuration error and must not silently redirect a caller
        # to localhost.
        uri = _non_empty("http://localhost:19530" if uri is None else uri, "uri")
        selected_database = _non_empty("default" if database is None else database, "database")
        if token is None and token_env:
            if not isinstance(token_env, str) or not token_env.strip() or "\x00" in token_env:
                raise ValueError("Milvus token_env must be a non-empty environment-variable name")
            token_env = token_env.strip()
            token = os.environ.get(token_env) or None
        elif token_env is not None:
            token_env = _non_empty(token_env, "token_env")
        pymilvus = _pymilvus_module()
        kwargs: dict[str, Any] = {"uri": uri, "db_name": selected_database}
        if token:
            kwargs["token"] = token
        try:
            client = pymilvus.MilvusClient(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"could not connect to Milvus: {_redact_milvus_error(exc, token)}") from exc
        try:
            index = cls(
                client,
                collection=collection,
                dimension=dimension or 1,
                metric=metric,
                database=selected_database,
                id_field=id_field,
                vector_field=vector_field,
                text_field=text_field,
                partition_names=partition_names,
                documents=documents,
                uri=uri,
                token_env=token_env,
                token=token,
                search_params=search_params,
                auto_load=auto_load,
            )
            index._introspect(configured_dimension=dimension)
            index._ensure_loaded()
            return index
        except Exception:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise

    @classmethod
    def from_config(cls, cfg: Any, documents: Mapping[str, Any] | None = None) -> MilvusIndex:
        index_cfg = cfg.index
        uri = getattr(index_cfg, "uri", None) or getattr(index_cfg, "url", None)
        path = getattr(index_cfg, "path", None)
        if not uri and isinstance(path, str) and "://" in path:
            uri = path
        text_field = getattr(index_cfg, "text_field", "content")
        if documents is None and not text_field:
            raise ValueError("Milvus text_field is required when no separate document store is configured")
        return cls.connect(
            uri=uri,
            token_env=getattr(index_cfg, "token_env", None) or "EMBEDFLOW_MILVUS_TOKEN",
            database=getattr(index_cfg, "database", "default") or "default",
            collection=getattr(index_cfg, "collection", None) or "documents",
            id_field=getattr(index_cfg, "id_field", "id"),
            vector_field=getattr(index_cfg, "vector_field", None) or None,
            text_field=text_field,
            partition_names=getattr(index_cfg, "partition_names", None),
            dimension=getattr(cfg.source, "dimension", None),
            metric=getattr(index_cfg, "metric", "cosine"),
            documents=documents,
            search_params=getattr(index_cfg, "search_params", None),
            auto_load=bool(getattr(index_cfg, "auto_load", False)),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Milvus index is closed")

    def _call(self, operation: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        try:
            with _quiet_sdk_logs():
                return fn(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Milvus {operation} failed: {_redact_milvus_error(exc, self._token)}") from exc

    def _describe_collection(self) -> Any:
        if self._schema is None:
            method = getattr(self.client, "describe_collection", None)
            if not callable(method):
                raise RuntimeError("Milvus client does not provide describe_collection")
            try:
                self._schema = self._call("describe_collection", method, collection_name=self.collection)
            except RuntimeError as exc:
                detail = str(exc).lower()
                if "can't find collection" in detail or "collection not found" in detail or "collection does not exist" in detail:
                    raise ValueError(f"Milvus collection {self.collection!r} was not found in database {self.database!r}") from exc
                raise
        return self._schema

    def _fields(self) -> list[Any]:
        schema = self._describe_collection()
        fields = _lookup(schema, "fields", None)
        if fields is None:
            nested = _lookup(schema, "schema", None)
            fields = _lookup(nested, "fields", None)
        if fields is None:
            raise ValueError(f"Milvus collection {self.collection!r} schema did not contain fields")
        if isinstance(fields, Mapping):
            fields = list(fields.values())
        if not isinstance(fields, (list, tuple)):
            try:
                fields = list(fields)
            except TypeError as exc:
                raise ValueError("Milvus collection fields must be a sequence") from exc
        return list(fields)

    def _introspect(self, configured_dimension: int | None = None) -> None:
        fields = self._fields()
        by_name = {str(_lookup(field, "name", "")): field for field in fields}
        if self.id_field not in by_name:
            # If a caller used the conventional default but the collection has
            # a different primary key, keep the error actionable rather than
            # silently querying an arbitrary scalar field.
            raise ValueError(f"Milvus ID field {self.id_field!r} was not found in collection {self.collection!r}")
        id_field = by_name[self.id_field]
        if _lookup(id_field, "is_primary", True) is False:
            raise ValueError(f"Milvus ID field {self.id_field!r} is not the collection primary key")
        id_kind = _field_type_name(id_field)
        if id_kind == "int64" or "int64" in id_kind:
            self._id_kind = "int"
        elif id_kind in {"varchar", "string"} or "varchar" in id_kind or "string" in id_kind:
            self._id_kind = "str"
        else:
            raise ValueError(f"Milvus ID field {self.id_field!r} must be INT64 or VARCHAR, not {id_kind!r}")
        compatible: list[Any] = []
        for field in fields:
            kind = _field_type_name(field)
            if kind in _UNSUPPORTED_TYPES:
                continue
            if kind in _SUPPORTED_DENSE_TYPES:
                compatible.append(field)
        if self.vector_field is None:
            if len(compatible) != 1:
                raise ValueError("Milvus vector_field is required when the collection has multiple or no compatible dense vector fields")
            self.vector_field = str(_lookup(compatible[0], "name"))
        if self.vector_field not in by_name:
            raise ValueError(f"Milvus vector field {self.vector_field!r} was not found in collection {self.collection!r}")
        vector = by_name[self.vector_field]
        kind = _field_type_name(vector)
        if kind in _UNSUPPORTED_TYPES or kind not in _SUPPORTED_DENSE_TYPES:
            raise ValueError(f"Milvus vector field {self.vector_field!r} has unsupported type {kind!r}; dense FLOAT_VECTOR is required")
        if self.text_field and self.documents is None:
            text = by_name.get(self.text_field)
            if text is None:
                raise ValueError(f"Milvus text field {self.text_field!r} was not found in collection {self.collection!r}")
            text_kind = _field_type_name(text)
            if text_kind not in {"varchar", "string"} and "varchar" not in text_kind and "string" not in text_kind:
                raise ValueError(f"Milvus text field {self.text_field!r} must be VARCHAR/STRING, not {text_kind!r}")
        inferred = _field_dimension(vector)
        if configured_dimension is not None:
            configured = _positive_int(configured_dimension, "source dimension")
            if inferred is not None and configured != inferred:
                raise ValueError(f"source encoder produces {configured} dimensions but Milvus {self.vector_field} is dimension {inferred}")
            self.dimension = configured
        elif inferred is not None:
            self.dimension = inferred
        # Resolve index metadata once at connection time when the SDK exposes
        # it. This is read-only and avoids control-plane calls on every query.
        self._load_index_description()
        self._metric_verified = self._extract_metric()
        if self._metric_verified and self._metric_verified != self.metric:
            raise ValueError(f"configured Milvus metric {self.metric} does not match collection index metric {self._metric_verified}")
        self._index_type = self._extract_index_type()
        list_partitions = getattr(self.client, "list_partitions", None)
        if callable(list_partitions):
            try:
                available = self._call("list_partitions", list_partitions, collection_name=self.collection)
                self._partitions_available = tuple(str(item) for item in (available or []))
                missing = [item for item in self.partition_names if item not in self._partitions_available]
                if missing:
                    raise ValueError(f"Milvus partition(s) {missing!r} were not found in collection {self.collection!r}")
            except RuntimeError:
                raise
            except ValueError:
                raise
            except Exception:
                self._partitions_available = None

    def _load_index_description(self) -> None:
        if self._index_description is not None:
            return
        list_method = getattr(self.client, "list_indexes", None)
        describe_method = getattr(self.client, "describe_index", None)
        if not callable(list_method) or not callable(describe_method):
            return
        try:
            names = self._call("list_indexes", list_method, collection_name=self.collection, field_name=self.vector_field)
            if isinstance(names, Mapping):
                names = _lookup(names, "index_names", _lookup(names, "indexes", []))
            names = list(names or [])
            if not names:
                return
            name = names[0]
            if not isinstance(name, str):
                name = str(_lookup(name, "index_name", name))
            self._index_description = self._call("describe_index", describe_method,
                                                 collection_name=self.collection, index_name=name)
        except Exception:
            # Some Milvus deployments restrict index metadata calls even when
            # search is permitted. Introspection is best-effort; never make a
            # healthy read-only adapter unusable solely for status decoration.
            self._index_description = None

    def _extract_metric(self) -> str | None:
        description = self._index_description
        if description is None:
            return None
        raw = _lookup(description, "metric_type", _lookup(description, "metric", None))
        if raw is None:
            params = _lookup(description, "params", None)
            raw = _lookup(params, "metric_type", _lookup(params, "metric", None))
        if raw is None:
            return None
        try:
            return normalize_milvus_metric(str(_enum_value(raw)))
        except ValueError:
            return None

    def _extract_index_type(self) -> str | None:
        raw = _lookup(self._index_description, "index_type", _lookup(self._index_description, "indexType", None))
        return None if raw is None else str(_enum_value(raw))

    def _ensure_loaded(self) -> None:
        method = getattr(self.client, "get_load_state", None)
        if not callable(method):
            return
        try:
            if self._loaded_state is None:
                state = self._call("get_load_state", method, collection_name=self.collection)
                self._loaded_state = _lookup(state, "state", state)
            value = _enum_value(self._loaded_state)
            name = str(getattr(value, "name", value)).lower()
            # Numeric LoadState values are supported by checking their string
            # representation as well; current SDKs expose the enum object.
            loaded = name.endswith("loaded") or name in {"loadstate.loaded", "3"}
            if loaded:
                return
            if self.auto_load:
                load = getattr(self.client, "load_collection", None)
                if not callable(load):
                    raise RuntimeError("Milvus collection is not loaded and the client cannot load it")
                self._call("load_collection", load, collection_name=self.collection)
                self._loaded_state = "Loaded"
                return
            raise RuntimeError(f"Milvus collection {self.collection!r} is not loaded; load it or set auto_load: true")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Milvus load-state check failed: {_redact_milvus_error(exc, self._token)}") from exc

    def _search_kwargs(self, k: int, value: np.ndarray) -> dict[str, Any]:
        search_params = self._effective_search_params(k)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "data": [value.tolist()],
            "anns_field": self.vector_field,
            "limit": k,
            "output_fields": [self.text_field] if self.documents is None and self.text_field else [],
            "partition_names": list(self.partition_names) or None,
            "search_params": search_params,
        }
        return kwargs

    def _effective_search_params(self, k: int | None = None) -> dict[str, Any]:
        """Return Milvus' native search-param shape without guessing index type.

        Milvus accepts ``{"metric_type": ..., "params": {...}}``.  For
        convenience, a config containing only ``ef`` or ``nprobe`` is wrapped
        into that shape. Explicit native mappings pass through unchanged.
        """

        if not self.search_params:
            return {}
        if "params" in self.search_params or "metric_type" in self.search_params:
            result = dict(self.search_params)
            configured_metric = result.setdefault("metric_type", _MILVUS_METRICS[self.metric])
            if normalize_milvus_metric(str(_enum_value(configured_metric))) != self.metric:
                raise ValueError(
                    f"Milvus search_params.metric_type {configured_metric!r} does not match configured metric {self.metric!r}"
                )
            params = result.get("params")
            if params is None:
                result["params"] = {}
            elif not isinstance(params, Mapping):
                raise ValueError("Milvus search_params.params must be a mapping")
            else:
                result["params"] = dict(params)
            # Milvus HNSW requires ef >= limit.  Preserve the configured
            # value for small queries, but raise it locally for a larger
            # candidate request so users receive the requested K instead of
            # an opaque server-side "ef should be larger than k" error.
            if k is not None and "ef" in result["params"]:
                ef = _positive_int(result["params"]["ef"], "search_params.ef")
                result["params"]["ef"] = max(ef, int(k))
            return result
        if set(self.search_params) <= {"ef", "nprobe", "radius", "range_filter"}:
            params = dict(self.search_params)
            if k is not None and "ef" in params:
                ef = _positive_int(params["ef"], "search_params.ef")
                params["ef"] = max(ef, int(k))
            return {"metric_type": _MILVUS_METRICS[self.metric], "params": params}
        raise ValueError("Milvus search_params must use native {metric_type, params} or ef/nprobe keys")

    def _first_hits(self, response: Any) -> list[Any]:
        if response is None:
            raise ValueError("Milvus search response was empty")
        if isinstance(response, Mapping):
            response = _lookup(response, "results", _lookup(response, "data", response))
        if not isinstance(response, (list, tuple)):
            try:
                response = list(response)
            except TypeError as exc:
                raise ValueError("Milvus search response must be a sequence") from exc
        if not response:
            return []
        first = response[0]
        # Current MilvusClient returns List[List[dict]].  A fake/legacy client
        # may return the inner hit list directly.
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(response)

    def _remember_text(self, hit: Any, document_id: str) -> None:
        if not self.text_field or self.documents is not None:
            return
        entity = _lookup(hit, "entity", None)
        value = _lookup(entity, self.text_field, None)
        # pymilvus 3.x Hit.entity is itself a view of the complete hit and
        # nests requested output fields under a second ``entity`` key. Older
        # clients expose the requested fields directly. Accept both shapes.
        if value is None:
            value = _lookup(_lookup(entity, "entity", None), self.text_field, None)
        if value is None:
            value = _lookup(hit, self.text_field, None)
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError(f"Milvus candidate {document_id!r} field {self.text_field!r} must be text")
        with self._lock:
            self._text_cache[document_id] = value

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        if k > _MAX_TOP_K:
            raise ValueError(f"Milvus top_k must be <= {_MAX_TOP_K}")
        value = validate_query_vector(query_vector, self.dimension)
        self._ensure_loaded()
        method = getattr(self.client, "search", None)
        if not callable(method):
            raise RuntimeError("Milvus client does not provide search")
        response = self._call("search", method, **self._search_kwargs(k, value))
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for rank, hit in enumerate(self._first_hits(response)):
            raw_id = _lookup(hit, "id", _lookup(hit, "pk", None))
            if raw_id is None:
                entity = _lookup(hit, "entity", None)
                raw_id = _lookup(entity, self.id_field, None)
            if raw_id is None:
                raise ValueError("Milvus search response contained a hit without an ID")
            document_id = str(raw_id)
            if not document_id:
                raise ValueError("Milvus search response contained an empty ID")
            if document_id in seen:
                raise ValueError(f"Milvus search returned duplicate document ID {document_id!r}")
            score_raw = _lookup(hit, "distance", _lookup(hit, "score", None))
            if score_raw is None:
                raise ValueError(f"Milvus hit {document_id!r} did not contain a distance/score")
            if isinstance(score_raw, bool):
                raise ValueError(f"Milvus hit {document_id!r} contained an invalid score")
            try:
                raw_score = float(score_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Milvus hit {document_id!r} contained an invalid score") from exc
            if not np.isfinite(raw_score):
                raise ValueError(f"Milvus hit {document_id!r} contained a non-finite score")
            # COSINE and IP are higher-is-better in Milvus. L2 is a distance.
            score = -raw_score if self.metric == "l2" else raw_score
            self._remember_text(hit, document_id)
            seen.add(document_id)
            hits.append(SearchHit(document_id, float(score), rank))
        return hits[:k]

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        self._ensure_open()
        normalized = list(dict.fromkeys(str(value) for value in ids))
        if not normalized:
            return {}
        if self.documents is not None:
            return {document_id: self.documents[document_id] for document_id in normalized if document_id in self.documents}
        if not self.text_field:
            raise RuntimeError("Milvus text_field is unset and no separate document store was configured")
        cached = getattr(self, "_text_cache", {})
        with self._lock:
            output = {document_id: cached[document_id] for document_id in normalized if document_id in cached}
        missing = [document_id for document_id in normalized if document_id not in output]
        if not missing:
            return output
        method = getattr(self.client, "get", None)
        if not callable(method):
            raise RuntimeError("Milvus client does not provide get for text-backed documents")
        # Milvus INT64 primary keys must be sent as ints, while VARCHAR keys
        # must remain strings. Canonical IDs remain strings at the EmbedFlow
        # boundary; conversion is performed only for this typed SDK call.
        # Only ask Milvus for uncached IDs.  Besides avoiding unnecessary
        # server work, this prevents a legitimate cached row from being
        # mistaken for a duplicate when the SDK returns all requested rows.
        lookup_ids: list[Any] = missing
        if self._id_kind == "int":
            lookup_ids = []
            for document_id in missing:
                try:
                    if str(int(document_id)) != document_id and document_id not in {"0", "-0"}:
                        raise ValueError
                    lookup_ids.append(int(document_id))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Milvus INT64 ID {document_id!r} is not an integer") from exc
        response = self._call(
            "get",
            method,
            collection_name=self.collection,
            ids=lookup_ids,
            output_fields=[self.id_field, self.text_field],
            partition_names=list(self.partition_names) or None,
        )
        rows = _normalise_rows(response)
        for row in rows:
            raw_id = _lookup(row, self.id_field, _lookup(row, "id", None))
            if raw_id is None:
                continue
            document_id = str(raw_id)
            value = _lookup(row, self.text_field, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"Milvus document {document_id!r} field {self.text_field!r} must be text")
            if document_id in output:
                raise ValueError(f"Milvus returned duplicate document ID {document_id!r}")
            output[document_id] = value
            with self._lock:
                self._text_cache[document_id] = value
        return {document_id: output[document_id] for document_id in normalized if document_id in output}

    def iter_ids(self) -> Iterator[str]:
        """Yield primary keys for explicit prewarm/export operations.

        Prefer Milvus' server-side ``query_iterator`` when available so
        explicit prewarm can enumerate collections larger than the ordinary
        query page limit.  Older clients without that API retain a bounded,
        fail-closed fallback rather than silently returning a partial ID set.
        """
        self._ensure_open()
        total = self.size()
        if total <= 0:
            return

        iterator_method = getattr(self.client, "query_iterator", None)
        if callable(iterator_method):
            iterator = self._call(
                "query_iterator",
                iterator_method,
                collection_name=self.collection,
                batch_size=min(1000, _MAX_TOP_K),
                limit=-1,
                filter="",
                output_fields=[self.id_field],
                partition_names=list(self.partition_names) or None,
            )
            seen: set[str] = set()
            try:
                next_method = getattr(iterator, "next", None)
                if not callable(next_method):
                    raise RuntimeError("Milvus query_iterator did not provide next()")
                while True:
                    batch = self._call("query_iterator.next", next_method)
                    if not batch:
                        break
                    for row in _normalise_rows(batch):
                        raw_id = _lookup(row, self.id_field, _lookup(row, "id", None))
                        if raw_id is None:
                            raise ValueError("Milvus query_iterator returned a row without an ID")
                        document_id = str(raw_id)
                        if document_id in seen:
                            raise ValueError(f"Milvus query_iterator returned duplicate document ID {document_id!r}")
                        seen.add(document_id)
                        yield document_id
            finally:
                close_iterator = getattr(iterator, "close", None)
                if callable(close_iterator):
                    try:
                        close_iterator()
                    except Exception:
                        pass
            if len(seen) < total:
                raise RuntimeError(
                    f"Milvus ID enumeration returned {len(seen)} of {total} entities; pass explicit document IDs"
                )
            return

        method = getattr(self.client, "query", None)
        if not callable(method):
            raise RuntimeError("Milvus client does not provide query; pass explicit document IDs to prewarm")
        limit = min(total, _MAX_TOP_K)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "filter": "",
            "output_fields": [self.id_field],
            "limit": limit,
            "partition_names": list(self.partition_names) or None,
        }
        rows = _normalise_rows(self._call("query", method, **kwargs))
        seen: set[str] = set()
        for row in rows:
            raw_id = _lookup(row, self.id_field, _lookup(row, "id", None))
            if raw_id is None:
                continue
            document_id = str(raw_id)
            if document_id in seen:
                raise ValueError(f"Milvus query returned duplicate document ID {document_id!r}")
            seen.add(document_id)
            yield document_id
        if total > limit and len(seen) < total:
            raise RuntimeError("Milvus ID enumeration exceeds one query page; pass explicit document IDs")

    def size(self) -> int:
        self._ensure_open()
        if self._size_cache is not None:
            return self._size_cache
        method = getattr(self.client, "get_collection_stats", None)
        if not callable(method):
            return 0
        stats = self._call("get_collection_stats", method, collection_name=self.collection)
        value = _lookup(stats, "row_count", _lookup(stats, "rowCount", 0))
        try:
            self._size_cache = max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            self._size_cache = 0
        return self._size_cache

    def _load_state_metadata(self) -> str:
        state = _enum_value(self._loaded_state)
        if state is None:
            return "UNKNOWN"
        if state in {0, 1, 2, 3}:
            return {0: "NotExist", 1: "NotLoad", 2: "Loading", 3: "Loaded"}[state]
        return str(getattr(state, "name", state))

    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        fields = self._fields()
        vector = next((field for field in fields if str(_lookup(field, "name", "")) == self.vector_field), None)
        kind = _field_type_name(vector) if vector is not None else "UNKNOWN"
        return {
            "backend": "milvus",
            "uri": _redact_uri(self.uri),
            "database": self.database,
            "collection": self.collection,
            "id_field": self.id_field,
            "vector_field": self.vector_field or "UNKNOWN",
            "text_field": self.text_field or "UNKNOWN",
            "vector_type": kind.upper(),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self._index_type or "UNKNOWN",
            "load_state": self._load_state_metadata(),
            "partitions": list(self.partition_names),
            "available_partitions": list(self._partitions_available) if self._partitions_available is not None else "UNKNOWN",
            "search_params": dict(self.search_params),
            "size": self.size(),
            "token_env": self.token_env,
        }

    def health_check(self) -> dict[str, Any]:
        try:
            self._describe_collection()
            self._ensure_loaded()
            return {"ok": True, "backend": "milvus", "database": self.database,
                    "collection": self.collection, "vector_count": self.size(),
                    "load_state": self._load_state_metadata()}
        except Exception as exc:
            return {"ok": False, "backend": "milvus", "database": self.database,
                    "collection": self.collection, "error": _redact_milvus_error(exc, self._token)}

    def audit(self, source_dimension: int | None = None) -> dict[str, Any]:
        checks: dict[str, Any] = {
            "sdk": {"ok": True, "detail": "pymilvus loaded"},
            "credentials": {"ok": True, "detail": "token configured" if self._token else "token not configured (local/no-auth or externally managed)"},
            "connection": self.health_check(),
        }
        checks["schema"] = {"ok": True, "collection": self.collection, "vector_field": self.vector_field,
                             "id_field": self.id_field, "vector_type": _field_type_name(next((f for f in self._fields() if str(_lookup(f, "name", "")) == self.vector_field), {}))}
        checks["dimension"] = {"configured": source_dimension or self.dimension, "collection": self.dimension,
                                "ok": source_dimension is None or int(source_dimension) == self.dimension}
        checks["metric"] = {"configured": self.metric, "index": self._metric_verified or "UNKNOWN",
                             "verified": self._metric_verified is not None,
                             "ok": self._metric_verified in {None, self.metric}}
        checks["partitions"] = {"configured": list(self.partition_names), "ok": True, "verified": False}
        try:
            probe = np.zeros(self.dimension, dtype="float32")
            probe[0] = 1.0
            hits = self.search(probe, 1)
            checks["candidate_query"] = {"ok": True, "matches": len(hits)}
            if self.documents is None and self.text_field and hits:
                text = self.fetch_documents([hits[0].document_id])
                checks["document_text"] = {"ok": hits[0].document_id in text, "mode": "milvus"}
            else:
                checks["document_text"] = {"ok": True, "mode": "external" if self.documents is not None else "milvus"}
        except Exception as exc:
            checks["candidate_query"] = {"ok": False, "error": _redact_milvus_error(exc, self._token)}
            checks["document_text"] = {"ok": False, "mode": "external" if self.documents is not None else "milvus"}
        ok = all(bool(value.get("ok", True)) for value in checks.values() if isinstance(value, Mapping))
        return {"backend": "milvus", "database": self.database, "collection": self.collection, "ok": ok, "checks": checks}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class _MilvusDocumentMapping(Mapping[str, str]):
    def __init__(self, store: MilvusDocumentStore) -> None:
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


class MilvusDocumentStore:
    """Lazy text resolver over a Milvus collection."""

    def __init__(self, index: MilvusIndex, *, text_field: str | None = None, owns_index: bool = True):
        self.index = index
        self.id_field = index.id_field
        self.text_field = text_field or index.text_field or "content"
        self.path = None
        self.documents: Mapping[str, str] = _MilvusDocumentMapping(self)
        self._owns_index = owns_index

    def get(self, document_ids: Any) -> dict[str, str]:
        ids = [str(value) for value in document_ids]
        values = self.index.fetch_documents(ids)
        missing = [value for value in ids if value not in values]
        if missing:
            raise KeyError(f"Milvus document text missing for IDs {missing[:5]} (field {self.text_field!r})")
        return {value: str(values[value]) for value in ids}

    def size(self) -> int:
        return self.index.size()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()


__all__ = ["MilvusIndex", "MilvusDocumentStore", "normalize_milvus_metric", "_redact_milvus_error", "_MAX_TOP_K"]
