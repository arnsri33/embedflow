"""Read-only Weaviate v4 adapter for existing dense vector collections.

The adapter deliberately sits behind the same small :class:`VectorIndex`
contract as the other remote backends.  Normal EmbedFlow operations only use
collection configuration, ``near_vector`` and batched object reads; no schema
or object mutation API is called here.  The ``weaviate-client`` dependency is
imported lazily so importing the base package remains lightweight.
"""

from __future__ import annotations

import os
import re
import threading
import uuid as uuid_module
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector

_METRIC_ALIASES = {
    "cosine": "cosine",
    "dot": "dot",
    "ip": "dot",
    "inner_product": "dot",
    "inner-product": "dot",
    "dotproduct": "dot",
    "l2": "l2",
    "euclidean": "l2",
    "squared_l2": "l2",
    "squaredl2": "l2",
    # Weaviate exposes this exact spelling in collection vector config for
    # its L2 distance implementation.  Keep it as an introspection alias;
    # users may still use the portable ``l2``/``euclidean`` names in YAML.
    "l2-squared": "l2",
    "squared-l2": "l2",
}
_WEAVIATE_METRICS = {"cosine": "cosine", "dot": "dot", "l2": "l2-squared"}
_MAX_TOP_K = 10_000


def normalize_weaviate_metric(metric: str) -> str:
    if not isinstance(metric, str) or metric.strip().lower() not in _METRIC_ALIASES:
        raise ValueError("Weaviate metric must be cosine, dot/inner_product, or l2/euclidean")
    return _METRIC_ALIASES[metric.strip().lower()]


def _lookup(value: Any, key: str, default: Any = None) -> Any:
    """Read dict-like, dataclass and SDK model fields (including camelCase)."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        parts = key.split("_")
        camel = parts[0] + "".join(part.title() for part in parts[1:])
        if camel in value:
            return value[camel]
        return default
    result = getattr(value, key, default)
    if result is not default:
        return result
    parts = key.split("_")
    camel = parts[0] + "".join(part.title() for part in parts[1:])
    return getattr(value, camel, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _redact_weaviate_error(exc: BaseException, secret: str | None = None) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    message = re.sub(r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(token\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(password\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)((?:https?|grpc|weaviate)://[^:/\s]+:)[^@\s]+(@)", r"\1<redacted>\2", message)
    message = re.sub(r"(?i)((?:https?|grpc|weaviate)://)[^/@\s]+(@)", r"\1<redacted>\2", message)
    return message or exc.__class__.__name__


def _redact_uri(uri: str | None) -> str:
    if not uri:
        return "UNKNOWN"
    value = str(uri)
    value = re.sub(r"(?i)^((?:https?|grpc|weaviate)://[^:/\s]+:)[^@\s]+(@)", r"\1<redacted>\2", value)
    value = re.sub(r"(?i)^((?:https?|grpc|weaviate)://)[^/@\s]+(@)", r"\1<redacted>\2", value)
    value = re.sub(r"(?i)([?&](?:token|password|api_key)=)[^&\s]+", r"\1<redacted>", value)
    return value


def _weaviate_module():
    try:
        import weaviate
    except ImportError as exc:  # pragma: no cover - exercised in clean installs
        raise RuntimeError('Weaviate support requires: pip install "embedflow[weaviate]"') from exc
    return weaviate


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Weaviate {label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"Weaviate {label} contains a NUL byte")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Weaviate {label} must be a positive integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Weaviate {label} must be a positive integer") from exc
    if not exact or parsed < 1:
        raise ValueError(f"Weaviate {label} must be a positive integer")
    return parsed


def _to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_dict(v) for k, v in value.items()}
    method = getattr(value, "to_dict", None)
    if callable(method):
        try:
            return _to_dict(method())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {str(k): _to_dict(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [_to_dict(v) for v in value]
    return str(value)


def _safe_config(value: Any, secret: str | None = None) -> Any:
    """Return collection configuration without credential-bearing values.

    Weaviate's configuration models are normally metadata-only, but modules
    and proxy deployments can include headers or token-like fields.  Status
    and telemetry must remain safe even when a server returns such fields.
    ``api_key_env`` is an environment-variable *name*, not a secret, so it is
    retained for troubleshooting.
    """
    converted = _to_dict(value)
    sensitive = {"api_key", "apikey", "token", "password", "secret", "authorization", "headers"}

    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for key, child in item.items():
                name = str(key).lower().replace("-", "_")
                if name in sensitive or ("api_key" in name and not name.endswith("_env")):
                    output[str(key)] = "<redacted>"
                else:
                    output[str(key)] = scrub(child)
            return output
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if secret and isinstance(item, str):
            return item.replace(secret, "<redacted>")
        return item

    return scrub(converted)


def _property_type(prop: Any) -> str:
    value = _lookup(prop, "data_type", _lookup(prop, "dataType", ""))
    value = _enum_value(value)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.replace("_", "")


def _vector_configs(config: Any) -> dict[str, Any]:
    raw = _lookup(config, "vector_config", _lookup(config, "vectorConfig", None))
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if raw is not None:
        return {"default": raw}
    # Single-vector collections expose the legacy vector_index_config.
    legacy = _lookup(config, "vector_index_config", _lookup(config, "vectorIndexConfig", None))
    return {"default": legacy} if legacy is not None else {}


def _vector_config_metric(vector: Any) -> str | None:
    raw = _vector_config_raw_metric(vector)
    if raw is None:
        return None
    try:
        return normalize_weaviate_metric(str(_enum_value(raw)))
    except ValueError:
        return None


def _vector_config_raw_metric(vector: Any) -> Any:
    if vector is None:
        return None
    index = _lookup(vector, "vector_index_config", _lookup(vector, "vectorIndexConfig", vector))
    raw = _lookup(index, "distance_metric", _lookup(index, "distance", None))
    if raw is None:
        raw = _lookup(vector, "distance_metric", _lookup(vector, "distance", None))
    return raw


def _vector_config_type(vector: Any) -> str | None:
    index = _lookup(vector, "vector_index_config", _lookup(vector, "vectorIndexConfig", vector))
    raw = _lookup(index, "vector_index_type", _lookup(index, "vectorIndexType", None))
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = None
    return None if raw is None else str(_enum_value(raw))


def _vector_config_dimension(vector: Any) -> int | None:
    """Extract a dimension only when the SDK/server exposes one explicitly."""
    candidates = (vector, _lookup(vector, "vector_index_config", None),
                  _lookup(vector, "vectorIndexConfig", None))
    for candidate in candidates:
        for key in ("dimension", "dim", "vector_dimension"):
            raw = _lookup(candidate, key, None)
            if raw is None:
                continue
            try:
                return _positive_int(raw, "collection dimension")
            except ValueError:
                return None
    return None


class WeaviateIndex(VectorIndex):
    """Read-only VectorIndex over an existing Weaviate v4 collection."""

    def __init__(
        self,
        collection: Any,
        *,
        collection_name: str,
        dimension: int,
        metric: str = "cosine",
        vector_name: str | None = None,
        text_property: str | None = "content",
        documents: Mapping[str, Any] | None = None,
        client: Any | None = None,
        uri: str | None = None,
        api_key_env: str | None = "WEAVIATE_API_KEY",
        api_key: str | None = None,
        tenant: str | None = None,
        collection_config: Any | None = None,
    ) -> None:
        if collection is None:
            raise ValueError("Weaviate collection client is required")
        self.collection_client = collection
        self.collection = _non_empty(collection_name, "collection")
        self.dimension = _positive_int(dimension, "dimension")
        self.metric = normalize_weaviate_metric(metric)
        self.vector_name = None if vector_name is None else _non_empty(vector_name, "vector_name")
        self.text_property = None if text_property is None else _non_empty(text_property, "text_property")
        self.documents = documents
        self.client = client
        self.uri = uri
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.tenant = None if tenant is None else _non_empty(tenant, "tenant")
        self._config = collection_config
        self._closed = False
        self._lock = threading.RLock()
        self._size_cache: int | None = None
        self._text_cache: dict[str, str] = {}
        self._vector_names: tuple[str, ...] = ()
        self._metric_verified: str | None = None
        self._index_type: str | None = None
        self._multi_tenancy: bool | None = None
        self._collection_dimension: int | None = None

    @classmethod
    def connect(
        cls,
        *,
        collection: str,
        dimension: int | None = None,
        metric: str = "cosine",
        vector_name: str | None = None,
        text_property: str | None = "content",
        documents: Mapping[str, Any] | None = None,
        uri: str | None = None,
        http_host: str = "localhost",
        http_port: int = 8080,
        grpc_host: str | None = None,
        grpc_port: int = 50051,
        secure: bool = False,
        grpc_secure: bool | None = None,
        api_key_env: str | None = "WEAVIATE_API_KEY",
        api_key: str | None = None,
        tenant: str | None = None,
    ) -> WeaviateIndex:
        weaviate = _weaviate_module()
        if api_key is None and api_key_env:
            api_key_env = _non_empty(api_key_env, "api_key_env")
            api_key = os.environ.get(api_key_env) or None
        elif api_key_env is not None:
            api_key_env = _non_empty(api_key_env, "api_key_env")
        endpoint = uri or http_host
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("Weaviate uri or http_host is required")
        endpoint = endpoint.strip()
        try:
            parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
            host = parsed.hostname or endpoint
            parsed_port = parsed.port
        except ValueError as exc:
            raise ValueError("Weaviate URI contains an invalid host or port") from exc
        if parsed_port is not None:
            http_port = parsed_port
        if grpc_host is None:
            grpc_host = host
        auth = api_key
        http_secure = bool(secure or parsed.scheme.lower() == "https")
        grpc_is_secure = bool(grpc_secure if grpc_secure is not None else http_secure)
        # String credentials are interpreted as API keys by the official v4
        # helpers. Cloud uses the cluster hostname and derives its gRPC host.
        try:
            if ".weaviate.cloud" in host or ".weaviate.network" in host:
                if not auth:
                    raise RuntimeError(f"Environment variable {api_key_env or 'WEAVIATE_API_KEY'} is not set.")
                client = weaviate.connect_to_weaviate_cloud(host, auth_credentials=auth)
                actual_uri = f"https://{host}"
            elif http_secure or grpc_is_secure or (grpc_host is not None and grpc_host != host):
                custom_kwargs = {
                    "http_host": host, "http_port": _positive_int(http_port, "http_port"),
                    "http_secure": http_secure, "grpc_host": _non_empty(grpc_host, "grpc_host"),
                    "grpc_port": _positive_int(grpc_port, "grpc_port"), "grpc_secure": grpc_is_secure,
                }
                if auth:
                    custom_kwargs["auth_credentials"] = auth
                client = weaviate.connect_to_custom(**custom_kwargs)
                actual_uri = f"{'https' if http_secure else 'http'}://{host}:{http_port}"
            else:
                local_kwargs = {
                    "host": host, "port": _positive_int(http_port, "http_port"),
                    "grpc_port": _positive_int(grpc_port, "grpc_port"),
                }
                if auth:
                    local_kwargs["auth_credentials"] = auth
                client = weaviate.connect_to_local(**local_kwargs)
                actual_uri = f"http://{host}:{http_port}"
            ready = getattr(client, "is_ready", None)
            if callable(ready) and not bool(ready()):
                raise RuntimeError("Weaviate server is not ready")
            collections = getattr(client, "collections", None)
            get_collection = getattr(collections, "get", None) if collections is not None else None
            # ``get`` is the read-only lookup in current v4 clients.  A few
            # compatible client facades expose only ``use``; accepting it as
            # a fallback keeps the adapter interoperable without ever calling
            # a collection-creation method.
            if not callable(get_collection):
                get_collection = getattr(collections, "use", None) if collections is not None else None
            if not callable(get_collection):
                raise RuntimeError("Weaviate client does not provide collections.get/use")
            collection_obj = get_collection(_non_empty(collection, "collection"))
            if dimension is None:
                raise ValueError("Weaviate source dimension is required because the collection API does not expose it reliably")
            index = cls(collection_obj, collection_name=collection, dimension=dimension, metric=metric,
                        vector_name=vector_name, text_property=text_property, documents=documents, client=client,
                        uri=actual_uri, api_key_env=api_key_env, api_key=api_key, tenant=tenant)
            index._introspect(configured_dimension=dimension)
            return index
        except ValueError:
            try:
                if 'client' in locals() and callable(getattr(client, "close", None)):
                    client.close()
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                if 'client' in locals() and callable(getattr(client, "close", None)):
                    client.close()
            except Exception:
                pass
            raise RuntimeError(f"Unable to reach configured Weaviate collection: {_redact_weaviate_error(exc, api_key)}") from exc

    @classmethod
    def from_config(cls, cfg: Any, documents: Mapping[str, Any] | None = None) -> WeaviateIndex:
        index_cfg = cfg.index
        text_property = getattr(index_cfg, "text_property", None)
        if text_property is None:
            text_property = getattr(index_cfg, "text_field", "content")
        if documents is None and not text_property:
            raise ValueError("Weaviate text_property is required when no separate document store is configured")
        host = getattr(index_cfg, "http_host", None) or "localhost"
        uri = getattr(index_cfg, "uri", None) or getattr(index_cfg, "url", None)
        return cls.connect(
            collection=getattr(index_cfg, "collection", None) or "Documents",
            dimension=getattr(cfg.source, "dimension", None), metric=getattr(index_cfg, "metric", "cosine"),
            vector_name=getattr(index_cfg, "vector_name", None), text_property=text_property, documents=documents,
            uri=uri, http_host=host, http_port=getattr(index_cfg, "http_port", 8080),
            grpc_host=getattr(index_cfg, "grpc_host", None), grpc_port=getattr(index_cfg, "grpc_port", 50051),
            secure=bool(getattr(index_cfg, "secure", False)), grpc_secure=getattr(index_cfg, "grpc_secure", None),
            api_key_env=getattr(index_cfg, "api_key_env", None) or "WEAVIATE_API_KEY",
            tenant=getattr(index_cfg, "tenant", None),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Weaviate index is closed")

    def _call(self, operation: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        self._ensure_open()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Weaviate {operation} failed: {_redact_weaviate_error(exc, self._api_key)}") from exc

    def _bound_collection(self) -> Any:
        if not self.tenant:
            return self.collection_client
        method = getattr(self.collection_client, "with_tenant", None)
        if not callable(method):
            raise RuntimeError("configured Weaviate tenant is unsupported by this client")
        return method(self.tenant)

    def _load_config(self) -> Any:
        if self._config is None:
            config_api = getattr(self.collection_client, "config", None)
            get = getattr(config_api, "get", None) if config_api is not None else None
            if not callable(get):
                raise RuntimeError("Weaviate collection does not provide config.get")
            try:
                self._config = self._call("collection config", get)
            except RuntimeError as exc:
                if "404" in str(exc) or "not found" in str(exc).lower():
                    raise ValueError(f"Weaviate collection {self.collection!r} was not found") from exc
                raise
        return self._config

    def _introspect(self, configured_dimension: int | None = None) -> None:
        config = self._load_config()
        vectors = _vector_configs(config)
        if not vectors:
            raise ValueError(f"Weaviate collection {self.collection!r} has no configured vector fields")
        self._vector_names = tuple(vectors)
        if self.vector_name is None and len(vectors) > 1:
            raise ValueError(f"Weaviate collection has multiple named vectors {list(vectors)!r}; vector_name is required")
        if self.vector_name is not None and vectors and self.vector_name not in vectors:
            raise ValueError(f"Weaviate vector_name {self.vector_name!r} was not found; available: {list(vectors)!r}")
        if self.vector_name is None and len(vectors) == 1 and next(iter(vectors)) != "default":
            self.vector_name = next(iter(vectors))
        selected = vectors.get(self.vector_name or "default")
        raw_metric = _vector_config_raw_metric(selected)
        self._metric_verified = _vector_config_metric(selected)
        if raw_metric is not None and self._metric_verified is None:
            raise ValueError(f"Weaviate collection uses unsupported distance metric {str(_enum_value(raw_metric))!r}")
        if self._metric_verified and self._metric_verified != self.metric:
            raise ValueError(f"configured Weaviate metric {self.metric} does not match collection metric {self._metric_verified}")
        self._index_type = _vector_config_type(selected)
        self._collection_dimension = _vector_config_dimension(selected)
        mt = _lookup(config, "multi_tenancy_config", _lookup(config, "multiTenancyConfig", None))
        enabled = _lookup(mt, "enabled", None)
        self._multi_tenancy = None if enabled is None else bool(enabled)
        if self.tenant and self._multi_tenancy is False:
            raise ValueError(f"Weaviate collection {self.collection!r} is not multi-tenant")
        properties = _lookup(config, "properties", []) or []
        by_name = {str(_lookup(prop, "name", "")): prop for prop in properties}
        if self.text_property and self.documents is None:
            prop = by_name.get(self.text_property)
            if prop is None:
                raise ValueError(f"Weaviate text property {self.text_property!r} was not found in collection {self.collection!r}")
            if _property_type(prop) not in {"text", "text[]"}:
                raise ValueError(f"Weaviate text property {self.text_property!r} must be text")
        # Weaviate does not expose vector dimensionality in all config
        # versions.  Keep the source contract dimension and report UNKNOWN
        # rather than inventing a value; the server validates query length.
        configured = None if configured_dimension is None else _positive_int(configured_dimension, "source dimension")
        if self._collection_dimension is not None and configured is not None and self._collection_dimension != configured:
            raise ValueError(
                f"source encoder dimension {configured} does not match Weaviate collection dimension {self._collection_dimension}"
            )
        if configured is not None:
            self.dimension = configured

    def _query_kwargs(self, value: np.ndarray, k: int) -> dict[str, Any]:
        try:
            from weaviate.classes.query import MetadataQuery
            metadata = MetadataQuery(distance=True)
        except Exception:
            metadata = {"distance": True}
        kwargs: dict[str, Any] = {
            "near_vector": value.tolist(), "limit": k, "include_vector": False,
            "return_metadata": metadata,
            "return_properties": [self.text_property] if self.documents is None and self.text_property else [],
        }
        if self.vector_name:
            kwargs["target_vector"] = self.vector_name
        return kwargs

    @staticmethod
    def _objects(response: Any) -> list[Any]:
        if response is None:
            raise ValueError("Weaviate near_vector response was empty")
        objects = _lookup(response, "objects", None)
        if objects is None:
            raise ValueError("Weaviate near_vector response did not contain objects")
        if not isinstance(objects, (list, tuple)):
            try:
                objects = list(objects)
            except TypeError as exc:
                raise ValueError("Weaviate response objects must be a sequence") from exc
        return list(objects)

    def _remember_text(self, obj: Any, document_id: str) -> None:
        if self.documents is not None or not self.text_property:
            return
        props = _lookup(obj, "properties", None)
        value = _lookup(props, self.text_property, None)
        if value is None:
            raise ValueError(
                f"Weaviate candidate {document_id!r} is missing text property {self.text_property!r}"
            )
        if not isinstance(value, str):
            raise ValueError(f"Weaviate candidate {document_id!r} property {self.text_property!r} must be text")
        with self._lock:
            self._text_cache[document_id] = value

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        if k > _MAX_TOP_K:
            raise ValueError(f"Weaviate top_k must be <= {_MAX_TOP_K}")
        value = validate_query_vector(query_vector, self.dimension)
        collection = self._bound_collection()
        query_api = getattr(collection, "query", None)
        near_vector = getattr(query_api, "near_vector", None) if query_api is not None else None
        if not callable(near_vector):
            raise RuntimeError("Weaviate collection does not provide query.near_vector")
        response = self._call("near_vector", near_vector, **self._query_kwargs(value, k))
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for rank, obj in enumerate(self._objects(response)):
            raw_id = _lookup(obj, "uuid", _lookup(obj, "id", None))
            if raw_id is None:
                raise ValueError("Weaviate response contained an object without a UUID")
            document_id = str(raw_id)
            if not document_id:
                raise ValueError("Weaviate response contained an empty UUID")
            if document_id in seen:
                raise ValueError(f"Weaviate search returned duplicate object UUID {document_id!r}")
            metadata = _lookup(obj, "metadata", None)
            distance = _lookup(metadata, "distance", None)
            if distance is not None:
                try:
                    raw_score = float(distance)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Weaviate object {document_id!r} contained an invalid distance") from exc
                if not np.isfinite(raw_score):
                    raise ValueError(f"Weaviate object {document_id!r} contained a non-finite distance")
                score = -raw_score
            else:
                raw_score = _lookup(metadata, "score", None)
                if raw_score is None:
                    raise ValueError(f"Weaviate object {document_id!r} did not contain distance/score metadata")
                try:
                    score = float(raw_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Weaviate object {document_id!r} contained an invalid score") from exc
                if not np.isfinite(score):
                    raise ValueError(f"Weaviate object {document_id!r} contained a non-finite score")
            self._remember_text(obj, document_id)
            seen.add(document_id)
            hits.append(SearchHit(document_id, float(score), rank))
        return hits[:k]

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        self._ensure_open()
        normalized = list(dict.fromkeys(str(value) for value in ids))
        if not normalized:
            return {}
        if self.documents is not None:
            return {doc_id: self.documents[doc_id] for doc_id in normalized if doc_id in self.documents}
        if not self.text_property:
            raise RuntimeError("Weaviate text_property is unset and no separate document store was configured")
        with self._lock:
            output = {doc_id: self._text_cache[doc_id] for doc_id in normalized if doc_id in self._text_cache}
        missing = [doc_id for doc_id in normalized if doc_id not in output]
        if not missing:
            return output
        query_api = getattr(self._bound_collection(), "query", None)
        fetch = getattr(query_api, "fetch_objects_by_ids", None) if query_api is not None else None
        if not callable(fetch):
            raise RuntimeError("Weaviate collection does not provide query.fetch_objects_by_ids")
        try:
            typed_ids = [uuid_module.UUID(doc_id) for doc_id in missing]
        except (ValueError, AttributeError) as exc:
            raise ValueError("Weaviate object IDs must be UUIDs for property-backed text") from exc
        response = self._call("fetch_objects_by_ids", fetch, typed_ids, include_vector=False, return_properties=[self.text_property])
        for obj in self._objects(response):
            raw_id = _lookup(obj, "uuid", _lookup(obj, "id", None))
            if raw_id is None:
                continue
            doc_id = str(raw_id)
            props = _lookup(obj, "properties", {}) or {}
            value = _lookup(props, self.text_property, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"Weaviate document {doc_id!r} property {self.text_property!r} must be text")
            if doc_id in output:
                raise ValueError(f"Weaviate returned duplicate object UUID {doc_id!r}")
            output[doc_id] = value
            with self._lock:
                self._text_cache[doc_id] = value
        return {doc_id: output[doc_id] for doc_id in normalized if doc_id in output}

    def iter_ids(self) -> Iterator[str]:
        self._ensure_open()
        iterator_method = getattr(self._bound_collection(), "iterator", None)
        if not callable(iterator_method):
            raise RuntimeError("Weaviate collection does not provide iterator; pass explicit document IDs to prewarm")
        iterator = self._call("iterator", iterator_method, include_vector=False, return_properties=[])
        seen: set[str] = set()
        try:
            for obj in iterator:
                raw_id = _lookup(obj, "uuid", _lookup(obj, "id", None))
                if raw_id is None:
                    raise ValueError("Weaviate iterator returned an object without a UUID")
                doc_id = str(raw_id)
                if doc_id in seen:
                    raise ValueError(f"Weaviate iterator returned duplicate object UUID {doc_id!r}")
                seen.add(doc_id)
                yield doc_id
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def size(self) -> int:
        self._ensure_open()
        if self._size_cache is not None:
            return self._size_cache
        aggregate = getattr(self._bound_collection(), "aggregate", None)
        over_all = getattr(aggregate, "over_all", None) if aggregate is not None else None
        if not callable(over_all):
            return 0
        result = self._call("aggregate", over_all, total_count=True)
        value = _lookup(result, "total_count", _lookup(result, "totalCount", 0))
        try:
            self._size_cache = max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            self._size_cache = 0
        return self._size_cache

    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return {
            "backend": "weaviate", "uri": _redact_uri(self.uri), "collection": self.collection,
            "vector_name": self.vector_name or "default", "vector_names": list(self._vector_names),
            "dimension": self._collection_dimension or self.dimension,
            "dimension_verified": self._collection_dimension is not None, "metric": self.metric,
            "index_type": self._index_type or "UNKNOWN", "metric_verified": self._metric_verified is not None,
            "text_property": self.text_property or "UNKNOWN", "tenant": self.tenant or "",
            "multi_tenancy": self._multi_tenancy if self._multi_tenancy is not None else "UNKNOWN",
            "size": self.size(), "api_key_env": self.api_key_env,
            "collection_config": _safe_config(self._config, self._api_key) if self._config is not None else "UNKNOWN",
        }

    def health_check(self) -> dict[str, Any]:
        try:
            ready = getattr(self.client, "is_ready", None)
            if callable(ready) and not bool(ready()):
                raise RuntimeError("Weaviate server is not ready")
            self._load_config()
            return {"ok": True, "backend": "weaviate", "collection": self.collection,
                    "tenant": self.tenant or "", "object_count": self.size()}
        except Exception as exc:
            return {"ok": False, "backend": "weaviate", "collection": self.collection,
                    "tenant": self.tenant or "", "error": _redact_weaviate_error(exc, self._api_key)}

    def audit(self, source_dimension: int | None = None) -> dict[str, Any]:
        checks: dict[str, Any] = {
            "sdk": {"ok": True, "detail": "weaviate-client v4 loaded"},
            "credentials": {"ok": True, "detail": "configured" if self._api_key else "local/no-auth or injected client"},
            "connection": self.health_check(),
            "schema": {"ok": True, "collection": self.collection, "vector_name": self.vector_name or "default"},
            "dimension": {"configured": source_dimension or self.dimension,
                           "collection": self._collection_dimension or "UNKNOWN",
                           "verified": self._collection_dimension is not None,
                           "ok": self._collection_dimension is None or source_dimension is None or int(source_dimension) == self._collection_dimension},
            "metric": {"configured": self.metric, "index": self._metric_verified or "UNKNOWN",
                        "verified": self._metric_verified is not None,
                        "ok": self._metric_verified in {None, self.metric}},
            "tenant": {"configured": self.tenant or "", "enabled": self._multi_tenancy, "ok": True},
        }
        try:
            probe = np.zeros(self.dimension, dtype="float32"); probe[0] = 1.0
            hits = self.search(probe, 1)
            checks["candidate_query"] = {"ok": True, "matches": len(hits)}
            if self.documents is None and self.text_property and hits:
                resolved = self.fetch_documents([hits[0].document_id])
                checks["document_text"] = {"ok": hits[0].document_id in resolved, "mode": "weaviate"}
            else:
                checks["document_text"] = {"ok": True, "mode": "external" if self.documents is not None else "weaviate"}
        except Exception as exc:
            checks["candidate_query"] = {"ok": False, "error": _redact_weaviate_error(exc, self._api_key)}
            checks["document_text"] = {"ok": False, "mode": "external" if self.documents is not None else "weaviate"}
        ok = all(bool(value.get("ok", True)) for value in checks.values() if isinstance(value, Mapping))
        return {"backend": "weaviate", "collection": self.collection, "ok": ok, "checks": checks}

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


class _WeaviateDocumentMapping(Mapping[str, str]):
    def __init__(self, store: WeaviateDocumentStore) -> None:
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


class WeaviateDocumentStore:
    """Lazy text resolver over Weaviate object properties."""

    def __init__(self, index: WeaviateIndex, *, text_property: str | None = None, owns_index: bool = True):
        self.index = index
        self.text_property = text_property or index.text_property or "content"
        self.path = None
        self.documents: Mapping[str, str] = _WeaviateDocumentMapping(self)
        self._owns_index = owns_index

    def get(self, document_ids: Any) -> dict[str, str]:
        ids = [str(value) for value in document_ids]
        values = self.index.fetch_documents(ids)
        missing = [value for value in ids if value not in values]
        if missing:
            raise KeyError(f"Weaviate document text missing for IDs {missing[:5]} (property {self.text_property!r})")
        return {value: str(values[value]) for value in ids}

    def size(self) -> int:
        return self.index.size()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()


__all__ = ["WeaviateIndex", "WeaviateDocumentStore", "normalize_weaviate_metric", "_redact_weaviate_error", "_redact_uri", "_MAX_TOP_K"]
