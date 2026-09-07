from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector


class QdrantIndex(VectorIndex):
    """Qdrant adapter. The dependency is optional and imported lazily."""

    def __init__(self, client: Any, collection: str, dimension: int,
                 documents: dict[str, Any] | None = None, metric: str = "cosine",
                 vector_name: str | None = None):
        try:
            dimension_value = int(dimension)
            dimension_exact = float(dimension) == dimension_value
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Qdrant vector dimension must be a positive integer") from exc
        if isinstance(dimension, bool) or not dimension_exact or dimension_value < 1:
            raise ValueError("Qdrant vector dimension must be a positive integer")
        metric = str(metric).lower()
        if metric not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        if not str(collection).strip():
            raise ValueError("Qdrant collection must be non-empty")
        self.client, self.collection, self.dimension = client, str(collection), dimension_value
        self.documents, self.metric, self.vector_name = documents or {}, metric, vector_name
        # Qdrant accepts integer IDs or UUIDs, while document stores commonly
        # use IDs such as ``doc-17``.  Keep a deterministic mapping for rows we
        # write so arbitrary application IDs round-trip through search/retrieve.
        self._id_map: dict[str, str] = {}
        self._closed = False

    @staticmethod
    def _storage_id(document_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"embedflow:{document_id}"))

    @classmethod
    def connect(cls, path_or_url: str, collection: str, dimension: int,
                documents: dict[str, Any] | None = None, metric: str = "cosine",
                api_key_env: str | None = "QDRANT_API_KEY", vector_name: str | None = None) -> QdrantIndex:
        try:
            dimension_value = int(dimension)
            dimension_exact = float(dimension) == dimension_value
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Qdrant vector dimension must be a non-negative integer") from exc
        if isinstance(dimension, bool) or not dimension_exact or dimension_value < 0:
            raise ValueError("Qdrant vector dimension must be a non-negative integer")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Qdrant backend requires qdrant-client; install it with `pip install qdrant-client`") from exc
        path_or_url = str(path_or_url)
        if "://" not in path_or_url and not path_or_url.startswith("http"):
            client = QdrantClient(path=path_or_url)
        else:
            api_key = os.environ.get(api_key_env) if api_key_env else None
            client = QdrantClient(url=path_or_url, api_key=api_key)
        if dimension_value == 0:
            try:
                info = client.get_collection(collection)
                vectors_config = getattr(getattr(info, "config", None), "params", None)
                vectors_config = getattr(vectors_config, "vectors", None)
                if isinstance(vectors_config, dict):
                    vectors_config = vectors_config.get(vector_name) if vector_name else next(iter(vectors_config.values()), None)
                dimension = int(getattr(vectors_config, "size", 0) or 0)
                if dimension <= 0:
                    raise ValueError("could not infer Qdrant collection vector dimension; set source.dimension")
            except Exception:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                raise
        return cls(client, collection, dimension, documents, metric, vector_name)

    @classmethod
    def build(cls, path_or_url: str, collection: str, vectors: np.ndarray, ids: list[str],
              documents: dict[str, Any] | None = None, metric: str = "cosine",
              api_key_env: str | None = "QDRANT_API_KEY", vector_name: str | None = None) -> QdrantIndex:
        values = np.asarray(vectors, dtype="float32")
        metric = str(metric).lower()
        if metric not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 1 or values.shape[0] != len(ids):
            raise ValueError("vectors must be a non-empty 2-D matrix aligned with IDs")
        if len(set(map(str, ids))) != len(ids):
            raise ValueError("duplicate document IDs are not allowed")
        if not np.isfinite(values).all():
            raise ValueError("vectors must be finite")
        if metric == "cosine" and np.any(np.linalg.norm(values, axis=1) <= 1e-12):
            raise ValueError("cosine vectors must be non-zero")
        result = cls.connect(path_or_url, collection, int(values.shape[1]), documents, metric,
                             api_key_env=api_key_env, vector_name=vector_name)
        try:
            from qdrant_client.models import Distance, VectorParams
            distance = Distance.COSINE if metric == "cosine" else Distance.DOT
            vectors_config = (VectorParams(size=result.dimension, distance=distance) if not vector_name
                              else {vector_name: VectorParams(size=result.dimension, distance=distance)})
            if hasattr(result.client, "collection_exists"):
                if result.client.collection_exists(collection):
                    result.client.delete_collection(collection)
                result.client.create_collection(collection_name=collection, vectors_config=vectors_config)
            else:  # qdrant-client versions before collection_exists
                result.client.recreate_collection(collection_name=collection, vectors_config=vectors_config)
            payloads = [{"text": documents[str(i)]} for i in ids] if documents else None
            result.upsert(ids, values, payloads)
        except ImportError as exc:
            result.close()
            raise RuntimeError("qdrant-client is required") from exc
        except Exception:
            result.close()
            raise
        return result

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if self._closed:
            raise RuntimeError("Qdrant index is closed")
        k = validate_k(k)
        value = validate_query_vector(query_vector, self.dimension).tolist()
        # qdrant-client renamed search -> query_points; support both APIs.
        if hasattr(self.client, "query_points"):
            kwargs = {"collection_name": self.collection, "query": value, "limit": k}
            if self.vector_name:
                kwargs["using"] = self.vector_name
            result = self.client.query_points(**kwargs).points
        else:
            kwargs = {"collection_name": self.collection, "query_vector": value, "limit": k}
            if self.vector_name:
                try:
                    result = self.client.search(**kwargs, using=self.vector_name)
                except TypeError:
                    result = self.client.search(collection_name=self.collection,
                                                query_vector=(self.vector_name, value), limit=k)
            else:
                result = self.client.search(**kwargs)
        hits = []
        for rank, p in enumerate(result):
            payload = p.payload or {}
            document_id = str(payload.get("_embedflow_document_id", p.id))
            self._id_map[document_id] = str(p.id)
            hits.append(SearchHit(document_id, float(p.score), rank))
        return hits

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Qdrant index is closed")
        if self.documents:
            return {str(i): self.documents[str(i)] for i in ids if str(i) in self.documents}
        if not ids:
            return {}
        storage_ids = [self._id_map.get(str(x), self._storage_id(str(x))) for x in ids]
        points = self.client.retrieve(collection_name=self.collection, ids=storage_ids, with_payload=True)
        output = {}
        for p in points:
            payload = p.payload or {}
            document_id = str(payload.get("_embedflow_document_id", p.id))
            output[document_id] = payload.get("text", payload)
        return output

    def size(self) -> int:
        if self._closed:
            raise RuntimeError("Qdrant index is closed")
        info = self.client.get_collection(self.collection)
        return int(getattr(info, "points_count", 0) or 0)

    def metadata(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Qdrant index is closed")
        return {"backend": "qdrant", "collection": self.collection, "dimension": self.dimension,
                "metric": self.metric, "vector_name": self.vector_name, "size": self.size()}

    def close(self) -> None:
        """Release a local file lock or remote client connection.

        ``QdrantClient.close`` is available in current qdrant-client releases;
        older clients simply have no close method, so teardown remains safe.
        The method is idempotent to support engine and application shutdown
        paths both calling it defensively.
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]] | None = None) -> None:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required") from exc
        ids = [str(value) for value in ids]
        values = np.asarray(vectors, dtype="float32")
        if values.ndim != 2 or values.shape != (len(ids), self.dimension):
            raise ValueError(f"Qdrant vectors must have shape ({len(ids)}, {self.dimension})")
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Qdrant upsert requires non-empty unique document IDs")
        if not np.isfinite(values).all():
            raise ValueError("Qdrant vectors must be finite")
        if self.metric == "cosine" and np.any(np.linalg.norm(values, axis=1) <= 1e-12):
            raise ValueError("Qdrant cosine vectors must be non-zero")
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError("Qdrant payload count must match document IDs")
        payloads = payloads or [{} for _ in ids]
        storage_ids = []
        normalized_payloads = []
        for document_id, payload in zip(ids, payloads):
            original_id = str(document_id)
            storage_id = self._storage_id(original_id)
            self._id_map[original_id] = storage_id
            storage_ids.append(storage_id)
            if payload is not None and not isinstance(payload, dict):
                raise ValueError("Qdrant payloads must be objects")
            item = dict(payload or {})
            item.setdefault("_embedflow_document_id", original_id)
            normalized_payloads.append(item)
        self.client.upsert(collection_name=self.collection,
                           points=[PointStruct(id=storage_id,
                                               vector=({self.vector_name: v.tolist()} if self.vector_name else v.tolist()),
                                               payload=payload)
                                   for storage_id, v, payload in zip(storage_ids, values, normalized_payloads)])
