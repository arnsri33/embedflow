from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector


class NumpyIndex(VectorIndex):
    """Tiny deterministic fallback used by the offline demo when FAISS is absent."""

    def __init__(self, vectors: np.ndarray, ids: list[str], path: str | Path | None = None,
                 metric: str = "cosine", documents: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None):
        self.vectors = np.asarray(vectors, dtype="float32"); self.ids = [str(x) for x in ids]; self.path = Path(path) if path else None
        self.metric, self.documents = str(metric).lower(), documents or {}; self._metadata = dict(metadata or {})
        if self.vectors.ndim != 2 or self.vectors.shape[0] < 1 or self.vectors.shape[0] != len(self.ids) or self.vectors.shape[1] < 1:
            raise ValueError("invalid numpy index vectors/IDs: expected a non-empty 2-D matrix aligned with IDs")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate document IDs are not allowed")
        if self.metric not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        if not np.isfinite(self.vectors).all():
            raise ValueError("index vectors must be finite")
        self.dimension = self.vectors.shape[1]

    @classmethod
    def load(cls, path: str | Path, documents: dict[str, Any] | None = None, metric: str = "cosine") -> NumpyIndex:
        try:
            with Path(path).open("rb") as handle:
                data = np.load(handle, allow_pickle=False)
                vectors, ids = data["vectors"], data["ids"].tolist()
            if not isinstance(ids, list):
                raise ValueError("embedded IDs must be a one-dimensional array")
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, EOFError) as exc:
            raise ValueError(f"invalid NumPy index artifact: {path}") from exc
        meta_path = Path(path).with_suffix(Path(path).suffix + ".meta.json")
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if not isinstance(meta, dict):
                raise ValueError("metadata must be a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid NumPy index metadata: {meta_path}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"invalid NumPy index metadata: {meta_path}")
        return cls(vectors, ids, path, metric, documents, meta)

    @classmethod
    def build(cls, vectors: np.ndarray, ids: list[str], path: str | Path | None = None,
              metric: str = "cosine", documents: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None,
              ids_path: str | Path | None = None) -> NumpyIndex:
        x = np.asarray(vectors, dtype="float32")
        metric = str(metric).lower()
        if metric not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        if x.ndim != 2 or x.shape[1] < 1 or x.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2-D matrix")
        if metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if np.any(norms <= 1e-12): raise ValueError("cosine vectors must be non-zero")
            x = x / norms
        result = cls(x, ids, path, metric, documents=documents, metadata=metadata)
        if path:
            target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle: np.savez(handle, vectors=x, ids=np.asarray(ids, dtype="U"))
            id_target = Path(ids_path) if ids_path else target.with_suffix(target.suffix + ".ids.json")
            id_target.parent.mkdir(parents=True, exist_ok=True)
            id_target.write_text(json.dumps(ids))
            if metadata:
                target.with_suffix(target.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return result

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        q = validate_query_vector(query_vector, self.dimension)
        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
            if norm <= 1e-12:
                raise ValueError("cosine query vector must be non-zero")
            q = q / norm
        scores = self.vectors @ q; order = np.lexsort((np.arange(len(scores)), -scores))[:min(k, len(self.ids))]
        return [SearchHit(self.ids[int(i)], float(scores[i]), rank) for rank, i in enumerate(order)]

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]: return {str(i): self.documents[str(i)] for i in ids if str(i) in self.documents}
    def size(self) -> int: return len(self.ids)
    def metadata(self) -> dict[str, Any]: return {"backend": "numpy_fallback", "dimension": self.dimension, "metric": self.metric, "size": self.size(), **self._metadata}
    def close(self) -> None: return None


class FaissIndex(VectorIndex):
    """FAISS wrapper supporting saved indexes and small in-process indexes."""

    def __init__(self, index: Any, ids: list[str], path: str | Path | None = None,
                 metric: str = "cosine", metadata: dict[str, Any] | None = None,
                 documents: dict[str, Any] | None = None):
        self.index = index
        self.ids = [str(x) for x in ids]
        self.path = Path(path) if path else None
        self.dimension = int(index.d)
        self.metric = metric
        self._metadata = dict(metadata or {})
        self.documents = documents or {}
        if int(index.ntotal) != len(self.ids):
            raise ValueError(f"FAISS index contains {index.ntotal} rows but {len(self.ids)} IDs were supplied")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate document IDs are not allowed")
        if self.dimension < 1:
            raise ValueError("FAISS index dimension must be positive")
        if str(metric).lower() not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        self.metric = str(metric).lower()
        self._metadata.update({"backend": "faiss", "dimension": self.dimension, "metric": self.metric,
                               "size": len(self.ids), "index_type": type(index).__name__})

    @classmethod
    def load(cls, path: str | Path, ids_path: str | Path | None = None,
             metric: str = "cosine", documents: dict[str, Any] | None = None,
             nprobe: int | None = None) -> FaissIndex:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS backend requires faiss-cpu or faiss-gpu") from exc
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            index = faiss.read_index(str(path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid FAISS index artifact: {path}") from exc
        candidates = [Path(ids_path)] if ids_path else []
        candidates += [Path(str(path) + ".ids.json"), path.with_suffix(path.suffix + ".ids.json"), path.with_suffix(".ids.json")]
        id_file = next((p for p in candidates if p.exists()), None)
        if id_file is None:
            raise FileNotFoundError(f"missing FAISS document ID sidecar; tried {[str(x) for x in candidates]}")
        if nprobe is not None and (isinstance(nprobe, bool) or int(nprobe) != nprobe or int(nprobe) < 1):
            raise ValueError("nprobe must be a positive integer")
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = min(max(1, int(nprobe)), int(getattr(index, "nlist", nprobe)))
        try:
            raw = json.loads(id_file.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid FAISS IDs sidecar: {id_file}") from exc
        ids = raw.get("ids", raw) if isinstance(raw, dict) else raw
        if not isinstance(ids, list):
            raise ValueError(f"invalid FAISS IDs sidecar: {id_file}")
        meta_file = path.with_suffix(path.suffix + ".meta.json")
        try:
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid FAISS metadata: {meta_file}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"invalid FAISS metadata: {meta_file}")
        return cls(index, [str(x) for x in ids], path=path, metric=metric, metadata=meta, documents=documents)

    @classmethod
    def build(cls, vectors: np.ndarray, ids: list[str], path: str | Path | None = None,
              metric: str = "cosine", nlist: int | None = None, nprobe: int = 64,
              documents: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None,
              ids_path: str | Path | None = None) -> FaissIndex:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS backend requires faiss-cpu or faiss-gpu") from exc
        x = np.asarray(vectors, dtype="float32")
        metric = str(metric).lower()
        if metric not in {"cosine", "dot", "inner_product"}:
            raise ValueError("metric must be cosine, dot, or inner_product")
        if isinstance(nprobe, bool) or int(nprobe) != nprobe or int(nprobe) < 1:
            raise ValueError("nprobe must be a positive integer")
        if nlist is not None and (isinstance(nlist, bool) or int(nlist) != nlist or int(nlist) < 1):
            raise ValueError("nlist must be a positive integer when provided")
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 1 or x.shape[0] != len(ids) or not np.isfinite(x).all():
            raise ValueError("vectors must be finite 2-D data aligned with unique IDs")
        if len(set(map(str, ids))) != len(ids):
            raise ValueError("duplicate document IDs are not allowed")
        if metric.lower() == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if np.any(norms <= 1e-12): raise ValueError("cosine vectors must be non-zero")
            x = x / norms
        d = x.shape[1]
        if nlist and len(x) >= max(39, nlist * 4):
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(int(nlist), len(x)), faiss.METRIC_INNER_PRODUCT)
            index.train(x)
            index.nprobe = min(int(nprobe), int(index.nlist))
        else:
            index = faiss.IndexFlatIP(d)
        index.add(x)
        result = cls(index, [str(x) for x in ids], path=path, metric=metric,
                     metadata={"nprobe": nprobe, "nlist": nlist, **(metadata or {})}, documents=documents)
        if path:
            result.save(path, ids_path=ids_path)
        return result

    def save(self, path: str | Path | None = None, ids_path: str | Path | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS backend requires faiss-cpu or faiss-gpu") from exc
        target = Path(path or self.path or "legacy.index")
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))
        id_target = Path(ids_path) if ids_path else target.with_suffix(target.suffix + ".ids.json")
        id_target.parent.mkdir(parents=True, exist_ok=True)
        id_target.write_text(json.dumps(self.ids, ensure_ascii=False))
        target.with_suffix(target.suffix + ".meta.json").write_text(json.dumps(self.metadata(), indent=2, sort_keys=True))
        self.path = target

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        value = validate_query_vector(query_vector, self.dimension)
        if self.metric == "cosine":
            norm = float(np.linalg.norm(value))
            if norm <= 1e-12:
                raise ValueError("cosine query vector must be non-zero")
            value = value / norm
        value = value[None, :]
        scores, indices = self.index.search(value, min(k, len(self.ids)))
        return [SearchHit(self.ids[int(i)], float(score), rank) for rank, (score, i) in enumerate(zip(scores[0], indices[0])) if int(i) >= 0]

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        return {str(i): self.documents[str(i)] for i in ids if str(i) in self.documents}

    def size(self) -> int:
        return len(self.ids)

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def close(self) -> None:
        # FAISS index objects are in-memory Python wrappers; there is no live
        # client to close. Keep the hook symmetric with other backends.
        return None


def load_faiss_index(path: str | Path, documents: dict[str, Any] | None = None,
                     metric: str = "cosine", nprobe: int | None = None) -> FaissIndex:
    return FaissIndex.load(path, metric=metric, documents=documents, nprobe=nprobe)
