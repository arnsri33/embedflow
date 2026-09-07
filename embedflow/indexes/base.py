from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float
    source_rank: int


class VectorIndex(ABC):
    dimension: int
    metric: str

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources.

        In-memory/FAISS indexes do not currently own a live resource, so the
        default is a no-op. Backends such as local Qdrant use this hook to
        release file locks and network clients when a serving engine stops.
        """
        return None


def validate_query_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    value = np.asarray(vector, dtype="float32")
    if value.ndim != 1 or value.shape[0] != int(dimension):
        raise ValueError(f"query vector dimension {value.shape} does not match index dimension {dimension}")
    if not np.isfinite(value).all():
        raise ValueError("query vector contains non-finite values")
    return value


def validate_k(k: int) -> int:
    """Validate a requested top-k before it reaches a backend API."""
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    return int(k)
