from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TargetVectorCache(ABC):
    @abstractmethod
    def get(self, document_ids: list[str], content_fingerprints: Any = None) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def put(self, document_ids: list[str], vectors: np.ndarray, content_fingerprints: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, document_ids: list[str], content_fingerprints: Any = None) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
