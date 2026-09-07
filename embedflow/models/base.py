from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingModel(ABC):
    model_id: str
    dimension: int
    fingerprint: str

    @abstractmethod
    def encode_queries(self, texts: list[str]):
        raise NotImplementedError

    @abstractmethod
    def encode_documents(self, texts: list[str], batch_size: int | None = None):
        raise NotImplementedError

    def encode_query(self, text: str):
        return self.encode_queries([text])[0]

    def encode_document(self, text: str):
        return self.encode_documents([text])[0]

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "dimension": self.dimension, "fingerprint": self.fingerprint}
