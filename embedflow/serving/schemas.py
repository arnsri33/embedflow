"""Pydantic request/response models for the public REST API."""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - imported only when API extras are used
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[misc]
        return default


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_depth: int | None = Field(default=None, ge=1)
    max_sync_misses: int | None = Field(default=None, ge=0)


class PrewarmRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    asynchronous: bool = True


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    migration: dict[str, Any]
    timing_ms: dict[str, float]
    # Flattened aliases make the response convenient for clients while the
    # nested ``migration`` object remains for backwards compatibility.
    state: str | None = None
    candidate_depth: int | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    sync_encoded: int | None = None
    async_queued: int | None = None
