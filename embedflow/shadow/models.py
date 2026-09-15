"""Small serializable models used by Shadow Mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Failure categories are part of the public, privacy-conscious telemetry
# contract.  Keep this allow-list deliberately closed: a third-party model or
# document store must not be able to smuggle arbitrary exception text (which
# may contain query/document data) into a persisted category field.
SAFE_FAILURE_CATEGORIES = frozenset({
    "SOURCE_CANDIDATE_ERROR",
    "TARGET_QUERY_ENCODING_ERROR",
    "DOCUMENT_RESOLUTION_ERROR",
    "TARGET_DOCUMENT_ENCODING_ERROR",
    "RERANK_ERROR",
    "CACHE_ERROR",
    "MATERIALIZATION_ERROR",
    "TIMEOUT",
    "INTERNAL",
})


@dataclass(frozen=True)
class ShadowObservation:
    """One privacy-conscious shadow execution outcome.

    Query/document text and vector payloads deliberately do not belong in this
    model.  The runner records only aggregate diagnostics and opaque request
    identifiers (when per-request retention is explicitly enabled).
    """

    status: str
    config_fingerprint: str
    request_id: str | None = None
    source_latency_ms: float | None = None
    shadow_latency_ms: float | None = None
    source_candidate_latency_ms: float | None = None
    target_query_encode_latency_ms: float | None = None
    target_rerank_latency_ms: float | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    target_candidates_available: int = 0
    target_candidates_missing: int = 0
    target_coverage: float | None = None
    top1_agreement: float | None = None
    top_k_overlap: float | None = None
    docs_queued: int = 0
    unique_docs_queued: int = 0
    docs_materialized: int = 0
    docs_failed: int = 0
    # Candidate IDs are retained only transiently so the telemetry writer can
    # update aggregate popularity counters. They are never written to the
    # per-observation table and are omitted from public report rendering.
    candidate_ids: tuple[str, ...] = ()
    missing_candidate_ids: tuple[str, ...] = ()
    failure_category: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["SAFE_FAILURE_CATEGORIES", "ShadowObservation"]
