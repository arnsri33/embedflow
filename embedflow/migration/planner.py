from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MigrationPlan:
    status: str
    target_model: str
    candidate_depth: int
    diagnostic: str
    ann_status: str
    corpus_size: int
    cached_target_vectors: int
    cache_fraction: float
    migration_strategy: str
    warnings: list[str]
    rationale: str = ""
    source_model: str = ""

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def make_plan(*, target_model: str, corpus_size: int, diagnostic: str,
              recommended_k: int | None, ann_status: str, cached_target_vectors: int,
              default_k: int = 50, throughput_docs_per_second: float | None = None,
              gpu_price_per_hour: float | None = None, source_model: str = "") -> MigrationPlan:
    diagnostic = str(diagnostic or "UNKNOWN").upper()
    if diagnostic not in {"SAFE", "EXPAND", "UNSAFE_OR_UNCERTAIN", "UNKNOWN"}:
        raise ValueError(f"unsupported diagnostic: {diagnostic}")
    ann_value = str(ann_status or "UNKNOWN").upper()
    if ann_value not in {"PASS", "WARNING", "UNKNOWN"}:
        raise ValueError(f"unsupported ANN status: {ann_status}")
    try:
        corpus_int = int(corpus_size)
        corpus_exact = float(corpus_size) == corpus_int
        cached_int = int(cached_target_vectors)
        cached_exact = float(cached_target_vectors) == cached_int
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("corpus_size and cached_target_vectors must be integers") from exc
    if isinstance(corpus_size, bool) or not corpus_exact or corpus_int < 0:
        raise ValueError("corpus_size must be a non-negative integer")
    if isinstance(cached_target_vectors, bool) or not cached_exact or cached_int < 0:
        raise ValueError("cached_target_vectors must be a non-negative integer")
    # ``candidate_depth: auto`` is a configuration convenience.  The actual
    # serving plan always resolves to an integer, using the probe recommendation
    # when available and the conservative default otherwise.
    if str(default_k).lower() == "auto":
        fallback_k = 50
    else:
        try:
            fallback_k = int(default_k)
            if isinstance(default_k, bool) or float(default_k) != fallback_k or fallback_k < 1:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("default_k must be auto or a positive integer") from exc
    raw_k = fallback_k if recommended_k is None else recommended_k
    try:
        k = int(raw_k)
        exact_k = float(raw_k) == k
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("recommended_k must be a positive integer") from exc
    if isinstance(raw_k, bool) or not exact_k or k < 1:
        raise ValueError("recommended_k must be a positive integer")
    warnings: list[str] = []
    if diagnostic != "SAFE":
        warnings.append("Compatibility is not a SAFE T2-v1 diagnostic; validate manually before relying on progressive migration.")
    if ann_value in {"WARNING", "UNKNOWN"}:
        warnings.append("ANN fidelity is not confirmed; T2-v1 does not diagnose ANN approximation error.")
    if cached_int > corpus_int:
        warnings.append("Cache count exceeds corpus size; check document identity and cache contract.")
    fraction = (cached_int / corpus_int) if corpus_int else 0.0
    strategy = "PROGRESSIVE" if diagnostic == "SAFE" else "PROGRESSIVE_WITH_REVIEW"
    status = "READY" if diagnostic == "SAFE" and ann_value in {"PASS", "UNKNOWN"} else "REVIEW"
    rationale = "Finite-pool diagnostic supports the selected starting K; this is an empirical recommendation, not a guarantee." if diagnostic == "SAFE" else "Use source-only/full-backfill fallback until compatibility and ANN health are reviewed."
    return MigrationPlan(status, target_model, k, diagnostic, ann_value, corpus_int,
                         cached_int, float(fraction), strategy, warnings, rationale, source_model)
