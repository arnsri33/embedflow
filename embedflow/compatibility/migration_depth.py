from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def observed_migration_depth(gap_curve: Sequence[Mapping[str, Any]], epsilon: float = 0.01) -> int | None:
    """Return observed K*_epsilon from an evaluated G(K) curve.

    This is only valid when the curve was computed against qrels/native target
    retrieval.  It must not be used for the no-target-index T2-v1 workflow.
    """
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be non-negative") from exc
    if not math.isfinite(epsilon_value) or epsilon_value < 0:
        raise ValueError("epsilon must be non-negative")
    seen: set[int] = set()
    normalized = []
    for row in gap_curve:
        try:
            k = int(row["k"])
            gap = float(row["candidate_gap"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("gap curve rows require numeric k and candidate_gap") from exc
        try:
            k_exact = float(row["k"]) == k
        except (TypeError, ValueError, OverflowError):
            k_exact = False
        if not k_exact or k < 1 or not math.isfinite(gap):
            raise ValueError("gap curve k must be positive and candidate_gap finite")
        if k in seen:
            raise ValueError(f"duplicate candidate depth K={k}")
        seen.add(k); normalized.append((k, gap))
    for k, gap in sorted(normalized):
        if gap <= epsilon_value:
            return k
    return None


def recommend_initial_k(probe_result: Mapping[str, Any], *, default: int = 50) -> tuple[int, str]:
    """Name the deployment recommendation without calling it K*.

    T2-v1 has no native target metric and therefore cannot identify an
    observed migration depth.  The returned label intentionally says
    ``recommended_initial_k`` rather than ``K*``.
    """
    if isinstance(default, bool) or int(default) != default or int(default) < 1:
        raise ValueError("default candidate depth must be a positive integer")
    if not isinstance(probe_result, Mapping):
        raise TypeError("probe_result must be a mapping")
    diagnostic = str(probe_result.get("diagnostic", "UNSAFE_OR_UNCERTAIN")).upper()
    raw_suggested = default if probe_result.get("recommended_k") is None else probe_result.get("recommended_k")
    try:
        suggested = int(raw_suggested)
        suggested_exact = float(raw_suggested) == suggested
    except (TypeError, ValueError, OverflowError):
        suggested, suggested_exact = 0, False
    if not suggested_exact or suggested < 1:
        raise ValueError("recommended candidate depth must be positive")
    if diagnostic == "SAFE":
        return suggested, "Finite-tail behavior supports this starting K; validate on production traffic."
    if diagnostic == "EXPAND":
        return max(suggested, int(default)), "Expand the source candidate depth and perform manual validation."
    return max(suggested, int(default)), "Evidence is insufficient; prefer manual validation or a full target backfill."
