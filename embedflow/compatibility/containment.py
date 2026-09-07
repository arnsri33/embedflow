from __future__ import annotations

from collections.abc import Sequence


def candidate_containment(source_ranked: Sequence[str], target_ranked: Sequence[str], k: int = 10) -> float:
    """Fraction of the target top-k that appears in the source candidate list."""
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    target = set(map(str, target_ranked[:k]))
    return float(len(target & set(map(str, source_ranked))) / len(target)) if target else 0.0


def containment_at_k(source_ranked: Sequence[str], target_ranked: Sequence[str], k: int = 10) -> float:
    """Alias retained for reports and downstream notebooks."""
    return candidate_containment(source_ranked, target_ranked, k=k)
