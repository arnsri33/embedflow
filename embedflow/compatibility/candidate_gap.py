from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .containment import candidate_containment
from .metrics import ndcg, paired_bootstrap
from .migration_depth import observed_migration_depth


@dataclass(frozen=True)
class CandidateGapCurve:
    """One point on an evaluated target-quality curve."""

    k: int
    native_target_quality: float
    restricted_target_quality: float
    candidate_gap: float
    containment: float
    queries: int
    gap_ci_low: float | None = None
    gap_ci_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_candidate_gap_curve(
    *,
    source_rankings: Mapping[str, Sequence[str]],
    target_scores: Mapping[str, Mapping[str, float]],
    native_target_rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k_values: Sequence[int] = (10, 20, 50, 100, 200, 500),
    quality_k: int = 10,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> list[CandidateGapCurve]:
    """Compute G(K) and containment from native target evaluation artifacts.

    ``target_scores`` must contain target scores for source candidates only;
    this function never treats containment as candidate gap.  A native target
    ranking is required to establish M_T.
    """
    if not k_values:
        raise ValueError("k_values must contain at least one positive depth")
    normalized_k = []
    seen_k: set[int] = set()
    for value in k_values:
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("k_values must contain positive integers")
        depth = int(value)
        if depth in seen_k:
            raise ValueError(f"duplicate candidate depth K={depth}")
        seen_k.add(depth); normalized_k.append(depth)
    if isinstance(quality_k, bool) or int(quality_k) != quality_k or int(quality_k) < 1:
        raise ValueError("quality_k must be a positive integer")
    query_ids = [str(q) for q in source_rankings if str(q) in target_scores and str(q) in native_target_rankings and str(q) in qrels]
    if not query_ids:
        raise ValueError("no query IDs overlap source rankings, target scores, native rankings, and qrels")
    points: list[CandidateGapCurve] = []
    native_values = {qid: ndcg(native_target_rankings[qid], qrels[qid], quality_k) for qid in query_ids}
    for raw_k in sorted(normalized_k):
        native_quality = [native_values[qid] for qid in query_ids]
        restricted_quality: list[float] = []
        contains: list[float] = []
        gaps: list[float] = []
        for qid in query_ids:
            source = [str(doc_id) for doc_id in source_rankings[qid]]
            score_map = target_scores[qid]
            candidates = source[:raw_k]
            positions: dict[str, int] = {}
            for position, doc_id in enumerate(source):
                positions.setdefault(doc_id, position)
            missing = [doc_id for doc_id in candidates if doc_id not in score_map]
            if missing:
                raise ValueError(f"target_scores missing {len(missing)} source candidates for query {qid!r}")
            if any(not np.isfinite(float(score_map[doc_id])) for doc_id in candidates):
                raise ValueError(f"target_scores contains non-finite score for query {qid!r}")
            ranked = sorted(candidates, key=lambda doc_id: (-float(score_map[doc_id]), positions[doc_id]))
            restricted_quality.append(ndcg(ranked, qrels[qid], quality_k))
            contains.append(candidate_containment(candidates, native_target_rankings[qid], quality_k))
            gaps.append(native_values[qid] - restricted_quality[-1])
        mean_gap = float(np.mean(gaps))
        low = high = None
        if bootstrap_resamples:
            _, low, high = paired_bootstrap(gaps, seed=seed, resamples=bootstrap_resamples)
        points.append(CandidateGapCurve(
            k=raw_k,
            native_target_quality=float(np.mean(native_quality)),
            restricted_target_quality=float(np.mean(restricted_quality)),
            candidate_gap=mean_gap,
            containment=float(np.mean(contains)),
            queries=len(query_ids),
            gap_ci_low=low,
            gap_ci_high=high,
        ))
    return points


def observed_k_epsilon(curve: Sequence[CandidateGapCurve], epsilon: float = 0.01) -> int | None:
    return observed_migration_depth([point.to_dict() for point in curve], epsilon=epsilon)
