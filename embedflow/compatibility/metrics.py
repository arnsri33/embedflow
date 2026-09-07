from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def ndcg(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """Compute nDCG@k for one query using graded relevance labels."""
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(qrels, Mapping):
        raise TypeError("qrels must be a mapping")
    clean_qrels: dict[str, int] = {}
    for doc_id, relevance in qrels.items():
        try:
            numeric = float(relevance)
        except (TypeError, ValueError) as exc:
            raise ValueError("qrel relevance values must be finite numbers") from exc
        if not math.isfinite(numeric) or numeric < 0 or int(numeric) != numeric:
            raise ValueError("qrel relevance values must be finite non-negative integers")
        clean_qrels[str(doc_id)] = int(numeric)
    # Retrieval backends should return unique IDs, but malformed sidecars and
    # user-provided rankings do occur.  Score only the first occurrence so a
    # duplicate cannot inflate DCG above the ideal ranking.
    ranked = []
    seen: set[str] = set()
    for value in ranked_ids:
        document_id = str(value)
        if document_id in seen:
            continue
        seen.add(document_id)
        ranked.append(document_id)
        if len(ranked) >= int(k):
            break
    values = [clean_qrels.get(x, 0) for x in ranked]
    dcg = sum((2**value - 1) / np.log2(position + 2) for position, value in enumerate(values))
    ideal = sorted(clean_qrels.values(), reverse=True)[: int(k)]
    idcg = sum((2**value - 1) / np.log2(position + 2) for position, value in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def recall(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int = 100) -> float:
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(qrels, Mapping):
        raise TypeError("qrels must be a mapping")
    positives = set()
    for doc_id, relevance in qrels.items():
        if isinstance(relevance, bool):
            raise ValueError("qrel relevance values must be finite non-negative numbers")
        try:
            numeric = float(relevance)
        except (TypeError, ValueError) as exc:
            raise ValueError("qrel relevance values must be finite non-negative numbers") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("qrel relevance values must be finite non-negative numbers")
        if numeric > 0:
            positives.add(str(doc_id))
    return float(len(set(map(str, ranked_ids[: int(k)])) & positives) / len(positives)) if positives else 0.0


def paired_bootstrap(values: Sequence[float], seed: int = 42, resamples: int = 10_000) -> tuple[float, float, float]:
    """Return mean and a percentile bootstrap interval for paired query values."""
    values = np.asarray(list(values), dtype="float64")
    if values.size == 0:
        raise ValueError("bootstrap requires at least one value")
    if isinstance(resamples, bool) or int(resamples) != resamples or int(resamples) < 1:
        raise ValueError("resamples must be a positive integer")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, values.size, size=(int(resamples), values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
