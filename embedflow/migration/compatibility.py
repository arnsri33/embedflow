from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

K_VALUES = (10, 20, 50, 100, 200, 500)


def _research_features(records: list[tuple[str, float, int]], target_scores: dict[str, float], ks: tuple[int, ...]) -> dict[str, float]:
    """Delegate the frozen feature implementation when this checkout has it."""
    try:
        if max(ks) < 500:
            raise ImportError("the frozen helper requires the complete K<=500 feature set")
        from src.probe_features import build_features
        return build_features(records, target_scores, ks=ks)
    except (ImportError, ModuleNotFoundError):
        final_ids = [x[0] for x in _rerank(records, target_scores, max(ks))[:10]]
        top50 = [x[0] for x in _rerank(records, target_scores, min(50, len(records)))[:10]]
        top500 = [x[0] for x in _rerank(records, target_scores, min(500, len(records)))[:10]]
        stability = len(set(top50) & set(final_ids)) / max(1, len(final_ids))
        return {"probe_residual_tail_50_mean": 1 - stability, "stability_to_500_50_mean": stability,
                "deepest_p90": float(max((x[2] for x in _rerank(records, target_scores, min(500, len(records)))[:10]), default=0)),
                "late_tail_area": float(len(set(top500) - set(top50)) / 10), "last_shell_any_rate": float(bool(set(top500) - set(top50))),
                "fraction_margin_nonpositive": 0.0}


def _rerank(records: list[tuple[str, float, int]], scores: dict[str, float], k: int) -> list[tuple[str, float, int]]:
    subset = records[:min(k, len(records))]
    return sorted(subset, key=lambda x: (-float(scores[x[0]]), int(x[2])))


def _mean_features(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty feature rows")
    keys = sorted({k for r in rows for k in r})
    output: dict[str, float] = {}
    for key in keys:
        values = [float(r.get(key, 0.0)) for r in rows]
        if not np.isfinite(values).all():
            raise ValueError(f"feature {key!r} contains non-finite values")
        output[key] = float(np.mean(values))
    return output


def recommend_k(per_query: list[dict[str, Any]], kmax: int) -> tuple[int, str]:
    if isinstance(kmax, bool) or int(kmax) != kmax or int(kmax) < 10:
        raise ValueError("kmax must be an integer >= 10")
    if not per_query: return min(50, int(kmax)), "No probe rows; using the configured conservative default."
    for k in K_VALUES:
        if k > kmax: continue
        try:
            vals = [float(row.get(f"stability_{k}", row.get("stability_to_500_50_mean", 0.0))) for row in per_query]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probe stability at K={k} must be numeric") from exc
        if not np.isfinite(vals).all() or any(value < 0 or value > 1 for value in vals):
            raise ValueError(f"probe stability at K={k} must be finite and in [0, 1]")
        if vals and float(np.mean(vals)) >= 0.90:
            return k, f"Mean finite-pool top-10 stability at K={k} is {float(np.mean(vals)):.3f}."
    return min(200, int(kmax)), "No tested K reached the 0.90 finite-pool stability heuristic; review before deployment."


def run_probe(source_model: Any, target_model: Any, index: Any, documents: Any,
              queries: Iterable[tuple[str, str]], kmax: int = 500, seed: int = 42,
              limit: int | None = None,
              target_document_vectors: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Run a finite candidate compatibility probe with frozen T2-v1 decision logic."""
    if isinstance(kmax, bool) or int(kmax) != kmax or int(kmax) < 10:
        raise ValueError("probe kmax must be an integer >= 10")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or int(limit) < 1):
        raise ValueError("probe limit must be a positive integer when provided")
    rows = list(queries)
    normalized_rows: list[tuple[str, str]] = []
    seen_query_ids: set[str] = set()
    for item in rows:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("probe queries must be (query_id, text) pairs")
        query_id, text = str(item[0]), item[1]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"probe query {query_id!r} must contain non-empty text")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate probe query ID {query_id!r}")
        seen_query_ids.add(query_id)
        normalized_rows.append((query_id, text))
    rows = normalized_rows
    if not rows:
        raise ValueError("probe queries must contain at least one query")
    rng = random.Random(seed); rng.shuffle(rows)
    rows = rows[:limit] if limit else rows
    kmax = min(int(kmax), index.size())
    if kmax < 10: raise ValueError("compatibility probe needs at least 10 index candidates")
    try:
        from src.t2_v1 import verify_t2_hash
        root = Path(__file__).resolve().parents[2]
        package_root = Path(__file__).resolve().parents[1]
        if (root / "frozen").exists():
            verify_t2_hash(root)
        elif (package_root / "frozen").exists():
            verify_t2_hash(package_root)
    except (ImportError, FileNotFoundError):
        pass
    per_query = []
    for query_id, text in rows:
        sq = np.asarray(source_model.encode_queries([text])[0], dtype="float32")
        if sq.ndim != 1 or sq.shape[0] != int(source_model.dimension) or not np.isfinite(sq).all():
            raise ValueError("source query encoder returned an invalid vector")
        candidates = index.search(sq, kmax)
        ids = [x.document_id for x in candidates]
        if not ids:
            raise ValueError("source index returned no candidates")
        if target_document_vectors is None:
            docs = documents.get(ids)
            tv = np.asarray(target_model.encode_documents([docs[x] for x in ids]), dtype="float32")
        else:
            missing_vectors = [document_id for document_id in ids if document_id not in target_document_vectors]
            if missing_vectors:
                raise ValueError(f"target probe vectors are missing {len(missing_vectors)} candidate IDs (e.g. {missing_vectors[:3]})")
            tv = np.asarray([target_document_vectors[document_id] for document_id in ids], dtype="float32")
        tq = np.asarray(target_model.encode_queries([text])[0], dtype="float32")
        if tv.ndim != 2 or tv.shape != (len(ids), int(target_model.dimension)) or not np.isfinite(tv).all():
            raise ValueError("target document dimension or finiteness mismatch in probe")
        if tq.ndim != 1 or tq.shape[0] != int(target_model.dimension) or not np.isfinite(tq).all():
            raise ValueError("target query encoder returned an invalid vector")
        scores = {did: float(vec @ tq) for did, vec in zip(ids, tv)}
        features = _research_features([(x.document_id, x.score, x.source_rank) for x in candidates], scores, tuple(k for k in K_VALUES if k <= kmax))
        features["query_id"] = str(query_id)
        features["target_scores"] = scores
        # Preserve per-K finite-pool stability for K recommendation.
        final = [x[0] for x in _rerank([(x.document_id, x.score, x.source_rank) for x in candidates], scores, kmax)[:10]]
        for k in K_VALUES:
            if k <= kmax:
                got = [x[0] for x in _rerank([(x.document_id, x.score, x.source_rank) for x in candidates], scores, k)[:10]]
                features[f"stability_{k}"] = len(set(got) & set(final)) / max(1, len(final))
        per_query.append(features)
    scalar_rows = [{k: v for k, v in row.items() if isinstance(v, (int, float, np.number))} for row in per_query]
    aggregate = _mean_features(scalar_rows)
    try:
        from src.t2_v1 import decide
        diagnostic = decide(aggregate)
        implementation = "src.t2_v1.decide"
    except (ImportError, KeyError) as exc:
        raise RuntimeError("frozen T2-v1 implementation is unavailable; refusing to invent a rule") from exc
    recommended, rationale = recommend_k(per_query, kmax)
    return {"diagnostic": diagnostic, "recommended_k": recommended, "rationale": rationale,
            "features": aggregate, "per_query": per_query, "queries": len(per_query),
            "kmax": kmax, "seed": seed, "implementation": implementation,
            "warning": "T2-v1 is an empirical finite-tail diagnostic, not a mathematical guarantee."}


def save_probe(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
