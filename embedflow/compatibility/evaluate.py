from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..indexes import FaissIndex, NumpyIndex, QdrantIndex
from ..migration.state import DocumentStore
from ..models import EmbeddingModel
from .candidate_gap import compute_candidate_gap_curve
from .containment import candidate_containment
from .metrics import ndcg, recall
from .report import migration_report


def load_queries(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid query JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"query row {number} in {path} must be a JSON object")
            query_id = str(row.get("id", row.get("query_id", number)))
            text = row.get("text", row.get("query"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"query {query_id!r} has no text")
            if query_id in seen:
                raise ValueError(f"duplicate query ID {query_id!r} in {path}")
            seen.add(query_id)
            rows.append((query_id, text))
    if not rows:
        raise ValueError(f"query file is empty: {path}")
    return rows


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"qrels is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("qrels must be a JSON object mapping query IDs to document relevance maps")
    output: dict[str, dict[str, int]] = {}
    for query_id, labels in raw.items():
        if not isinstance(labels, Mapping):
            raise ValueError(f"qrels for query {query_id!r} must be an object")
        cleaned: dict[str, int] = {}
        for document_id, value in labels.items():
            if isinstance(value, bool):
                raise ValueError("qrel relevance values must be finite non-negative integers")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("qrel relevance values must be finite non-negative integers") from exc
            if not math.isfinite(numeric) or numeric < 0 or int(numeric) != numeric:
                raise ValueError("qrel relevance values must be finite non-negative integers")
            cleaned[str(document_id)] = int(numeric)
        output[str(query_id)] = cleaned
    return output


def _load_index(path: str | Path, backend: str, metric: str, documents: dict[str, str], dimension: int | None = None):
    if backend == "faiss":
        try:
            return FaissIndex.load(path, metric=metric, documents=documents)
        except RuntimeError as exc:
            try:
                return NumpyIndex.load(path, metric=metric, documents=documents)
            except Exception as fallback_exc:
                raise exc from fallback_exc
    if backend == "qdrant":
        return QdrantIndex.connect(str(path), "embedflow", int(dimension or 0), documents=documents, metric=metric)
    raise ValueError(f"unsupported index backend: {backend}")


def _rank_scores(scores: Mapping[str, float], *, tie_order: Sequence[str]) -> list[str]:
    return sorted((str(doc_id) for doc_id in scores), key=lambda doc_id: (-float(scores[doc_id]), tie_order.index(doc_id) if doc_id in tie_order else len(tie_order)))


def ann_fidelity(source_rankings: Mapping[str, Sequence[str]], reference_rankings: Mapping[str, Sequence[str]], k: int = 100) -> dict[str, Any]:
    """Compare source ANN rankings with an exact/reference source ranking."""
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("ANN audit k must be a positive integer")
    values = []
    for query_id, ranking in source_rankings.items():
        reference = reference_rankings.get(query_id)
        if reference is None:
            continue
        values.append(candidate_containment(ranking, reference, k=k))
    if not values:
        return {"status": "UNKNOWN", "queries": 0, "note": "No overlapping source/reference rankings were supplied."}
    mean = float(np.mean(values))
    return {
        "status": "PASS" if mean >= 0.95 else "WARNING",
        "queries": len(values),
        "k": int(k),
        "mean_overlap": mean,
        "note": "Overlap with the supplied exact/reference source retrieval; this is not a T2-v1 measurement.",
    }


def evaluate_rankings(
    *,
    source_rankings: Mapping[str, Sequence[str]],
    native_target_rankings: Mapping[str, Sequence[str]],
    target_scores: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    source_model: str,
    target_model: str,
    corpus_size: int,
    k_values: Sequence[int] = (10, 20, 50, 100, 200, 500),
    quality_k: int = 10,
    epsilon: float = 0.01,
    bootstrap_resamples: int = 0,
    seed: int = 42,
    diagnostic: str | None = None,
    recommended_k: int | None = None,
    ann_status: str = "UNKNOWN",
) -> dict[str, Any]:
    if not k_values:
        raise ValueError("k_values must contain at least one positive depth")
    if len(set(k_values)) != len(k_values) or any(isinstance(k, bool) or int(k) != k or int(k) < 1 for k in k_values):
        raise ValueError("k_values must contain unique positive integers")
    if isinstance(quality_k, bool) or int(quality_k) != quality_k or int(quality_k) < 1:
        raise ValueError("quality_k must be a positive integer")
    curve = compute_candidate_gap_curve(
        source_rankings=source_rankings,
        target_scores=target_scores,
        native_target_rankings=native_target_rankings,
        qrels=qrels,
        k_values=k_values,
        quality_k=quality_k,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    shared = [qid for qid in source_rankings if qid in qrels and qid in native_target_rankings]
    source_quality = [ndcg(source_rankings[qid], qrels[qid], quality_k) for qid in shared]
    source_recall = [recall(source_rankings[qid], qrels[qid], max(k_values)) for qid in shared]
    native_quality = [ndcg(native_target_rankings[qid], qrels[qid], quality_k) for qid in shared]
    result = migration_report(
        source_model=source_model,
        target_model=target_model,
        corpus_size=corpus_size,
        curve=curve,
        diagnostic=diagnostic,
        recommended_k=recommended_k,
        ann_status=ann_status,
        epsilon=epsilon,
        native_target_index_used=True,
    )
    result["source_quality"] = {"ndcg_at_k": float(np.mean(source_quality)) if source_quality else 0.0,
                                 "recall_at_k": float(np.mean(source_recall)) if source_recall else 0.0,
                                 "k": max(k_values)}
    result["native_target_quality"] = {"ndcg_at_k": float(np.mean(native_quality)) if native_quality else 0.0, "k": quality_k}
    result["queries_evaluated"] = len(shared)
    result["bootstrap_resamples"] = int(bootstrap_resamples)
    return result


def evaluate_models(
    *,
    source_model: EmbeddingModel,
    target_model: EmbeddingModel,
    source_index: Any,
    documents: DocumentStore,
    queries: Iterable[tuple[str, str]],
    qrels: Mapping[str, Mapping[str, int]],
    native_target_index: Any | None = None,
    reference_source_index: Any | None = None,
    k_values: Sequence[int] = (10, 20, 50, 100, 200, 500),
    quality_k: int = 10,
    epsilon: float = 0.01,
    bootstrap_resamples: int = 0,
    seed: int = 42,
    diagnostic: str | None = None,
    recommended_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate Mode A using qrels and optional native/reference indexes.

    If a native target index is omitted, the target model encodes the corpus
    in-process to establish M_T.  That is correct but potentially expensive;
    production-scale users should provide a native target index or saved
    rankings instead.
    """
    queries = [(str(qid), str(text)) for qid, text in queries if str(qid) in qrels]
    if not queries:
        raise ValueError("no evaluation queries overlap qrels")
    if source_index.size() < 1:
        raise ValueError("source index is empty")
    if not k_values or any(isinstance(k, bool) or int(k) != k or int(k) < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    max_k = min(max(int(k) for k in k_values), source_index.size())
    source_rankings: dict[str, list[str]] = {}
    source_vectors: dict[str, np.ndarray] = {}
    target_query_vectors: dict[str, np.ndarray] = {}
    for query_id, text in queries:
        source_vector = np.asarray(source_model.encode_query(text), dtype="float32")
        source_vectors[query_id] = source_vector
        source_rankings[query_id] = [hit.document_id for hit in source_index.search(source_vector, max_k)]
        target_query_vectors[query_id] = np.asarray(target_model.encode_query(text), dtype="float32")

    candidate_ids = list(dict.fromkeys(doc_id for ids in source_rankings.values() for doc_id in ids))
    candidate_texts = documents.get(candidate_ids)
    candidate_vectors = target_model.encode_documents([candidate_texts[doc_id] for doc_id in candidate_ids], batch_size=32)
    candidate_vectors = np.asarray(candidate_vectors, dtype="float32")
    target_scores = {
        query_id: {doc_id: float(candidate_vectors[index] @ target_query_vectors[query_id]) for index, doc_id in enumerate(candidate_ids)}
        for query_id in source_rankings
    }

    native_target_rankings: dict[str, list[str]] = {}
    if native_target_index is not None:
        for query_id, _ in queries:
            native_target_rankings[query_id] = [hit.document_id for hit in native_target_index.search(target_query_vectors[query_id], documents.size())]
    else:
        all_ids = list(documents.documents)
        all_texts = documents.get(all_ids)
        all_vectors = np.asarray(target_model.encode_documents([all_texts[doc_id] for doc_id in all_ids], batch_size=32), dtype="float32")
        for query_id, _ in queries:
            scores = all_vectors @ target_query_vectors[query_id]
            order = np.lexsort((np.arange(len(all_ids)), -scores))
            native_target_rankings[query_id] = [all_ids[int(index)] for index in order]

    ann_status = "UNKNOWN"
    if reference_source_index is not None:
        reference_rankings = {query_id: [hit.document_id for hit in reference_source_index.search(source_vectors[query_id], max_k)] for query_id, _ in queries}
        ann_status = ann_fidelity(source_rankings, reference_rankings, k=max_k)["status"]
    return evaluate_rankings(
        source_rankings=source_rankings,
        native_target_rankings=native_target_rankings,
        target_scores=target_scores,
        qrels=qrels,
        source_model=source_model.model_id,
        target_model=target_model.model_id,
        corpus_size=documents.size(),
        k_values=k_values,
        quality_k=quality_k,
        epsilon=epsilon,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        diagnostic=diagnostic,
        recommended_k=recommended_k,
        ann_status=ann_status,
    )


def evaluate_with_native_rankings(
    *,
    source_model: EmbeddingModel,
    target_model: EmbeddingModel,
    source_index: Any,
    documents: DocumentStore,
    queries: Iterable[tuple[str, str]],
    native_target_rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    reference_source_index: Any | None = None,
    k_values: Sequence[int] = (10, 20, 50, 100, 200, 500),
    quality_k: int = 10,
    epsilon: float = 0.01,
    bootstrap_resamples: int = 0,
    seed: int = 42,
    diagnostic: str | None = None,
    recommended_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate against saved native target rankings without a target index."""
    queries = [(str(qid), str(text)) for qid, text in queries if str(qid) in qrels and str(qid) in native_target_rankings]
    if not queries:
        raise ValueError("no evaluation queries overlap qrels and native target rankings")
    if source_index.size() < 1:
        raise ValueError("source index is empty")
    if not k_values or any(isinstance(k, bool) or int(k) != k or int(k) < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    max_k = min(max(int(k) for k in k_values), source_index.size())
    source_rankings: dict[str, list[str]] = {}
    target_query_vectors: dict[str, np.ndarray] = {}
    for query_id, text in queries:
        source_rankings[query_id] = [hit.document_id for hit in source_index.search(source_model.encode_query(text), max_k)]
        target_query_vectors[query_id] = np.asarray(target_model.encode_query(text), dtype="float32")
    candidate_ids = list(dict.fromkeys(doc_id for ids in source_rankings.values() for doc_id in ids))
    candidate_texts = documents.get(candidate_ids)
    vectors = np.asarray(target_model.encode_documents([candidate_texts[doc_id] for doc_id in candidate_ids], batch_size=32), dtype="float32")
    target_scores = {qid: {doc_id: float(vectors[i] @ target_query_vectors[qid]) for i, doc_id in enumerate(candidate_ids)} for qid, _ in queries}
    ann_status = "UNKNOWN"
    if reference_source_index is not None:
        reference = {qid: [hit.document_id for hit in reference_source_index.search(source_model.encode_query(text), max_k)] for qid, text in queries}
        ann_status = ann_fidelity(source_rankings, reference, k=max_k)["status"]
    return evaluate_rankings(
        source_rankings=source_rankings,
        native_target_rankings={qid: list(map(str, native_target_rankings[qid])) for qid, _ in queries},
        target_scores=target_scores,
        qrels=qrels,
        source_model=source_model.model_id,
        target_model=target_model.model_id,
        corpus_size=documents.size(),
        k_values=k_values,
        quality_k=quality_k,
        epsilon=epsilon,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        diagnostic=diagnostic,
        recommended_k=recommended_k,
        ann_status=ann_status,
    )
