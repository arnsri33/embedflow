from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import SQLiteVectorCache
from ..metrics import summarize
from ..migration.materializer import MaterializationWorker
from ..migration.planner import MigrationPlan, make_plan
from ..migration.state import DocumentStore, MigrationState


def _elapsed(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1_000_000.0


class MigrationEngine:
    """Backend-agnostic source retrieval + target reranking serving engine."""

    def __init__(self, cfg: Any, source_model: Any, target_model: Any, source_index: Any,
                 target_cache: SQLiteVectorCache, documents: DocumentStore,
                 target_index: Any | None = None, probe: dict[str, Any] | None = None,
                 ann_status: str = "UNKNOWN", start_worker: bool = True):
        self.cfg, self.source_model, self.target_model = cfg, source_model, target_model
        self.source_index, self.target_index, self.cache, self.documents = source_index, target_index, target_cache, documents
        self.state = MigrationState(cfg.state_path, cfg)
        self.probe = probe or {}
        self.ann_status = ann_status
        self.plan: MigrationPlan = make_plan(target_model=cfg.target.model, corpus_size=documents.size(),
            diagnostic=self.probe.get("diagnostic", "UNKNOWN"), recommended_k=self.probe.get("recommended_k"),
            ann_status=ann_status, cached_target_vectors=target_cache.stats()["cached_target_vectors"],
            default_k=cfg.migration.candidate_depth, throughput_docs_per_second=cfg.economics.target_docs_per_second,
            gpu_price_per_hour=cfg.economics.gpu_price_per_hour, source_model=cfg.source.model)
        queue_path = Path(cfg.cache.path) / "materialization_queue.sqlite3"
        self.worker = MaterializationWorker(target_model, documents, target_cache, queue_path,
                                            batch_size=cfg.migration.background_batch_size,
                                            max_retries=cfg.migration.max_retries, state=self.state)
        self.records: list[dict[str, Any]] = []
        telemetry_path = getattr(getattr(cfg, "telemetry", None), "latency_log", None)
        self.metrics_path = Path(telemetry_path) if telemetry_path else Path(cfg.cache.path) / "latency.jsonl"
        if self.metrics_path.exists():
            try:
                with self.metrics_path.open() as handle:
                    loaded = [json.loads(line) for line in handle if line.strip()]
                self.records = [row for row in loaded if isinstance(row, dict)]
            except (OSError, json.JSONDecodeError):
                # A malformed metrics tail must not make cached vectors
                # unusable; fresh measurements continue in a new process.
                self.records = []
        self._lock = threading.RLock()
        self._closed = False
        self.state.update(status="SERVING", diagnostic=self.plan.diagnostic, ann_status=ann_status,
                          candidate_depth=self.plan.candidate_depth, corpus_size=documents.size(),
                          migration_strategy=self.plan.migration_strategy)
        if start_worker: self.worker.start()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("migration engine is closed")

    def _record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(row)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                handle.flush()
            self.state.update(queries=int(self.state.snapshot().get("queries", 0)) + 1,
                              last_latency_ms=row["total_ms"], cache=self.cache.stats(), worker=self.worker.stats())

    def search(self, query: str, top_k: int = 10, candidate_depth: int | None = None,
               max_sync_misses: int | None = None) -> dict[str, Any]:
        self._ensure_open()
        if not isinstance(query, str) or not query.strip(): raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or int(top_k) != top_k or int(top_k) < 1:
            raise ValueError("top_k must be a positive integer")
        if candidate_depth is not None and (isinstance(candidate_depth, bool) or int(candidate_depth) != candidate_depth or int(candidate_depth) < 1):
            raise ValueError("candidate_depth must be a positive integer")
        if max_sync_misses is not None and (isinstance(max_sync_misses, bool) or int(max_sync_misses) != max_sync_misses or int(max_sync_misses) < 0):
            raise ValueError("max_sync_misses must be a non-negative integer")
        K = int(candidate_depth or self.plan.candidate_depth or self.cfg.migration.candidate_depth)
        if K < 1:
            raise ValueError("candidate_depth must be a positive integer")
        max_sync = self.cfg.migration.max_sync_misses if max_sync_misses is None else int(max_sync_misses)
        total_start = time.perf_counter_ns(); row: dict[str, Any] = {"mode": "embedflow", "K": K}
        t = time.perf_counter_ns(); source_vector = np.asarray(self.source_model.encode_query(query), dtype="float32"); row["source_query_encode_ms"] = _elapsed(t)
        if source_vector.ndim != 1 or source_vector.shape[0] != int(self.source_model.dimension) or not np.isfinite(source_vector).all():
            raise ValueError("source query encoder returned an invalid vector")
        t = time.perf_counter_ns(); candidates = self.source_index.search(source_vector, K); row["source_ann_ms"] = _elapsed(t)
        candidate_ids = [hit.document_id for hit in candidates]
        t = time.perf_counter_ns(); target_vector = np.asarray(self.target_model.encode_query(query), dtype="float32"); row["target_query_encode_ms"] = _elapsed(t)
        if target_vector.ndim != 1 or target_vector.shape[0] != int(self.target_model.dimension) or not np.isfinite(target_vector).all():
            raise ValueError("target query encoder returned an invalid vector")
        t = time.perf_counter_ns(); cached = self.cache.get(candidate_ids); row["cache_lookup_ms"] = _elapsed(t)
        missing = [doc_id for doc_id in candidate_ids if doc_id not in cached]
        initial_hits = len(cached)
        sync_ids = missing[:max_sync]
        if sync_ids:
            t = time.perf_counter_ns()
            docs = self.documents.get(sync_ids)
            vectors = self.target_model.encode_documents([docs[x] for x in sync_ids], batch_size=self.cfg.migration.background_batch_size)
            self.cache.put(sync_ids, vectors); cached.update({x: np.asarray(v, dtype="float32") for x, v in zip(sync_ids, vectors)})
            row["synchronous_target_encode_ms"] = _elapsed(t)
        else: row["synchronous_target_encode_ms"] = 0.0
        async_ids = [doc_id for doc_id in missing if doc_id not in cached]
        if async_ids: self.worker.enqueue(async_ids)
        scored_ids = [doc_id for doc_id in candidate_ids if doc_id in cached]
        t = time.perf_counter_ns()
        if scored_ids:
            scores = {doc_id: float(np.asarray(cached[doc_id], dtype="float32") @ target_vector) for doc_id in scored_ids}
            candidate_positions = {doc_id: position for position, doc_id in enumerate(candidate_ids)}
            target_order = sorted(scored_ids, key=lambda x: (-scores[x], candidate_positions[x]))
        else:
            scores, target_order = {}, candidate_ids[:]
        row["target_score_ms"] = _elapsed(t)
        t = time.perf_counter_ns(); selected = target_order[:max(1, int(top_k))]; row["topk_ms"] = _elapsed(t)
        target_rank = {doc_id: rank for rank, doc_id in enumerate(target_order, 1)}
        source_rank = {hit.document_id: hit.source_rank + 1 for hit in candidates}
        # Resolve result text in one batch.  PgVectorDocumentStore turns this
        # into a single ``WHERE id = ANY(...)`` query instead of issuing one
        # SQL request per selected document; the regular JSONL store keeps the
        # same behavior behind its existing DocumentStore contract.
        selected_texts = self.documents.get(selected) if selected else {}
        results = []
        for doc_id in selected:
            results.append({"id": doc_id, "text": selected_texts[doc_id], "target_score": scores.get(doc_id),
                            "source_rank": source_rank.get(doc_id), "target_rank": target_rank.get(doc_id), "target_vector_cached": doc_id in cached})
        # State describes vectors available for this response, including the
        # bounded synchronous budget.  ``cache_hits`` remains the persistent
        # lookup count; callers can use ``sync_encoded`` and
        # ``target_vectors_available`` to distinguish a cold lookup that was
        # fully satisfied synchronously from a genuinely partial ranking.
        available_count = len(scored_ids)
        if available_count == len(candidate_ids): status = "WARM"
        elif available_count == 0: status = "COLD"
        else: status = "PARTIAL"
        row["total_ms"] = _elapsed(total_start); row.update({"cache_hits": initial_hits, "cache_misses": len(missing),
            "sync_misses_encoded": len(sync_ids), "async_misses_queued": len(async_ids), "status": status,
            "target_vectors_available": available_count, "candidate_ids": candidate_ids})
        self._record(row)
        timings = {k: float(row[k]) for k in ("source_query_encode_ms", "source_ann_ms", "target_query_encode_ms", "cache_lookup_ms",
                                               "synchronous_target_encode_ms", "target_score_ms", "topk_ms", "total_ms")}
        return {"results": results, "migration": {"status": status, "source_candidates": len(candidate_ids), "target_cache_hits": initial_hits,
                 "target_cache_misses": len(missing), "cache_hit_rate": initial_hits / max(1, len(candidate_ids)),
                 "target_vectors_available": available_count,
                 "ranking_cache_hit_rate": available_count / max(1, len(candidate_ids)),
                 "sync_misses_encoded": len(sync_ids), "async_misses_queued": len(async_ids),
                 # Short aliases mirror the public REST schema while the
                 # target_* names remain for backwards compatibility.
                 "cache_hits": initial_hits, "cache_misses": len(missing),
                 "sync_encoded": len(sync_ids), "async_queued": len(async_ids),
                 "candidate_depth": K},
                "timing_ms": timings}

    def prewarm(self, document_ids: list[str], asynchronous: bool = False) -> dict[str, Any]:
        self._ensure_open()
        ids = list(dict.fromkeys(str(x) for x in document_ids)); missing = [x for x in ids if x not in self.cache.contains(ids)]
        if asynchronous:
            queued = self.worker.enqueue(missing); return {"requested": len(ids), "queued": queued, "asynchronous": True}
        if missing:
            batch_size = max(1, int(self.cfg.migration.background_batch_size))
            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                docs = self.documents.get(chunk)
                vectors = self.target_model.encode_documents([docs[x] for x in chunk], batch_size=batch_size)
                self.cache.put(chunk, vectors)
        return {"requested": len(ids), "materialized": len(missing), "asynchronous": False}

    def status(self) -> dict[str, Any]:
        self._ensure_open()
        cache = self.cache.stats(); worker = self.worker.stats(); data = self.state.snapshot()
        plan = make_plan(target_model=self.cfg.target.model, corpus_size=self.documents.size(), diagnostic=self.plan.diagnostic,
                         recommended_k=self.plan.candidate_depth, ann_status=self.ann_status,
                         cached_target_vectors=cache["cached_target_vectors"], default_k=self.cfg.migration.candidate_depth,
                         source_model=self.cfg.source.model)
        values: list[float] = []
        for row in self.records:
            try:
                value = float(row.get("total_ms"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0:
                values.append(value)
        lat = summarize(values) if values else {"count": 0}
        return {"migration": plan.to_dict(), "index": self.source_index.metadata(), "cache": cache,
                "worker": worker, "state": data, "latency": lat}

    def close(self, close_models: bool = True, close_indexes: bool = True) -> None:
        """Stop background work and close resources.

        ``close_models=False`` is used by the public facade when the caller
        supplied model instances that it still owns.  Existing CLI callers
        retain the original behavior through the default.
        """
        if self._closed:
            return
        try:
            self.worker.close()
        finally:
            try:
                self.cache.close()
            finally:
                try:
                    # Qdrant owns a file lock/client connection; release both
                    # source and optional target indexes when the engine owns
                    # the serving session. In-memory/FAISS implementations are
                    # no-ops, and identity avoids closing a shared object twice.
                    if close_indexes:
                        closed_indexes: set[int] = set()
                        for index in (self.source_index, self.target_index):
                            if index is None or id(index) in closed_indexes:
                                continue
                            close_index = getattr(index, "close", None)
                            if callable(close_index):
                                close_index()
                            closed_indexes.add(id(index))
                finally:
                    if close_models:
                        try:
                            self.source_model.close()
                        finally:
                            self.target_model.close()
                    self._closed = True
