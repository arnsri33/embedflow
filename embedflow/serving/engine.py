from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import SQLiteVectorCache
from ..metrics import summarize
from ..migration.materializer import MaterializationWorker, ReadOnlyMaterializationWorker
from ..migration.planner import MigrationPlan, make_plan
from ..migration.state import DocumentStore, MigrationState
from ..shadow import ShadowRunner, ShadowTaskError, ShadowTelemetry, index_identity_from_config, migration_fingerprint


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
        self.runtime_mode = str(getattr(getattr(cfg, "runtime", None), "mode", "migration") or "migration").lower()
        shadow_cfg = getattr(cfg, "shadow", None)
        shadow_mode = self.runtime_mode == "shadow"
        source_only_mode = self.runtime_mode in {"source", "shadow"}
        shadow_active = shadow_mode and bool(getattr(shadow_cfg, "enabled", False))
        # Source-only modes must not even open the normal persistent
        # materialization queue.  Opening that SQLite queue resets stale
        # processing rows and starts a worker, which would be an unnecessary
        # side effect when Shadow Mode is disabled (or when the explicit
        # ``source`` mode is selected).
        if source_only_mode and (not shadow_active or not bool(getattr(shadow_cfg, "materialize", True))):
            self.worker = ReadOnlyMaterializationWorker()
        else:
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
        self._shadow_closed = False
        self._metrics_queue: queue.Queue[Any] | None = None
        self._metrics_stop: threading.Event | None = None
        self._metrics_thread: threading.Thread | None = None
        # Source-only and Shadow requests must not wait for the latency/state
        # writer.  Keep that observability path bounded and asynchronous; the
        # historical migration path continues to use the synchronous writer.
        if source_only_mode:
            self._metrics_queue = queue.Queue(maxsize=max(16, min(int(getattr(shadow_cfg, "queue_capacity", 1000) or 1000), 1000)))
            self._metrics_stop = threading.Event()
            self._metrics_thread = threading.Thread(target=self._metrics_loop,
                                                     name="embedflow-source-metrics", daemon=True)
            self._metrics_thread.start()
        self.shadow_telemetry: ShadowTelemetry | None = None
        self.shadow_runner: ShadowRunner | None = None
        if shadow_active:
            telemetry_cfg = getattr(shadow_cfg, "telemetry", None)
            telemetry_path = getattr(telemetry_cfg, "path", None) or (Path(cfg.cache.path) / "shadow.sqlite3")
            index_meta = source_index.metadata()
            # Fingerprint the configured source identity rather than a
            # backend-specific metadata rendering.  The report command can
            # therefore reopen telemetry without loading a client/model and
            # still select exactly the same migration configuration.
            index_identity = index_identity_from_config(cfg.index)
            shadow_k = getattr(shadow_cfg, "candidate_k", None) or self.plan.candidate_depth or cfg.migration.candidate_depth
            config_fingerprint = migration_fingerprint(source_fingerprint=source_model.fingerprint,
                                                        target_fingerprint=target_model.fingerprint,
                                                        # Use the configured backend for the stable report key.  A FAISS
                                                        # deployment may legitimately use the NumPy fallback on a machine
                                                        # without faiss-cpu; exposing that implementation detail here would
                                                        # make ``shadow report`` unable to reopen the same telemetry.
                                                        backend=str(cfg.index.backend),
                                                        index_identity=index_identity, candidate_k=int(shadow_k))
            self.shadow_telemetry = ShadowTelemetry(
                telemetry_path,
                config_fingerprint=config_fingerprint,
                enabled=bool(getattr(telemetry_cfg, "enabled", True)),
                max_records=int(getattr(telemetry_cfg, "max_records", 10_000)),
                retain_query_records=bool(getattr(telemetry_cfg, "retain_query_records", False)),
                retain_query_text=bool(getattr(telemetry_cfg, "retain_query_text", False)),
                retention_days=getattr(telemetry_cfg, "retention_days", None),
                min_target_coverage_for_ranking=float(getattr(telemetry_cfg, "min_target_coverage_for_ranking", 1.0)),
                source_fingerprint=source_model.fingerprint,
                target_fingerprint=target_model.fingerprint,
                backend=str(getattr(cfg.index, "backend", index_meta.get("backend", "unknown"))),
                index_identity=index_identity,
            )
            self.shadow_runner = ShadowRunner(
                self._run_shadow_task,
                self.shadow_telemetry,
                sample_rate=float(getattr(shadow_cfg, "sample_rate", 0.1)),
                sample_seed=int(getattr(shadow_cfg, "sample_seed", 42)),
                max_inflight=int(getattr(shadow_cfg, "max_inflight", 32)),
                queue_capacity=int(getattr(shadow_cfg, "queue_capacity", 1000)),
                timeout_ms=int(getattr(shadow_cfg, "timeout_ms", 10_000)),
                shutdown_grace_ms=int(getattr(shadow_cfg, "shutdown_grace_ms", 1_000)),
                config_fingerprint=config_fingerprint,
            )
            self.worker.on_materialized = self._on_shadow_materialized
            self.worker.on_failed = self._on_shadow_materialization_failed
        self.state.update(status="SERVING", diagnostic=self.plan.diagnostic, ann_status=ann_status,
                          candidate_depth=self.plan.candidate_depth, corpus_size=documents.size(),
                          migration_strategy=self.plan.migration_strategy, runtime_mode=self.runtime_mode)
        if start_worker and (not source_only_mode or (shadow_active and bool(getattr(shadow_cfg, "materialize", True)))): self.worker.start()

    def _on_shadow_materialized(self, count: int) -> None:
        if self.shadow_telemetry is not None:
            try:
                self.shadow_telemetry.increment("target_docs_materialized", int(count))
            except Exception:
                pass

    def _on_shadow_materialization_failed(self, count: int) -> None:
        if self.shadow_telemetry is not None:
            try:
                self.shadow_telemetry.increment("target_docs_failed", int(count))
            except Exception:
                pass

    def _shadow_queue_snapshot(self) -> dict[str, Any]:
        """Return both bounded shadow-dispatch and materialization queues."""
        runner = self.shadow_runner.stats() if self.shadow_runner is not None else {}
        worker = self.worker.stats()
        materialization = worker.get("queue", {}) if isinstance(worker, Mapping) else {}
        return {**runner, "materialization_queue": materialization}

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

    def _metrics_loop(self) -> None:
        """Persist source latency/state off the source response critical path."""
        assert self._metrics_queue is not None and self._metrics_stop is not None
        while True:
            try:
                row = self._metrics_queue.get(timeout=0.05)
            except queue.Empty:
                if self._metrics_stop.is_set():
                    return
                continue
            try:
                if row is None:
                    return
                self._record(row)
            except Exception:
                # A broken observability destination must not affect source
                # serving or terminate the writer before later rows can be
                # attempted.
                pass
            finally:
                self._metrics_queue.task_done()

    def _record_source_nonblocking(self, row: dict[str, Any]) -> None:
        queue_ = self._metrics_queue
        if queue_ is None:
            try:
                self._record(row)
            except Exception:
                pass
            return
        try:
            queue_.put_nowait(dict(row))
        except queue.Full:
            # Dropping an observability row is preferable to making the
            # source request wait under backpressure. Shadow telemetry keeps
            # its own bounded dropped counter for sampled work.
            pass

    def _validate_search(self, query: str, top_k: int, candidate_depth: int | None,
                         max_sync_misses: int | None) -> tuple[int, int, int]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        def integer(value: Any, label: str, minimum: int) -> int:
            if isinstance(value, bool):
                raise ValueError(f"{label} must be an integer")
            try:
                parsed = int(value)
                exact = float(value) == parsed
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{label} must be an integer") from None
            if not exact or parsed < minimum:
                if minimum == 1:
                    raise ValueError(f"{label} must be a positive integer")
                raise ValueError(f"{label} must be an integer >= {minimum}")
            return parsed

        top_k = integer(top_k, "top_k", 1)
        if candidate_depth is not None:
            candidate_depth = integer(candidate_depth, "candidate_depth", 1)
        if max_sync_misses is not None:
            max_sync_misses = integer(max_sync_misses, "max_sync_misses", 0)
        K = int(candidate_depth or self.plan.candidate_depth or self.cfg.migration.candidate_depth)
        if K < 1:
            raise ValueError("candidate_depth must be a positive integer")
        # Remote vector APIs impose finite per-query limits.  Validate the
        # effective request before invoking a backend so a direct Python/API
        # caller receives a stable EmbedFlow error instead of an opaque SDK
        # traceback (or an accidentally enormous local request).
        index_cfg = getattr(self.cfg, "index", None)
        backend = str(getattr(index_cfg, "backend", "")).lower()
        backend_limits = {"pinecone": 10_000, "milvus": 16_384, "weaviate": 10_000}
        limit = backend_limits.get(backend)
        if limit is not None and max(int(top_k), K) > limit:
            raise ValueError(f"{backend} candidate depth/top_k must be <= {limit}")
        max_sync = self.cfg.migration.max_sync_misses if max_sync_misses is None else int(max_sync_misses)
        return int(top_k), K, max_sync

    def _source_search(self, query: str, top_k: int, candidate_depth: int | None,
                       max_sync_misses: int | None, *, request_id: str | None = None) -> dict[str, Any]:
        """Return the source-authoritative response used by Shadow Mode.

        This path intentionally performs no target encoding or cache lookup
        before returning.  The resulting schema mirrors the normal search
        response, while target-specific fields are explicitly ``None``.
        """
        if candidate_depth is None and self.runtime_mode == "shadow":
            configured_shadow_k = getattr(getattr(self.cfg, "shadow", None), "candidate_k", None)
            candidate_depth = configured_shadow_k
        top_k, K, _ = self._validate_search(query, top_k, candidate_depth, max_sync_misses)
        total_start = time.perf_counter_ns()
        row: dict[str, Any] = {"mode": self.runtime_mode, "K": K}
        started = time.perf_counter_ns()
        source_vector = np.asarray(self.source_model.encode_query(query), dtype="float32")
        row["source_query_encode_ms"] = _elapsed(started)
        if source_vector.ndim != 1 or source_vector.shape[0] != int(self.source_model.dimension) or not np.isfinite(source_vector).all():
            raise ValueError("source query encoder returned an invalid vector")
        started = time.perf_counter_ns()
        candidates = self.source_index.search(source_vector, max(K, top_k))
        row["source_ann_ms"] = _elapsed(started)
        selected = list(candidates[:top_k])
        selected_ids = [str(hit.document_id) for hit in selected]
        texts = self.documents.get(selected_ids) if selected_ids else {}
        results = [{"id": str(hit.document_id), "text": texts[str(hit.document_id)],
                    "target_score": None,
                    "source_rank": int(hit.source_rank) + 1, "target_rank": None,
                    "target_vector_cached": False} for hit in selected]
        zero_stages = {"target_query_encode_ms": 0.0, "cache_lookup_ms": 0.0,
                       "synchronous_target_encode_ms": 0.0, "target_score_ms": 0.0, "topk_ms": 0.0}
        row.update(zero_stages)
        row["total_ms"] = _elapsed(total_start)
        row.update({"cache_hits": 0, "cache_misses": 0, "sync_misses_encoded": 0,
                    "async_misses_queued": 0, "status": "SOURCE", "target_vectors_available": 0,
                    "candidate_ids": [str(hit.document_id) for hit in candidates]})
        # Latency/state persistence is observability, not part of the
        # source-authoritative response contract.  In source-only modes this
        # handoff is explicitly non-blocking; a broken/slow destination is
        # handled by the bounded writer thread.
        if self.runtime_mode in {"source", "shadow"}:
            self._record_source_nonblocking(row)
        else:
            try:
                self._record(row)
            except Exception:
                pass
        scheduled = False
        if self.shadow_runner is not None and not self._shadow_closed:
            payload = {"query": query, "candidate_hits": tuple(candidates),
                       # Keep the full source candidate ordering for shadow
                       # overlap diagnostics.  The user-facing response may
                       # request top_k=1 while the shadow candidate pool is
                       # K=100; truncating here would make the configured
                       # report_k overlap denominator meaningless.
                       "source_ids": tuple(str(hit.document_id) for hit in candidates),
                       "source_latency_ms": row["total_ms"],
                       "source_candidate_latency_ms": row["source_ann_ms"],
                       "candidate_k": K}
            try:
                scheduled = self.shadow_runner.dispatch(payload, request_id=request_id)
            except Exception:
                # Scheduling is deliberately outside the primary failure path.
                scheduled = False
        migration = {"status": "SOURCE", "source_authoritative": True,
                     "source_candidates": len(candidates), "target_cache_hits": 0,
                     "target_cache_misses": 0, "cache_hit_rate": 0.0,
                     "target_vectors_available": 0, "ranking_cache_hit_rate": 0.0,
                     "sync_misses_encoded": 0, "async_misses_queued": 0,
                     "cache_hits": 0, "cache_misses": 0, "sync_encoded": 0,
                     "async_queued": 0, "candidate_depth": K,
                     "shadow_enabled": self.shadow_runner is not None,
                     "shadow_scheduled": scheduled}
        return {"results": results, "migration": migration,
                "timing_ms": {key: float(row[key]) for key in ("source_query_encode_ms", "source_ann_ms",
                    "target_query_encode_ms", "cache_lookup_ms", "synchronous_target_encode_ms",
                    "target_score_ms", "topk_ms", "total_ms")}}

    def _rank_cached_candidates(self, candidate_ids: list[str], cached: Mapping[str, Any],
                                target_vector: np.ndarray) -> tuple[list[str], dict[str, float], float]:
        """Score cached target vectors using the shared migration ranking path.

        Shadow Mode deliberately has a different *execution policy* (no
        synchronous misses and no source-response gating), but it must not
        grow a second ranking implementation.  Keeping validation, tie
        breaking, and score direction here means normal migration and shadow
        observations use exactly the same target scoring semantics.
        """
        started = time.perf_counter_ns()
        scores: dict[str, float] = {}
        positions = {document_id: position for position, document_id in enumerate(candidate_ids)}
        scored_ids = [document_id for document_id in candidate_ids if document_id in cached]
        try:
            for document_id in scored_ids:
                vector = np.asarray(cached[document_id], dtype="float32")
                if vector.ndim != 1 or vector.shape[0] != int(self.target_model.dimension) or not np.isfinite(vector).all():
                    raise ValueError(f"cached target vector for {document_id!r} is invalid")
                score = float(vector @ target_vector)
                if not np.isfinite(score):
                    raise ValueError(f"cached target score for {document_id!r} is non-finite")
                scores[document_id] = score
        except Exception:
            # Preserve the caller's existing error classification/UX while
            # ensuring a single helper owns all score validation.
            raise
        target_order = sorted(scored_ids, key=lambda document_id: (-scores[document_id], positions[document_id]))
        return target_order, scores, _elapsed(started)

    def _cache_content_expectations(self, document_ids: list[str]) -> dict[str, str] | None:
        """Return current JSONL content fingerprints when available.

        Remote document stores may resolve text lazily and should not incur an
        extra lookup on the source response path.  In-memory/local stores can
        cheaply bind cache validity to current content; the materializer still
        performs the authoritative check for remote stores.
        """
        fingerprint = getattr(self.cache, "content_fingerprint", None)
        # Remote backend document stores expose a lazy Mapping proxy whose
        # ``__getitem__`` performs one network lookup.  Content binding is
        # authoritative in the materializer for those stores; do not turn the
        # primary search path into an N+1 fetch.  Direct in-memory mappings are
        # safe to fingerprint inline.
        if type(self.documents) is dict:
            mapping = self.documents
        elif isinstance(self.documents, Mapping):
            # Lazy backend Mapping proxies perform a remote lookup per key;
            # leave content binding to the materializer for those stores.
            mapping = None
        else:
            candidate_mapping = getattr(self.documents, "documents", None)
            # A concrete dict is the local JSONL/in-memory case.  Do not
            # index arbitrary Mapping proxies here for the same N+1 reason.
            mapping = candidate_mapping if type(candidate_mapping) is dict else None
        if not isinstance(mapping, Mapping) or not callable(fingerprint):
            return None
        return {str(document_id): fingerprint(mapping[document_id])
                for document_id in document_ids if document_id in mapping}

    def _cache_put(self, document_ids: list[str], vectors: Any,
                   documents: Mapping[str, Any] | None = None) -> None:
        """Write target vectors while preserving older cache implementations.

        The built-in SQLite cache binds vectors to document content.  External
        cache implementations written against the pre-0.8 contract may only
        accept ``put(ids, vectors)``; keeping this compatibility shim here
        avoids making the shared migration engine unusable for those adapters.
        """
        fingerprint_fn = getattr(self.cache, "content_fingerprint", None)
        if callable(fingerprint_fn) and documents is not None:
            fingerprints = {document_id: fingerprint_fn(documents[document_id])
                            for document_id in document_ids if document_id in documents}
            if len(fingerprints) == len(document_ids):
                try:
                    self.cache.put(document_ids, vectors, content_fingerprints=fingerprints)
                    return
                except TypeError as exc:
                    if "content_fingerprint" not in str(exc):
                        raise
        self.cache.put(document_ids, vectors)

    def _run_shadow_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run the target path after the source response has been returned."""
        if self._shadow_closed:
            raise ShadowTaskError("INTERNAL", "shadow runner is shutting down")
        shadow_cfg = getattr(self.cfg, "shadow", None)
        materialize = bool(getattr(shadow_cfg, "materialize", True))
        hits = list(payload.get("candidate_hits", ()))
        candidate_ids = list(dict.fromkeys(str(hit.document_id) for hit in hits))
        if not candidate_ids:
            return {"status": "completed", "source_latency_ms": payload.get("source_latency_ms"),
                    "source_candidate_latency_ms": payload.get("source_candidate_latency_ms"),
                    "target_coverage": 1.0, "target_candidates_available": 0,
                    "target_candidates_missing": 0, "candidate_ids": tuple(),
                    "missing_candidate_ids": tuple()}
        started = time.perf_counter_ns()
        try:
            target_vector = np.asarray(self.target_model.encode_query(str(payload.get("query", ""))), dtype="float32")
        except Exception as exc:
            # Keep private query text and provider payloads out of telemetry;
            # the category is actionable while the raw exception is not.
            raise ShadowTaskError("TARGET_QUERY_ENCODING_ERROR", "target query encoding failed") from exc
        target_query_ms = _elapsed(started)
        if target_vector.ndim != 1 or target_vector.shape[0] != int(self.target_model.dimension) or not np.isfinite(target_vector).all():
            raise ShadowTaskError("TARGET_QUERY_ENCODING_ERROR", "target query encoder returned an invalid vector")
        try:
            if materialize:
                expected = self._cache_content_expectations(candidate_ids)
                cached = self.cache.get(candidate_ids, content_fingerprints=expected) if expected is not None else self.cache.get(candidate_ids)
            else:
                peek = getattr(self.cache, "peek", None)
                if not callable(peek):
                    # Falling back to ``get`` would mutate hit/miss counters
                    # (and some cache implementations refresh access times),
                    # violating the explicit materialize=false guarantee.
                    raise ShadowTaskError("CACHE_ERROR", "target cache does not support read-only lookup")
                expected = self._cache_content_expectations(candidate_ids)
                cached = peek(candidate_ids, content_fingerprints=expected) if expected is not None else peek(candidate_ids)
        except Exception as exc:
            raise ShadowTaskError("CACHE_ERROR", "target cache lookup failed") from exc
        missing = [document_id for document_id in candidate_ids if document_id not in cached]
        queued = 0
        if materialize and missing:
            try:
                queued = int(self.worker.enqueue(missing))
            except Exception as exc:
                raise ShadowTaskError("MATERIALIZATION_ERROR", "target materialization enqueue failed") from exc
        try:
            target_order, scores, target_rerank_ms = self._rank_cached_candidates(candidate_ids, cached, target_vector)
        except Exception as exc:
            raise ShadowTaskError("RERANK_ERROR", "target reranking failed") from exc
        scored_ids = list(target_order)
        coverage = len(scored_ids) / max(1, len(candidate_ids))
        minimum = float(getattr(getattr(shadow_cfg, "telemetry", None), "min_target_coverage_for_ranking", 1.0))
        status = "completed" if coverage >= minimum else "partial"
        source_ids = list(payload.get("source_ids", ()))
        report_k = int(getattr(getattr(shadow_cfg, "telemetry", None), "report_k", 10))
        top1 = None if not target_order or not source_ids else float(target_order[0] == source_ids[0])
        source_top = set(source_ids[:report_k]); target_top = set(target_order[:report_k])
        # A backend may return fewer than the configured report K (for
        # example, a small/partially available candidate set).  Use the
        # common available depth as the denominator so a short target list
        # cannot make overlap look better than the evidence supports.
        overlap = None if not source_top or not target_top else len(source_top & target_top) / max(
            1, min(report_k, len(source_top), len(target_top))
        )
        return {"status": status, "source_latency_ms": payload.get("source_latency_ms"),
                "source_candidate_latency_ms": payload.get("source_candidate_latency_ms"),
                "target_query_encode_latency_ms": target_query_ms, "target_rerank_latency_ms": target_rerank_ms,
                "cache_hits": len(cached), "cache_misses": len(missing),
                "target_candidates_available": len(scored_ids), "target_candidates_missing": len(missing),
                "candidate_ids": tuple(candidate_ids), "missing_candidate_ids": tuple(missing),
                "target_coverage": coverage, "top1_agreement": top1, "top_k_overlap": overlap,
                # ``queued`` is the count actually accepted by the
                # persistent deduplicating queue.  A candidate may be missing
                # from cache yet already pending from an earlier shadow
                # request, so reporting ``len(missing)`` here would double
                # count materialization work.
                "docs_queued": queued, "unique_docs_queued": queued}

    def search(self, query: str, top_k: int = 10, candidate_depth: int | None = None,
               max_sync_misses: int | None = None, *, request_id: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        # ``runtime.mode=shadow`` is source-authoritative even when the
        # feature's explicit ``shadow.enabled`` kill switch is false.  In
        # that state we keep the source-only response contract but simply do
        # not schedule any shadow work; falling through to the normal
        # migration path would unexpectedly rerank user traffic.
        if self.shadow_runner is not None or self.runtime_mode in {"source", "shadow"}:
            return self._source_search(query, top_k, candidate_depth, max_sync_misses, request_id=request_id)
        top_k, K, max_sync = self._validate_search(query, top_k, candidate_depth, max_sync_misses)
        total_start = time.perf_counter_ns(); row: dict[str, Any] = {"mode": "embedflow", "K": K}
        t = time.perf_counter_ns(); source_vector = np.asarray(self.source_model.encode_query(query), dtype="float32"); row["source_query_encode_ms"] = _elapsed(t)
        if source_vector.ndim != 1 or source_vector.shape[0] != int(self.source_model.dimension) or not np.isfinite(source_vector).all():
            raise ValueError("source query encoder returned an invalid vector")
        t = time.perf_counter_ns(); candidates = self.source_index.search(source_vector, K); row["source_ann_ms"] = _elapsed(t)
        candidate_ids = [hit.document_id for hit in candidates]
        t = time.perf_counter_ns(); target_vector = np.asarray(self.target_model.encode_query(query), dtype="float32"); row["target_query_encode_ms"] = _elapsed(t)
        if target_vector.ndim != 1 or target_vector.shape[0] != int(self.target_model.dimension) or not np.isfinite(target_vector).all():
            raise ValueError("target query encoder returned an invalid vector")
        t = time.perf_counter_ns()
        expected_content = self._cache_content_expectations(candidate_ids)
        cached = (self.cache.get(candidate_ids, content_fingerprints=expected_content)
                  if expected_content is not None else self.cache.get(candidate_ids))
        row["cache_lookup_ms"] = _elapsed(t)
        missing = [doc_id for doc_id in candidate_ids if doc_id not in cached]
        initial_hits = len(cached)
        sync_ids = missing[:max_sync]
        if sync_ids:
            t = time.perf_counter_ns()
            docs = self.documents.get(sync_ids)
            vectors = self.target_model.encode_documents([docs[x] for x in sync_ids], batch_size=self.cfg.migration.background_batch_size)
            self._cache_put(sync_ids, vectors, docs)
            cached.update({x: np.asarray(v, dtype="float32") for x, v in zip(sync_ids, vectors)})
            row["synchronous_target_encode_ms"] = _elapsed(t)
        else: row["synchronous_target_encode_ms"] = 0.0
        async_ids = [doc_id for doc_id in missing if doc_id not in cached]
        if async_ids: self.worker.enqueue(async_ids)
        scored_ids = [doc_id for doc_id in candidate_ids if doc_id in cached]
        target_order_cached, scores, row["target_score_ms"] = self._rank_cached_candidates(candidate_ids, cached, target_vector)
        if scored_ids:
            target_order = target_order_cached
        else:
            scores, target_order = {}, candidate_ids[:]
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
        ids = list(dict.fromkeys(str(x) for x in document_ids))
        expected_content = self._cache_content_expectations(ids)
        warm = (self.cache.contains(ids, content_fingerprints=expected_content)
                 if expected_content is not None else self.cache.contains(ids))
        missing = [x for x in ids if x not in warm]
        if asynchronous:
            queued = self.worker.enqueue(missing); return {"requested": len(ids), "queued": queued, "asynchronous": True}
        if missing:
            batch_size = max(1, int(self.cfg.migration.background_batch_size))
            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                docs = self.documents.get(chunk)
                vectors = self.target_model.encode_documents([docs[x] for x in chunk], batch_size=batch_size)
                self._cache_put(chunk, vectors, docs)
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
        shadow = None
        if self.shadow_runner is not None and self.shadow_telemetry is not None:
            shadow_k = getattr(self.cfg.shadow, "candidate_k", None) or self.plan.candidate_depth
            shadow_queue_for_report = self._shadow_queue_snapshot()
            shadow = self.shadow_telemetry.snapshot(queue=shadow_queue_for_report, candidate_k=shadow_k,
                                                     sample_rate=self.shadow_runner.sample_rate)
            runner_stats = self.shadow_runner.stats()
            # The persistence writer is asynchronous by design.  Merge the
            # runner's lock-protected counters into the live status snapshot
            # so an operator sees dispatches immediately without putting a
            # SQLite write on the source response path.
            shadow.update(runner_stats)
            live_counters = dict(shadow.get("counters", {}))
            live_counters.update(runner_stats.get("counters", {}))
            shadow["counters"] = live_counters
            shadow["materialize"] = bool(getattr(self.cfg.shadow, "materialize", True))
            shadow["telemetry_enabled"] = bool(getattr(getattr(self.cfg.shadow, "telemetry", None), "enabled", True))
        return {"migration": plan.to_dict(), "index": self.source_index.metadata(), "cache": cache,
                "worker": worker, "state": data, "latency": lat,
                "runtime": {"mode": self.runtime_mode, "shadow_enabled": self.shadow_runner is not None},
                "shadow": shadow or {"enabled": False, "mode": self.runtime_mode}}

    def close(self, close_models: bool = True, close_indexes: bool = True) -> None:
        """Stop background work and close resources.

        ``close_models=False`` is used by the public facade when the caller
        supplied model instances that it still owns.  Existing CLI callers
        retain the original behavior through the default.
        """
        if self._closed:
            return
        self._shadow_closed = True
        try:
            if self._metrics_stop is not None:
                self._metrics_stop.set()
                if self._metrics_thread is not None:
                    # Keep shutdown bounded even when a filesystem/state
                    # writer is stuck.  The thread is a daemon and is never
                    # allowed to hold up closing the source/index resources.
                    self._metrics_thread.join(timeout=0.25)
            if self.shadow_runner is not None:
                self.shadow_runner.close()
            try:
                # A provider can still be inside a target-document encode at
                # shutdown.  Shadow mode promises a bounded grace period;
                # leave a daemon worker detached rather than allowing that
                # provider to hold the server process indefinitely.  The
                # normal migration path keeps its historical five-second
                # worker-close behavior.
                worker_timeout = 5.0
                if self.runtime_mode == "shadow":
                    worker_timeout = max(0.0, float(getattr(getattr(self.cfg, "shadow", None),
                                                           "shutdown_grace_ms", 1_000)) / 1000.0)
                self.worker.close(timeout=worker_timeout)
            except Exception:
                # Shutdown diagnostics are best effort.  In particular, a
                # stuck materializer must not prevent source/index resources
                # from being released or turn a successful serving session
                # into a noisy exception on process exit.
                pass
        finally:
            try:
                self.cache.close()
            finally:
                try:
                    if self.shadow_telemetry is not None:
                        self.shadow_telemetry.close()
                finally:
                    try:
                        # Qdrant owns a file lock/client connection; release
                        # both source and optional target indexes when the
                        # engine owns the serving session. In-memory/FAISS
                        # implementations are no-ops, and identity avoids
                        # closing a shared object twice.
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
