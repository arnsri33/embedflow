"""Bounded asynchronous execution for source-authoritative Shadow Mode."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from .models import SAFE_FAILURE_CATEGORIES, ShadowObservation
from .telemetry import ShadowTelemetry


class ShadowTaskError(RuntimeError):
    """Failure raised by a shadow task with a safe public category."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        normalized = str(category).upper()
        self.category = normalized if normalized in SAFE_FAILURE_CATEGORIES else "INTERNAL"


class ShadowRunner:
    """Run shadow jobs on daemon workers with bounded queue/backpressure.

    The runner never raises task failures to the caller of :meth:`dispatch`.
    A timeout is best-effort cancellation (Python cannot kill a running
    thread), but the timed-out job is detached from the source request and
    worker capacity is reclaimed.
    """

    def __init__(self, task: Callable[[Mapping[str, Any]], Mapping[str, Any]], telemetry: ShadowTelemetry,
                 *, sample_rate: float = 0.1, sample_seed: int = 42, max_inflight: int = 32,
                 queue_capacity: int = 1000, timeout_ms: int = 10_000, shutdown_grace_ms: int = 1_000,
                 config_fingerprint: str = ""):
        if isinstance(sample_rate, bool) or not 0.0 <= float(sample_rate) <= 1.0:
            raise ValueError("shadow.sample_rate must be between 0 and 1")
        integer_values = {
            "max_inflight": max_inflight,
            "queue_capacity": queue_capacity,
            "timeout_ms": timeout_ms,
            "shutdown_grace_ms": shutdown_grace_ms,
        }
        limits = {"max_inflight": 1024, "queue_capacity": 1_000_000,
                  "timeout_ms": 3_600_000, "shutdown_grace_ms": 300_000}
        parsed_values: dict[str, int] = {}
        for name, value in integer_values.items():
            if isinstance(value, bool):
                raise ValueError(f"shadow.{name} must be a positive integer")
            try:
                parsed = int(value)
                exact = float(value) == parsed
            except (TypeError, ValueError, OverflowError):
                exact, parsed = False, 0
            if not exact or parsed < 1:
                raise ValueError(f"shadow.{name} must be a positive integer")
            if parsed > limits[name]:
                raise ValueError(f"shadow.{name} exceeds the safe limit of {limits[name]}")
            parsed_values[name] = parsed
        self.task = task
        self.telemetry = telemetry
        self.sample_rate = float(sample_rate)
        self.sample_seed = int(sample_seed)
        self.max_inflight = parsed_values["max_inflight"]
        self.queue_capacity = parsed_values["queue_capacity"]
        self.timeout_ms = parsed_values["timeout_ms"]
        self.shutdown_grace_ms = parsed_values["shutdown_grace_ms"]
        self.config_fingerprint = str(config_fingerprint)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_capacity)
        self._stop = threading.Event()
        self._telemetry_stop = threading.Event()
        self._lock = threading.RLock()
        self._active = 0
        self._detached_jobs = 0
        # A worker waits on a provider call in a small helper thread so the
        # timeout can be enforced.  Keep an explicit execution semaphore so
        # timed-out helper threads still consume their slot until they return;
        # otherwise ``max_inflight`` would describe only the queue workers and
        # a set of hung providers could silently double the actual concurrency.
        self._execution_slots = threading.BoundedSemaphore(self.max_inflight)
        self._closed = False
        self._counts: defaultdict[str, int] = defaultdict(int)
        # Telemetry persistence is deliberately decoupled from dispatch.  A
        # slow/corrupt SQLite file (or a test double that blocks forever) must
        # never sit on the primary source-response path.  The queue is bounded
        # so observability cannot become an unbounded memory sink either.
        self._telemetry_queue: queue.Queue[Any] = queue.Queue(maxsize=max(16, min(self.queue_capacity, 1000)))
        self._workers = [threading.Thread(target=self._worker, name=f"embedflow-shadow-{i}", daemon=True)
                         for i in range(self.max_inflight)]
        self._telemetry_worker = threading.Thread(target=self._telemetry_loop,
                                                   name="embedflow-shadow-telemetry", daemon=True)
        for worker in self._workers:
            worker.start()
        self._telemetry_worker.start()

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex

    def should_sample(self, request_id: str) -> bool:
        digest = hashlib.sha256(f"{self.sample_seed}:{request_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        return value < self.sample_rate

    def _count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += max(0, int(amount))

    def _emit_telemetry(self, operation: str, value: Any = None, *, timestamp: float | None = None) -> None:
        """Best-effort non-blocking handoff to the telemetry writer."""
        try:
            self._telemetry_queue.put_nowait((operation, value, timestamp))
        except queue.Full:
            self._count("telemetry_dropped_total")

    def _telemetry_loop(self) -> None:
        while True:
            try:
                item = self._telemetry_queue.get(timeout=0.05)
            except queue.Empty:
                if self._telemetry_stop.is_set():
                    return
                continue
            operation, value, timestamp = item
            try:
                if operation == "primary":
                    self.telemetry.record_primary(eligible=bool(value), timestamp=timestamp)
                elif operation == "sampled":
                    self.telemetry.record_sampled(timestamp=timestamp)
                elif operation == "dropped":
                    self.telemetry.record_dropped(timestamp=timestamp)
                elif operation == "observation":
                    self.telemetry.record_observation(value, timestamp=timestamp)
                elif operation == "candidate_documents":
                    payload = value if isinstance(value, Mapping) else {}
                    self.telemetry.record_candidate_documents(
                        payload.get("document_ids", ()),
                        timestamp=timestamp,
                        config_fingerprint=payload.get("config_fingerprint"),
                    )
                elif operation == "candidate_misses":
                    payload = value if isinstance(value, Mapping) else {}
                    self.telemetry.record_candidate_misses(
                        payload.get("document_ids", ()),
                        timestamp=timestamp,
                        config_fingerprint=payload.get("config_fingerprint"),
                    )
            except BaseException:
                # Telemetry failures are intentionally swallowed.  The local
                # counters remain available through ``stats`` and the source
                # request is never affected.
                pass
            finally:
                self._telemetry_queue.task_done()

    def dispatch(self, payload: Mapping[str, Any], *, request_id: str | None = None) -> bool:
        request_id = str(request_id or self.new_request_id())
        now = time.time()
        self._count("primary_requests_total")
        self._count("shadow_eligible_total")
        self._emit_telemetry("primary", True, timestamp=now)
        # Close and enqueue are serialized so a request racing shutdown is
        # either accepted before the close snapshot or rejected afterward;
        # it can never be stranded in a queue after workers have stopped.
        with self._lock:
            if self._closed or not self.should_sample(request_id):
                return False
            self._count("shadow_sampled_total")
            self._emit_telemetry("sampled", timestamp=now)
            item = (request_id, dict(payload), now)
            try:
                self._queue.put_nowait(item)
                candidate_ids = payload.get("source_ids", payload.get("candidate_ids", ()))
                self._emit_telemetry(
                    "candidate_documents",
                    {"document_ids": tuple(str(value) for value in candidate_ids),
                     "config_fingerprint": self.config_fingerprint},
                    timestamp=now,
                )
                return True
            except queue.Full:
                self._count("shadow_dropped_total")
                self._emit_telemetry("dropped", timestamp=now)
                return False

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                return
            request_id, payload, submitted_at = item
            with self._lock:
                self._active += 1
            try:
                observation = self._execute(request_id, payload, submitted_at)
                self._account_observation(observation)
                self._emit_telemetry("observation", observation)
            except BaseException:  # telemetry/runner errors stay isolated
                # Do not persist third-party exception payloads: encoders and
                # document stores occasionally include query/document text in
                # their messages.  Category-level diagnostics are sufficient
                # for a privacy-preserving shadow report.
                observation = ShadowObservation(status="failed", config_fingerprint=self.config_fingerprint,
                                                request_id=request_id, failure_category="INTERNAL", error="shadow task failed")
                self._account_observation(observation)
                self._emit_telemetry("observation", observation)
            finally:
                with self._lock:
                    self._active = max(0, self._active - 1)
                self._queue.task_done()

    def _release_detached(self) -> None:
        with self._lock:
            self._detached_jobs = max(0, self._detached_jobs - 1)

    def _account_observation(self, observation: ShadowObservation) -> None:
        status = str(observation.status).lower()
        counter = {"completed": "shadow_completed_total", "partial": "shadow_partial_total",
                   "failed": "shadow_failed_total", "timeout": "shadow_timeout_total",
                   "timed_out": "shadow_timeout_total"}.get(status)
        if counter:
            self._count(counter)
        missing_ids = getattr(observation, "missing_candidate_ids", ())
        if missing_ids:
            self._emit_telemetry(
                "candidate_misses",
                {"document_ids": tuple(str(value) for value in missing_ids),
                 "config_fingerprint": self.config_fingerprint},
            )
        for name, value in (("target_cache_hits", observation.cache_hits),
                            ("target_cache_misses", observation.cache_misses),
                            ("target_docs_queued", observation.docs_queued),
                            ("target_unique_docs_queued", observation.unique_docs_queued),
                            ("target_docs_materialized", observation.docs_materialized),
                            ("target_docs_failed", observation.docs_failed)):
            self._count(name, int(value or 0))

    def _execute(self, request_id: str, payload: Mapping[str, Any], submitted_at: float) -> ShadowObservation:
        started = time.perf_counter()
        holder: dict[str, Any] = {}
        done = threading.Event()
        completion_lock = threading.Lock()
        completion = {"finished": False, "timed_out": False}
        source_latency = payload.get("source_latency_ms")
        source_candidate_latency = payload.get("source_candidate_latency_ms")

        def failed(status: str, category: str, error: str) -> ShadowObservation:
            return ShadowObservation(status=status, config_fingerprint=self.config_fingerprint,
                                     request_id=request_id, source_latency_ms=source_latency,
                                     source_candidate_latency_ms=source_candidate_latency,
                                     shadow_latency_ms=(time.perf_counter() - started) * 1000.0,
                                     failure_category=category, error=error)

        def invoke() -> None:
            try:
                holder["value"] = self.task(payload)
            except BaseException as exc:
                holder["error"] = exc
            finally:
                with completion_lock:
                    completion["finished"] = True
                    done.set()
                    timed_out = bool(completion["timed_out"])
                if timed_out:
                    self._release_detached()
                self._execution_slots.release()

        # A timed-out Python thread cannot be force-killed safely.  The
        # semaphore is acquired before invoking the provider and released only
        # after the helper really returns, so detached calls remain bounded and
        # do not allow a second hidden pool of work to grow without limit.
        if not self._execution_slots.acquire(blocking=False):
            return failed("timeout", "TIMEOUT", "shadow capacity is occupied by timed-out jobs")
        try:
            thread = threading.Thread(target=invoke, name="embedflow-shadow-job", daemon=True)
            thread.start()
        except Exception:
            self._execution_slots.release()
            return failed("failed", "INTERNAL", "shadow task could not be started")
        if not done.wait(self.timeout_ms / 1000.0):
            with completion_lock:
                if not completion["finished"]:
                    completion["timed_out"] = True
                    with self._lock:
                        self._detached_jobs += 1
            return failed("timeout", "TIMEOUT", "shadow task exceeded configured timeout")
        error = holder.get("error")
        if error is not None:
            category = getattr(error, "category", "INTERNAL")
            normalized = str(category).upper()
            if normalized not in SAFE_FAILURE_CATEGORIES:
                normalized = "INTERNAL"
            return failed("failed", normalized, f"{normalized} failed")
        value = holder.get("value")
        if not isinstance(value, Mapping):
            return failed("failed", "INTERNAL", "shadow task returned malformed result")
        data = dict(value)
        status = str(data.pop("status", "completed")).lower()
        if status not in {"completed", "partial", "failed", "timeout", "timed_out"}:
            status = "failed"
        data.update({"status": status, "config_fingerprint": self.config_fingerprint,
                     "request_id": request_id,
                     "shadow_latency_ms": float(data.get("shadow_latency_ms", (time.perf_counter() - started) * 1000.0))})
        return ShadowObservation(**{key: data[key] for key in ShadowObservation.__dataclass_fields__ if key in data})

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            detached = self._detached_jobs
        return {"active": active, "max_inflight": self.max_inflight,
                "detached_jobs": detached,
                "queue_depth": self._queue.qsize(), "queue_capacity": self.queue_capacity,
                "sample_rate": self.sample_rate, "sample_seed": self.sample_seed,
                "timeout_ms": self.timeout_ms, "closed": self._closed,
                "workers_alive": sum(worker.is_alive() for worker in self._workers),
                "telemetry_queue_depth": self._telemetry_queue.qsize(),
                "telemetry_dropped_total": self._counts.get("telemetry_dropped_total", 0),
                "counters": dict(self._counts)}

    def close(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Cancel queued work and account for it as dropped. Running jobs are
        # allowed only the bounded grace period; daemon threads prevent a
        # stuck third-party encoder from hanging process shutdown.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
                self._count("shadow_dropped_total")
                self._emit_telemetry("dropped")
        self._stop.set()
        grace = self.shutdown_grace_ms / 1000.0 if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + grace
        for worker in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(remaining)
        self._telemetry_stop.set()
        remaining = max(0.0, deadline - time.monotonic())
        self._telemetry_worker.join(remaining)


__all__ = ["ShadowRunner", "ShadowTaskError"]
