"""Bounded execution of traffic-aware prewarm plans."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..migration.materializer import MaterializationWorker
from .models import MAX_INLINE_PREWARM_IDS, PrewarmPlan


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False))
    temporary.replace(path)


def _identity_fingerprint(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


class PrewarmRunner:
    """Materialize only the IDs selected by one validated plan artifact.

    The existing :class:`MaterializationWorker` and :class:`SQLiteVectorCache`
    remain the only target encoding/write path.  A plan-specific queue makes
    retries/resume durable without mixing unrelated serving work.
    """

    def __init__(self, target_model: Any, documents: Any, cache: Any, *,
                 queue_path: str | Path | None = None, batch_size: int = 32,
                 max_retries: int = 3, source_fingerprint: str | None = None,
                 target_fingerprint: str | None = None, backend: str | None = None,
                 index_identity: str | None = None, candidate_k: int | None = None,
                 config_fingerprint: str | None = None, state_dir: str | Path | None = None):
        self.target_model = target_model
        self.documents = documents
        self.cache = cache
        cache_root = Path(getattr(cache, "root", Path.cwd()))
        self.state_dir = Path(state_dir) if state_dir else cache_root / "prewarm"
        self.queue_path = Path(queue_path) if queue_path else None
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.source_fingerprint = source_fingerprint
        self.target_fingerprint = target_fingerprint or getattr(target_model, "fingerprint", None)
        self.backend = backend
        self.index_identity = index_identity
        self.candidate_k = candidate_k
        self.config_fingerprint = config_fingerprint
        if self.batch_size < 1 or self.max_retries < 1:
            raise ValueError("prewarm batch_size and max_retries must be positive")

    @classmethod
    def from_engine(cls, engine: Any) -> PrewarmRunner:
        cfg = engine.cfg
        prewarm = getattr(cfg, "prewarm", None)
        # ``open_engine(..., start_worker=False)`` is a valid prewarm-run
        # entry point, so there may be no live ShadowTelemetry object to
        # borrow.  Recompute the same credential-free migration fingerprint
        # used by Shadow Mode from the loaded model contracts/configuration.
        telemetry = getattr(engine, "shadow_telemetry", None)
        config_fingerprint = getattr(telemetry, "config_fingerprint", None)
        index_identity = getattr(telemetry, "index_identity", None)
        if not config_fingerprint:
            try:
                from ..shadow import index_identity_from_config, migration_fingerprint

                raw_identity = index_identity_from_config(cfg.index)
                candidate = getattr(cfg.shadow, "candidate_k", None) or getattr(engine, "plan", None).candidate_depth
                if isinstance(candidate, str):
                    candidate = 50
                config_fingerprint = migration_fingerprint(
                    source_fingerprint=engine.source_model.fingerprint,
                    target_fingerprint=engine.target_model.fingerprint,
                    backend=str(cfg.index.backend), index_identity=raw_identity,
                    candidate_k=int(candidate),
                )
                # ShadowTelemetry stores a one-way digest of the raw index
                # identity.  Prewarm plan artifacts hash that stored value
                # once more; mirror it here so a plan made with Shadow mode
                # can be executed from the normal migration runtime too.
                index_identity = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
            except Exception:
                # Validation remains conservative: if the runtime cannot
                # reconstruct an identity, a plan with an explicit identity
                # will be rejected rather than silently mixed.
                config_fingerprint = None
        effective_candidate_k = getattr(cfg.shadow, "candidate_k", None)
        if effective_candidate_k is None:
            effective_candidate_k = getattr(getattr(engine, "plan", None), "candidate_depth", None)
        if effective_candidate_k is None:
            effective_candidate_k = cfg.migration.candidate_depth
        if isinstance(effective_candidate_k, str) and effective_candidate_k.strip().lower() == "auto":
            effective_candidate_k = 50
        return cls(
            engine.target_model, engine.documents, engine.cache,
            batch_size=int(getattr(prewarm, "batch_size", None) or cfg.migration.background_batch_size),
            max_retries=int(getattr(prewarm, "max_retries", None) or cfg.migration.max_retries),
            source_fingerprint=getattr(engine.source_model, "fingerprint", None),
            target_fingerprint=getattr(engine.target_model, "fingerprint", None),
            backend=getattr(cfg.index, "backend", None),
            index_identity=index_identity,
            candidate_k=effective_candidate_k,
            config_fingerprint=config_fingerprint,
        )

    def _expected_content(self, chunk: list[str]) -> dict[str, str] | None:
        """Resolve current document fingerprints without trusting stale cache rows."""
        if not callable(getattr(self.cache, "content_fingerprint", None)) or self.documents is None:
            return None
        values: Mapping[str, Any] | None = None
        # Use the store's batch resolver before looking at its ``documents``
        # attribute.  Lazy backend mappings implement Mapping but each
        # ``mapping[id]`` call is a remote request, so preferring that path
        # would create an avoidable N+1 fetch during every prewarm run.
        if isinstance(self.documents, Mapping):
            values = {str(document_id): self.documents[document_id] for document_id in chunk if document_id in self.documents}
        elif callable(getattr(self.documents, "get", None)):
            try:
                resolved = self.documents.get(chunk)
                if isinstance(resolved, Mapping):
                    values = {str(key): value for key, value in resolved.items()}
            except Exception:
                # A strict store may reject a batch containing one deleted
                # document.  Resolve only that exceptional batch one ID at a
                # time so known documents still get content-safe cache checks;
                # this is never used by a healthy batched run.
                partial: dict[str, Any] = {}
                for document_id in chunk:
                    try:
                        resolved = self.documents.get([document_id])
                        if isinstance(resolved, Mapping) and document_id in resolved:
                            partial[document_id] = resolved[document_id]
                    except Exception:
                        continue
                values = partial
        else:
            mapping = getattr(self.documents, "documents", None)
            if isinstance(mapping, Mapping):
                values = {str(document_id): mapping[document_id] for document_id in chunk if document_id in mapping}
        if values is None:
            return None
        return {str(document_id): self.cache.content_fingerprint(text)
                for document_id, text in values.items()}

    def _warm(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        warm: set[str] = set()
        lookup_batch = int(getattr(self.cache, "_LOOKUP_BATCH_SIZE", 900))
        for offset in range(0, len(ids), max(1, lookup_batch)):
            chunk = ids[offset:offset + max(1, lookup_batch)]
            expected = self._expected_content(chunk)
            try:
                peek = getattr(self.cache, "peek", None)
                if callable(peek):
                    try:
                        rows = peek(chunk, content_fingerprints=expected) if expected is not None else peek(chunk)
                    except TypeError:
                        rows = peek(chunk)
                else:
                    rows = self.cache.contains(chunk)
                warm.update(str(value) for value in rows)
            except Exception:
                # Corrupt vectors are cold and will be replaced by the worker.
                continue
        return warm

    def validate_plan(self, plan: PrewarmPlan) -> None:
        if not isinstance(plan, PrewarmPlan):
            raise TypeError("plan must be a PrewarmPlan")
        if plan.schema_version != 1:
            raise ValueError("unsupported prewarm plan schema_version")
        if plan.strategy != "traffic_hotset":
            raise ValueError("unsupported prewarm strategy")
        selected = plan.selected_ids
        if len(selected) > MAX_INLINE_PREWARM_IDS:
            raise ValueError("prewarm plan contains too many inline IDs; use a bounded manifest")
        if len(selected) != len(set(selected)):
            raise ValueError("prewarm plan contains duplicate document IDs")
        declared_documents = plan.selection.get("documents")
        if declared_documents is not None:
            try:
                declared_int = int(declared_documents)
                exact = float(declared_documents) == declared_int
            except (TypeError, ValueError, OverflowError):
                declared_int, exact = -1, False
            if isinstance(declared_documents, bool) or not exact or declared_int != len(selected):
                raise ValueError("prewarm plan selection.documents must match selected_ids")
        maximum = plan.budget.get("max_docs")
        if maximum is None:
            raise ValueError("prewarm plan must declare a hard max_docs budget")
        if maximum is not None:
            try:
                maximum_int = int(maximum)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("prewarm plan max_docs budget is invalid") from exc
            if isinstance(maximum, bool) or maximum_int < 0 or float(maximum) != maximum_int:
                raise ValueError("prewarm plan max_docs budget is invalid")
            if len(selected) > maximum_int:
                raise ValueError("prewarm plan exceeds its max_docs budget")
        migration = plan.migration
        # A plan is an executable artifact, not merely a ranked ID list.  All
        # identity bindings are required at run time so a hand-edited or
        # incomplete artifact cannot silently be applied to another migration.
        required_bindings = {
            "target_fingerprint": migration.get("target_fingerprint"),
            "source_fingerprint": migration.get("source_fingerprint"),
            "backend": migration.get("backend"),
            "telemetry_fingerprint": migration.get("telemetry_fingerprint"),
            "index_identity_fingerprint": migration.get("index_identity_fingerprint"),
        }
        missing_bindings = [name for name, value in required_bindings.items() if value is None or not str(value).strip()]
        if missing_bindings:
            raise ValueError("prewarm plan is missing required migration binding(s): " + ", ".join(missing_bindings))
        checks = (
            ("target model", migration.get("target_fingerprint"), self.target_fingerprint),
            ("source migration", migration.get("source_fingerprint"), self.source_fingerprint),
            ("backend", migration.get("backend"), self.backend),
            ("telemetry", migration.get("telemetry_fingerprint"), self.config_fingerprint),
        )
        for label, expected, actual in checks:
            if actual is None or not str(actual).strip():
                raise ValueError(f"current {label} identity is unavailable; refusing to execute the prewarm plan")
            if str(expected) != str(actual):
                raise ValueError(f"prewarm plan {label} fingerprint/configuration does not match current configuration")
        planned_identity = migration.get("index_identity_fingerprint")
        if self.index_identity is None or not str(self.index_identity).strip():
            raise ValueError("current source index identity is unavailable; refusing to execute the prewarm plan")
        if planned_identity != _identity_fingerprint(self.index_identity):
            raise ValueError("prewarm plan source index identity does not match current configuration")
        planned_target = migration.get("target_fingerprint")
        cache_fp = getattr(self.cache, "model_fingerprint", None)
        if cache_fp is None or not str(cache_fp).strip():
            raise ValueError("target cache model fingerprint is unavailable; refusing to execute the prewarm plan")
        if str(planned_target) != str(cache_fp):
            raise ValueError("prewarm plan target fingerprint does not match target cache")
        planned_k = migration.get("candidate_k")
        if planned_k is not None:
            try:
                planned_k_int = int(planned_k)
                if float(planned_k) != planned_k_int or planned_k_int < 1:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                raise ValueError("prewarm plan candidate K is invalid") from None
            if self.candidate_k is not None and planned_k_int != int(self.candidate_k):
                raise ValueError("prewarm plan candidate K does not match current migration configuration")

    def _state_path(self, plan: PrewarmPlan) -> Path:
        return self.state_dir / f"{plan.fingerprint}.state.json"

    def run(self, plan: PrewarmPlan, *, max_runtime_seconds: float | None = None,
            progress: Any | None = None) -> dict[str, Any]:
        self.validate_plan(plan)
        if max_runtime_seconds is not None:
            try:
                runtime_limit = float(max_runtime_seconds)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_runtime_seconds must be finite and non-negative") from exc
            if not math.isfinite(runtime_limit) or runtime_limit < 0:
                raise ValueError("max_runtime_seconds must be finite and non-negative")
        else:
            runtime_limit = None
        selected = plan.selected_ids
        started = time.monotonic()
        state_path = self._state_path(plan)
        state: dict[str, Any] = {
            "schema_version": 1, "plan_fingerprint": plan.fingerprint,
            "status": "STARTING", "selected": len(selected), "encoded": 0,
            "already_warm": 0, "failed": 0, "remaining": len(selected),
            "updated_at": time.time(),
        }
        _atomic_json(state_path, state)
        initially_warm = self._warm(selected)
        cold = [document_id for document_id in selected if document_id not in initially_warm]
        state.update({"status": "RUNNING", "already_warm": len(initially_warm), "remaining": len(cold),
                      "started_at": time.time(), "updated_at": time.time()})
        _atomic_json(state_path, state)
        if not cold:
            result = self._result(plan, started, len(initially_warm), 0, 0, 0, "NOOP", state_path)
            state.update(result); state["updated_at"] = time.time(); _atomic_json(state_path, state)
            return result

        queue_path = self.queue_path or (self.state_dir / f"{plan.fingerprint}.queue.sqlite3")
        try:
            worker = MaterializationWorker(self.target_model, self.documents, self.cache, queue_path,
                                           batch_size=self.batch_size, max_retries=self.max_retries)
        except Exception as exc:
            # Persist a categorical terminal state before returning the
            # initialization error.  The original exception may contain a
            # credential-bearing path/provider payload, so do not copy it to
            # the durable run state.
            state.update({"status": "FAILED", "failed": len(cold), "remaining": len(cold),
                          "error": f"materializer initialization failed ({type(exc).__name__})",
                          "updated_at": time.time()})
            _atomic_json(state_path, state)
            raise RuntimeError("prewarm materializer could not be initialized") from exc
        timed_out = False
        try:
            worker.enqueue(cold)
            # ``done`` rows from an earlier plan run are only reusable when
            # the current cache still has a matching content fingerprint.
            # Requeue terminal stale rows while preserving active work owned
            # by another runner.
            requeue = getattr(worker.queue, "requeue_stale", None)
            if callable(requeue):
                requeue(cold)
            worker.start()
            while True:
                queue_stats = worker.queue.stats()
                warm_now = self._warm(selected)
                remaining = len(selected) - len(warm_now)
                failed = int(queue_stats.get("error", 0))
                state.update({"status": "RUNNING", "encoded": max(0, len(warm_now) - len(initially_warm)),
                              "already_warm": len(initially_warm), "failed": failed, "remaining": remaining,
                              "queue": queue_stats, "updated_at": time.time()})
                _atomic_json(state_path, state)
                if callable(progress):
                    progress(dict(state))
                if int(queue_stats.get("pending", 0)) + int(queue_stats.get("processing", 0)) == 0:
                    break
                if runtime_limit is not None and time.monotonic() - started >= runtime_limit:
                    timed_out = True
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            state.update({"status": "INTERRUPTED", "remaining": len(selected) - len(self._warm(selected)),
                          "updated_at": time.time()})
            _atomic_json(state_path, state)
            raise
        except Exception as exc:
            state.update({"status": "FAILED", "remaining": len(selected) - len(self._warm(selected)),
                          "error": f"prewarm execution failed ({type(exc).__name__})",
                          "updated_at": time.time()})
            _atomic_json(state_path, state)
            # Provider/document-store exceptions can contain private text,
            # DSNs, or credentials.  Persist only the categorical state above
            # and expose a stable, sanitized CLI/API error.
            raise RuntimeError(f"prewarm execution failed ({type(exc).__name__})") from exc
        finally:
            try:
                worker.close(timeout=1.0)
            except Exception:
                # ``MaterializationWorker.close`` raises when a provider is
                # still running after the bounded grace period.  Do not close
                # its SQLite connection underneath the live worker: that can
                # turn a slow but finite encode into a queue corruption race.
                # The worker thread is daemonized and will close/reconcile on
                # its own once the provider returns; a later run can reclaim
                # any processing row through the durable queue startup logic.
                if not (worker.thread and worker.thread.is_alive()):
                    try:
                        worker.queue.close()
                    except Exception:
                        pass
                elif worker.thread is not None:
                    # Reclaim the queue connection once the daemon worker has
                    # naturally unwound.  This watcher itself is bounded to
                    # one daemon thread per unusually slow shutdown and does
                    # not create additional provider work.
                    def close_after_worker() -> None:
                        try:
                            worker.thread.join()
                            worker.queue.close()
                        except Exception:
                            pass
                    threading.Thread(target=close_after_worker,
                                     name="embedflow-prewarm-queue-close", daemon=True).start()
        final_warm = self._warm(selected)
        queue_stats = {}
        try:
            queue_stats = worker.queue.stats()
        except Exception:
            queue_stats = state.get("queue", {})
        encoded = max(0, len(final_warm) - len(initially_warm))
        failed = int(queue_stats.get("error", state.get("failed", 0)) or 0)
        remaining = len(selected) - len(final_warm)
        status = "INTERRUPTED" if timed_out else ("COMPLETED" if remaining == 0 and failed == 0 else "PARTIAL")
        result = self._result(plan, started, len(initially_warm), encoded, failed, remaining, status, state_path)
        result["queue"] = queue_stats
        state.update(result); state["updated_at"] = time.time(); _atomic_json(state_path, state)
        return result

    def _result(self, plan: PrewarmPlan, started: float, already_warm: int, encoded: int,
                failed: int, remaining: int, status: str, state_path: Path) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - started)
        throughput = encoded / elapsed if encoded and elapsed > 0 else None
        before = plan.baseline.get("observed_candidate_coverage")
        after = plan.selection.get("estimated_total_candidate_coverage") if remaining == 0 else None
        return {
            "schema_version": 1, "status": status, "plan_fingerprint": plan.fingerprint,
            "planned_documents": len(plan.selected_ids), "already_warm": already_warm,
            "encoded": encoded, "failed": failed, "remaining": remaining,
            "elapsed_seconds": elapsed,
            "measured_docs_per_second": throughput,
            "observed_candidate_coverage_before": before,
            "observed_candidate_coverage_after": after,
            "projected_plan_coverage": plan.selection.get("estimated_total_candidate_coverage"),
            "state_path": str(state_path),
            "assumptions": ["Coverage after is reported only when every selected document is warm; it is observed candidate-occurrence coverage, not retrieval quality."],
        }


def load_prewarm_plan(path: str | Path) -> PrewarmPlan:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # YAML is accepted for symmetry with the plan renderer.  JSON remains
        # the recommended executable artifact because it has no implicit type
        # resolution surprises, but both formats pass through the same strict
        # schema/fingerprint validation below.
        try:
            import yaml
        except ImportError as yaml_exc:
            raise ValueError(f"invalid prewarm plan JSON/YAML: {path}") from yaml_exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as yaml_exc:
            raise ValueError(f"invalid prewarm plan JSON/YAML: {path}") from yaml_exc
    return PrewarmPlan.from_dict(raw)


def prewarm_status(cache: Any, state_dir: str | Path | None = None) -> dict[str, Any]:
    """Return cache and durable prewarm run state without opening a source index."""
    root = Path(state_dir) if state_dir else Path(getattr(cache, "root", Path.cwd())) / "prewarm"
    states: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.glob("*.state.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                states.append(value)
    return {"schema_version": 1, "cache": cache.stats(), "state_dir": str(root),
            "runs": states[-20:]}


__all__ = ["PrewarmRunner", "load_prewarm_plan", "prewarm_status"]
