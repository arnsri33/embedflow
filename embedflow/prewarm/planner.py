"""Traffic-aware selection of uncached target documents."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from ..planner.economics import estimate_economics, quantity
from .models import MAX_INLINE_PREWARM_IDS, PREWARM_SCHEMA_VERSION, PrewarmPlan, PrewarmWarning


def _identity_fingerprint(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _dtype_bytes(dtype: Any) -> int:
    text = str(dtype).strip().lower().replace("-", "")
    if text in {"bfloat16", "bf16"}:
        return 2
    try:
        parsed = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"unsupported target vector dtype: {dtype!r}") from exc
    if parsed.kind not in {"f", "i", "u"} or parsed.itemsize < 1:
        raise ValueError("target vector dtype must be numeric")
    return int(parsed.itemsize)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not exact or parsed < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return parsed


def _coverage(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("target_observed_coverage must be between 0 and 1")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_observed_coverage must be between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("target_observed_coverage must be between 0 and 1")
    return parsed


def _occurrences(value: Any, *, document_id: str) -> int:
    """Validate one telemetry count instead of silently fixing corruption."""
    if isinstance(value, bool):
        raise ValueError(f"candidate_occurrences for {document_id!r} must be a non-negative integer")
    try:
        parsed = int(value or 0)
        exact = float(value or 0) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"candidate_occurrences for {document_id!r} must be a non-negative integer") from exc
    if not exact or parsed < 0:
        raise ValueError(f"candidate_occurrences for {document_id!r} must be a non-negative integer")
    return parsed


class PrewarmPlanner:
    """Build a deterministic, cache-aware traffic hot-set plan.

    The planner consumes aggregate Shadow Mode document statistics and only
    reads the target cache/document store.  It never opens or mutates a source
    vector index.  ``traffic_hotset`` is intentionally transparent: priority
    is observed candidate occurrences, then canonical document ID.
    """

    def __init__(self, telemetry: Any, cache: Any, *, config_fingerprint: str | None = None,
                 source_fingerprint: str | None = None, target_fingerprint: str | None = None,
                 backend: str | None = None, index_identity: str | None = None,
                 candidate_k: int | None = None, target_dimension: int | None = None,
                 target_dtype: str = "float32", corpus_size: int | None = None,
                 documents: Any | None = None, config: Any | None = None):
        self.telemetry = telemetry
        self.cache = cache
        self.config = config
        self.config_fingerprint = str(config_fingerprint or getattr(telemetry, "config_fingerprint", ""))
        self.source_fingerprint = source_fingerprint or getattr(telemetry, "source_fingerprint", None)
        self.target_fingerprint = target_fingerprint or getattr(telemetry, "target_fingerprint", None)
        self.backend = backend or getattr(telemetry, "backend", None)
        self.index_identity = index_identity or getattr(telemetry, "index_identity", None)
        self.candidate_k = None if candidate_k is None else _positive_int(candidate_k, "candidate_k")
        self.target_dimension = None if target_dimension is None else _positive_int(target_dimension, "target_dimension")
        self.target_dtype = str(target_dtype or "float32")
        self.corpus_size = corpus_size
        self.documents = documents
        cache_fingerprint = getattr(cache, "model_fingerprint", None)
        if cache_fingerprint and self.target_fingerprint and str(cache_fingerprint) != str(self.target_fingerprint):
            raise ValueError("target cache fingerprint does not match the configured target model")

    @classmethod
    def from_config(cls, cfg: Any, telemetry: Any, cache: Any, *, documents: Any | None = None,
                    candidate_k: int | None = None) -> PrewarmPlanner:
        index_identity = getattr(telemetry, "index_identity", None)
        if candidate_k is None:
            candidate_k = getattr(getattr(cfg, "shadow", None), "candidate_k", None)
            if candidate_k is None:
                candidate_k = getattr(getattr(cfg, "migration", None), "candidate_depth", None)
            if isinstance(candidate_k, str) and candidate_k.strip().lower() == "auto":
                candidate_k = 50
        return cls(telemetry, cache, config_fingerprint=getattr(telemetry, "config_fingerprint", None),
                    source_fingerprint=getattr(cfg.source, "fingerprint", None),
                    target_fingerprint=getattr(cfg.target, "fingerprint", None),
                    backend=getattr(cfg.index, "backend", None), index_identity=index_identity,
                    candidate_k=candidate_k, target_dimension=getattr(cfg.target, "dimension", None),
                    target_dtype=getattr(cfg.target, "dtype", "float32"),
                    corpus_size=(documents.size() if documents is not None and callable(getattr(documents, "size", None)) else None),
                    documents=documents, config=cfg)

    def _rows(self, *, start: float | None, end: float | None) -> Iterator[dict[str, Any]]:
        iterator = getattr(self.telemetry, "iter_document_stats", None)
        if callable(iterator):
            yield from iterator(start=start, end=end, config_fingerprint=self.config_fingerprint)
            return
        provider = getattr(self.telemetry, "document_stats", None)
        if not callable(provider):
            return
        yield from provider(start=start, end=end, config_fingerprint=self.config_fingerprint)

    def _current_warm(self, ids: list[str]) -> set[str]:
        """Return IDs with a valid current-target (and, when available, content) vector."""
        if not ids:
            return set()
        warm: set[str] = set()
        lookup_batch = int(getattr(self.cache, "_LOOKUP_BATCH_SIZE", 900))
        for offset in range(0, len(ids), max(1, lookup_batch)):
            chunk = ids[offset:offset + max(1, lookup_batch)]
            expected: dict[str, str] | None = None
            if self.documents is not None and callable(getattr(self.cache, "content_fingerprint", None)):
                try:
                    # Prefer the store's batch resolver.  Backend document
                    # mappings (Milvus/Weaviate/Pinecone/pgvector) implement
                    # ``Mapping`` lazily; indexing those proxies one ID at a
                    # time would turn planning into an N+1 network walk.
                    if isinstance(self.documents, Mapping):
                        expected = {document_id: self.cache.content_fingerprint(self.documents[document_id])
                                    for document_id in chunk if document_id in self.documents}
                    elif callable(getattr(self.documents, "get", None)):
                        resolved = self.documents.get(chunk)
                        if isinstance(resolved, Mapping):
                            expected = {document_id: self.cache.content_fingerprint(resolved[document_id])
                                        for document_id in chunk if document_id in resolved}
                    else:
                        mapping = getattr(self.documents, "documents", None)
                        if isinstance(mapping, Mapping):
                            expected = {document_id: self.cache.content_fingerprint(mapping[document_id])
                                        for document_id in chunk if document_id in mapping}
                except Exception:
                    expected = None
            try:
                peek = getattr(self.cache, "peek", None)
                if callable(peek):
                    try:
                        values = peek(chunk, content_fingerprints=expected) if expected is not None else peek(chunk)
                    except TypeError:
                        values = peek(chunk)
                else:
                    values = self.cache.contains(chunk)
                warm.update(str(value) for value in values)
            except Exception:
                # A corrupt/stale cache entry is cold for planning. The run
                # path will materialize a replacement or report the failure.
                continue
        return warm

    def _defaults(self) -> dict[str, Any]:
        prewarm = getattr(self.config, "prewarm", None)
        planner = getattr(self.config, "planner", None)
        economics = getattr(self.config, "economics", None)
        planner_price = getattr(planner, "gpu_hourly_cost", None)
        economics_price = getattr(economics, "gpu_price_per_hour", None)
        return {
            "max_docs": getattr(prewarm, "max_docs", 1000),
            "strategy": getattr(prewarm, "strategy", "traffic_hotset"),
            "target_observed_coverage": getattr(prewarm, "target_observed_coverage", None),
            "max_storage_gb": getattr(prewarm, "max_storage_gb", None),
            "max_runtime_seconds": getattr(prewarm, "max_runtime_seconds", None),
            "batch_size": getattr(prewarm, "batch_size", None),
            "max_retries": getattr(prewarm, "max_retries", None),
            "docs_per_second": getattr(planner, "target_docs_per_second", None) or getattr(economics, "target_docs_per_second", None),
            "gpu_hourly_cost": planner_price if planner_price is not None else economics_price,
        }

    def plan(self, *, since_seconds: float | None = None, start: float | None = None,
             end: float | None = None, max_docs: int | None = None,
             target_observed_coverage: float | None = None, max_storage_gb: float | None = None,
             max_runtime_seconds: float | None = None, docs_per_second: float | None = None,
             gpu_hourly_cost: float | None = None, strategy: str | None = None) -> PrewarmPlan:
        defaults = self._defaults()
        max_docs = defaults["max_docs"] if max_docs is None else max_docs
        max_docs = _positive_int(max_docs, "max_docs", allow_zero=True)
        if max_docs > MAX_INLINE_PREWARM_IDS:
            raise ValueError(
                f"max_docs exceeds the inline prewarm plan limit ({MAX_INLINE_PREWARM_IDS}); "
                "use smaller bounded plans until manifest support is available"
            )
        # Accept the documented CLI spelling as an alias while persisting one
        # canonical strategy value in plan artifacts.
        strategy = str(strategy or defaults["strategy"]).strip().lower().replace("-", "_")
        if strategy != "traffic_hotset":
            raise ValueError("prewarm strategy must be traffic_hotset")
        target = defaults["target_observed_coverage"] if target_observed_coverage is None else target_observed_coverage
        target = None if target is None else _coverage(target)
        max_storage_gb = defaults["max_storage_gb"] if max_storage_gb is None else max_storage_gb
        max_runtime_seconds = defaults["max_runtime_seconds"] if max_runtime_seconds is None else max_runtime_seconds
        docs_per_second = defaults["docs_per_second"] if docs_per_second is None else docs_per_second
        gpu_hourly_cost = defaults["gpu_hourly_cost"] if gpu_hourly_cost is None else gpu_hourly_cost
        if max_storage_gb is not None:
            if isinstance(max_storage_gb, bool) or not math.isfinite(float(max_storage_gb)) or float(max_storage_gb) < 0:
                raise ValueError("max_storage_gb must be finite and non-negative")
        if max_runtime_seconds is not None:
            if isinstance(max_runtime_seconds, bool) or not math.isfinite(float(max_runtime_seconds)) or float(max_runtime_seconds) < 0:
                raise ValueError("max_runtime_seconds must be finite and non-negative")
        if docs_per_second is not None:
            if isinstance(docs_per_second, bool) or not math.isfinite(float(docs_per_second)) or float(docs_per_second) <= 0:
                raise ValueError("docs_per_second must be finite and positive")
        if gpu_hourly_cost is not None:
            if isinstance(gpu_hourly_cost, bool) or not math.isfinite(float(gpu_hourly_cost)) or float(gpu_hourly_cost) < 0:
                raise ValueError("gpu_hourly_cost must be finite and non-negative")

        now = time.time()
        if since_seconds is not None:
            if isinstance(since_seconds, bool) or not math.isfinite(float(since_seconds)) or float(since_seconds) < 0:
                raise ValueError("since_seconds must be finite and non-negative")
            start, end = now - float(since_seconds), now
        start = None if start is None else float(start)
        end = now if end is None else float(end)
        for label, value in (("start", start), ("end", end)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"prewarm window {label} must be finite")
        if start is not None and end < start:
            raise ValueError("prewarm window end must be greater than or equal to start")

        item_bytes = _dtype_bytes(self.target_dtype)
        per_vector_bytes = None if self.target_dimension is None else self.target_dimension * item_bytes
        effective_cap = max_docs
        limiting_budget = "max_docs"
        if max_storage_gb is not None and per_vector_bytes is not None:
            storage_cap = int(float(max_storage_gb) * (1024 ** 3)) // max(1, per_vector_bytes)
            if storage_cap < effective_cap:
                effective_cap, limiting_budget = storage_cap, "max_storage_gb"
        if max_runtime_seconds is not None and docs_per_second is not None:
            runtime_cap = int(math.floor(float(max_runtime_seconds) * float(docs_per_second)))
            if runtime_cap < effective_cap:
                effective_cap, limiting_budget = runtime_cap, "max_runtime_seconds"
        effective_cap = max(0, int(effective_cap))

        # ``iter_document_stats`` is SQL-ordered and yields one aggregate row
        # at a time.  Resolve cache state in bounded batches and retain only
        # the top ``effective_cap`` cold rows, so a large telemetry table does
        # not become an equally large Python object graph.
        total_occurrences = 0
        warm_occurrences = 0
        row_count = 0
        unique_ids_count = 0
        warm_ids: set[str] = set()
        cold_rows: list[dict[str, Any]] = []
        batch: list[dict[str, Any]] = []

        def consume(rows_batch: list[dict[str, Any]]) -> None:
            nonlocal total_occurrences, warm_occurrences, row_count, unique_ids_count
            if not rows_batch:
                return
            if any(row.get("document_id") is None for row in rows_batch):
                raise ValueError("candidate telemetry contains a row without a document ID")
            ids = [str(row.get("document_id")) for row in rows_batch]
            warm_batch = self._current_warm(ids)
            for row in rows_batch:
                document_id = str(row.get("document_id"))
                if not document_id.strip():
                    raise ValueError("candidate telemetry contains an empty document ID")
                occurrences = _occurrences(row.get("candidate_occurrences", 0), document_id=document_id)
                # A zero-count aggregate carries no observed traffic signal;
                # treating it as a selectable hot-set member could cause an
                # empty/corrupt telemetry window to warm arbitrary documents.
                if occurrences == 0:
                    continue
                row_count += 1
                unique_ids_count += 1
                total_occurrences += occurrences
                if document_id in warm_batch:
                    warm_ids.add(document_id)
                    warm_occurrences += occurrences
                elif effective_cap:
                    cold_rows.append({**row, "document_id": document_id,
                                      "candidate_occurrences": occurrences})
                    # The provider normally sorts these rows, but retaining
                    # only the current deterministic top cap also protects
                    # injected/fake telemetry providers that do not.
                    if len(cold_rows) > effective_cap * 2 + 32:
                        cold_rows.sort(key=lambda value: (-int(value["candidate_occurrences"]),
                                                           str(value["document_id"])))
                        del cold_rows[effective_cap:]

        for row in self._rows(start=start, end=end):
            if str(row.get("config_fingerprint", self.config_fingerprint)) != self.config_fingerprint:
                continue
            batch.append(row)
            if len(batch) >= 900:
                consume(batch)
                batch = []
        consume(batch)
        cold_rows.sort(key=lambda row: (-int(row["candidate_occurrences"]), str(row["document_id"])))
        if len(cold_rows) > effective_cap:
            del cold_rows[effective_cap:]
        has_rows = row_count > 0

        warnings: list[PrewarmWarning] = []
        if not has_rows:
            warnings.append(PrewarmWarning("NO_COMPATIBLE_TRAFFIC", "WARN",
                                           "No compatible Shadow Mode candidate traffic was found in the selected window.",
                                           "Enable Shadow Mode and collect traffic for this source/target/K configuration."))
        if max_runtime_seconds is not None and docs_per_second is None:
            warnings.append(PrewarmWarning("THROUGHPUT_UNKNOWN", "WARN",
                                           "A runtime budget was supplied but target encoding throughput is unknown.",
                                           "Provide measured or user-supplied docs_per_second; the runtime budget cannot be converted to a document cap."))
        if max_storage_gb is not None and per_vector_bytes is None:
            warnings.append(PrewarmWarning("DIMENSION_UNKNOWN", "WARN",
                                           "A storage budget was supplied but target vector dimension is unknown.",
                                           "Provide target.dimension (or an introspected target model contract) before relying on storage bounds."))
        selected: list[str] = []
        selected_set: set[str] = set()
        selected_occurrences = 0
        baseline_coverage = warm_occurrences / max(1, total_occurrences)
        desired_occurrences = None if target is None else max(0, int(math.ceil(target * total_occurrences)) - warm_occurrences)
        if desired_occurrences != 0:
            for row in cold_rows:
                if len(selected) >= effective_cap:
                    break
                document_id = str(row.get("document_id"))
                if document_id in selected_set:
                    continue
                selected.append(document_id)
                selected_set.add(document_id)
                selected_occurrences += max(0, int(row.get("candidate_occurrences", 0) or 0))
                if desired_occurrences is not None and selected_occurrences >= desired_occurrences:
                    limiting_budget = "target_observed_coverage"
                    break

        projected_coverage = (warm_occurrences + selected_occurrences) / max(1, total_occurrences)
        if target is not None and projected_coverage + 1e-12 < target and has_rows:
            warnings.append(PrewarmWarning("TARGET_COVERAGE_UNMET", "WARN",
                                           f"The requested observed candidate coverage ({target:.2%}) cannot be reached within the selected budget.",
                                           "Increase max_docs/storage/runtime budget or collect more traffic."))
        if selected and len(selected) >= effective_cap and target is not None and projected_coverage < target:
            warnings.append(PrewarmWarning("BUDGET_LIMITED", "WARN",
                                           "Selection stopped at a hard prewarm budget before the requested coverage was reached.",
                                           "Review the marginal coverage curve before increasing the budget."))
        if has_rows and docs_per_second is None:
            warnings.append(PrewarmWarning("TARGET_THROUGHPUT_UNKNOWN", "INFO",
                                           "Target encoding throughput is unknown; runtime and cost remain UNKNOWN.",
                                           "Run a small measured profile or provide target_docs_per_second."))

        checkpoints = [0, 1000, 5000, 10000, 25000, 50000, 100000, len(selected)]
        checkpoints = sorted({value for value in checkpoints if 0 <= value <= len(selected)})
        curve: list[dict[str, Any]] = []
        cumulative = 0
        occurrence_by_id = {str(row.get("document_id")): _occurrences(row.get("candidate_occurrences", 0),
                                                                        document_id=str(row.get("document_id")))
                            for row in cold_rows}
        for checkpoint in checkpoints:
            if checkpoint:
                cumulative = sum(occurrence_by_id.get(document_id, 0) for document_id in selected[:checkpoint])
            curve.append({"documents": checkpoint,
                          "observed_candidate_coverage": (warm_occurrences + cumulative) / max(1, total_occurrences),
                          "candidate_occurrences_added": cumulative})

        traffic = {}
        try:
            report = self.telemetry.report(start=start, end=end, config_fingerprint=self.config_fingerprint,
                                           candidate_k=self.candidate_k)
            traffic = report.get("traffic", {}) if isinstance(report, Mapping) else {}
        except Exception:
            traffic = {}
        corpus_size = self.corpus_size
        if corpus_size is None and isinstance(self.documents, Mapping):
            corpus_size = len(self.documents)
        economics = estimate_economics(corpus_size, self.target_dimension, dtype=self.target_dtype,
                                       docs_per_second=docs_per_second, gpu_hourly_cost=gpu_hourly_cost,
                                       cached_documents=len(warm_ids))
        selected_storage = None if per_vector_bytes is None else len(selected) * per_vector_bytes
        economics["selected_documents"] = len(selected)
        economics["selected_raw_vector_storage"] = quantity(
            selected_storage, "bytes", "modeled" if selected_storage is not None else "unknown",
            assumptions=["selected documents × target dimension × dtype bytes"] if selected_storage is not None else ["target dimension is unknown"])
        economics["selected_runtime"] = quantity(
            None if docs_per_second is None else len(selected) / float(docs_per_second), "seconds",
            "modeled" if docs_per_second is not None else "unknown",
            assumptions=["linear target encoding throughput"] if docs_per_second is not None else ["target encoding throughput is unknown"])
        selected_hours = None if docs_per_second is None else len(selected) / float(docs_per_second) / 3600.0
        economics["selected_gpu_hours"] = quantity(
            selected_hours, "hours", "modeled" if selected_hours is not None else "unknown",
            assumptions=["one worker/GPU and linear target encoding throughput"] if selected_hours is not None else ["target encoding throughput is unknown"])
        selected_cost = None if selected_hours is None or gpu_hourly_cost is None else selected_hours * float(gpu_hourly_cost)
        economics["selected_cost"] = quantity(
            selected_cost, "currency", "modeled" if selected_cost is not None else "unknown",
            assumptions=["user-supplied or configured GPU hourly price"] if selected_cost is not None else ["throughput or GPU hourly price is missing"])
        if docs_per_second is not None and max_runtime_seconds is not None:
            economics["selected_runtime_budget_seconds"] = float(max_runtime_seconds)

        if not has_rows:
            status = "NO_COMPATIBLE_TRAFFIC"
        elif not selected:
            status = "NOOP"
        else:
            status = "READY"
        reason = (
            "No uncached candidate documents were observed in this window."
            if not selected and has_rows else
            "No compatible Shadow Mode candidate traffic was observed."
            if not has_rows else
            "Documents ranked by observed candidate occurrences (descending), with canonical ID tie-breaking."
        )
        migration = {
            "backend": self.backend,
            "source_model": getattr(getattr(self.config, "source", None), "model", None),
            "source_fingerprint": self.source_fingerprint,
            "target_model": getattr(getattr(self.config, "target", None), "model", None),
            "target_fingerprint": self.target_fingerprint,
            "target_dimension": self.target_dimension,
            "target_dtype": self.target_dtype,
            "index_identity_fingerprint": _identity_fingerprint(self.index_identity),
            "candidate_k": self.candidate_k,
            "telemetry_fingerprint": self.config_fingerprint,
        }
        window = {"start": start, "end": end, "shadow_observations": int(traffic.get("shadow_sampled_total", 0) or 0),
                  "bucket_seconds": 60}
        baseline = {
            "candidate_occurrences": total_occurrences,
            "unique_candidate_docs": unique_ids_count,
            "already_cached_docs": len(warm_ids),
            "warm_candidate_occurrences": warm_occurrences,
            "observed_candidate_coverage": baseline_coverage,
            "uncached_candidate_docs": max(0, unique_ids_count - len(warm_ids)),
        }
        selection = {
            "documents": len(selected), "selected_ids": selected,
            "selected_candidate_occurrences": selected_occurrences,
            "estimated_incremental_candidate_coverage": projected_coverage - baseline_coverage,
            "estimated_total_candidate_coverage": projected_coverage,
            "reason": reason, "limiting_budget": limiting_budget,
        }
        budget = {"max_docs": max_docs, "target_observed_coverage": target,
                  "max_storage_gb": max_storage_gb,
                  "max_storage_bytes": None if max_storage_gb is None else int(float(max_storage_gb) * 1024 ** 3),
                  "max_runtime_seconds": max_runtime_seconds, "effective_document_cap": effective_cap}
        assumptions = [
            "Traffic hot-set priority is candidate_occurrences, not retrieval quality or recall.",
            "Only observations matching the current migration telemetry fingerprint are included.",
            "Target cache state is checked at planning time and must be checked again at run time.",
            "Popularity windows are aggregated in one-minute buckets; boundary values are approximate to that bucket granularity.",
        ]
        return PrewarmPlan(
            schema_version=PREWARM_SCHEMA_VERSION, migration=migration, window=window,
            strategy=strategy, baseline=baseline, selection=selection, budget=budget,
            cost=economics, coverage_curve=curve, warnings=warnings,
            assumptions=assumptions, status=status,
        )


TrafficHotsetPlanner = PrewarmPlanner


__all__ = ["PrewarmPlanner", "TrafficHotsetPlanner"]
