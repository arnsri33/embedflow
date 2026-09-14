"""Reusable advisory migration planner.

The planner composes existing EmbedFlow components: backend preflight,
registry matching, the frozen T2-v1 probe, the target-vector cache contract,
latency telemetry, and the economics formulas.  It does not route traffic or
write to a source index.  Probe target vectors are kept in a planner-local
temporary cache by default so ``embedflow plan`` cannot pollute serving state.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import SQLiteVectorCache
from ..config import EmbedFlowConfig, PlannerConfig, from_dict, load_config, save_config
from ..metrics import summarize
from ..migration.compatibility import run_probe
from ..registry import MATCH_EXACT, MATCH_NONE, MATCH_PRIOR, MATCH_RELATED, EvidenceRecord, load_evidence, match_config
from ..runtime import open_engine
from .economics import estimate_economics, quantity
from .models import PlanResult, PlanWarning

T2_SAFE = "SAFE"
T2_EXPAND = "EXPAND"
T2_UNCERTAIN = "UNSAFE_OR_UNCERTAIN"
T2_NOT_RUN = "NOT_RUN"
RECOMMENDATIONS = {"PROCEED", "PROCEED_WITH_CAUTION", "EXPAND_PROBE", "DEFER", "BLOCKED"}
DEFAULT_K_GRID = (20, 50, 100, 200, 500)
T2_MAX_K = 500
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "PASSWORD", "SECRET", "DSN")


def _warning(code: str, severity: str, message: str, remediation: str) -> PlanWarning:
    return PlanWarning(code, severity, message, remediation)


def _redact_text(value: Any) -> str:
    """Redact environment-backed credentials from diagnostics and metadata."""
    text = str(value)
    for name, secret in os.environ.items():
        if secret and len(secret) >= 4 and any(marker in name.upper() for marker in _SECRET_ENV_MARKERS):
            text = text.replace(secret, "<redacted>")
            # A database/endpoint environment variable may contain a full
            # URI while an SDK exception echoes only its password component.
            # Scrub that component independently as well; otherwise a
            # diagnostic containing just ``password_fragment`` could bypass
            # the full-value replacement above.
            if "://" in secret:
                match = re.search(r"://[^:/\s]+:([^@/\s]+)@", secret)
                if match and len(match.group(1)) >= 4:
                    text = text.replace(match.group(1), "<redacted>")
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)=([^\s,;]+)", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(://[^:/\s]+:)[^@/\s]+(@)", r"\1<redacted>\2", text)
    return text


def _safe(value: Any) -> Any:
    """Convert SDK/numpy values to JSON-safe values without leaking secrets."""
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        sensitive = {"password", "token", "secret", "authorization", "api_key", "apikey"}
        for key, child in value.items():
            name = str(key).lower().replace("-", "_")
            sensitive_name = (
                name in sensitive
                or any(marker in name for marker in ("password", "token", "secret", "authorization"))
                or ("api_key" in name and not name.endswith("_env"))
            )
            if sensitive_name and not name.endswith("_env"):
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = _safe(child)
        return output
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_text(value)


def _encode_query(model: Any, text: str) -> np.ndarray:
    method = getattr(model, "encode_query", None)
    if callable(method):
        value = method(text)
    else:
        value = model.encode_queries([text])[0]
    return np.asarray(value, dtype="float32")


def _encode_documents(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    method = getattr(model, "encode_documents", None)
    if not callable(method):
        raise TypeError("target model does not expose encode_documents")
    try:
        value = method(list(texts), batch_size=batch_size)
    except TypeError as exc:
        # A small user-supplied test/model adapter may implement the protocol
        # without the optional keyword. Preserve compatibility without hiding
        # unrelated TypeErrors raised from the encoder itself.
        if "batch_size" not in str(exc):
            raise
        value = method(list(texts))
    return np.asarray(value, dtype="float32")


def _validate_nonnegative(value: Any, label: str, *, upper: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0 or (upper is not None and parsed > upper):
        suffix = f" <= {upper}" if upper is not None else ""
        raise ValueError(f"{label} must be a finite non-negative number{suffix}")
    return parsed


def _validate_positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
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


def _validate_integer(value: Any, label: str) -> int:
    """Validate an integer-valued option whose seed may be negative."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not exact:
        raise ValueError(f"{label} must be an integer")
    return parsed


def load_probe_queries(value: str | Path | Iterable[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """Load and validate the small JSONL probe format used by ``plan``.

    Each row is ``{"query": "..."}``, with optional ``id``/``query_id``;
    ``text`` is accepted as the same alias used by the existing analysis CLI.
    Full query text is never copied into a :class:`PlanResult`.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid query JSON at {path}:{number}") from exc
                if not isinstance(raw, Mapping):
                    raise ValueError(f"query row {number} in {path} must be a JSON object")
                raw_query_id = raw.get("id", raw.get("query_id", number))
                if raw_query_id is None or (isinstance(raw_query_id, str) and not raw_query_id.strip()):
                    raise ValueError(f"query row {number} in {path} has an empty id")
                query_id = str(raw_query_id)
                text = raw.get("query", raw.get("text"))
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"query {query_id!r} at {path}:{number} has no non-empty query text")
                if query_id in seen:
                    raise ValueError(f"duplicate query ID {query_id!r} in {path}")
                seen.add(query_id)
                rows.append((query_id, text))
        # An explicitly empty probe file is equivalent to omitting probes: the
        # planner can still perform preflight, registry, and economics work,
        # but it must report T2-v1 as NOT_RUN.
        return rows
    try:
        rows = list(value)
    except TypeError as exc:
        raise ValueError("probe queries must be a JSONL path or iterable of (query_id, text) pairs") from exc
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, item in enumerate(rows, 1):
        if isinstance(item, Mapping):
            raw_query_id = item.get("id", item.get("query_id", number))
            if raw_query_id is None or (isinstance(raw_query_id, str) and not raw_query_id.strip()):
                raise ValueError(f"query row {number} has an empty id")
            query_id = str(raw_query_id)
            text = item.get("query", item.get("text"))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            if item[0] is None or (isinstance(item[0], str) and not item[0].strip()):
                raise ValueError(f"query row {number} has an empty id")
            query_id, text = str(item[0]), item[1]
        else:
            raise ValueError("probe queries must contain (query_id, text) pairs")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"query {query_id!r} has no non-empty query text")
        if query_id in seen:
            raise ValueError(f"duplicate query ID {query_id!r}")
        seen.add(query_id)
        output.append((query_id, text))
    # Keep the no-probe API path useful for callers that build a list
    # programmatically; an empty list intentionally means preflight-only.
    return output


def _sample_queries(rows: Sequence[tuple[str, str]], max_probes: int, seed: int) -> list[tuple[str, str]]:
    if isinstance(max_probes, bool) or int(max_probes) != max_probes or int(max_probes) < 1:
        raise ValueError("max_probes must be a positive integer")
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[: int(max_probes)]


def _deduplicate_probe_content(rows: Sequence[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """Remove repeated probe text from the analysis sample.

    Query IDs remain the caller's identity boundary, so duplicate IDs are
    rejected by :func:`load_probe_queries`.  Repeated text with different IDs
    is allowed as input, but it must not masquerade as independent semantic
    evidence or multiply target-encoding work.  Keep the first deterministic
    ID and report how many repeated rows were removed.
    """
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    duplicates = 0
    for query_id, text in rows:
        key = text.strip()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append((query_id, text))
    return unique, duplicates


def _k_grid(values: Iterable[int], *, maximum: int | None = None) -> list[int]:
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError("k_grid must be an iterable of positive integers") from exc
    if not raw:
        raise ValueError("k_grid must contain at least one positive integer")
    output: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("k_grid must contain positive integers")
        try:
            parsed = int(value)
            exact = float(value) == parsed
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("k_grid must contain positive integers") from exc
        if not exact or parsed < 1:
            raise ValueError("k_grid must contain positive integers")
        output.append(parsed)
    if len(set(output)) != len(output):
        raise ValueError("k_grid must not contain duplicates")
    output = sorted(output)
    if maximum is not None:
        output = [value for value in output if value <= int(maximum)]
    return output


def _quantity_values(records: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any] | None:
    values: list[float] = []
    for record in records:
        try:
            value = float(record.get(stage, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    if not values:
        return None
    summary = summarize(values)
    return {
        "p50": quantity(summary["p50_ms"], "ms", "measured", assumptions=[f"{len(values)} telemetry records"]),
        "p95": quantity(summary["p95_ms"], "ms", "measured", assumptions=[f"{len(values)} telemetry records"]),
        "count": len(values),
    }


def _read_latency(path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path:
        return [], []
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    malformed += 1
    except OSError:
        return [], ["latency telemetry could not be read"]
    warnings = [f"{malformed} malformed latency telemetry rows were ignored"] if malformed else []
    return rows, warnings


def _load_access_trace(value: str | Path | None) -> dict[str, int]:
    if value is None:
        return {}
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid access trace JSON at {path}:{number}") from exc
            if not isinstance(row, Mapping) or "document_id" not in row:
                raise ValueError(f"access trace row {number} must contain document_id")
            document_id = str(row["document_id"])
            if not document_id.strip():
                raise ValueError(f"access trace row {number} must contain a non-empty document_id")
            raw_count = row.get("count", 1)
            if isinstance(raw_count, bool):
                raise ValueError(f"access trace count at {path}:{number} must be a positive integer")
            try:
                count = int(raw_count)
                exact = float(raw_count) == count
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"access trace count at {path}:{number} must be a positive integer") from exc
            if not exact or count < 1:
                raise ValueError(f"access trace count at {path}:{number} must be a positive integer")
            counts[document_id] = counts.get(document_id, 0) + count
    return counts


class _CachedVectorMapping(Mapping[str, np.ndarray]):
    """A mapping view over the isolated planner cache.

    ``run_probe`` requests vectors by canonical ID as it walks each query's
    candidates.  Keeping this view backed by SQLite avoids retaining every
    target vector in a Python dictionary when a large probe set has many
    unique candidates; the existing cache contract still validates dimension,
    checksum, finiteness, and model fingerprint on every read.
    """

    def __init__(self, cache: SQLiteVectorCache):
        self._cache = cache
        self._pending: dict[str, np.ndarray] = {}

    def __getitem__(self, document_id: str) -> np.ndarray:
        key = str(document_id)
        pending = self._pending.pop(key, None)
        if pending is not None:
            return pending
        values = self._cache.get([key])
        if key not in values:
            raise KeyError(key)
        return values[key]

    def __iter__(self):  # pragma: no cover - run_probe uses membership/index access
        return iter(())

    def __len__(self) -> int:
        return int(self._cache.stats().get("cached_target_vectors", 0))

    def __contains__(self, document_id: object) -> bool:
        key = str(document_id)
        values = self._cache.get([key])
        if key in values:
            # ``run_probe`` checks membership and then indexes the mapping;
            # retain that one value to avoid a second local lookup while
            # keeping the overall vector store disk-backed.
            self._pending[key] = values[key]
            return True
        return False


def _model_contract(model: Any, configured: Any) -> dict[str, Any]:
    contract = configured.contract() if hasattr(configured, "contract") else {}
    output = dict(contract)
    # ``local_path`` is a deployment detail, not part of the semantic model
    # contract (and can disclose a user's filesystem layout in a plan
    # artifact).  The fingerprint already excludes it; omit it here too.
    output.pop("local_path", None)
    output["model"] = str(getattr(model, "model_id", getattr(configured, "model", "UNKNOWN")))
    output["dimension"] = int(getattr(model, "dimension", configured.dimension or 0))
    output["fingerprint"] = str(getattr(model, "fingerprint", configured.fingerprint))
    return _safe(output)


def _model_id(model: Any, configured: Any) -> str:
    return str(getattr(model, "model_id", getattr(configured, "model", "UNKNOWN")))


def _model_fingerprint(model: Any, configured: Any) -> str:
    return str(getattr(model, "fingerprint", getattr(configured, "fingerprint", "")))


def _canonical_metric(value: Any) -> str | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    text = str(value).strip().lower().replace("-", "_")
    return {
        "cos": "cosine",
        "cosine": "cosine",
        "dot": "dot",
        "dotproduct": "dot",
        "inner_product": "dot",
        "ip": "dot",
        "l2": "l2",
        "euclidean": "l2",
        "squared_l2": "l2",
        "squaredl2": "l2",
    }.get(text, text)


def _index_health(index: Any, source_dimension: int | None, expected_metric: Any = None) -> tuple[dict[str, Any], list[PlanWarning]]:
    """Reuse a backend's existing audit method where it exposes one."""
    warnings: list[PlanWarning] = []
    checks: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    try:
        raw_metadata = index.metadata()
        if not isinstance(raw_metadata, Mapping):
            metadata = {}
            checks["metadata"] = {"status": "FAIL", "detail": "backend metadata must be a mapping"}
            warnings.append(_warning("MALFORMED_INDEX_METADATA", "ERROR", "The source backend returned malformed index metadata.", "Upgrade/fix the backend adapter and rerun the planner."))
        else:
            metadata = _safe(raw_metadata)
            checks["metadata"] = {"status": "PASS", "detail": "backend metadata loaded"}
    except Exception as exc:
        checks["metadata"] = {"status": "FAIL", "detail": _redact_text(exc)}
    size: int | None = None
    try:
        raw_size = index.size()
        size = _validate_positive_int(raw_size, "index size", allow_zero=True)
        if size < 0:
            raise ValueError("index size must be non-negative")
        checks["corpus_size"] = {"status": "PASS" if size else "FAIL", "value": size,
                                  "detail": "source index contains vectors" if size else "source index is empty"}
        if not size:
            warnings.append(_warning("EMPTY_SOURCE_INDEX", "ERROR", "The source index contains no vectors.", "Populate the existing source index before planning a migration."))
    except Exception as exc:
        checks["corpus_size"] = {"status": "FAIL", "detail": _redact_text(exc)}
    audit_fn = getattr(index, "audit", None)
    audit: dict[str, Any] | None = None
    if callable(audit_fn):
        try:
            try:
                raw_audit = audit_fn(source_dimension=source_dimension)
            except TypeError:
                raw_audit = audit_fn()
            audit = _safe(raw_audit if isinstance(raw_audit, Mapping) else {})
            audit_ok = bool(audit.get("ok", False))
            checks["backend_audit"] = {"status": "PASS" if audit_ok else "FAIL", "detail": audit}
            if not audit_ok:
                warnings.append(_warning("BACKEND_AUDIT_FAILED", "ERROR", "The source backend audit did not pass.", "Resolve the reported connection, schema, dimension, or document-text checks before planning."))
        except Exception as exc:
            checks["backend_audit"] = {"status": "FAIL", "detail": _redact_text(exc)}
            warnings.append(_warning("BACKEND_AUDIT_FAILED", "ERROR", f"Source backend audit failed: {_redact_text(exc)}", "Run the backend-specific audit-index command and correct the failure."))
    else:
        checks["backend_audit"] = {"status": "WARN", "detail": "backend does not expose an audit method"}
        warnings.append(_warning("BACKEND_AUDIT_UNAVAILABLE", "WARN", "The backend does not expose a detailed audit method.", "Use the backend's native health and exact/reference retrieval checks separately."))
    if source_dimension is not None:
        raw_index_dim = metadata.get("dimension")
        try:
            index_dim = int(raw_index_dim)
        except (TypeError, ValueError):
            index_dim = None
        if index_dim is None:
            checks["dimension"] = {"status": "WARN", "configured": int(source_dimension), "index": "UNKNOWN"}
            warnings.append(_warning("UNKNOWN_INDEX_DIMENSION", "WARN", "The backend did not expose a reliable source dimension.", "Run a small source query or inspect the native index schema before rollout."))
        else:
            match = index_dim == int(source_dimension)
            checks["dimension"] = {"status": "PASS" if match else "FAIL", "configured": int(source_dimension), "index": index_dim}
            if not match:
                warnings.append(_warning("SOURCE_DIMENSION_MISMATCH", "ERROR", f"Source encoder dimension {source_dimension} does not match index dimension {index_dim}.", "Use the encoder contract that created the source index or rebuild only in an explicitly separate migration setup."))
    metric = metadata.get("metric")
    if metric is None:
        checks["metric"] = {"status": "WARN", "configured": "UNKNOWN", "index": "UNKNOWN"}
        warnings.append(_warning("UNKNOWN_INDEX_METRIC", "WARN", "The source metric was not exposed by backend introspection.", "Verify metric semantics with a small exact/reference check."))
    else:
        metric_matches = expected_metric is None or _canonical_metric(metric) == _canonical_metric(expected_metric)
        checks["metric"] = {"status": "PASS" if metric_matches else "FAIL", "configured": expected_metric or "UNKNOWN", "index": metric}
        if not metric_matches:
            warnings.append(_warning(
                "SOURCE_METRIC_MISMATCH",
                "ERROR",
                f"Configured source metric {expected_metric!r} differs from index metric {metric!r}.",
                "Use the metric that was configured when the source index was built, then rerun the planner.",
            ))
    ann_status = "UNKNOWN"
    if audit and isinstance(audit, Mapping):
        raw_ann = audit.get("ann_fidelity") or audit.get("ann_status")
        if str(raw_ann).upper() in {"PASS", "WARNING", "FAIL"}:
            ann_status = str(raw_ann).upper()
    return {
        "status": "FAIL" if any(item.get("status") == "FAIL" for item in checks.values()) else
        "WARN" if any(item.get("status") == "WARN" for item in checks.values()) else "PASS",
        "checks": checks,
        "index": metadata,
        "backend_audit": audit,
        "corpus_documents": size,
        "ann_fidelity": ann_status,
    }, warnings


def _stability_rows(probe: Mapping[str, Any], k_grid: Sequence[int]) -> list[dict[str, Any]]:
    rows = probe.get("per_query", [])
    output: list[dict[str, Any]] = []
    for k in k_grid:
        values: list[float] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            source_scores = row.get("target_scores")
            if not isinstance(source_scores, Mapping):
                continue
            ids = [str(item) for item in source_scores]
            if not ids:
                continue
            try:
                numeric_scores = {item: float(source_scores[item]) for item in ids}
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("T2-v1 returned a non-numeric target score") from exc
            if not all(math.isfinite(value) for value in numeric_scores.values()):
                raise ValueError("T2-v1 returned a non-finite target score")
            final_ids = sorted(ids, key=lambda item: (-numeric_scores[item], ids.index(item)))[:10]
            candidates = ids[: min(int(k), len(ids))]
            ranked = sorted(candidates, key=lambda item: (-numeric_scores[item], ids.index(item)))[:10]
            values.append(len(set(ranked) & set(final_ids)) / max(1, len(final_ids)))
        output.append({
            "k": int(k),
            "mean_top10_stability": float(np.mean(values)) if values else None,
            "minimum_top10_stability": float(np.min(values)) if values else None,
            "queries": len(values),
            "provenance": "t2_v1_no_qrels",
        })
    return output


def _next_k(maximum: int, backend_limit: int | None = None) -> int | None:
    candidate = max(10, int(maximum) * 2)
    if backend_limit is not None and candidate > backend_limit:
        return None
    return candidate


def _confidence(*, t2_status: str, probe_count: int, source_contract: str, ann: str, exact_registry: bool) -> str:
    if t2_status == T2_NOT_RUN or probe_count == 0:
        return "INSUFFICIENT"
    if t2_status != T2_SAFE:
        return "LIMITED" if probe_count < 50 else "MODERATE"
    if probe_count >= 100 and source_contract == "VERIFIED" and (ann == "PASS" or exact_registry):
        return "STRONG"
    if probe_count >= 50:
        return "MODERATE"
    return "LIMITED"


def _rollout(recommendation: str, recommended_k: int | None, confidence: str) -> dict[str, Any]:
    phases = [
        {"order": 0, "name": "Validate", "action": "Run audit-index, verify model contracts, and retain the plan as an artifact."},
        {"order": 1, "name": "Shadow", "action": "Keep the source path authoritative; collect target ranking, cache, and latency telemetry out of band."},
    ]
    if recommendation in {"PROCEED", "PROCEED_WITH_CAUTION"}:
        phases.extend([
            {"order": 2, "name": "Small canary", "action": "Suggested 5% traffic canary only after shadow checks pass; operator-controlled."},
            {"order": 3, "name": "Expanded canary", "action": "Suggested 25% expansion after reviewing error rate, p95 latency, queue depth, and ranking evaluation."},
            {"order": 4, "name": "Target-primary path", "action": "Move traffic in stages while continuing progressive target materialization."},
        ])
    else:
        phases.append({"order": 2, "name": "Hold", "action": "Do not canary; expand probes or resolve blocking/uncertain evidence first."})
    next_action = {
        "PROCEED": f"Run a shadow evaluation with K={recommended_k} before an operator-controlled canary.",
        "PROCEED_WITH_CAUTION": f"Run a longer shadow evaluation with K={recommended_k or 'the tested grid maximum'} before canary traffic.",
        "EXPAND_PROBE": "Expand the probe set or test a larger candidate depth before canarying.",
        "DEFER": "Collect qrels/native-target evidence or resolve the unsafe finite-tail behavior before deployment.",
        "BLOCKED": "Fix the preflight failure, then rerun the planner.",
    }[recommendation]
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "phases": phases,
        "next_action": next_action,
        "stop_conditions": [
            "source retrieval or target encoding errors increase",
            "p95 migration-path latency exceeds the operator's budget",
            "materialization queue grows without draining",
            "cache hit rate or ranking evaluation is worse than the shadow baseline",
        ],
        "operator_controlled": True,
    }


class MigrationPlanner:
    """Compose an advisory plan from an existing EmbedFlow configuration."""

    def __init__(self, config: str | Path | EmbedFlowConfig | Mapping[str, Any], *,
                 device: str | None = None, demo: bool = False,
                 model_root: str | Path | None = None,
                 source_model: Any | None = None, target_model: Any | None = None,
                 source_index: Any | None = None, documents: Any | None = None,
                 registry_records: Iterable[Any] | None = None,
                 t2_runner: Callable[..., dict[str, Any]] | None = None) -> None:
        if isinstance(config, (str, Path)):
            self.config_path = Path(config).expanduser().resolve()
            self.config = load_config(self.config_path)
        elif isinstance(config, EmbedFlowConfig):
            self.config_path = None
            self.config = copy.deepcopy(config)
            self.config.validate()
            self.config.resolve_paths(Path.cwd().resolve())
        elif isinstance(config, Mapping):
            self.config_path = None
            self.config = from_dict(dict(config))
            self.config.resolve_paths(Path.cwd().resolve())
        else:
            raise TypeError("config must be a path, mapping, or EmbedFlowConfig")
        self.device = device
        self.demo = bool(demo)
        self.model_root = Path(model_root).expanduser().resolve() if model_root else None
        self._source_model = source_model
        self._target_model = target_model
        self._source_index = source_index
        self._documents = documents
        if registry_records is None:
            self.registry_records = None
        else:
            self.registry_records = [
                item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_dict(item)
                for item in registry_records
            ]
        self.t2_runner = t2_runner or run_probe
        self._runtime_engine: Any | None = None

    def _runtime(self) -> tuple[tuple[Any, Any, Any, Any], bool, tempfile.TemporaryDirectory[str] | None, Any | None]:
        supplied = (self._source_model, self._target_model, self._source_index, self._documents)
        if all(item is not None for item in supplied):
            return supplied, False, None, None  # type: ignore[return-value]
        if any(item is not None for item in supplied):
            raise ValueError("source_model, target_model, source_index, and documents must be supplied together")
        temporary = tempfile.TemporaryDirectory(prefix="embedflow-planner-")
        root = Path(temporary.name)
        cfg = copy.deepcopy(self.config)
        # Every plan gets an isolated target cache/state/telemetry location;
        # planning therefore cannot alter a serving cache by accident.
        cfg.cache.path = str(root / "cache")
        cfg.state_path = str(root / "state.json")
        cfg.telemetry.latency_log = str(root / "latency.jsonl")
        if self.model_root:
            registry = {
                "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
                "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b",
                "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
                "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
            }
            for model in (cfg.source, cfg.target):
                staged = self.model_root / registry.get(model.model, model.model)
                if staged.exists():
                    model.local_path = str(staged)
        config_path = root / "embedflow.yaml"
        try:
            save_config(cfg, config_path)
            engine = open_engine(config_path, device=self.device, demo=self.demo, start_worker=False,
                                 allow_empty_index=True)
        except Exception:
            # Do not leave a temporary directory behind when model/backend
            # initialization fails before the engine owns its resources.
            temporary.cleanup()
            raise
        self._runtime_engine = engine
        return (engine.source_model, engine.target_model, engine.source_index, engine.documents), True, temporary, engine

    def plan(self, probe_queries: str | Path | Iterable[tuple[str, str]] | None = None, *,
             queries: str | Path | Iterable[tuple[str, str]] | None = None,
             max_probes: int | None = None, seed: int | None = None,
             k_grid: Iterable[int] | str | None = None, max_candidates: int | None = None,
             max_target_encodes: int | None = None, profile: bool = False,
             gpu_hourly_cost: float | None = None, target_docs_per_second: float | None = None,
             queries_per_second: float | None = None, daily_queries: float | None = None,
             cache_hit_rate: float | None = None, latency_budget_ms: float | None = None,
             access_trace: str | Path | None = None, corpus_name: str | None = None,
             corpus_fingerprint: str | None = None,
             progress: Callable[[str], None] | None = None) -> PlanResult:
        """Generate a structured plan; recommendation states never route traffic."""
        if probe_queries is not None and queries is not None:
            raise ValueError("provide probe_queries or queries, not both")
        query_source = probe_queries if probe_queries is not None else queries
        if query_source is None:
            # Reuse the project's existing probe configuration when present;
            # an explicit ``--queries``/API argument still takes precedence.
            query_source = self.config.probe.queries
        planner_cfg: PlannerConfig = copy.deepcopy(self.config.planner)
        max_probes = planner_cfg.max_probes if max_probes is None else max_probes
        seed = planner_cfg.seed if seed is None else seed
        if k_grid is None:
            grid_values: Iterable[int] = planner_cfg.k_grid or DEFAULT_K_GRID
        elif isinstance(k_grid, str):
            try:
                grid_values = [int(item.strip()) for item in k_grid.split(",") if item.strip()]
            except ValueError as exc:
                raise ValueError("k_grid must be comma-separated integers") from exc
        else:
            grid_values = k_grid
        max_candidates = planner_cfg.max_candidates if max_candidates is None else max_candidates
        max_target_encodes = planner_cfg.max_target_encodes if max_target_encodes is None else max_target_encodes
        gpu_hourly_cost = planner_cfg.gpu_hourly_cost if gpu_hourly_cost is None else gpu_hourly_cost
        target_docs_per_second = planner_cfg.target_docs_per_second if target_docs_per_second is None else target_docs_per_second
        queries_per_second = planner_cfg.queries_per_second if queries_per_second is None else queries_per_second
        daily_queries = planner_cfg.daily_queries if daily_queries is None else daily_queries
        cache_hit_rate = planner_cfg.cache_hit_rate if cache_hit_rate is None else cache_hit_rate
        latency_budget_ms = planner_cfg.latency_budget_ms if latency_budget_ms is None else latency_budget_ms
        access_trace = planner_cfg.access_trace if access_trace is None else access_trace
        corpus_name = planner_cfg.corpus_name if corpus_name is None else corpus_name
        corpus_fingerprint = planner_cfg.corpus_fingerprint if corpus_fingerprint is None else corpus_fingerprint
        max_probes = _validate_positive_int(max_probes, "max_probes")
        seed = _validate_integer(seed, "seed")
        if max_candidates is not None:
            max_candidates = _validate_positive_int(max_candidates, "max_candidates")
        if max_target_encodes is not None:
            max_target_encodes = _validate_positive_int(max_target_encodes, "max_target_encodes")
        gpu_hourly_cost = _validate_nonnegative(gpu_hourly_cost, "gpu_hourly_cost")
        target_docs_per_second = _validate_nonnegative(target_docs_per_second, "target_docs_per_second")
        queries_per_second = _validate_nonnegative(queries_per_second, "queries_per_second")
        daily_queries = _validate_nonnegative(daily_queries, "daily_queries")
        cache_hit_rate = _validate_nonnegative(cache_hit_rate, "cache_hit_rate", upper=1.0)
        latency_budget_ms = _validate_nonnegative(latency_budget_ms, "latency_budget_ms")
        if target_docs_per_second is not None and target_docs_per_second <= 0:
            raise ValueError("target_docs_per_second must be greater than zero when supplied")
        if access_trace is not None and not str(access_trace).strip():
            raise ValueError("access_trace must be a non-empty path")
        if progress:
            progress("Preflight...")
        source_model = target_model = source_index = documents = None
        runtime_engine = None
        owned_runtime = False
        temporary: tempfile.TemporaryDirectory[str] | None = None
        planner_cache_temp: tempfile.TemporaryDirectory[str] | None = None
        planner_cache: SQLiteVectorCache | None = None
        warnings: list[PlanWarning] = []
        limitations = [
            "This plan is advisory and does not route traffic or mutate the source index.",
            "T2-v1 is a no-qrels finite-tail diagnostic, not a retrieval-quality guarantee.",
            "ANN fidelity is UNKNOWN unless an exact/reference source comparison is supplied.",
        ]
        try:
            runtime = self._runtime()
            (source_model, target_model, source_index, documents), owned_runtime, temporary, runtime_engine = runtime
            if source_model is None or target_model is None or source_index is None or documents is None:
                raise RuntimeError("planner runtime could not be initialized")
            preflight, preflight_warnings = _index_health(source_index, int(getattr(source_model, "dimension", 0) or 0), self.config.index.metric)
            warnings.extend(preflight_warnings)
            if progress:
                progress("Registry evidence...")
            for label, configured, model in (("source", self.config.source.dimension, source_model),
                                             ("target", self.config.target.dimension, target_model)):
                if configured is not None and int(getattr(model, "dimension", 0) or 0) != int(configured):
                    warnings.append(_warning(
                        f"{label.upper()}_MODEL_DIMENSION_MISMATCH",
                        "ERROR",
                        f"Configured {label} model dimension {configured} does not match the loaded encoder dimension {getattr(model, 'dimension', 'UNKNOWN')}.",
                        f"Set {label}.dimension to the loaded encoder dimension or load the intended model contract.",
                    ))
                    preflight["status"] = "FAIL"
            source_size = preflight.get("corpus_documents")
            if source_size is None:
                try:
                    source_size = int(documents.size())
                except Exception:
                    source_size = None
            source_contract_state = "UNKNOWN"
            stored_fp = (preflight.get("index") or {}).get("model_fingerprint") if isinstance(preflight.get("index"), Mapping) else None
            if stored_fp:
                source_contract_state = "VERIFIED" if str(stored_fp) == _model_fingerprint(source_model, self.config.source) else "MISMATCH"
                if source_contract_state == "MISMATCH":
                    warnings.append(_warning("SOURCE_CONTRACT_MISMATCH", "ERROR", "The source index fingerprint differs from the configured source model contract.", "Use the exact source revision/pooling/normalization contract that created the index."))
            else:
                warnings.append(_warning("UNKNOWN_SOURCE_CONTRACT", "WARN", "The existing source index does not expose a verifiable model contract.", "Record the source model revision, pooling, prompts, normalization, and dtype before rollout."))
            if source_contract_state == "MISMATCH":
                preflight["status"] = "FAIL"
            if int(getattr(source_model, "dimension", 0) or 0) < 1 or int(getattr(target_model, "dimension", 0) or 0) < 1:
                warnings.append(_warning("UNKNOWN_MODEL_DIMENSION", "ERROR", "A model dimension could not be established.", "Configure explicit source/target dimensions or provide loadable model contracts."))
                preflight["status"] = "FAIL"
            if preflight.get("status") == "FAIL":
                recommendation = "BLOCKED"
            registry = match_config(
                self.config,
                corpus_fingerprint=corpus_fingerprint,
                corpus_name=corpus_name,
                corpus_size=source_size,
                records=self.registry_records if self.registry_records is not None else load_evidence(),
            )
            registry_dict = registry.to_dict()
            registry_dict["records_used"] = [
                {"evidence_id": row.evidence_id, "dataset": _safe(row.dataset),
                 "candidate_depths": sorted(row.candidate_gap), "provenance": _safe(row.raw.get("provenance", {}))}
                for row in registry.records
            ]
            if registry.level == MATCH_NONE:
                warnings.append(_warning("NO_REGISTRY_EVIDENCE", "WARN", "No relevant retained migration evidence matches this model transition.", "Rely on representative probes and, where possible, qrels/native-target evaluation."))
            elif registry.level == MATCH_PRIOR:
                warnings.append(_warning("PRIOR_EVIDENCE_ONLY", "INFO", "Model contracts match prior evidence, but the current corpus is different or unidentified.", "Use registry depths as starting points only; current probes remain authoritative."))
            elif registry.level == MATCH_RELATED:
                warnings.append(_warning("RELATED_EVIDENCE_ONLY", "WARN", "Only related model-family evidence was found.", "Do not treat related evidence as an exact compatibility result."))
            if source_size is None:
                warnings.append(_warning("DOCUMENT_COUNT_UNKNOWN", "WARN", "Source document count is unavailable.", "Supply corpus metadata for storage and backfill estimates."))
            rows = load_probe_queries(query_source) if query_source is not None else []
            supplied_probe_count = len(rows)
            sampled = _sample_queries(rows, max_probes, seed) if rows else []
            if rows and len(sampled) < len(rows):
                warnings.append(_warning("PROBE_SAMPLED", "INFO", f"Only {len(sampled)} of {len(rows)} supplied probes were sampled.", "Increase --max-probes if the sample is not representative."))
            selected, duplicate_probe_count = _deduplicate_probe_content(sampled)
            if duplicate_probe_count:
                warnings.append(_warning(
                    "DUPLICATE_PROBE_CONTENT",
                    "WARN",
                    f"Removed {duplicate_probe_count} repeated probe texts from the analysis sample; evidence counts use unique text.",
                    "Provide semantically diverse probes so confidence is not driven by repeated wording.",
                ))
            if selected and len(selected) < 25:
                warnings.append(_warning("LIMITED_PROBE_COUNT", "WARN", f"Only {len(selected)} probe queries were analyzed.", "Use a larger representative probe set before a production canary."))
            if not selected:
                warnings.append(_warning("NO_PROBES", "WARN", "No probe queries were supplied; T2-v1 was not run.", "Provide --queries or planner/access probe configuration before making a compatibility decision."))
            actual_size = None if source_size is None else max(0, int(source_size))
            all_requested_grid = _k_grid(grid_values)
            beyond_t2 = [k for k in all_requested_grid if k > T2_MAX_K]
            if beyond_t2:
                warnings.append(_warning(
                    "K_BEYOND_T2_FROZEN",
                    "WARN",
                    f"Requested K values {beyond_t2} exceed the frozen T2-v1 feature depth of {T2_MAX_K}.",
                    "Treat those depths as separate expansion experiments; they are not certified by this planner.",
                ))
            requested_grid = [k for k in all_requested_grid if k <= T2_MAX_K]
            below_t2 = [k for k in requested_grid if k < 10]
            if below_t2:
                warnings.append(_warning(
                    "K_BELOW_T2_MINIMUM",
                    "WARN",
                    f"Requested K values {below_t2} are below T2-v1's minimum feature depth of 10.",
                    "Keep them as serving experiments only; include K>=10 before asking the planner to certify a starting depth.",
                ))
            if max_candidates is not None:
                if isinstance(max_candidates, bool) or int(max_candidates) != max_candidates or int(max_candidates) < 1:
                    raise ValueError("max_candidates must be a positive integer")
                requested_grid = [k for k in requested_grid if k <= int(max_candidates)]
            if actual_size is not None:
                requested_grid = [k for k in requested_grid if k <= actual_size]
            if not requested_grid:
                requested_grid = []
                warnings.append(_warning("K_GRID_UNAVAILABLE", "ERROR", "No candidate depth in the requested grid is valid for this source index.", "Choose positive K values no larger than the source index and backend API limit."))
            if requested_grid and max(requested_grid) > T2_MAX_K:
                warnings.append(_warning("T2_MAX_K", "WARN", "T2-v1 finite-tail features are frozen through K=500; larger depths were not certified by this planner.", "Treat larger K as an expansion experiment, not as a T2-safe result."))
            # A configured grid is not evidence that it was actually tested.
            # Only expose depths as ``tested_k`` when a valid probe run will
            # execute; this avoids a no-probe or failed-preflight plan looking
            # like it certified a candidate depth.
            t2_grid = (
                requested_grid
                if preflight.get("status") != "FAIL" and selected and requested_grid and max(requested_grid) >= 10
                else []
            )
            candidate_depth: dict[str, Any] = {
                "requested_k": all_requested_grid,
                "tested_k": t2_grid,
                "maximum_tested_k": max(t2_grid) if t2_grid else None,
                "recommended_k": None,
                "next_expansion_k": None,
                "t2_status": T2_NOT_RUN,
                "t2_diagnostics": None,
                "candidate_gap": None,
                "candidate_gap_provenance": "UNKNOWN — no qrels/native-target evaluation was supplied",
                "k_results": [],
                "recommendation_reason": "Probe analysis was not run.",
            }
            unique_candidates: list[str] = []
            candidate_seen: set[str] = set()
            target_vectors: Mapping[str, np.ndarray] | None = None
            # T2-v1 requires a finite pool of at least ten candidates.  A
            # tiny source index may still be inspectable for preflight and
            # economics, but it must not cause an opaque ``run_probe`` error.
            if (preflight.get("status") != "FAIL" and selected and requested_grid
                    and max(requested_grid) >= 10):
                probe_kmax = min(max(requested_grid), T2_MAX_K)
                planned_occurrences = len(selected) * probe_kmax
                if planned_occurrences > 100_000:
                    warnings.append(_warning(
                        "LARGE_PLANNER_WORK",
                        "WARN",
                        f"The bounded probe grid may inspect up to {planned_occurrences:,} candidate occurrences before deduplication.",
                        "Reduce --max-probes/--max-candidates or use a representative subset before running an expensive plan.",
                    ))
                if progress:
                    progress("Retrieving probe candidates...")
                for _, text in selected:
                    source_vector = _encode_query(source_model, text)
                    if source_vector.ndim != 1 or source_vector.shape[0] != int(source_model.dimension) or not np.isfinite(source_vector).all():
                        raise ValueError("source query encoder returned an invalid vector during planning")
                    hits = source_index.search(source_vector, probe_kmax)
                    if hits is None or not isinstance(hits, Sequence):
                        raise ValueError("source index returned a malformed candidate list during planning")
                    if len(hits) > probe_kmax:
                        raise ValueError(f"source index returned {len(hits)} candidates for K={probe_kmax}")
                    query_candidate_ids: set[str] = set()
                    for hit in hits:
                        if not hasattr(hit, "document_id"):
                            raise ValueError("source index returned a candidate without a document ID")
                        document_id = str(hit.document_id)
                        if not document_id:
                            raise ValueError("source index returned an empty document ID")
                        if document_id in query_candidate_ids:
                            raise ValueError(f"source index returned duplicate candidate ID {document_id!r}")
                        query_candidate_ids.add(document_id)
                        if not hasattr(hit, "score"):
                            raise ValueError(f"source index returned candidate {document_id!r} without a score")
                        try:
                            score = float(hit.score)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(f"source index returned a non-numeric score for candidate {document_id!r}") from exc
                        if not math.isfinite(score):
                            raise ValueError(f"source index returned a non-finite score for candidate {document_id!r}")
                        if not hasattr(hit, "source_rank"):
                            raise ValueError(f"source index returned candidate {document_id!r} without a source rank")
                        try:
                            source_rank = int(hit.source_rank)
                            if float(hit.source_rank) != source_rank or source_rank < 0:
                                raise ValueError
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(f"source index returned an invalid source rank for candidate {document_id!r}") from exc
                        if document_id not in candidate_seen:
                            candidate_seen.add(document_id)
                            unique_candidates.append(document_id)
                if max_target_encodes is not None:
                    if isinstance(max_target_encodes, bool) or int(max_target_encodes) != max_target_encodes or int(max_target_encodes) < 1:
                        raise ValueError("max_target_encodes must be a positive integer")
                    if len(unique_candidates) > int(max_target_encodes):
                        raise ValueError(f"planner needs {len(unique_candidates)} unique target candidates, exceeding max_target_encodes={int(max_target_encodes)}")
                if progress:
                    progress("Encoding target candidates...")
                batch_size = int(planner_cfg.background_batch_size or self.config.migration.background_batch_size)
                planner_cache_temp = tempfile.TemporaryDirectory(prefix="embedflow-planner-vectors-")
                planner_cache = SQLiteVectorCache(
                    Path(planner_cache_temp.name) / "cache",
                    _model_fingerprint(target_model, self.config.target),
                    int(target_model.dimension),
                )
                target_vectors = _CachedVectorMapping(planner_cache)
                for start in range(0, len(unique_candidates), max(1, batch_size)):
                    chunk = unique_candidates[start:start + max(1, batch_size)]
                    resolved = documents.get(chunk)
                    missing = [item for item in chunk if item not in resolved]
                    if missing:
                        raise ValueError(f"document text is unavailable for planner candidates (e.g. {missing[:3]})")
                    texts = [resolved[item] for item in chunk]
                    invalid_text = [item for item, text_value in zip(chunk, texts) if not isinstance(text_value, str) or not text_value.strip()]
                    if invalid_text:
                        raise ValueError(f"planner candidate text must be non-empty strings (e.g. {invalid_text[:3]})")
                    values = _encode_documents(target_model, texts, batch_size)
                    if values.ndim != 2 or values.shape != (len(chunk), int(target_model.dimension)) or not np.isfinite(values).all():
                        raise ValueError("target document encoder returned invalid planner vectors")
                    planner_cache.put(chunk, values)
                if progress:
                    progress("Running frozen T2-v1...")
                probe_result = self.t2_runner(
                    source_model,
                    target_model,
                    source_index,
                    documents,
                    selected,
                    kmax=probe_kmax,
                    seed=int(seed),
                    limit=None,
                    # Do not use ``mapping or {}`` here: Mapping truthiness
                    # calls ``__len__`` (an SQLite stats query) and an empty
                    # planner cache would silently replace the contract with
                    # a plain dict.  Preserve the cache-backed mapping.
                    target_document_vectors=target_vectors if target_vectors is not None else {},
                )
                diagnostic = str(probe_result.get("diagnostic", T2_UNCERTAIN)).upper()
                if diagnostic not in {T2_SAFE, T2_EXPAND, T2_UNCERTAIN}:
                    raise ValueError(f"T2-v1 returned unsupported diagnostic {diagnostic!r}")
                candidate_depth["t2_status"] = diagnostic
                candidate_depth["t2_diagnostics"] = {
                    "features": _safe(probe_result.get("features", {})),
                    "implementation": probe_result.get("implementation", "src.t2_v1.decide"),
                    "warning": probe_result.get("warning", "T2-v1 is an empirical finite-tail diagnostic, not a compatibility guarantee."),
                    "queries": len(selected),
                    "seed": int(seed),
                }
                candidate_depth["k_results"] = _stability_rows(probe_result, requested_grid)
                t2_recommended = probe_result.get("recommended_k")
                if t2_recommended is not None:
                    t2_recommended = _validate_positive_int(t2_recommended, "T2 recommended_k")
                    if t2_recommended > probe_kmax:
                        raise ValueError("T2 recommended_k cannot exceed the tested candidate depth")
                # Candidate pools are expected to be nested.  A stability
                # curve that drops at a deeper tested K is therefore either
                # contradictory upstream evidence or a backend/probe bug.
                # Do not certify a shallow isolated point in that case: only
                # a K whose entire tested suffix remains stable is defensible.
                k_results = candidate_depth["k_results"]
                stable: list[int] = []
                nonmonotonic = False
                for position, row in enumerate(k_results):
                    value = row.get("mean_top10_stability")
                    if value is None:
                        continue
                    current_stable = float(value) >= 0.90
                    suffix_values = [later.get("mean_top10_stability") for later in k_results[position:]]
                    suffix_stable = all(item is not None and float(item) >= 0.90 for item in suffix_values)
                    if current_stable and suffix_stable:
                        stable.append(int(row["k"]))
                    if current_stable and any(item is not None and float(item) < 0.90 for item in suffix_values[1:]):
                        nonmonotonic = True
                if nonmonotonic:
                    warnings.append(_warning(
                        "NONMONOTONIC_K_STABILITY",
                        "WARN",
                        "The finite-pool stability curve contains a stable shallow K followed by an unstable deeper K; the shallow point was not certified.",
                        "Inspect backend/probe behavior and expand or correct the candidate-depth experiment before deployment.",
                    ))
                if diagnostic == T2_SAFE:
                    eligible = [k for k in stable if k >= 10 and (t2_recommended is None or k >= int(t2_recommended))]
                    if eligible:
                        candidate_depth["recommended_k"] = min(eligible)
                        candidate_depth["recommendation_reason"] = (
                            f"K={min(eligible)} was the first tested depth with at least 0.90 mean finite-pool top-10 stability "
                            f"after the frozen T2-v1 diagnostic returned SAFE. This is an empirical recommendation, not a guarantee."
                        )
                    else:
                        candidate_depth["next_expansion_k"] = _next_k(max(requested_grid), T2_MAX_K)
                        candidate_depth["recommendation_reason"] = "T2-v1 returned SAFE, but the requested grid did not contain a defensible stable starting depth."
                        warnings.append(_warning("K_EVIDENCE_INCOMPLETE", "WARN", "No tested K met the planner's finite-pool stability guard.", "Expand the K grid and probe set before canarying."))
                elif diagnostic == T2_EXPAND:
                    candidate_depth["next_expansion_k"] = _next_k(max(requested_grid), T2_MAX_K)
                    candidate_depth["recommendation_reason"] = "T2-v1 requested a larger candidate pool; the tested maximum is not certified SAFE."
                    warnings.append(_warning("T2_EXPAND", "WARN", "Finite-tail behavior remained unstable at the tested depth.", "Test a larger K if backend limits and cost allow, then rerun the planner."))
                else:
                    candidate_depth["recommendation_reason"] = "T2-v1 classified the finite-tail behavior UNSAFE_OR_UNCERTAIN; no safe K is reported."
                    warnings.append(_warning("T2_UNSAFE_OR_UNCERTAIN", "ERROR", "Probe behavior does not support a safe progressive recommendation.", "Defer canary traffic and collect stronger evaluation evidence or reconsider the migration."))
            elif preflight.get("status") == "FAIL":
                candidate_depth["recommendation_reason"] = "Preflight failed; T2-v1 was intentionally not run."
            elif selected and requested_grid:
                candidate_depth["recommendation_reason"] = "No tested depth reaches T2-v1's minimum finite pool of 10 candidates; T2-v1 was intentionally not run."
                warnings.append(_warning(
                    "K_BELOW_T2_MINIMUM",
                    "WARN",
                    "The source index or requested grid is smaller than T2-v1's minimum depth of 10.",
                    "Use a larger source index/grid for a T2 diagnostic; treat this plan as preflight-only.",
                ))
            elif selected and not requested_grid:
                candidate_depth["recommendation_reason"] = "The requested candidate grid is unavailable for this source/backend; T2-v1 was intentionally not run."
            else:
                candidate_depth["recommendation_reason"] = "No probes were supplied; T2-v1 was intentionally not run."
            if preflight.get("status") == "FAIL":
                recommendation = "BLOCKED"
            elif candidate_depth["t2_status"] == T2_UNCERTAIN:
                recommendation = "DEFER"
            elif candidate_depth["t2_status"] == T2_EXPAND:
                recommendation = "EXPAND_PROBE"
            elif candidate_depth["t2_status"] == T2_SAFE and candidate_depth.get("recommended_k") is not None:
                # A healthy finite-tail probe is not enough to promote a plan
                # to the strongest state when backend preflight is only a
                # warning (for example, ANN fidelity is unknown).  Keep the
                # operator-facing recommendation conservative while preserving
                # the empirical K result.
                recommendation = "PROCEED" if (
                    len(selected) >= 100
                    and source_contract_state == "VERIFIED"
                    and preflight.get("status") == "PASS"
                ) else "PROCEED_WITH_CAUTION"
            else:
                recommendation = "EXPAND_PROBE"
            if not selected and preflight.get("status") != "FAIL":
                recommendation = "EXPAND_PROBE"
            if recommendation not in RECOMMENDATIONS:
                recommendation = "BLOCKED"
            latency_rows, latency_warnings = _read_latency(self.config.telemetry.latency_log)
            for message in latency_warnings:
                warnings.append(_warning("LATENCY_TELEMETRY", "WARN", message, "Inspect or regenerate the telemetry file."))
            performance: dict[str, Any] = {
                "measured": {stage: _quantity_values(latency_rows, stage) for stage in (
                    "source_query_encode_ms", "source_ann_ms", "target_query_encode_ms", "cache_lookup_ms",
                    "synchronous_target_encode_ms", "target_score_ms", "topk_ms", "total_ms")},
                "modeled": {},
                "user_supplied": {
                    "latency_budget_ms": quantity(latency_budget_ms, "ms", "user_supplied") if latency_budget_ms is not None else quantity(None, "ms", "unknown"),
                    "queries_per_second": quantity(queries_per_second, "queries/second", "user_supplied") if queries_per_second is not None else quantity(None, "queries/second", "unknown"),
                    "daily_queries": quantity(daily_queries, "queries/day", "user_supplied") if daily_queries is not None else quantity(None, "queries/day", "unknown"),
                    "cache_hit_rate": quantity(cache_hit_rate, "fraction", "user_supplied") if cache_hit_rate is not None else quantity(None, "fraction", "unknown"),
                },
                "assumptions": ["Measured values come only from the configured telemetry file; missing stages remain UNKNOWN."],
            }
            measured_total = (performance["measured"].get("total_ms") or {}).get("p50")
            if measured_total:
                performance["modeled"]["warm_path_p50"] = quantity(measured_total["value"], "ms", "modeled", assumptions=["warm-path estimate reuses observed total p50; target-cache state may differ"])
            elif latency_budget_ms is not None:
                warnings.append(_warning("LATENCY_UNKNOWN", "WARN", "No measured latency telemetry is available.", "Run a small local profile or collect serving telemetry before comparing with a latency budget."))
            profile_result: dict[str, Any] | None = None
            if profile and selected and preflight.get("status") != "FAIL":
                profile_result = self._profile(source_model, target_model, source_index, documents, selected[: min(20, len(selected))],
                                                min(max(requested_grid or [10]), T2_MAX_K))
                performance["profile"] = profile_result
                throughput = profile_result.get("target_docs_per_second")
                if throughput is not None:
                    target_docs_per_second = float(throughput)
            economics = estimate_economics(
                source_size,
                int(getattr(target_model, "dimension", 0) or 0) or None,
                dtype=getattr(self.config.target, "dtype", "float32"),
                docs_per_second=target_docs_per_second if target_docs_per_second is not None else self.config.economics.target_docs_per_second,
                gpu_hourly_cost=gpu_hourly_cost if gpu_hourly_cost is not None else self.config.economics.gpu_price_per_hour,
                cached_documents=0,
            )
            if economics["full_backfill"]["wall_time"]["value"] is None:
                warnings.append(_warning("TARGET_THROUGHPUT_UNKNOWN", "WARN", "Target encoding throughput is unknown; backfill time/cost remains UNKNOWN.", "Supply a measured --target-docs-per-second or run --profile."))
            if economics["full_backfill"]["cost"]["value"] is None:
                warnings.append(_warning("COST_MODEL_INCOMPLETE", "INFO", "GPU hourly price or throughput was not supplied; cost remains UNKNOWN.", "Provide both throughput and --gpu-hourly-cost for a modeled cost."))
            cache_plan = self._cache_plan(
                planner_cfg,
                candidate_depth,
                unique_candidates,
                selected,
                access_trace,
                source_size,
                default_sync_misses=self.config.migration.max_sync_misses,
                default_background_batch_size=self.config.migration.background_batch_size,
            )
            cache_plan["traffic_cache_hit_rate"] = quantity(
                cache_hit_rate,
                "fraction",
                "user_supplied" if cache_hit_rate is not None else "unknown",
                assumptions=["operator-provided traffic/cache input"] if cache_hit_rate is not None else (),
            )
            trace_counts = _load_access_trace(access_trace) if access_trace else {}
            if trace_counts:
                cache_plan["access_trace"] = {"documents": len(trace_counts), "events": sum(trace_counts.values()), "provenance": "user_supplied"}
                cache_plan["prewarm_coverage"] = self._trace_coverage(trace_counts, source_size)
            else:
                cache_plan["access_trace"] = {"documents": 0, "events": 0, "provenance": "unknown"}
                cache_plan["prewarm_coverage"] = None
            confidence = _confidence(t2_status=candidate_depth["t2_status"], probe_count=len(selected),
                                     source_contract="VERIFIED" if source_contract_state == "VERIFIED" else "UNKNOWN",
                                     ann=str(preflight.get("ann_fidelity", "UNKNOWN")),
                                     exact_registry=registry.level == MATCH_EXACT)
            if confidence == "LIMITED" and selected:
                warnings.append(_warning("LIMITED_EVIDENCE", "WARN", "The recommendation is based on limited probe/evidence coverage.", "Use representative production-like probes and shadow telemetry before canarying."))
            rollout = _rollout(recommendation, candidate_depth.get("recommended_k"), confidence)
            if progress:
                progress("Building plan...")
            plan = PlanResult(
                schema_version=1,
                recommendation=recommendation,
                confidence=confidence,
                source={
                    "backend": str(self.config.index.backend).lower(),
                    "model": _model_id(source_model, self.config.source),
                    "model_contract": _model_contract(source_model, self.config.source),
                    "dimension": int(source_model.dimension),
                    "corpus_documents": source_size,
                    "metric": (preflight.get("index") or {}).get("metric", self.config.index.metric),
                    "index_health": preflight.get("status"),
                    "index_metadata": preflight.get("index", {}),
                    "ann_fidelity": preflight.get("ann_fidelity", "UNKNOWN"),
                    "source_contract_status": source_contract_state,
                },
                target={
                    "model": _model_id(target_model, self.config.target),
                    "model_contract": _model_contract(target_model, self.config.target),
                    "dimension": int(target_model.dimension),
                    "device": self.device or getattr(self.config.target, "device", None) or "cpu",
                },
                preflight=preflight,
                evidence={
                    "registry_match_class": registry.level,
                    # Keep the concise name used by the public JSON example
                    # alongside the explicit field used by the matcher.
                    "registry_match": registry.level,
                    "registry_rows_used": registry_dict.get("records_used", []),
                    "exact_corpus_match": bool(registry.exact_corpus),
                    "exact_contract_match": bool(registry.exact_source_contract and registry.exact_target_contract),
                    "related_evidence": registry.level == MATCH_RELATED,
                    "probe_queries_supplied": supplied_probe_count,
                    "probe_queries_sampled": len(sampled),
                    "probe_queries_used": len(selected),
                    "duplicate_probe_texts_removed": duplicate_probe_count,
                    "sampling_seed": int(seed),
                    "registry": registry_dict,
                },
                candidate_depth=candidate_depth,
                cache=cache_plan,
                performance=performance,
                economics=economics,
                rollout=rollout,
                warnings=warnings,
                limitations=limitations,
            )
            return plan
        except Exception:
            # A structural/runtime failure is a BLOCKED plan rather than an
            # optimistic partial result.  The CLI still reports ordinary
            # malformed config errors as non-zero; callers using the Python API
            # can catch the exception and inspect its redacted message.
            raise
        finally:
            if runtime_engine is not None:
                try:
                    runtime_engine.close()
                except Exception:
                    pass
            elif owned_runtime:
                for resource in (source_index, source_model, target_model, documents):
                    close = getattr(resource, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
            if temporary is not None:
                temporary.cleanup()
            if planner_cache is not None:
                planner_cache.close()
            if planner_cache_temp is not None:
                planner_cache_temp.cleanup()
            self._runtime_engine = None

    @staticmethod
    def _cache_plan(planner_cfg: PlannerConfig, candidate_depth: Mapping[str, Any], unique_candidates: Sequence[str],
                    selected: Sequence[tuple[str, str]], access_trace: str | Path | None, corpus_size: int | None,
                    *, default_sync_misses: int | None = None,
                    default_background_batch_size: int | None = None) -> dict[str, Any]:
        # Serving defaults remain authoritative unless planner-specific values
        # are supplied.  The final fallback is only for lightweight callers
        # that construct a PlannerConfig outside a full EmbedFlowConfig.
        sync = planner_cfg.max_sync_misses if planner_cfg.max_sync_misses is not None else default_sync_misses
        batch = planner_cfg.background_batch_size if planner_cfg.background_batch_size is not None else default_background_batch_size
        if sync is None:
            sync = 4
        if batch is None:
            batch = 32
        occurrences = len(selected) * int(candidate_depth.get("maximum_tested_k") or 0)
        overlap = (len(unique_candidates) / occurrences) if occurrences else None
        dedup_fraction = (1.0 - overlap) if overlap is not None else None
        policy = "trace-driven explicit/popularity" if access_trace else "traffic-driven progressive warming"
        return {
            "recommended_max_sync_misses": int(sync),
            "background_batch_size": int(batch),
            "suggested_prewarm_policy": policy,
            "planner_cache": "ephemeral; normal serving cache is not modified",
            "candidate_occurrences": occurrences,
            "unique_candidate_documents": len(unique_candidates),
            "candidate_deduplication_fraction": dedup_fraction,
            "unique_candidate_fraction": overlap,
            "modeled_hit_rate_scenarios": [
                {"hit_rate": rate, "provenance": "modeled", "latency": "UNKNOWN"}
                for rate in (0.50, 0.80, 0.90, 0.95, 0.99)
            ],
            "corpus_documents": corpus_size,
        }

    @staticmethod
    def _trace_coverage(counts: Mapping[str, int], corpus_size: int | None) -> dict[str, Any]:
        ordered = sorted(counts.items(), key=lambda pair: (-int(pair[1]), pair[0]))
        total = sum(int(value) for value in counts.values())
        return {
            "top_documents": [{"document_id": key, "count": int(value)} for key, value in ordered[:20]],
            "top_1_percent_event_coverage": (sum(value for _, value in ordered[: max(1, math.ceil(len(ordered) * .01))]) / total) if total else None,
            "trace_documents": len(counts),
            "corpus_documents": corpus_size,
            "provenance": "modeled_from_user_access_trace",
        }

    @staticmethod
    def _profile(source_model: Any, target_model: Any, source_index: Any, documents: Any,
                 queries: Sequence[tuple[str, str]], k: int) -> dict[str, Any]:
        source_times: list[float] = []
        target_query_times: list[float] = []
        ann_times: list[float] = []
        target_doc_times: list[float] = []
        encoded_docs = 0

        # Warm model/tokenizer kernels once outside the measurements.  The
        # warmup follows the same source-query, candidate lookup, target-query
        # and target-document path as the measured samples, but its elapsed
        # time and document count are intentionally excluded from results.
        warmup_excluded = False
        if queries:
            _, warmup_text = queries[0]
            warmup_source = _encode_query(source_model, warmup_text)
            warmup_hits = source_index.search(warmup_source, k)
            _encode_query(target_model, warmup_text)
            warmup_ids = [str(hit.document_id) for hit in warmup_hits]
            if warmup_ids:
                warmup_resolved = documents.get(warmup_ids)
                warmup_texts = [warmup_resolved[item] for item in warmup_ids]
                if any(not isinstance(value, str) or not value.strip() for value in warmup_texts):
                    raise ValueError("planner profile candidate text must be non-empty strings")
                _encode_documents(target_model, warmup_texts, min(32, len(warmup_ids)))
            warmup_excluded = True
        for _, text in queries:
            start = time.perf_counter_ns(); source_vector = _encode_query(source_model, text); source_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns(); hits = source_index.search(source_vector, k); ann_times.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns(); _encode_query(target_model, text); target_query_times.append((time.perf_counter_ns() - start) / 1e6)
            ids = [str(hit.document_id) for hit in hits]
            if ids:
                resolved = documents.get(ids)
                texts = [resolved[item] for item in ids]
                if any(not isinstance(value, str) or not value.strip() for value in texts):
                    raise ValueError("planner profile candidate text must be non-empty strings")
                start = time.perf_counter_ns(); _encode_documents(target_model, texts, min(32, len(ids))); target_doc_times.append((time.perf_counter_ns() - start) / 1e6); encoded_docs += len(ids)
        def measured(values: Sequence[float], unit: str = "ms") -> dict[str, Any]:
            summary = summarize(values)
            return {"p50": quantity(summary["p50_ms"], unit, "measured", assumptions=["small optional local profile", "one unmeasured warmup pass excluded"]),
                    "p95": quantity(summary["p95_ms"], unit, "measured", assumptions=["small optional local profile"]), "count": len(values)}
        docs_per_second = None
        if target_doc_times and sum(target_doc_times) > 0:
            docs_per_second = encoded_docs / (sum(target_doc_times) / 1000.0)
        return {
            "sample_count": len(queries),
            "k": int(k),
            "warmup_excluded": warmup_excluded,
            "device": getattr(target_model, "device", "unknown"),
            "source_query_encode": measured(source_times),
            "source_candidate_search": measured(ann_times),
            "target_query_encode": measured(target_query_times),
            "target_document_encode": measured(target_doc_times) if target_doc_times else None,
            "target_docs_per_second": docs_per_second,
            "target_docs_per_second_provenance": "measured" if docs_per_second is not None else "unknown",
        }


def plan_migration(config: str | Path | EmbedFlowConfig | Mapping[str, Any],
                   probe_queries: str | Path | Iterable[tuple[str, str]] | None = None,
                   **kwargs: Any) -> PlanResult:
    """Convenience function for applications and notebooks.

    ``probe_queries`` is accepted positionally for a small, ergonomic public
    API while the CLI and keyword callers can continue using ``queries=``.
    """
    runtime_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in {
        "device", "demo", "model_root", "source_model", "target_model", "source_index", "documents", "registry_records", "t2_runner"
    }}
    planner = MigrationPlanner(config, **runtime_kwargs)
    return planner.plan(probe_queries=probe_queries, **kwargs)


__all__ = ["MigrationPlanner", "load_probe_queries", "plan_migration"]
