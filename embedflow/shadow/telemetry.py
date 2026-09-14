"""Bounded, privacy-conscious Shadow Mode telemetry persistence."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import SAFE_FAILURE_CATEGORIES, ShadowObservation

SCHEMA_VERSION = 1
_STATUS_COUNTERS = {
    "completed": "shadow_completed_total",
    "partial": "shadow_partial_total",
    "failed": "shadow_failed_total",
    "timeout": "shadow_timeout_total",
    "timed_out": "shadow_timeout_total",
}
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "PASSWORD", "SECRET", "DSN")


def migration_fingerprint(*, source_fingerprint: str, target_fingerprint: str,
                          backend: str, index_identity: str, candidate_k: int) -> str:
    """Return a stable, credential-free configuration fingerprint."""
    payload = "|".join((str(source_fingerprint), str(target_fingerprint), str(backend),
                        str(index_identity), str(int(candidate_k))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_identity_from_config(index: Any) -> str:
    """Build a stable, credential-free input for migration fingerprints.

    Endpoint/collection fields are deliberately hashed before they appear in
    telemetry.  Including them here prevents two deployments that happen to
    use the same collection name (but different hosts, tenants, or databases)
    from being aggregated into one shadow report.
    """
    keys = (
        "backend", "path", "url", "uri", "host", "index_name", "namespace",
        "database", "collection", "table", "schema", "http_host", "http_port",
        "grpc_host", "grpc_port", "tenant", "vector_name", "vector_field",
        "id_field", "text_field", "text_property", "partition_names",
    )
    values: list[str] = []
    for key in keys:
        value = getattr(index, key, "")
        if isinstance(value, (list, tuple, set)):
            value = tuple(str(item) for item in value)
        values.append(f"{key}={value!s}")
    return "|".join(values)


def _sanitize(value: Any) -> str:
    text = str(value)
    for name, secret in os.environ.items():
        if secret and len(secret) >= 4 and any(marker in name.upper() for marker in _SECRET_ENV_MARKERS):
            text = text.replace(secret, "<redacted>")
            if "://" in secret:
                match = re.search(r"://[^:/\s]+:([^@/\s]+)@", secret)
                if match:
                    text = text.replace(match.group(1), "<redacted>")
    text = re.sub(r"(?i)(api[_-]?key|token|password|secret)=([^\s,;]+)", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(://[^:/\s]+:)[^@/\s]+(@)", r"\1<redacted>\2", text)
    return text[:2000]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: Any) -> int:
    """Normalize a persisted counter without letting corrupt rows break reports."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(number) or number < 0:
        return 0
    return int(round(number))


class ShadowTelemetry:
    """Persistent aggregate/event telemetry with failure isolation.

    SQLite writes are deliberately small and bounded.  If the file is
    unavailable or corrupt, the class degrades to in-memory counters and the
    serving path continues; callers can inspect ``available``/``error``.
    """

    def __init__(self, path: str | Path | None, *, config_fingerprint: str = "",
                 enabled: bool = True, max_records: int = 10_000,
                 retain_query_records: bool = False, retain_query_text: bool = False,
                 min_target_coverage_for_ranking: float = 1.0,
                 source_fingerprint: str | None = None, target_fingerprint: str | None = None,
                 backend: str | None = None, index_identity: str | None = None,
                 retention_days: int | None = None):
        self.path = Path(path) if path else None
        self.config_fingerprint = str(config_fingerprint)
        self.enabled = bool(enabled)
        if isinstance(max_records, bool) or int(max_records) != max_records or int(max_records) < 1:
            raise ValueError("shadow telemetry max_records must be a positive integer")
        self.max_records = int(max_records)
        self.retain_query_records = bool(retain_query_records)
        # Raw query text is never accepted by the telemetry API.  Retaining
        # this flag only makes the privacy choice visible to status/config.
        self.retain_query_text = bool(retain_query_text)
        coverage = float(min_target_coverage_for_ranking)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("shadow telemetry target coverage must be between 0 and 1")
        self.min_target_coverage_for_ranking = coverage
        if retention_days is not None:
            if isinstance(retention_days, bool) or int(retention_days) != retention_days or int(retention_days) < 1:
                raise ValueError("shadow telemetry retention_days must be a positive integer")
            self.retention_days = int(retention_days)
        else:
            self.retention_days = None
        # These are semantic/operational fingerprints only; no endpoint,
        # credential, query, document, or vector payload is retained.
        self.source_fingerprint = str(source_fingerprint) if source_fingerprint else None
        self.target_fingerprint = str(target_fingerprint) if target_fingerprint else None
        self.backend = str(backend) if backend else None
        # Endpoint strings can contain DSNs, usernames, or opaque provider
        # tokens.  Reports need a stable identity for correlation, not the
        # endpoint itself, so retain only a one-way digest.
        self.index_identity = (
            hashlib.sha256(str(index_identity).encode("utf-8")).hexdigest()
            if index_identity else None
        )
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self.available = False
        self.error: str | None = None
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._events: list[dict[str, Any]] = []
        self._counter_events: list[dict[str, Any]] = []
        self._counter_event_writes = 0
        self._t2_windows: list[dict[str, Any]] = []
        if self.enabled and self.path is not None:
            self._open()

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=0.5)
            self._db.execute("PRAGMA busy_timeout=500")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("""CREATE TABLE IF NOT EXISTS shadow_meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS shadow_counters (
                bucket INTEGER NOT NULL, config_fingerprint TEXT NOT NULL,
                name TEXT NOT NULL, value REAL NOT NULL,
                PRIMARY KEY(bucket, config_fingerprint, name))""")
            # Bucketed counters are cheap for long-running deployments, while
            # this bounded event ledger keeps explicit report windows exact
            # (rather than over-counting a whole minute at a boundary).
            self._db.execute("""CREATE TABLE IF NOT EXISTS shadow_counter_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at REAL NOT NULL,
                config_fingerprint TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL)""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS shadow_observations (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at REAL NOT NULL,
                config_fingerprint TEXT NOT NULL,
                request_id TEXT,
                status TEXT NOT NULL,
                source_latency_ms REAL,
                shadow_latency_ms REAL,
                source_candidate_latency_ms REAL,
                target_query_encode_latency_ms REAL,
                target_rerank_latency_ms REAL,
                cache_hits INTEGER NOT NULL DEFAULT 0,
                cache_misses INTEGER NOT NULL DEFAULT 0,
                target_candidates_available INTEGER NOT NULL DEFAULT 0,
                target_candidates_missing INTEGER NOT NULL DEFAULT 0,
                target_coverage REAL,
                top1_agreement REAL,
                top_k_overlap REAL,
                docs_queued INTEGER NOT NULL DEFAULT 0,
                unique_docs_queued INTEGER NOT NULL DEFAULT 0,
                docs_materialized INTEGER NOT NULL DEFAULT 0,
                docs_failed INTEGER NOT NULL DEFAULT 0,
                failure_category TEXT,
                error TEXT)""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS shadow_t2_windows (
                window_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at REAL NOT NULL,
                config_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                query_count INTEGER NOT NULL,
                diagnostics TEXT)""")
            self._db.execute("INSERT OR REPLACE INTO shadow_meta(key,value) VALUES('schema_version',?)",
                             (str(SCHEMA_VERSION),))
            if self.retention_days is not None:
                cutoff = time.time() - self.retention_days * 86400
                self._db.execute("DELETE FROM shadow_observations WHERE observed_at < ?", (cutoff,))
                self._db.execute("DELETE FROM shadow_t2_windows WHERE observed_at < ?", (cutoff,))
                self._db.execute("DELETE FROM shadow_counters WHERE bucket < ?", (self._bucket(cutoff),))
                self._db.execute("DELETE FROM shadow_counter_events WHERE observed_at < ?", (cutoff,))
            # Rehydrate the live all-time snapshot after a process restart.
            # Reports already read the durable event ledger directly, but
            # ``status`` is expected to show persisted counters immediately
            # rather than resetting them to zero until the next observation.
            counter_where = " WHERE config_fingerprint=?" if self.config_fingerprint else ""
            counter_values = (self.config_fingerprint,) if self.config_fingerprint else ()
            persisted = self._db.execute(
                "SELECT name,SUM(value) FROM shadow_counter_events" + counter_where + " GROUP BY name",
                counter_values,
            ).fetchall()
            if not persisted:
                # Databases created by the first Shadow release may not have
                # the exact-time event ledger yet; retain their bucketed
                # counters for the live snapshot until new events arrive.
                persisted = self._db.execute(
                    "SELECT name,SUM(value) FROM shadow_counters" + counter_where + " GROUP BY name",
                    counter_values,
                ).fetchall()
            for name, value in persisted:
                try:
                    self._counters[str(name)] = max(0.0, float(value or 0.0))
                except (TypeError, ValueError, OverflowError):
                    continue
            self._db.commit()
            self.available = True
        except Exception as exc:  # pragma: no cover - exercised by corruption tests
            self.error = _sanitize(exc)
            self.available = False
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
                self._db = None

    def _bucket(self, timestamp: float) -> int:
        return int(timestamp // 60) * 60

    def _safe_db(self) -> sqlite3.Connection | None:
        return self._db if self.available and self._db is not None else None

    def _disable_db(self, exc: Exception) -> None:
        """Disable a broken telemetry connection without affecting serving."""
        self.error = _sanitize(exc)
        self.available = False
        db = self._db
        self._db = None
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    def _append_counter_event_locked(self, *, timestamp: float, config_fingerprint: str,
                                     name: str, value: float, db: sqlite3.Connection | None) -> None:
        """Append an exact-time counter event while the telemetry lock is held."""
        self._counter_events.append({"observed_at": timestamp, "config_fingerprint": config_fingerprint,
                                     "name": name, "value": value})
        limit = max(1_024, self.max_records * 16)
        if len(self._counter_events) > limit:
            self._counter_events = self._counter_events[-limit:]
        if db is None:
            return
        db.execute("INSERT INTO shadow_counter_events(observed_at,config_fingerprint,name,value) VALUES(?,?,?,?)",
                   (timestamp, config_fingerprint, name, value))
        self._counter_event_writes += 1
        # Keep this ledger bounded without adding a DELETE to every request's
        # hot telemetry transaction.  A few hundred excess events are harmless
        # and are removed on the next periodic cleanup/open.
        if self._counter_event_writes % 256 == 0:
            db.execute("DELETE FROM shadow_counter_events WHERE event_id NOT IN "
                       "(SELECT event_id FROM shadow_counter_events ORDER BY event_id DESC LIMIT ?)",
                       (limit,))

    def increment(self, name: str, amount: int | float = 1, *, timestamp: float | None = None,
                  config_fingerprint: str | None = None) -> None:
        """Increment an aggregate counter without exposing exceptions."""
        try:
            value = float(amount)
            if not math.isfinite(value) or value < 0:
                return
            now = float(timestamp if timestamp is not None else time.time())
            if not math.isfinite(now):
                now = time.time()
            fp = str(config_fingerprint if config_fingerprint is not None else self.config_fingerprint)
            with self._lock:
                self._counters[name] += value
                db = self._safe_db()
                self._append_counter_event_locked(timestamp=now, config_fingerprint=fp,
                                                   name=name, value=value, db=db)
                if db is None:
                    return
                bucket = self._bucket(now)
                db.execute("""INSERT INTO shadow_counters(bucket,config_fingerprint,name,value)
                    VALUES(?,?,?,?) ON CONFLICT(bucket,config_fingerprint,name)
                    DO UPDATE SET value=value+excluded.value""", (bucket, fp, name, value))
                db.commit()
        except Exception as exc:
            self.error = _sanitize(exc)
            self.available = False

    def record_primary(self, *, eligible: bool = True, timestamp: float | None = None) -> None:
        self.increment("primary_requests_total", timestamp=timestamp)
        if eligible:
            self.increment("shadow_eligible_total", timestamp=timestamp)

    def record_sampled(self, *, timestamp: float | None = None) -> None:
        self.increment("shadow_sampled_total", timestamp=timestamp)

    def record_dropped(self, *, timestamp: float | None = None) -> None:
        self.increment("shadow_dropped_total", timestamp=timestamp)

    def record_observation(self, observation: ShadowObservation | Mapping[str, Any], *, timestamp: float | None = None) -> None:
        """Persist one completed/partial/failed/timeout observation."""
        try:
            data = observation.to_dict() if isinstance(observation, ShadowObservation) else dict(observation)
            status = str(data.get("status", "failed")).lower()
            if status not in {"completed", "partial", "failed", "timeout", "timed_out"}:
                status = "failed"
            now = float(timestamp if timestamp is not None else time.time())
            if not math.isfinite(now):
                now = time.time()
            fp = str(data.get("config_fingerprint", self.config_fingerprint))
            row = {
                "observed_at": now,
                "config_fingerprint": fp,
                # Even an explicitly retained request ID is treated as
                # untrusted input.  Hashing keeps it joinable without
                # persisting a caller accidentally placing query text in it.
                "request_id": hashlib.sha256(str(data.get("request_id")).encode("utf-8")).hexdigest()
                if self.retain_query_records and data.get("request_id") is not None else None,
                "status": status,
                "source_latency_ms": _finite(data.get("source_latency_ms")),
                "shadow_latency_ms": _finite(data.get("shadow_latency_ms")),
                "source_candidate_latency_ms": _finite(data.get("source_candidate_latency_ms")),
                "target_query_encode_latency_ms": _finite(data.get("target_query_encode_latency_ms")),
                "target_rerank_latency_ms": _finite(data.get("target_rerank_latency_ms")),
                "cache_hits": _nonnegative_int(data.get("cache_hits", 0)),
                "cache_misses": _nonnegative_int(data.get("cache_misses", 0)),
                "target_candidates_available": _nonnegative_int(data.get("target_candidates_available", 0)),
                "target_candidates_missing": _nonnegative_int(data.get("target_candidates_missing", 0)),
                "target_coverage": _finite(data.get("target_coverage")),
                "top1_agreement": _finite(data.get("top1_agreement")),
                "top_k_overlap": _finite(data.get("top_k_overlap")),
                "docs_queued": _nonnegative_int(data.get("docs_queued", 0)),
                "unique_docs_queued": _nonnegative_int(data.get("unique_docs_queued", data.get("docs_queued", 0))),
                "docs_materialized": _nonnegative_int(data.get("docs_materialized", 0)),
                "docs_failed": _nonnegative_int(data.get("docs_failed", 0)),
                # Categories are allow-listed before persistence.  Sanitizing
                # arbitrary provider text is not sufficient because it may
                # contain private query/document data that is not an
                # environment secret.
                "failure_category": (
                    str(data.get("failure_category")).upper()
                    if str(data.get("failure_category")).upper() in SAFE_FAILURE_CATEGORIES
                    else ("INTERNAL" if data.get("failure_category") else None)
                ),
                # Persist only a category-level message.  Third-party error
                # strings can contain private query/document text even after
                # generic redaction, while the category is enough for the
                # report and is stable for operators.
                "error": "shadow task failed" if data.get("error") else None,
            }
            with self._lock:
                self._events.append(dict(row))
                self._events = self._events[-self.max_records:]
                counter = _STATUS_COUNTERS.get(status)
                if counter:
                    self._counters[counter] += 1
                counter_fields = (("target_cache_hits", "cache_hits"),
                                  ("target_cache_misses", "cache_misses"),
                                  ("target_docs_queued", "docs_queued"),
                                  ("target_unique_docs_queued", "unique_docs_queued"),
                                  ("target_docs_materialized", "docs_materialized"),
                                  ("target_docs_failed", "docs_failed"))
                for key, field in counter_fields:
                    value = row[field]
                    self._counters[key] += value
                increments = []
                if counter:
                    increments.append((counter, 1.0))
                for key, field in counter_fields:
                    increments.append((key, float(row[field])))
                db = self._safe_db()
                if db is None:
                    # Keep the exact-time ledger in the in-memory fallback as
                    # well.  Without these events a report filtered by a
                    # window would see primary counters but silently miss
                    # completed/partial/failed observations.
                    for key, value in increments:
                        self._append_counter_event_locked(
                            timestamp=now, config_fingerprint=fp,
                            name=key, value=float(value), db=None
                        )
                    return
                columns = ",".join(row)
                placeholders = ",".join("?" for _ in row)
                db.execute(f"INSERT INTO shadow_observations({columns}) VALUES({placeholders})", tuple(row.values()))
                db.execute("""DELETE FROM shadow_observations WHERE event_id NOT IN
                    (SELECT event_id FROM shadow_observations ORDER BY event_id DESC LIMIT ?)""", (self.max_records,))
                bucket = self._bucket(now)
                for key, value in increments:
                    self._append_counter_event_locked(timestamp=now, config_fingerprint=fp,
                                                       name=key, value=float(value), db=db)
                    db.execute("""INSERT INTO shadow_counters(bucket,config_fingerprint,name,value)
                        VALUES(?,?,?,?) ON CONFLICT(bucket,config_fingerprint,name)
                        DO UPDATE SET value=value+excluded.value""", (bucket, fp, key, value))
                db.commit()
        except Exception as exc:
            # Telemetry must never be on the primary request's failure path.
            self.error = _sanitize(exc)
            self.available = False

    def record_t2_window(self, status: str, query_count: int, *, diagnostics: str | None = None,
                         timestamp: float | None = None, config_fingerprint: str | None = None) -> None:
        status = str(status).upper()
        if status not in {"SAFE", "EXPAND", "UNSAFE_OR_UNCERTAIN", "NOT_RUN"}:
            raise ValueError("T2 window status is invalid")
        if isinstance(query_count, bool) or int(query_count) != query_count or int(query_count) < 0:
            raise ValueError("T2 window query_count must be a non-negative integer")
        now = float(timestamp if timestamp is not None else time.time())
        if not math.isfinite(now):
            now = time.time()
        fp = str(config_fingerprint if config_fingerprint is not None else self.config_fingerprint)
        row = {"observed_at": now, "config_fingerprint": fp, "status": status,
               "query_count": int(query_count), "diagnostics": _sanitize(diagnostics) if diagnostics else None}
        try:
            with self._lock:
                self._t2_windows.append(row)
                self._t2_windows = self._t2_windows[-self.max_records:]
                db = self._safe_db()
                if db is not None:
                    db.execute("INSERT INTO shadow_t2_windows(observed_at,config_fingerprint,status,query_count,diagnostics) VALUES(?,?,?,?,?)",
                               tuple(row.values()))
                    db.commit()
        except Exception as exc:
            self.error = _sanitize(exc)
            self.available = False

    def _read_events(self, start: float | None, end: float | None, fp: str | None) -> list[dict[str, Any]]:
        db = self._safe_db()
        if db is None:
            rows = list(self._events)
        else:
            where, values = [], []
            if start is not None:
                where.append("observed_at>=?"); values.append(float(start))
            if end is not None:
                where.append("observed_at<=?"); values.append(float(end))
            if fp:
                where.append("config_fingerprint=?"); values.append(str(fp))
            query = "SELECT * FROM shadow_observations" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY event_id"
            try:
                cursor = db.execute(query, values)
                names = [item[0] for item in cursor.description]
                rows = [dict(zip(names, row)) for row in cursor.fetchall()]
            except Exception as exc:
                # A database can become unreadable after startup (partial
                # copy, disk error, operator truncation).  Treat it like a
                # telemetry outage and fall back to the bounded in-memory
                # window rather than taking status/API serving down.
                self._disable_db(exc)
                rows = list(self._events)
        if start is not None:
            rows = [row for row in rows if float(row.get("observed_at", 0)) >= start]
        if end is not None:
            rows = [row for row in rows if float(row.get("observed_at", 0)) <= end]
        if fp:
            rows = [row for row in rows if str(row.get("config_fingerprint", "")) == str(fp)]
        return rows

    def _read_counters(self, start: float | None, end: float | None, fp: str | None) -> dict[str, float]:
        db = self._safe_db()
        if db is None:
            rows = self._counter_events
            if rows:
                output: defaultdict[str, float] = defaultdict(float)
                for row in rows:
                    observed_at = float(row.get("observed_at", 0))
                    if start is not None and observed_at < start:
                        continue
                    if end is not None and observed_at > end:
                        continue
                    if fp and str(row.get("config_fingerprint", "")) != str(fp):
                        continue
                    output[str(row.get("name", ""))] += float(row.get("value", 0) or 0)
                return dict(output)
            # The in-memory fallback has no timestamped aggregate rows.  Do
            # not accidentally turn a bounded report window into an all-time
            # report when the database was unavailable before the event ledger
            # was populated.
            return dict(self._counters) if start is None and end is None and fp is None else {}
        where, values = [], []
        if start is not None:
            where.append("observed_at>=?"); values.append(float(start))
        if end is not None:
            where.append("observed_at<=?"); values.append(float(end))
        if fp:
            where.append("config_fingerprint=?"); values.append(str(fp))
        try:
            query = "SELECT name,SUM(value) FROM shadow_counter_events" + (" WHERE " + " AND ".join(where) if where else "") + " GROUP BY name"
            rows = db.execute(query, values).fetchall()
            # Databases created by a pre-event-ledger release have aggregate
            # rows but no exact events.  Preserve those historical reports;
            # once at least one event exists, an empty filtered result is a
            # genuine empty window and must not fall back to all-time totals.
            event_count = int(db.execute("SELECT COUNT(*) FROM shadow_counter_events").fetchone()[0] or 0)
            if rows or event_count:
                return {str(name): float(value or 0) for name, value in rows}
            aggregate_where, aggregate_values = [], []
            if start is not None:
                aggregate_where.append("bucket>=?"); aggregate_values.append(self._bucket(float(start)))
            if end is not None:
                aggregate_where.append("bucket<=?"); aggregate_values.append(self._bucket(float(end)))
            if fp:
                aggregate_where.append("config_fingerprint=?"); aggregate_values.append(str(fp))
            aggregate_query = "SELECT name,SUM(value) FROM shadow_counters" + (" WHERE " + " AND ".join(aggregate_where) if aggregate_where else "") + " GROUP BY name"
            return {str(name): float(value or 0) for name, value in db.execute(aggregate_query, aggregate_values).fetchall()}
        except Exception as exc:
            self._disable_db(exc)
            return self._read_counters(start, end, fp)

    def _read_t2(self, start: float | None, end: float | None, fp: str | None) -> dict[str, Any] | None:
        db = self._safe_db()
        if db is None:
            rows = list(self._t2_windows)
        else:
            where, values = [], []
            if start is not None:
                where.append("observed_at>=?"); values.append(float(start))
            if end is not None:
                where.append("observed_at<=?"); values.append(float(end))
            if fp:
                where.append("config_fingerprint=?"); values.append(str(fp))
            query = "SELECT observed_at,config_fingerprint,status,query_count,diagnostics FROM shadow_t2_windows" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY window_id DESC LIMIT 1"
            try:
                row = db.execute(query, values).fetchone()
                rows = [] if row is None else [{"observed_at": row[0], "config_fingerprint": row[1], "status": row[2], "query_count": row[3], "diagnostics": row[4]}]
            except Exception as exc:
                self._disable_db(exc)
                rows = list(self._t2_windows)
        if start is not None:
            rows = [row for row in rows if float(row.get("observed_at", 0)) >= start]
        if end is not None:
            rows = [row for row in rows if float(row.get("observed_at", 0)) <= end]
        if fp:
            rows = [row for row in rows if str(row.get("config_fingerprint", "")) == str(fp)]
        return rows[-1] if rows else None

    def report(self, *, since_seconds: float | None = None, start: float | None = None,
               end: float | None = None, config_fingerprint: str | None = None,
               queue: Mapping[str, Any] | None = None, candidate_k: int | None = None,
               sample_rate: float | None = None, telemetry_error: str | None = None) -> dict[str, Any]:
        now = time.time()
        if since_seconds is not None:
            if not math.isfinite(float(since_seconds)) or float(since_seconds) < 0:
                raise ValueError("since_seconds must be finite and non-negative")
            start = now - float(since_seconds)
        start = None if start is None else float(start)
        end = now if end is None else float(end)
        with self._lock:
            queue_data = dict(queue or {})
            # ``queue`` historically referred to the bounded ShadowRunner
            # dispatch queue.  When the serving engine supplies the
            # persistent target-materialization queue as a nested mapping,
            # report its depth separately so operators do not mistake pending
            # shadow jobs for pending document encodes.
            materialization_queue = queue_data.get("materialization_queue")
            if not isinstance(materialization_queue, Mapping):
                materialization_queue = {}
            materialization_depth = _nonnegative_int(
                materialization_queue.get("queue_depth", materialization_queue.get("pending", 0)) or 0
            )
            if "queue_depth" not in materialization_queue and "processing" in materialization_queue:
                materialization_depth += _nonnegative_int(materialization_queue.get("processing", 0) or 0)
            events = self._read_events(start, end, config_fingerprint)
            counters = self._read_counters(start, end, config_fingerprint)
            t2 = self._read_t2(start, end, config_fingerprint)
            if not events and not counters:
                return {
                    "schema_version": SCHEMA_VERSION, "mode": "shadow",
                    "window": {"start": start, "end": end},
                    "migration": {"config_fingerprint": config_fingerprint or self.config_fingerprint,
                                   "source_fingerprint": self.source_fingerprint,
                                   "target_fingerprint": self.target_fingerprint,
                                   "backend": self.backend,
                                   "index_identity": self.index_identity,
                                   "candidate_k": candidate_k},
                    "traffic": {"primary_requests_total": 0, "shadow_eligible_total": 0, "shadow_sampled_total": 0,
                                "shadow_completed_total": 0, "shadow_partial_total": 0, "shadow_failed_total": 0,
                                "shadow_timeout_total": 0, "shadow_dropped_total": 0},
                    "cache": {"hits": 0, "misses": 0, "hit_rate": 0.0},
                    "materialization": {"docs_queued": 0, "unique_docs_queued": 0, "docs_materialized": 0, "docs_failed": 0,
                                         "queue_depth": materialization_depth,
                                         "shadow_queue_depth": _nonnegative_int(queue_data.get("queue_depth", 0) or 0),
                                         "shadow_queue_capacity": _nonnegative_int(queue_data.get("queue_capacity", 0) or 0)},
                    "latency": {}, "ranking": {}, "target_coverage": {},
                    "t2": {"status": (t2 or {}).get("status", "NOT_RUN"), "queries": _nonnegative_int((t2 or {}).get("query_count", 0) or 0)},
                    "recommendation": "CONTINUE_SHADOW", "evidence": "NO_SHADOW_OBSERVATIONS",
                    "warnings": ["No shadow observations in the selected window."],
                    "warning_details": [{"code": "NO_SHADOW_OBSERVATIONS", "severity": "info",
                                         "message": "No shadow observations in the selected window.",
                                         "remediation": "Collect sampled traffic before evaluating the migration."}],
                }
            traffic_names = ("primary_requests_total", "shadow_eligible_total", "shadow_sampled_total",
                             "shadow_completed_total", "shadow_partial_total", "shadow_failed_total",
                             "shadow_timeout_total", "shadow_dropped_total")
            traffic = {name: _nonnegative_int(counters.get(name, 0)) for name in traffic_names}
            hits = _nonnegative_int(counters.get("target_cache_hits", 0)); misses = _nonnegative_int(counters.get("target_cache_misses", 0))
            completed = [row for row in events if str(row.get("status")) == "completed"]
            eligible = [row for row in completed if (_finite(row.get("target_coverage")) or 0.0) >= self.min_target_coverage_for_ranking]

            def values(field: str) -> list[float]:
                output = []
                for row in events:
                    value = _finite(row.get(field))
                    if value is not None and value >= 0:
                        output.append(value)
                return output

            def summary(field: str) -> dict[str, Any]:
                vals = values(field)
                if not vals:
                    return {"count": 0}
                vals.sort()
                def quantile(q: float) -> float:
                    if len(vals) == 1:
                        return vals[0]
                    position = (len(vals) - 1) * q
                    low, high = int(math.floor(position)), int(math.ceil(position))
                    if low == high:
                        return vals[low]
                    return vals[low] + (vals[high] - vals[low]) * (position - low)
                return {"count": len(vals), "p50_ms": quantile(.5), "p95_ms": quantile(.95), "mean_ms": sum(vals) / len(vals)}

            def mean(field: str) -> float | None:
                vals = [_finite(row.get(field)) for row in eligible]
                vals = [float(x) for x in vals if x is not None]
                return sum(vals) / len(vals) if vals else None

            coverage_values = [_finite(row.get("target_coverage")) for row in events]
            coverage_values = [float(x) for x in coverage_values if x is not None]
            warning_list: list[str] = []
            warning_details: list[dict[str, str]] = []
            if traffic["shadow_sampled_total"] < 20:
                warning_list.append("Evidence is LIMITED: fewer than 20 sampled shadow requests were observed.")
                warning_details.append({"code": "LIMITED_EVIDENCE", "severity": "warning", "message": warning_list[-1],
                                        "remediation": "Continue shadow observation until the sample is representative."})
            # A sampled request that was dropped, timed out, failed, or
            # remained partial is not evidence for a complete source-vs-target
            # comparison.  Do not let a manually recorded SAFE T2 window (or
            # a large sampled count) promote a report with no eligible
            # completed observations to canary guidance.
            if traffic["shadow_sampled_total"] and len(eligible) < 20:
                warning_list.append("Complete ranking evidence is LIMITED: fewer than 20 fully covered observations were recorded.")
                warning_details.append({"code": "LIMITED_COMPLETE_EVIDENCE", "severity": "warning", "message": warning_list[-1],
                                        "remediation": "Continue shadow observation until fully covered target comparisons accumulate."})
            if coverage_values and (sum(coverage_values) / len(coverage_values)) < self.min_target_coverage_for_ranking:
                warning_list.append("Target coverage is incomplete; ranking aggregates include only fully covered observations.")
                warning_details.append({"code": "PARTIAL_TARGET_COVERAGE", "severity": "warning", "message": warning_list[-1],
                                        "remediation": "Enable materialization or wait for the target cache to warm."})
            materialization_failures = _nonnegative_int(counters.get("target_docs_failed", 0))
            if materialization_failures:
                warning_list.append(f"Target materialization failures observed: {materialization_failures}; investigate document resolution or encoding.")
                warning_details.append({"code": "MATERIALIZATION_FAILURE", "severity": "error", "message": warning_list[-1],
                                        "remediation": "Inspect document-store and target-encoder errors before canary evaluation."})
            if traffic["shadow_sampled_total"]:
                failure_rate = (traffic["shadow_failed_total"] + traffic["shadow_timeout_total"]) / traffic["shadow_sampled_total"]
                if failure_rate > 0.05:
                    warning_list.append("Shadow failure/timeout rate is above 5%; investigate before canary evaluation.")
                    warning_details.append({"code": "SHADOW_FAILURE_RATE", "severity": "error", "message": warning_list[-1],
                                            "remediation": "Fix shadow failures/timeouts and continue observation."})
            # ANN fidelity is not measured by Shadow Mode itself.  Keep this
            # distinction visible without making an unknown (rather than
            # failed) health signal block a mature operational report.
            warning_list.append("ANN fidelity is UNKNOWN; Shadow Mode does not run an exact recall audit.")
            warning_details.append({"code": "ANN_FIDELITY_UNKNOWN", "severity": "info",
                                    "message": warning_list[-1],
                                    "remediation": "Run audit-index with an exact/reference index if ANN recall matters."})
            t2_status = str((t2 or {}).get("status", "NOT_RUN")).upper()
            blocking_warning = any(item.get("severity") in {"error", "critical"}
                                   for item in warning_details)
            # Operational failures take precedence over an analytical
            # request to expand K.  A larger candidate pool cannot repair a
            # broken target encoder, materializer, or overloaded shadow
            # path, so do not present EXPAND_K as the next action while a
            # blocking error is active.
            if blocking_warning or t2_status == "UNSAFE_OR_UNCERTAIN":
                recommendation = "INVESTIGATE"
            elif t2_status == "EXPAND":
                recommendation = "EXPAND_K"
            elif t2_status == "SAFE" and len(eligible) >= 20 and not blocking_warning and not any(
                item.get("severity") == "warning" for item in warning_details
            ):
                recommendation = "READY_FOR_CANARY_EVALUATION"
            else:
                recommendation = "CONTINUE_SHADOW"
            return {
                "schema_version": SCHEMA_VERSION, "mode": "shadow",
                "window": {"start": start, "end": end},
                "migration": {"config_fingerprint": config_fingerprint or self.config_fingerprint,
                               "source_fingerprint": self.source_fingerprint,
                               "target_fingerprint": self.target_fingerprint,
                               "backend": self.backend,
                               "index_identity": self.index_identity,
                               "candidate_k": candidate_k},
                "traffic": traffic,
                "cache": {"hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses)},
                "materialization": {"docs_queued": _nonnegative_int(counters.get("target_docs_queued", 0)),
                                     "unique_docs_queued": _nonnegative_int(counters.get("target_unique_docs_queued", 0)),
                                     "docs_materialized": _nonnegative_int(counters.get("target_docs_materialized", 0)),
                                     "docs_failed": _nonnegative_int(counters.get("target_docs_failed", 0)),
                                     "queue_depth": materialization_depth,
                                     "shadow_queue_depth": _nonnegative_int(queue_data.get("queue_depth", 0) or 0),
                                     "shadow_queue_capacity": _nonnegative_int(queue_data.get("queue_capacity", 0) or 0)},
                "latency": {"source": summary("source_latency_ms"), "shadow": summary("shadow_latency_ms"),
                            "source_candidate": summary("source_candidate_latency_ms"),
                            "target_query_encode": summary("target_query_encode_latency_ms"),
                            "target_rerank": summary("target_rerank_latency_ms")},
                "ranking": {"top1_agreement": mean("top1_agreement"), "top_k_overlap": mean("top_k_overlap"),
                            "eligible_completed_observations": len(eligible),
                            "completed_observations": len(completed)},
                "target_coverage": {"mean": sum(coverage_values) / len(coverage_values) if coverage_values else None,
                                     "min": min(coverage_values) if coverage_values else None,
                                     "observations": len(coverage_values),
                                     "minimum_for_complete_ranking": self.min_target_coverage_for_ranking},
                "t2": {"status": t2_status, "queries": _nonnegative_int((t2 or {}).get("query_count", len(completed)) or 0),
                       "diagnostics": (t2 or {}).get("diagnostics")},
                "sample_rate": sample_rate,
                "recommendation": recommendation,
                "evidence": "RANKING DISAGREEMENT AND OPERATIONAL DIAGNOSTICS; NOT QREL-BASED QUALITY",
                "warnings": warning_list,
                "warning_details": warning_details,
                "telemetry": {"enabled": self.enabled, "persistent": self.available, "error": telemetry_error or self.error},
            }

    def snapshot(self, *, queue: Mapping[str, Any] | None = None, candidate_k: int | None = None,
                 sample_rate: float | None = None) -> dict[str, Any]:
        with self._lock:
            counters = {key: int(round(value)) for key, value in self._counters.items()}
        return {"enabled": self.enabled, "persistent": self.available, "path": str(self.path) if self.path else None,
                "config_fingerprint": self.config_fingerprint, "counters": counters,
                "source_fingerprint": self.source_fingerprint, "target_fingerprint": self.target_fingerprint,
                "backend": self.backend, "index_identity": self.index_identity,
                "queue": dict(queue or {}), "candidate_k": candidate_k, "sample_rate": sample_rate,
                "error": self.error}

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
                self._db = None
                self.available = False


__all__ = ["SCHEMA_VERSION", "ShadowTelemetry", "index_identity_from_config", "migration_fingerprint"]
