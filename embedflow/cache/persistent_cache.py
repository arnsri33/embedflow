from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .base import TargetVectorCache


class CacheCorruptionError(RuntimeError):
    pass


class SQLiteVectorCache(TargetVectorCache):
    """Persistent target-vector cache keyed by document and model fingerprint."""

    # SQLite builds differ in their maximum number of bound variables.  Keep
    # lookups below the conservative default so prewarming large corpora works
    # on older and embedded SQLite versions as well as on modern builds.
    _LOOKUP_BATCH_SIZE = 900

    def __init__(self, path: str | Path, model_fingerprint: str, dimension: int,
                 dtype: str = "float32", *, read_only: bool = False):
        p = Path(path)
        if p.suffix in {".sqlite", ".sqlite3", ".db"}:
            self.db_path = p
            self.root = p.parent
        else:
            self.root = p
            self.db_path = p / "cache.sqlite3"
        self.read_only = bool(read_only)
        # A planning/status inspection must not create a cache directory or
        # migrate its schema.  Normal serving retains the historical
        # create-or-open behavior; read-only callers use SQLite's URI mode and
        # receive an empty view when the cache has not been initialized yet.
        if not self.read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = str(model_fingerprint).strip()
        if not self.model_fingerprint:
            raise ValueError("model_fingerprint must be non-empty")
        try:
            parsed_dimension = int(dimension)
            exact = float(dimension) == parsed_dimension
        except (TypeError, ValueError, OverflowError):
            parsed_dimension, exact = 0, False
        if isinstance(dimension, bool) or not exact or parsed_dimension < 1:
            raise ValueError("cache dimension must be a positive integer")
        self.dimension = parsed_dimension
        try:
            self.dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported cache dtype: {dtype!r}") from exc
        if self.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("cache dtype must be a numeric vector dtype")
        self._lock = threading.RLock()
        self._db = None
        self._has_content_fingerprint = True
        self._has_target_vectors = True
        try:
            if self.read_only:
                if self.db_path.exists():
                    self._db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True,
                                                check_same_thread=False, timeout=30)
                    self._db.execute("PRAGMA busy_timeout=30000")
                    columns = {str(row[1]) for row in self._db.execute(
                        "PRAGMA table_info(target_vectors)").fetchall()}
                    self._has_target_vectors = bool(columns)
                    self._has_content_fingerprint = "content_fingerprint" in columns
                self._closed = False
                return
            self._db = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
            self._db.execute("PRAGMA busy_timeout=30000")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("""CREATE TABLE IF NOT EXISTS target_vectors (
                document_id TEXT NOT NULL,
                model_fingerprint TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                checksum TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                content_fingerprint TEXT,
                PRIMARY KEY(document_id, model_fingerprint)
            )""")
            # Caches created by pre-0.8 releases do not have a content
            # fingerprint.  Keep those vectors readable for existing serving
            # callers, while allowing prewarming to require a current-content
            # match when one is available.
            columns = {str(row[1]) for row in self._db.execute("PRAGMA table_info(target_vectors)").fetchall()}
            if "content_fingerprint" not in columns:
                self._db.execute("ALTER TABLE target_vectors ADD COLUMN content_fingerprint TEXT")
            self._has_content_fingerprint = True
            self._has_target_vectors = True
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_target_vectors_model ON target_vectors(model_fingerprint)")
            self._db.execute("""CREATE TABLE IF NOT EXISTS cache_counters (
                model_fingerprint TEXT PRIMARY KEY,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0
            )""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS cache_access_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_fingerprint TEXT NOT NULL,
                hit INTEGER NOT NULL,
                accessed_at REAL NOT NULL
            )""")
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_cache_events_model ON cache_access_events(model_fingerprint, event_id)")
            self._db.execute("INSERT OR IGNORE INTO cache_counters(model_fingerprint) VALUES (?)", (self.model_fingerprint,))
            self._db.commit()
        except sqlite3.DatabaseError as exc:
            if self._db is not None:
                self._db.close()
            self._db = None
            raise CacheCorruptionError(f"invalid SQLite target-vector cache: {self.db_path}") from exc
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed or (self._db is None and not self.read_only):
            raise RuntimeError("target vector cache is closed")

    def _decode(self, row: tuple[Any, ...]) -> np.ndarray:
        document_id, _, dimension, dtype, blob, checksum, *_ = row
        if int(dimension) != self.dimension or np.dtype(dtype) != self.dtype:
            raise CacheCorruptionError(f"cache contract mismatch for document {document_id}")
        if hashlib.sha256(blob).hexdigest() != checksum:
            raise CacheCorruptionError(f"cache checksum mismatch for document {document_id}")
        value = np.frombuffer(blob, dtype=self.dtype).copy()
        if value.size != self.dimension or not np.isfinite(value).all():
            raise CacheCorruptionError(f"invalid cached vector for document {document_id}")
        return value.astype("float32", copy=False)

    @staticmethod
    def content_fingerprint(text: Any) -> str:
        """Return the stable fingerprint used to bind a vector to content."""
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_content_fingerprints(document_ids: list[str], values: Any) -> dict[str, str] | None:
        if values is None:
            return None
        if isinstance(values, Mapping):
            return {str(key): str(value) for key, value in values.items()}
        try:
            sequence = list(values)
        except TypeError as exc:
            raise ValueError("content_fingerprints must be a mapping or sequence") from exc
        if len(sequence) != len(document_ids):
            raise ValueError("content_fingerprints must match document_ids length")
        return {str(document_id): str(value) for document_id, value in zip(document_ids, sequence)}

    def get(self, document_ids: list[str], content_fingerprints: Any = None) -> dict[str, np.ndarray]:
        if self.read_only:
            # Preserve the familiar lookup contract for inspection callers
            # without updating hit/miss/access timestamps in a read-only
            # SQLite connection.
            return self.peek(document_ids, content_fingerprints=content_fingerprints)
        ids = [str(x) for x in document_ids]
        if not ids: return {}
        expected_content = self._normalize_content_fingerprints(ids, content_fingerprints)
        with self._lock:
            self._ensure_open()
            rows: list[tuple[Any, ...]] = []
            for start in range(0, len(ids), self._LOOKUP_BATCH_SIZE):
                chunk = ids[start:start + self._LOOKUP_BATCH_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                columns = ",content_fingerprint" if self._has_content_fingerprint else ""
                rows.extend(self._db.execute(
                    f"SELECT document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at{columns} "
                    f"FROM target_vectors WHERE model_fingerprint=? AND document_id IN ({placeholders})",
                    [self.model_fingerprint, *chunk]).fetchall())
            now = time.time()
            if expected_content is not None:
                # Rows from pre-0.8 caches have no content binding.  Preserve
                # their historical serving behavior while requiring an exact
                # match whenever a binding is present (new writes and
                # materialized vectors).
                if self._has_content_fingerprint:
                    rows = [row for row in rows if row[8] is None or str(row[8]) == expected_content.get(str(row[0]))]
            values = {str(row[0]): self._decode(row) for row in rows}
            if rows:
                self._db.executemany("UPDATE target_vectors SET accessed_at=? WHERE document_id=? AND model_fingerprint=?",
                                     [(now, str(row[0]), self.model_fingerprint) for row in rows])
            # Count accesses, rather than unique rows, so the hit-rate remains
            # meaningful when a caller supplies duplicate IDs.
            hits = sum(document_id in values for document_id in ids)
            misses = len(ids) - hits
            self._db.execute("UPDATE cache_counters SET hits=hits+?,misses=misses+? WHERE model_fingerprint=?",
                             (hits, misses, self.model_fingerprint))
            self._db.executemany("INSERT INTO cache_access_events(model_fingerprint,hit,accessed_at) VALUES (?,?,?)",
                                 [(self.model_fingerprint, int(document_id in values), now) for document_id in ids])
            # Keep the persistent telemetry table bounded while retaining a
            # useful recent-hit-rate window for status and dashboards.
            self._db.execute("""DELETE FROM cache_access_events
                WHERE model_fingerprint=? AND event_id NOT IN
                (SELECT event_id FROM cache_access_events WHERE model_fingerprint=? ORDER BY event_id DESC LIMIT 1000)""",
                             (self.model_fingerprint, self.model_fingerprint))
            self._db.commit()
            return values

    def peek(self, document_ids: list[str], content_fingerprints: Any = None) -> dict[str, np.ndarray]:
        """Read cached vectors without changing hit/miss telemetry.

        Shadow analysis with ``materialize=false`` must not make ordinary
        serving cache statistics look like migration traffic.  This method is
        intentionally a read-only fast path; callers that need accounting
        should continue to use :meth:`get`.
        """
        ids = [str(x) for x in document_ids]
        if not ids:
            return {}
        expected_content = self._normalize_content_fingerprints(ids, content_fingerprints)
        with self._lock:
            self._ensure_open()
            if self._db is None or not self._has_target_vectors:
                return {}
            rows: list[tuple[Any, ...]] = []
            for start in range(0, len(ids), self._LOOKUP_BATCH_SIZE):
                chunk = ids[start:start + self._LOOKUP_BATCH_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                columns = ",content_fingerprint" if self._has_content_fingerprint else ""
                rows.extend(self._db.execute(
                    f"SELECT document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at{columns} "
                    f"FROM target_vectors WHERE model_fingerprint=? AND document_id IN ({placeholders})",
                    [self.model_fingerprint, *chunk]).fetchall())
            if expected_content is not None and self._has_content_fingerprint:
                rows = [row for row in rows if row[8] is None or str(row[8]) == expected_content.get(str(row[0]))]
            return {str(row[0]): self._decode(row) for row in rows}

    def put(self, document_ids: list[str], vectors: np.ndarray, content_fingerprints: Any = None) -> None:
        if self.read_only:
            raise RuntimeError("target vector cache is read-only")
        ids = [str(x) for x in document_ids]
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape != (len(ids), self.dimension):
            raise ValueError(f"vectors must have shape ({len(ids)}, {self.dimension})")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate IDs in cache write")
        if not np.isfinite(values).all():
            raise ValueError("cannot cache non-finite vectors")
        content = self._normalize_content_fingerprints(ids, content_fingerprints)
        now = time.time(); records = []
        for document_id, value in zip(ids, values):
            blob = np.ascontiguousarray(value).tobytes()
            records.append((document_id, self.model_fingerprint, self.dimension, self.dtype.name, sqlite3.Binary(blob),
                            hashlib.sha256(blob).hexdigest(), now, now,
                            content.get(document_id) if content is not None else None))
        with self._lock:
            self._ensure_open()
            if content is None:
                # Older callers do not know document content.  Updating the
                # vector must not erase a binding written by the new
                # materializer, otherwise a later content change could be
                # mistaken for a valid cache hit.  New rows remain unbound and
                # therefore retain historical compatibility.
                self._db.executemany("""INSERT INTO target_vectors
                    (document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at,content_fingerprint)
                    VALUES (?,?,?,?,?,?,?,?,NULL)
                    ON CONFLICT(document_id,model_fingerprint) DO UPDATE SET
                        dimension=excluded.dimension,dtype=excluded.dtype,vector=excluded.vector,
                        checksum=excluded.checksum,accessed_at=excluded.accessed_at""",
                                       [record[:8] for record in records])
            else:
                self._db.executemany("""INSERT INTO target_vectors
                    (document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at,content_fingerprint)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(document_id,model_fingerprint) DO UPDATE SET
                        dimension=excluded.dimension,dtype=excluded.dtype,vector=excluded.vector,
                        checksum=excluded.checksum,accessed_at=excluded.accessed_at,
                        content_fingerprint=excluded.content_fingerprint""", records)
            self._db.commit()

    def contains(self, document_ids: list[str], content_fingerprints: Any = None) -> set[str]:
        if self.read_only:
            return set(self.peek(document_ids, content_fingerprints=content_fingerprints))
        return set(self.get(document_ids, content_fingerprints=content_fingerprints))

    def content_fingerprints(self, document_ids: list[str]) -> dict[str, str | None]:
        """Inspect content bindings without changing cache hit/miss counters."""
        ids = [str(x) for x in document_ids]
        if not ids:
            return {}
        with self._lock:
            self._ensure_open()
            if self._db is None or not self._has_target_vectors or not self._has_content_fingerprint:
                return {}
            rows: list[tuple[Any, ...]] = []
            for start in range(0, len(ids), self._LOOKUP_BATCH_SIZE):
                chunk = ids[start:start + self._LOOKUP_BATCH_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(self._db.execute(
                    f"SELECT document_id,content_fingerprint FROM target_vectors "
                    f"WHERE model_fingerprint=? AND document_id IN ({placeholders})",
                    [self.model_fingerprint, *chunk]).fetchall())
            return {str(row[0]): (None if row[1] is None else str(row[1])) for row in rows}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            if self._db is None or not self._has_target_vectors:
                return {"cached_target_vectors": 0, "all_model_vectors": 0,
                        "model_fingerprint": self.model_fingerprint, "dimension": self.dimension,
                        "cache_path": str(self.db_path), "total_cache_hits": 0,
                        "total_cache_misses": 0, "recent_hit_rate": 0.0}
            count = self._db.execute("SELECT COUNT(*) FROM target_vectors WHERE model_fingerprint=?", (self.model_fingerprint,)).fetchone()[0]
            total = self._db.execute("SELECT COUNT(*) FROM target_vectors").fetchone()[0]
            counters = self._db.execute("SELECT hits,misses FROM cache_counters WHERE model_fingerprint=?",
                                        (self.model_fingerprint,)).fetchone() or (0, 0)
            recent = self._db.execute("""SELECT SUM(hit),COUNT(*) FROM cache_access_events
                WHERE model_fingerprint=? AND event_id IN
                (SELECT event_id FROM cache_access_events WHERE model_fingerprint=? ORDER BY event_id DESC LIMIT 100)""",
                                     (self.model_fingerprint, self.model_fingerprint)).fetchone()
        total_hits, total_misses = int(counters[0]), int(counters[1])
        recent_hits, recent_count = int(recent[0] or 0), int(recent[1] or 0)
        return {"cached_target_vectors": int(count), "all_model_vectors": int(total),
                "model_fingerprint": self.model_fingerprint, "dimension": self.dimension,
                "cache_path": str(self.db_path), "total_cache_hits": total_hits,
                "total_cache_misses": total_misses,
                "recent_hit_rate": (recent_hits / recent_count) if recent_count else 0.0}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                if self._db is not None:
                    self._db.close()
                self._db = None
                self._closed = True


__all__ = ["SQLiteVectorCache", "CacheCorruptionError"]
