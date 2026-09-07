from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
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
                 dtype: str = "float32"):
        p = Path(path)
        if p.suffix in {".sqlite", ".sqlite3", ".db"}:
            self.db_path = p
            self.root = p.parent
        else:
            self.root = p
            self.db_path = p / "cache.sqlite3"
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
        try:
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
                PRIMARY KEY(document_id, model_fingerprint)
            )""")
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
        if self._closed or self._db is None:
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

    def get(self, document_ids: list[str]) -> dict[str, np.ndarray]:
        ids = [str(x) for x in document_ids]
        if not ids: return {}
        with self._lock:
            self._ensure_open()
            rows: list[tuple[Any, ...]] = []
            for start in range(0, len(ids), self._LOOKUP_BATCH_SIZE):
                chunk = ids[start:start + self._LOOKUP_BATCH_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(self._db.execute(
                    f"SELECT document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at "
                    f"FROM target_vectors WHERE model_fingerprint=? AND document_id IN ({placeholders})",
                    [self.model_fingerprint, *chunk]).fetchall())
            now = time.time()
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

    def put(self, document_ids: list[str], vectors: np.ndarray) -> None:
        ids = [str(x) for x in document_ids]
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape != (len(ids), self.dimension):
            raise ValueError(f"vectors must have shape ({len(ids)}, {self.dimension})")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate IDs in cache write")
        if not np.isfinite(values).all():
            raise ValueError("cannot cache non-finite vectors")
        now = time.time(); records = []
        for document_id, value in zip(ids, values):
            blob = np.ascontiguousarray(value).tobytes()
            records.append((document_id, self.model_fingerprint, self.dimension, self.dtype.name, sqlite3.Binary(blob),
                            hashlib.sha256(blob).hexdigest(), now, now))
        with self._lock:
            self._ensure_open()
            self._db.executemany("""INSERT INTO target_vectors
                (document_id,model_fingerprint,dimension,dtype,vector,checksum,created_at,accessed_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id,model_fingerprint) DO UPDATE SET
                    dimension=excluded.dimension,dtype=excluded.dtype,vector=excluded.vector,
                    checksum=excluded.checksum,accessed_at=excluded.accessed_at""", records)
            self._db.commit()

    def contains(self, document_ids: list[str]) -> set[str]:
        return set(self.get(document_ids))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
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
                self._db.close()
                self._db = None
                self._closed = True


__all__ = ["SQLiteVectorCache", "CacheCorruptionError"]
