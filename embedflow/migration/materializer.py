from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import SQLiteVectorCache


class PersistentWorkQueue:
    def __init__(self, path: str | Path, max_retries: int = 3):
        p = Path(path); self.path = p if p.suffix else p / "materialization_queue.sqlite3"
        try:
            retries = int(max_retries)
            exact = float(max_retries) == retries
        except (TypeError, ValueError, OverflowError):
            retries, exact = 0, False
        if isinstance(max_retries, bool) or not exact or retries < 1:
            raise ValueError("max_retries must be a positive integer")
        self.path.parent.mkdir(parents=True, exist_ok=True); self.max_retries = retries
        self.db = None; self.lock = threading.RLock()
        try:
            self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            self.db.execute("PRAGMA busy_timeout=30000")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS work (
                document_id TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT, enqueued_at REAL NOT NULL, updated_at REAL NOT NULL)""")
            self.db.execute("UPDATE work SET status='pending',updated_at=? WHERE status='processing'", (time.time(),)); self.db.commit()
        except sqlite3.DatabaseError as exc:
            if self.db is not None:
                self.db.close()
            self.db = None
            raise ValueError(f"invalid materialization queue database: {self.path}") from exc

    def _ensure_open(self) -> None:
        if self.db is None:
            raise RuntimeError("materialization queue is closed")

    def enqueue(self, ids: Iterable[str]) -> int:
        now = time.time(); n = 0
        with self.lock:
            self._ensure_open()
            for document_id in dict.fromkeys(str(x) for x in ids):
                cur = self.db.execute("INSERT OR IGNORE INTO work(document_id,status,enqueued_at,updated_at) VALUES(?,?,?,?)",
                                      (document_id, "pending", now, now)); n += cur.rowcount
            self.db.commit()
        return n

    def claim(self, limit: int) -> list[str]:
        if isinstance(limit, bool) or int(limit) != limit or int(limit) < 1:
            raise ValueError("queue claim limit must be a positive integer")
        with self.lock:
            self._ensure_open()
            # A write transaction makes SELECT+UPDATE atomic across multiple
            # worker processes sharing the same SQLite queue.  Without it,
            # two workers can claim the same pending document before either
            # marks it processing.
            self.db.execute("BEGIN IMMEDIATE")
            try:
                rows = self.db.execute("SELECT document_id FROM work WHERE status='pending' AND attempts<? ORDER BY enqueued_at,document_id LIMIT ?",
                                       (self.max_retries, int(limit))).fetchall()
                ids = [str(x[0]) for x in rows]; now = time.time()
                self.db.executemany("UPDATE work SET status='processing',attempts=attempts+1,updated_at=? WHERE document_id=?",
                                    [(now, x) for x in ids]); self.db.commit(); return ids
            except Exception:
                self.db.rollback()
                raise

    def complete(self, ids: Iterable[str]) -> None:
        with self.lock:
            self._ensure_open()
            self.db.executemany("UPDATE work SET status='done',updated_at=? WHERE document_id=?", [(time.time(), str(x)) for x in ids]); self.db.commit()

    def fail(self, ids: Iterable[str], error: str) -> None:
        if not str(error).strip():
            error = "unknown materialization failure"
        with self.lock:
            self._ensure_open()
            self.db.executemany("UPDATE work SET status=CASE WHEN attempts<? THEN 'pending' ELSE 'error' END,error=?,updated_at=? WHERE document_id=?",
                                [(self.max_retries, str(error), time.time(), str(x)) for x in ids]); self.db.commit()

    def stats(self) -> dict[str, int]:
        with self.lock:
            self._ensure_open()
            rows = self.db.execute("SELECT status,COUNT(*) FROM work GROUP BY status").fetchall()
        out = {"pending": 0, "processing": 0, "done": 0, "error": 0}; out.update({str(k): int(v) for k, v in rows}); return out

    def close(self) -> None:
        with self.lock:
            if self.db is not None:
                self.db.close()
                self.db = None


class MaterializationWorker:
    def __init__(self, target_model: Any, documents: Any, cache: SQLiteVectorCache,
                 queue_path: str | Path, batch_size: int = 32, max_retries: int = 3,
                 state: Any | None = None):
        self.target_model, self.documents, self.cache = target_model, documents, cache
        try:
            parsed_batch_size = int(batch_size)
            exact = float(batch_size) == parsed_batch_size
        except (TypeError, ValueError, OverflowError):
            parsed_batch_size, exact = 0, False
        if isinstance(batch_size, bool) or not exact or parsed_batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.queue = PersistentWorkQueue(queue_path, max_retries); self.batch_size = parsed_batch_size; self.state = state
        self.stop_event = threading.Event(); self.thread: threading.Thread | None = None
        self._closed = False
        self._stats = {"materialized": 0, "errors": 0, "started_at": None, "last_throughput_docs_sec": 0.0}

    def enqueue(self, ids: Iterable[str]) -> int:
        if self._closed:
            raise RuntimeError("materialization worker is closed")
        return self.queue.enqueue(ids)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("materialization worker is closed")
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear(); self._stats["started_at"] = time.time(); self.thread = threading.Thread(target=self._run, name="embedflow-materializer", daemon=True); self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                ids = self.queue.claim(self.batch_size)
            except RuntimeError:
                if self.stop_event.is_set() or self._closed:
                    return
                raise
            except Exception as exc:
                self._stats["errors"] += 1
                if self.state:
                    self.state.add_error(f"materialization queue failed: {exc}")
                time.sleep(0.1)
                continue
            if not ids:
                time.sleep(0.05)
                continue
            try:
                # Resolve documents individually so one missing/deleted row is
                # marked as an error without poisoning the rest of a batch.
                available: list[str] = []
                texts: list[str] = []
                missing: list[str] = []
                for document_id in ids:
                    try:
                        value = self.documents.get([document_id]).get(document_id)
                    except Exception:
                        value = None
                    if value is None:
                        missing.append(document_id)
                    else:
                        available.append(document_id); texts.append(str(value))
                if missing:
                    self.queue.fail(missing, "document text is unavailable")
                    self._stats["errors"] += len(missing)
                if not available:
                    continue
                vectors = self.target_model.encode_documents(texts, batch_size=self.batch_size)
                self.cache.put(available, np.asarray(vectors, dtype="float32")); self.queue.complete(available)
                self._stats["materialized"] += len(available)
                elapsed = max(time.time() - float(self._stats["started_at"] or time.time()), 1e-6)
                self._stats["last_throughput_docs_sec"] = self._stats["materialized"] / elapsed
                if self.state: self.state.update(materialized_documents=self._stats["materialized"], materializer_throughput_docs_sec=self._stats["last_throughput_docs_sec"], queue=self.queue.stats())
            except Exception as exc:
                self.queue.fail(ids, repr(exc)); self._stats["errors"] += len(ids)
                if self.state: self.state.add_error(f"materialization failed for {len(ids)} docs: {exc}")

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise RuntimeError("materialization worker did not stop before timeout")

    def stats(self) -> dict[str, Any]: return {**self._stats, "queue": self.queue.stats()}

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self.stop(timeout=timeout)
        self.queue.close()
        self._closed = True
