from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import EmbedFlowConfig


class DocumentStore:
    """Memory-indexed JSONL document store for progressive materialization."""

    def __init__(self, path: str | Path, id_field: str = "id", text_field: str = "text"):
        self.path = Path(path)
        if not self.path.exists(): raise FileNotFoundError(self.path)
        if not isinstance(id_field, str) or not id_field.strip() or not isinstance(text_field, str) or not text_field.strip():
            raise ValueError("document id_field and text_field must be non-empty strings")
        self.documents: dict[str, str] = {}
        self.id_field, self.text_field = id_field, text_field
        with self.path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip(): continue
                try: row = json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON at {self.path}:{line_no}") from exc
                if not isinstance(row, dict): raise ValueError(f"document row {line_no} in {self.path} must be a JSON object")
                if id_field not in row or text_field not in row: raise ValueError(f"missing {id_field}/{text_field} at {self.path}:{line_no}")
                document_id = str(row[id_field])
                if not document_id.strip(): raise ValueError(f"empty document ID at {self.path}:{line_no}")
                if document_id in self.documents: raise ValueError(f"duplicate document ID {document_id!r}")
                text = row[text_field]
                if text is None or not str(text).strip(): raise ValueError(f"missing document text for {document_id!r}")
                self.documents[document_id] = str(text)
        if not self.documents: raise ValueError(f"document file is empty: {self.path}")

    def get(self, document_ids: Iterable[str]) -> dict[str, str]:
        ids = [str(x) for x in document_ids]
        missing = [x for x in ids if x not in self.documents]
        if missing: raise KeyError(f"document text missing for IDs: {missing[:5]}")
        return {x: self.documents[x] for x in ids}

    def size(self) -> int: return len(self.documents)


class MigrationState:
    def __init__(self, path: str | Path, cfg: EmbedFlowConfig):
        self.path = Path(path); self._lock = threading.RLock()
        self.data: dict[str, Any] = {
            "source_model": cfg.source.model, "target_model": cfg.target.model,
            "target_model_fingerprint": cfg.target.fingerprint, "status": "INITIALIZING",
            "diagnostic": "UNKNOWN", "ann_status": "UNKNOWN", "candidate_depth": cfg.migration.candidate_depth,
            "migration_strategy": "PROGRESSIVE", "queries": 0, "errors": [],
        }
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError) as exc: raise ValueError(f"invalid state file {self.path}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"state file {self.path} must contain a JSON object")
            stored_fingerprint = loaded.get("target_model_fingerprint")
            if stored_fingerprint and str(stored_fingerprint) != str(cfg.target.fingerprint):
                raise ValueError("state file target model fingerprint does not match the configured target contract")
            self.data.update(loaded)
        self.save()

    def update(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values); self.save()

    def add_error(self, message: str) -> None:
        with self._lock:
            errors = self.data.setdefault("errors", []); errors.append(str(message)); self.data["errors"] = errors[-100:]; self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False)); tmp.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock: return json.loads(json.dumps(self.data))
