"""Read-only pgvector adapter for existing PostgreSQL vector tables.

The adapter intentionally has the same small surface as the FAISS and Qdrant
backends.  It only retrieves candidates and document text; it never creates,
alters, or drops a user's table, extension, or index.  PostgreSQL identifiers
are composed with :mod:`psycopg.sql`, while vectors and IDs remain query
parameters throughout.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .base import SearchHit, VectorIndex, validate_k, validate_query_vector

_METRIC_ALIASES = {
    "cosine": "cosine",
    "l2": "l2",
    "euclidean": "l2",
    "dot": "inner_product",
    "inner_product": "inner_product",
}
_METRIC_OPERATORS = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}
_VECTOR_DIMENSION = re.compile(r"^vector\((\d+)\)$", re.IGNORECASE)


def normalize_pgvector_metric(metric: str) -> str:
    """Return the canonical pgvector metric name.

    ``euclidean`` and ``dot`` are accepted as user-facing aliases to retain
    EmbedFlow's existing terminology.  Internally pgvector uses ``l2`` and
    ``inner_product`` for the corresponding operators.
    """

    if not isinstance(metric, str) or metric.strip().lower() not in _METRIC_ALIASES:
        raise ValueError("pgvector metric must be cosine, l2/euclidean, or inner_product/dot")
    return _METRIC_ALIASES[metric.strip().lower()]


def _identifier(value: str, label: str) -> str:
    """Validate an identifier before passing it to ``sql.Identifier``.

    Psycopg performs the actual quoting.  Rejecting NUL avoids a confusing
    driver error while still allowing quoted names, reserved words, and other
    valid PostgreSQL identifiers.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pgvector {label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"pgvector {label} contains a NUL byte")
    return value


def _vector_literal(vector: np.ndarray, dimension: int) -> str:
    """Serialize a checked vector for a parameterized ``::vector`` cast."""

    value = validate_query_vector(vector, dimension)
    # ``repr`` preserves enough float32 precision for a database distance
    # calculation.  The resulting string is still a bound parameter, never
    # interpolated into SQL text.
    return "[" + ",".join(repr(float(item)) for item in value) + "]"


def _redact_error(exc: BaseException) -> str:
    """Return a connection error without exposing a password-bearing DSN."""

    message = str(exc)
    # Psycopg normally omits passwords, but drivers/proxies are not required to
    # do so.  Keep the error useful while removing URI credentials if present.
    message = re.sub(r"(postgres(?:ql)?://[^:/\s]+:)[^@\s]+(@)", r"\1<redacted>\2", message, flags=re.IGNORECASE)
    message = re.sub(r"([?&](?:password|passfile|sslpassword)=)[^&\s]+", r"\1<redacted>", message, flags=re.IGNORECASE)
    message = re.sub(r"(password\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", message, flags=re.IGNORECASE)
    return message or exc.__class__.__name__


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised in clean installs
        raise RuntimeError('pgvector support requires psycopg; install it with `pip install "embedflow[pgvector]"`') from exc
    return psycopg


class PgVectorIndex(VectorIndex):
    """Read-only ``VectorIndex`` backed by a user-owned pgvector table.

    ``connect`` is the normal entry point.  ``connection`` may be supplied to
    the constructor by tests or applications that manage a psycopg connection
    themselves.  The adapter serializes operations on one connection with a
    lock, which is safe for concurrent FastAPI requests without creating a
    connection per candidate lookup.
    """

    def __init__(
        self,
        connection: Any,
        *,
        schema: str = "public",
        table: str = "documents",
        id_column: str = "id",
        vector_column: str = "embedding",
        text_column: str | None = "content",
        dimension: int,
        metric: str = "cosine",
        dsn_env: str | None = "EMBEDFLOW_PGVECTOR_DSN",
        hnsw_ef_search: int | None = None,
        ivfflat_probes: int | None = None,
        owns_connection: bool = True,
    ) -> None:
        if connection is None:
            raise ValueError("pgvector connection is required")
        try:
            dimension_value = int(dimension)
            dimension_exact = float(dimension) == dimension_value
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pgvector dimension must be a positive integer") from exc
        if isinstance(dimension, bool) or not dimension_exact or dimension_value < 1:
            raise ValueError("pgvector dimension must be a positive integer")
        self.schema = _identifier(schema, "schema")
        self.table = _identifier(table, "table")
        self.id_column = _identifier(id_column, "id_column")
        self.vector_column = _identifier(vector_column, "vector_column")
        self.text_column = None if text_column is None else _identifier(text_column, "text_column")
        self.metric = normalize_pgvector_metric(metric)
        self.dimension = dimension_value
        self.dsn_env = dsn_env
        self.hnsw_ef_search = self._setting(hnsw_ef_search, "hnsw_ef_search")
        self.ivfflat_probes = self._setting(ivfflat_probes, "ivfflat_probes")
        self.connection = connection
        self._owns_connection = bool(owns_connection)
        self._lock = threading.RLock()
        self._closed = False
        self._size_cache: int | None = None
        self._size_is_approximate = False
        self._extension_version: str | None = None
        self._ann_indexes: list[dict[str, str]] | None = None

    @staticmethod
    def _setting(value: int | None, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"pgvector {label} must be a positive integer")
        try:
            parsed = int(value)
            exact = float(value) == parsed
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"pgvector {label} must be a positive integer") from exc
        if not exact or parsed < 1:
            raise ValueError(f"pgvector {label} must be a positive integer")
        return parsed

    @classmethod
    def connect(
        cls,
        dsn: str | None = None,
        *,
        dsn_env: str | None = "EMBEDFLOW_PGVECTOR_DSN",
        schema: str = "public",
        table: str = "documents",
        id_column: str = "id",
        vector_column: str = "embedding",
        text_column: str | None = "content",
        dimension: int | None = None,
        metric: str = "cosine",
        hnsw_ef_search: int | None = None,
        ivfflat_probes: int | None = None,
        allow_missing_text: bool = False,
    ) -> PgVectorIndex:
        """Connect to an existing table and validate its vector contract.

        ``dsn`` is intended for an application that already owns a secret. In
        YAML deployments, leave it unset and provide the environment variable
        named by ``dsn_env`` instead.
        """

        if dsn is None:
            env_name = dsn_env or "EMBEDFLOW_PGVECTOR_DSN"
            if not isinstance(env_name, str) or not env_name.strip():
                raise ValueError("pgvector dsn_env must be a non-empty environment-variable name")
            dsn = os.environ.get(env_name.strip())
            if not dsn:
                raise RuntimeError(f"pgvector DSN not found; set the {env_name.strip()} environment variable")
            dsn_env = env_name.strip()
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("pgvector DSN must be a non-empty string")
        psycopg = _psycopg()
        try:
            try:
                connection = psycopg.connect(dsn, autocommit=False)
            except TypeError:
                # Small connection doubles used by applications/tests may not
                # accept psycopg's keyword; real psycopg does.
                connection = psycopg.connect(dsn)
        except Exception as exc:
            raise RuntimeError(f"could not connect to pgvector database: {_redact_error(exc)}") from exc
        try:
            index = cls(
                connection,
                schema=schema,
                table=table,
                id_column=id_column,
                vector_column=vector_column,
                text_column=text_column,
                dimension=dimension if dimension is not None else 1,
                metric=metric,
                dsn_env=dsn_env,
                hnsw_ef_search=hnsw_ef_search,
                ivfflat_probes=ivfflat_probes,
            )
            index._validate_table_contract(configured_dimension=dimension, allow_missing_text=allow_missing_text)
            return index
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
            raise

    @classmethod
    def from_config(cls, cfg: Any, documents: Mapping[str, Any] | None = None) -> PgVectorIndex:
        """Build an adapter from an ``EmbedFlowConfig``-like object."""

        index_cfg = cfg.index
        explicit = getattr(index_cfg, "url", None)
        path_value = getattr(index_cfg, "path", None)
        dsn = explicit if explicit else (path_value if isinstance(path_value, str) and "://" in path_value else None)
        text_column = getattr(index_cfg, "text_column", "content")
        if documents is None and text_column is None:
            raise ValueError("pgvector text_column is required when no separate document store is configured")
        return cls.connect(
            dsn=dsn,
            dsn_env=getattr(index_cfg, "dsn_env", None) or "EMBEDFLOW_PGVECTOR_DSN",
            schema=getattr(index_cfg, "schema", "public"),
            table=getattr(index_cfg, "table", "documents"),
            id_column=getattr(index_cfg, "id_column", "id"),
            vector_column=getattr(index_cfg, "vector_column", "embedding"),
            text_column=text_column,
            dimension=getattr(cfg.source, "dimension", None),
            metric=getattr(index_cfg, "metric", "cosine"),
            hnsw_ef_search=getattr(index_cfg, "hnsw_ef_search", None),
            ivfflat_probes=getattr(index_cfg, "ivfflat_probes", None),
            allow_missing_text=documents is not None,
        )

    @property
    def _qualified_table(self):
        psycopg = _psycopg()
        return psycopg.sql.SQL("{}.{}").format(psycopg.sql.Identifier(self.schema), psycopg.sql.Identifier(self.table))

    def _column_identifier(self, name: str):
        return _psycopg().sql.Identifier(name)

    def _commit(self) -> None:
        commit = getattr(self.connection, "commit", None)
        if not callable(commit):
            return
        try:
            commit()
        except Exception as exc:
            rollback = getattr(self.connection, "rollback", None)
            if callable(rollback):
                rollback()
            raise RuntimeError(f"pgvector transaction failed: {_redact_error(exc)}") from exc

    def _execute(self, statement: Any, params: tuple[Any, ...] | list[Any] = (), *, fetch: bool = True) -> list[tuple[Any, ...]]:
        """Execute one statement and close the cursor/transaction cleanly."""

        self._ensure_open()
        with self._lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(statement, params)
                rows = cursor.fetchall() if fetch else []
            except Exception as exc:
                rollback = getattr(self.connection, "rollback", None)
                if callable(rollback):
                    rollback()
                raise RuntimeError(f"pgvector query failed: {_redact_error(exc)}") from exc
            finally:
                close = getattr(cursor, "close", None)
                if callable(close):
                    close()
            self._commit()
            return list(rows)

    def _execute_with_settings(self, statement: Any, params: tuple[Any, ...], settings: list[tuple[str, int]]) -> list[tuple[Any, ...]]:
        """Run a retrieval query with transaction-local ANN settings."""

        self._ensure_open()
        with self._lock:
            cursor = self.connection.cursor()
            try:
                for name, value in settings:
                    # ``set_config(..., true)`` is parameterized and local to
                    # this transaction; unlike a bare SET LOCAL it works with
                    # values supplied by the caller without SQL interpolation.
                    cursor.execute("SELECT set_config(%s, %s, true)", (name, str(value)))
                cursor.execute(statement, params)
                rows = cursor.fetchall()
            except Exception as exc:
                rollback = getattr(self.connection, "rollback", None)
                if callable(rollback):
                    rollback()
                raise RuntimeError(f"pgvector query failed: {_redact_error(exc)}") from exc
            finally:
                close = getattr(cursor, "close", None)
                if callable(close):
                    close()
            self._commit()
            return list(rows)

    def _table_columns(self) -> dict[str, dict[str, Any]]:
        rows = self._execute(
            """SELECT column_name, data_type, udt_name, is_nullable
               FROM information_schema.columns
               WHERE table_schema = %s AND table_name = %s""",
            (self.schema, self.table),
        )
        return {
            str(row[0]): {"data_type": row[1], "udt_name": row[2], "is_nullable": row[3]}
            for row in rows
        }

    def _infer_dimension(self) -> int | None:
        psycopg = _psycopg()
        rows = self._execute(
            psycopg.sql.SQL(
                "SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) "
                "FROM pg_catalog.pg_attribute AS a "
                "JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
                "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s AND a.attnum > 0 AND NOT a.attisdropped"
            ),
            (self.schema, self.table, self.vector_column),
        )
        if rows:
            type_name = str(rows[0][0] or "")
            match = _VECTOR_DIMENSION.match(type_name)
            if match:
                return int(match.group(1))
        # An unbounded ``vector`` column can still reveal its dimension from a
        # non-null row.  This query is bounded and only runs at connect time.
        rows = self._execute(
            psycopg.sql.SQL("SELECT vector_dims({vector}) FROM {table} WHERE {vector} IS NOT NULL LIMIT 1").format(
                vector=self._column_identifier(self.vector_column), table=self._qualified_table
            ),
        )
        if rows and rows[0][0] is not None:
            return int(rows[0][0])
        return None

    def _validate_table_contract(self, configured_dimension: int | None, *, allow_missing_text: bool = False) -> None:
        columns = self._table_columns()
        if not columns:
            raise ValueError(f"pgvector table {self.schema}.{self.table} was not found")
        for field, name in (("id column", self.id_column), ("vector column", self.vector_column)):
            if name not in columns:
                raise ValueError(f"pgvector {field} {self.schema}.{self.table}.{name} was not found")
        if self.text_column is not None and self.text_column not in columns and not allow_missing_text:
            raise ValueError(f"pgvector text column {self.schema}.{self.table}.{self.text_column} was not found")
        if self.text_column is not None and self.text_column not in columns and allow_missing_text:
            self.text_column = None
        vector_info = columns[self.vector_column]
        if str(vector_info.get("udt_name", "")).lower() != "vector":
            raise ValueError(f"{self.schema}.{self.table}.{self.vector_column} is not a pgvector column")
        inferred = self._infer_dimension()
        if configured_dimension is not None:
            try:
                configured = int(configured_dimension)
                configured_exact = float(configured_dimension) == configured
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("source.dimension must be a positive integer") from exc
            if isinstance(configured_dimension, bool) or not configured_exact or configured < 1:
                raise ValueError("source.dimension must be a positive integer")
            if inferred is not None and configured != inferred:
                raise ValueError(
                    f"source encoder produces {configured} dimensions but "
                    f"{self.schema}.{self.table}.{self.vector_column} is vector({inferred})"
                )
            self.dimension = configured
        elif inferred is not None:
            self.dimension = inferred
        else:
            raise ValueError("could not infer pgvector dimension; set source.dimension")

    def health_check(self) -> dict[str, Any]:
        """Run a lightweight connection/table health check."""

        try:
            self._execute("SELECT 1")
            columns = self._table_columns()
            ok = self.id_column in columns and self.vector_column in columns
            return {"ok": ok, "backend": "pgvector", "schema": self.schema, "table": self.table}
        except Exception as exc:
            return {"ok": False, "backend": "pgvector", "schema": self.schema, "table": self.table,
                    "error": _redact_error(exc)}

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = validate_k(k)
        literal = _vector_literal(query_vector, self.dimension)
        psycopg = _psycopg()
        operator = _METRIC_OPERATORS[self.metric]
        vector = self._column_identifier(self.vector_column)
        statement = psycopg.sql.SQL(
            "SELECT {id}, ({vector} {operator} %s::vector) AS distance "
            "FROM {table} WHERE {vector} IS NOT NULL AND {id} IS NOT NULL "
            "ORDER BY {vector} {operator} %s::vector, {id} LIMIT %s"
        ).format(
            id=self._column_identifier(self.id_column),
            vector=vector,
            operator=psycopg.sql.SQL(operator),
            table=self._qualified_table,
        )
        settings: list[tuple[str, int]] = []
        if self.hnsw_ef_search is not None:
            settings.append(("hnsw.ef_search", self.hnsw_ef_search))
        if self.ivfflat_probes is not None:
            settings.append(("ivfflat.probes", self.ivfflat_probes))
        params = (literal, literal, k)
        rows = self._execute_with_settings(statement, params, settings) if settings else self._execute(statement, params)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        # PostgreSQL enforces LIMIT, but truncating defensively keeps the
        # VectorIndex contract intact for connection proxies/test doubles that
        # may return extra rows.
        for rank, row in enumerate(rows[:k]):
            if len(row) < 2 or row[0] is None:
                raise ValueError("pgvector retrieval returned a NULL document ID")
            document_id = str(row[0])
            if document_id in seen:
                raise ValueError(f"pgvector retrieval returned duplicate canonical document ID {document_id!r}")
            seen.add(document_id)
            distance = float(row[1])
            if not np.isfinite(distance):
                raise ValueError("pgvector retrieval returned a non-finite distance")
            score = 1.0 - distance if self.metric == "cosine" else -distance
            hits.append(SearchHit(document_id, float(score), rank))
        return hits

    def fetch_documents(self, ids: list[str]) -> dict[str, Any]:
        self._ensure_open()
        if self.text_column is None:
            raise RuntimeError("pgvector text_column is unset and no separate document store was configured")
        normalized = list(dict.fromkeys(str(value) for value in ids))
        if not normalized:
            return {}
        psycopg = _psycopg()
        statement = psycopg.sql.SQL(
            "SELECT {id}, {text} FROM {table} WHERE ({id})::text = ANY(%s::text[])"
        ).format(
            id=self._column_identifier(self.id_column),
            text=self._column_identifier(self.text_column),
            table=self._qualified_table,
        )
        rows = self._execute(statement, (normalized,))
        output: dict[str, str] = {}
        for row in rows:
            if row[0] is None:
                continue
            document_id = str(row[0])
            if document_id in output:
                raise ValueError(f"pgvector table contains duplicate canonical document ID {document_id!r}")
            if row[1] is not None:
                output[document_id] = str(row[1])
        return output

    def iter_ids(self) -> Iterator[str]:
        """Yield table IDs for explicit prewarm/export operations."""

        psycopg = _psycopg()
        statement = psycopg.sql.SQL("SELECT {id} FROM {table} ORDER BY {id}").format(
            id=self._column_identifier(self.id_column), table=self._qualified_table
        )
        seen: set[str] = set()
        for row in self._execute(statement):
            if row[0] is not None:
                document_id = str(row[0])
                if document_id in seen:
                    raise ValueError(f"pgvector table contains duplicate canonical document ID {document_id!r}")
                seen.add(document_id)
                yield document_id

    def size(self) -> int:
        self._ensure_open()
        if self._size_cache is None:
            # ``pg_class.reltuples`` is maintained by PostgreSQL's statistics
            # collector and avoids a full table scan for large production
            # tables. Newly-created/unanalysed tables report ``-1``; fall back
            # to an exact count in that case so small demos and empty tables
            # remain useful.
            rows = self._execute(
                """SELECT c.reltuples::bigint
                   FROM pg_catalog.pg_class AS c
                   JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                   WHERE n.nspname = %s AND c.relname = %s""",
                (self.schema, self.table),
            )
            estimate = int(rows[0][0]) if rows and rows[0][0] is not None else -1
            if estimate >= 0:
                self._size_cache = estimate
                self._size_is_approximate = True
            else:
                exact_rows = self._execute(
                    _psycopg().sql.SQL("SELECT count(*) FROM {table}").format(table=self._qualified_table),
                )
                self._size_cache = int(exact_rows[0][0]) if exact_rows else 0
        return self._size_cache

    def _load_ann_indexes(self) -> list[dict[str, str]]:
        if self._ann_indexes is not None:
            return self._ann_indexes
        rows = self._execute(
            """SELECT index_class.relname, access_method.amname
               FROM pg_catalog.pg_index AS index_info
               JOIN pg_catalog.pg_class AS table_class ON table_class.oid = index_info.indrelid
               JOIN pg_catalog.pg_namespace AS namespace_info ON namespace_info.oid = table_class.relnamespace
               JOIN pg_catalog.pg_class AS index_class ON index_class.oid = index_info.indexrelid
               JOIN pg_catalog.pg_am AS access_method ON access_method.oid = index_class.relam
               WHERE namespace_info.nspname = %s AND table_class.relname = %s
                 AND index_info.indisvalid AND access_method.amname IN ('hnsw', 'ivfflat')""",
            (self.schema, self.table),
        )
        self._ann_indexes = [{"name": str(row[0]), "type": str(row[1])} for row in rows]
        return self._ann_indexes

    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        if self._extension_version is None:
            rows = self._execute("SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'")
            self._extension_version = str(rows[0][0]) if rows and rows[0][0] is not None else "UNKNOWN"
        return {
            "backend": "pgvector",
            "schema": self.schema,
            "table": self.table,
            "id_column": self.id_column,
            "vector_column": self.vector_column,
            "text_column": self.text_column,
            "dimension": self.dimension,
            "metric": self.metric,
            "size": self.size(),
            "size_is_approximate": self._size_is_approximate,
            "extension_version": self._extension_version,
            "ann_indexes": self._load_ann_indexes(),
            "hnsw_ef_search": self.hnsw_ef_search,
            "ivfflat_probes": self.ivfflat_probes,
            "dsn_env": self.dsn_env,
        }

    def audit(self) -> dict[str, Any]:
        """Return read-only table/index checks for ``embedflow audit-index``."""

        checks: dict[str, Any] = {}
        health = self.health_check()
        checks["connection"] = health
        try:
            columns = self._table_columns() if health.get("ok") else {}
        except Exception as exc:
            columns = {}
            checks["columns"] = {"ok": False, "error": _redact_error(exc)}
        checks["table_exists"] = bool(columns)
        checks["id_column"] = self.id_column in columns
        checks["vector_column"] = self.vector_column in columns and str(columns.get(self.vector_column, {}).get("udt_name", "")).lower() == "vector"
        checks["text_column"] = self.text_column is None or self.text_column in columns
        if checks["vector_column"]:
            try:
                inferred = self._infer_dimension()
                checks["dimension"] = {
                    "configured": self.dimension,
                    "index": inferred,
                    "ok": inferred is None or int(inferred) == int(self.dimension),
                }
            except Exception as exc:
                checks["dimension"] = {"configured": self.dimension, "index": None, "ok": False, "error": _redact_error(exc)}
        else:
            checks["dimension"] = {"configured": self.dimension, "index": None, "ok": False}
        if checks["table_exists"] and checks["id_column"] and checks["vector_column"]:
            try:
                rows = self._execute(
                    _psycopg().sql.SQL(
                        "SELECT count(*) FILTER (WHERE {vector} IS NULL), "
                        "count(*) FILTER (WHERE {id} IS NULL), count(*)::bigint, "
                        "count(DISTINCT ({id})::text)::bigint FROM {table}"
                    ).format(vector=self._column_identifier(self.vector_column), id=self._column_identifier(self.id_column), table=self._qualified_table)
                )
                null_vectors, null_ids, total, distinct_ids = (int(rows[0][0]), int(rows[0][1]), int(rows[0][2]), int(rows[0][3])) if rows else (0, 0, 0, 0)
                checks["null_vectors"] = {"count": null_vectors, "ok": null_vectors == 0}
                checks["null_ids"] = {"count": null_ids, "ok": null_ids == 0}
                checks["duplicate_ids"] = {"count": total - null_ids - distinct_ids, "ok": total - null_ids == distinct_ids}
            except Exception as exc:
                checks["row_checks"] = {"ok": False, "error": _redact_error(exc)}
        try:
            checks["ann_indexes"] = {"indexes": self._load_ann_indexes(), "status": "KNOWN"}
        except Exception as exc:
            checks["ann_indexes"] = {"status": "UNKNOWN", "error": _redact_error(exc)}
        if checks["vector_column"]:
            try:
                probe = self._execute(
                    _psycopg().sql.SQL("SELECT 1 FROM {table} WHERE {vector} IS NOT NULL AND {id} IS NOT NULL LIMIT 1").format(
                        table=self._qualified_table, vector=self._column_identifier(self.vector_column), id=self._column_identifier(self.id_column)
                    )
                )
                checks["retrieval_probe"] = {"ok": bool(probe)}
            except Exception as exc:
                checks["retrieval_probe"] = {"ok": False, "error": _redact_error(exc)}
        else:
            checks["retrieval_probe"] = {"ok": False, "error": "vector column is unavailable"}
        ok = all(bool(value.get("ok", True)) if isinstance(value, dict) else bool(value) for value in checks.values())
        return {"backend": "pgvector", "schema": self.schema, "table": self.table, "ok": ok, "checks": checks}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_connection:
            close = getattr(self.connection, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Teardown should remain idempotent even when PostgreSQL
                    # has already dropped the socket.
                    pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pgvector index is closed")


class _PgVectorDocumentMapping(Mapping[str, str]):
    def __init__(self, store: PgVectorDocumentStore):
        self.store = store

    def __getitem__(self, key: str) -> str:
        values = self.store.get([str(key)])
        if str(key) not in values:
            raise KeyError(str(key))
        return values[str(key)]

    def __iter__(self) -> Iterator[str]:
        return self.store.index.iter_ids()

    def __len__(self) -> int:
        return self.store.size()


class PgVectorDocumentStore:
    """DocumentStore-compatible lazy resolver backed by pgvector text rows."""

    def __init__(self, index: PgVectorIndex, *, id_field: str = "id", text_field: str = "content", owns_index: bool = True):
        self.index = index
        self.id_field = id_field
        self.text_field = text_field
        self.path = None
        self.documents: Mapping[str, str] = _PgVectorDocumentMapping(self)
        self._owns_index = owns_index

    def get(self, document_ids: Any) -> dict[str, str]:
        ids = [str(value) for value in document_ids]
        values = self.index.fetch_documents(ids)
        missing = [value for value in ids if value not in values]
        if missing:
            raise KeyError(f"document text missing for IDs: {missing[:5]}")
        return {value: values[value] for value in ids}

    def size(self) -> int:
        return self.index.size()

    def close(self) -> None:
        if self._owns_index:
            self.index.close()


__all__ = ["PgVectorIndex", "PgVectorDocumentStore", "normalize_pgvector_metric"]
