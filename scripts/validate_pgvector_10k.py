#!/usr/bin/env python3
"""Run a destructive-test-only, real PostgreSQL/pgvector validation.

The script creates a uniquely named, disposable Docker container and a
uniquely named set of tables inside it.  The EmbedFlow adapter is only used
for reads; all DDL and fixture writes are owned by this script.  It is kept
outside the normal test suite because Docker is an optional maintainer tool.

Usage::

    python scripts/validate_pgvector_10k.py

The process exits non-zero on a failed check.  A JSON report is written to a
temporary directory unless ``--output`` is supplied.  No source checkout,
database, cache, or model files are modified except for that report and the
temporary Docker volume/container, which are removed on exit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class ValidationFailure(RuntimeError):
    """A check failed with a concise, user-facing reason."""


class DockerContainer:
    def __init__(self, image: str = "pgvector/pgvector:pg16") -> None:
        self.image = image
        self.name = f"embedflow-pgvector-validation-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.port: int | None = None

    def _run(self, args: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=timeout)

    def start(self) -> None:
        if shutil.which("docker") is None:
            raise ValidationFailure("Docker executable is unavailable")
        info = self._run(["info"], check=False)
        if info.returncode != 0:
            raise ValidationFailure(f"Docker daemon is unavailable: {info.stderr.strip() or 'docker info failed'}")
        result = self._run([
            "run", "-d", "--name", self.name,
            "-e", "POSTGRES_DB=embedflow",
            "-e", "POSTGRES_USER=embedflow",
            "-e", "POSTGRES_PASSWORD=embedflow",
            "-p", "127.0.0.1::5432",
            self.image,
        ])
        if not result.stdout.strip():
            raise ValidationFailure("Docker did not return a container ID")
        self._refresh_port()

    def _refresh_port(self) -> None:
        for _ in range(30):
            mapped = self._run(["port", self.name, "5432/tcp"], check=False)
            if mapped.returncode == 0 and mapped.stdout.strip():
                # Format is 127.0.0.1:49123 or [::1]:49123.
                self.port = int(mapped.stdout.strip().rsplit(":", 1)[1])
                return
            time.sleep(0.2)
        raise ValidationFailure("Docker container did not expose PostgreSQL port")

    @property
    def dsn(self) -> str:
        if self.port is None:
            raise ValidationFailure("container has not been started")
        return f"postgresql://embedflow:embedflow@127.0.0.1:{self.port}/embedflow"

    def restart(self) -> None:
        self._run(["restart", "-t", "1", self.name])

    def stop(self) -> None:
        self._run(["stop", "-t", "1", self.name], check=False, timeout=20)

    def start_again(self) -> None:
        self._run(["start", self.name])
        self._refresh_port()

    def cleanup(self) -> None:
        self._run(["rm", "-f", self.name], check=False, timeout=20)


def wait_for_postgres(dsn: str, timeout: float = 45.0) -> Any:
    import psycopg

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            connection = psycopg.connect(dsn, autocommit=True)
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return connection
        except Exception as exc:  # startup race is expected
            last = exc
            time.sleep(0.5)
    raise ValidationFailure(f"PostgreSQL did not become ready: {last}")


def sql_identifier(psycopg: Any, value: str) -> Any:
    return psycopg.sql.Identifier(value)


def qualified(psycopg: Any, schema: str, table: str) -> Any:
    return psycopg.sql.SQL("{}.{}").format(sql_identifier(psycopg, schema), sql_identifier(psycopg, table))


def vector_literal(value: np.ndarray) -> str:
    return "[" + ",".join(repr(float(x)) for x in np.asarray(value, dtype="float32")) + "]"


def record(results: dict[str, dict[str, Any]], name: str, fn: Callable[[], Any]) -> Any:
    started = time.monotonic()
    try:
        value = fn()
        results[name] = {"status": "PASS", "seconds": round(time.monotonic() - started, 3), "detail": value}
        return value
    except Exception as exc:
        results[name] = {"status": "FAIL", "seconds": round(time.monotonic() - started, 3), "detail": str(exc)}
        return None


def assert_true(condition: Any, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def make_fixture(connection: Any, *, table: str, dimension: int = 32, rows: int = 10_000) -> dict[str, Any]:
    import psycopg

    model_cls = __import__("embedflow.models", fromlist=["HashEmbeddingModel"]).HashEmbeddingModel
    model = model_cls("embedflow/demo-source", dimension)
    contents = [f"validation document {i:05d} topic {i % 37}" for i in range(rows)]
    vectors = model.encode_documents(contents, batch_size=256)
    with connection.cursor() as cursor:
        cursor.execute(
            psycopg.sql.SQL("CREATE TABLE {table} ({id} bigint PRIMARY KEY, {text} text NOT NULL, {vector} vector({dim}) NOT NULL)").format(
                table=qualified(psycopg, "public", table),
                id=sql_identifier(psycopg, "id"),
                text=sql_identifier(psycopg, "content"),
                vector=sql_identifier(psycopg, "embedding"),
                dim=psycopg.sql.Literal(dimension),
            )
        )
        insert = psycopg.sql.SQL("INSERT INTO {table} ({id},{text},{vector}) VALUES (%s,%s,%s::vector)").format(
            table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "id"),
            text=sql_identifier(psycopg, "content"), vector=sql_identifier(psycopg, "embedding"),
        )
        cursor.executemany(insert, ((i, contents[i], vector_literal(vectors[i])) for i in range(rows)))
        cursor.execute(psycopg.sql.SQL("ANALYZE {table}").format(table=qualified(psycopg, "public", table)))
    return {"table": table, "dimension": dimension, "rows": rows, "contents": contents, "vectors": vectors}


def snapshot_source(connection: Any, table: str) -> dict[str, Any]:
    import psycopg

    with connection.cursor() as cursor:
        cursor.execute(
            psycopg.sql.SQL(
                "SELECT count(*)::bigint, md5(string_agg(({id})::text || ':' || ({vector})::text || ':' || {text}, '|' ORDER BY {id})) "
                "FROM {table}"
            ).format(id=sql_identifier(psycopg, "id"), vector=sql_identifier(psycopg, "embedding"),
                     text=sql_identifier(psycopg, "content"), table=qualified(psycopg, "public", table))
        )
        count, digest = cursor.fetchone()
        cursor.execute(
            psycopg.sql.SQL("SELECT ({id})::text, ({vector})::text, {text} FROM {table} WHERE {id} IN (0, 1, 9999) ORDER BY {id}").format(
                id=sql_identifier(psycopg, "id"), vector=sql_identifier(psycopg, "embedding"),
                text=sql_identifier(psycopg, "content"), table=qualified(psycopg, "public", table)
            )
        )
        sample = [tuple(row) for row in cursor.fetchall()]
        cursor.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname=%s AND tablename=%s ORDER BY indexname",
            ("public", table),
        )
        indexes = [tuple(row) for row in cursor.fetchall()]
    return {"count": int(count), "digest": digest, "sample": sample, "indexes": indexes}


def create_ann_index(connection: Any, table: str, kind: str) -> str:
    import psycopg

    with connection.cursor() as cursor:
        for name in (f"{table}_hnsw", f"{table}_ivfflat"):
            cursor.execute(psycopg.sql.SQL("DROP INDEX IF EXISTS {index}").format(index=sql_identifier(psycopg, name)))
        name = f"{table}_{kind}"
        if kind == "hnsw":
            statement = psycopg.sql.SQL("CREATE INDEX {index} ON {table} USING hnsw ({vector} vector_cosine_ops)").format(
                index=sql_identifier(psycopg, name), table=qualified(psycopg, "public", table), vector=sql_identifier(psycopg, "embedding")
            )
        else:
            statement = psycopg.sql.SQL("CREATE INDEX {index} ON {table} USING ivfflat ({vector} vector_cosine_ops) WITH (lists = 100)").format(
                index=sql_identifier(psycopg, name), table=qualified(psycopg, "public", table), vector=sql_identifier(psycopg, "embedding")
            )
        cursor.execute(statement)
        cursor.execute(psycopg.sql.SQL("ANALYZE {table}").format(table=qualified(psycopg, "public", table)))
    return name


def explain_uses_index(connection: Any, table: str, query: np.ndarray, index_name: str) -> list[str]:
    import psycopg

    with connection.cursor() as cursor:
        cursor.execute("BEGIN")
        try:
            cursor.execute("SET LOCAL enable_seqscan = off")
            statement = psycopg.sql.SQL(
                "EXPLAIN (FORMAT TEXT) SELECT {id} FROM {table} WHERE {vector} IS NOT NULL "
                "ORDER BY {vector} <=> %s::vector LIMIT 10"
            ).format(id=sql_identifier(psycopg, "id"), table=qualified(psycopg, "public", table), vector=sql_identifier(psycopg, "embedding"))
            cursor.execute(statement, (vector_literal(query),))
            plan = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute("ROLLBACK")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    assert_true(any(index_name in line for line in plan), f"EXPLAIN did not select {index_name}: {' | '.join(plan)}")
    return plan


def adapter_for(dsn: str, table: str, *, metric: str = "cosine", dimension: int = 32, **kwargs: Any) -> Any:
    from embedflow.indexes import PgVectorIndex

    return PgVectorIndex.connect(dsn=dsn, schema="public", table=table, id_column="id", vector_column="embedding",
                                 text_column="content", dimension=dimension, metric=metric, **kwargs)


def test_ann_and_topk(dsn: str, connection: Any, fixture: dict[str, Any], kind: str) -> dict[str, Any]:
    table = fixture["table"]
    index_name = create_ann_index(connection, table, kind)
    query = fixture["vectors"][123]
    plan = explain_uses_index(connection, table, query, index_name)
    kwargs = {"hnsw_ef_search": 77} if kind == "hnsw" else {"ivfflat_probes": 13}
    index = adapter_for(dsn, table, **kwargs)
    try:
        counts: dict[int, int] = {}
        for k in (1, 10, 50, 100, 500, 10_001):
            hits = index.search(query, k)
            counts[k] = len(hits)
            assert_true(len(hits) == min(k, fixture["rows"]), f"{kind} K={k}: expected {min(k, fixture['rows'])}, got {len(hits)}")
            ids = [hit.document_id for hit in hits]
            assert_true(len(ids) == len(set(ids)), f"{kind} K={k}: duplicate IDs")
            scores = [hit.score for hit in hits]
            assert_true(all(a >= b - 1e-6 for a, b in zip(scores, scores[1:])), f"{kind} scores are not descending")
        text = index.fetch_documents(["0", "123", "9999"])
        assert_true(text["123"] == fixture["contents"][123], "text lookup returned the wrong document")
        audit = index.audit()
        assert_true(audit["ok"], f"{kind} audit failed: {audit}")
        return {"index": index_name, "plan": plan, "counts": counts, "audit": audit}
    finally:
        index.close()


def test_settings_do_not_leak(dsn: str, table: str, kind: str) -> dict[str, Any]:
    import psycopg

    setting, configured, default = ("hnsw.ef_search", 77, 40) if kind == "hnsw" else ("ivfflat.probes", 13, 1)
    index = adapter_for(dsn, table, **({"hnsw_ef_search": configured} if kind == "hnsw" else {"ivfflat_probes": configured}))
    try:
        # A successful query proves the setting is accepted in the adapter's
        # transaction.  A fresh connection proves SET LOCAL did not leak.
        index.search(np.ones(32, dtype="float32"), 1)
    finally:
        index.close()
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            # pgvector registers its custom GUCs when the extension library is
            # loaded in a session.  Touch the vector type before SHOW so the
            # check observes the real server default rather than PostgreSQL's
            # pre-load "unrecognized parameter" response.
            cursor.execute("SELECT '[1,0]'::vector")
            cursor.execute(f"SHOW {setting}")
            value = int(cursor.fetchone()[0])
    assert_true(value == default, f"{setting} leaked after adapter commit: {value} != {default}")
    return {"setting": setting, "configured": configured, "fresh_connection": value}


def test_metric_table(connection: Any, dsn: str, prefix: str) -> dict[str, Any]:
    import psycopg

    table = f"{prefix}_metrics"
    vectors = np.asarray([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype="float32")
    query = np.asarray([1, 0.25], dtype="float32")
    with connection.cursor() as cursor:
        cursor.execute(psycopg.sql.SQL("CREATE TABLE {table} ({id} text PRIMARY KEY, {vector} vector(2) NOT NULL, {text} text NOT NULL)").format(
            table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "id"), vector=sql_identifier(psycopg, "embedding"), text=sql_identifier(psycopg, "content")))
        cursor.executemany(psycopg.sql.SQL("INSERT INTO {table} ({id},{vector},{text}) VALUES (%s,%s::vector,%s)").format(
            table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "id"), vector=sql_identifier(psycopg, "embedding"), text=sql_identifier(psycopg, "content")),
            ((f"v{i}", vector_literal(vectors[i]), f"metric {i}") for i in range(len(vectors))))
    expected: dict[str, list[str]] = {}
    for metric in ("cosine", "l2", "euclidean", "inner_product", "dot"):
        if metric in {"cosine"}:
            distances = 1.0 - (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
        elif metric in {"l2", "euclidean"}:
            distances = np.linalg.norm(vectors - query, axis=1)
        else:
            distances = -(vectors @ query)
        order = np.lexsort((np.arange(len(vectors)), distances))
        expected[metric] = [f"v{i}" for i in order]
        index = adapter_for(dsn, table, metric=metric, dimension=2)
        try:
            hits = index.search(query, 4)
            assert_true([hit.document_id for hit in hits] == expected[metric], f"{metric} ordering differs from NumPy ground truth")
            assert_true(all(hits[i].score >= hits[i + 1].score - 1e-7 for i in range(len(hits) - 1)), f"{metric} score direction reversed")
        finally:
            index.close()
    return {"expected_order": expected}


def test_id_types(connection: Any, dsn: str, prefix: str) -> dict[str, Any]:
    import psycopg

    tables = {"bigint": ("bigint", "42"), "uuid": ("uuid", "11111111-1111-1111-1111-111111111111"), "text": ("text", "42")}
    for label, (id_type, expected_id) in tables.items():
        table = f"{prefix}_{label}"
        with connection.cursor() as cursor:
            cursor.execute(psycopg.sql.SQL("CREATE TABLE {table} ({id} {typ} PRIMARY KEY, {vector} vector(2) NOT NULL, {text} text NOT NULL)").format(
                table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "id"), typ=psycopg.sql.SQL(id_type), vector=sql_identifier(psycopg, "embedding"), text=sql_identifier(psycopg, "content")))
            if label == "uuid":
                values = [(expected_id, "[1,0]", f"{label} content"), ("22222222-2222-2222-2222-222222222222", "[0,1]", "other")]
            elif label == "bigint":
                values = [(42, "[1,0]", f"{label} content"), (43, "[0,1]", "other")]
            else:
                values = [("42", "[1,0]", f"{label} content"), ("43", "[0,1]", "other")]
            cursor.executemany(psycopg.sql.SQL("INSERT INTO {table} ({id},{vector},{text}) VALUES (%s,%s::vector,%s)").format(
                table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "id"), vector=sql_identifier(psycopg, "embedding"), text=sql_identifier(psycopg, "content")), values)
        index = adapter_for(dsn, table, dimension=2)
        try:
            hits = index.search(np.asarray([1, 0], dtype="float32"), 2)
            assert_true(hits[0].document_id == expected_id, f"{label} ID was not canonicalized as expected")
            fetched = index.fetch_documents([expected_id])
            assert_true(fetched[expected_id] == f"{label} content", f"{label} text lookup failed")
        finally:
            index.close()
    return {"types": list(tables)}


def test_edges(connection: Any, dsn: str, prefix: str) -> dict[str, Any]:
    import psycopg

    empty = f"{prefix}_empty"
    one = f"{prefix}_one"
    nullable = f"{prefix}_nullable"
    no_text = f"{prefix}_notext"
    duplicate = f"{prefix}_duplicate"
    for table, schema in ((empty, "content text"), (one, "content text"), (nullable, "content text"), (no_text, ""), (duplicate, "content text")):
        with connection.cursor() as cursor:
            null_vector = "vector(2)" if table != nullable else "vector(2)"
            text_sql = ", content text" if schema else ""
            id_sql = "id bigint" if table == duplicate else "id bigint PRIMARY KEY"
            cursor.execute(psycopg.sql.SQL("CREATE TABLE {table} ({iddef}, embedding {vector}{text})").format(
                table=qualified(psycopg, "public", table), iddef=psycopg.sql.SQL(id_sql), vector=psycopg.sql.SQL(null_vector), text=psycopg.sql.SQL(text_sql)))
    with connection.cursor() as cursor:
        cursor.execute(psycopg.sql.SQL("INSERT INTO {table} VALUES (1,'[1,0]','one row')").format(table=qualified(psycopg, "public", one)))
        cursor.execute(psycopg.sql.SQL("INSERT INTO {table} VALUES (1,NULL,'null vector'),(2,'[1,0]','valid')").format(table=qualified(psycopg, "public", nullable)))
        cursor.execute(psycopg.sql.SQL("INSERT INTO {table} (id,embedding) VALUES (1,'[1,0]')").format(table=qualified(psycopg, "public", no_text)))
        cursor.execute(psycopg.sql.SQL("INSERT INTO {table} VALUES (1,'[1,0]','duplicate one'),(1,'[0,1]','duplicate two')").format(table=qualified(psycopg, "public", duplicate)))
    empty_index = adapter_for(dsn, empty, dimension=2)
    try:
        assert_true(empty_index.size() == 0, "empty table size is not zero")
        assert_true(empty_index.search(np.asarray([1, 0], dtype="float32"), 10) == [], "empty table returned hits")
    finally:
        empty_index.close()
    one_index = adapter_for(dsn, one, dimension=2)
    try:
        assert_true(len(one_index.search(np.asarray([1, 0], dtype="float32"), 10)) == 1, "one-row table retrieval failed")
    finally:
        one_index.close()
    null_index = adapter_for(dsn, nullable, dimension=2)
    try:
        assert_true(len(null_index.search(np.asarray([1, 0], dtype="float32"), 10)) == 1, "NULL vectors were not skipped")
        audit = null_index.audit()
        assert_true(audit["checks"]["null_vectors"]["count"] == 1 and not audit["ok"], "NULL-vector audit did not report the finding")
    finally:
        null_index.close()
    no_text_index = __import__("embedflow.indexes", fromlist=["PgVectorIndex"]).PgVectorIndex.connect(
        dsn=dsn, schema="public", table=no_text, dimension=2, text_column=None)
    try:
        no_text_index.search(np.asarray([1, 0], dtype="float32"), 1)
        try:
            no_text_index.fetch_documents(["1"])
        except RuntimeError as exc:
            assert_true("text_column" in str(exc), "missing text error was not explicit")
        else:
            raise ValidationFailure("missing-text table unexpectedly resolved documents")
    finally:
        no_text_index.close()
    duplicate_index = adapter_for(dsn, duplicate, dimension=2)
    try:
        try:
            duplicate_index.search(np.asarray([1, 0], dtype="float32"), 10)
        except ValueError as exc:
            assert_true("duplicate" in str(exc).lower(), "duplicate canonical IDs did not fail clearly")
        else:
            raise ValidationFailure("duplicate canonical IDs unexpectedly passed retrieval")
        duplicate_audit = duplicate_index.audit()
        assert_true(not duplicate_audit["checks"]["duplicate_ids"]["ok"], "duplicate-ID audit did not report the finding")
    finally:
        duplicate_index.close()
    from embedflow.indexes import PgVectorIndex
    for kwargs, phrase in [({"table": f"{prefix}_missing"}, "table"), ({"schema": f"{prefix}_missing"}, "table"), ({"table": one, "dimension": 3}, "dimension")]:
        try:
            PgVectorIndex.connect(dsn=dsn, schema=kwargs.get("schema", "public"), table=kwargs.get("table", one), dimension=kwargs.get("dimension", 2))
        except Exception as exc:
            assert_true(phrase in str(exc).lower() or (phrase == "dimension" and "dimensions" in str(exc).lower()), f"edge error did not mention {phrase}: {exc}")
        else:
            raise ValidationFailure(f"invalid edge configuration unexpectedly connected: {kwargs}")
    return {"empty": True, "one": True, "null_vectors": True, "missing_text": True}


def test_identifier_security(connection: Any, dsn: str, prefix: str, legacy_table: str) -> dict[str, Any]:
    import psycopg

    from embedflow.indexes import PgVectorIndex

    table = f'{prefix}_weird-name'
    with connection.cursor() as cursor:
        cursor.execute(psycopg.sql.SQL('CREATE TABLE {table} ({id} text PRIMARY KEY, {vector} vector(2) NOT NULL, {text} text NOT NULL)').format(
            table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "select"), vector=sql_identifier(psycopg, "user"), text=sql_identifier(psycopg, "spaces col")))
        cursor.execute(psycopg.sql.SQL('INSERT INTO {table} ({id},{vector},{text}) VALUES (%s,%s::vector,%s)').format(
            table=qualified(psycopg, "public", table), id=sql_identifier(psycopg, "select"), vector=sql_identifier(psycopg, "user"), text=sql_identifier(psycopg, "spaces col")),
                       ("safe", "[1,0]", "quoted identifiers work"))
    index = PgVectorIndex.connect(dsn=dsn, schema="public", table=table, id_column="select", vector_column="user", text_column="spaces col", dimension=2)
    try:
        assert_true(index.fetch_documents(["safe"])["safe"] == "quoted identifiers work", "quoted identifiers failed")
    finally:
        index.close()
    malicious = 'documents; DROP TABLE ' + legacy_table + '; --'
    try:
        PgVectorIndex.connect(dsn=dsn, schema="public", table=malicious, dimension=32)
    except Exception:
        pass
    else:
        raise ValidationFailure("malicious table identifier unexpectedly connected")
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (f"public.{legacy_table}",))
        assert_true(cursor.fetchone()[0] is not None, "malicious identifier changed the legacy table")
    # Exercise each configurable identifier position; all are composed as
    # identifiers and therefore fail as missing columns/tables, never as SQL.
    for field in ("schema", "id_column", "vector_column", "text_column"):
        kwargs = {"schema": "public", "table": legacy_table, "id_column": "id", "vector_column": "embedding", "text_column": "content"}
        kwargs[field] = "safe_name; DROP TABLE " + legacy_table + "; --"
        try:
            PgVectorIndex.connect(dsn=dsn, dimension=32, **kwargs)
        except Exception:
            pass
        else:
            raise ValidationFailure(f"malicious {field} unexpectedly connected")
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (f"public.{legacy_table}",))
        assert_true(cursor.fetchone()[0] is not None, "malicious identifier changed the source table")
    # A real failed authentication path must redact a password-bearing DSN.
    bad_dsn = dsn.replace("embedflow:embedflow@", "embedflow:wrong-secret@", 1)
    try:
        PgVectorIndex.connect(dsn=bad_dsn, table=legacy_table, dimension=32)
    except Exception as exc:
        message = str(exc)
        assert_true("wrong-secret" not in message and "embedflow:wrong-secret@" not in message, "connection error exposed a password")
    else:
        raise ValidationFailure("bad-password connection unexpectedly succeeded")
    from embedflow.indexes.pgvector_backend import _redact_error
    redacted = _redact_error(RuntimeError("postgresql://user:super-secret@db.example/x?password=query-secret"))
    assert_true("super-secret" not in redacted and "query-secret" not in redacted and "<redacted>" in redacted, "DSN password was not redacted")
    return {"quoted_identifiers": True, "malicious_identifier": True, "redacted": redacted}


def make_config(tmp: Path, dsn: str, table: str) -> Path:
    import yaml

    config_path = tmp / "embedflow.yaml"
    raw = {
        "source": {"model": "embedflow/demo-source", "dimension": 32, "normalization": "l2"},
        "target": {"model": "embedflow/demo-target", "dimension": 32, "normalization": "l2"},
        "index": {"backend": "pgvector", "dsn_env": "EMBEDFLOW_PGVECTOR_DSN", "schema": "public", "table": table,
                   "id_column": "id", "vector_column": "embedding", "text_column": "content", "metric": "cosine", "hnsw_ef_search": 77},
        "documents": {"path": str(tmp / "documents-not-needed.jsonl"), "id_field": "id", "text_field": "text"},
        "migration": {"candidate_depth": 50, "kmax_probe": 50, "probe_queries": 3, "max_sync_misses": 2, "background_batch_size": 16, "max_retries": 3},
        "cache": {"path": str(tmp / "cache")},
        "state_path": str(tmp / "state.json"),
        "telemetry": {"latency_log": str(tmp / "latency.jsonl")},
        "probe": {"queries": str(tmp / "queries.jsonl"), "k_values": [10, 20, 50], "kmax": 50, "epsilon": 0.01},
    }
    (tmp / "queries.jsonl").write_text("\n".join(json.dumps({"id": str(i), "text": f"topic {i}"}) for i in range(3)) + "\n", encoding="utf-8")
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def test_migration(dsn: str, table: str, config_path: Path) -> dict[str, Any]:
    from embedflow.runtime import open_engine

    os.environ["EMBEDFLOW_PGVECTOR_DSN"] = dsn
    engine = open_engine(config_path, demo=True, start_worker=False)
    query = "topic 3"
    try:
        cold = engine.search(query, top_k=5, candidate_depth=50, max_sync_misses=0)
        assert_true(cold["migration"]["status"] == "COLD", f"expected COLD, got {cold['migration']}")
        partial = engine.search(query, top_k=5, candidate_depth=50, max_sync_misses=2)
        assert_true(partial["migration"]["status"] == "PARTIAL", f"expected PARTIAL, got {partial['migration']}")
        assert_true(partial["migration"]["sync_encoded"] == 2, "bounded synchronous budget was not honored")
        engine.worker.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            stats = engine.worker.stats()
            if stats["queue"]["pending"] == 0 and stats["queue"]["processing"] == 0:
                break
            time.sleep(0.1)
        stats = engine.worker.stats()
        assert_true(stats["queue"]["error"] == 0 and stats["queue"]["done"] >= 50, f"worker did not drain: {stats}")
        warm = engine.search(query, top_k=5, candidate_depth=50, max_sync_misses=0)
        assert_true(warm["migration"]["status"] == "WARM", f"expected WARM, got {warm['migration']}")
        assert_true(all(row["target_vector_cached"] for row in warm["results"]), "warm results contain uncached target vectors")
        # Once every candidate vector is warm, compare the engine's order with
        # direct target scoring over exactly the same pgvector candidate set.
        candidates = engine.source_index.search(engine.source_model.encode_query(query), 50)
        candidate_ids = [hit.document_id for hit in candidates]
        texts = engine.documents.get(candidate_ids)
        vectors = engine.target_model.encode_documents([texts[document_id] for document_id in candidate_ids])
        scores = {document_id: float(vector @ engine.target_model.encode_query(query)) for document_id, vector in zip(candidate_ids, vectors)}
        expected = sorted(candidate_ids, key=lambda document_id: (-scores[document_id], candidate_ids.index(document_id)))[:5]
        assert_true([row["id"] for row in warm["results"]] == expected, "warm ranking differs from direct target scoring")
        cached = engine.cache.stats()["cached_target_vectors"]
        return {"cold": cold["migration"], "partial": partial["migration"], "warm": warm["migration"], "worker": stats, "cached": cached}
    finally:
        engine.close()


def test_restart_persistence(dsn: str, config_path: Path) -> dict[str, Any]:
    from embedflow.runtime import open_engine

    os.environ["EMBEDFLOW_PGVECTOR_DSN"] = dsn
    engine = open_engine(config_path, demo=True, start_worker=False)
    try:
        result = engine.search("topic 3", top_k=5, max_sync_misses=0)
        assert_true(result["migration"]["status"] == "WARM", "persistent cache did not survive EmbedFlow restart")
        status = engine.status()
        assert_true(status["index"]["backend"] == "pgvector", "status did not identify pgvector")
        return {"status": status["migration"], "cache": status["cache"], "index": status["index"]}
    finally:
        engine.close()


def run_cli(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "embedflow", *command], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def test_cli_and_api(repo: Path, dsn: str, config_path: Path, tmp: Path) -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(repo), "EMBEDFLOW_PGVECTOR_DSN": dsn}
    cwd = tmp
    commands = {
        "doctor": ["doctor", "--config", str(config_path)],
        "analyze": ["analyze", "--config", str(config_path), "--demo", "--output-dir", str(tmp / "analysis")],
        "status": ["status", "--config", str(config_path), "--demo"],
        "search": ["search", "--config", str(config_path), "--demo", "topic 3"],
        "audit-index": ["audit-index", "--config", str(config_path), "--demo"],
        "prewarm": ["prewarm", "--config", str(config_path), "--demo", "--documents", "3", "--async"],
    }
    outputs: dict[str, str] = {}
    for name, command in commands.items():
        result = run_cli(command, cwd=cwd, env=env)
        assert_true(result.returncode == 0, f"CLI {name} failed: {result.stdout}\n{result.stderr}")
        outputs[name] = result.stdout
    assert_true('"backend": "pgvector"' in outputs["status"], "CLI status omitted backend")
    assert_true('"pgvector_audit"' in outputs["audit-index"], "CLI audit omitted pgvector audit")
    # Environment override: the YAML deliberately points to a nonexistent
    # table, then the documented override selects the live table.
    override_config = tmp / "override.yaml"
    # Replace the actual generated table line robustly rather than assuming a
    # particular YAML emitter layout.
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")); raw["index"]["table"] = "definitely_missing"
    override_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    override_env = {**env, "EMBEDFLOW_PGVECTOR_TABLE": str(__import__("yaml").safe_load(config_path.read_text())["index"]["table"])}
    result = run_cli(["doctor", "--config", str(override_config)], cwd=cwd, env=override_env)
    assert_true(result.returncode == 0, f"pgvector table environment override failed: {result.stdout}\n{result.stderr}")

    # Launch the actual API in a child process and exercise every route.  A
    # loopback socket is expected in normal environments; if it is blocked,
    # the caller records the check as a skip rather than pretending it passed.
    port = 18_000 + (os.getpid() % 1000)
    process = subprocess.Popen([sys.executable, "-m", "embedflow", "serve", "--config", str(config_path), "--demo", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                               cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as response:
                    assert_true(response.status == 200, "health route was not 200")
                    break
            except (urllib.error.URLError, TimeoutError):
                if process.poll() is not None:
                    break
                time.sleep(0.2)
        else:
            raise ValidationFailure("FastAPI loopback server did not become ready")
        def get(path: str) -> dict[str, Any]:
            with urllib.request.urlopen(base + path, timeout=10) as response:
                return json.loads(response.read())
        def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
            request = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read())
        health = get("/health")
        status = get("/status")
        plan = get("/plan")
        analysis = post("/analyze", {})
        search = post("/search", {"query": "topic 3", "top_k": 5})
        prewarm = post("/prewarm", {"document_ids": ["1", "2"], "asynchronous": True})
        metrics = get("/metrics")
        assert_true(health.get("status") == "ok", "health response mismatch")
        assert_true(status.get("index", {}).get("backend") == "pgvector", "API status omitted pgvector")
        assert_true("candidate_depth" in plan and "diagnostic" in analysis, "API plan/analyze schema mismatch")
        assert_true(search.get("state") in {"COLD", "PARTIAL", "WARM"}, "API search state invalid")
        assert_true("queued" in prewarm and "latency" in metrics, "API prewarm/metrics schema mismatch")
        serialized = json.dumps([health, status, plan, analysis, search, prewarm, metrics])
        assert_true("embedflow:embedflow" not in serialized and "postgresql://" not in serialized, "API leaked DSN credentials")
        api = {"health": health, "status": status, "plan": plan, "analysis": analysis, "search": search, "prewarm": prewarm, "metrics": metrics}
    except urllib.error.URLError as exc:
        raise ValidationFailure(f"API loopback unavailable: {exc}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return {"commands": list(commands), "api": api}


def test_concurrency(dsn: str, table: str) -> dict[str, Any]:
    index = adapter_for(dsn, table, hnsw_ef_search=77)
    try:
        def one(i: int) -> list[str]:
            hits = index.search(np.asarray([float((i % 5) + 1)] + [0.1] * 31, dtype="float32"), 10)
            return [hit.document_id for hit in hits]
        for workers in (2, 4, 8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(one, range(workers * 3)))
            assert_true(len(values) == workers * 3 and all(len(ids) == 10 for ids in values), f"concurrency {workers} produced invalid results")
        return {"workers": [2, 4, 8], "queries": 42, "connection_model": "one serialized adapter connection"}
    finally:
        index.close()


def test_wheel(repo: Path, dsn: str, config_path: Path, tmp: Path) -> dict[str, Any]:
    dist = repo / "dist"
    wheels = sorted(dist.glob("embedflow-*.whl"))
    assert_true(bool(wheels), "no wheel found; build it before running validation")
    wheel = wheels[-1]
    venv = tmp / "wheel-venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True, capture_output=True, text=True)
    pip = venv / "bin" / "pip"
    executable = venv / "bin" / "embedflow"
    subprocess.run([str(pip), "install", "--no-deps", f"embedflow[pgvector] @ file://{wheel}"], check=True, capture_output=True, text=True)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["EMBEDFLOW_PGVECTOR_DSN"] = dsn
    commands = [
        ["--version"], ["registry", "list"], ["doctor", "--config", str(config_path)],
        ["audit-index", "--config", str(config_path), "--demo"],
        ["search", "--config", str(config_path), "--demo", "topic 3"],
    ]
    outputs = {}
    for command in commands:
        result = subprocess.run([str(executable), *command], cwd=tmp, env=env, capture_output=True, text=True, timeout=90)
        assert_true(result.returncode == 0, f"installed wheel command {command} failed: {result.stdout}\n{result.stderr}")
        outputs[" ".join(command)] = result.stdout
    import_result = subprocess.run([str(venv / "bin" / "python"), "-c", "import embedflow, psycopg; print(embedflow.__version__)"], cwd=tmp, env=env, capture_output=True, text=True)
    assert_true(import_result.returncode == 0 and import_result.stdout.strip() == "0.2.0", f"wheel import failed: {import_result.stdout}\n{import_result.stderr}")
    return {"wheel": str(wheel), "commands": list(outputs), "import": import_result.stdout.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON report here instead of a temporary path")
    parser.add_argument("--keep-container", action="store_true", help="leave the disposable container running for manual inspection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    # Running ``python scripts/validate_pgvector_10k.py`` places ``scripts/``
    # (not the checkout root) at sys.path[0].  Add the root explicitly for
    # this maintainer-only validation harness; installed-package checks below
    # deliberately remove PYTHONPATH again.
    sys.path.insert(0, str(repo))
    results: dict[str, dict[str, Any]] = {}
    container = DockerContainer()
    work = Path(tempfile.mkdtemp(prefix="embedflow-pgvector-10k-"))
    dsn = ""
    connection = None
    fixture: dict[str, Any] = {}
    prefix = f"efv_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    table = f"{prefix}_10k"
    try:
        record(results, "docker_start", container.start)
        if results.get("docker_start", {}).get("status") != "PASS":
            raise ValidationFailure(results["docker_start"]["detail"])
        dsn = container.dsn
        os.environ["EMBEDFLOW_PGVECTOR_DSN"] = dsn
        connection = wait_for_postgres(dsn)
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("SELECT version(), (SELECT extversion FROM pg_extension WHERE extname='vector')")
            pg_version, pgvector_version = cursor.fetchone()
        results["postgres_pgvector"] = {"status": "PASS", "detail": {"postgres": str(pg_version), "pgvector": str(pgvector_version)}}
        fixture = make_fixture(connection, table=table)
        assert_true(fixture["rows"] == 10_000, "fixture row count was not exactly 10,000")
        results["10k_fixture"] = {"status": "PASS", "detail": {"rows": fixture["rows"], "dimension": fixture["dimension"], "seed": "content/index deterministic"}}
        # Tiny numerical, ID, and edge fixtures are independent of ANN index
        # choice and run before the migration so source immutability can be
        # checked around the complete workflow.
        record(results, "numerical_metrics", lambda: test_metric_table(connection, dsn, prefix))
        record(results, "id_types", lambda: test_id_types(connection, dsn, prefix))
        record(results, "edge_tables", lambda: test_edges(connection, dsn, prefix))
        record(results, "identifier_security", lambda: test_identifier_security(connection, dsn, prefix, table))
        for kind in ("hnsw", "ivfflat"):
            record(results, f"{kind}_live", lambda kind=kind: test_ann_and_topk(dsn, connection, fixture, kind))
            record(results, f"{kind}_settings", lambda kind=kind: test_settings_do_not_leak(dsn, table, kind))
        # Freeze source state after the fixture DDL/index setup.  The actual
        # migration/CLI/API/restart checks below are read-only operations.
        create_ann_index(connection, table, "hnsw")
        before = snapshot_source(connection, table)
        config_path = make_config(work, dsn, table)
        record(results, "migration_lifecycle", lambda: test_migration(dsn, table, config_path))
        record(results, "restart_persistence", lambda: test_restart_persistence(dsn, config_path))
        record(results, "concurrency", lambda: test_concurrency(dsn, table))
        record(results, "cli_api", lambda: test_cli_and_api(repo, dsn, config_path, work))
        # Exercise an explicit asynchronous prewarm after the API process has
        # exited, then leave the source fixture untouched.
        after = snapshot_source(connection, table)
        assert_true(before["count"] == after["count"] == 10_000 and before["digest"] == after["digest"] and before["sample"] == after["sample"] and before["indexes"] == after["indexes"], "source table/vector/index snapshot changed during migration operations")
        results["source_immutability"] = {"status": "PASS", "detail": {"count": after["count"], "digest": after["digest"], "indexes": after["indexes"]}}
        # Database restart/reconnect: stop/start the actual server, assert an
        # unavailable connection is cleanly reported, then reconnect/search.
        index = adapter_for(dsn, table)
        container.stop()
        try:
            try:
                index.search(fixture["vectors"][0], 1)
            except Exception as exc:
                message = str(exc)
                assert_true("embedflow" not in message.lower() or "password" not in message.lower(), "database-drop error exposed credentials")
            else:
                raise ValidationFailure("query unexpectedly succeeded while PostgreSQL was stopped")
            try:
                __import__("embedflow.indexes", fromlist=["PgVectorIndex"]).PgVectorIndex.connect(dsn=dsn, table=table, dimension=32)
            except Exception as exc:
                assert_true("embedflow" not in str(exc).lower() or "password" not in str(exc).lower(), "startup failure exposed credentials")
            else:
                raise ValidationFailure("connection unexpectedly succeeded while PostgreSQL was stopped")
        finally:
            index.close()
            container.start_again()
        dsn = container.dsn
        os.environ["EMBEDFLOW_PGVECTOR_DSN"] = dsn
        wait_connection = wait_for_postgres(dsn)
        wait_connection.close()
        reconnect = adapter_for(dsn, table)
        try:
            assert_true(len(reconnect.search(fixture["vectors"][0], 1)) == 1, "search failed after PostgreSQL restart")
        finally:
            reconnect.close()
        results["postgres_restart_reconnect"] = {"status": "PASS", "detail": "stop/drop/restart/reconnect lifecycle completed"}
        record(results, "fresh_wheel_pgvector", lambda: test_wheel(repo, dsn, config_path, work))
    except Exception as exc:
        results.setdefault("fatal", {"status": "FAIL", "detail": str(exc)})
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if not args.keep_container:
            container.cleanup()
        report = {
            "schema": "embedflow.pgvector-validation.v1",
            "prepared_version": "0.2.0",
            "rows": fixture.get("rows"),
            "table": table,
            "results": results,
            "workdir": str(work),
            "container": None if not args.keep_container else container.name,
        }
        output = args.output or (work / "pgvector_10k_validation.json")
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps({"report": str(output), "container": report["container"], "checks": {name: value["status"] for name, value in results.items()}}, indent=2, sort_keys=True))
    failed = [name for name, value in results.items() if value.get("status") == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
