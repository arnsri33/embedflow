from __future__ import annotations

import numpy as np
import pytest

from embedflow.config import from_dict
from embedflow.indexes.pgvector_backend import PgVectorDocumentStore, PgVectorIndex, _redact_error, _vector_literal


class _Connection:
    """Small connection double; search tests replace the SQL executor below."""

    def commit(self):
        pass

    def rollback(self):
        pass


def _index(metric: str, rows: list[tuple[object, float]], **kwargs) -> tuple[PgVectorIndex, list[tuple[object, tuple]]]:
    index = PgVectorIndex(_Connection(), dimension=2, metric=metric, **kwargs)
    calls: list[tuple[object, tuple]] = []

    def execute(statement, params=(), **_):
        calls.append((statement, tuple(params)))
        return rows

    index._execute = execute  # type: ignore[method-assign]
    return index, calls


@pytest.mark.parametrize(
    ("metric", "rows", "expected"),
    [
        ("cosine", [0.0, 1.0, 2.0], [1.0, 0.0, -1.0]),
        ("l2", [0.0, 1.0, 2.0], [-0.0, -1.0, -2.0]),
        ("inner_product", [-2.0, -1.0, 0.0], [2.0, 1.0, 0.0]),
    ],
)
def test_pgvector_distance_to_embedflow_score_conversion(metric, rows, expected):
    index, _ = _index(metric, [("best", rows[0]), ("middle", rows[1]), ("worst", rows[2])])
    hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), 3)
    assert [hit.document_id for hit in hits] == ["best", "middle", "worst"]
    assert [hit.score for hit in hits] == expected


@pytest.mark.parametrize("metric", ["cosine", "l2", "inner_product"])
def test_pgvector_metric_order_matches_numpy_ground_truth(metric):
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.25], dtype=np.float32)
    if metric == "cosine":
        distances = 1.0 - (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
    elif metric == "l2":
        distances = np.linalg.norm(vectors - query, axis=1)
    else:
        # pgvector's <#> operator returns the negative inner product.
        distances = -(vectors @ query)
    order = np.lexsort((np.arange(len(vectors)), distances))
    rows = [(f"doc-{int(index)}", float(distances[index])) for index in order]
    index, _ = _index(metric, rows)
    hits = index.search(query, len(vectors))
    assert [hit.document_id for hit in hits] == [f"doc-{int(index)}" for index in order]


def test_pgvector_query_is_parameterized_and_identifiers_are_quoted():
    malicious_table = 'documents"; DROP TABLE users; --'
    index, calls = _index("cosine", [("42", 0.25)], table=malicious_table)
    vector = np.asarray([0.25, 0.75], dtype=np.float32)
    hits = index.search(vector, 1)
    assert hits[0].score == pytest.approx(0.75)
    statement, params = calls[0]
    sql_text = statement.as_string(None)
    assert '"documents""; DROP TABLE users; --"' in sql_text
    assert "DROP TABLE users" not in sql_text.replace('"documents""; DROP TABLE users; --"', "")
    assert params[0] == params[1] == "[0.25,0.75]"
    assert "0.25" not in sql_text


def test_pgvector_fetch_documents_uses_bound_ids_and_quoted_columns():
    index = PgVectorIndex(_Connection(), dimension=2, table="select", id_column="from", text_column="content")
    calls = []

    def execute(statement, params=(), **_):
        calls.append((statement, tuple(params)))
        return [(42, "answer"), ("uuid-value", "text")]

    index._execute = execute  # type: ignore[method-assign]
    result = index.fetch_documents([42, "uuid-value"])
    assert result == {"42": "answer", "uuid-value": "text"}
    statement, params = calls[0]
    sql_text = statement.as_string(None)
    assert '"select"' in sql_text and '"from"' in sql_text and '"content"' in sql_text
    assert params == (["42", "uuid-value"],)


def test_pgvector_vector_validation_and_metric_aliases():
    assert _vector_literal(np.asarray([1, 2], dtype=np.float32), 2) == "[1.0,2.0]"
    assert PgVectorIndex(_Connection(), dimension=2, metric="euclidean").metric == "l2"
    assert PgVectorIndex(_Connection(), dimension=2, metric="dot").metric == "inner_product"
    with pytest.raises(ValueError, match="metric"):
        PgVectorIndex(_Connection(), dimension=2, metric="manhattan")
    with pytest.raises(ValueError, match="NUL"):
        PgVectorIndex(_Connection(), dimension=2, table="bad\x00name")
    with pytest.raises(ValueError, match="finite"):
        _vector_literal(np.asarray([np.nan, 1], dtype=np.float32), 2)


def test_pgvector_settings_are_positive_integers():
    index = PgVectorIndex(_Connection(), dimension=2, hnsw_ef_search=100, ivfflat_probes=20)
    assert index.hnsw_ef_search == 100
    assert index.ivfflat_probes == 20
    with pytest.raises(ValueError, match="positive"):
        PgVectorIndex(_Connection(), dimension=2, hnsw_ef_search=0)


def test_pgvector_config_accepts_table_contract_and_l2():
    cfg = from_dict(
        {
            "source": {"model": "source", "dimension": 2},
            "target": {"model": "target", "dimension": 3},
            "index": {
                "backend": "pgvector",
                "metric": "l2",
                "dsn_env": "TEST_DSN",
                "schema": "public",
                "table": "documents",
                "id_column": "id",
                "vector_column": "embedding",
                "text_column": "content",
                "hnsw_ef_search": 80,
                "ivfflat_probes": 12,
            },
        }
    )
    assert cfg.index.backend == "pgvector"
    assert cfg.index.metric == "l2"
    assert cfg.index.hnsw_ef_search == 80


def test_pgvector_config_rejects_bad_identifier_and_metric():
    base = {"source": {"model": "source", "dimension": 2}, "target": {"model": "target", "dimension": 3}}
    with pytest.raises(ValueError, match="metric"):
        from_dict({**base, "index": {"backend": "pgvector", "metric": "manhattan"}})
    with pytest.raises(ValueError, match="table"):
        from_dict({**base, "index": {"backend": "pgvector", "table": ""}})


def test_pgvector_config_round_trip_and_environment_overrides(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
source: {model: source, dimension: 2}
target: {model: target, dimension: 3}
index:
  backend: pgvector
  dsn_env: BASE_DSN
  schema: public
  table: documents
  metric: cosine
"""
    )
    monkeypatch.setenv("EMBEDFLOW_PGVECTOR_TABLE", "quoted-table")
    monkeypatch.setenv("EMBEDFLOW_PGVECTOR_HNSW_EF_SEARCH", "75")
    from embedflow.config import load_config

    cfg = load_config(path)
    assert cfg.index.table == "quoted-table"
    assert cfg.index.hnsw_ef_search == 75
    assert cfg.index.dsn_env == "BASE_DSN"


def test_pgvector_error_redacts_dsn_credentials():
    message = _redact_error(RuntimeError("could not connect postgresql://alice:super-secret@example.test/db?password=query-secret password=another"))
    assert "super-secret" not in message
    assert "query-secret" not in message
    assert "another" not in message
    assert "<redacted>" in message


def test_pgvector_document_store_is_lazy_and_strict():
    class FakeIndex:
        def size(self):
            return 2

        def iter_ids(self):
            yield from ("42", "abc")

        def fetch_documents(self, ids):
            return {key: f"text-{key}" for key in ids}

    store = PgVectorDocumentStore(FakeIndex())
    assert store.size() == 2
    assert list(store.documents) == ["42", "abc"]
    assert store.get([42, "abc"]) == {"42": "text-42", "abc": "text-abc"}
    assert store.documents["42"] == "text-42"


def test_postgres_dsn_is_inferred_as_pgvector_for_direct_facade_and_cli(tmp_path, monkeypatch):
    from argparse import Namespace

    from embedflow.cli import _direct_analysis_config

    args = Namespace(
        documents="docs.jsonl", index="postgresql://user:pass@localhost/db",
        source_model="source", target_model="target", backend=None,
        output_dir=str(tmp_path), collection="embedflow", vector_name=None,
        api_key_env="QDRANT_API_KEY", metric="cosine", index_ids=None,
        dsn_env="EMBEDFLOW_PGVECTOR_DSN", schema="public", table="documents",
        id_column="id", vector_column="embedding", text_column="content",
        hnsw_ef_search=None, ivfflat_probes=None, model_root=None,
        kmax=10, limit=None, seed=42, probe_queries="queries.jsonl",
    )
    monkeypatch.delenv("EMBEDFLOW_PGVECTOR_DSN", raising=False)
    config_path, cfg = _direct_analysis_config(args)
    assert cfg.index.backend == "pgvector"
    assert "secret" not in config_path.read_text()
    assert cfg.index.url is None
