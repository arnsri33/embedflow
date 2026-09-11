from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from embedflow.cache import SQLiteVectorCache
from embedflow.config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    from_dict,
    load_config,
)
from embedflow.indexes import PineconeDocumentStore, PineconeIndex, normalize_pinecone_metric
from embedflow.indexes.pinecone_backend import _MAX_TOP_K, _redact_pinecone_error
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel
from embedflow.serving.engine import MigrationEngine


class FakePineconeIndex:
    def __init__(self, matches=None, *, dimension=2, namespaces=None, vectors=None):
        self.matches = list(matches or [])
        self.dimension = dimension
        self.namespaces = namespaces if namespaces is not None else {"": {"vector_count": len(self.matches)}}
        self.vectors = vectors or {}
        self.calls = []
        self.closed = False
        self.fetch_calls = []

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return {"matches": list(self.matches)}

    def describe_index_stats(self):
        self.calls.append(("describe_index_stats", {}))
        return {"dimension": self.dimension, "namespaces": self.namespaces,
                "total_vector_count": sum(int(v.get("vector_count", 0)) for v in self.namespaces.values())}

    def fetch(self, **kwargs):
        self.fetch_calls.append(kwargs)
        return {"vectors": {key: value for key, value in self.vectors.items() if key in kwargs["ids"]}}

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return list(self.vectors)

    def close(self):
        self.closed = True


def test_metric_aliases_and_score_direction():
    assert normalize_pinecone_metric("cosine") == "cosine"
    assert normalize_pinecone_metric("dot") == "dotproduct"
    assert normalize_pinecone_metric("inner_product") == "dotproduct"
    assert normalize_pinecone_metric("l2") == "euclidean"
    for metric, scores, expected in (
        ("cosine", [0.9, 0.2], [0.9, 0.2]),
        ("dotproduct", [5.0, 1.0], [5.0, 1.0]),
        ("euclidean", [0.25, 4.0], [-0.25, -4.0]),
    ):
        client = FakePineconeIndex([{"id": f"d{i}", "score": score} for i, score in enumerate(scores)])
        index = PineconeIndex(client, dimension=2, metric=metric)
        hits = index.search(np.array([1, 0], dtype="float32"), 2)
        assert [hit.document_id for hit in hits] == ["d0", "d1"]
        assert [hit.score for hit in hits] == expected


def test_query_shape_is_read_only_and_metadata_is_opt_in():
    client = FakePineconeIndex([{"id": "42", "score": 0.8, "metadata": {"text": "answer"}}])
    index = PineconeIndex(client, dimension=2, namespace="production")
    index.search(np.array([1, 0], dtype="float32"), 500)
    name, kwargs = client.calls[0]
    assert name == "query"
    assert kwargs["top_k"] == 500
    assert kwargs["namespace"] == "production"
    assert kwargs["include_values"] is False
    assert kwargs["include_metadata"] is False
    assert not any(name in {"upsert", "delete", "create_index", "delete_index"} for name, _ in client.calls)

    metadata_client = FakePineconeIndex([{"id": "42", "score": 0.8, "metadata": {"text": "answer"}}])
    metadata_index = PineconeIndex(metadata_client, dimension=2, text_metadata_field="text")
    assert metadata_index.search(np.array([1, 0], dtype="float32"), 1)[0].document_id == "42"
    assert metadata_client.calls[0][1]["include_metadata"] is True
    assert metadata_client.calls[0][1]["include_values"] is False


def test_ids_stay_strings_and_duplicate_or_malformed_matches_fail():
    client = FakePineconeIndex([{"id": 42, "score": 1.0}, {"id": "uuid-looking", "score": 0.5}])
    index = PineconeIndex(client, dimension=2)
    assert [hit.document_id for hit in index.search(np.ones(2, dtype="float32"), 2)] == ["42", "uuid-looking"]
    duplicate = PineconeIndex(FakePineconeIndex([{"id": "42", "score": 1}, {"id": 42, "score": 0.5}]), dimension=2)
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.search(np.ones(2, dtype="float32"), 2)
    malformed = PineconeIndex(FakePineconeIndex([{"id": "x"}]), dimension=2)
    with pytest.raises(ValueError, match="score"):
        malformed.search(np.ones(2, dtype="float32"), 1)


def test_external_document_store_batches_without_fetching_pinecone_metadata():
    client = FakePineconeIndex([{"id": "1", "score": 1}, {"id": "2", "score": 0.5}])
    index = PineconeIndex(client, dimension=2, text_metadata_field="text", documents={"1": "one", "2": "two"})
    assert index.fetch_documents(["1", "2"]) == {"1": "one", "2": "two"}
    index.search(np.ones(2, dtype="float32"), 2)
    assert client.calls[0][1]["include_metadata"] is False
    assert client.fetch_calls == []


def test_metadata_document_store_fetches_one_batch_and_reports_missing_text():
    client = FakePineconeIndex(
        dimension=2,
        vectors={"1": {"id": "1", "metadata": {"text": "one"}}, "2": {"id": "2", "metadata": {}}},
    )
    index = PineconeIndex(client, dimension=2, text_metadata_field="text", namespace="ns")
    store = PineconeDocumentStore(index)
    assert store.get(["1"]) == {"1": "one"}
    assert client.fetch_calls == [{"ids": ["1"], "namespace": "ns"}]
    with pytest.raises(KeyError, match="metadata field"):
        store.get(["2"])


def test_namespace_stats_and_metadata_are_safe():
    client = FakePineconeIndex(
        dimension=3,
        namespaces={"production": {"vector_count": 7}, "other": {"vector_count": 2}},
    )
    index = PineconeIndex(client, dimension=3, namespace="production", host="host.example")
    assert index.size() == 7
    metadata = index.metadata()
    assert metadata["namespace"] == "production"
    assert metadata["size"] == 7
    assert metadata["host"] == "host.example"
    assert "PINECONE_SUPER_SECRET_123" not in json.dumps(metadata)


def test_current_sdk_shaped_stats_and_list_pages_are_supported():
    page = SimpleNamespace(vectors=[SimpleNamespace(id="one"), SimpleNamespace(id="two")])
    client = FakePineconeIndex(
        dimension=3,
        namespaces={"": {"vector_count": 2}},
    )
    client.list = lambda **kwargs: iter([page])
    client.describe_index_stats = lambda: SimpleNamespace(
        dimension=3,
        metric="dotproduct",
        vector_type="dense",
        namespaces={"": SimpleNamespace(vector_count=2)},
        total_vector_count=2,
    )
    index = PineconeIndex(client, dimension=3, metric="dot")
    assert list(index.iter_ids()) == ["one", "two"]
    metadata = index.metadata()
    assert metadata["metric"] == "dotproduct"
    assert metadata["vector_type"] == "dense"
    json.dumps(metadata)


def test_audit_checks_candidate_query_and_text():
    client = FakePineconeIndex([{"id": "1", "score": 1, "metadata": {"text": "one"}}], dimension=2)
    index = PineconeIndex(client, dimension=2, text_metadata_field="text")
    audit = index.audit(source_dimension=2)
    assert audit["ok"]
    assert audit["checks"]["candidate_query"]["ok"]
    assert audit["checks"]["document_text"]["ok"]


def test_top_k_limit_and_vector_validation():
    index = PineconeIndex(FakePineconeIndex(), dimension=2)
    with pytest.raises(ValueError, match="<= 10000"):
        index.search(np.ones(2, dtype="float32"), _MAX_TOP_K + 1)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(3, dtype="float32"), 1)


def test_connect_uses_host_and_secret_free_errors(monkeypatch):
    data_index = FakePineconeIndex(dimension=2)

    class FakePinecone:
        seen_key = None

        def __init__(self, api_key):
            FakePinecone.seen_key = api_key

        def Index(self, **kwargs):
            assert kwargs == {"host": "idx.svc.pinecone.io"}
            return data_index

    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone))
    monkeypatch.setenv("PINECONE_API_KEY", "PINECONE_SUPER_SECRET_123")
    index = PineconeIndex.connect(host="idx.svc.pinecone.io", dimension=2)
    assert index.host == "idx.svc.pinecone.io"
    assert FakePinecone.seen_key == "PINECONE_SUPER_SECRET_123"
    assert "PINECONE_SUPER_SECRET_123" not in json.dumps(index.metadata())

    class FailingPinecone(FakePinecone):
        def Index(self, **_):
            raise RuntimeError("api_key=PINECONE_SUPER_SECRET_123 denied")

    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FailingPinecone))
    with pytest.raises(RuntimeError) as exc:
        PineconeIndex.connect(host="idx.svc.pinecone.io", dimension=2)
    assert "PINECONE_SUPER_SECRET_123" not in str(exc.value)


def test_connect_resolves_index_name_and_dimension(monkeypatch):
    data_index = FakePineconeIndex(dimension=4)

    class FakePinecone:
        def __init__(self, api_key):
            assert api_key == "key"

        def describe_index(self, *, name):
            assert name == "demo"
            return {"host": "demo.svc.pinecone.io", "dimension": 4, "metric": "dotproduct"}

        def Index(self, **kwargs):
            assert kwargs == {"host": "demo.svc.pinecone.io"}
            return data_index

    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone))
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    index = PineconeIndex.connect(index_name="demo", dimension=4, metric="dot")
    assert index.host == "demo.svc.pinecone.io"
    assert index.index_name == "demo"
    with pytest.raises(ValueError, match="dimension"):
        PineconeIndex.connect(index_name="demo", dimension=3, metric="dot")


def test_optional_dependency_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pinecone", None)
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    with pytest.raises(RuntimeError, match=r"embedflow\[pinecone\]"):
        PineconeIndex.connect(host="idx.svc.pinecone.io", dimension=2)


def test_config_and_environment_overrides(tmp_path, monkeypatch):
    raw = {
        "source": {"model": "source", "dimension": 2},
        "target": {"model": "target", "dimension": 3},
        "index": {"backend": "pinecone", "host": "idx.svc.pinecone.io", "namespace": "production", "metric": "dot"},
    }
    cfg = from_dict(raw)
    assert cfg.index.backend == "pinecone"
    assert cfg.index.api_key_env == "PINECONE_API_KEY"
    assert cfg.index.metric == "dot"
    path = tmp_path / "embedflow.yaml"
    path.write_text("source: {model: source, dimension: 2}\ntarget: {model: target, dimension: 3}\nindex:\n  backend: pinecone\n  host: original.svc.pinecone.io\n")
    monkeypatch.setenv("EMBEDFLOW_PINECONE_HOST", "override.svc.pinecone.io")
    monkeypatch.setenv("EMBEDFLOW_PINECONE_NAMESPACE", "ns")
    loaded = load_config(path)
    assert loaded.index.host == "override.svc.pinecone.io"
    assert loaded.index.namespace == "ns"
    no_host = tmp_path / "env-only.yaml"
    no_host.write_text("source: {model: source, dimension: 2}\ntarget: {model: target, dimension: 3}\nindex:\n  backend: pinecone\n")
    monkeypatch.setenv("EMBEDFLOW_PINECONE_HOST", "env-only.svc.pinecone.io")
    env_only = load_config(no_host)
    assert env_only.index.host == "env-only.svc.pinecone.io"
    with pytest.raises(ValueError, match="host or index_name"):
        from_dict({"source": {"model": "s", "dimension": 2}, "target": {"model": "t", "dimension": 3}, "index": {"backend": "pinecone"}})


def test_secret_redaction_helper():
    message = _redact_pinecone_error(RuntimeError("api-key=PINECONE_SUPER_SECRET_123 bearer abc"), "PINECONE_SUPER_SECRET_123")
    assert "PINECONE_SUPER_SECRET_123" not in message
    assert "bearer abc" not in message


def test_concurrent_queries_keep_namespace_and_results():
    client = FakePineconeIndex([{"id": "x", "score": 1}], dimension=2)
    index = PineconeIndex(client, dimension=2, namespace="fixed")
    errors = []

    def run():
        try:
            assert index.search(np.ones(2, dtype="float32"), 1)[0].document_id == "x"
        except Exception as exc:  # pragma: no cover - assertion context
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert all(call[1]["namespace"] == "fixed" for call in client.calls if call[0] == "query")


def test_shared_migration_engine_reaches_cold_partial_warm_with_pinecone(tmp_path):
    rows = [{"id": f"doc-{i}", "text": f"document {i}"} for i in range(4)]
    document_path = tmp_path / "documents.jsonl"
    document_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    docs = DocumentStore(document_path)
    source = HashEmbeddingModel("pinecone-source", 2)
    target = HashEmbeddingModel("pinecone-target", 2)
    fake = FakePineconeIndex(
        [{"id": row["id"], "score": 1.0 - position * 0.1} for position, row in enumerate(rows)],
        dimension=2,
    )
    source_index = PineconeIndex(fake, dimension=2, metric="cosine", documents=docs.documents, host="test.svc.pinecone.io")
    cfg = EmbedFlowConfig(
        source=ModelConfig("pinecone-source", dimension=2),
        target=ModelConfig("pinecone-target", dimension=2),
        index=IndexConfig(backend="pinecone", host="test.svc.pinecone.io", path="./legacy.index"),
        documents=DocumentsConfig(path=str(document_path)),
        migration=MigrationConfig(candidate_depth=4, kmax_probe=4, max_sync_misses=2, background_batch_size=2),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        state_path=str(tmp_path / "state.json"),
    )
    cache = SQLiteVectorCache(cfg.cache.path, target.fingerprint, target.dimension)
    engine = MigrationEngine(cfg, source, target, source_index, cache, docs, start_worker=False)
    try:
        cold = engine.search("query", top_k=2, max_sync_misses=0)
        assert cold["migration"]["status"] == "COLD"
        partial = engine.search("query", top_k=2, max_sync_misses=2)
        assert partial["migration"]["status"] == "PARTIAL"
        engine.prewarm([row["id"] for row in rows], asynchronous=False)
        warm = engine.search("query", top_k=2, max_sync_misses=0)
        assert warm["migration"]["status"] == "WARM"
        assert not any(name in {"upsert", "delete", "update", "create_index", "delete_index"} for name, _ in fake.calls)
    finally:
        engine.close()
