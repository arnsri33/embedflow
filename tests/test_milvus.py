from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from embedflow.config import load_config
from embedflow.indexes import MilvusDocumentStore, MilvusIndex, normalize_milvus_metric
from embedflow.indexes.milvus_backend import _MAX_TOP_K, _redact_milvus_error


class FakeMilvus:
    def __init__(self, *, fields=None, hits=None, rows=None, dimension=2, state="Loaded"):
        self.fields = fields or [
            {"name": "id", "data_type": 5, "is_primary": True},
            {"name": "embedding", "data_type": 101, "params": {"dim": dimension}},
            {"name": "content", "data_type": 21},
        ]
        self.hits = hits if hits is not None else [[{"id": "1", "distance": 0.9}, {"id": "2", "distance": 0.1}]]
        self.rows = rows or [{"id": 1, "content": "one"}, {"id": 2, "content": "two"}]
        self.dimension = dimension
        self.state = state
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def describe_collection(self, **kwargs):
        self.calls.append(("describe_collection", kwargs))
        return {"fields": self.fields}

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        return ["idx_embedding"]

    def describe_index(self, **kwargs):
        self.calls.append(("describe_index", kwargs))
        return {"index_type": "HNSW", "metric_type": "COSINE", "params": {"ef": 64}}

    def get_load_state(self, **kwargs):
        self.calls.append(("get_load_state", kwargs))
        return {"state": self.state}

    def load_collection(self, **kwargs):
        self.calls.append(("load_collection", kwargs))
        self.state = "Loaded"

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return self.hits

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        requested = {str(x) for x in kwargs["ids"]}
        return [row for row in self.rows if str(row.get("id")) in requested]

    def get_collection_stats(self, **kwargs):
        self.calls.append(("get_collection_stats", kwargs))
        return {"row_count": len(self.rows)}

    def close(self):
        self.closed = True


def make_index(client=None, **kwargs):
    client = client or FakeMilvus()
    return MilvusIndex(client, collection="documents", dimension=2, **kwargs)


def test_metric_aliases_and_score_direction():
    assert normalize_milvus_metric("cosine") == "cosine"
    assert normalize_milvus_metric("dot") == "ip"
    assert normalize_milvus_metric("inner_product") == "ip"
    assert normalize_milvus_metric("euclidean") == "l2"
    for metric, raw, expected in (("cosine", [0.9, 0.1], [0.9, 0.1]), ("ip", [3, 1], [3, 1]), ("l2", [0.1, 3], [-0.1, -3])):
        client = FakeMilvus(hits=[[{"id": "a", "distance": raw[0]}, {"id": "b", "distance": raw[1]}]])
        got = make_index(client, metric=metric).search(np.ones(2, dtype=np.float32), 2)
        assert [item.document_id for item in got] == ["a", "b"]
        assert [item.score for item in got] == expected


def test_request_shape_external_documents_and_metadata_text():
    client = FakeMilvus()
    index = make_index(client, documents={"1": "one"}, partition_names=["p1"], search_params={"ef": 64})
    index.search(np.array([1, 0], dtype=np.float32), 500)
    request = next(args for name, args in client.calls if name == "search")
    assert request["collection_name"] == "documents"
    assert request["limit"] == 500
    assert request["anns_field"] == "embedding"
    assert request["partition_names"] == ["p1"]
    assert request["output_fields"] == []
    assert request["search_params"] == {"metric_type": "COSINE", "params": {"ef": 500}}
    assert request["data"] == [[1.0, 0.0]]

    metadata_client = FakeMilvus(hits=[[{"id": "1", "distance": 1, "entity": {"content": "one"}}]])
    metadata_index = make_index(metadata_client, text_field="content")
    metadata_index.search(np.ones(2, dtype=np.float32), 1)
    request = next(args for name, args in metadata_client.calls if name == "search")
    assert request["output_fields"] == ["content"]


def test_hnsw_ef_is_never_sent_below_requested_limit():
    client = FakeMilvus()
    index = make_index(client, search_params={"ef": 8})
    index.search(np.ones(2, dtype=np.float32), 50)
    request = next(args for name, args in client.calls if name == "search")
    assert request["search_params"]["params"]["ef"] == 50


def test_get_is_batched_and_integer_ids_are_typed():
    client = FakeMilvus()
    index = make_index(client)
    index._id_kind = "int"
    assert index.fetch_documents(["1", "2"]) == {"1": "one", "2": "two"}
    get = next(args for name, args in client.calls if name == "get")
    assert get["ids"] == [1, 2]
    assert sum(name == "get" for name, _ in client.calls) == 1


def test_schema_checks_multiple_vectors_and_unsupported_types():
    fields = [
        {"name": "id", "data_type": 21, "is_primary": True},
        {"name": "a", "data_type": 101, "params": {"dim": 2}},
        {"name": "b", "data_type": 101, "params": {"dim": 2}},
    ]
    with pytest.raises(ValueError, match="vector_field"):
        MilvusIndex.connect  # keep this branch explicit for type checkers
        index = MilvusIndex(FakeMilvus(fields=fields), collection="documents", dimension=2, vector_field=None)
        index._introspect()
    sparse = [
        {"name": "id", "data_type": 5, "is_primary": True},
        {"name": "embedding", "data_type": 104, "params": {"dim": 2}},
    ]
    with pytest.raises(ValueError, match="unsupported"):
        index = MilvusIndex(FakeMilvus(fields=sparse), collection="documents", dimension=2)
        index._introspect()
    bad_id = [
        {"name": "id", "data_type": 4, "is_primary": True},
        {"name": "embedding", "data_type": 101, "params": {"dim": 2}},
    ]
    with pytest.raises(ValueError, match="INT64 or VARCHAR"):
        MilvusIndex(FakeMilvus(fields=bad_id), collection="documents", dimension=2)._introspect()


def test_text_field_contract_is_checked_at_connection_time():
    fields = [
        {"name": "id", "data_type": 5, "is_primary": True},
        {"name": "embedding", "data_type": 101, "params": {"dim": 2}},
    ]
    with pytest.raises(ValueError, match="text field"):
        MilvusIndex(FakeMilvus(fields=fields), collection="documents", dimension=2, text_field="content")._introspect()


def test_dimension_load_state_and_auto_load():
    with pytest.raises(ValueError, match="produces 3"):
        index = MilvusIndex(FakeMilvus(dimension=2), collection="documents", dimension=3)
        index._introspect(configured_dimension=3)
    unloaded = FakeMilvus(state="NotLoad")
    index = make_index(unloaded)
    with pytest.raises(RuntimeError, match="not loaded"):
        index._ensure_loaded()
    auto = make_index(FakeMilvus(state="NotLoad"), auto_load=True)
    auto._ensure_loaded()
    assert any(name == "load_collection" for name, _ in auto.client.calls)


def test_malformed_response_and_input_validation():
    for hits, pattern in (([[{"distance": 1}]], "without an ID"), ([[{"id": "a"}]], "distance/score"), ([[{"id": "a", "distance": float("nan")}]], "non-finite"), ([[{"id": "a", "distance": 1}, {"id": "a", "distance": 0}]], "duplicate")):
        index = make_index(FakeMilvus(hits=hits))
        with pytest.raises(ValueError, match=pattern):
            index.search(np.ones(2, dtype=np.float32), 2)
    index = make_index()
    with pytest.raises(ValueError, match="<="):
        index.search(np.ones(2, dtype=np.float32), _MAX_TOP_K + 1)
    with pytest.raises(ValueError, match="non-finite"):
        index.search(np.array([np.inf, 0], dtype=np.float32), 1)


def test_response_object_shapes_and_metadata_are_safe():
    match = SimpleNamespace(id="x", distance=0.7, entity={"content": "unicode ✓"})
    client = FakeMilvus(hits=[[match]])
    index = make_index(client)
    assert index.search(np.ones(2, dtype=np.float32), 1)[0].document_id == "x"
    index._introspect()
    metadata = index.metadata()
    assert metadata["backend"] == "milvus"
    assert metadata["index_type"] == "HNSW"
    assert "MILVUS_SUPER_SECRET_123" not in json.dumps(metadata)


def test_secret_redaction():
    message = _redact_milvus_error(RuntimeError("token=MILVUS_SUPER_SECRET_123 password=bad"), "MILVUS_SUPER_SECRET_123")
    assert "MILVUS_SUPER_SECRET_123" not in message
    assert "password=bad" not in message


def test_document_store_missing_text_is_explicit():
    client = FakeMilvus(rows=[{"id": 1, "content": None}])
    store = MilvusDocumentStore(make_index(client))
    with pytest.raises(KeyError, match="missing"):
        store.get(["1"])


def test_config_and_environment_overrides(tmp_path, monkeypatch):
    path = tmp_path / "embedflow.yaml"
    path.write_text("""source: {model: source, dimension: 2}\ntarget: {model: target, dimension: 3}\nindex:\n  backend: milvus\n  uri: http://localhost:19530\n  collection: documents\n""")
    monkeypatch.setenv("EMBEDFLOW_MILVUS_URI", "http://override:19530")
    monkeypatch.setenv("EMBEDFLOW_MILVUS_COLLECTION", "quoted")
    monkeypatch.setenv("EMBEDFLOW_MILVUS_PARTITIONS", "a,b")
    cfg = load_config(path)
    assert cfg.index.uri == "http://override:19530"
    assert cfg.index.collection == "quoted"
    assert cfg.index.partition_names == ["a", "b"]


def test_concurrent_queries_preserve_client_state():
    client = FakeMilvus(hits=[[{"id": "x", "distance": 1}]])
    index = make_index(client, partition_names=["fixed"])
    errors = []

    def run():
        try:
            assert index.search(np.ones(2, dtype=np.float32), 1)[0].document_id == "x"
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert all(call[1]["partition_names"] == ["fixed"] for call in client.calls if call[0] == "search")
