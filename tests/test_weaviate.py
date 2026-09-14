from __future__ import annotations

import json
import threading
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from embedflow.config import load_config
from embedflow.indexes import WeaviateDocumentStore, WeaviateIndex, normalize_weaviate_metric
from embedflow.indexes.weaviate_backend import _MAX_TOP_K, _redact_uri, _redact_weaviate_error


class FakeCollection:
    def __init__(self, *, objects=None, config=None, count=3):
        self.objects = objects if objects is not None else [
            SimpleNamespace(uuid=uuid.UUID("00000000-0000-0000-0000-000000000001"), metadata=SimpleNamespace(distance=.1), properties={"content": "one"}),
            SimpleNamespace(uuid=uuid.UUID("00000000-0000-0000-0000-000000000002"), metadata=SimpleNamespace(distance=.2), properties={"content": "two"}),
        ]
        vector = SimpleNamespace(vectorIndexType="hnsw", vectorIndexConfig=SimpleNamespace(distance="cosine"))
        self.config_value = config or SimpleNamespace(
            vectorConfig={"default": vector},
            properties=[SimpleNamespace(name="content", dataType="text")],
            multiTenancyConfig=SimpleNamespace(enabled=False),
        )
        self.count = count
        self.calls: list[tuple[str, dict]] = []
        self.query = SimpleNamespace(near_vector=self.near_vector, fetch_objects_by_ids=self.fetch_objects_by_ids)
        self.aggregate = SimpleNamespace(over_all=self.over_all)
        self.config = SimpleNamespace(get=self.config)

    def config(self):  # pragma: no cover - replaced by the instance attribute above
        return self.config_value

    def near_vector(self, **kwargs):
        self.calls.append(("near_vector", kwargs))
        return SimpleNamespace(objects=self.objects)

    def fetch_objects_by_ids(self, ids, **kwargs):
        self.calls.append(("fetch_objects_by_ids", {"ids": ids, **kwargs}))
        wanted = {str(item) for item in ids}
        return SimpleNamespace(objects=[obj for obj in self.objects if str(obj.uuid) in wanted])

    def over_all(self, **kwargs):
        self.calls.append(("over_all", kwargs))
        return SimpleNamespace(total_count=self.count)

    def iterator(self, **kwargs):
        self.calls.append(("iterator", kwargs))
        return iter(self.objects)


def make_index(collection=None, **kwargs):
    collection = collection or FakeCollection()
    return WeaviateIndex(collection, collection_name="Documents", dimension=2, **kwargs)


def test_metric_aliases_and_l2_score_direction():
    assert normalize_weaviate_metric("cosine") == "cosine"
    assert normalize_weaviate_metric("dotproduct") == "dot"
    assert normalize_weaviate_metric("inner_product") == "dot"
    assert normalize_weaviate_metric("euclidean") == "l2"
    assert normalize_weaviate_metric("l2-squared") == "l2"
    index = make_index(FakeCollection(objects=[SimpleNamespace(uuid="a", metadata=SimpleNamespace(distance=.1), properties={}),
                                                SimpleNamespace(uuid="b", metadata=SimpleNamespace(distance=2), properties={})]))
    # This fixture exercises score normalization only; use the external text
    # mode so the objects do not need a content property.
    index.documents = {}
    index.metric = "l2"
    got = index.search(np.ones(2, dtype=np.float32), 2)
    assert [hit.document_id for hit in got] == ["a", "b"]
    assert [hit.score for hit in got] == [-.1, -2.0]


def test_request_shape_external_and_metadata_text():
    collection = FakeCollection()
    index = make_index(collection, documents={str(collection.objects[0].uuid): "external"})
    index.search(np.array([1, 0], dtype=np.float32), 500)
    request = next(args for name, args in collection.calls if name == "near_vector")
    assert request["limit"] == 500
    assert request["near_vector"] == [1.0, 0.0]
    assert request["include_vector"] is False
    assert request["return_properties"] == []
    assert "target_vector" not in request

    named = make_index(FakeCollection(), vector_name="default", text_property="content")
    named.search(np.array([1, 0], dtype=np.float32), 10)
    request = next(args for name, args in named.collection_client.calls if name == "near_vector")
    assert request["return_properties"] == ["content"]
    assert request["include_vector"] is False
    assert "target_vector" in request


def test_named_vector_ambiguity_and_metric_mismatch():
    vector = SimpleNamespace(vectorIndexType="hnsw", vectorIndexConfig=SimpleNamespace(distance="cosine"))
    config = SimpleNamespace(vectorConfig={"a": vector, "b": vector}, properties=[], multiTenancyConfig=SimpleNamespace(enabled=False))
    with pytest.raises(ValueError, match="multiple named vectors"):
        index = make_index(FakeCollection(config=config), text_property=None, documents={"x": "x"})
        index._introspect()
    config = SimpleNamespace(vectorConfig={"default": SimpleNamespace(vectorIndexType="flat", vectorIndexConfig=SimpleNamespace(distance="dot"))}, properties=[], multiTenancyConfig=SimpleNamespace(enabled=False))
    with pytest.raises(ValueError, match="does not match"):
        index = make_index(FakeCollection(config=config), text_property=None, documents={"x": "x"})
        index.metric = "cosine"
        index._introspect()
    unsupported = SimpleNamespace(vectorConfig={"default": SimpleNamespace(vectorIndexConfig=SimpleNamespace(distance="manhattan"))}, properties=[])
    with pytest.raises(ValueError, match="unsupported distance metric"):
        make_index(FakeCollection(config=unsupported), text_property=None, documents={"x": "x"})._introspect()


def test_text_fetch_is_batched_and_uuid_typed():
    collection = FakeCollection()
    index = make_index(collection)
    ids = [str(obj.uuid) for obj in collection.objects]
    assert index.fetch_documents(ids) == {ids[0]: "one", ids[1]: "two"}
    fetch_calls = [args for name, args in collection.calls if name == "fetch_objects_by_ids"]
    assert len(fetch_calls) == 1
    assert all(isinstance(value, uuid.UUID) for value in fetch_calls[0]["ids"])


def test_empty_and_malformed_responses_are_explicit():
    for objects, pattern in ((None, "did not contain objects"), ([], None)):
        collection = FakeCollection(objects=objects)
        if objects is None:
            collection.objects = None
        index = make_index(collection, documents={})
        if pattern:
            with pytest.raises(ValueError, match=pattern):
                index.search(np.ones(2, dtype=np.float32), 1)
        else:
            assert index.search(np.ones(2, dtype=np.float32), 1) == []
    malformed = FakeCollection(objects=[SimpleNamespace(uuid="x", metadata=SimpleNamespace(distance=float("nan")), properties={})])
    with pytest.raises(ValueError, match="non-finite"):
        make_index(malformed, documents={}).search(np.ones(2, dtype=np.float32), 1)
    duplicate = FakeCollection(objects=[SimpleNamespace(uuid="x", metadata=SimpleNamespace(distance=.1), properties={}),
                                        SimpleNamespace(uuid="x", metadata=SimpleNamespace(distance=.2), properties={})])
    with pytest.raises(ValueError, match="duplicate"):
        make_index(duplicate, documents={}).search(np.ones(2, dtype=np.float32), 2)


def test_adversarial_metadata_values_are_rejected():
    for value in (None, 1, 1.5, False, [], {}, "ok"):
        object_id = "00000000-0000-0000-0000-000000000009"
        obj = SimpleNamespace(uuid=object_id, metadata=SimpleNamespace(distance=.1), properties={"content": value})
        index = make_index(FakeCollection(objects=[obj]))
        if isinstance(value, str):
            assert index.fetch_documents([object_id]) == {object_id: "ok"}
        else:
            # A null value is treated as missing and is surfaced by the lazy
            # document store; non-string values are rejected immediately.
            if value is None:
                with pytest.raises(KeyError, match="missing"):
                    WeaviateDocumentStore(index).get([object_id])
            else:
                with pytest.raises(ValueError, match="must be text"):
                    index.fetch_documents([object_id])


def test_missing_text_property_is_reported_during_candidate_query():
    object_id = "00000000-0000-0000-0000-000000000010"
    obj = SimpleNamespace(uuid=object_id, metadata=SimpleNamespace(distance=.1), properties={})
    with pytest.raises(ValueError, match="missing text property 'content'"):
        make_index(FakeCollection(objects=[obj])).search(np.ones(2, dtype=np.float32), 1)


def test_ids_remain_strings_and_k_input_is_validated():
    obj = SimpleNamespace(uuid="42", metadata=SimpleNamespace(distance=.1), properties={})
    index = make_index(FakeCollection(objects=[obj]), documents={"42": "text"})
    assert index.search(np.ones(2, dtype=np.float32), 1)[0].document_id == "42"
    with pytest.raises(ValueError, match="positive"):
        index.search(np.ones(2, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="<="):
        index.search(np.ones(2, dtype=np.float32), _MAX_TOP_K + 1)


def test_close_is_idempotent_and_health_metadata_are_safe():
    class Client:
        def __init__(self): self.closed = 0
        def close(self): self.closed += 1
        def is_ready(self): return True
    client = Client(); index = make_index(client=client, uri="https://user:WEAVIATE_SUPER_SECRET_123@example.test")
    index._introspect()
    index._api_key = "WEAVIATE_SUPER_SECRET_123"
    index._config = {"note": "server echoed WEAVIATE_SUPER_SECRET_123", "safe": "value"}
    assert index.metadata()["backend"] == "weaviate"
    assert "WEAVIATE_SUPER_SECRET_123" not in json.dumps(index.metadata())
    index.close(); index.close(); assert client.closed == 1
    assert "WEAVIATE_SUPER_SECRET_123" not in _redact_uri("https://user:WEAVIATE_SUPER_SECRET_123@example.test")
    assert "WEAVIATE_SUPER_SECRET_123" not in _redact_weaviate_error(RuntimeError("token=WEAVIATE_SUPER_SECRET_123"), "WEAVIATE_SUPER_SECRET_123")


def test_config_and_environment_overrides(tmp_path, monkeypatch):
    path = tmp_path / "embedflow.yaml"
    path.write_text("""source: {model: source, dimension: 2}\ntarget: {model: target, dimension: 3}\nindex:\n  backend: weaviate\n  collection: Documents\n  text_property: content\n""")
    monkeypatch.setenv("EMBEDFLOW_WEAVIATE_HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("EMBEDFLOW_WEAVIATE_HTTP_PORT", "8081")
    monkeypatch.setenv("EMBEDFLOW_WEAVIATE_GRPC_PORT", "50052")
    monkeypatch.setenv("EMBEDFLOW_WEAVIATE_TENANT", "tenant-a")
    cfg = load_config(path)
    assert cfg.index.http_host == "127.0.0.1"
    assert cfg.index.http_port == 8081
    assert cfg.index.grpc_port == 50052
    assert cfg.index.tenant == "tenant-a"


def test_concurrent_queries_preserve_request_state():
    collection = FakeCollection()
    index = make_index(collection, documents={str(collection.objects[0].uuid): "one"})
    errors = []

    def run():
        try:
            index.search(np.ones(2, dtype=np.float32), 1)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(16)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors
    assert all(args["limit"] == 1 and args["include_vector"] is False for name, args in collection.calls if name == "near_vector")
