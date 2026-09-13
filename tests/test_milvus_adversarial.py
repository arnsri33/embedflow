"""Independent Milvus audit tests.

These intentionally exercise response-shape and boundary assumptions that the
normal backend contract tests do not cover.  They use a deterministic client
double, so they remain runnable without a Milvus server; the real standalone
fixture is driven separately by ``scripts/validate_milvus.py``.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from embedflow.config import from_dict
from embedflow.indexes import MilvusDocumentStore, MilvusIndex
from embedflow.indexes.milvus_backend import _redact_milvus_error, _redact_uri

_MISSING = object()


class AuditClient:
    def __init__(self, *, hits=_MISSING, response_mode: str = "list", state: object = "Loaded"):
        self.fields = [
            {"name": "id", "data_type": 5, "is_primary": True},
            {"name": "embedding", "data_type": 101, "params": {"dim": 3}},
            {"name": "content", "data_type": 21},
        ]
        self.hits = [[{"id": "0", "distance": 1.0, "entity": {"content": "zero"}}]] if hits is _MISSING else hits
        self.get_content = "text"
        self.response_mode = response_mode
        self.state = state
        self.calls: list[tuple[str, dict]] = []

    def describe_collection(self, **kwargs):
        self.calls.append(("describe_collection", kwargs))
        return {"fields": self.fields}

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        return []

    def get_load_state(self, **kwargs):
        self.calls.append(("get_load_state", kwargs))
        return {"state": self.state}

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.response_mode == "mapping":
            return {"data": self.hits}
        if self.response_mode == "inner":
            return self.hits[0]
        return self.hits

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        wanted = {str(value) for value in kwargs["ids"]}
        return [{"id": value, "content": self.get_content if self.get_content != "text" else f"text-{value}"} for value in wanted]

    def get_collection_stats(self, **kwargs):
        self.calls.append(("get_collection_stats", kwargs))
        return {"row_count": 3}


def make_index(client: AuditClient | None = None, **kwargs) -> MilvusIndex:
    return MilvusIndex(client or AuditClient(), collection="documents", dimension=3, **kwargs)


def test_response_shape_variants_are_normalized_without_mutation():
    for mode in ("list", "mapping", "inner"):
        client = AuditClient(response_mode=mode)
        index = make_index(client)
        original = client.hits
        got = index.search(np.asarray([1, 0, 0], dtype="float32"), 1)
        assert got[0].document_id == "0"
        assert client.hits == original


@pytest.mark.parametrize(
    "hits, pattern",
    [
        ([[{"distance": 1.0}]], "without an ID"),
        ([[{"id": "x"}]], "distance/score"),
        ([[{"id": "x", "distance": float("nan")}]], "non-finite"),
        ([[{"id": "x", "distance": 1.0}, {"id": "x", "distance": 0.0}]], "duplicate"),
        ([[{"id": "x", "distance": float("inf")}]], "non-finite"),
    ],
)
def test_malformed_search_responses_fail_explicitly(hits, pattern):
    index = make_index(AuditClient(hits=hits))
    with pytest.raises(ValueError, match=pattern):
        index.search(np.asarray([1, 0, 0], dtype="float32"), 1)


def test_k_is_bounded_and_query_vector_is_finite():
    index = make_index()
    with pytest.raises(ValueError, match="positive integer"):
        index.search(np.ones(3, dtype="float32"), 0)
    with pytest.raises(ValueError, match="<="):
        index.search(np.ones(3, dtype="float32"), 16_385)
    with pytest.raises(ValueError, match="non-finite"):
        index.search(np.asarray([1, np.nan, 0], dtype="float32"), 1)


def test_missing_or_null_match_container_is_not_silently_treated_as_success():
    for hits in (None, {"data": None}):
        client = AuditClient(hits=hits)
        index = make_index(client)
        with pytest.raises(ValueError):
            index.search(np.ones(3, dtype="float32"), 1)


def test_external_document_fetch_is_one_batched_call():
    client = AuditClient(hits=[[{"id": str(i), "distance": float(i)} for i in range(500)]])
    index = make_index(client, text_field="content")
    index.search(np.ones(3, dtype="float32"), 500)
    assert sum(name == "search" for name, _ in client.calls) == 1
    assert index.fetch_documents([str(i) for i in range(500)]) == {str(i): f"text-{i}" for i in range(500)}
    assert sum(name == "get" for name, _ in client.calls) == 1


def test_metadata_text_shapes_and_type_errors():
    for value in ("", "✓ unicode\ntext"):
        client = AuditClient(hits=[[{"id": "x", "distance": 1.0, "entity": {"content": value}}]])
        assert make_index(client).search(np.ones(3, dtype="float32"), 1)[0].document_id == "x"
    for value in (None, 3, [], {"text": "bad"}):
        client = AuditClient(hits=[[{"id": "x", "distance": 1.0, "entity": {"content": value}}]])
        index = make_index(client)
        if value is None:
            # Missing/null metadata is surfaced when the shared document store
            # asks for text, rather than silently becoming the string "None".
            client.get_content = None
            assert index.search(np.ones(3, dtype="float32"), 1)
            with pytest.raises(KeyError, match="missing"):
                MilvusDocumentStore(index).get(["x"])
        else:
            with pytest.raises(ValueError, match="must be text"):
                index.search(np.ones(3, dtype="float32"), 1)


def test_config_boundaries_fail_before_client_creation():
    base = {"source": {"model": "source", "dimension": 3}, "target": {"model": "target", "dimension": 4}}
    for update, pattern in (
        ({"backend": "milvus", "collection": "x"}, "index.uri"),
        ({"backend": "milvus", "uri": "http://localhost:19530", "search_params": {"oops": 1}}, "unsupported"),
        ({"backend": "milvus", "uri": "http://localhost:19530", "partition_names": ["a", "a"]}, "duplicates"),
        ({"backend": "milvus", "uri": "http://localhost:19530", "auto_load": "yes"}, "boolean"),
    ):
        raw = {**base, "index": update}
        with pytest.raises(ValueError, match=pattern):
            from_dict(raw)


def test_omitted_vector_field_preserves_ambiguity_detection():
    raw = {
        "source": {"model": "source", "dimension": 3},
        "target": {"model": "target", "dimension": 4},
        "index": {"backend": "milvus", "uri": "http://localhost:19530", "collection": "documents"},
    }
    cfg = from_dict(raw)
    assert cfg.index.vector_field is None


def test_legacy_path_uri_is_accepted_when_uri_field_is_omitted():
    raw = {
        "source": {"model": "source", "dimension": 3},
        "target": {"model": "target", "dimension": 4},
        "index": {"backend": "milvus", "path": "http://localhost:19530", "collection": "documents"},
    }
    cfg = from_dict(raw)
    assert cfg.index.uri is None
    assert cfg.index.path == "http://localhost:19530"


def test_explicit_blank_endpoint_is_not_silently_redirected_to_localhost():
    with pytest.raises(ValueError, match="uri"):
        MilvusIndex.connect(uri="", collection="documents")
    with pytest.raises(ValueError, match="database"):
        MilvusIndex.connect(uri="http://127.0.0.1:19530", database="", collection="documents")


def test_metamorphic_score_rankings_are_metric_invariant():
    rng = np.random.default_rng(123)
    corpus = rng.normal(size=(32, 3)).astype("float32")
    query = rng.normal(size=3).astype("float32")
    cosine = corpus @ query / (np.linalg.norm(corpus, axis=1) * np.linalg.norm(query))
    cosine_scaled = (corpus * 7.0) @ query / (np.linalg.norm(corpus * 7.0, axis=1) * np.linalg.norm(query))
    assert np.array_equal(np.argsort(-cosine), np.argsort(-cosine_scaled))
    dot = corpus @ query
    assert np.array_equal(np.argsort(-dot), np.argsort(-(corpus @ (query * 3.0))))
    shift = np.asarray([2.0, -1.0, 4.0], dtype="float32")
    l2 = np.sum((corpus - query) ** 2, axis=1)
    l2_shifted = np.sum(((corpus + shift) - (query + shift)) ** 2, axis=1)
    assert np.array_equal(np.argsort(l2), np.argsort(l2_shifted))


def test_property_fuzz_two_thousand_normalization_examples():
    rng = random.Random(20260912)
    for _ in range(2_000):
        count = rng.randint(1, 32)
        k = rng.randint(1, 32)
        ids = [f"id-{rng.randrange(10**9)}-{i}" for i in range(count)]
        scores = [rng.uniform(-100.0, 100.0) for _ in ids]
        client = AuditClient(hits=[[{"id": item, "distance": score} for item, score in zip(ids, scores)]])
        got = make_index(client).search(np.ones(3, dtype="float32"), k)
        assert len(got) <= k
        assert [item.document_id for item in got] == ids[: len(got)]
        assert all(isinstance(item.document_id, str) for item in got)
        assert all(np.isfinite(item.score) for item in got)


def test_token_and_uri_redaction():
    secret = "MILVUS_SUPER_SECRET_123"
    assert secret not in _redact_milvus_error(RuntimeError(f"token={secret}"), secret)
    assert secret not in _redact_milvus_error(RuntimeError(f"milvus://user:{secret}@host"), secret)
    assert secret not in _redact_uri(f"milvus://user:{secret}@host")


def test_load_state_numeric_and_client_failure_are_actionable():
    assert make_index(AuditClient(state=3)).health_check()["ok"]
    with pytest.raises(RuntimeError, match="not loaded"):
        make_index(AuditClient(state=2))._ensure_loaded()


def test_response_object_with_nested_entity_is_supported():
    hit = SimpleNamespace(id="x", distance=0.5, entity=SimpleNamespace(entity={"content": "nested"}))
    index = make_index(AuditClient(hits=[[hit]]))
    assert index.search(np.ones(3, dtype="float32"), 1)[0].document_id == "x"


def test_dict_like_sdk_hit_properties_are_used_when_keys_are_not_materialized():
    class Hit(dict):
        id = "property-id"
        distance = 0.75

        @property
        def entity(self):
            return {"content": "property text"}

    index = make_index(AuditClient(hits=[[Hit()]]))
    assert index.search(np.ones(3, dtype="float32"), 1)[0].document_id == "property-id"


def test_iter_ids_uses_query_iterator_beyond_single_query_page():
    rows = [{"id": i} for i in range(16_500)]

    class Iterator:
        def __init__(self):
            self.offset = 0
            self.closed = False

        def next(self):
            batch = rows[self.offset:self.offset + 1_000]
            self.offset += len(batch)
            return batch

        def close(self):
            self.closed = True

    class IteratorClient(AuditClient):
        def __init__(self):
            super().__init__()
            self.iterator = Iterator()

        def get_collection_stats(self, **kwargs):
            self.calls.append(("get_collection_stats", kwargs))
            return {"row_count": len(rows)}

        def query_iterator(self, **kwargs):
            self.calls.append(("query_iterator", kwargs))
            return self.iterator

    client = IteratorClient()
    got = list(make_index(client, text_field=None).iter_ids())
    assert len(got) == len(rows)
    assert got[0] == "0" and got[-1] == "16499"
    assert client.iterator.closed
    request = next(args for name, args in client.calls if name == "query_iterator")
    assert request["batch_size"] == 1_000
    assert request["limit"] == -1
