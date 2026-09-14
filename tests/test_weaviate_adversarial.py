"""Independent Weaviate normalization and metric-invariant checks.

These tests deliberately use a small transport-shaped client double rather
than calling adapter helpers directly.  They exercise the response parser and
score convention with inputs beyond the hand-written fixtures.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from embedflow.indexes import WeaviateIndex


class ResponseCollection:
    def __init__(self, objects):
        self.objects = objects
        self.query = SimpleNamespace(near_vector=self.near_vector)

    def near_vector(self, **kwargs):
        return SimpleNamespace(objects=self.objects)


def _index(objects, *, metric="cosine"):
    return WeaviateIndex(ResponseCollection(objects), collection_name="Documents", dimension=4,
                         metric=metric, text_property=None, documents={})


def test_property_fuzz_two_thousand_response_lists():
    rng = random.Random(20260913)
    for example in range(2_000):
        count = rng.randint(1, 32)
        requested = rng.randint(1, 40)
        ids = [f"id-{example}-{i}-✓" for i in range(count)]
        distances = [rng.uniform(0.0, 100.0) for _ in ids]
        objects = [SimpleNamespace(uuid=doc_id, metadata=SimpleNamespace(distance=distance), properties={})
                   for doc_id, distance in zip(ids, distances)]
        original_ids = [obj.uuid for obj in objects]
        got = _index(objects).search(np.ones(4, dtype="float32"), requested)
        assert len(got) == min(count, requested)
        assert [hit.document_id for hit in got] == ids[:len(got)]
        assert all(isinstance(hit.document_id, str) and np.isfinite(hit.score) for hit in got)
        assert [obj.uuid for obj in objects] == original_ids


def test_metamorphic_metric_rankings_preserve_order():
    rng = np.random.default_rng(1209)
    corpus = rng.normal(size=(32, 4)).astype("float32")
    query = rng.normal(size=4).astype("float32")
    shift = np.asarray([2.0, -1.0, 4.0, 0.5], dtype="float32")

    def run(metric, values, q):
        if metric == "cosine":
            distances = 1.0 - (values @ q) / (np.linalg.norm(values, axis=1) * np.linalg.norm(q))
        elif metric == "dot":
            distances = 1.0 - values @ q
        else:
            distances = np.sum((values - q) ** 2, axis=1)
        order = np.argsort(distances, kind="stable")
        objects = [SimpleNamespace(uuid=f"{metric}-{i}", metadata=SimpleNamespace(distance=float(distances[i])), properties={})
                   for i in order]
        got = _index(objects, metric=metric).search(q, len(values))
        return [hit.document_id for hit in got]

    cosine = run("cosine", corpus, query)
    assert cosine == run("cosine", corpus * 7.0, query)
    dot = run("dot", corpus, query)
    assert dot == run("dot", corpus, query * 3.0)
    l2 = run("l2", corpus, query)
    assert l2 == run("l2", corpus + shift, query + shift)
