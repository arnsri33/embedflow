#!/usr/bin/env python3
"""Independent live Weaviate validation for the EmbedFlow read-only adapter.

The supplied 10k collection is treated as an existing source.  Only uniquely
named collections created by this script are deleted during cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from embedflow.cache import SQLiteVectorCache
from embedflow.config import CacheConfig, DocumentsConfig, EmbedFlowConfig, IndexConfig, MigrationConfig, ModelConfig
from embedflow.indexes import WeaviateDocumentStore, WeaviateIndex
from embedflow.models import EmbeddingModel
from embedflow.serving.engine import MigrationEngine


class DeterministicModel(EmbeddingModel):
    def __init__(self, model_id: str, dimension: int, query_vector: np.ndarray):
        self.model_id = model_id
        self.dimension = dimension
        self.fingerprint = f"weaviate-validation-{model_id}-{dimension}"
        self.query_vector = np.asarray(query_vector, dtype="float32")
        self.document_calls = 0

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.repeat(self.query_vector[None, :], len(texts), axis=0)

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        self.document_calls += len(texts)
        rows = []
        for text in texts:
            value = np.zeros(self.dimension, dtype="float32")
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            value[int.from_bytes(digest[:8], "little") % self.dimension] = 1.0
            rows.append(value)
        return np.asarray(rows, dtype="float32")

    def close(self) -> None:
        return None


def _host_ports(uri: str, http_port: int, grpc_port: int) -> tuple[str, int]:
    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    return parsed.hostname or uri, parsed.port or http_port


def _client(uri: str, http_port: int, grpc_port: int):
    try:
        import weaviate
    except ImportError as exc:
        raise RuntimeError('Weaviate validation requires: pip install "embedflow[weaviate]"') from exc
    host, parsed_port = _host_ports(uri, http_port, grpc_port)
    if parsed_port == 8080 and grpc_port == 50051:
        return weaviate.connect_to_local(host=host, port=parsed_port, grpc_port=grpc_port)
    return weaviate.connect_to_custom(http_host=host, http_port=parsed_port, http_secure=False,
                                      grpc_host=host, grpc_port=grpc_port, grpc_secure=False)


def _distance(metric: str):
    from weaviate.classes.config import VectorDistances
    return {"cosine": VectorDistances.COSINE, "dot": VectorDistances.DOT,
            "l2": VectorDistances.L2_SQUARED}[metric]


def _create_collection(client: Any, name: str, dimension: int, *, metric: str = "cosine",
                       vector_names: tuple[str, ...] = ("default",), text: bool = True,
                       index_type: str = "flat") -> Any:
    from weaviate.classes.config import Configure, DataType, Property
    vector_configs = []
    for vector_name in vector_names:
        index = Configure.VectorIndex.flat(distance_metric=_distance(metric)) if index_type == "flat" else Configure.VectorIndex.hnsw(distance_metric=_distance(metric), ef=64)
        vector_configs.append(Configure.Vectors.self_provided(name=vector_name, vector_index_config=index))
    config: Any = vector_configs[0] if len(vector_configs) == 1 else vector_configs
    properties = []
    if text:
        properties.append(Property(name="content", data_type=DataType.TEXT))
    properties.append(Property(name="fixture_marker", data_type=DataType.TEXT))
    return client.collections.create(name, vector_config=config, properties=properties)


def _insert(collection: Any, vectors: np.ndarray, *, start: int = 0, marker: str = "embedflow-weaviate-validation") -> None:
    from weaviate.classes.data import DataObject
    ns = uuid.UUID("6f1ec4d0-1eb4-5e0a-9f0e-5bded477a9ce")
    objects = []
    for offset, vector in enumerate(vectors):
        i = start + offset
        objects.append(DataObject(uuid=uuid.uuid5(ns, f"object-{i}"),
                                  properties={"content": f"live Weaviate text {i}", "fixture_marker": marker},
                                  vector=vector.tolist()))
    for begin in range(0, len(objects), 200):
        result = collection.data.insert_many(objects[begin:begin + 200])
        errors = getattr(result, "errors", {}) or {}
        if errors:
            raise RuntimeError(f"Weaviate fixture insert failed for {len(errors)} objects")


def _wait_count(collection: Any, expected: int, timeout: float = 180.0) -> float:
    started = time.monotonic()
    deadline = started + timeout
    observed = 0
    while time.monotonic() < deadline:
        observed = int(collection.aggregate.over_all(total_count=True).total_count or 0)
        if observed >= expected:
            return time.monotonic() - started
        time.sleep(0.5)
    raise RuntimeError(f"Weaviate fixture visibility timed out: expected {expected}, observed {observed}")


def _uuid_for(i: int) -> str:
    return str(uuid.uuid5(uuid.UUID("6f1ec4d0-1eb4-5e0a-9f0e-5bded477a9ce"), f"object-{i}"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = _client(args.uri, args.http_port, args.grpc_port)
    temporary: list[str] = []
    result: dict[str, Any] = {"collection": args.collection, "rows": args.rows, "dimension": args.dimension}
    source = None
    try:
        source_collection = client.collections.get(args.collection)
        before_config = json.dumps(source_collection.config.get(), sort_keys=True, default=str)
        before_count = int(source_collection.aggregate.over_all(total_count=True).total_count or 0)
        source = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                                       collection=args.collection, dimension=args.dimension, metric=args.metric,
                                       vector_name=args.vector_name, text_property="content")
        _assert(before_count == args.rows, f"expected {args.rows} source objects, got {before_count}")
        result["before_config_sha256"] = hashlib.sha256(before_config.encode()).hexdigest()
        result["before_rows"] = before_count
        result["index_type"] = source.metadata()["index_type"]
        result["vector_names"] = source.metadata()["vector_names"]
        query = np.zeros(args.dimension, dtype="float32"); query[0] = 1.0
        topk = {}
        for k in (1, 10, 50, 100, 500, 1000):
            hits = source.search(query, k)
            _assert(len(hits) <= k, f"K={k} returned too many objects")
            _assert(len({hit.document_id for hit in hits}) == len(hits), f"K={k} returned duplicate IDs")
            _assert(all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)), f"K={k} score ordering")
            topk[str(k)] = len(hits)
        result["top_k"] = topk
        result["text_sample"] = source.fetch_documents([source.search(query, 1)[0].document_id])

        after_config = json.dumps(source_collection.config.get(), sort_keys=True, default=str)
        _assert(before_config == after_config, "source collection config changed during read-only search")

        # Exact metric fixtures use the real server and the same adapter.
        metric_results = {}
        values = np.asarray([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype="float32")
        q = np.asarray([1, .25], dtype="float32")
        expected_orders = {"cosine": [0, 2, 1, 3], "dot": [2, 0, 1, 3], "l2": [0, 2, 1, 3]}
        for metric in ("cosine", "dot", "l2"):
            name = f"EmbedFlowMetric{uuid.uuid4().hex[:10]}"; temporary.append(name)
            coll = _create_collection(client, name, 2, metric=metric)
            _insert(coll, values)
            _wait_count(coll, 4)
            idx = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                                        collection=name, dimension=2, metric=metric, text_property="content")
            got = idx.search(q, 4)
            expected = [_uuid_for(i) for i in expected_orders[metric]]
            _assert([hit.document_id for hit in got] == expected, f"{metric} live ranking mismatch")
            metric_results[metric] = {"ids": [hit.document_id for hit in got], "scores": [hit.score for hit in got]}
            idx.close()
        result["metrics"] = metric_results

        # Named vectors must be selected explicitly and cannot be guessed.
        name = f"EmbedFlowNamed{uuid.uuid4().hex[:10]}"; temporary.append(name)
        coll = _create_collection(client, name, 2, vector_names=("a", "b"), text=False)
        from weaviate.classes.data import DataObject
        ids = [_uuid_for(1), _uuid_for(2)]
        coll.data.insert_many([DataObject(uuid=ids[0], properties={"fixture_marker": "named"}, vector={"a": [1, 0], "b": [0, 1]}),
                               DataObject(uuid=ids[1], properties={"fixture_marker": "named"}, vector={"a": [0, 1], "b": [1, 0]})])
        _wait_count(coll, 2)
        mapping = {ids[0]: "a", ids[1]: "b"}
        a = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                                  collection=name, dimension=2, metric="cosine", vector_name="a", text_property=None, documents=mapping)
        b = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                                  collection=name, dimension=2, metric="cosine", vector_name="b", text_property=None, documents=mapping)
        _assert(a.search(np.asarray([1, 0], dtype="float32"), 1)[0].document_id == ids[0], "named vector A")
        _assert(b.search(np.asarray([1, 0], dtype="float32"), 1)[0].document_id == ids[1], "named vector B")
        try:
            ambiguous = WeaviateIndex(coll, collection_name=name, dimension=2, metric="cosine", text_property=None, documents=mapping)
            ambiguous._introspect()
        except ValueError as exc:
            _assert("multiple named vectors" in str(exc), "ambiguous named-vector error")
        else:
            raise AssertionError("multiple named vectors were guessed")
        a.close(); b.close(); result["named_vectors"] = "PASS"

        # Shared engine lifecycle: actual candidate/text calls go through the
        # live Weaviate collection, while target vectors remain local SQLite.
        docs = WeaviateDocumentStore(source, owns_index=False)
        cache_root = Path(args.cache).expanduser().resolve()
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache = SQLiteVectorCache(cache_root, "weaviate-target", args.dimension)
        cfg = EmbedFlowConfig(source=ModelConfig("weaviate-source", dimension=args.dimension),
                              target=ModelConfig("weaviate-target", dimension=args.dimension),
                              index=IndexConfig(backend="weaviate", uri=args.uri, collection=args.collection,
                                                vector_name=args.vector_name, text_property="content", metric=args.metric,
                                                http_host=_host_ports(args.uri, args.http_port, args.grpc_port)[0],
                                                http_port=args.http_port, grpc_port=args.grpc_port),
                              documents=DocumentsConfig(path="/tmp/no-weaviate-documents.jsonl"),
                              migration=MigrationConfig(candidate_depth=50, kmax_probe=100,
                                                         max_sync_misses=1, background_batch_size=8),
                              cache=CacheConfig(path=str(cache_root)), state_path=str(cache_root / "state.json"))
        cfg.validate()
        source_model = DeterministicModel("source", args.dimension, query)
        target_model = DeterministicModel("target", args.dimension, query)
        engine = MigrationEngine(cfg, source_model, target_model, source, cache, docs, start_worker=False)
        cold = engine.search("live", top_k=5, candidate_depth=50, max_sync_misses=0)
        partial = engine.search("live", top_k=5, candidate_depth=50, max_sync_misses=1)
        _assert(cold["migration"]["status"] == "COLD", "cold lifecycle state")
        _assert(partial["migration"]["status"] in {"PARTIAL", "WARM"}, "partial lifecycle state")
        engine.worker.start()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            qstats = engine.worker.stats()["queue"]
            if qstats.get("pending", 0) == 0 and qstats.get("processing", 0) == 0:
                break
            time.sleep(.1)
        qstats = engine.worker.stats()["queue"]
        _assert(qstats.get("pending", 0) == 0 and qstats.get("processing", 0) == 0, "queue did not drain")
        warm = engine.search("live", top_k=5, candidate_depth=50, max_sync_misses=0)
        _assert(warm["migration"]["status"] == "WARM", "warm lifecycle state")
        engine.close(close_models=False, close_indexes=False); source_model.close(); target_model.close()
        source2 = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                                         collection=args.collection, dimension=args.dimension, metric=args.metric,
                                         vector_name=args.vector_name, text_property="content")
        docs2 = WeaviateDocumentStore(source2, owns_index=False)
        cache2 = SQLiteVectorCache(cache_root, "weaviate-target", args.dimension)
        engine2 = MigrationEngine(cfg, DeterministicModel("source", args.dimension, query),
                                  DeterministicModel("target", args.dimension, query), source2, cache2, docs2, start_worker=False)
        restarted = engine2.search("live", top_k=5, candidate_depth=50, max_sync_misses=0)
        _assert(restarted["migration"]["status"] == "WARM", "cache restart state")
        engine2.close(close_models=False, close_indexes=False); source2.close()
        result["migration"] = {"cold": cold["migration"], "partial": partial["migration"],
                                "warm": warm["migration"], "queue": qstats, "restart": restarted["migration"]}

        with ThreadPoolExecutor(max_workers=8) as pool:
            concurrent = list(pool.map(lambda _: source.search(query, 10), range(16)))
        _assert(all([hit.document_id for hit in rows] == [hit.document_id for hit in concurrent[0]] for rows in concurrent), "concurrent result crossover")
        result["concurrency"] = "PASS"

        after_count = int(source_collection.aggregate.over_all(total_count=True).total_count or 0)
        final_config = json.dumps(source_collection.config.get(), sort_keys=True, default=str)
        _assert(after_count == before_count, "source row count changed")
        _assert(final_config == before_config, "source schema/index config changed")
        result["after_rows"] = after_count
        result["after_config_sha256"] = hashlib.sha256(final_config.encode()).hexdigest()
        result["source_immutability"] = "PASS"
        return result
    finally:
        if source is not None:
            source.close()
        for name in temporary:
            try:
                if client.collections.exists(name):
                    client.collections.delete(name)
            except Exception as exc:
                print(f"cleanup warning for {name}: {type(exc).__name__}: {exc}", flush=True)
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Weaviate adapter validation")
    parser.add_argument("--uri", default=os.environ.get("EMBEDFLOW_WEAVIATE_URI", "http://127.0.0.1:8080"))
    parser.add_argument("--http-port", type=int, default=int(os.environ.get("EMBEDFLOW_WEAVIATE_HTTP_PORT", "8080")))
    parser.add_argument("--grpc-port", type=int, default=int(os.environ.get("EMBEDFLOW_WEAVIATE_GRPC_PORT", "50051")))
    parser.add_argument("--collection", default=os.environ.get("EMBEDFLOW_WEAVIATE_COLLECTION", "Documents10k"))
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--metric", choices=["cosine", "dot", "l2"], default="cosine")
    parser.add_argument("--vector-name", default="default")
    parser.add_argument("--cache", default="/tmp/embedflow-weaviate-validation-cache")
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args)
    result["duration_seconds_diagnostic_only"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
