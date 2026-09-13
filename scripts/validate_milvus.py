#!/usr/bin/env python3
"""Independent local Milvus validation for the read-only EmbedFlow adapter.

This script is deliberately separate from production code.  It creates only
uniquely named disposable collections, exercises the real ``pymilvus`` client,
and removes every fixture in a ``finally`` block.  It never writes to a
collection supplied through ``--collection``; that collection is used only for
read-only checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from embedflow.cache import SQLiteVectorCache
from embedflow.config import CacheConfig, DocumentsConfig, EmbedFlowConfig, IndexConfig, MigrationConfig, ModelConfig
from embedflow.indexes import MilvusDocumentStore, MilvusIndex
from embedflow.models import EmbeddingModel
from embedflow.serving.engine import MigrationEngine


def _client(uri: str, database: str):
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:
        raise SystemExit('Milvus validation requires: python -m pip install "embedflow[milvus]"') from exc
    return MilvusClient(uri=uri, db_name=database or "default"), DataType


def _wait_loaded(client: Any, collection: str, timeout: float = 120.0) -> None:
    client.load_collection(collection_name=collection)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get_load_state(collection_name=collection).get("state")
        name = str(getattr(state, "name", state)).lower()
        if name.endswith("loaded") or str(state) == "Loaded":
            return
        time.sleep(0.25)
    raise RuntimeError(f"Milvus collection {collection!r} did not become loaded")


def _wait_rows(client: Any, collection: str, expected: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            count = int(client.get_collection_stats(collection_name=collection).get("row_count", 0))
        except Exception:
            count = 0
        if count >= expected:
            return
        time.sleep(0.25)
    raise RuntimeError(f"Milvus collection {collection!r} expected {expected} rows")


def _create_collection(client: Any, DataType: Any, name: str, *, dimension: int, metric: str,
                       index_type: str = "FLAT", id_type: Any | None = None,
                       vectors: tuple[str, ...] = ("embedding",), text: bool = True) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    actual_id_type = id_type or DataType.INT64
    id_kwargs: dict[str, Any] = {"field_name": "id", "datatype": actual_id_type, "is_primary": True}
    if actual_id_type == DataType.VARCHAR:
        id_kwargs["max_length"] = 256
    schema.add_field(**id_kwargs)
    if text:
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
    for field_name in vectors:
        schema.add_field(field_name=field_name, datatype=DataType.FLOAT_VECTOR, dim=dimension)
    params = client.prepare_index_params()
    for field_name in vectors:
        if index_type == "HNSW":
            params.add_index(field_name=field_name, index_type="HNSW", metric_type=metric,
                             params={"M": 16, "efConstruction": 100})
        elif index_type == "IVF_FLAT":
            params.add_index(field_name=field_name, index_type="IVF_FLAT", metric_type=metric,
                             params={"nlist": 16})
        else:
            params.add_index(field_name=field_name, index_type="FLAT", metric_type=metric)
    client.create_collection(collection_name=name, schema=schema, index_params=params)


def _insert(client: Any, collection: str, rows: list[dict[str, Any]], *, partition: str | None = None,
            expected: int | None = None) -> None:
    kwargs = {"collection_name": collection, "data": rows}
    if partition:
        kwargs["partition_name"] = partition
    client.insert(**kwargs)
    client.flush(collection_name=collection)
    if expected is not None:
        _wait_rows(client, collection, expected)


def _tiny_vectors(metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype="float32")
    query = np.asarray([1.0, 0.25], dtype="float32")
    if metric == "COSINE":
        score = values @ query / (np.linalg.norm(values, axis=1) * np.linalg.norm(query))
        order = np.argsort(-score, kind="stable")
    elif metric == "IP":
        order = np.argsort(-(values @ query), kind="stable")
    else:
        order = np.argsort(np.sum((values - query) ** 2, axis=1), kind="stable")
    return values, query, [str(int(x)) for x in order]


class _DeterministicModel(EmbeddingModel):
    def __init__(self, model_id: str, dimension: int, *, query_value: np.ndarray | None = None):
        self.model_id = model_id
        self.dimension = dimension
        self.fingerprint = f"audit-{model_id}-{dimension}"
        self.query_value = np.asarray(query_value if query_value is not None else np.ones(dimension), dtype="float32")

    def encode_queries(self, texts: list[str]):
        return np.repeat(self.query_value[None, :], len(texts), axis=0)

    def encode_documents(self, texts: list[str], batch_size: int | None = None):
        rows = []
        for text in texts:
            value = np.zeros(self.dimension, dtype="float32")
            # Python's process-randomized ``hash`` would make this fixture
            # differ between validation runs.  Use a stable digest so target
            # materialization is reproducible across processes.
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            value[int.from_bytes(digest[:8], "little") % self.dimension] = 1.0
            rows.append(value)
        return np.asarray(rows, dtype="float32")

    def close(self) -> None:
        return None


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(args: argparse.Namespace) -> dict[str, Any]:
    client, DataType = _client(args.uri, args.database)
    temporary: list[str] = []
    checks: dict[str, Any] = {}
    try:
        print("[milvus-validation] connecting to read-only 10k fixture", flush=True)
        # The caller's existing 10k collection is strictly read-only here.
        source = MilvusIndex.connect(uri=args.uri, database=args.database, collection=args.collection,
                                     dimension=args.dimension, id_field="id", vector_field="embedding",
                                     text_field="content", metric="cosine", search_params={"ef": 64})
        before_schema = source.metadata()
        before_stats = client.get_collection_stats(collection_name=args.collection)
        before_sample = client.get(collection_name=args.collection, ids=[0, 1, 42], output_fields=["id", "embedding", "content"])
        for index_type in ("HNSW", "IVF_FLAT"):
            print(f"[milvus-validation] top-k matrix {index_type}", flush=True)
            index = source if index_type == "HNSW" else MilvusIndex.connect(
                uri=args.uri, database=args.database, collection=args.ivf_collection,
                dimension=args.dimension, id_field="id", vector_field="embedding", text_field="content",
                metric="cosine", search_params={"nprobe": 16})
            meta = index.metadata()
            _assert(meta["index_type"].upper() == index_type, f"expected {index_type}, got {meta['index_type']}")
            rows = []
            for k in (1, 10, 50, 100, 500, 1000, 10001):
                print(f"[milvus-validation] {index_type} search k={k}", flush=True)
                hits = index.search(np.eye(args.dimension, dtype="float32")[0], k)
                _assert(len(hits) <= k, f"{index_type} returned >K")
                _assert(len({hit.document_id for hit in hits}) == len(hits), f"{index_type} duplicate IDs")
                _assert(all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)), f"{index_type} score order")
                rows.append({"k": k, "returned": len(hits)})
            checks[index_type.lower()] = {"metadata": meta, "top_k": rows,
                                          "text": index.fetch_documents(["0", "42"])}
            if index is not source:
                index.close()
        checks["10k_fixture"] = {"rows": int(before_stats.get("row_count", 0)), "dimension": source.dimension}
        _assert(checks["10k_fixture"]["rows"] == 10000, "10k fixture row count mismatch")
        checks["source_snapshot_before"] = {"metadata": before_schema, "stats": before_stats,
                                             "sample": [dict(row) for row in before_sample]}
        # Exact metric checks use disposable FLAT collections and independent
        # NumPy rankings.  The adapter must preserve Milvus's ordering and
        # negate only L2 distances.
        metric_checks: dict[str, Any] = {}
        for metric in ("COSINE", "IP", "L2"):
            print(f"[milvus-validation] exact metric {metric}", flush=True)
            name = f"embedflow_milvus_metric_{uuid.uuid4().hex[:10]}"
            temporary.append(name)
            _create_collection(client, DataType, name, dimension=2, metric=metric, index_type="FLAT")
            values, query, expected = _tiny_vectors(metric)
            _insert(client, name, [{"id": i, "content": f"metric {i}", "embedding": values[i].tolist()} for i in range(len(values))], expected=len(values))
            _wait_loaded(client, name)
            index = MilvusIndex.connect(uri=args.uri, database=args.database, collection=name, dimension=2,
                                        id_field="id", vector_field="embedding", text_field="content", metric=metric)
            got = index.search(query, 4)
            _assert([hit.document_id for hit in got] == expected, f"{metric} ranking mismatch")
            _assert(all(got[i].score >= got[i + 1].score for i in range(len(got) - 1)), f"{metric} score direction")
            metric_checks[metric.lower()] = {"ids": [hit.document_id for hit in got], "scores": [hit.score for hit in got]}
            index.close()
        checks["metrics"] = metric_checks
        # INT64 and VARCHAR IDs, plus partition isolation.
        ids_checks: dict[str, Any] = {}
        for id_type, values in ((DataType.INT64, [0, 42, 2**40]), (DataType.VARCHAR, ["42", "uuid-✓", "a/b"])):
            print(f"[milvus-validation] id fixture {getattr(id_type, 'name', id_type)}", flush=True)
            name = f"embedflow_milvus_ids_{uuid.uuid4().hex[:10]}"
            temporary.append(name)
            _create_collection(client, DataType, name, dimension=2, metric="COSINE", id_type=id_type)
            _insert(client, name, [{"id": value, "content": f"id text {value}", "embedding": [1.0, 0.0]} for value in values], expected=len(values))
            _wait_loaded(client, name)
            index = MilvusIndex.connect(uri=args.uri, database=args.database, collection=name, dimension=2,
                                        id_field="id", vector_field="embedding", text_field="content", metric="cosine")
            got = index.search(np.asarray([1.0, 0.0], dtype="float32"), len(values))
            fetched = index.fetch_documents([str(value) for value in values])
            _assert(set(fetched) == {str(value) for value in values}, f"{id_type} text resolution")
            ids_checks[str(getattr(id_type, "name", id_type))] = {"ids": [hit.document_id for hit in got], "fetched": fetched}
            index.close()
        checks["ids"] = ids_checks
        # Two vector fields must be selected explicitly and produce different
        # results.  Omitting the field is rejected as ambiguous.
        multi = f"embedflow_milvus_multi_{uuid.uuid4().hex[:10]}"
        print("[milvus-validation] multiple vector fields", flush=True)
        temporary.append(multi)
        _create_collection(client, DataType, multi, dimension=2, metric="COSINE", vectors=("embedding_a", "embedding_b"), text=False)
        _insert(client, multi, [{"id": 1, "embedding_a": [1.0, 0.0], "embedding_b": [0.0, 1.0]},
                                {"id": 2, "embedding_a": [0.0, 1.0], "embedding_b": [1.0, 0.0]}], expected=2)
        _wait_loaded(client, multi)
        a = MilvusIndex.connect(uri=args.uri, database=args.database, collection=multi, dimension=2,
                                id_field="id", vector_field="embedding_a", text_field=None, metric="cosine", documents={"1": "a", "2": "b"})
        b = MilvusIndex.connect(uri=args.uri, database=args.database, collection=multi, dimension=2,
                                id_field="id", vector_field="embedding_b", text_field=None, metric="cosine", documents={"1": "a", "2": "b"})
        _assert(a.search(np.asarray([1.0, 0.0], dtype="float32"), 1)[0].document_id == "1", "embedding_a selection")
        _assert(b.search(np.asarray([1.0, 0.0], dtype="float32"), 1)[0].document_id == "2", "embedding_b selection")
        try:
            ambiguous = MilvusIndex(client, collection=multi, dimension=2, id_field="id", vector_field=None, text_field=None,
                                     metric="cosine", documents={"1": "a", "2": "b"})
            ambiguous._introspect()
        except ValueError as exc:
            _assert("vector_field" in str(exc), "ambiguous vector field error")
        else:
            raise AssertionError("ambiguous vector field was accepted")
        a.close(); b.close(); checks["multiple_vector_fields"] = "PASS"
        # Partition restrictions are passed through to both search and get.
        part = f"embedflow_milvus_part_{uuid.uuid4().hex[:10]}"
        print("[milvus-validation] partitions/load state", flush=True)
        temporary.append(part)
        _create_collection(client, DataType, part, dimension=2, metric="COSINE")
        client.create_partition(collection_name=part, partition_name="A")
        client.create_partition(collection_name=part, partition_name="B")
        _insert(client, part, [{"id": 1, "content": "A", "embedding": [1.0, 0.0]}], partition="A", expected=1)
        _insert(client, part, [{"id": 2, "content": "B", "embedding": [1.0, 0.0]}], partition="B", expected=2)
        _wait_loaded(client, part)
        restricted = MilvusIndex.connect(uri=args.uri, database=args.database, collection=part, dimension=2,
                                         id_field="id", vector_field="embedding", text_field="content", metric="cosine",
                                         partition_names=["A"])
        _assert([hit.document_id for hit in restricted.search(np.asarray([1.0, 0.0], dtype="float32"), 10)] == ["1"], "partition isolation")
        _assert(set(restricted.fetch_documents(["1"])) == {"1"}, "partition text resolution")
        restricted.close(); checks["partition_isolation"] = "PASS"
        # Load-state policy: false is actionable, true explicitly loads.
        unloaded = MilvusIndex.connect(uri=args.uri, database=args.database, collection=part, dimension=2,
                                       id_field="id", vector_field="embedding", text_field="content", metric="cosine")
        # The connection is loaded by setup; release it through the fixture
        # client, then use an injected client to avoid a second control path.
        client.release_collection(collection_name=part)
        try:
            unloaded._loaded_state = None
            unloaded._ensure_loaded()
        except RuntimeError as exc:
            _assert("not loaded" in str(exc), "unloaded collection error")
        else:
            raise AssertionError("unloaded collection unexpectedly searched")
        loaded = MilvusIndex.connect(uri=args.uri, database=args.database, collection=part, dimension=2,
                                     id_field="id", vector_field="embedding", text_field="content", metric="cosine", auto_load=True)
        _assert(loaded.search(np.asarray([1.0, 0.0], dtype="float32"), 1), "auto-load search")
        loaded.close(); unloaded.close(); checks["load_state"] = "PASS"
        # Shared migration engine uses the exact same MilvusIndex and has no
        # source writes.  Cache/restart is included here to exercise COLD,
        # PARTIAL, WARM and persistent target vectors.
        source = MilvusIndex.connect(uri=args.uri, database=args.database, collection=args.collection, dimension=args.dimension,
                                     id_field="id", vector_field="embedding", text_field="content", metric="cosine", search_params={"ef": 64})
        docs = MilvusDocumentStore(source, owns_index=False)
        cache_root = Path(args.cache).expanduser().resolve()
        cache = SQLiteVectorCache(cache_root, "milvus-audit-target", args.dimension)
        cfg = EmbedFlowConfig(source=ModelConfig("milvus-audit-source", dimension=args.dimension),
                              target=ModelConfig("milvus-audit-target", dimension=args.dimension),
                              index=IndexConfig(backend="milvus", path="./legacy.index", uri=args.uri, collection=args.collection,
                                                id_field="id", vector_field="embedding", text_field="content", metric="cosine"),
                              documents=DocumentsConfig(path="./documents.jsonl"),
                              migration=MigrationConfig(candidate_depth=10, kmax_probe=20, max_sync_misses=2, background_batch_size=8),
                              cache=CacheConfig(path=str(cache_root)), state_path=str(cache_root / "state.json"))
        cfg.validate()
        source_model = _DeterministicModel("source", args.dimension, query_value=np.eye(args.dimension, dtype="float32")[0])
        target_model = _DeterministicModel("target", args.dimension, query_value=np.eye(args.dimension, dtype="float32")[0])
        engine = MigrationEngine(cfg, source_model, target_model, source, cache, docs, start_worker=False)
        print("[milvus-validation] migration cold/partial/warm", flush=True)
        cold = engine.search("fixed", top_k=5, candidate_depth=10, max_sync_misses=0)
        partial = engine.search("fixed", top_k=5, candidate_depth=10, max_sync_misses=2)
        candidate_ids = [hit.document_id for hit in source.search(source_model.encode_query("fixed"), 10)]
        engine.prewarm(candidate_ids, asynchronous=False)
        warm = engine.search("fixed", top_k=5, candidate_depth=10, max_sync_misses=0)
        _assert(cold["migration"]["status"] == "COLD", "migration cold")
        _assert(partial["migration"]["status"] == "PARTIAL", "migration partial")
        _assert(warm["migration"]["status"] == "WARM", "migration warm")
        engine.close(close_models=False, close_indexes=False)
        source_model.close(); target_model.close()
        source2 = MilvusIndex.connect(uri=args.uri, database=args.database, collection=args.collection, dimension=args.dimension,
                                      id_field="id", vector_field="embedding", text_field="content", metric="cosine", search_params={"ef": 64})
        docs2 = MilvusDocumentStore(source2, owns_index=False)
        cache2 = SQLiteVectorCache(cache_root, "milvus-audit-target", args.dimension)
        source_model2 = _DeterministicModel("source", args.dimension, query_value=np.eye(args.dimension, dtype="float32")[0])
        target_model2 = _DeterministicModel("target", args.dimension, query_value=np.eye(args.dimension, dtype="float32")[0])
        engine2 = MigrationEngine(cfg, source_model2, target_model2, source2, cache2, docs2, start_worker=False)
        print("[milvus-validation] migration restart", flush=True)
        restarted = engine2.search("fixed", top_k=5, candidate_depth=10, max_sync_misses=0)
        _assert(restarted["migration"]["status"] == "WARM", "cache restart")
        engine2.close(close_models=False, close_indexes=False); source_model2.close(); target_model2.close()
        checks["migration"] = {"cold": cold["migration"], "partial": partial["migration"], "warm": warm["migration"],
                                "restart": restarted["migration"]}
        after_stats = client.get_collection_stats(collection_name=args.collection)
        after_sample = client.get(collection_name=args.collection, ids=[0, 1, 42], output_fields=["id", "embedding", "content"])
        _assert(before_stats == after_stats, "source row count changed")
        _assert([dict(row) for row in before_sample] == [dict(row) for row in after_sample], "source sample changed")
        checks["source_immutability"] = "PASS"
        source.close()
        return checks
    finally:
        for name in temporary:
            try:
                if client.has_collection(collection_name=name):
                    client.drop_collection(collection_name=name)
            except Exception as exc:
                print(f"cleanup warning for {name}: {exc}", flush=True)
        try:
            client.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real local Milvus adapter validation")
    parser.add_argument("--uri", default=os.environ.get("EMBEDFLOW_MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--database", default=os.environ.get("EMBEDFLOW_MILVUS_DATABASE", "default"))
    parser.add_argument("--collection", default=os.environ.get("EMBEDFLOW_MILVUS_COLLECTION", "embedflow_milvus_10k"))
    parser.add_argument("--ivf-collection", default=os.environ.get("EMBEDFLOW_MILVUS_IVF_COLLECTION", "embedflow_milvus_10k_ivf"))
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--cache", default="/tmp/embedflow-milvus-validation-cache")
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args)
    result["duration_seconds_diagnostic_only"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
