#!/usr/bin/env python3
"""Create a deterministic disposable Milvus fixture for integration tests.

This setup utility is intentionally separate from ``MilvusIndex``. It is the
only project code that creates a collection, inserts entities, or creates an
index. Production EmbedFlow operations remain read-only.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _client(uri: str, database: str):
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:
        raise SystemExit('Milvus fixture requires: python -m pip install "embedflow[milvus]"') from exc
    return MilvusClient(uri=uri, db_name=database or "default"), DataType


def create_fixture(uri: str, collection: str, *, rows: int = 10_000, dimension: int = 64,
                   metric: str = "COSINE", index_type: str = "HNSW", database: str = "default",
                   batch_size: int = 500) -> dict[str, object]:
    if rows < 1 or dimension < 1:
        raise ValueError("rows and dimension must be positive")
    client, DataType = _client(uri, database)
    if client.has_collection(collection_name=collection):
        raise RuntimeError(f"Milvus fixture collection {collection!r} already exists; choose a unique name (no implicit drop)")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_type = index_type.upper()
    if index_type == "HNSW":
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type=metric,
                               params={"M": 16, "efConstruction": 100})
    elif index_type in {"IVF_FLAT", "IVFFLAT"}:
        index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type=metric,
                               params={"nlist": 128})
        index_type = "IVF_FLAT"
    elif index_type == "FLAT":
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type=metric)
    else:
        raise ValueError("index_type must be HNSW, IVF_FLAT, or FLAT")
    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
    rng = np.random.default_rng(20260912)
    vectors = rng.normal(size=(rows, dimension)).astype("float32")
    if metric.upper() == "COSINE":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for start in range(0, rows, batch_size):
        end = min(rows, start + batch_size)
        payload = [{"id": int(i), "content": f"deterministic Milvus document {i}", "embedding": vectors[i].tolist()}
                   for i in range(start, end)]
        client.insert(collection_name=collection, data=payload)
    client.flush(collection_name=collection)
    # Insert is asynchronous on some deployments. Poll stats, then load.
    deadline = time.monotonic() + 180
    count = 0
    while time.monotonic() < deadline:
        try:
            stats = client.get_collection_stats(collection_name=collection)
            count = int(stats.get("row_count", 0))
        except Exception:
            count = 0
        if count >= rows:
            break
        time.sleep(1)
    if count < rows:
        raise RuntimeError(f"Milvus fixture visibility timed out: expected {rows}, observed {count}")
    client.load_collection(collection_name=collection)
    load_deadline = time.monotonic() + 180
    while time.monotonic() < load_deadline:
        state = client.get_load_state(collection_name=collection).get("state")
        if str(getattr(state, "name", state)).lower().endswith("loaded") or str(state) == "Loaded":
            break
        time.sleep(1)
    return {"uri": uri, "database": database, "collection": collection, "rows": rows,
            "dimension": dimension, "metric": metric.upper(), "index_type": index_type, "count": count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic Milvus integration fixture")
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--database", default="default")
    parser.add_argument("--collection", default="embedflow_milvus_10k")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--metric", default="COSINE", choices=["COSINE", "IP", "L2"])
    parser.add_argument("--index-type", default="HNSW", choices=["HNSW", "IVF_FLAT", "FLAT"])
    args = parser.parse_args()
    print(create_fixture(args.uri, args.collection, rows=args.rows, dimension=args.dimension,
                         metric=args.metric, index_type=args.index_type, database=args.database))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
