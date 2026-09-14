#!/usr/bin/env python3
"""Create a deterministic disposable externally-vectorized Weaviate fixture.

This script is test/setup code.  The production ``WeaviateIndex`` never calls
collection creation or data mutation APIs.
"""

from __future__ import annotations

import argparse
import time
import uuid
from urllib.parse import urlparse

import numpy as np


def _client(uri: str, http_port: int, grpc_port: int):
    try:
        import weaviate
    except ImportError as exc:
        raise SystemExit('Weaviate fixture requires: python -m pip install "embedflow[weaviate]"') from exc
    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    host = parsed.hostname or uri
    if parsed.port is not None:
        http_port = parsed.port
    # connect_to_local is deliberately used for the standard Docker ports;
    # connect_to_custom keeps this script useful with remapped ports.
    if http_port == 8080 and grpc_port == 50051:
        return weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
    return weaviate.connect_to_custom(http_host=host, http_port=http_port, http_secure=False,
                                      grpc_host=host, grpc_port=grpc_port, grpc_secure=False)


def create_fixture(*, uri: str = "127.0.0.1", http_port: int = 8080, grpc_port: int = 50051,
                   collection_name: str = "Documents", rows: int = 10_000,
                   dimension: int = 64, metric: str = "cosine", seed: int = 20260913,
                   batch_size: int = 200) -> dict[str, object]:
    if rows < 1 or dimension < 1:
        raise ValueError("rows and dimension must be positive")
    if not collection_name or not collection_name[0].isupper():
        raise ValueError("Weaviate collection names must start with an uppercase letter")
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances
    from weaviate.classes.data import DataObject

    metric_key = metric.strip().lower()
    distance = {"cosine": VectorDistances.COSINE, "dot": VectorDistances.DOT,
                "inner_product": VectorDistances.DOT, "l2": VectorDistances.L2_SQUARED,
                "euclidean": VectorDistances.L2_SQUARED}.get(metric_key)
    if distance is None:
        raise ValueError("metric must be cosine, dot, or l2")
    client = _client(uri, http_port, grpc_port)
    try:
        if client.collections.exists(collection_name):
            raise RuntimeError(f"fixture collection {collection_name!r} already exists; choose a unique name")
        config = Configure.Vectors.self_provided(
            name="default", vector_index_config=Configure.VectorIndex.hnsw(distance_metric=distance, ef=64)
        )
        collection = client.collections.create(
            collection_name,
            vector_config=config,
            properties=[Property(name="content", data_type=DataType.TEXT),
                        Property(name="category", data_type=DataType.TEXT),
                        Property(name="fixture_marker", data_type=DataType.TEXT)],
        )
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(rows, dimension)).astype("float32")
        if metric_key == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.maximum(norms, np.finfo("float32").eps)
        namespace = uuid.UUID("b4be0e6b-8f6d-5d1c-9bb7-104c8406f6d4")
        inserted = 0
        for start in range(0, rows, batch_size):
            end = min(rows, start + batch_size)
            objects = [DataObject(
                uuid=uuid.uuid5(namespace, f"doc-{i}"),
                properties={"content": f"deterministic Weaviate document {i}",
                            "category": f"category-{i % 8}", "fixture_marker": "embedflow-weaviate-fixture"},
                vector=vectors[i].tolist(),
            ) for i in range(start, end)]
            result = collection.data.insert_many(objects)
            errors = getattr(result, "errors", {}) or {}
            if errors:
                raise RuntimeError(f"Weaviate fixture batch failed: {len(errors)} objects")
            inserted = end
        deadline = time.monotonic() + 180
        observed = 0
        while time.monotonic() < deadline:
            observed = int(collection.aggregate.over_all(total_count=True).total_count or 0)
            if observed >= rows:
                break
            time.sleep(0.5)
        if observed < rows:
            raise RuntimeError(f"fixture visibility timed out: expected {rows}, observed {observed}")
        return {"collection": collection_name, "rows": observed, "dimension": dimension,
                "metric": metric_key, "seed": seed, "inserted": inserted}
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic Weaviate test collection")
    parser.add_argument("--uri", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--collection", default="Documents")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--metric", choices=["cosine", "dot", "l2"], default="cosine")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    print(create_fixture(uri=args.uri, http_port=args.http_port, grpc_port=args.grpc_port,
                         collection_name=args.collection, rows=args.rows, dimension=args.dimension,
                         metric=args.metric, seed=args.seed, batch_size=args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
