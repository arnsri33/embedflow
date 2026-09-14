#!/usr/bin/env python3
"""Non-mutating Weaviate smoke check for an existing collection."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from embedflow.indexes import WeaviateIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only EmbedFlow Weaviate smoke test")
    parser.add_argument("--uri", default="http://127.0.0.1:8080")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--grpc-host")
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--vector-name")
    parser.add_argument("--text-property", default="content")
    parser.add_argument("--tenant")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    index = None
    try:
        index = WeaviateIndex.connect(uri=args.uri, http_port=args.http_port, grpc_host=args.grpc_host,
                                      grpc_port=args.grpc_port, collection=args.collection, dimension=args.dimension,
                                      metric=args.metric, vector_name=args.vector_name,
                                      text_property=args.text_property, tenant=args.tenant)
        query = np.zeros(args.dimension, dtype="float32")
        query[0] = 1.0
        hits = index.search(query, args.top_k)
        print(json.dumps({"health": index.health_check(), "metadata": index.metadata(),
                          "hits": [hit.__dict__ for hit in hits]}, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"EmbedFlow Weaviate smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()


if __name__ == "__main__":
    raise SystemExit(main())
