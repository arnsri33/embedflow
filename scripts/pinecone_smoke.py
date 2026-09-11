#!/usr/bin/env python3
"""Read-only smoke test for an existing Pinecone index.

This helper intentionally has no setup or cleanup operations: it never calls
create_index, upsert, update, delete, or namespace administration APIs.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from embedflow.indexes import PineconeIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Pinecone EmbedFlow smoke test")
    parser.add_argument("--host", required=True, help="existing Pinecone data-plane host")
    parser.add_argument("--index-name")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--api-key-env", default="PINECONE_API_KEY")
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "dot", "dotproduct", "inner_product", "l2", "euclidean"])
    parser.add_argument("--query-vector", help="optional JSON array; no query is issued when omitted")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        index = PineconeIndex.connect(
            host=args.host,
            index_name=args.index_name,
            api_key_env=args.api_key_env,
            namespace=args.namespace,
            dimension=args.dimension,
            metric=args.metric,
        )
        try:
            print(json.dumps(index.metadata(), indent=2, ensure_ascii=False, default=str))
            print(json.dumps(index.health_check(), indent=2, ensure_ascii=False, default=str))
            if args.query_vector is not None:
                vector = np.asarray(json.loads(args.query_vector), dtype="float32")
                print(json.dumps([hit.__dict__ for hit in index.search(vector, args.top_k)], indent=2, ensure_ascii=False))
        finally:
            index.close()
    except Exception as exc:
        print(f"EmbedFlow Pinecone smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
