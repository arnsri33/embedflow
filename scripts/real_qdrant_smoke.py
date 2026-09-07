#!/usr/bin/env python3
"""Exercise a real local Qdrant migration with retained public model snapshots.

This is a release smoke test, not a benchmark.  It intentionally uses a tiny
synthetic corpus and only local model files supplied with ``--model-root``.
No network, Qdrant Cloud account, or credentials are required.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from embedflow.config import ModelConfig
from embedflow.indexes import QdrantIndex
from embedflow.migration.facade import migrate
from embedflow.migration.state import DocumentStore
from embedflow.models import load_embedding_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, help="directory containing minilm_l6/ and qwen3_0_6b/")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    model_root = Path(args.model_root).expanduser().resolve()
    work = Path(tempfile.mkdtemp(prefix="embedflow-real-qdrant-"))
    rows = [{"id": f"doc-{i}", "text": f"aurora retrieval migration topic {i % 4}"} for i in range(24)]
    documents_path = work / "documents.jsonl"
    documents_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    documents = DocumentStore(documents_path)

    source = load_embedding_model(
        # Known model IDs hydrate the same frozen research contracts used by
        # the registry; model_root resolves the staged local snapshots.
        ModelConfig("sentence-transformers/all-MiniLM-L6-v2", local_path=str(model_root / "minilm_l6")),
        model_root=model_root,
        device=args.device,
    )
    target = load_embedding_model(
        ModelConfig("Qwen/Qwen3-Embedding-0.6B", local_path=str(model_root / "qwen3_0_6b")),
        model_root=model_root,
        device=args.device,
    )
    try:
        storage = work / "legacy-qdrant"
        built = QdrantIndex.build(
            storage,
            "legacy",
            source.encode_documents([row["text"] for row in rows]),
            [row["id"] for row in rows],
            documents=documents.documents,
        )
        # Local Qdrant uses an exclusive storage lock.  The migration facade
        # opens its own client, so the builder must be closed first.
        built.close()
        with migrate(
            index=storage,
            old_model=source,
            new_model=target,
            documents=documents_path,
            backend="qdrant",
            collection="legacy",
            cache_path=work / "cache",
            state_path=work / "state.json",
            candidate_depth=10,
            kmax_probe=20,
            start_worker=False,
        ) as session:
            query = "what explains aurora migration"
            cold = session.search(query, top_k=3, max_sync_misses=0)
            partial = session.search(query, top_k=3, max_sync_misses=2)
            candidate_ids = [hit.document_id for hit in session.engine.source_index.search(source.encode_query(query), 10)]
            session.prewarm(candidate_ids, asynchronous=False)
            warm = session.search(query, top_k=3, max_sync_misses=0)
            payload = {
                "backend": session.engine.source_index.metadata(),
                "models": {"source": source.model_id, "target": target.model_id},
                "states": {"cold": cold["migration"], "partial": partial["migration"], "warm": warm["migration"]},
                "warm_result_ids": [row["id"] for row in warm["results"]],
                "note": "Tiny synthetic corpus; functional lifecycle smoke test, not a benchmark.",
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        # The facade does not close caller-owned model instances.
        source.close()
        target.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
