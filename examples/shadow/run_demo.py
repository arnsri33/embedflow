"""Deterministic, offline Shadow Mode demonstration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow the example to be launched as ``python examples/shadow/run_demo.py``
# from a source checkout without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embedflow.cache import SQLiteVectorCache
from embedflow.config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    RuntimeConfig,
    ShadowConfig,
    ShadowTelemetryConfig,
    TelemetryConfig,
    save_config,
)
from embedflow.indexes import NumpyIndex
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel
from embedflow.serving.engine import MigrationEngine
from embedflow.shadow import render_shadow_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    root = args.path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    documents_path = root / "documents.jsonl"
    rows = [{"id": f"doc-{i}", "text": f"offline shadow document {i % 8}"} for i in range(64)]
    documents_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    docs = DocumentStore(documents_path)
    source = HashEmbeddingModel("embedflow/demo-shadow-source", 64)
    target = HashEmbeddingModel("embedflow/demo-shadow-target", 64)
    vectors = source.encode_documents([row["text"] for row in rows])
    index = NumpyIndex.build(vectors, [row["id"] for row in rows], documents=docs.documents)
    cfg = EmbedFlowConfig(
        source=ModelConfig(source.model_id, dimension=source.dimension),
        target=ModelConfig(target.model_id, dimension=target.dimension),
        index=IndexConfig(path=str(root / "source.index")),
        documents=DocumentsConfig(path=str(documents_path)),
        migration=MigrationConfig(candidate_depth=12, kmax_probe=20, max_sync_misses=0, background_batch_size=8),
        cache=CacheConfig(path=str(root / "target-cache")),
        telemetry=TelemetryConfig(latency_log=str(root / "latency.jsonl")),
        state_path=str(root / "state.json"),
        runtime=RuntimeConfig(mode="shadow"),
        shadow=ShadowConfig(enabled=True, sample_rate=1.0, candidate_k=12, materialize=True,
                            max_inflight=2, queue_capacity=32, timeout_ms=2000,
                            telemetry=ShadowTelemetryConfig(path=str(root / "shadow.sqlite3"))),
    )
    # Persist a self-contained, secret-free config so the CLI report can be
    # exercised against the same deterministic telemetry artifact.
    save_config(cfg, root / "embedflow.yaml")
    cache = SQLiteVectorCache(cfg.cache.path, target.fingerprint, target.dimension)
    engine = MigrationEngine(cfg, source, target, index, cache, docs, start_worker=True)
    try:
        result = engine.search("offline shadow document 3", top_k=5, request_id="demo-request")
        print("Source response returned:")
        print(json.dumps({"ids": [row["id"] for row in result["results"]],
                          "source_authoritative": result["migration"]["source_authoritative"]}, indent=2))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            report = engine.shadow_telemetry.report(queue=engine.shadow_runner.stats(), candidate_k=12, sample_rate=1.0)
            if report["traffic"]["shadow_completed_total"] + report["traffic"]["shadow_partial_total"] >= 1:
                break
            time.sleep(0.02)
        # Once the bounded worker has warmed the candidate set, the next
        # source-authoritative request observes complete target coverage.
        warm_deadline = time.monotonic() + 3
        while time.monotonic() < warm_deadline and engine.cache.stats()["cached_target_vectors"] < 12:
            time.sleep(0.02)
        engine.search("offline shadow document 3", top_k=5, request_id="demo-warm")
        time.sleep(0.1)
        report = engine.shadow_telemetry.report(queue=engine.shadow_runner.stats(), candidate_k=12, sample_rate=1.0)
        print(render_shadow_report(report))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
