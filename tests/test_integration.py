from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embedflow.cache import SQLiteVectorCache
from embedflow.config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    TelemetryConfig,
)
from embedflow.indexes import NumpyIndex
from embedflow.metrics import summarize
from embedflow.migration.planner import make_plan
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel
from embedflow.serving.engine import MigrationEngine


def _engine(tmp_path: Path) -> MigrationEngine:
    document_path = tmp_path / "documents.jsonl"
    rows = [{"id": f"doc-{i}", "text": f"topic {i % 4} reference {i}"} for i in range(32)]
    document_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cfg = EmbedFlowConfig(
        source=ModelConfig("embedflow/source", dimension=32),
        target=ModelConfig("embedflow/target", dimension=32),
        index=IndexConfig(path=str(tmp_path / "legacy.index")),
        documents=DocumentsConfig(path=str(document_path)),
        migration=MigrationConfig(candidate_depth=10, kmax_probe=20, max_sync_misses=2, background_batch_size=4),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        telemetry=TelemetryConfig(latency_log=str(tmp_path / "latency.jsonl")),
        state_path=str(tmp_path / "state.json"),
    )
    docs = DocumentStore(document_path)
    source = HashEmbeddingModel("embedflow/source", 32)
    target = HashEmbeddingModel("embedflow/target", 32)
    index = NumpyIndex.build(source.encode_documents(list(docs.documents.values())), list(docs.documents), documents=docs.documents)
    cache = SQLiteVectorCache(cfg.cache.path, target.fingerprint, target.dimension)
    return MigrationEngine(cfg, source, target, index, cache, docs, start_worker=False)


def test_progressive_search_is_cold_then_warm_and_deterministic(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        cold = engine.search("topic 1", top_k=5, max_sync_misses=0)
        assert cold["migration"]["status"] == "COLD"
        assert cold["migration"]["target_cache_misses"] == 10
        partial = engine.search("topic 1", top_k=5, max_sync_misses=2)
        assert partial["migration"]["status"] == "PARTIAL"
        assert partial["migration"]["sync_encoded"] == 2
        assert partial["migration"]["target_vectors_available"] == 2
        candidate_ids = [hit.document_id for hit in engine.source_index.search(engine.source_model.encode_query("topic 1"), 10)]
        engine.prewarm(candidate_ids, asynchronous=False)
        warm = engine.search("topic 1", top_k=5, max_sync_misses=0)
        assert warm["migration"]["status"] == "WARM"
        assert warm["migration"]["target_cache_hits"] == 10
        assert all(row["target_vector_cached"] for row in warm["results"])
        assert [row["id"] for row in warm["results"]] == [row["id"] for row in engine.search("topic 1", top_k=5)["results"]]
    finally:
        engine.close()
    with __import__("pytest").raises(RuntimeError, match="closed"):
        engine.search("topic 1")


def test_api_routes_expose_status_search_and_metrics(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from embedflow.serving.api import create_app
    from embedflow.serving.schemas import SearchRequest

    engine = _engine(tmp_path)
    try:
        app = create_app(engine)
        routes = {route.path: route for route in app.routes}
        assert routes["/health"].endpoint() == {"status": "ok"}
        response = routes["/search"].endpoint(SearchRequest(query="topic 2", top_k=3, max_sync_misses=0))
        assert response["migration"]["status"] == "COLD"
        assert routes["/analyze"].endpoint()["ann_status"] == "UNKNOWN"
        assert routes["/metrics"].endpoint()["latency"]["count"] == 1
    finally:
        engine.close()


def test_qdrant_local_backend_preserves_document_ids(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("qdrant_client")
    from embedflow.indexes import QdrantIndex

    ids = ["doc-0", "doc-with/slash", "doc-2"]
    documents = {document_id: f"text for {document_id}" for document_id in ids}
    index = QdrantIndex.build(tmp_path / "qdrant", "legacy", np.eye(3, dtype="float32"), ids, documents=documents)
    try:
        assert index.metadata()["backend"] == "qdrant"
        assert index.search(np.array([1, 0, 0], dtype="float32"), 2)[0].document_id == "doc-0"
        assert index.fetch_documents(["doc-with/slash"])["doc-with/slash"] == "text for doc-with/slash"
    finally:
        index.close()


def test_qdrant_persisted_collection_can_be_reopened_after_close(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("qdrant_client")
    from embedflow.indexes import QdrantIndex

    path = tmp_path / "qdrant"
    ids = ["a", "b"]
    documents = {"a": "alpha", "b": "beta"}
    built = QdrantIndex.build(path, "legacy", np.eye(2, dtype="float32"), ids, documents=documents)
    built.close()
    reopened = QdrantIndex.connect(str(path), "legacy", 2, documents=documents)
    try:
        assert reopened.size() == 2
        assert reopened.search(np.array([1, 0], dtype="float32"), 1)[0].document_id == "a"
    finally:
        reopened.close()
    with pytest.raises(RuntimeError, match="closed"):
        reopened.search(np.array([1, 0], dtype="float32"), 1)


def test_qdrant_rejects_zero_norm_cosine_vectors(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("qdrant_client")
    from embedflow.indexes import QdrantIndex

    with pytest.raises(ValueError, match="non-zero"):
        QdrantIndex.build(tmp_path / "qdrant", "legacy", np.array([[0.0, 0.0]], dtype="float32"), ["a"])


def test_qdrant_named_vector_roundtrip_and_dimension_inference(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("qdrant_client")
    from embedflow.indexes import QdrantIndex

    path = tmp_path / "qdrant-named"
    built = QdrantIndex.build(path, "legacy", np.eye(2, dtype="float32"), ["a", "b"], vector_name="text")
    built.close()
    reopened = QdrantIndex.connect(str(path), "legacy", 0, vector_name="text")
    try:
        assert reopened.dimension == 2
        assert reopened.search(np.array([1, 0], dtype="float32"), 1)[0].document_id == "a"
    finally:
        reopened.close()


def test_economics_and_latency_are_explicit_projections():
    from embedflow.cli import economics_for

    result = economics_for(1_000_000_000, 106.98, 3.29, cached=100)
    assert result["status"] == "ESTIMATE"
    assert result["estimated_full_backfill_gpu_hours"] > result["estimated_remaining_gpu_hours"]
    assert "projection" in result["note"].lower()
    assert summarize([1.0, 2.0])["p50_ms"] == 1.5


def test_plan_never_turns_unknown_ann_into_pass():
    plan = make_plan(target_model="target", corpus_size=10, diagnostic="SAFE", recommended_k=50,
                     ann_status="UNKNOWN", cached_target_vectors=0)
    assert plan.status == "READY"
    assert plan.ann_status == "UNKNOWN"
    assert any("ANN fidelity" in warning for warning in plan.warnings)
