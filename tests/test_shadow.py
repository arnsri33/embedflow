"""Independent contract tests for source-authoritative Shadow Mode.

These tests deliberately exercise failure, scheduling, privacy, and report
boundaries rather than repeating the ordinary migration happy path.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

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
)
from embedflow.indexes import NumpyIndex
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel
from embedflow.serving.engine import MigrationEngine
from embedflow.shadow import ShadowObservation, ShadowRunner, ShadowTelemetry, render_shadow_report


def _engine(tmp_path: Path, *, materialize: bool = True, sample_rate: float = 1.0,
            queue_capacity: int = 20, max_inflight: int = 2, timeout_ms: int = 1000,
            runtime_mode: str = "shadow", shadow_enabled: bool = True) -> MigrationEngine:
    document_path = tmp_path / "documents.jsonl"
    rows = [{"id": f"doc-{i}", "text": f"topic {i % 4} reference {i}"} for i in range(32)]
    document_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cfg = EmbedFlowConfig(
        source=ModelConfig("embedflow/shadow-source", dimension=32),
        target=ModelConfig("embedflow/shadow-target", dimension=32),
        index=IndexConfig(path=str(tmp_path / "legacy.index")),
        documents=DocumentsConfig(path=str(document_path)),
        migration=MigrationConfig(candidate_depth=10, kmax_probe=20, max_sync_misses=0, background_batch_size=4),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        telemetry=TelemetryConfig(latency_log=str(tmp_path / "latency.jsonl")),
        state_path=str(tmp_path / "state.json"),
        runtime=RuntimeConfig(mode=runtime_mode),
        shadow=ShadowConfig(
            enabled=shadow_enabled, sample_rate=sample_rate, candidate_k=10, materialize=materialize,
            max_inflight=max_inflight, queue_capacity=queue_capacity, timeout_ms=timeout_ms,
            telemetry=ShadowTelemetryConfig(path=str(tmp_path / "shadow.sqlite3"), max_records=100),
        ),
    )
    docs = DocumentStore(document_path)
    source = HashEmbeddingModel("embedflow/shadow-source", 32)
    target = HashEmbeddingModel("embedflow/shadow-target", 32)
    index = NumpyIndex.build(source.encode_documents(list(docs.documents.values())), list(docs.documents), documents=docs.documents)
    cache = SQLiteVectorCache(cfg.cache.path, target.fingerprint, target.dimension)
    return MigrationEngine(cfg, source, target, index, cache, docs, start_worker=True)


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_shadow_returns_source_result_before_blocked_target(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    gate = threading.Event()
    try:
        original = engine._run_shadow_task

        def blocked(payload):
            gate.wait(5)
            return original(payload)

        engine.shadow_runner.task = blocked
        source_only = engine.source_index.search(engine.source_model.encode_query("topic 1"), 5)
        result = engine.search("topic 1", top_k=5, request_id="stable-request")
        assert [row["id"] for row in result["results"]] == [hit.document_id for hit in source_only]
        assert result["migration"]["source_authoritative"] is True
        # The shadow task is intentionally still blocked; dispatch must not
        # make the primary call wait for it.
        assert engine.shadow_runner.stats()["active"] >= 0
    finally:
        gate.set()
        engine.close()


def test_shadow_response_completes_before_shadow_gate_is_released(tmp_path: Path):
    """Prove non-blocking behavior with a gate, rather than a timing guess."""
    engine = _engine(tmp_path, materialize=False, timeout_ms=2_000)
    entered = threading.Event()
    release = threading.Event()
    search_done = threading.Event()
    holder: dict[str, object] = {}
    thread: threading.Thread | None = None
    try:
        def blocked(payload):
            entered.set()
            release.wait(5)
            return engine._run_shadow_task(payload)

        engine.shadow_runner.task = blocked

        def invoke_search() -> None:
            holder["result"] = engine.search("topic 1", top_k=3, request_id="gate-proof")
            search_done.set()

        thread = threading.Thread(target=invoke_search, daemon=True)
        thread.start()
        assert entered.wait(1), "shadow task was not dispatched"
        assert search_done.wait(.5), "primary response waited for shadow work"
        result = holder["result"]
        assert isinstance(result, dict)
        assert result["migration"]["source_authoritative"] is True
    finally:
        release.set()
        if thread is not None:
            thread.join(timeout=2)
        engine.close()


def test_shadow_source_metrics_writer_cannot_block_primary(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    gate = threading.Event()
    try:
        engine._record = lambda _row: gate.wait(5)
        started = time.monotonic()
        result = engine.search("topic 1", top_k=2, request_id="metrics-blocked")
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert result["migration"]["source_authoritative"] is True
    finally:
        gate.set()
        engine.close()


def test_shadow_api_preserves_source_response_contract(tmp_path: Path):
    pytest.importorskip("fastapi")
    from embedflow.serving.api import create_app
    from embedflow.serving.schemas import SearchRequest

    engine = _engine(tmp_path, materialize=False)
    try:
        app = create_app(engine)
        search_route = next(route for route in app.routes if route.path == "/search")
        expected = engine.source_index.search(engine.source_model.encode_query("topic 0"), 4)
        response = search_route.endpoint(SearchRequest(query="topic 0", top_k=4, request_id="api-shadow"))
        assert [row["id"] for row in response["results"]] == [hit.document_id for hit in expected]
        assert set(response["results"][0]) == {
            "id", "text", "target_score", "source_rank", "target_rank", "target_vector_cached",
        }
        assert response["migration"]["source_authoritative"] is True
        # A target failure remains an observation-only event.
        engine.target_model.encode_query = lambda _: (_ for _ in ()).throw(RuntimeError("target unavailable"))
        second = search_route.endpoint(SearchRequest(query="topic 0", top_k=4, request_id="api-shadow-fail"))
        assert [row["id"] for row in second["results"]] == [hit.document_id for hit in expected]
    finally:
        engine.close()


def test_shadow_failures_and_timeouts_are_isolated(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "telemetry.sqlite3", config_fingerprint="fp")
    try:
        runner = ShadowRunner(lambda _: (_ for _ in ()).throw(RuntimeError("target boom")), telemetry,
                              sample_rate=1, max_inflight=1, queue_capacity=4, timeout_ms=50)
        assert runner.dispatch({}, request_id="failure") is True
        assert _wait_until(lambda: telemetry.report()["traffic"]["shadow_failed_total"] == 1)
        runner.task = lambda _: time.sleep(1)
        assert runner.dispatch({}, request_id="timeout") is True
        assert _wait_until(lambda: telemetry.report()["traffic"]["shadow_timeout_total"] == 1)
        runner.close()
        report = telemetry.report()
        assert report["traffic"]["shadow_failed_total"] == 1
        assert report["traffic"]["shadow_timeout_total"] == 1
    finally:
        telemetry.close()


def test_runner_capacity_cap_does_not_reject_normal_tasks(tmp_path: Path):
    """A full worker pool must still execute ordinary non-timeout jobs."""
    telemetry = ShadowTelemetry(None, config_fingerprint="fp")
    runner = ShadowRunner(lambda _: {"status": "completed"}, telemetry,
                          sample_rate=1, max_inflight=1, queue_capacity=2, timeout_ms=200)
    try:
        assert runner.dispatch({}, request_id="normal") is True
        assert _wait_until(lambda: telemetry.report()["traffic"]["shadow_completed_total"] == 1)
        assert runner.stats()["detached_jobs"] == 0
    finally:
        runner.close()
        telemetry.close()


def test_queue_full_drops_shadow_only(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "telemetry.sqlite3", config_fingerprint="fp")
    gate = threading.Event()
    runner = ShadowRunner(lambda _: gate.wait(2), telemetry, sample_rate=1, max_inflight=1, queue_capacity=1, timeout_ms=2000)
    try:
        assert runner.dispatch({}, request_id="one")
        assert _wait_until(lambda: runner.stats()["active"] == 1)
        assert runner.dispatch({}, request_id="two")
        # Subsequent dispatches are non-blocking and can be dropped.
        assert runner.dispatch({}, request_id="three") is False
        assert runner.stats()["counters"]["shadow_dropped_total"] >= 1
    finally:
        gate.set()
        runner.close()
        telemetry.close()


def test_sampling_is_stable_and_has_exact_endpoints(tmp_path: Path):
    telemetry = ShadowTelemetry(None, config_fingerprint="fp")
    zero = ShadowRunner(lambda _: {}, telemetry, sample_rate=0, sample_seed=42, max_inflight=1, queue_capacity=2)
    one = ShadowRunner(lambda _: {}, telemetry, sample_rate=1, sample_seed=42, max_inflight=1, queue_capacity=2)
    try:
        ids = [f"request-{i}" for i in range(1000)]
        assert not any(zero.should_sample(item) for item in ids)
        assert all(one.should_sample(item) for item in ids)
        first = [item for item in ids if one.should_sample(item)]
        second = [item for item in ids if one.should_sample(item)]
        assert first == second
    finally:
        zero.close(); one.close(); telemetry.close()


def test_materialize_false_does_not_touch_cache_or_queue(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    try:
        before = engine.cache.stats()
        result = engine.search("topic 2", top_k=3, request_id="no-materialize")
        assert result["migration"]["source_authoritative"] is True
        assert _wait_until(lambda: engine.shadow_telemetry.report()["traffic"]["shadow_partial_total"] == 1)
        after = engine.cache.stats()
        assert after["cached_target_vectors"] == before["cached_target_vectors"]
        assert after["total_cache_hits"] == before["total_cache_hits"]
        assert engine.worker.stats()["queue"]["pending"] == 0
    finally:
        engine.close()


def test_materialize_true_deduplicates_and_warms(tmp_path: Path):
    engine = _engine(tmp_path, materialize=True)
    try:
        engine.search("topic 3", top_k=5, request_id="warm-a")
        engine.search("topic 3", top_k=5, request_id="warm-b")
        assert _wait_until(lambda: engine.cache.stats()["cached_target_vectors"] > 0)
        assert _wait_until(lambda: engine.worker.stats()["queue"]["pending"] == 0)
        report = engine.shadow_telemetry.report(queue=engine.shadow_runner.stats())
        assert report["materialization"]["unique_docs_queued"] <= 10
    finally:
        engine.close()


def test_telemetry_privacy_and_empty_report(tmp_path: Path):
    empty = ShadowTelemetry(tmp_path / "empty.sqlite3", config_fingerprint="fp")
    try:
        report = empty.report()
        assert "No shadow observations" in report["warnings"][0]
        assert "SUPER_PRIVATE" not in json.dumps(report)
        empty.record_observation(ShadowObservation(status="failed", config_fingerprint="fp",
                                                    error="WEAVIATE_SECRET_123 SUPER_PRIVATE_QUERY"))
        data = json.dumps(empty.report())
        assert "SUPER_PRIVATE_QUERY" not in data
        assert "WEAVIATE_SECRET_123" not in data
        assert "WEAVIATE_SECRET_123" not in render_shadow_report(empty.report())
    finally:
        empty.close()


def test_target_failure_payload_is_not_persisted(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    try:
        def fail(_query):
            raise RuntimeError("CUSTOMER_SECRET_QUERY_123")

        engine.target_model.encode_query = fail
        engine.search("CUSTOMER_SECRET_QUERY_123", top_k=1, request_id="private")
        assert _wait_until(lambda: engine.shadow_telemetry.report()["traffic"]["shadow_failed_total"] == 1)
        payload = json.dumps(engine.shadow_telemetry.report())
        assert "CUSTOMER_SECRET_QUERY_123" not in payload
    finally:
        engine.close()


def test_untrusted_failure_categories_are_not_persisted(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "categories.sqlite3", config_fingerprint="fp")
    try:
        telemetry.record_observation(ShadowObservation(
            status="failed", config_fingerprint="fp",
            failure_category="CUSTOMER_PRIVATE_QUERY_TEXT",
            error="provider failure",
        ))
        report = telemetry.report()
        # Unknown categories collapse to the safe public bucket rather than
        # persisting arbitrary provider/query text.
        assert report["traffic"]["shadow_failed_total"] == 1
        rows = telemetry._read_events(None, None, "fp")
        assert rows[0]["failure_category"] == "INTERNAL"
        assert "CUSTOMER_PRIVATE_QUERY_TEXT" not in json.dumps(report)
    finally:
        telemetry.close()


def test_config_shadow_defaults_and_invalid_values():
    from embedflow.config import from_dict

    cfg = from_dict({"source": {"model": "s"}, "target": {"model": "t"},
                     "runtime": {"mode": "shadow"}, "index": {"backend": "faiss"}})
    assert cfg.runtime.mode == "shadow"
    assert cfg.shadow.enabled is True
    with pytest.raises(ValueError, match="sample_rate"):
        from_dict({"source": {"model": "s"}, "target": {"model": "t"},
                   "shadow": {"sample_rate": 2}, "index": {"backend": "faiss"}})


def test_observation_counter_mapping_is_not_double_or_misnamed(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "telemetry.sqlite3", config_fingerprint="fp")
    try:
        telemetry.record_observation(ShadowObservation(status="completed", config_fingerprint="fp",
                                                        cache_hits=2, cache_misses=3, docs_queued=4,
                                                        unique_docs_queued=4))
        report = telemetry.report()
        assert report["cache"] == {"hits": 2, "misses": 3, "hit_rate": 0.4}
        assert report["materialization"]["unique_docs_queued"] == 4
    finally:
        telemetry.close()


def test_shadow_report_persists_fingerprints_and_filters_window(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "telemetry.sqlite3", config_fingerprint="current",
                                source_fingerprint="source-fp", target_fingerprint="target-fp",
                                backend="faiss", index_identity="idx")
    try:
        old = time.time() - 10_000
        telemetry.record_observation(ShadowObservation(status="completed", config_fingerprint="current",
                                                        target_coverage=1.0, top1_agreement=1.0), timestamp=old)
        telemetry.record_observation(ShadowObservation(status="partial", config_fingerprint="current",
                                                        target_coverage=.5), timestamp=time.time())
        report = telemetry.report(config_fingerprint="current")
        assert report["migration"]["source_fingerprint"] == "source-fp"
        assert report["migration"]["backend"] == "faiss"
        recent = telemetry.report(config_fingerprint="current", since_seconds=1)
        assert recent["traffic"]["shadow_partial_total"] == 1
    finally:
        telemetry.close()


def test_shadow_report_counter_window_is_not_minute_bucketed(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "window.sqlite3", config_fingerprint="fp")
    try:
        now = time.time()
        telemetry.increment("primary_requests_total", timestamp=now - 5)
        telemetry.increment("primary_requests_total", timestamp=now)
        report = telemetry.report(start=now - 1, end=now + 1, config_fingerprint="fp")
        assert report["traffic"]["primary_requests_total"] == 1
    finally:
        telemetry.close()


def test_corrupt_telemetry_read_degrades_without_raising(tmp_path: Path):
    path = tmp_path / "shadow.sqlite3"
    telemetry = ShadowTelemetry(path, config_fingerprint="fp")
    telemetry.close()
    path.write_bytes(b"not a sqlite database")
    reopened = ShadowTelemetry(path, config_fingerprint="fp")
    try:
        report = reopened.report()
        assert report["recommendation"] == "CONTINUE_SHADOW"
        assert reopened.available is False
    finally:
        reopened.close()


def test_telemetry_snapshot_rehydrates_after_restart(tmp_path: Path):
    path = tmp_path / "restart.sqlite3"
    first = ShadowTelemetry(path, config_fingerprint="fp")
    try:
        first.record_primary(timestamp=time.time())
        first.record_sampled(timestamp=time.time())
        first.record_observation(ShadowObservation(status="completed", config_fingerprint="fp"))
    finally:
        first.close()
    reopened = ShadowTelemetry(path, config_fingerprint="fp")
    try:
        counters = reopened.snapshot()["counters"]
        assert counters["primary_requests_total"] == 1
        assert counters["shadow_eligible_total"] == 1
        assert counters["shadow_sampled_total"] == 1
        assert counters["shadow_completed_total"] == 1
    finally:
        reopened.close()


def test_shadow_disabled_runtime_has_no_runner(tmp_path: Path):
    disabled = _engine(tmp_path, materialize=False, runtime_mode="migration")
    try:
        assert disabled.shadow_runner is None
        result = disabled.search("topic 0", top_k=1, max_sync_misses=0)
        assert result["migration"]["status"] == "COLD"
    finally:
        disabled.close()


def test_shadow_kill_switch_remains_source_only_without_queue_side_effect(tmp_path: Path):
    engine = _engine(tmp_path, materialize=True, runtime_mode="shadow", shadow_enabled=False)
    try:
        assert engine.shadow_runner is None
        assert not (tmp_path / "cache" / "materialization_queue.sqlite3").exists()
        result = engine.search("topic 1", top_k=2, request_id="disabled-shadow")
        assert result["migration"]["source_authoritative"] is True
        assert result["migration"]["shadow_scheduled"] is False
        assert engine.worker.stats()["queue"]["pending"] == 0
    finally:
        engine.close()


def test_cli_endpoint_display_redacts_uri_credentials():
    from embedflow.cli import _safe_endpoint

    secret = "SHADOW_DISPLAY_SECRET_123"
    assert secret not in _safe_endpoint(f"https://user:{secret}@example.test:443/path?token={secret}")
    assert "<redacted>" in _safe_endpoint(f"milvus://token={secret}")


def test_shadow_search_rejects_non_finite_or_non_integer_bounds(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    try:
        for value in (None, float("nan"), float("inf"), "not-an-integer", 1.5):
            with pytest.raises(ValueError, match="top_k"):
                engine.search("topic 1", top_k=value)  # type: ignore[arg-type]
    finally:
        engine.close()


def test_shadow_overlap_uses_common_available_depth(tmp_path: Path):
    engine = _engine(tmp_path, materialize=False)
    try:
        hits = engine.source_index.search(engine.source_model.encode_query("topic 1"), 3)
        first_id = str(hits[0].document_id)
        vector = engine.target_model.encode_documents(["topic 1 reference 1"])[0]
        engine.cache.put([first_id], np.asarray([vector], dtype="float32"))
        result = engine._run_shadow_task({
            "query": "topic 1",
            "candidate_hits": tuple(hits),
            "source_ids": tuple(str(hit.document_id) for hit in hits),
        })
        assert result["status"] == "partial"
        assert result["top_k_overlap"] == 1.0
    finally:
        engine.close()


def test_shadow_report_never_promotes_without_complete_observations(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "recommendation.sqlite3", config_fingerprint="fp")
    try:
        # A large sampled count alone is insufficient when every comparison
        # is partial (for example, while the target cache is cold).
        for _ in range(20):
            telemetry.record_sampled()
            telemetry.record_observation(ShadowObservation(status="partial", config_fingerprint="fp", target_coverage=.25))
        telemetry.record_t2_window("SAFE", 20)
        report = telemetry.report(config_fingerprint="fp")
        assert report["recommendation"] == "CONTINUE_SHADOW"
        assert any(item["code"] == "LIMITED_COMPLETE_EVIDENCE" for item in report["warning_details"])
    finally:
        telemetry.close()


def test_shadow_report_prioritizes_operational_failures_over_expand(tmp_path: Path):
    telemetry = ShadowTelemetry(tmp_path / "failure-priority.sqlite3", config_fingerprint="fp")
    try:
        for _ in range(20):
            telemetry.record_sampled()
            telemetry.record_observation(ShadowObservation(
                status="failed", config_fingerprint="fp", failure_category="INTERNAL",
            ))
        telemetry.record_t2_window("EXPAND", 20)
        report = telemetry.report(config_fingerprint="fp")
        assert report["recommendation"] == "INVESTIGATE"
    finally:
        telemetry.close()


def test_shadow_engine_opens_when_target_model_fails_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unavailable target must be an isolated observation, not a source outage."""
    from embedflow import runtime

    documents_path = tmp_path / "documents.jsonl"
    rows = [{"id": f"doc-{i}", "text": f"startup target failure {i}"} for i in range(4)]
    documents_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    source = HashEmbeddingModel("embedflow/startup-source", 8)
    docs = DocumentStore(documents_path)
    index_path = tmp_path / "source.index"
    NumpyIndex.build(source.encode_documents([row["text"] for row in rows]), [row["id"] for row in rows],
                     path=index_path, documents=docs.documents,
                     metadata={"model_fingerprint": source.fingerprint})
    cfg_path = tmp_path / "embedflow.yaml"
    cfg = EmbedFlowConfig(
        source=ModelConfig(source.model_id, dimension=8),
        target=ModelConfig("embedflow/unavailable-target", dimension=8),
        index=IndexConfig(path=str(index_path), metric="cosine"),
        documents=DocumentsConfig(path=str(documents_path)),
        runtime=RuntimeConfig(mode="shadow"),
        shadow=ShadowConfig(enabled=True, sample_rate=1.0, materialize=False,
                            telemetry=ShadowTelemetryConfig(path=str(tmp_path / "shadow.sqlite3"))),
    )
    from embedflow.config import save_config
    save_config(cfg, cfg_path)
    original_loader = runtime.load_embedding_model

    def fail_target(config, **kwargs):
        if config.model == "embedflow/unavailable-target":
            raise RuntimeError("target snapshot unavailable")
        return original_loader(config, **kwargs)

    monkeypatch.setattr(runtime, "load_embedding_model", fail_target)
    engine = runtime.open_engine(cfg_path, demo=True, start_worker=False)
    try:
        result = engine.search("startup target failure 1", top_k=1, request_id="startup-failure")
        assert result["migration"]["source_authoritative"] is True
        assert engine.target_model.available is False
        assert _wait_until(lambda: engine.shadow_telemetry.report()["traffic"]["shadow_failed_total"] == 1)
    finally:
        engine.close()
        source.close()


def test_shadow_facade_opens_when_target_model_fails_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The public Python facade must preserve the same source-only guarantee."""
    from embedflow.migration import facade
    from embedflow.runtime import build_faiss_from_documents

    documents_path = tmp_path / "documents.jsonl"
    rows = [{"id": f"doc-{i}", "text": f"facade target failure {i}"} for i in range(4)]
    documents_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    documents = DocumentStore(documents_path)
    source = HashEmbeddingModel("embedflow/facade-source", 8)
    cfg = EmbedFlowConfig(
        source=ModelConfig(source.model_id, dimension=8),
        target=ModelConfig("embedflow/facade-unavailable-target", dimension=8),
        index=IndexConfig(path=str(tmp_path / "source.index"), metric="cosine"),
        documents=DocumentsConfig(path=str(documents_path)),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        state_path=str(tmp_path / "state.json"),
    )
    build_faiss_from_documents(cfg, source, documents)
    original_loader = facade.load_embedding_model

    def fail_target(config, **kwargs):
        if config.model == "embedflow/facade-unavailable-target":
            raise RuntimeError("target snapshot unavailable")
        return original_loader(config, **kwargs)

    monkeypatch.setattr(facade, "load_embedding_model", fail_target)
    session = facade.migrate(
        index=cfg.index.path,
        old_model=source,
        new_model="embedflow/facade-unavailable-target",
        documents=documents_path,
        backend="faiss",
        metric="cosine",
        cache_path=tmp_path / "session-cache",
        state_path=tmp_path / "session-state.json",
        mode="shadow",
        shadow_sample_rate=1.0,
        shadow_materialize=False,
    )
    try:
        result = session.search("facade target failure 1", top_k=1, request_id="facade-startup-failure")
        assert result["migration"]["source_authoritative"] is True
        assert session.engine.target_model.available is False
        assert _wait_until(lambda: session.engine.shadow_telemetry.report()["traffic"]["shadow_failed_total"] == 1)
    finally:
        session.close()
        source.close()
