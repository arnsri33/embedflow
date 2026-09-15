"""Focused contract tests for traffic-aware target-vector prewarming."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from embedflow.cache import SQLiteVectorCache
from embedflow.config import from_dict
from embedflow.models import HashEmbeddingModel
from embedflow.prewarm import PrewarmPlan, PrewarmPlanner, PrewarmRunner
from embedflow.shadow import ShadowTelemetry


class Documents:
    def __init__(self, values: dict[str, str]):
        self.documents = dict(values)

    def get(self, ids):
        ids = [str(value) for value in ids]
        missing = [value for value in ids if value not in self.documents]
        if missing:
            raise KeyError(missing)
        return {value: self.documents[value] for value in ids}

    def size(self):
        return len(self.documents)


def _fixture(tmp_path: Path, counts: dict[str, int], *, fp: str = "migration-fp"):
    values = {document_id: f"document text {document_id}" for document_id in counts}
    docs = Documents(values)
    telemetry = ShadowTelemetry(tmp_path / "shadow.sqlite3", config_fingerprint=fp, max_records=1000)
    timestamp = 1_000_000.0
    for document_id, count in counts.items():
        for _ in range(count):
            telemetry.record_candidate_documents([document_id], timestamp=timestamp, config_fingerprint=fp)
    target = HashEmbeddingModel("embedflow/prewarm-target", 4)
    cache = SQLiteVectorCache(tmp_path / "cache", target.fingerprint, target.dimension)
    planner = PrewarmPlanner(
        telemetry, cache, config_fingerprint=fp, source_fingerprint="source-fp",
        target_fingerprint=target.fingerprint, backend="faiss", index_identity="fixture-index",
        candidate_k=10, target_dimension=target.dimension, documents=docs,
    )
    return docs, telemetry, target, cache, planner


def _close(*resources):
    for resource in resources:
        close = getattr(resource, "close", None)
        if callable(close):
            close()


def test_traffic_hotset_cache_awareness_and_coverage(tmp_path: Path):
    docs, telemetry, target, cache, planner = _fixture(tmp_path, {"A": 1000, "B": 700, "C": 100, "D": 10})
    try:
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=2)
        assert plan.selected_ids == ["A", "B"]
        assert plan.baseline["candidate_occurrences"] == 1810
        assert plan.selection["selected_candidate_occurrences"] == 1700
        assert plan.selection["estimated_total_candidate_coverage"] == pytest.approx(1700 / 1810)
        assert all(isinstance(item, dict) for item in plan.to_dict()["warnings"])

        text = docs.documents["A"]
        cache.put(["A"], np.ones((1, 4), dtype="float32"),
                  content_fingerprints={"A": cache.content_fingerprint(text)})
        cached_plan = planner.plan(start=999_000, end=1_001_000, max_docs=2)
        assert cached_plan.selected_ids == ["B", "C"]
        assert cached_plan.baseline["already_cached_docs"] == 1
        assert cached_plan.baseline["observed_candidate_coverage"] == pytest.approx(1000 / 1810)
    finally:
        _close(telemetry, cache)


def test_ties_windows_and_configuration_segments_are_deterministic(tmp_path: Path):
    docs, telemetry, target, cache, planner = _fixture(tmp_path, {"z": 100, "a": 100, "b": 100})
    try:
        tied = planner.plan(start=999_000, end=1_001_000, max_docs=2)
        assert tied.selected_ids == ["a", "b"]

        telemetry.record_candidate_documents(["z"], timestamp=2_000_000, config_fingerprint="other-fp")
        telemetry.record_candidate_documents(["b"], timestamp=2_000_000, config_fingerprint="migration-fp")
        current = planner.plan(start=1_999_000, end=2_001_000, max_docs=1)
        assert current.selected_ids == ["b"]

        other = PrewarmPlanner(
            telemetry, cache, config_fingerprint="other-fp", source_fingerprint="source-fp",
            target_fingerprint=target.fingerprint, backend="faiss", index_identity="fixture-index",
            candidate_k=10, target_dimension=4, documents=docs,
        ).plan(start=1_999_000, end=2_001_000, max_docs=1)
        assert other.selected_ids == ["z"]
    finally:
        _close(telemetry, cache)


def test_plan_round_trip_and_fingerprint_reject_tampering(tmp_path: Path):
    _, telemetry, _, cache, planner = _fixture(tmp_path, {"A": 3})
    try:
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=1)
        round_trip = PrewarmPlan.from_dict(json.loads(plan.to_json()))
        assert round_trip.fingerprint == plan.fingerprint
        yaml_path = tmp_path / "plan.yaml"
        yaml_path.write_text(plan.to_yaml(), encoding="utf-8")
        from embedflow.prewarm import load_prewarm_plan
        assert load_prewarm_plan(yaml_path).fingerprint == plan.fingerprint
        raw = json.loads(plan.to_json())
        raw["selection"]["selected_ids"] = ["tampered"]
        with pytest.raises(ValueError, match="fingerprint"):
            PrewarmPlan.from_dict(raw)
    finally:
        _close(telemetry, cache)


def test_content_fingerprint_makes_changed_document_cold(tmp_path: Path):
    docs, telemetry, target, cache, planner = _fixture(tmp_path, {"A": 4, "B": 2})
    try:
        cache.put(["A"], np.ones((1, 4), dtype="float32"),
                  content_fingerprints={"A": cache.content_fingerprint(docs.documents["A"])})
        docs.documents["A"] = "changed document text A"
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=1)
        assert plan.selected_ids == ["A"]
    finally:
        _close(telemetry, cache)


def test_read_only_inspection_does_not_create_or_mutate_state(tmp_path: Path):
    target = HashEmbeddingModel("embedflow/prewarm-read-only", 4)
    cache_path = tmp_path / "missing-cache"
    telemetry_path = tmp_path / "missing-shadow.sqlite3"
    from embedflow.cache import SQLiteVectorCache

    cache = SQLiteVectorCache(cache_path, target.fingerprint, target.dimension, read_only=True)
    telemetry = ShadowTelemetry(telemetry_path, config_fingerprint="fp", read_only=True)
    try:
        assert cache.stats()["cached_target_vectors"] == 0
        assert telemetry.document_stats() == []
        assert not cache_path.exists()
        assert not telemetry_path.exists()
    finally:
        cache.close(); telemetry.close()


def test_prewarmer_accepts_mapping_document_store(tmp_path: Path):
    docs, telemetry, target, cache, planner = _fixture(tmp_path, {"A": 3})
    try:
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=1)
        runner = PrewarmRunner(
            target, dict(docs.documents), cache, state_dir=tmp_path / "prewarm-mapping",
            source_fingerprint="source-fp", target_fingerprint=target.fingerprint,
            backend="faiss", index_identity="fixture-index", candidate_k=10,
            config_fingerprint="migration-fp",
        )
        assert runner.run(plan)["status"] == "COMPLETED"
    finally:
        _close(telemetry, cache)


def test_cache_update_without_content_metadata_preserves_existing_binding(tmp_path: Path):
    target = HashEmbeddingModel("embedflow/prewarm-content-preserve", 4)
    cache = SQLiteVectorCache(tmp_path / "cache", target.fingerprint, target.dimension)
    try:
        fingerprint = cache.content_fingerprint("original")
        cache.put(["A"], np.ones((1, 4), dtype="float32"), content_fingerprints={"A": fingerprint})
        cache.put(["A"], np.zeros((1, 4), dtype="float32"))
        assert cache.content_fingerprints(["A"])["A"] == fingerprint
        assert "A" not in cache.peek(["A"], content_fingerprints={"A": cache.content_fingerprint("changed")})
    finally:
        cache.close()


def test_prewarmer_is_bounded_fingerprinted_and_idempotent(tmp_path: Path):
    docs, telemetry, target, cache, planner = _fixture(tmp_path, {"A": 3, "B": 2, "C": 1})
    try:
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=2)
        runner = PrewarmRunner(
            target, docs, cache, state_dir=tmp_path / "prewarm",
            source_fingerprint="source-fp", target_fingerprint=target.fingerprint,
            backend="faiss", index_identity="fixture-index", candidate_k=10,
            config_fingerprint="migration-fp",
        )
        first = runner.run(plan)
        assert first["status"] == "COMPLETED"
        assert first["encoded"] == 2
        assert first["remaining"] == 0
        second = runner.run(plan)
        assert second["status"] == "NOOP"
        assert second["already_warm"] == 2
        assert second["encoded"] == 0
        assert cache.stats()["cached_target_vectors"] == 2

        # A plan is reusable only while the selected document content remains
        # the same.  A changed document is stale despite the same model/ID and
        # must be requeued rather than hidden behind a durable ``done`` row.
        docs.documents[plan.selected_ids[0]] = "new content after planning"
        changed = runner.run(plan)
        assert changed["status"] == "COMPLETED"
        assert changed["encoded"] == 1

        bad = PrewarmPlan.from_dict(plan.to_dict())
        bad.migration["target_fingerprint"] = "different-target"
        with pytest.raises(ValueError, match="target model"):
            runner.validate_plan(bad)
    finally:
        _close(telemetry, cache)


def test_large_stream_keeps_selection_bounded(tmp_path: Path):
    class StreamingTelemetry:
        config_fingerprint = "stream-fp"

        def iter_document_stats(self, **kwargs):
            del kwargs
            for index in range(20_000):
                yield {"document_id": f"doc-{index:05d}", "config_fingerprint": "stream-fp",
                       "candidate_occurrences": 20_000 - index, "shadow_queries_seen": 1,
                       "cache_misses": 1, "first_seen": 1.0, "last_seen": 1.0}

        def report(self, **kwargs):
            del kwargs
            return {"traffic": {"shadow_sampled_total": 20_000}}

    target = HashEmbeddingModel("embedflow/stream-target", 4)
    cache = SQLiteVectorCache(tmp_path / "cache", target.fingerprint, target.dimension)
    try:
        planner = PrewarmPlanner(StreamingTelemetry(), cache, config_fingerprint="stream-fp",
                                 source_fingerprint="s", target_fingerprint=target.fingerprint,
                                 backend="faiss", index_identity="idx", candidate_k=10,
                                 target_dimension=4)
        plan = planner.plan(max_docs=7)
        assert len(plan.selected_ids) == 7
        assert plan.selected_ids == [f"doc-{index:05d}" for index in range(7)]
        assert plan.baseline["unique_candidate_docs"] == 20_000
    finally:
        cache.close()


def test_empty_telemetry_never_selects_random_documents(tmp_path: Path):
    _, telemetry, _, cache, planner = _fixture(tmp_path, {})
    try:
        plan = planner.plan(max_docs=10)
        assert plan.status == "NO_COMPATIBLE_TRAFFIC"
        assert plan.selected_ids == []
        assert any(item.code == "NO_COMPATIBLE_TRAFFIC" for item in plan.warnings)
    finally:
        _close(telemetry, cache)


def test_config_exposes_bounded_prewarm_defaults():
    cfg = from_dict({"source": {"model": "source", "dimension": 4},
                     "target": {"model": "target", "dimension": 4},
                     "prewarm": {"max_docs": 12, "strategy": "traffic_hotset", "batch_size": 4}})
    assert cfg.prewarm.max_docs == 12
    assert cfg.prewarm.batch_size == 4
    with pytest.raises(ValueError, match="prewarm.strategy"):
        from_dict({"source": {"model": "source", "dimension": 4},
                   "target": {"model": "target", "dimension": 4},
                   "prewarm": {"strategy": "random"}})


def test_cli_prewarmer_uses_persisted_probe_candidate_depth(tmp_path: Path):
    """Plan/report fingerprints must match the runtime's probe-selected K."""
    from embedflow.cli import _effective_shadow_candidate_k

    cfg = from_dict({"source": {"model": "source", "dimension": 4},
                     "target": {"model": "target", "dimension": 4},
                     "state_path": str(tmp_path / "state.json"),
                     "migration": {"candidate_depth": 20}})
    probe_path = tmp_path / "probe_result.json"
    probe_path.write_text(json.dumps({"diagnostic": "SAFE", "recommended_k": 7}), encoding="utf-8")
    assert _effective_shadow_candidate_k(cfg) == 7
    cfg.shadow.candidate_k = 11
    assert _effective_shadow_candidate_k(cfg) == 11


def test_traffic_hotset_property_invariants_two_thousand_examples():
    """Cheap deterministic property sweep when Hypothesis is not installed."""
    class Telemetry:
        config_fingerprint = "property-fp"

        def __init__(self, rows):
            self.rows = rows

        def iter_document_stats(self, **kwargs):
            del kwargs
            yield from self.rows

        def report(self, **kwargs):
            del kwargs
            return {"traffic": {"shadow_sampled_total": len(self.rows)}}

    class Cache:
        model_fingerprint = "target-fp"

        def __init__(self, warm):
            self.warm = set(warm)

        def peek(self, ids, **kwargs):
            del kwargs
            return {document_id: np.zeros(4, dtype="float32") for document_id in ids if document_id in self.warm}

    rng = random.Random(20260915)
    for _ in range(2_000):
        count = rng.randrange(0, 16)
        ids = [f"doc-{index}" for index in range(count)]
        rows = [{"document_id": document_id, "config_fingerprint": "property-fp",
                 "candidate_occurrences": rng.randrange(0, 10_000), "shadow_queries_seen": 1,
                 "cache_misses": 1, "first_seen": 1.0, "last_seen": 1.0} for document_id in ids]
        warm = {document_id for document_id in ids if rng.randrange(2)}
        budget = rng.randrange(0, count + 1) if count else 0
        planner = PrewarmPlanner(Telemetry(rows), Cache(warm), config_fingerprint="property-fp",
                                 source_fingerprint="source-fp", target_fingerprint="target-fp",
                                 backend="faiss", index_identity="property-index", candidate_k=10,
                                 target_dimension=4)
        plan = planner.plan(max_docs=budget, start=0.0, end=2.0)
        assert len(plan.selected_ids) <= budget
        assert len(plan.selected_ids) == len(set(plan.selected_ids))
        assert not (set(plan.selected_ids) & warm)
        assert 0.0 <= plan.baseline["observed_candidate_coverage"] <= 1.0
        assert 0.0 <= plan.selection["estimated_total_candidate_coverage"] <= 1.0
