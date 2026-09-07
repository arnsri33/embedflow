"""Hostile, small fixtures for the public release surface.

These tests intentionally exercise malformed inputs and boundary conditions,
not just the normal examples.  They avoid downloading model weights or
requiring a running external vector database.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from embedflow.cache import SQLiteVectorCache
from embedflow.compatibility.candidate_gap import compute_candidate_gap_curve
from embedflow.compatibility.containment import candidate_containment
from embedflow.compatibility.metrics import ndcg, paired_bootstrap, recall
from embedflow.compatibility.migration_depth import observed_migration_depth, recommend_initial_k
from embedflow.compatibility.t2 import diagnose_t2
from embedflow.config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    ProbeConfig,
    from_dict,
    hydrate_research_contract,
    load_config,
)
from embedflow.indexes import NumpyIndex
from embedflow.metrics import aggregate_records, summarize
from embedflow.migration.materializer import MaterializationWorker, PersistentWorkQueue
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel
from embedflow.registry import (
    MATCH_EXACT,
    MATCH_NONE,
    MATCH_PRIOR,
    MATCH_RELATED,
    EvidenceRecord,
    contract_fingerprint,
    load_benchmark_profiles,
    load_evidence,
    load_summaries,
    match_evidence,
    verify_registry,
)
from embedflow.registry.loader import dataset_fingerprint


def _safe_features(**overrides: float) -> dict[str, float]:
    values = {
        "probe_residual_tail_50_mean": 0.01,
        "deepest_p90": 100.0,
        "late_tail_area": 0.1,
        "stability_to_500_50_mean": 0.99,
        "last_shell_any_rate": 0.0,
        "fraction_margin_nonpositive": 0.0,
    }
    values.update(overrides)
    return values


def test_ndcg_relevance_and_edge_cases():
    assert ndcg(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)
    assert ndcg(["a", "a", "b"], {"a": 2, "b": 1}, 3) == pytest.approx(1.0)
    assert ndcg(["x"], {}, 10) == 0.0
    with pytest.raises(ValueError, match="positive integer"):
        ndcg(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError, match="non-negative"):
        ndcg(["a"], {"a": -1}, 1)
    with pytest.raises(ValueError, match="finite"):
        ndcg(["a"], {"a": math.nan}, 1)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg(["a"], {"a": 1}, True)


def test_recall_deduplicates_results_and_rejects_bad_qrels():
    assert recall(["a", "a", "x"], {"a": 1, "b": 1}, 3) == pytest.approx(0.5)
    assert recall([], {}, 1) == 0.0
    with pytest.raises(ValueError, match="finite"):
        recall(["a"], {"a": math.inf}, 1)
    with pytest.raises(ValueError, match="positive integer"):
        recall(["a"], {"a": 1}, -1)


def test_bootstrap_is_reproducible_and_rejects_nonfinite_or_empty():
    values = [0.1, -0.2, 0.4, 0.4]
    assert paired_bootstrap(values, seed=7, resamples=100) == paired_bootstrap(values, seed=7, resamples=100)
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap([])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap([math.nan])
    with pytest.raises(ValueError, match="positive integer"):
        paired_bootstrap(values, resamples=0)


def _gap_fixture(**kwargs):
    defaults = dict(
        source_rankings={"q": ["a", "b", "c"]},
        target_scores={"q": {"a": 0.1, "b": 0.9, "c": 0.2}},
        native_target_rankings={"q": ["b", "a", "c"]},
        qrels={"q": {"a": 2, "b": 1}},
        k_values=[1, 2, 3],
    )
    defaults.update(kwargs)
    return compute_candidate_gap_curve(**defaults)


def test_gap_supports_negative_and_nonmonotonic_values_without_assumptions():
    curve = _gap_fixture()
    # A target ranking can outperform the saved native ranking, so G(K) may be
    # negative; the evaluator must preserve the sign.
    assert any(point.candidate_gap < 0 for point in curve)
    nonmonotonic = [{"k": 100, "candidate_gap": 0.02}, {"k": 10, "candidate_gap": 0.005}, {"k": 50, "candidate_gap": 0.03}]
    assert observed_migration_depth(nonmonotonic, epsilon=0.01) == 10
    with pytest.raises(ValueError, match="duplicate"):
        observed_migration_depth([{"k": 1, "candidate_gap": 0}, {"k": 1, "candidate_gap": 0}])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("k_values", [], "at least one"),
        ("k_values", [1, 1], "duplicate"),
        ("k_values", [0], "positive"),
        ("quality_k", 0, "positive"),
        ("target_scores", {"q": {"a": math.nan, "b": 0.2, "c": 0.1}}, "non-finite"),
    ],
)
def test_gap_rejects_malformed_inputs(field, value, match):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        _gap_fixture(**kwargs)


def test_gap_rejects_missing_candidates_and_missing_query_overlap():
    with pytest.raises(ValueError, match="missing"):
        _gap_fixture(target_scores={"q": {"a": 0.1}})
    with pytest.raises(ValueError, match="no query IDs"):
        _gap_fixture(source_rankings={"other": ["a"]})


def test_containment_boundary_semantics():
    assert candidate_containment(["a", "b"], ["b", "a"], 2) == 1.0
    assert candidate_containment(["x"], ["a", "b"], 2) == 0.0
    assert candidate_containment(["a", "a"], ["a", "a"], 2) == 1.0
    assert candidate_containment(["a"], [], 1) == 0.0
    with pytest.raises(ValueError, match="positive integer"):
        candidate_containment(["a"], ["a"], 0)


def test_migration_depth_boundaries_and_recommendation():
    assert observed_migration_depth([{"k": 5, "candidate_gap": 0.01}], 0.01) == 5
    assert observed_migration_depth([{"k": 5, "candidate_gap": 0.0100001}], 0.01) is None
    assert observed_migration_depth([], 0.01) is None
    with pytest.raises(ValueError, match="positive"):
        observed_migration_depth([{"k": 1.5, "candidate_gap": 0.0}], 0.01)
    assert recommend_initial_k({"diagnostic": "SAFE", "recommended_k": 20}, default=50) == (
        20,
        "Finite-tail behavior supports this starting K; validate on production traffic.",
    )
    assert recommend_initial_k({"diagnostic": "EXPAND", "recommended_k": 20}, default=50)[0] == 50
    assert recommend_initial_k({"diagnostic": "UNSAFE_OR_UNCERTAIN"}, default=50)[0] == 50
    with pytest.raises(ValueError, match="non-negative"):
        observed_migration_depth([], -0.1)


def test_t2_golden_vocabulary_and_leakage_guards():
    assert diagnose_t2(_safe_features()).diagnostic == "SAFE"
    assert diagnose_t2(_safe_features(probe_residual_tail_50_mean=0.06)).diagnostic == "EXPAND"
    assert diagnose_t2(_safe_features(probe_residual_tail_50_mean=0.11)).diagnostic == "UNSAFE_OR_UNCERTAIN"
    assert diagnose_t2(_safe_features(probe_residual_tail_50_mean=0.06, last_shell_any_rate=0.3)).diagnostic == "UNSAFE_OR_UNCERTAIN"
    with pytest.raises(ValueError, match="label/native"):
        diagnose_t2({**_safe_features(), "qrels": 0})
    with pytest.raises(ValueError, match="missing"):
        diagnose_t2({"probe_residual_tail_50_mean": 0.1})
    with pytest.raises(ValueError, match="finite"):
        diagnose_t2(_safe_features(deepest_p90=math.inf))
    with pytest.raises(ValueError, match="boolean"):
        diagnose_t2(_safe_features(deepest_p90=True))


def test_t2_retained_golden_fixture_reproduces_frozen_decisions():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "t2_v1_golden.json").read_text())
    for case in fixture:
        assert diagnose_t2(case["features"]).diagnostic == case["diagnostic"], case["case"]


def test_registry_rows_are_golden_validated_and_consistent():
    rows = load_evidence()
    assert len(rows) == 15
    assert [row.evidence_id for row in rows] == sorted(row.evidence_id for row in rows)
    for row in rows:
        raw = row.to_dict()
        assert EvidenceRecord.from_dict(raw).to_dict() == raw
        gaps = row.candidate_gap
        if row.raw.get("observed_migration_depth") is not None:
            expected = min((k for k, value in gaps.items() if value <= row.epsilon), default=None)
            assert expected == row.raw["observed_migration_depth"]
        assert row.raw["provenance"]["artifact_sha256"] and len(row.raw["provenance"]["artifact_sha256"]) == 64
    assert verify_registry()["ok"] is True
    assert len(load_summaries()) == 2
    assert len(load_benchmark_profiles()) == 3
    # External research artifacts are intentionally not shipped in the public
    # repository.  A maintainer can opt into byte-level provenance checks with
    # EMBEDFLOW_PROVENANCE_ROOT; a clean stranger checkout should still run
    # the packaged-registry test without assuming our workstation layout.
    provenance_root = os.environ.get("EMBEDFLOW_PROVENANCE_ROOT")
    if not provenance_root:
        pytest.skip("retained research artifacts are optional; set EMBEDFLOW_PROVENANCE_ROOT to audit them")
    retained = verify_registry(provenance_root=Path(provenance_root).expanduser())
    assert retained["ok"] is True
    assert retained["provenance_artifacts_checked"] >= 20
    assert retained["semantic_values_checked"] >= 500


def test_benchmark_profile_rejects_nonfinite_measurements():
    profile = load_benchmark_profiles()[0].to_dict()
    profile["measurements"]["p50_ms"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        type(load_benchmark_profiles()[0]).from_dict(profile)


def test_registry_schema_rejects_tampered_contract_or_curve():
    row = load_evidence()[0].to_dict()
    row["source"]["max_length"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        EvidenceRecord.from_dict(row)
    row = load_evidence()[0].to_dict()
    row["candidate_depths"] = [10, 20]
    with pytest.raises(ValueError, match="candidate_depths"):
        EvidenceRecord.from_dict(row)
    row = load_evidence()[0].to_dict()
    row["candidate_gap"]["10"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        EvidenceRecord.from_dict(row)


def test_registry_external_provenance_hashes_are_checked(tmp_path: Path):
    source_root = tmp_path / "research"
    artifact = source_root / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("canonical\n")
    row = load_evidence()[0].to_dict()
    row["provenance"] = {
        "artifact": "artifact.csv",
        "artifact_sha256": "0" * 64,
        "status": "retained_external",
    }
    result = __import__("embedflow.registry.loader", fromlist=["verify_registry"]).verify_registry
    # Build a one-row temporary registry directory so the normal schema and
    # external-byte verification path are exercised together.
    data = tmp_path / "registry"
    data.mkdir()
    (data / "migrations.jsonl").write_text(json.dumps(row) + "\n")
    (data / "benchmark_profiles.jsonl").write_text("")
    (data / "research_summaries.json").write_text(json.dumps({"summaries": []}))
    (data / "schema_version.json").write_text(json.dumps({"schema_version": "1", "registry_version": "0.1.0"}))
    (data / "registry_manifest.json").write_text(json.dumps({"schema_version": "1", "registry_version": "0.1.0", "record_count": 1, "profile_count": 0, "summary_count": 0, "files": {}}))
    verified = result(data, provenance_root=source_root)
    assert verified["ok"] is False
    assert any("checksum mismatch" in message for message in verified["errors"])


def test_registry_semantic_verifier_fails_closed_on_malformed_artifact(tmp_path: Path):
    from embedflow.registry.loader import _verify_semantic_provenance

    artifact = tmp_path / "scale_curves.csv"
    artifact.write_text("pair,corpus_size,K,n_queries,source_ndcg,native_target_ndcg,restricted_ndcg,G,containment10\n"
                        "minilm_l6_to_qwen3_8b,100000,not-a-k,1,0,0,0,0,0\n")
    row = load_evidence()[3].to_dict()
    row["provenance"] = {"artifact": str(artifact)}
    record = EvidenceRecord.from_dict(row)
    errors: list[str] = []
    warnings: list[str] = []
    _verify_semantic_provenance(tmp_path, [record], [], [], errors, warnings)
    assert errors


def test_registry_semantic_provenance_catches_transcribed_value_drift():
    """A validly shaped row must still agree with its retained source table."""
    from embedflow.registry.loader import _verify_semantic_provenance

    provenance_root = os.environ.get("EMBEDFLOW_PROVENANCE_ROOT")
    if not provenance_root:
        pytest.skip("retained research artifacts are optional; set EMBEDFLOW_PROVENANCE_ROOT to audit them")
    row = load_evidence()[0].to_dict()
    row["candidate_gap"]["50"] = float(row["candidate_gap"]["50"]) + 0.001
    tampered = EvidenceRecord.from_dict(row)
    errors: list[str] = []
    warnings: list[str] = []
    checked = _verify_semantic_provenance(Path(provenance_root).expanduser(), [tampered], [], [], errors, warnings)
    assert checked > 0
    assert any("candidate gap disagrees" in message for message in errors)


def test_registry_matching_exact_prior_related_and_none():
    rows = load_evidence()
    source = hydrate_research_contract(ModelConfig("Qwen/Qwen3-Embedding-4B"))
    target = hydrate_research_contract(ModelConfig("Qwen/Qwen3-Embedding-8B"))
    exact = match_evidence(source_model=source, target_model=target, corpus_name="beir_nq_1m_prefix_1000000", corpus_size=1_000_000, records=rows)
    assert exact.level == MATCH_EXACT and exact.exact_corpus
    prior = match_evidence(source_model=source, target_model=target, corpus_name="customer-corpus", corpus_size=123, records=rows)
    assert prior.level == MATCH_PRIOR and not prior.exact_corpus
    altered = hydrate_research_contract(ModelConfig("Qwen/Qwen3-Embedding-8B", max_length=1024))
    related = match_evidence(source_model=source, target_model=altered, records=rows)
    assert related.level == MATCH_RELATED and related.exact_target_contract is False
    family = match_evidence(source_model="Qwen/Qwen3-Embedding-7B", target_model="Qwen/Qwen3-Embedding-8B", records=rows)
    assert family.level == MATCH_RELATED and family.records
    none = match_evidence(source_model="unseen/source", target_model="unseen/target", records=rows)
    assert none.level == MATCH_NONE and not none.records


def test_registry_matching_consumes_generators_once():
    rows = load_evidence()
    result = match_evidence(source_model="Qwen3-4B", target_model="Qwen3-8B", records=(row for row in rows))
    assert result.level == MATCH_RELATED
    assert result.records


def test_contract_fingerprint_changes_for_every_relevant_field():
    base = {
        "canonical_model_id": "demo/model", "revision": "r1", "dimension": 3,
        "max_length": 8, "pooling": "mean_tokens", "padding_side": "right",
        "truncation_side": "right", "query_instruction": "", "document_instruction": "",
        "normalization": "l2", "dtype": "float32",
    }
    for key, value in {
        "canonical_model_id": "demo/other", "revision": "r2", "dimension": 4,
        "max_length": 9, "pooling": "last_token", "padding_side": "left",
        "truncation_side": "left", "query_instruction": "Q:{text}",
        "document_instruction": "D:{text}", "normalization": "none", "dtype": "float16",
    }.items():
        changed = dict(base, **{key: value})
        assert contract_fingerprint(changed) != contract_fingerprint(base), key
    assert contract_fingerprint(base) == contract_fingerprint(dict(base))


def test_dataset_fingerprint_is_stable_and_contract_sensitive(tmp_path: Path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text('{"id": 1, "text": "alpha"}\n{"id": 2, "text": "beta"}\n')
    first = dataset_fingerprint(docs)
    assert first == dataset_fingerprint(docs)
    docs.write_text('{"id": 2, "text": "beta"}\n{"id": 1, "text": "alpha"}\n')
    assert first != dataset_fingerprint(docs)
    docs.write_text('{"id": 1, "text": "alpha"}\nnot-json\n')
    with pytest.raises(ValueError, match="invalid JSONL"):
        dataset_fingerprint(docs)


def test_sqlite_cache_torture_and_checksum_detection(tmp_path: Path):
    cache = SQLiteVectorCache(tmp_path / "cache", "fp-a", 3)
    try:
        cache.put(["a"], np.array([[1, 2, 3]], dtype="float32"))
        with pytest.raises(ValueError, match="duplicate"):
            cache.put(["a", "a"], np.ones((2, 3), dtype="float32"))
        with pytest.raises(ValueError, match="shape"):
            cache.put(["b"], np.ones((1, 2), dtype="float32"))
        with pytest.raises(ValueError, match="finite"):
            cache.put(["b"], np.array([[math.nan, 1, 2]], dtype="float32"))
        assert len(cache.get(["a"] * 1200)) == 1  # exercises SQLite variable chunking
        db = sqlite3.connect(cache.db_path)
        db.execute("UPDATE target_vectors SET vector=? WHERE document_id='a'", (sqlite3.Binary(np.ones(3, dtype="float32").tobytes()),))
        db.commit(); db.close()
        with pytest.raises(RuntimeError, match="checksum"):
            cache.get(["a"])
    finally:
        cache.close()
    reopened = SQLiteVectorCache(tmp_path / "cache", "fp-b", 3)
    try:
        assert reopened.get(["a"]) == {}
    finally:
        reopened.close()


def test_persistent_queue_dedup_retry_and_restart(tmp_path: Path):
    path = tmp_path / "queue.sqlite3"
    queue = PersistentWorkQueue(path, max_retries=2)
    assert queue.enqueue(["a", "a", "b"]) == 2
    with pytest.raises(ValueError, match="positive integer"):
        queue.claim(0)
    first = queue.claim(10)
    assert first == ["a", "b"]
    queue.fail(["a"], "transient")
    queue.complete(["b"])
    assert queue.stats() == {"pending": 1, "processing": 0, "done": 1, "error": 0}
    second = queue.claim(1)
    assert second == ["a"]
    queue.fail(second, "permanent")
    assert queue.stats()["error"] == 1
    queue.close()
    reopened = PersistentWorkQueue(path, max_retries=2)
    try:
        assert reopened.stats()["error"] == 1
    finally:
        reopened.close()


def test_materialization_worker_isolates_missing_documents_and_retries(tmp_path: Path):
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text('{"id":"a","text":"alpha"}\n{"id":"b","text":"beta"}\n')
    docs = DocumentStore(docs_path)
    model = HashEmbeddingModel("worker-target", 4)
    cache = SQLiteVectorCache(tmp_path / "cache", model.fingerprint, model.dimension)
    worker = MaterializationWorker(model, docs, cache, tmp_path / "queue.sqlite3", batch_size=2, max_retries=2)
    try:
        assert worker.enqueue(["a", "missing", "b", "a"]) == 3
        worker.start()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            state = worker.stats()["queue"]
            if state["done"] == 2 and state["error"] == 1:
                break
            time.sleep(0.03)
        state = worker.stats()["queue"]
        assert state["done"] == 2 and state["error"] == 1
        assert cache.contains(["a", "b"]) == {"a", "b"}
        assert cache.contains(["missing"]) == set()
    finally:
        worker.close()
        worker.close()  # shutdown is idempotent
        with pytest.raises(RuntimeError, match="closed"):
            worker.start()
        with pytest.raises(RuntimeError, match="closed"):
            worker.enqueue(["a"])
        model.close()
        cache.close()


def test_numpy_index_rejects_malformed_inputs_and_is_deterministic():
    with pytest.raises(ValueError, match="non-empty 2-D"):
        NumpyIndex(np.array([]), [])
    with pytest.raises(ValueError, match="duplicate"):
        NumpyIndex(np.eye(2, dtype="float32"), ["a", "a"])
    with pytest.raises(ValueError, match="finite"):
        NumpyIndex(np.array([[math.nan]], dtype="float32"), ["a"])
    index = NumpyIndex.build(np.eye(2, dtype="float32"), ["b", "a"], metric="dot")
    with pytest.raises(ValueError, match="positive integer"):
        index.search(np.array([1, 0], dtype="float32"), 0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1, 0, 0], dtype="float32"), 1)
    # Ties preserve insertion order, making warm results reproducible.
    tied = NumpyIndex.build(np.array([[1, 0], [1, 0]], dtype="float32"), ["first", "second"], metric="dot")
    assert [hit.document_id for hit in tied.search(np.array([1, 0], dtype="float32"), 2)] == ["first", "second"]


def test_numpy_index_roundtrip_and_corrupt_artifacts(tmp_path: Path):
    path = tmp_path / "legacy.index"
    built = NumpyIndex.build(np.eye(3, dtype="float32"), ["a", "b", "c"], path=path, metric="cosine")
    loaded = NumpyIndex.load(path)
    assert loaded.size() == built.size() and loaded.dimension == 3
    path.write_bytes(b"not-an-index")
    with pytest.raises(ValueError, match="invalid NumPy"):
        NumpyIndex.load(path)


def test_faiss_matrix_when_dependency_is_available(tmp_path: Path):
    pytest.importorskip("faiss")
    from embedflow.indexes import FaissIndex

    path = tmp_path / "legacy.faiss"
    FaissIndex.build(np.eye(4, dtype="float32"), ["a", "b", "c", "d"], path=path, nprobe=1)
    loaded = FaissIndex.load(path, nprobe=1)
    assert loaded.search(np.array([1, 0, 0, 0], dtype="float32"), 20)[0].document_id == "a"
    with pytest.raises(ValueError, match="positive integer"):
        FaissIndex.build(np.eye(2, dtype="float32"), ["a", "b"], nprobe=0)
    path.with_suffix(path.suffix + ".ids.json").write_text("not-json")
    with pytest.raises(ValueError, match="IDs sidecar"):
        FaissIndex.load(path)


def test_engine_rejects_invalid_search_bounds_and_double_close(tmp_path: Path):
    from embedflow.serving.engine import MigrationEngine

    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text("\n".join(json.dumps({"id": f"d{i}", "text": f"topic {i}"}) for i in range(3)) + "\n")
    docs = DocumentStore(docs_path)
    source = HashEmbeddingModel("source", 4)
    target = HashEmbeddingModel("target", 4)
    index = NumpyIndex.build(source.encode_documents(list(docs.documents.values())), list(docs.documents), documents=docs.documents)
    cfg = EmbedFlowConfig(source=ModelConfig("source", dimension=4), target=ModelConfig("target", dimension=4),
                          index=IndexConfig(path=str(tmp_path / "index")), documents=DocumentsConfig(path=str(docs_path)),
                          migration=MigrationConfig(candidate_depth=2, kmax_probe=10),
                          cache=CacheConfig(path=str(tmp_path / "cache")), state_path=str(tmp_path / "state.json"))
    cache = SQLiteVectorCache(cfg.cache.path, target.fingerprint, 4)
    engine = MigrationEngine(cfg, source, target, index, cache, docs, start_worker=False)
    try:
        for kwargs, message in [({"top_k": 0}, "top_k"), ({"candidate_depth": 0}, "candidate_depth"), ({"max_sync_misses": -1}, "max_sync_misses")]:
            with pytest.raises(ValueError, match=message):
                engine.search("topic", **kwargs)
    finally:
        engine.close()
        engine.close()


def test_api_request_models_reject_bad_types_or_bounds():
    pytest.importorskip("pydantic")
    from embedflow.serving.schemas import PrewarmRequest, SearchRequest

    with pytest.raises(Exception):
        SearchRequest(query="", top_k=1)
    with pytest.raises(Exception):
        SearchRequest(query="ok", top_k=0)
    with pytest.raises(Exception):
        SearchRequest(query="ok", top_k=101)
    request = PrewarmRequest(document_ids=["a", "a"], asynchronous=False)
    assert request.document_ids == ["a", "a"]


def test_public_analyze_migration_api_delegates_to_same_probe(tmp_path: Path):
    from embedflow import analyze_migration
    from embedflow.runtime import build_faiss_from_documents

    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text("\n".join(json.dumps({"id": f"d{i}", "text": f"topic {i % 3}"}) for i in range(12)) + "\n")
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text('{"id":"q1","text":"topic 1"}\n')
    docs = DocumentStore(docs_path)
    source = HashEmbeddingModel("embedflow/demo-source", 8)
    cfg = EmbedFlowConfig(
        source=ModelConfig("embedflow/demo-source", dimension=8),
        target=ModelConfig("embedflow/demo-target", dimension=8),
        index=IndexConfig(path=str(tmp_path / "legacy.index")),
        documents=DocumentsConfig(path=str(docs_path)),
        probe=ProbeConfig(queries=str(queries_path), k_values=[10], kmax=10, limit=1),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        state_path=str(tmp_path / "state.json"),
    )
    build_faiss_from_documents(cfg, source, docs)
    source.close()
    report = analyze_migration(cfg, probe_queries=queries_path, demo=True, output_dir=tmp_path / "report")
    assert report["diagnostic"] in {"SAFE", "EXPAND", "UNSAFE_OR_UNCERTAIN"}
    assert report["registry"]["level"] in {"NO REGISTRY MATCH", "RELATED EVIDENCE ONLY"}
    assert (tmp_path / "report" / "migration_report.json").exists()


def test_config_fuzz_like_invalid_values_are_explained():
    base = {
        "source": {"model": "embedflow/source", "dimension": 8},
        "target": {"model": "embedflow/target", "dimension": 8},
    }
    cases = [
        ({"index": {"backend": "pinecone"}}, "backend"),
        ({"migration": {"candidate_depth": 0}}, "candidate_depth"),
        ({"migration": {"background_batch_size": 0}}, "background_batch_size"),
        ({"probe": {"k_values": [10, 10]}}, "duplicates"),
        ({"probe": {"k_values": [5000], "kmax": 10}}, "exceed"),
        ({"source": {"model": "embedflow/source", "dimension": 8, "max_length": 0}}, "max_length"),
        ({"source": {"model": "same", "dimension": 8}, "target": {"model": "same", "dimension": 8}}, "differ"),
    ]
    for override, match in cases:
        merged = json.loads(json.dumps(base))
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        with pytest.raises((ValueError, TypeError), match=match):
            from_dict(merged)


def test_model_config_semantic_fingerprint_ignores_deployment_only_fields():
    one = ModelConfig("demo", dimension=8, local_path="/a", device="cpu")
    two = ModelConfig("demo", dimension=8, local_path="/b", device="cuda")
    assert one.fingerprint == two.fingerprint
    assert ModelConfig("demo", dimension=8, padding_side="left").fingerprint != one.fingerprint


def test_latency_helpers_reject_bad_rows_and_keep_percentiles_finite():
    assert summarize([0, 0])['qps'] == 0.0
    with pytest.raises(ValueError, match="finite"):
        summarize([math.inf])
    with pytest.raises(ValueError, match="non-negative"):
        summarize([-1])
    rows = [{"mode": "x", "K": 1, "nprobe": 1, "total_ms": 2.0}, {"mode": "x", "K": 1, "nprobe": 1, "total_ms": "bad"}]
    aggregate = aggregate_records(rows + [None])
    assert aggregate[0]["total_ms_count"] == 1
    assert aggregate[0]["malformed_rows_skipped"] == 2


def test_hash_model_empty_and_large_batch_contract():
    model = HashEmbeddingModel("demo", 4)
    assert model.encode_documents([]).shape == (0, 4)
    vectors = model.encode_documents([str(i) for i in range(5000)], batch_size=37)
    assert vectors.shape == (5000, 4) and np.isfinite(vectors).all()
    model.close()


def test_registry_schema_fails_closed_on_non_objects_and_bad_keys():
    with pytest.raises(ValueError, match="evidence row must be an object"):
        EvidenceRecord.from_dict([])  # type: ignore[arg-type]
    row = load_evidence()[0].to_dict()
    row["containment"] = {"not-a-depth": 0.5}
    with pytest.raises(ValueError, match="invalid K"):
        EvidenceRecord.from_dict(row)
    profile = load_benchmark_profiles()[0].to_dict()
    profile["profile_id"] = ""
    with pytest.raises(ValueError, match="profile_id"):
        type(load_benchmark_profiles()[0]).from_dict(profile)


def test_config_environment_override_errors_are_actionable(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "source:\n  model: embedflow/source\n  dimension: 8\n"
        "target:\n  model: embedflow/target\n  dimension: 8\n"
    )
    monkeypatch.setenv("EMBEDFLOW_CANDIDATE_DEPTH", "not-an-integer")
    with pytest.raises(ValueError, match="EMBEDFLOW_CANDIDATE_DEPTH"):
        load_config(config_path)


def test_closed_cache_and_queue_fail_explicitly(tmp_path: Path):
    cache = SQLiteVectorCache(tmp_path / "cache", "fp", 2)
    cache.close(); cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.stats()
    queue = PersistentWorkQueue(tmp_path / "queue.sqlite3")
    queue.close(); queue.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue.stats()
