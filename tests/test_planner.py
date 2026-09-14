from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

import embedflow.cli as cli
from embedflow import plan as public_plan
from embedflow.config import CacheConfig, DocumentsConfig, EmbedFlowConfig, IndexConfig, ModelConfig, PlannerConfig
from embedflow.indexes import NumpyIndex
from embedflow.models import HashEmbeddingModel
from embedflow.planner import MigrationPlanner, estimate_economics, load_probe_queries
from embedflow.planner.planner import _stability_rows
from embedflow.registry.schema import REGISTRY_VERSION, contract_fingerprint


class _Documents:
    def __init__(self, values: dict[str, str]):
        self.documents = dict(values)

    def get(self, ids):
        normalized = [str(item) for item in ids]
        missing = [item for item in normalized if item not in self.documents]
        if missing:
            raise KeyError(missing)
        return {item: self.documents[item] for item in normalized}

    def size(self):
        return len(self.documents)


def _fixture(tmp_path: Path, *, count: int = 32):
    source_cfg = ModelConfig("embedflow/test-source", dimension=8)
    target_cfg = ModelConfig("embedflow/test-target", dimension=8)
    config = EmbedFlowConfig(
        source=source_cfg,
        target=target_cfg,
        index=IndexConfig(backend="faiss", path=str(tmp_path / "legacy.index")),
        documents=DocumentsConfig(path=str(tmp_path / "documents.jsonl")),
        cache=CacheConfig(path=str(tmp_path / "cache")),
        planner=PlannerConfig(max_probes=250, seed=7, k_grid=[10, 20, 50]),
    )
    documents = _Documents({f"doc-{i}": f"topic {i % 3} document {i}" for i in range(count)})
    source_model = HashEmbeddingModel(source_cfg.model, source_cfg.dimension or 8)
    target_model = HashEmbeddingModel(target_cfg.model, target_cfg.dimension or 8)
    vectors = source_model.encode_documents(list(documents.documents.values()))
    index = NumpyIndex(
        vectors,
        list(documents.documents),
        metric="cosine",
        documents=documents.documents,
        metadata={"model_fingerprint": source_model.fingerprint, "model_id": source_model.model_id},
    )
    config.validate()
    return config, source_model, target_model, index, documents


def _t2(status: str):
    def runner(source_model, target_model, index, documents, queries, **kwargs):
        per_query = []
        for query_id, _ in queries:
            hits = index.search(source_model.encode_query("shared topic"), kwargs["kmax"])
            # The target ranking deliberately follows the source order, making
            # this a deterministic strong finite-pool case.
            scores = {str(hit.document_id): float(len(hits) - rank) for rank, hit in enumerate(hits)}
            row = {"query_id": query_id, "target_scores": scores}
            for k in (10, 20, 50, 100, 200, 500):
                row[f"stability_{k}"] = 1.0
            per_query.append(row)
        return {
            "diagnostic": status,
            "recommended_k": 10,
            "features": {"probe_residual_tail_50_mean": 0.0},
            "per_query": per_query,
            "implementation": "src.t2_v1.decide",
        }

    return runner


def test_plan_is_structured_and_does_not_include_probe_text(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    result = MigrationPlanner(
        config,
        source_model=source,
        target_model=target,
        source_index=index,
        documents=documents,
        t2_runner=_t2("SAFE"),
    ).plan([("q1", "private query text"), ("q2", "another private query")])
    assert result.schema_version == 1
    assert result.t2_status == "SAFE"
    assert result.recommended_k == 10
    assert result.recommendation == "PROCEED_WITH_CAUTION"
    assert "private query text" not in result.to_json()
    assert result.evidence["probe_queries_used"] == 2
    assert result.candidate_depth["candidate_gap"] is None
    assert result.evidence["registry_match"] == result.evidence["registry_match_class"]


def test_public_plan_accepts_positional_probe_queries(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    result = public_plan(config, [("q", "probe")], source_model=source, target_model=target,
                         source_index=index, documents=documents, t2_runner=_t2("SAFE"))
    assert result.evidence["probe_queries_used"] == 1


@pytest.mark.parametrize(("status", "recommendation"), [("EXPAND", "EXPAND_PROBE"), ("UNSAFE_OR_UNCERTAIN", "DEFER")])
def test_t2_states_are_conservative(tmp_path: Path, status: str, recommendation: str):
    config, source, target, index, documents = _fixture(tmp_path)
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents, t2_runner=_t2(status)).plan([("q", "probe")])
    assert result.t2_status == status
    assert result.recommendation == recommendation
    if status != "SAFE":
        assert result.recommended_k is None


def test_no_probes_never_claims_safe(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index, documents=documents).plan()
    assert result.t2_status == "NOT_RUN"
    assert result.recommendation == "EXPAND_PROBE"
    assert result.confidence == "INSUFFICIENT"
    assert result.candidate_depth["tested_k"] == []
    assert any(item.code == "NO_PROBES" for item in result.warnings)


def test_sensitive_backend_metadata_is_redacted(tmp_path: Path):
    config, source, target, _, documents = _fixture(tmp_path)
    index = NumpyIndex(
        source.encode_documents(list(documents.documents.values())),
        list(documents.documents),
        metric="cosine",
        documents=documents.documents,
        metadata={
            "model_fingerprint": source.fingerprint,
            "access_token": "do-not-emit",
            "client_secret": "also-do-not-emit",
            "api_key_env": "PINECONE_API_KEY",
        },
    )
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents).plan()
    rendered = result.to_json()
    assert "do-not-emit" not in rendered
    assert "also-do-not-emit" not in rendered
    assert "PINECONE_API_KEY" in rendered


def test_dsn_password_fragment_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, source, target, _, documents = _fixture(tmp_path)
    dsn = "postgresql://embedflow:POSTGRES_SUPER_SECRET_123@localhost/embedflow"
    monkeypatch.setenv("EMBEDFLOW_PGVECTOR_DSN", dsn)
    index = NumpyIndex(
        source.encode_documents(list(documents.documents.values())),
        list(documents.documents), metric="cosine", documents=documents.documents,
        metadata={"diagnostic": "POSTGRES_SUPER_SECRET_123"},
    )
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents).plan()
    assert "POSTGRES_SUPER_SECRET_123" not in result.to_json()


def test_model_contract_does_not_disclose_local_path(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    config.source.local_path = "/private/user/model-source"
    config.target.local_path = "/private/user/model-target"
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents).plan()
    rendered = result.to_json()
    assert "/private/user/model-source" not in rendered
    assert "/private/user/model-target" not in rendered


def test_planner_rejects_invalid_cli_numeric_overrides(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    planner = MigrationPlanner(config, source_model=source, target_model=target, source_index=index, documents=documents)
    with pytest.raises(ValueError, match="queries_per_second"):
        planner.plan(queries_per_second=-1)
    with pytest.raises(ValueError, match="cache_hit_rate"):
        planner.plan(cache_hit_rate=1.1)
    with pytest.raises(ValueError, match="max_probes"):
        planner.plan(max_probes=0)


def test_candidate_documents_are_encoded_once_across_queries(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    calls: list[int] = []
    original = target.encode_documents

    def counted(texts, batch_size=None):
        calls.append(len(texts))
        return original(texts, batch_size=batch_size)

    target.encode_documents = counted  # type: ignore[method-assign]
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents, t2_runner=_t2("SAFE")).plan(
                                  [("q1", "first wording"), ("q2", "second wording"), ("q3", "third wording")])
    assert result.cache["unique_candidate_documents"] < result.cache["candidate_occurrences"]
    assert sum(calls) == result.cache["unique_candidate_documents"]


def test_duplicate_probe_text_does_not_inflate_evidence(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path, count=64)
    probes = [(f"q-{i}", "the same semantic probe") for i in range(100)]
    result = MigrationPlanner(
        config, source_model=source, target_model=target, source_index=index,
        documents=documents, t2_runner=_t2("SAFE"),
    ).plan(probes)
    assert result.evidence["probe_queries_supplied"] == 100
    assert result.evidence["probe_queries_sampled"] == 100
    assert result.evidence["probe_queries_used"] == 1
    assert result.evidence["duplicate_probe_texts_removed"] == 99
    assert result.confidence == "LIMITED"
    assert any(item.code == "DUPLICATE_PROBE_CONTENT" for item in result.warnings)


def test_dimension_preflight_blocks_t2(tmp_path: Path):
    config, source, target, _, documents = _fixture(tmp_path)
    bad_index = NumpyIndex(np.ones((32, 7), dtype="float32"), list(documents.documents), metric="cosine")
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("T2 must not run after failed preflight")

    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=bad_index,
                              documents=documents, t2_runner=should_not_run).plan([("q", "probe")])
    assert result.recommendation == "BLOCKED"
    assert result.t2_status == "NOT_RUN"
    assert not called
    assert any(item.code == "SOURCE_DIMENSION_MISMATCH" for item in result.warnings)


def test_probe_loader_validates_jsonl_and_sampling(tmp_path: Path):
    path = tmp_path / "probes.jsonl"
    path.write_text(json.dumps({"id": "q1", "query": "alpha"}) + "\n" + json.dumps({"query": "beta"}) + "\n")
    rows = load_probe_queries(path)
    assert rows[0] == ("q1", "alpha")
    assert len(rows) == 2
    with pytest.raises(ValueError, match="duplicate"):
        load_probe_queries([("q", "a"), ("q", "b")])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert load_probe_queries(empty) == []


def test_economics_are_provenance_aware_and_nonnegative():
    result = estimate_economics(1_000, 256, docs_per_second=100, gpu_hourly_cost=2.0)
    assert result["raw_vector_storage"]["value"] == 1_000 * 256 * 4
    assert result["raw_vector_storage"]["provenance"] == "modeled"
    assert result["full_backfill"]["gpu_hours"]["value"] > 0
    with pytest.raises(ValueError, match="non-negative"):
        estimate_economics(-1, 256)
    with pytest.raises(ValueError, match="greater than zero"):
        estimate_economics(1, 256, docs_per_second=0)
    with pytest.raises(ValueError, match="positive integer"):
        estimate_economics(1, 0)
    # NumPy builds without a native bfloat16 dtype still have a well-defined
    # two-byte embedding representation; the planner keeps that estimate
    # explicit instead of failing an otherwise valid model contract.
    bf16 = estimate_economics(10, 4, dtype="bfloat16")
    assert bf16["dtype"] == "bfloat16"
    assert bf16["raw_vector_storage"]["value"] == 10 * 4 * 2


def test_cli_plan_json_demo(tmp_path: Path):
    # ``cmd_plan`` opens a real demo runtime from a serialized config. Build a
    # tiny artifact and use the public CLI formatter with injected resources
    # covered by the library tests above; here we only assert parser wiring.
    args = cli.build_parser().parse_args(["plan", "--config", str(tmp_path / "missing.yaml"), "--format", "json", "--quiet"])
    assert args.command == "plan"
    assert args.format == "json"
    assert args.quiet is True


def test_atomic_plan_output_preserves_previous_file_on_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    destination = tmp_path / "migration-plan.json"
    destination.write_text("previous plan", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(cli.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        cli._atomic_write_text(destination, "new plan")
    assert destination.read_text(encoding="utf-8") == "previous plan"
    assert list(tmp_path.glob("*.tmp")) == []


def test_negative_seed_is_a_valid_reproducible_seed_and_serving_defaults_are_reported(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    config.migration.max_sync_misses = 7
    config.migration.background_batch_size = 11
    config.planner.seed = -9
    config.validate()
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index, documents=documents).plan()
    assert result.evidence["sampling_seed"] == -9
    assert result.cache["recommended_max_sync_misses"] == 7
    assert result.cache["background_batch_size"] == 11


def test_tiny_source_does_not_call_t2_or_raise_probe_minimum_error(tmp_path: Path):
    config, source, target, _, _ = _fixture(tmp_path, count=5)
    documents = _Documents({f"doc-{i}": f"tiny {i}" for i in range(5)})
    vectors = source.encode_documents(list(documents.documents.values()))
    index = NumpyIndex(vectors, list(documents.documents), metric="cosine", documents=documents.documents,
                       metadata={"model_fingerprint": source.fingerprint})
    config.planner.k_grid = [1, 5]
    config.validate()
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("T2 must not run for a source index smaller than ten rows")

    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents, t2_runner=should_not_run).plan([("q", "probe")])
    assert not called
    assert result.t2_status == "NOT_RUN"
    assert result.recommendation == "EXPAND_PROBE"
    assert any(item.code == "K_BELOW_T2_MINIMUM" for item in result.warnings)


def test_registry_prior_is_not_exact_corpus_match(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    # A custom transition row is intentionally omitted; the packaged registry
    # should therefore remain NO REGISTRY MATCH rather than being guessed from
    # vector dimensions.  The exact/prior distinction itself is covered by the
    # matcher, and this verifies the planner surfaces its level unchanged.
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index, documents=documents).plan()
    assert result.evidence["registry_match_class"] == "NO REGISTRY MATCH"
    assert not result.evidence["exact_corpus_match"]


def test_prior_registry_evidence_does_not_upgrade_confidence(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path, count=32)
    source_contract = config.source.contract()
    source_contract["canonical_model_id"] = config.source.model
    source_contract["contract_fingerprint"] = contract_fingerprint(source_contract)
    target_contract = config.target.contract()
    target_contract["canonical_model_id"] = config.target.model
    target_contract["contract_fingerprint"] = contract_fingerprint(target_contract)
    record = {
        "registry_version": REGISTRY_VERSION,
        "evidence_id": "synthetic-prior",
        "source": source_contract,
        "target": target_contract,
        "dataset": {"name": "different-corpus", "corpus_size": 999, "canonical_construction": True},
        "metric": "ndcg@10",
        "candidate_gap": {"10": 0.0},
        "containment": {"10": 1.0},
        "observed_migration_depth": 10,
        "candidate_depths": [10],
        "epsilon": 0.01,
        "provenance": {"artifact": "synthetic"},
    }
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents, registry_records=[record], t2_runner=_t2("SAFE")).plan(
                                  [(str(i), "probe") for i in range(100)])
    assert result.evidence["registry_match_class"] == "PRIOR EVIDENCE AVAILABLE"
    assert result.confidence != "STRONG"


def test_t2_recommended_depth_is_respected_when_larger_than_first_stable_k(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path, count=64)

    def runner(source_model, target_model, source_index, document_store, queries, **kwargs):
        rows = []
        for query_id, _ in queries:
            hits = source_index.search(source_model.encode_query("shared topic"), kwargs["kmax"])
            scores = {str(hit.document_id): float(len(hits) - rank) for rank, hit in enumerate(hits)}
            rows.append({"query_id": query_id, "target_scores": scores})
        return {"diagnostic": "SAFE", "recommended_k": 50, "per_query": rows,
                "features": {"probe_residual_tail_50_mean": 0.0}}

    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents, t2_runner=runner).plan([("q", "probe")])
    assert result.recommended_k == 50


def test_nonmonotonic_stability_does_not_certify_shallow_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A contradictory deeper instability must invalidate an isolated shallow point."""
    import importlib

    planner_module = importlib.import_module("embedflow.planner.planner")
    config, source, target, index, documents = _fixture(tmp_path, count=128)
    config.planner.k_grid = [20, 50, 100]

    def contradictory_rows(_probe: dict, k_grid: list[int]) -> list[dict[str, object]]:
        values = {20: 0.95, 50: 0.40, 100: 0.95}
        return [
            {"k": k, "mean_top10_stability": values[k], "minimum_top10_stability": values[k],
             "queries": 1, "provenance": "synthetic-adversarial"}
            for k in k_grid
        ]

    monkeypatch.setattr(planner_module, "_stability_rows", contradictory_rows)
    result = MigrationPlanner(
        config, source_model=source, target_model=target, source_index=index,
        documents=documents, t2_runner=_t2("SAFE"),
    ).plan([("q", "probe")])
    assert result.recommended_k == 100
    assert any(item.code == "NONMONOTONIC_K_STABILITY" for item in result.warnings)


def test_modeled_latency_keeps_modeled_provenance(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path)
    config.telemetry.latency_log = str(tmp_path / "latency.jsonl")
    Path(config.telemetry.latency_log).write_text(json.dumps({"total_ms": 12.5}) + "\n")
    result = MigrationPlanner(config, source_model=source, target_model=target, source_index=index,
                              documents=documents).plan()
    assert result.performance["modeled"]["warm_path_p50"]["provenance"] == "modeled"


def test_profile_excludes_one_warmup_and_keeps_measured_provenance(tmp_path: Path):
    config, source, target, index, documents = _fixture(tmp_path, count=32)
    result = MigrationPlanner(
        config, source_model=source, target_model=target, source_index=index,
        documents=documents, t2_runner=_t2("SAFE"),
    ).plan([("q", "probe")], profile=True)
    profile = result.performance["profile"]
    assert profile["warmup_excluded"] is True
    assert profile["sample_count"] == 1
    assert profile["target_docs_per_second_provenance"] == "measured"
    assert profile["target_docs_per_second"] > 0
    assert profile["source_query_encode"]["p50"]["provenance"] == "measured"


def test_stability_normalization_properties_2000_examples():
    """Exercise normalization invariants over varied valid SDK-like rows."""
    rng = random.Random(20260913)
    k_grid = [10, 20, 50, 100, 500]
    for _ in range(2_000):
        count = rng.randrange(1, 80)
        ids = [f"id-{i}-{rng.randrange(10**9)}-{rng.choice(('α', '🙂', 'x'))}" for i in range(count)]
        scores = {document_id: rng.uniform(-100.0, 100.0) for document_id in ids}
        payload = {"per_query": [{"query_id": "q", "target_scores": scores}]}
        before = copy.deepcopy(payload)
        rows = _stability_rows(payload, k_grid)
        assert payload == before
        assert len(rows) == len(k_grid)
        assert all(
            row["mean_top10_stability"] is None
            or 0.0 <= float(row["mean_top10_stability"]) <= 1.0
            for row in rows
        )
