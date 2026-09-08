from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

import embedflow.cli as cli
from embedflow import __version__
from embedflow.cache import SQLiteVectorCache
from embedflow.compatibility.candidate_gap import compute_candidate_gap_curve
from embedflow.compatibility.migration_depth import observed_migration_depth
from embedflow.compatibility.t2 import diagnose_t2
from embedflow.config import ModelConfig, from_dict, load_config


def test_model_fingerprint_excludes_deployment_details_but_includes_contract():
    a = ModelConfig("demo", dimension=8, local_path="/one", device="cpu")
    b = ModelConfig("demo", dimension=8, local_path="/two", device="cuda")
    assert a.fingerprint == b.fingerprint
    c = ModelConfig("demo", dimension=8, query_instruction="Q:{text}")
    assert c.fingerprint != a.fingerprint


def test_config_accepts_auto_candidate_depth_and_public_probe_section():
    cfg = from_dict({
        "source": {"model": "embedflow/demo-source", "dimension": 8},
        "target": {"model": "embedflow/demo-target", "dimension": 8},
        "migration": {"candidate_depth": "auto", "kmax_probe": 20},
        "probe": {"k_values": [10, 20], "epsilon": 0.01},
    })
    assert cfg.migration.candidate_depth == "auto"
    assert cfg.probe.k_values == [10, 20]


def test_gap_is_distinct_from_containment_and_observed_k():
    curve = compute_candidate_gap_curve(
        source_rankings={"q": ["a", "b", "c"]},
        target_scores={"q": {"a": 0.9, "b": 0.8, "c": 0.1}},
        native_target_rankings={"q": ["a", "b", "c"]},
        qrels={"q": {"a": 2, "b": 1}},
        k_values=[1, 2, 3],
    )
    # Containment is measured against the native target top-quality-k set
    # (quality_k defaults to 10, so this three-document fixture contains one
    # of the three native results at K=1). It is intentionally not the gap.
    assert curve[0].containment == 1 / 3
    assert curve[0].candidate_gap >= 0.0
    assert observed_migration_depth([row.to_dict() for row in curve], epsilon=0.01) == 2


def test_frozen_t2_wrapper_preserves_vocabulary():
    safe = {
        "probe_residual_tail_50_mean": 0.01,
        "deepest_p90": 100,
        "late_tail_area": 0.1,
        "stability_to_500_50_mean": 0.99,
        "last_shell_any_rate": 0,
        "fraction_margin_nonpositive": 0,
    }
    result = diagnose_t2(safe)
    assert result.diagnostic == "SAFE"
    assert "not a compatibility guarantee" in result.warning


def test_cache_rejects_cross_model_reuse(tmp_path: Path):
    values = np.ones((1, 3), dtype="float32")
    first = SQLiteVectorCache(tmp_path / "cache", "model-a", 3)
    first.put(["doc-1"], values)
    assert first.get(["doc-1", "missing"])["doc-1"].shape == (3,)
    assert first.stats()["total_cache_hits"] == 1
    assert first.stats()["total_cache_misses"] == 1
    first.close()
    second = SQLiteVectorCache(tmp_path / "cache", "model-b", 3)
    assert second.stats()["cached_target_vectors"] == 0
    second.close()


def test_environment_overrides_are_explicit_and_validated(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "embedflow.yaml"
    config_path.write_text(
        "source:\n  model: embedflow/demo-source\n  dimension: 8\n"
        "target:\n  model: embedflow/demo-target\n  dimension: 8\n"
        "index:\n  path: ./legacy.index\n"
        "documents:\n  path: ./documents.jsonl\n"
    )
    monkeypatch.setenv("EMBEDFLOW_CANDIDATE_DEPTH", "auto")
    monkeypatch.setenv("EMBEDFLOW_MAX_SYNC_MISSES", "0")
    cfg = load_config(config_path)
    assert cfg.migration.candidate_depth == "auto"
    assert cfg.migration.max_sync_misses == 0


def test_cli_version_is_available(capsys):
    try:
        cli.build_parser().parse_args(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--version should terminate argparse successfully")
    assert f"embedflow {__version__}" in capsys.readouterr().out


def test_doctor_treats_optional_runtime_modules_as_warnings(monkeypatch, capsys):
    original_find_spec = cli.importlib.util.find_spec
    optional_modules = {"faiss", "torch", "fastapi", "qdrant_client"}

    def missing_optional(name: str):
        if name in optional_modules:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(cli.importlib.util, "find_spec", missing_optional)
    assert cli.cmd_doctor(Namespace(config=None, json=False)) == 0
    output = capsys.readouterr().out
    assert "WARN  pytorch:" in output
    assert "Overall: PASS" in output
