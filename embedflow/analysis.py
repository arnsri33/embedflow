"""Programmatic Mode-B migration analysis.

The CLI is the most discoverable interface, but applications and notebooks
often already hold a parsed EmbedFlowConfig. This module delegates to the
same runtime and frozen T2-v1 implementation used by ``embedflow analyze``;
it does not introduce a second scoring path.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .compatibility.report import report_markdown, write_report
from .config import EmbedFlowConfig, from_dict, load_config, save_config
from .migration.compatibility import run_probe, save_probe
from .registry import load_evidence, match_config
from .runtime import load_documents, open_engine


def _query_rows(value: str | Path | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid query JSON at {path}:{number}") from exc
                if not isinstance(item, Mapping):
                    raise ValueError(f"query row {number} in {path} must be a JSON object")
                query_id = str(item.get("id", item.get("query_id", number)))
                text = item.get("text", item.get("query"))
                if query_id in seen:
                    raise ValueError(f"duplicate query ID {query_id!r} in {path}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"query {query_id!r} has no text")
                seen.add(query_id); rows.append((query_id, text))
        if not rows:
            raise ValueError(f"query file is empty: {value}")
        return rows
    try:
        rows = [(str(query_id), text) for query_id, text in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("probe_queries must be an iterable of (query_id, text) pairs") from exc
    if len({query_id for query_id, _ in rows}) != len(rows):
        raise ValueError("probe_queries must not contain duplicate query IDs")
    if not rows or any(not isinstance(text, str) or not text.strip() for _, text in rows):
        raise ValueError("probe_queries must contain non-empty query text")
    return rows


def _registry_values(match: Any) -> list[dict[str, Any]]:
    """Return only the canonical measurements explicitly reused by a match."""
    values: list[dict[str, Any]] = []
    for row in match.records:
        data = row.to_dict()
        values.append({
            "evidence_id": row.evidence_id,
            "dataset": data.get("dataset", {}).get("name"),
            "candidate_gap": data.get("candidate_gap", {}),
            "containment": data.get("containment", {}),
            "source_quality": data.get("source_quality"),
            "native_target_quality": data.get("native_target_quality"),
            "restricted_target_quality": data.get("restricted_target_quality", {}),
            "observed_migration_depth": data.get("observed_migration_depth"),
            "ci_certified_migration_depth": data.get("ci_certified_migration_depth"),
            "epsilon": data.get("epsilon"),
            "provenance": data.get("provenance", {}),
        })
    return values


def analyze_migration(
    config: str | Path | EmbedFlowConfig | Mapping[str, Any],
    *,
    probe_queries: str | Path | Iterable[tuple[str, str]] | None = None,
    device: str | None = None,
    demo: bool = False,
    corpus_name: str | None = None,
    corpus_fingerprint: str | None = None,
    use_registry: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run leakage-safe no-target-index analysis and return a JSON report.

    ``use_registry`` only succeeds for an exact contract-and-corpus match. A
    prior or related row is displayed in the returned ``registry`` field but
    never converted into a current-corpus compatibility decision.
    """
    temporary_config: Path | None = None
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        cfg = load_config(config_path)
    else:
        cfg = config if isinstance(config, EmbedFlowConfig) else from_dict(dict(config))
        cfg.validate()
        # open_engine intentionally has one configuration-loading path.
        # Serialize an in-memory config to a short-lived file so this API
        # inherits all of its validation, model, and index checks.
        cfg.resolve_paths(Path.cwd().resolve())
        handle = tempfile.NamedTemporaryFile(prefix="embedflow-analysis-", suffix=".yaml", delete=False)
        handle.close()
        temporary_config = Path(handle.name)
        save_config(cfg, temporary_config)
        config_path = temporary_config
    if probe_queries is None:
        probe_queries = cfg.probe.queries
    if probe_queries is None:
        raise ValueError("probe_queries or config.probe.queries is required")
    engine = None
    try:
        docs = load_documents(cfg)
        registry = match_config(
            cfg,
            corpus_fingerprint=corpus_fingerprint,
            corpus_name=corpus_name,
            corpus_size=docs.size(),
            records=load_evidence(),
        )
        if use_registry and registry.level != "EXACT REGISTRY MATCH":
            raise ValueError("--use-registry requires an EXACT REGISTRY MATCH")
        engine = open_engine(config_path, device=device, demo=demo, start_worker=False)
        result = run_probe(
            engine.source_model,
            engine.target_model,
            engine.source_index,
            docs,
            _query_rows(probe_queries),
            kmax=cfg.probe.kmax,
            seed=cfg.probe.seed,
            limit=cfg.probe.limit,
        )
        report: dict[str, Any] = {
            "schema_version": "0.1",
            "source_model": cfg.source.model,
            "target_model": cfg.target.model,
            "corpus_documents": docs.size(),
            "diagnostic": result["diagnostic"],
            "recommended_initial_k": result.get("recommended_k"),
            "observed_k_epsilon": None,
            "epsilon": cfg.probe.epsilon,
            "ann_status": "UNKNOWN",
            "native_target_index_used": False,
            "registry": registry.to_dict(),
            "registry_reused": bool(use_registry),
            "registry_reuse": {
                "used": bool(use_registry),
                "evidence_ids": [row.evidence_id for row in registry.records] if use_registry else [],
                "reused_fields": ["candidate_gap", "containment", "native_target_quality", "source_quality", "observed_migration_depth", "ci_certified_migration_depth"] if use_registry else [],
                "note": "Canonical values are prior measurements; the current-corpus probe remains the deployment analysis." if use_registry else "Canonical rows were displayed but not reused.",
            },
            "registry_reused_values": _registry_values(registry) if use_registry else [],
            "registry_evidence": [row.to_dict() for row in registry.records],
            "probe": result,
            "candidate_gap_curve": [],
            "t2_warning": result.get("warning"),
            "recommendation": (
                "Progressive migration is a reasonable candidate for further deployment validation."
                if str(result.get("diagnostic", "")).upper() == "SAFE"
                else "Further validation is required before relying on progressive migration."
            ),
            "limitations": [
                "No native target index or qrels were used; this is Mode B deployment analysis.",
                "Recommended initial K is not observed K*.",
                "ANN fidelity is UNKNOWN until an exact/reference source comparison is supplied.",
            ],
        }
        if output_dir is not None:
            destination = Path(output_dir).expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            save_probe(result, destination / "probe_result.json")
            write_report(destination / "migration_report.json", report)
            (destination / "report.md").write_text(report_markdown(report), encoding="utf-8")
        return report
    finally:
        if engine is not None:
            engine.close()
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


__all__ = ["analyze_migration"]
