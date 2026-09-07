from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .candidate_gap import CandidateGapCurve, observed_k_epsilon


def write_curve_csv(path: str | Path, curve: Sequence[CandidateGapCurve], *, field: str = "candidate_gap") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [point.to_dict() for point in curve]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["k", field])
        writer.writeheader()
        writer.writerows(rows)


def migration_report(
    *,
    source_model: str,
    target_model: str,
    corpus_size: int,
    curve: Sequence[CandidateGapCurve],
    diagnostic: str | None = None,
    recommended_k: int | None = None,
    ann_status: str = "UNKNOWN",
    t2_warning: str = "T2-v1 is an empirical finite-tail diagnostic, not a compatibility guarantee.",
    epsilon: float = 0.01,
    native_target_index_used: bool = True,
) -> dict[str, Any]:
    observed = observed_k_epsilon(curve, epsilon=epsilon) if native_target_index_used else None
    if str(diagnostic or "").upper() == "SAFE":
        recommendation = "Progressive migration is a reasonable candidate for further deployment validation."
    elif str(diagnostic or "").upper() == "EXPAND":
        recommendation = "Expand source retrieval depth and validate manually before deployment."
    else:
        recommendation = "Prefer manual validation, a larger probe, or a full target backfill before relying on progressive migration."
    return {
        "schema_version": "0.1",
        "source_model": source_model,
        "target_model": target_model,
        "corpus_documents": int(corpus_size),
        "diagnostic": diagnostic,
        "recommended_initial_k": recommended_k,
        "observed_k_epsilon": observed,
        "epsilon": float(epsilon),
        "ann_status": ann_status,
        "recommendation": recommendation,
        "native_target_index_used": bool(native_target_index_used),
        "t2_warning": t2_warning,
        "candidate_gap_curve": [point.to_dict() for point in curve],
        "limitations": [
            "Candidate gap and containment are separate metrics.",
            "A SAFE T2-v1 diagnostic is empirical and is not a compatibility guarantee.",
            "ANN fidelity is UNKNOWN unless an exact/reference source comparison was supplied.",
        ],
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False, default=float) + "\n")


def report_markdown(report: Mapping[str, Any]) -> str:
    curve = list(report.get("candidate_gap_curve", []))
    corpus_value = report.get("corpus_documents", "unknown")
    try:
        corpus_display = f"{int(corpus_value):,}"
    except (TypeError, ValueError):
        corpus_display = str(corpus_value)
    lines = [
        "# EmbedFlow Migration Report", "", "## Models", "",
        f"- Source: `{report.get('source_model', 'unknown')}`",
        f"- Target: `{report.get('target_model', 'unknown')}`",
        f"- Corpus documents: `{corpus_display}`",
        "", "## Decision", "",
        f"- T2-v1 diagnostic: **{report.get('diagnostic') or 'not run'}**",
        f"- Recommended initial candidate depth: **K={report.get('recommended_initial_k') or 'n/a'}**",
        f"- Observed K*_epsilon: **{report.get('observed_k_epsilon') or 'not computed'}** (epsilon={report.get('epsilon', 0.01)})",
        f"- Finite-tail behavior: **{report.get('finite_tail_behavior', 'not established')}**",
        f"- ANN health: **{report.get('ann_status', 'UNKNOWN')}**",
        f"- Recommendation: {report.get('recommendation', 'Further validation required.')}",
        "", "> SAFE is an empirical finite-tail diagnostic, not a compatibility guarantee.",
        "", "## Candidate gap curve", "", "| K | native target | target within source candidates | G(K) | containment |", "|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        lines.append(f"| {row.get('k')} | {float(row.get('native_target_quality', 0)):.4f} | {float(row.get('restricted_target_quality', 0)):.4f} | {float(row.get('candidate_gap', 0)):.4f} | {float(row.get('containment', 0)):.4f} |")
    reused = report.get("registry_reused_values") or []
    if reused:
        lines.extend(["", "## Reused registry evidence", "", "> These are canonical prior measurements reused because the source/target contracts and corpus construction matched exactly. They are not a new-corpus probe result.", "", "| Evidence | Dataset | G(50) | Observed K* | CI-certified K* |", "|---|---|---:|---:|---:|"])
        for item in reused:
            gaps = item.get("candidate_gap") or {}
            gap_50 = gaps.get("50", gaps.get(50, "unavailable"))
            lines.append(f"| `{item.get('evidence_id', 'unknown')}` | {item.get('dataset', 'unknown')} | {float(gap_50):.5f} | {item.get('observed_migration_depth') or 'unavailable'} | {item.get('ci_certified_migration_depth') or 'unavailable'} |" if gap_50 != "unavailable" else f"| `{item.get('evidence_id', 'unknown')}` | {item.get('dataset', 'unknown')} | unavailable | {item.get('observed_migration_depth') or 'unavailable'} | {item.get('ci_certified_migration_depth') or 'unavailable'} |")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report.get("limitations", [])], ""])
    return "\n".join(lines)
