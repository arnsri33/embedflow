"""Human rendering for structured Shadow Mode reports."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_shadow_report(report: dict[str, Any]) -> str:
    if not report.get("traffic", {}).get("shadow_sampled_total") and report.get("warnings") == ["No shadow observations in the selected window."]:
        return "SHADOW MIGRATION REPORT\n========================\n\nNo shadow observations in selected window.\n"
    traffic = report.get("traffic", {}); cache = report.get("cache", {}); material = report.get("materialization", {})
    latency = report.get("latency", {}); ranking = report.get("ranking", {}); coverage = report.get("target_coverage", {})
    t2 = report.get("t2", {}); migration = report.get("migration", {})
    lines = ["SHADOW MIGRATION REPORT", "========================", "",
             f"Recommendation: {report.get('recommendation', 'CONTINUE_SHADOW')}",
             "Ranking disagreement is diagnostic; it is not qrels-based retrieval quality.", "",
             "Migration", f"  Backend: {migration.get('backend') or 'unknown'}",
             f"  Source fingerprint: {_fmt(migration.get('source_fingerprint'))}",
             f"  Target fingerprint: {_fmt(migration.get('target_fingerprint'))}",
             f"  Configuration fingerprint: {_fmt(migration.get('config_fingerprint'))}", "",
             "Window", f"  Start: {_fmt(report.get('window', {}).get('start'))}", f"  End:   {_fmt(report.get('window', {}).get('end'))}", "",
             "Traffic",
             f"  Primary: {traffic.get('primary_requests_total', 0)}",
             f"  Eligible/sample: {traffic.get('shadow_eligible_total', 0)} / {traffic.get('shadow_sampled_total', 0)}",
             f"  Completed/partial: {traffic.get('shadow_completed_total', 0)} / {traffic.get('shadow_partial_total', 0)}",
             f"  Failed/timed out/dropped: {traffic.get('shadow_failed_total', 0)} / {traffic.get('shadow_timeout_total', 0)} / {traffic.get('shadow_dropped_total', 0)}", "",
             "Candidate configuration", f"  K: {migration.get('candidate_k', 'configured')}", "",
             "Cache", f"  Hit rate: {_fmt(cache.get('hit_rate'), '0.000')}", f"  Hits/misses: {cache.get('hits', 0)} / {cache.get('misses', 0)}", "",
             "Materialization", f"  Unique queued: {material.get('unique_docs_queued', 0)}", f"  Materialized: {material.get('docs_materialized', 0)}", f"  Queue depth: {material.get('queue_depth', 0)}", "",
             "Latency (shadow is off the primary critical path)", f"  Source p50/p95: {_fmt(latency.get('source', {}).get('p50_ms'))} / {_fmt(latency.get('source', {}).get('p95_ms'))} ms", f"  Shadow p50/p95: {_fmt(latency.get('shadow', {}).get('p50_ms'))} / {_fmt(latency.get('shadow', {}).get('p95_ms'))} ms", "",
             "Ranking diagnostics", f"  Top-1 agreement: {_fmt(ranking.get('top1_agreement'))}", f"  Top-k overlap: {_fmt(ranking.get('top_k_overlap'))}", f"  Target coverage mean/min: {_fmt(coverage.get('mean'))} / {_fmt(coverage.get('min'))}", "",
             "T2-v1 window", f"  Status: {t2.get('status', 'NOT_RUN')}", f"  Queries: {t2.get('queries', 0)}", "",
             "Warnings"]
    warnings = report.get("warnings") or ["None"]
    lines.extend(f"  - {warning}" for warning in warnings)
    lines.extend(["", "Next: continue shadow evaluation; READY_FOR_CANARY_EVALUATION is guidance only and never routes traffic automatically.", ""])
    return "\n".join(lines)


__all__ = ["render_shadow_report"]
