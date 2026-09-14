"""Human and machine renderers for migration plans."""

from __future__ import annotations

from typing import Any

from .models import PlanResult


def _value(value: Any, default: str = "UNKNOWN") -> str:
    return default if value is None or value == "" else str(value)


def render_plan(plan: PlanResult) -> str:
    """Render a concise terminal report without probe text or ANSI escapes."""
    data = plan.to_dict()
    source, target = data.get("source", {}), data.get("target", {})
    preflight, evidence = data.get("preflight", {}), data.get("evidence", {})
    depth, cache, economics = data.get("candidate_depth", {}), data.get("cache", {}), data.get("economics", {})
    rollout = data.get("rollout", {})
    lines = [
        "MIGRATION PLAN",
        "=" * 64,
        "",
        "Recommendation",
        f"  {_value(data.get('recommendation'))} (evidence: {_value(data.get('confidence'))})",
        "",
        "Source",
        f"  Backend:        {_value(source.get('backend'))}",
        f"  Model:          {_value(source.get('model'))}",
        f"  Dimension:      {_value(source.get('dimension'))}",
        f"  Documents:      {_value(source.get('corpus_documents'))}",
        f"  Metric:         {_value(source.get('metric'))}",
        f"  Index health:   {_value(preflight.get('status'))}",
        "",
        "Target",
        f"  Model:          {_value(target.get('model'))}",
        f"  Dimension:      {_value(target.get('dimension'))}",
        f"  Device:         {_value(target.get('device'))}",
        "",
        "Evidence",
        f"  Registry:       {_value(evidence.get('registry_match_class'))}",
        f"  Exact corpus:   {_value(evidence.get('exact_corpus_match'))}",
        f"  Probe queries:  {_value(evidence.get('probe_queries_used'), '0')}",
        f"  ANN fidelity:   {_value(preflight.get('ann_fidelity'))}",
        "",
        "Candidate depth",
        f"  K tested:       {_value(depth.get('tested_k'))}",
        f"  Recommended K:  {_value(depth.get('recommended_k'))}",
        f"  T2-v1:          {_value(depth.get('t2_status'))}",
        f"  Reason:          {_value(depth.get('recommendation_reason'))}",
        "",
        "Cache strategy",
        f"  Sync misses:    {_value(cache.get('recommended_max_sync_misses'))}",
        f"  Background:     {_value(cache.get('background_batch_size'))}",
        f"  Prewarm:        {_value(cache.get('suggested_prewarm_policy'))}",
        "",
        "Economics",
        f"  Raw vectors:    {_quantity_value(economics.get('raw_vector_storage'))}",
        f"  Full backfill:  {_quantity_value((economics.get('full_backfill') or {}).get('wall_time'))}",
        f"  Cost:           {_quantity_value((economics.get('full_backfill') or {}).get('cost'))}",
        "",
        "Rollout",
    ]
    for phase in rollout.get("phases", []) or []:
        if isinstance(phase, dict):
            lines.append(f"  {phase.get('order', '')}. {phase.get('name', 'Phase')}: {phase.get('action', '')}")
        else:
            lines.append(f"  - {phase}")
    warnings = data.get("warnings", []) or []
    if warnings:
        lines.extend(["", "Warnings"])
        for warning in warnings:
            if isinstance(warning, dict):
                lines.append(f"  [{warning.get('severity', 'WARN')}] {warning.get('code')}: {warning.get('message')}")
            else:
                lines.append(f"  - {warning}")
    lines.extend(["", f"Next: {_value(rollout.get('next_action'), 'run a shadow evaluation before canary traffic.')}"])
    return "\n".join(lines) + "\n"


def _quantity_value(value: Any) -> str:
    if not isinstance(value, dict):
        return "UNKNOWN"
    raw = value.get("value")
    if raw is None:
        return "UNKNOWN"
    unit = value.get("unit", "")
    provenance = str(value.get("provenance", "unknown")).upper()
    return f"{raw} {unit} ({provenance})"


__all__ = ["render_plan"]
