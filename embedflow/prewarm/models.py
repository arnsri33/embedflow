"""Serializable artifacts for traffic-aware target-vector prewarming."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

PREWARM_SCHEMA_VERSION = 1
MAX_INLINE_PREWARM_IDS = 1_000_000


@dataclass(frozen=True)
class PrewarmWarning:
    code: str
    severity: str
    message: str
    remediation: str

    def __post_init__(self) -> None:
        severity = str(self.severity).upper()
        if severity not in {"INFO", "WARN", "ERROR"}:
            raise ValueError("prewarm warning severity must be INFO, WARN, or ERROR")
        if not str(self.code).strip() or not str(self.message).strip() or not str(self.remediation).strip():
            raise ValueError("prewarm warning code, message, and remediation are required")
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrewarmPlan:
    """Bounded, fingerprinted traffic-hotset plan.

    Selected IDs are intentionally inline for the first release.  The planner
    enforces a practical maximum through ``max_docs`` and refuses malformed
    duplicate IDs.  A future manifest format can be added without changing the
    schema version contract for these fields.
    """

    schema_version: int
    migration: dict[str, Any]
    window: dict[str, Any]
    strategy: str
    baseline: dict[str, Any]
    selection: dict[str, Any]
    budget: dict[str, Any]
    cost: dict[str, Any]
    coverage_curve: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[PrewarmWarning | dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    status: str = "READY"

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "migration": self.migration,
            "window": self.window,
            "strategy": self.strategy,
            "baseline": self.baseline,
            "selection": self.selection,
            "budget": self.budget,
            "cost": self.cost,
            "coverage_curve": self.coverage_curve,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
            "status": self.status,
        }

    @staticmethod
    def _convert(value: Any) -> Any:
        if isinstance(value, PrewarmWarning):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): PrewarmPlan._convert(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [PrewarmPlan._convert(child) for child in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        converted = self._convert(self._core_dict())
        converted["plan_fingerprint"] = self.fingerprint
        return converted

    @property
    def selected_ids(self) -> list[str]:
        return [str(value) for value in self.selection.get("selected_ids", [])]

    @property
    def fingerprint(self) -> str:
        core = self._convert(self._core_dict())
        payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict_without_fingerprint(self) -> dict[str, Any]:
        return self._convert(self._core_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PrewarmPlan:
        if not isinstance(raw, dict):
            raise ValueError("prewarm plan must contain a JSON object")
        schema = raw.get("schema_version")
        if isinstance(schema, bool) or not isinstance(schema, int) or schema != PREWARM_SCHEMA_VERSION:
            raise ValueError(f"unsupported prewarm plan schema_version: {schema!r}")
        sections = ("migration", "window", "baseline", "selection", "budget", "cost")
        for section in sections:
            value = raw.get(section)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"prewarm plan {section} must be an object")
        selection_raw = raw.get("selection") or {}
        selected = selection_raw.get("selected_ids", [])
        if not isinstance(selected, list) or any(not isinstance(value, str) or not value.strip() for value in selected):
            raise ValueError("prewarm plan selection.selected_ids must be a list of strings")
        if len(selected) > MAX_INLINE_PREWARM_IDS:
            raise ValueError("prewarm plan contains too many inline IDs; use a bounded plan/manifest")
        if len(set(selected)) != len(selected):
            raise ValueError("prewarm plan selection.selected_ids contains duplicates")
        declared_documents = selection_raw.get("documents")
        if declared_documents is not None:
            try:
                declared_int = int(declared_documents)
                exact = float(declared_documents) == declared_int
            except (TypeError, ValueError, OverflowError):
                declared_int, exact = -1, False
            if isinstance(declared_documents, bool) or not exact or declared_int != len(selected):
                raise ValueError("prewarm plan selection.documents must match selected_ids")
        coverage_curve = raw.get("coverage_curve") or []
        warnings = raw.get("warnings") or []
        assumptions = raw.get("assumptions") or []
        if not isinstance(coverage_curve, list):
            raise ValueError("prewarm plan coverage_curve must be a list")
        if not isinstance(warnings, list):
            raise ValueError("prewarm plan warnings must be a list")
        if not isinstance(assumptions, list):
            raise ValueError("prewarm plan assumptions must be a list")
        normalized_warnings: list[dict[str, Any]] = []
        for item in warnings:
            if isinstance(item, PrewarmWarning):
                normalized_warnings.append(item.to_dict())
            elif isinstance(item, dict):
                try:
                    normalized_warnings.append(PrewarmWarning(
                        code=item.get("code", ""), severity=item.get("severity", ""),
                        message=item.get("message", ""), remediation=item.get("remediation", "")
                    ).to_dict())
                except (TypeError, ValueError) as exc:
                    raise ValueError("prewarm plan warnings must contain structured warning objects") from exc
            else:
                raise ValueError("prewarm plan warnings must contain structured warning objects")
        if any(not isinstance(item, dict) for item in coverage_curve):
            raise ValueError("prewarm plan coverage_curve entries must be objects")
        plan = cls(
            schema_version=int(schema), migration=dict(raw.get("migration") or {}),
            window=dict(raw.get("window") or {}), strategy=str(raw.get("strategy", "")),
            baseline=dict(raw.get("baseline") or {}), selection=dict(selection_raw),
            budget=dict(raw.get("budget") or {}), cost=dict(raw.get("cost") or {}),
            coverage_curve=list(coverage_curve),
            warnings=normalized_warnings, assumptions=[str(value) for value in assumptions],
            status=str(raw.get("status", "READY")),
        )
        if plan.strategy != "traffic_hotset":
            raise ValueError("unsupported prewarm strategy in plan")
        expected = raw.get("plan_fingerprint")
        if expected is not None and str(expected) != plan.fingerprint:
            raise ValueError("prewarm plan fingerprint does not match its contents")
        return plan

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)

    def to_yaml(self) -> str:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML output requires PyYAML") from exc
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


__all__ = ["PREWARM_SCHEMA_VERSION", "MAX_INLINE_PREWARM_IDS", "PrewarmPlan", "PrewarmWarning"]
