"""Structured models returned by the advisory migration planner.

The planner is intentionally separate from :mod:`embedflow.migration.planner`,
which contains the small serving-plan object used by the runtime.  These
objects describe an analysis result and are safe to serialize as JSON/YAML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanWarning:
    """A first-class planner warning with an actionable remediation."""

    code: str
    severity: str
    message: str
    remediation: str

    def __post_init__(self) -> None:
        severity = str(self.severity).upper()
        if severity not in {"INFO", "WARN", "ERROR"}:
            raise ValueError("planner warning severity must be INFO, WARN, or ERROR")
        object.__setattr__(self, "severity", severity)
        if not str(self.code).strip() or not str(self.message).strip():
            raise ValueError("planner warning code and message are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    """Typed, serializable migration plan.

    Nested sections remain dictionaries on purpose: backend metadata and future
    planner fields can evolve without changing the core migration engine or
    forcing users to parse terminal strings.
    """

    schema_version: int
    recommendation: str
    confidence: str
    source: dict[str, Any]
    target: dict[str, Any]
    preflight: dict[str, Any]
    evidence: dict[str, Any]
    candidate_depth: dict[str, Any]
    cache: dict[str, Any]
    performance: dict[str, Any]
    economics: dict[str, Any]
    rollout: dict[str, Any]
    warnings: list[PlanWarning | dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, PlanWarning):
                return value.to_dict()
            if isinstance(value, dict):
                return {str(key): convert(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(child) for child in value]
            return value

        return convert(asdict(self))

    @property
    def recommended_k(self) -> int | None:
        value = self.candidate_depth.get("recommended_k")
        return None if value is None else int(value)

    @property
    def t2_status(self) -> str:
        return str(self.candidate_depth.get("t2_status", "NOT_RUN"))

    @property
    def status(self) -> str:
        """Compatibility alias for callers that used the old plan object."""
        return self.recommendation

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)

    def to_yaml(self) -> str:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - PyYAML is a base dependency
            raise RuntimeError("YAML output requires PyYAML") from exc
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


__all__ = ["PlanResult", "PlanWarning"]
