from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class T2Diagnostic:
    """A frozen finite-tail diagnostic, never a compatibility guarantee."""

    diagnostic: str
    features: dict[str, float]
    implementation: str
    warning: str = "T2-v1 is an empirical finite-tail diagnostic, not a compatibility guarantee."

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic": self.diagnostic,
            "features": self.features,
            "implementation": self.implementation,
            "warning": self.warning,
        }


def diagnose_t2(features: Mapping[str, Any], root: str | Path | None = None) -> T2Diagnostic:
    """Run the repository's frozen T2-v1 rule without reimplementing it."""
    from src.t2_v1 import decide, verify_t2_hash

    if not isinstance(features, Mapping):
        raise TypeError("T2-v1 features must be a mapping")
    forbidden = ("qrel", "native", "ndcg", "candidate_gap", "g(", "target_rank", "label", "observed_k")
    leaked = [str(key) for key in features if any(token in str(key).lower() for token in forbidden)]
    if leaked:
        raise ValueError(f"T2-v1 refuses label/native-target data in features: {leaked}")
    required = {
        "probe_residual_tail_50_mean", "deepest_p90", "late_tail_area",
        "stability_to_500_50_mean", "last_shell_any_rate", "fraction_margin_nonpositive",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"T2-v1 feature row is missing required fields: {missing}")

    if root is None:
        root = Path(__file__).resolve().parents[2]
        if not (Path(root) / "frozen").exists():
            root = Path(__file__).resolve().parents[1]
    root = Path(root)
    frozen = root / "frozen" / "T2_V1_FROZEN_SPEC.md"
    if frozen.exists():
        verify_t2_hash(root)
    scalar = {}
    for key, value in features.items():
        if isinstance(value, bool):
            raise ValueError(f"T2-v1 feature {key!r} must be numeric, not boolean")
        try:
            scalar[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T2-v1 feature {key!r} must be numeric") from exc
    import math
    if not all(math.isfinite(value) for value in scalar.values()):
        raise ValueError("T2-v1 features must be finite")
    return T2Diagnostic(decide(scalar), scalar, "src.t2_v1.decide")
