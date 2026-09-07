from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REGISTRY_VERSION = "0.1.0"
SCHEMA_VERSION = "1"
MATCH_EXACT = "EXACT REGISTRY MATCH"
MATCH_PRIOR = "PRIOR EVIDENCE AVAILABLE"
MATCH_RELATED = "RELATED EVIDENCE ONLY"
MATCH_NONE = "NO REGISTRY MATCH"


class RegistryError(ValueError):
    """Raised when packaged evidence is malformed or internally inconsistent."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise RegistryError(f"{label} must be finite")
    return result


def _positive_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    """Parse an integer without accepting lossy values such as ``1.5``."""
    if isinstance(value, bool):
        raise RegistryError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == number
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or (number < 0 if allow_zero else number < 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise RegistryError(f"{label} must be a {qualifier} integer")
    return number


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    """Hash the behaviorally relevant embedding contract deterministically."""
    required = {
        # Keep the key names identical to ``ModelConfig.fingerprint``.  This
        # lets registry matching prove that a packaged row and a runtime
        # cache use the same semantic contract, rather than merely the same
        # model display name.
        "model": contract.get("canonical_model_id", contract.get("model", "")),
        "revision": contract.get("revision"),
        "dimension": contract.get("dimension"),
        "max_length": contract.get("max_length"),
        "pooling": contract.get("pooling"),
        "padding_side": contract.get("padding_side"),
        "truncation_side": contract.get("truncation_side"),
        "query_instruction": contract.get("query_instruction", ""),
        "document_instruction": contract.get("document_instruction", ""),
        "normalization": contract.get("normalization"),
        "dtype": contract.get("dtype", "float32"),
    }
    return hashlib.sha256(json.dumps(required, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class EvidenceRecord:
    """Validated view of one canonical migration evidence row."""

    raw: dict[str, Any]

    @property
    def evidence_id(self) -> str:
        return str(self.raw["evidence_id"])

    @property
    def source(self) -> Mapping[str, Any]:
        return self.raw["source"]

    @property
    def target(self) -> Mapping[str, Any]:
        return self.raw["target"]

    @property
    def dataset(self) -> Mapping[str, Any]:
        return self.raw["dataset"]

    @property
    def candidate_gap(self) -> dict[int, float]:
        return {int(k): float(v) for k, v in self.raw.get("candidate_gap", {}).items()}

    @property
    def containment(self) -> dict[int, float]:
        return {int(k): float(v) for k, v in self.raw.get("containment", {}).items()}

    @property
    def epsilon(self) -> float:
        return float(self.raw.get("epsilon", 0.01))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw, sort_keys=True))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvidenceRecord:
        if not isinstance(raw, Mapping):
            raise RegistryError("evidence row must be an object")
        try:
            row = json.loads(json.dumps(dict(raw), ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise RegistryError("evidence row must contain JSON-serializable values") from exc
        for key in ("registry_version", "evidence_id", "source", "target", "dataset", "metric", "provenance"):
            if key not in row:
                raise RegistryError(f"evidence row missing {key!r}")
        if str(row["registry_version"]) != REGISTRY_VERSION:
            raise RegistryError(f"{row['evidence_id']}: unsupported registry_version {row['registry_version']!r}")
        for side in ("source", "target"):
            contract = row[side]
            if not isinstance(contract, dict):
                raise RegistryError(f"{row['evidence_id']}: {side} contract must be an object")
            if not str(contract.get("canonical_model_id", "")).strip():
                raise RegistryError(f"{row['evidence_id']}: {side}.canonical_model_id is required")
            stored = contract.get("contract_fingerprint")
            if not stored:
                raise RegistryError(f"{row['evidence_id']}: {side}.contract_fingerprint is required")
            expected = contract_fingerprint(contract)
            if str(stored) != expected:
                raise RegistryError(f"{row['evidence_id']}: {side} contract fingerprint does not match contract fields")
            if contract.get("dimension") is not None:
                _positive_integer(contract["dimension"], f"{row['evidence_id']}: {side}.dimension")
        dataset = row["dataset"]
        if not isinstance(dataset, dict) or not str(dataset.get("name", "")).strip():
            raise RegistryError(f"{row['evidence_id']}: dataset.name is required")
        for key in ("corpus_size", "query_count"):
            if dataset.get(key) is not None:
                _positive_integer(dataset[key], f"{row['evidence_id']}: dataset.{key}", allow_zero=True)
        metric = str(row["metric"])
        if metric.lower() not in {"ndcg@10", "ndcg@k"}:
            raise RegistryError(f"{row['evidence_id']}: unsupported metric {metric!r}")
        epsilon = _finite(row.get("epsilon", 0.01), f"{row['evidence_id']}.epsilon")
        if epsilon < 0:
            raise RegistryError(f"{row['evidence_id']}: epsilon cannot be negative")
        gaps = row.get("candidate_gap", {}) or {}
        if not isinstance(gaps, dict):
            raise RegistryError(f"{row['evidence_id']}: candidate_gap must be an object")
        depths = []
        for key, value in gaps.items():
            try:
                depth = _positive_integer(key, f"{row['evidence_id']}: candidate depth {key!r}")
            except RegistryError as exc:
                raise RegistryError(f"{row['evidence_id']}: invalid candidate depth {key!r}") from exc
            if str(depth) != str(key):
                raise RegistryError(f"{row['evidence_id']}: candidate depths must be positive canonical integers")
            _finite(value, f"{row['evidence_id']}.candidate_gap[{key}]")
            depths.append(depth)
        if len(depths) != len(set(depths)):
            raise RegistryError(f"{row['evidence_id']}: candidate_gap depths must be unique")
        containment = row.get("containment", {}) or {}
        if not isinstance(containment, dict):
            raise RegistryError(f"{row['evidence_id']}: containment must be an object")
        for key, value in containment.items():
            try:
                containment_depth = _positive_integer(key, f"{row['evidence_id']}: containment depth {key!r}")
            except RegistryError as exc:
                raise RegistryError(f"{row['evidence_id']}: containment has invalid K={key!r}") from exc
            if containment_depth not in depths:
                raise RegistryError(f"{row['evidence_id']}: containment has unknown K={key}")
            fraction = _finite(value, f"{row['evidence_id']}.containment[{key}]")
            if not 0.0 <= fraction <= 1.0:
                raise RegistryError(f"{row['evidence_id']}: containment must be in [0, 1]")
        declared_depths = row.get("candidate_depths")
        if declared_depths is not None:
            if not isinstance(declared_depths, list):
                raise RegistryError(f"{row['evidence_id']}: candidate_depths must be a list")
            parsed_declared = [_positive_integer(value, f"{row['evidence_id']}: candidate_depths item") for value in declared_depths]
            if parsed_declared != sorted(set(parsed_declared)):
                raise RegistryError(f"{row['evidence_id']}: candidate_depths must be sorted and unique")
            if parsed_declared != sorted(depths):
                raise RegistryError(f"{row['evidence_id']}: candidate_depths disagrees with candidate_gap keys")
        observed = row.get("observed_migration_depth")
        if observed is not None:
            observed = _positive_integer(observed, f"{row['evidence_id']}: observed migration depth")
            if observed not in depths:
                raise RegistryError(f"{row['evidence_id']}: observed migration depth is not in candidate_gap")
            expected = next((k for k in sorted(depths) if float(gaps[str(k)]) <= epsilon), None)
            if expected != observed:
                raise RegistryError(f"{row['evidence_id']}: observed migration depth disagrees with candidate_gap and epsilon")
        ci = row.get("ci_certified_migration_depth")
        if ci is not None and _positive_integer(ci, f"{row['evidence_id']}: CI-certified depth") not in depths:
            raise RegistryError(f"{row['evidence_id']}: CI-certified depth is not in candidate_gap")
        provenance = row["provenance"]
        if not isinstance(provenance, dict) or not str(provenance.get("artifact", "")).strip():
            raise RegistryError(f"{row['evidence_id']}: provenance.artifact is required")
        digest = provenance.get("artifact_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower())):
            raise RegistryError(f"{row['evidence_id']}: provenance.artifact_sha256 must be a SHA-256 digest")
        return cls(row)


@dataclass(frozen=True)
class BenchmarkProfile:
    """Measured latency/throughput profile kept separate from compatibility evidence."""

    raw: dict[str, Any]

    @property
    def profile_id(self) -> str:
        return str(self.raw["profile_id"])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw, sort_keys=True))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BenchmarkProfile:
        if not isinstance(raw, Mapping):
            raise RegistryError("benchmark profile must be an object")
        try:
            row = json.loads(json.dumps(dict(raw), ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise RegistryError("benchmark profile must contain JSON-serializable values") from exc
        for key in ("registry_version", "profile_id", "kind", "provenance", "measurements"):
            if key not in row:
                raise RegistryError(f"benchmark profile missing {key!r}")
        if not str(row["profile_id"]).strip():
            raise RegistryError("benchmark profile profile_id must be non-empty")
        if not str(row["kind"]).strip():
            raise RegistryError(f"{row['profile_id']}: kind must be non-empty")
        if str(row["registry_version"]) != REGISTRY_VERSION:
            raise RegistryError(f"{row['profile_id']}: unsupported registry_version")
        if not isinstance(row["measurements"], dict):
            raise RegistryError(f"{row['profile_id']}: measurements must be an object")
        if not isinstance(row["provenance"], dict) or not row["provenance"].get("artifact"):
            raise RegistryError(f"{row['profile_id']}: provenance.artifact is required")
        digest = row["provenance"].get("artifact_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower())):
            raise RegistryError(f"{row['profile_id']}: provenance.artifact_sha256 must be a SHA-256 digest")

        def validate_measurement(value: Any, label: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    validate_measurement(child, f"{label}.{key}")
                return
            if value is None:
                return
            if isinstance(value, bool):
                raise RegistryError(f"{row['profile_id']}: {label} must be numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise RegistryError(f"{row['profile_id']}: {label} must be numeric") from exc
            if not math.isfinite(numeric):
                raise RegistryError(f"{row['profile_id']}: {label} must be finite")

        validate_measurement(row["measurements"], "measurements")
        return cls(row)


def canonical_json_hash(value: Any) -> str:
    """Hash a JSON value for manifest/checksum use."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
