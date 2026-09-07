from __future__ import annotations

import csv
import hashlib
import importlib.resources as resources
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .schema import REGISTRY_VERSION, SCHEMA_VERSION, BenchmarkProfile, EvidenceRecord, RegistryError


def _resource_root() -> Any:
    return resources.files("embedflow.data.registry")


def _read_json(name: str, root: Any | None = None) -> dict[str, Any]:
    resource = (root or _resource_root()).joinpath(name)
    try:
        return json.loads(resource.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"registry file is missing: {name}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"registry file is not valid JSON: {name}: {exc}") from exc


def _read_jsonl(name: str, root: Any | None = None) -> list[dict[str, Any]]:
    resource = (root or _resource_root()).joinpath(name)
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RegistryError(f"registry file is missing: {name}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RegistryError(f"{name}:{line_number}: each row must be a JSON object")
        rows.append(value)
    return rows


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the packaged registry manifest or a directory-level manifest."""
    if path is None:
        return _read_json("registry_manifest.json")
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "registry_manifest.json"
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry manifest is not valid JSON: {candidate}") from exc
    if not isinstance(payload, dict):
        raise RegistryError(f"registry manifest must be a JSON object: {candidate}")
    return payload


def _root_for(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_dir() else candidate.parent


def load_evidence(path: str | Path | None = None) -> list[EvidenceRecord]:
    """Load and validate deterministic migration records."""
    root = _root_for(path)
    name = "migrations.jsonl"
    rows = _read_jsonl(name, root) if root is not None else _read_jsonl(name)
    evidence = [EvidenceRecord.from_dict(row) for row in rows]
    return sorted(evidence, key=lambda item: item.evidence_id)


def load_benchmark_profiles(path: str | Path | None = None) -> list[BenchmarkProfile]:
    """Load measured latency/throughput profiles, separate from compatibility rows."""
    root = _root_for(path)
    name = "benchmark_profiles.jsonl"
    rows = _read_jsonl(name, root) if root is not None else _read_jsonl(name)
    profiles = [BenchmarkProfile.from_dict(row) for row in rows]
    return sorted(profiles, key=lambda item: item.profile_id)


def load_summaries(path: str | Path | None = None) -> list[dict[str, Any]]:
    root = _root_for(path)
    name = "research_summaries.json"
    payload = _read_json(name, root) if root is not None else _read_json(name)
    summaries = payload.get("summaries", [])
    if not isinstance(summaries, list) or not all(isinstance(row, dict) for row in summaries):
        raise RegistryError("research_summaries.json field 'summaries' must be a list of objects")
    return list(summaries)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resource_bytes(root: Any, name: str) -> bytes:
    return root.joinpath(name).read_bytes()


def _verify_checksums(root: Any, manifest: Mapping[str, Any], errors: list[str]) -> int:
    checked = 0
    files = manifest.get("files") or {}
    if not isinstance(files, Mapping):
        errors.append("manifest field 'files' must be an object")
        files = {}
    for name, expected in files.items():
        if not isinstance(expected, dict) or not expected.get("sha256"):
            errors.append(f"manifest entry {name!r} lacks sha256")
            continue
        try:
            actual = _sha256_bytes(_resource_bytes(root, name))
        except FileNotFoundError:
            errors.append(f"manifest file is missing: {name}")
            continue
        checked += 1
        if actual != str(expected["sha256"]):
            errors.append(f"checksum mismatch for {name}: expected {expected['sha256']}, got {actual}")
    checksum_resource = root.joinpath("checksums.sha256")
    if checksum_resource.is_file():
        for line_number, line in enumerate(checksum_resource.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or len(parts[0]) != 64:
                errors.append(f"checksums.sha256:{line_number}: malformed checksum line")
                continue
            name, expected = parts[1].lstrip(" *"), parts[0]
            try:
                actual = _sha256_bytes(_resource_bytes(root, name))
            except FileNotFoundError:
                errors.append(f"checksums.sha256:{line_number}: missing {name}")
                continue
            if actual != expected:
                errors.append(f"checksums.sha256:{line_number}: checksum mismatch for {name}")
    return checked


def _verify_external_provenance(
    provenance_root: str | Path | None,
    evidence: list[EvidenceRecord],
    profiles: list[BenchmarkProfile],
    summaries: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> int:
    """Check retained artifact bytes when the research checkout is present.

    Installed users normally do not have the private/research checkout, so a
    missing external artifact is a warning.  A present artifact with a wrong
    digest is a release-blocking error: packaged numbers must remain tied to
    the exact retained file from which they were transcribed.
    """
    if provenance_root is None:
        return 0
    root = Path(provenance_root).expanduser().resolve()
    checked = 0
    rows: list[tuple[str, Mapping[str, Any]]] = []
    rows.extend((row.evidence_id, row.raw.get("provenance", {})) for row in evidence)
    rows.extend((profile.profile_id, profile.raw.get("provenance", {})) for profile in profiles)
    rows.extend((str(row.get("summary_id", "summary")), row.get("provenance", {})) for row in summaries)
    for identifier, provenance in rows:
        if not isinstance(provenance, Mapping):
            warnings.append(f"{identifier}: external artifact digest is unavailable")
            continue
        # A summary may name a primary row-level table and a supporting audit
        # table (for example, the 63-cell result table plus its CI headline
        # audit).  Verify every declared digest rather than silently trusting
        # only the first path.
        artifacts = [("artifact", provenance.get("artifact"), provenance.get("artifact_sha256"))]
        if provenance.get("supporting_artifact") or provenance.get("supporting_artifact_sha256"):
            artifacts.append(("supporting_artifact", provenance.get("supporting_artifact"), provenance.get("supporting_artifact_sha256")))
        for label, artifact, expected in artifacts:
            if not artifact or not expected:
                warnings.append(f"{identifier}: {label} digest is unavailable")
                continue
            candidate = Path(str(artifact))
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.exists():
                warnings.append(f"{identifier}: retained {label} not found at {candidate}")
                continue
            try:
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError as exc:
                errors.append(f"{identifier}: cannot read retained {label} {candidate}: {exc}")
                continue
            checked += 1
            if actual != str(expected):
                errors.append(f"{identifier}: retained {label} checksum mismatch: expected {expected}, got {actual}")
    return checked


def _artifact_path(provenance_root: Path, artifact: Any) -> Path:
    """Resolve a provenance path relative to the research checkout root."""
    candidate = Path(str(artifact)).expanduser()
    return candidate if candidate.is_absolute() else provenance_root / candidate


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise RegistryError(f"{label} is not finite")
    return result


def _same_measurement(actual: Any, expected: Any) -> bool:
    """Compare retained decimal output without requiring byte-identical floats."""
    try:
        actual_value = _as_float(actual, "artifact value")
        expected_value = _as_float(expected, "registry value")
    except RegistryError:
        return False
    return math.isclose(actual_value, expected_value, rel_tol=1e-10, abs_tol=1e-12)


def _verify_curve_row(record: EvidenceRecord, row: Mapping[str, Any], *, source_fields: tuple[str, str], gap_field: str, containment_field: str, errors: list[str]) -> int:
    """Verify one registry migration row against one retained curve row."""
    checked = 0
    try:
        k = int(row["K"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{record.evidence_id}: retained curve row has an invalid K")
        return checked
    expected_gap = record.candidate_gap.get(k)
    if expected_gap is None:
        errors.append(f"{record.evidence_id}: registry has no candidate_gap for retained K={k}")
        return checked
    comparisons = (
        (record.raw.get("source_quality"), source_fields[0], "source quality"),
        (record.raw.get("native_target_quality"), source_fields[1], "native target quality"),
        (record.raw.get("restricted_target_quality", {}).get(str(k)), "restricted_ndcg" if gap_field == "G" else "restricted_target_ndcg", "restricted target quality"),
        (expected_gap, gap_field, "candidate gap"),
    )
    for expected, field, label in comparisons:
        if expected is None or field not in row or not _same_measurement(row[field], expected):
            errors.append(f"{record.evidence_id}: retained {label} disagrees at K={k}")
        else:
            checked += 1
    expected_containment = record.containment.get(k)
    if expected_containment is not None:
        if containment_field not in row or not _same_measurement(row[containment_field], expected_containment):
            errors.append(f"{record.evidence_id}: retained containment disagrees at K={k}")
        else:
            checked += 1
    if record.dataset.get("query_count") is not None:
        try:
            if int(row.get("n_queries", row.get("queries"))) != int(record.dataset["query_count"]):
                errors.append(f"{record.evidence_id}: retained query count disagrees")
            else:
                checked += 1
        except (TypeError, ValueError):
            errors.append(f"{record.evidence_id}: retained query count is invalid")
    return checked


def _verify_semantic_provenance(
    provenance_root: str | Path | None,
    evidence: list[EvidenceRecord],
    profiles: list[BenchmarkProfile],
    summaries: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> int:
    """Cross-check transcribed registry values against known retained tables.

    Checksums establish that an artifact was not changed; these checks establish
    that the values in the public registry are actually the values in that
    artifact.  Unknown future artifact formats are left to contributors rather
    than guessed here.
    """
    if provenance_root is None:
        return 0
    root = Path(provenance_root).expanduser().resolve()
    checked = 0
    by_artifact: dict[str, list[EvidenceRecord]] = {}
    for record in evidence:
        artifact = record.raw.get("provenance", {}).get("artifact")
        if artifact:
            by_artifact.setdefault(str(artifact), []).append(record)
    for artifact, records in by_artifact.items():
        path = _artifact_path(root, artifact)
        if not path.is_file():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            errors.append(f"semantic provenance: cannot parse {path}: {exc}")
            continue
        name = path.name
        if name == "scale_curves.csv":
            pair_by_source = {
                "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6_to_qwen3_8b",
                "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b_to_qwen3_8b",
                "Qwen/Qwen3-Embedding-4B": "qwen3_4b_to_qwen3_8b",
            }
            for record in records:
                pair = pair_by_source.get(str(record.source.get("canonical_model_id")))
                corpus_size = record.dataset.get("corpus_size")
                if pair is None or corpus_size is None:
                    warnings.append(f"{record.evidence_id}: no semantic parser mapping for {name}")
                    continue
                matches = [r for r in rows if r.get("pair") == pair and r.get("corpus_size") == str(corpus_size)]
                if len(matches) != len(record.candidate_gap):
                    errors.append(f"{record.evidence_id}: retained scale curve has {len(matches)} rows; expected {len(record.candidate_gap)}")
                    continue
                for row in matches:
                    checked += _verify_curve_row(record, row, source_fields=("source_ndcg", "native_target_ndcg"), gap_field="G", containment_field="containment10", errors=errors)
        elif name == "bright_pooled_appendix_values.csv":
            source_by_key = {
                "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
                "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b",
                "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
            }
            for record in records:
                source_key = source_by_key.get(str(record.source.get("canonical_model_id")))
                if source_key is None:
                    warnings.append(f"{record.evidence_id}: no semantic parser mapping for {name}")
                    continue
                matches = [r for r in rows if r.get("source") == source_key]
                if len(matches) != len(record.candidate_gap):
                    errors.append(f"{record.evidence_id}: retained BRIGHT curve has {len(matches)} rows; expected {len(record.candidate_gap)}")
                    continue
                by_k: dict[int, Mapping[str, Any]] = {}
                invalid_k = False
                for retained in matches:
                    if not retained.get("K"):
                        invalid_k = True
                        continue
                    try:
                        parsed_k = int(retained["K"])
                    except (TypeError, ValueError):
                        invalid_k = True
                        continue
                    if parsed_k in by_k:
                        invalid_k = True
                    by_k[parsed_k] = retained
                if invalid_k:
                    errors.append(f"{record.evidence_id}: retained BRIGHT curve contains an invalid or duplicate K")
                if set(by_k) != set(record.candidate_gap):
                    errors.append(f"{record.evidence_id}: retained BRIGHT K grid disagrees")
                    continue
                for row in matches:
                    checked += _verify_curve_row(record, row, source_fields=("source_ndcg", "target_ndcg"), gap_field="absolute_candidate_gap", containment_field="target_top10_containment", errors=errors)
                # The BRIGHT appendix explicitly records the two depth notions;
                # check both against the first qualifying K, preserving null.
                point_depth = next((k for k, r in by_k.items() if str(r.get("observed_near_target_g_le_0_01", "")).lower() == "true"), None)
                ci_depth = next((k for k, r in by_k.items() if str(r.get("target_noninferior_margin_0_01", "")).lower() == "true"), None)
                if record.raw.get("observed_migration_depth") != point_depth:
                    errors.append(f"{record.evidence_id}: retained BRIGHT observed depth disagrees")
                else:
                    checked += 1
                if record.raw.get("ci_certified_migration_depth") != ci_depth:
                    errors.append(f"{record.evidence_id}: retained BRIGHT CI-certified depth disagrees")
                else:
                    checked += 1
        else:
            warnings.append(f"semantic provenance: no parser for retained artifact {path}")
    # Measured serving profiles are kept separate from migration evidence, but
    # their headline values are checked against the retained latency/profile
    # artifacts as well.
    profile_by_artifact: dict[str, list[BenchmarkProfile]] = {}
    for profile in profiles:
        artifact = profile.raw.get("provenance", {}).get("artifact")
        if artifact:
            profile_by_artifact.setdefault(str(artifact), []).append(profile)
    for artifact, profile_rows in profile_by_artifact.items():
        path = _artifact_path(root, artifact)
        if not path.is_file():
            continue
        if path.name == "latency_summary.csv":
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    latency_rows = list(csv.DictReader(handle))
            except (OSError, csv.Error) as exc:
                errors.append(f"semantic provenance: cannot parse {path}: {exc}")
                continue
            for profile in profile_rows:
                config = profile.raw.get("configuration") or {}
                if profile.profile_id.startswith("latency_25k_qwen3_4b"):
                    selected = [r for r in latency_rows if r.get("source_model") == "qwen3_4b" and r.get("mode") == "embedflow_warm" and r.get("K") == str(config.get("K")) and r.get("nprobe") == str(config.get("nprobe"))]
                elif profile.profile_id.startswith("latency_25k_native"):
                    selected = [r for r in latency_rows if r.get("source_model") == "native_target" and r.get("mode") == "native_target"]
                else:
                    warnings.append(f"{profile.profile_id}: no semantic parser mapping for {path.name}")
                    continue
                if len(selected) != 1:
                    errors.append(f"{profile.profile_id}: retained latency selection has {len(selected)} rows")
                    continue
                row = selected[0]
                measurements = profile.raw.get("measurements") or {}
                for key in ("p50_ms", "p95_ms"):
                    if key not in measurements or not _same_measurement(row.get(key), measurements[key]):
                        errors.append(f"{profile.profile_id}: retained {key} disagrees")
                    else:
                        checked += 1
                if "added_vs_native_p50_ms" in measurements:
                    native = next((r for r in latency_rows if r.get("source_model") == "native_target" and r.get("mode") == "native_target"), None)
                    try:
                        actual_added = float(row["p50_ms"]) - float(native["p50_ms"]) if native is not None else math.nan
                    except (KeyError, TypeError, ValueError):
                        actual_added = math.nan
                    if not _same_measurement(actual_added, measurements["added_vs_native_p50_ms"]):
                        errors.append(f"{profile.profile_id}: retained added p50 disagrees")
                    else:
                        checked += 1
                if "added_vs_native_p95_ms" in measurements:
                    native = next((r for r in latency_rows if r.get("source_model") == "native_target" and r.get("mode") == "native_target"), None)
                    try:
                        actual_added = float(row["p95_ms"]) - float(native["p95_ms"]) if native is not None else math.nan
                    except (KeyError, TypeError, ValueError):
                        actual_added = math.nan
                    if not _same_measurement(actual_added, measurements["added_vs_native_p95_ms"]):
                        errors.append(f"{profile.profile_id}: retained added p95 disagrees")
                    else:
                        checked += 1
                for registry_name, csv_name in {
                    "source_query_encode": "mean_source_query_encode_ms",
                    "source_ann": "mean_source_ann_search_ms",
                    "target_query_encode": "mean_target_query_encode_ms",
                    "target_score": "mean_target_score_ms",
                    "topk": "mean_topk_ms",
                }.items():
                    expected = (measurements.get("mean_stage_ms") or {}).get(registry_name)
                    if expected is not None:
                        if not _same_measurement(row.get(csv_name), expected):
                            errors.append(f"{profile.profile_id}: retained stage {registry_name} disagrees")
                        else:
                            checked += 1
                expected = (measurements.get("mean_stage_ms") or {}).get("candidate_cache_lookup")
                if expected is not None:
                    try:
                        actual = float(row.get("mean_candidate_lookup_ms", 0.0)) + float(row.get("mean_cache_lookup_ms", 0.0))
                    except (TypeError, ValueError):
                        actual = math.nan
                    if not _same_measurement(actual, expected):
                        errors.append(f"{profile.profile_id}: retained stage candidate_cache_lookup disagrees")
                    else:
                        checked += 1
        elif path.name == "qwen3_8b_latency.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                batch_rows = payload.get("batch_sizes", [])
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"semantic provenance: cannot parse {path}: {exc}")
                continue
            if not isinstance(batch_rows, list) or not all(isinstance(row, Mapping) for row in batch_rows):
                errors.append(f"semantic provenance: {path} batch_sizes must be a list of objects")
                continue
            for profile in profile_rows:
                measurements = profile.raw.get("measurements") or {}
                best_batch = measurements.get("best_batch_size")
                selected = [r for r in batch_rows if r.get("batch_size") == best_batch]
                if len(selected) != 1:
                    errors.append(f"{profile.profile_id}: retained throughput selection has {len(selected)} rows")
                    continue
                row = selected[0]
                for registry_name, artifact_name in (("best_rows_per_second", "rows_per_second"), ("best_mean_seconds", "mean_seconds")):
                    if not _same_measurement(row.get(artifact_name), measurements.get(registry_name)):
                        errors.append(f"{profile.profile_id}: retained {registry_name} disagrees")
                    else:
                        checked += 1
                all_rows = measurements.get("all_batch_rows_per_second") or {}
                for batch, expected in all_rows.items():
                    match = next((r for r in batch_rows if str(r.get("batch_size")) == str(batch)), None)
                    if match is None or not _same_measurement(match.get("rows_per_second"), expected):
                        errors.append(f"{profile.profile_id}: retained throughput disagrees for batch {batch}")
                    else:
                        checked += 1
        else:
            warnings.append(f"semantic provenance: no parser for retained profile artifact {path}")

    # Summary values are count audits rather than curves.  They are already
    # byte-hash protected; verify the expected shape and headline counts when
    # the canonical files are available so a swapped summary cannot pass
    # unnoticed.  The development audit has two retained representations: the
    # 63-row result table carries G(50), while the appendix table carries the
    # existing CI-certification flag.
    for summary in summaries:
        provenance = summary.get("provenance") or {}
        declared_artifacts: list[tuple[str, Any]] = []
        if provenance.get("artifact"):
            declared_artifacts.append(("artifact", provenance.get("artifact")))
        if provenance.get("supporting_artifact"):
            declared_artifacts.append(("supporting_artifact", provenance.get("supporting_artifact")))
        for artifact_label, artifact in declared_artifacts:
            path = _artifact_path(root, artifact)
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, csv.Error) as exc:
                errors.append(f"semantic provenance: cannot parse {path}: {exc}")
                continue
            summary_id = str(summary.get("summary_id"))
            if summary.get("kind") == "development_sweep_summary":
                expected_cells = int(summary.get("cells", -1))
                if len(rows) != expected_cells:
                    errors.append(f"{summary_id}: retained {artifact_label} development row count {len(rows)} != {expected_cells}")
                    continue
                checked += 1
                # Both the production result table and the appendix table
                # encode the point-estimate G(50) under different names.
                gap_field = "exact_top50_gap_ndcg" if rows and "exact_top50_gap_ndcg" in rows[0] else "signed_candidate_gap_g50"
                if rows and gap_field in rows[0]:
                    try:
                        point_count = sum(_as_float(row.get(gap_field), f"{path} {gap_field}") <= 0.01 for row in rows)
                    except RegistryError as exc:
                        errors.append(f"{summary_id}: invalid development gap value in {path}: {exc}")
                    else:
                        declared_point = int(summary.get("point_estimate_g50_le_0_01", -1))
                        if point_count != declared_point:
                            errors.append(f"{summary_id}: retained point-estimate count {point_count} != {declared_point}")
                        else:
                            checked += 1
                        non_nano = [row for row in rows if str(row.get("dataset", "")).lower() not in {"nanoquora", "nanomsmarco"}]
                        expected_excluded = summary.get("excluding_nanoquora_and_nanomsmarco", {}).get("point_estimate_g50_le_0_01")
                        if expected_excluded is not None:
                            try:
                                excluded_count = sum(_as_float(row.get(gap_field), f"{path} {gap_field}") <= 0.01 for row in non_nano)
                            except RegistryError as exc:
                                errors.append(f"{summary_id}: invalid non-Nano development gap value in {path}: {exc}")
                            else:
                                if excluded_count != int(expected_excluded):
                                    errors.append(f"{summary_id}: retained non-Nano point count {excluded_count} != {expected_excluded}")
                                else:
                                    checked += 1
                # Only the appendix representation contains the canonical
                # existing CI flag.  Do not substitute a different budget
                # column from the development result table.
                ci_field = "ci_noninferior_margin_0_01" if rows and "ci_noninferior_margin_0_01" in rows[0] else None
                if ci_field:
                    ci_count = sum(str(row.get(ci_field, "")).strip().lower() == "true" for row in rows)
                    declared_ci = int(summary.get("ci_certified_cells", -1))
                    if ci_count != declared_ci:
                        errors.append(f"{summary_id}: retained CI-certified count {ci_count} != {declared_ci}")
                    else:
                        checked += 1
                    non_nano = [row for row in rows if str(row.get("dataset", "")).lower() not in {"nanoquora", "nanomsmarco"}]
                    expected_excluded_ci = summary.get("excluding_nanoquora_and_nanomsmarco", {}).get("ci_certified_cells")
                    if expected_excluded_ci is not None:
                        excluded_ci_count = sum(str(row.get(ci_field, "")).strip().lower() == "true" for row in non_nano)
                        if excluded_ci_count != int(expected_excluded_ci):
                            errors.append(f"{summary_id}: retained non-Nano CI count {excluded_ci_count} != {expected_excluded_ci}")
                        else:
                            checked += 1
            elif summary.get("kind") == "t2_holdout_summary":
                expected_cells = int(summary.get("cells", -1))
                if len(rows) != expected_cells:
                    errors.append(f"{summary_id}: retained T2 row count {len(rows)} != {expected_cells}")
                    continue
                checked += 1
                def _count_true(field: str) -> int:
                    return sum(str(row.get(field, "")).strip().lower() == "true" for row in rows)
                checks = (("compatible_cells", "compatible_eps_001"), ("safe_predictions", "prediction"), ("observed_false_safe_predictions", "false_safe"))
                for summary_key, field in checks:
                    if summary_key == "safe_predictions":
                        actual = sum(str(row.get(field, "")).strip().upper() == "SAFE" for row in rows)
                    else:
                        actual = _count_true(field)
                    expected_value = int(summary.get(summary_key, -1))
                    if actual != expected_value:
                        errors.append(f"{summary_id}: retained {summary_key} count {actual} != {expected_value}")
                    else:
                        checked += 1
                # Verify the four published source/target group totals when
                # the ledger carries those identifiers.  This catches a row
                # swap that preserves only the aggregate 28/22/17/0 counts.
                aliases = {"qwen3_0_6b": "Qwen3-Embedding-0.6B", "qwen3_4b": "Qwen3-Embedding-4B", "qwen3_8b": "Qwen3-Embedding-8B", "minilm_l6": "MiniLM-L6"}
                for group in summary.get("groups", []):
                    expected_source = str(group.get("source", ""))
                    expected_target = str(group.get("target", ""))
                    selected = [row for row in rows if aliases.get(str(row.get("source_model")), str(row.get("source_model"))) == expected_source and aliases.get(str(row.get("target_model")), str(row.get("target_model"))) == expected_target]
                    expected_group_cells = int(group.get("cells", -1))
                    if len(selected) != expected_group_cells:
                        errors.append(f"{summary_id}: group {expected_source}->{expected_target} has {len(selected)} rows != {expected_group_cells}")
                        continue
                    for key, field, predicate in (("compatible", "compatible_eps_001", lambda value: str(value).lower() == "true"), ("safe", "prediction", lambda value: str(value).upper() == "SAFE"), ("false_safe", "false_safe", lambda value: str(value).lower() == "true")):
                        actual = sum(predicate(row.get(field, "")) for row in selected)
                        if actual != int(group.get(key, -1)):
                            errors.append(f"{summary_id}: group {expected_source}->{expected_target} {key} count {actual} != {group.get(key)}")
                        else:
                            checked += 1
    return checked


def verify_registry(path: str | Path | None = None, *, provenance_root: str | Path | None = None) -> dict[str, Any]:
    """Validate schema, hashes, consistency, and provenance metadata.

    Provenance paths point to the retained research checkout and are not
    expected to exist in an installed package.  The recorded artifact digest
    is checked for shape, while the packaged bytes are checked by the manifest.
    When ``provenance_root`` is available, known canonical CSV summaries are
    also checked field-by-field against the transcribed registry values.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if path is None:
        root = _resource_root()
        manifest = load_manifest()
    else:
        root = _root_for(path)
        try:
            manifest = load_manifest(path)
        except Exception as exc:
            return {"ok": False, "errors": [str(exc)], "warnings": [], "record_count": 0, "profile_count": 0}
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    if str(manifest.get("registry_version")) != REGISTRY_VERSION:
        errors.append(f"unsupported registry_version: {manifest.get('registry_version')!r}")
    # Keep the standalone schema marker meaningful.  It is shipped alongside
    # the data so downstream tooling can reject a registry before attempting
    # to parse rows.
    try:
        schema_marker = _read_json("schema_version.json", root)
        if str(schema_marker.get("schema_version")) != SCHEMA_VERSION:
            errors.append(f"schema_version.json declares {schema_marker.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
        if str(schema_marker.get("registry_version")) != REGISTRY_VERSION:
            errors.append(f"schema_version.json declares registry_version {schema_marker.get('registry_version')!r}, expected {REGISTRY_VERSION!r}")
    except Exception as exc:
        errors.append(str(exc))
    checked_files = _verify_checksums(root, manifest, errors)
    try:
        evidence = load_evidence(path)
    except Exception as exc:
        evidence, profiles = [], []
        errors.append(str(exc))
    else:
        try:
            profiles = load_benchmark_profiles(path)
        except Exception as exc:
            profiles = []
            errors.append(str(exc))
    expected_count = manifest.get("record_count")
    if expected_count is not None:
        try:
            count_value = int(expected_count)
        except (TypeError, ValueError, OverflowError):
            errors.append("manifest record_count must be an integer")
        else:
            if count_value != len(evidence):
                errors.append(f"manifest record_count={expected_count} but loaded {len(evidence)}")
    expected_profiles = manifest.get("profile_count")
    if expected_profiles is not None:
        try:
            profiles_value = int(expected_profiles)
        except (TypeError, ValueError, OverflowError):
            errors.append("manifest profile_count must be an integer")
        else:
            if profiles_value != len(profiles):
                errors.append(f"manifest profile_count={expected_profiles} but loaded {len(profiles)}")
    try:
        summaries = load_summaries(path)
    except Exception as exc:
        summaries = []
        errors.append(str(exc))
    expected_summaries = manifest.get("summary_count")
    if expected_summaries is not None:
        try:
            summaries_value = int(expected_summaries)
        except (TypeError, ValueError, OverflowError):
            errors.append("manifest summary_count must be an integer")
        else:
            if summaries_value != len(summaries):
                errors.append(f"manifest summary_count={expected_summaries} but loaded {len(summaries)}")
    for summary in summaries:
        if not isinstance(summary, dict) or not str(summary.get("summary_id", "")).strip():
            errors.append("research summary lacks summary_id")
            continue
        provenance = summary.get("provenance") or {}
        for digest_name in ("artifact_sha256", "supporting_artifact_sha256"):
            digest = provenance.get(digest_name)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
                errors.append(f"{summary['summary_id']}: invalid provenance {digest_name} digest")
    ids = [row.evidence_id for row in evidence]
    if len(ids) != len(set(ids)):
        errors.append("evidence_id values are not unique")
    profile_ids = [profile.profile_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("profile_id values are not unique")
    for row in evidence:
        provenance = row.raw.get("provenance", {})
        digest = provenance.get("artifact_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
            errors.append(f"{row.evidence_id}: provenance.artifact_sha256 is not a SHA-256 digest")
        if provenance.get("status") not in {"retained_external", "packaged", "derived_from_retained_external"}:
            errors.append(f"{row.evidence_id}: provenance.status must declare whether the artifact is retained externally")
        ann = row.raw.get("ann") or {}
        if ann.get("status") == "UNKNOWN" and ann.get("tested_configurations"):
            warnings.append(f"{row.evidence_id}: ANN status UNKNOWN despite tested configurations; review semantics")
    for profile in profiles:
        provenance = profile.raw.get("provenance", {})
        digest = provenance.get("artifact_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            errors.append(f"{profile.profile_id}: invalid provenance artifact digest")
    provenance_checked = _verify_external_provenance(provenance_root, evidence, profiles, summaries, errors, warnings)
    semantic_checked = _verify_semantic_provenance(provenance_root, evidence, profiles, summaries, errors, warnings)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(evidence),
        "profile_count": len(profiles),
        "summary_count": len(summaries),
        "checked_files": checked_files,
        "provenance_artifacts_checked": provenance_checked,
        "semantic_values_checked": semantic_checked,
        "registry_version": manifest.get("registry_version"),
        "schema_version": manifest.get("schema_version"),
    }


def _canonical_record(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def dataset_fingerprint(
    path: str | Path,
    *,
    id_field: str = "id",
    text_field: str = "text",
    query_path: str | Path | None = None,
    qrels_path: str | Path | None = None,
) -> str:
    """Create a stable fingerprint for JSONL corpus/query/qrels construction.

    The hash includes field names, canonicalized row values, and row counts.
    It is deliberately not inferred from a dataset name alone.
    """
    digest = hashlib.sha256()
    count = 0
    digest.update(f"documents:{id_field}:{text_field}\n".encode())
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or id_field not in row or text_field not in row:
                raise ValueError(f"row {line_number} in {path} lacks {id_field!r}/{text_field!r}")
            digest.update(_canonical_record({"id": str(row[id_field]), "text": str(row[text_field])}))
            count += 1
    digest.update(f"documents_count:{count}\n".encode())
    for label, extra in (("queries", query_path), ("qrels", qrels_path)):
        if extra is None:
            continue
        extra_digest = hashlib.sha256(Path(extra).read_bytes()).hexdigest()
        digest.update(f"{label}:{extra_digest}\n".encode())
    return digest.hexdigest()


def find_evidence(
    *,
    source_model: Any,
    target_model: Any,
    corpus_fingerprint: str | None = None,
    corpus_name: str | None = None,
    corpus_size: int | None = None,
    records: Iterable[EvidenceRecord] | None = None,
):
    """Find evidence using the public matching API (lazy import avoids cycles)."""
    from .matcher import match_evidence

    return match_evidence(
        source_model=source_model,
        target_model=target_model,
        corpus_fingerprint=corpus_fingerprint,
        corpus_name=corpus_name,
        corpus_size=corpus_size,
        records=list(records) if records is not None else load_evidence(),
    )
