"""Pure, provenance-aware economics helpers for migration planning."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def _nonnegative_int(value: Any, label: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
        exact = float(value) == parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if not exact or parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _nonnegative_float(value: Any, label: str, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return parsed


def quantity(value: Any, unit: str, provenance: str, *, assumptions: Iterable[str] = ()) -> dict[str, Any]:
    """Attach provenance to one measured, supplied, modeled, or unknown value."""
    return {
        "value": value,
        "unit": str(unit),
        "provenance": str(provenance).lower(),
        "assumptions": [str(item) for item in assumptions],
    }


def estimate_economics(
    corpus_size: int | None,
    target_dimension: int | None,
    *,
    dtype: str = "float32",
    docs_per_second: float | None = None,
    gpu_hourly_cost: float | None = None,
    cached_documents: int = 0,
    scenarios: Iterable[float] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0),
) -> dict[str, Any]:
    """Return non-negative storage/time/cost projections with explicit provenance.

    Missing throughput, price, corpus size, or dimension stays ``UNKNOWN``;
    no hardware or cloud price is guessed.  Raw vector bytes intentionally do
    not include ANN indexes, metadata, replicas, or backups.
    """
    corpus = _nonnegative_int(corpus_size, "corpus_size", allow_none=True)
    dimension = _nonnegative_int(target_dimension, "target_dimension", allow_none=True)
    if dimension is not None and dimension < 1:
        raise ValueError("target_dimension must be a positive integer when supplied")
    cached = _nonnegative_int(cached_documents, "cached_documents") or 0
    if corpus is not None and cached > corpus:
        raise ValueError("cached_documents cannot exceed corpus_size")
    # NumPy does not expose ``bfloat16`` on every supported version, while it
    # is a common model/storage spelling.  Keep its well-defined two-byte
    # width as an explicit alias and use NumPy for the remaining numeric
    # dtypes so invalid/object types are still rejected.
    dtype_text = str(dtype).strip().lower().replace("-", "")
    if dtype_text in {"bfloat16", "bf16"}:
        normalized_dtype = "bfloat16"
        itemsize = 2
        dtype_kind = "f"
    else:
        try:
            parsed_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported vector dtype: {dtype!r}") from exc
        normalized_dtype = parsed_dtype.name
        itemsize = int(parsed_dtype.itemsize)
        dtype_kind = parsed_dtype.kind
    if itemsize < 1 or dtype_kind not in {"f", "i", "u"}:
        raise ValueError("vector dtype must be a numeric dtype")
    throughput = _nonnegative_float(docs_per_second, "docs_per_second")
    if throughput is not None and throughput <= 0:
        raise ValueError("docs_per_second must be greater than zero when supplied")
    price = _nonnegative_float(gpu_hourly_cost, "gpu_hourly_cost")

    result: dict[str, Any] = {
        "corpus_documents": corpus,
        "cached_documents": cached,
        "remaining_documents": None if corpus is None else max(0, corpus - cached),
        "target_dimension": dimension,
        "dtype": normalized_dtype,
        "bytes_per_element": itemsize,
        "raw_vector_storage": quantity(None, "bytes", "unknown", assumptions=["dimension or corpus size is unknown"]),
        "full_backfill": {
            "wall_time": quantity(None, "seconds", "unknown", assumptions=["target encoding throughput was not supplied"]),
            "gpu_hours": quantity(None, "hours", "unknown", assumptions=["target encoding throughput was not supplied"]),
            "cost": quantity(None, "currency", "unknown", assumptions=["throughput or GPU hourly cost is missing"]),
        },
        "progressive_scenarios": [],
        "assumptions": [
            "Storage is raw target-vector bytes only; ANN index, metadata, replicas, and backups are excluded.",
            "Time and cost are linear projections, not guarantees.",
        ],
    }
    if corpus is not None and dimension is not None:
        raw_bytes = corpus * dimension * itemsize
        result["raw_vector_storage"] = quantity(raw_bytes, "bytes", "modeled", assumptions=["documents × dimension × dtype bytes"])
    if corpus is not None and throughput is not None and throughput > 0:
        seconds = corpus / throughput
        hours = seconds / 3600.0
        result["full_backfill"]["wall_time"] = quantity(seconds, "seconds", "modeled", assumptions=["user-supplied or measured docs/sec"])
        result["full_backfill"]["gpu_hours"] = quantity(hours, "hours", "modeled", assumptions=["one worker/GPU and linear throughput"])
        if price is not None:
            result["full_backfill"]["cost"] = quantity(hours * price, "currency", "modeled", assumptions=["user-supplied hourly price"])
    for raw_fraction in scenarios:
        fraction = _nonnegative_float(raw_fraction, "scenario fraction", allow_none=False)
        if fraction is None or fraction <= 0 or fraction > 1:
            raise ValueError("scenario fractions must be in (0, 1]")
        docs = None if corpus is None else int(math.ceil(corpus * fraction))
        storage = None if docs is None or dimension is None else docs * dimension * itemsize
        scenario: dict[str, Any] = {
            "fraction": fraction,
            "documents": docs,
            "storage": quantity(storage, "bytes", "modeled" if storage is not None else "unknown",
                                  assumptions=["raw vectors only"] if storage is not None else ["corpus size or dimension is unknown"]),
            "wall_time": quantity(None, "seconds", "unknown", assumptions=["throughput was not supplied"]),
            "cost": quantity(None, "currency", "unknown", assumptions=["throughput or price is missing"]),
        }
        if docs is not None and throughput is not None and throughput > 0:
            scenario["wall_time"] = quantity(docs / throughput, "seconds", "modeled", assumptions=["linear encoding throughput"])
            if price is not None:
                scenario["cost"] = quantity(docs / throughput / 3600.0 * price, "currency", "modeled", assumptions=["linear encoding throughput", "user-supplied hourly price"])
        result["progressive_scenarios"].append(scenario)
    result["inputs"] = {
        "target_docs_per_second": quantity(throughput, "documents/second", "user_supplied" if throughput is not None else "unknown"),
        "gpu_hourly_cost": quantity(price, "currency/hour", "user_supplied" if price is not None else "unknown"),
    }
    return result


__all__ = ["estimate_economics", "quantity"]
