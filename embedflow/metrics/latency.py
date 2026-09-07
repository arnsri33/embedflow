from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

STAGES = ("source_query_encode_ms", "source_ann_ms", "target_query_encode_ms", "cache_lookup_ms",
          "synchronous_target_encode_ms", "target_score_ms", "topk_ms", "total_ms")


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype="float64")
    if arr.size == 0: return {"count": 0}
    if not np.isfinite(arr).all() or (arr < 0).any(): raise ValueError("latency values must be finite and non-negative")
    return {"count": int(arr.size), "mean_ms": float(np.mean(arr)), "p50_ms": float(np.quantile(arr, .50)),
            "p90_ms": float(np.quantile(arr, .90)), "p95_ms": float(np.quantile(arr, .95)),
            "p99_ms": float(np.quantile(arr, .99)), "min_ms": float(np.min(arr)), "max_ms": float(np.max(arr)),
            "std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            # A zero-duration synthetic fixture has no meaningful finite QPS;
            # return 0 rather than leaking infinity into JSON reports.
            "qps": float(1000.0 / np.mean(arr)) if float(np.mean(arr)) > 0 else 0.0}


def aggregate_records(records: list[dict[str, Any]], group_keys: tuple[str, ...] = ("mode", "K", "nprobe")) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    malformed = 0
    for row in records:
        if not isinstance(row, dict):
            malformed += 1
            continue
        try:
            values = [float(row.get(stage, 0.0)) for stage in STAGES]
            if not np.isfinite(values).all() or (np.asarray(values) < 0).any():
                raise ValueError
        except (TypeError, ValueError):
            # Telemetry is append-only and may contain a partially written or
            # hand-edited line. Skip that row while making the omission visible.
            malformed += 1
            continue
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    for key, rows in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        result = dict(zip(group_keys, key))
        for stage in STAGES: result.update({f"{stage}_{metric}": value for metric, value in summarize(float(r.get(stage, 0.0)) for r in rows).items()})
        if malformed:
            result["malformed_rows_skipped"] = malformed
        out.append(result)
    return out
