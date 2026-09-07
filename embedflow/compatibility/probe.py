from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def run_finite_pool_probe(
    source_model: Any,
    target_model: Any,
    source_index: Any,
    documents: Any,
    queries: Iterable[tuple[str, str]],
    *,
    kmax: int = 500,
    seed: int = 42,
    limit: int | None = None,
    target_document_vectors: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the existing leakage-safe T2-v1 probe implementation."""
    from ..migration.compatibility import run_probe

    return run_probe(
        source_model,
        target_model,
        source_index,
        documents,
        queries,
        kmax=kmax,
        seed=seed,
        limit=limit,
        target_document_vectors=target_document_vectors,
    )
