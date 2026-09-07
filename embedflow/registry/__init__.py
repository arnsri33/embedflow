"""Versioned, provenance-aware migration evidence shipped with EmbedFlow.

The registry contains only retained research results.  It is intentionally
separate from the runtime cache and never contains vectors or indexes.
"""

from .loader import (
    dataset_fingerprint,
    find_evidence,
    load_benchmark_profiles,
    load_evidence,
    load_manifest,
    load_summaries,
    verify_registry,
)
from .matcher import RegistryMatch, match_config, match_evidence
from .schema import (
    MATCH_EXACT,
    MATCH_NONE,
    MATCH_PRIOR,
    MATCH_RELATED,
    BenchmarkProfile,
    EvidenceRecord,
    RegistryError,
    contract_fingerprint,
)

__all__ = [
    "BenchmarkProfile",
    "EvidenceRecord",
    "MATCH_EXACT",
    "MATCH_NONE",
    "MATCH_PRIOR",
    "MATCH_RELATED",
    "RegistryError",
    "contract_fingerprint",
    "RegistryMatch",
    "dataset_fingerprint",
    "find_evidence",
    "load_benchmark_profiles",
    "load_evidence",
    "load_manifest",
    "load_summaries",
    "match_config",
    "match_evidence",
    "verify_registry",
]
