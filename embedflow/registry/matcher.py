from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .schema import (
    MATCH_EXACT,
    MATCH_NONE,
    MATCH_PRIOR,
    MATCH_RELATED,
    EvidenceRecord,
    contract_fingerprint,
)

_ALIASES = {
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3_0_6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen/qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3_4b": "Qwen/Qwen3-Embedding-4B",
    "qwen/qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3_8b": "Qwen/Qwen3-Embedding-8B",
    "qwen/qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}


def canonical_model_id(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("canonical_model_id", value.get("model", value.get("model_id", "")))
    else:
        value = getattr(value, "model", getattr(value, "model_id", value))
    text = str(value or "").strip()
    return _ALIASES.get(text.lower(), text)


def _contract(value: Any) -> tuple[str, str | None]:
    if isinstance(value, Mapping):
        ident = canonical_model_id(value)
        fp = value.get("contract_fingerprint") or value.get("fingerprint")
        if fp is None and any(key in value for key in ("revision", "dimension", "pooling", "max_length", "normalization")):
            candidate = dict(value)
            candidate["canonical_model_id"] = ident
            fp = contract_fingerprint(candidate)
        return ident, str(fp) if fp else None
    ident = canonical_model_id(value)
    if hasattr(value, "contract"):
        raw = dict(value.contract())
        raw["canonical_model_id"] = ident
        return ident, contract_fingerprint(raw)
    if hasattr(value, "fingerprint"):
        return ident, str(value.fingerprint)
    # A bare model ID can identify a prior transition for display, but cannot
    # establish exact contract equivalence (revision/prompt/etc. are unknown).
    return ident, None


def _family(value: str) -> str:
    text = value.lower().replace("embedding", "").replace("sentence-transformers/", "")
    if "minilm" in text:
        return "minilm"
    # Keep the family intentionally broad: a Qwen3 revision/size that is not
    # in the core rows can still receive RELATED evidence, but never an
    # automatic decision.  Contract-level matching above remains strict.
    if re.search(r"qwen3", text):
        return "qwen3"
    return text


@dataclass(frozen=True)
class RegistryMatch:
    """Evidence lookup result; only EXACT matches can be reused as results."""

    level: str
    records: tuple[EvidenceRecord, ...]
    exact_source_contract: bool
    exact_target_contract: bool
    exact_corpus: bool
    reason: str
    recommended_k: tuple[int, ...] = ()

    @property
    def prior_datasets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(row.dataset.get("name")) for row in self.records))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "exact_source_contract": self.exact_source_contract,
            "exact_target_contract": self.exact_target_contract,
            "exact_corpus": self.exact_corpus,
            "reason": self.reason,
            "prior_datasets": list(self.prior_datasets),
            "recommended_k": list(self.recommended_k),
            "evidence_ids": [row.evidence_id for row in self.records],
        }


def match_evidence(
    *,
    source_model: Any,
    target_model: Any,
    corpus_fingerprint: str | None = None,
    corpus_name: str | None = None,
    corpus_size: int | None = None,
    records: Iterable[EvidenceRecord],
) -> RegistryMatch:
    source_id, source_fp = _contract(source_model)
    target_id, target_fp = _contract(target_model)
    # ``records`` is intentionally typed as an iterable so callers can stream
    # custom registries.  Materialise it once: transition and family matching
    # both need to inspect the same rows, and consuming a generator twice
    # would otherwise produce a false ``NO REGISTRY MATCH`` result.
    rows = tuple(records)
    source_family, target_family = _family(source_id), _family(target_id)
    transition = tuple(
        row for row in rows
        if canonical_model_id(row.source) == source_id and canonical_model_id(row.target) == target_id
    )
    if not transition:
        family_rows = tuple(
            row for row in rows
            if _family(canonical_model_id(row.source)) == source_family
            and _family(canonical_model_id(row.target)) == target_family
        )
        return RegistryMatch(
            MATCH_RELATED if family_rows else MATCH_NONE,
            family_rows,
            False,
            False,
            False,
            "No identical transition was found; related family evidence is not a compatibility decision." if family_rows else "No registry record matches this transition.",
            _priority_k(family_rows),
        )
    exact_source = source_fp is not None and all(str(row.source.get("contract_fingerprint")) == source_fp for row in transition)
    exact_target = target_fp is not None and all(str(row.target.get("contract_fingerprint")) == target_fp for row in transition)
    corpus_matches = []
    for row in transition:
        dataset = row.dataset
        fingerprint_match = corpus_fingerprint is not None and dataset.get("fingerprint") is not None and str(dataset.get("fingerprint")) == str(corpus_fingerprint)
        canonical_match = (
            corpus_name is not None and str(dataset.get("canonical_dataset_id", dataset.get("name"))) == str(corpus_name)
            and corpus_size is not None and dataset.get("corpus_size") is not None and int(dataset["corpus_size"]) == int(corpus_size)
            and bool(dataset.get("canonical_construction", False))
        )
        corpus_matches.append(bool(fingerprint_match or canonical_match))
    exact_corpus = any(corpus_matches)
    if exact_source and exact_target and exact_corpus:
        selected = tuple(row for row, matched in zip(transition, corpus_matches) if matched)
        return RegistryMatch(MATCH_EXACT, selected, True, True, True, "Source and target contracts and corpus construction match a canonical row.", _priority_k(selected))
    if exact_source and exact_target:
        return RegistryMatch(MATCH_PRIOR, transition, True, True, False, "Contracts match, but the supplied corpus is new or its fingerprint is unavailable.", _priority_k(transition))
    return RegistryMatch(MATCH_RELATED, transition, exact_source, exact_target, False, "A model transition matches by canonical name, but its full contract fingerprint differs.", _priority_k(transition))


def _priority_k(records: Iterable[EvidenceRecord]) -> tuple[int, ...]:
    depths: set[int] = set()
    for row in records:
        observed = row.raw.get("observed_migration_depth")
        if observed is not None:
            depths.add(int(observed))
        depths.update(int(k) for k in row.candidate_gap)
    return tuple(sorted(depths))


def match_config(config: Any, *, corpus_fingerprint: str | None = None, corpus_name: str | None = None, corpus_size: int | None = None, records: Iterable[EvidenceRecord] | None = None) -> RegistryMatch:
    """Match an ``EmbedFlowConfig`` without making network/model calls."""
    from ..config import hydrate_research_contract
    from .loader import load_evidence

    if corpus_size is None:
        docs = getattr(config, "documents", None)
        if docs is not None:
            corpus_size = None
    source = config.source if hasattr(config, "source") else config.get("source")
    target = config.target if hasattr(config, "target") else config.get("target")
    # ``load_config`` already hydrates the frozen contracts.  Hydrating here as
    # well makes the Python API safe for callers who construct a config object
    # directly, while explicit non-default contract fields remain untouched.
    if hasattr(source, "model"):
        source = hydrate_research_contract(source)
    if hasattr(target, "model"):
        target = hydrate_research_contract(target)
    return match_evidence(
        source_model=source,
        target_model=target,
        corpus_fingerprint=corpus_fingerprint,
        corpus_name=corpus_name,
        corpus_size=corpus_size,
        records=list(records) if records is not None else load_evidence(),
    )
