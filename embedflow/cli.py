from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .compatibility.evaluate import evaluate_models, evaluate_with_native_rankings, load_qrels, load_queries
from .compatibility.report import report_markdown, write_report
from .config import (
    CacheConfig,
    DocumentsConfig,
    EmbedFlowConfig,
    IndexConfig,
    MigrationConfig,
    ModelConfig,
    ProbeConfig,
    TelemetryConfig,
    hydrate_research_contract,
    load_config,
    save_config,
)
from .migration.compatibility import run_probe, save_probe
from .migration.state import DocumentStore
from .models import HashEmbeddingModel, load_embedding_model
from .registry import (
    MATCH_EXACT,
    load_benchmark_profiles,
    load_evidence,
    match_config,
    match_evidence,
    verify_registry,
)
from .runtime import build_faiss_from_documents, load_documents, open_engine


def _json(value: Any) -> None: print(json.dumps(value, indent=2, ensure_ascii=False, default=float))


def _normalize_device(value: str | None) -> str | None:
    """Accept the common human spelling ``gpu`` while PyTorch uses ``cuda``."""
    if value is None:
        return None
    value = str(value).strip()
    return "cuda" if value.lower() == "gpu" else value


def _load_queries(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for i, line in enumerate(handle, 1):
            if not line.strip(): continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid query JSON at {path}:{i}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"query row {i} in {path} must be a JSON object")
            qid = str(row.get("id", row.get("query_id", i))); text = row.get("text", row.get("query"))
            if qid in seen:
                raise ValueError(f"duplicate query ID {qid!r} in {path}")
            if not isinstance(text, str) or not text.strip(): raise ValueError(f"query {qid} has no text")
            seen.add(qid); rows.append((qid, text))
    if not rows:
        raise ValueError(f"query file is empty: {path}")
    return rows


def _load_rankings(path: str | Path) -> dict[str, list[str]]:
    """Load JSONL native rankings: {id/query_id, ranked_ids/ranking}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, list[str]] = {}
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid ranking JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"ranking row {number} in {path} must be a JSON object")
            query_id = str(row.get("id", row.get("query_id", number)))
            if query_id in rows:
                raise ValueError(f"duplicate ranking query ID {query_id!r} in {path}")
            ranking = row.get("ranked_ids", row.get("ranking", row.get("results")))
            if not isinstance(ranking, list):
                raise ValueError(f"ranking {query_id!r} must contain a list")
            rows[query_id] = [str(item.get("id")) if isinstance(item, dict) else str(item) for item in ranking]
    if not rows:
        raise ValueError(f"ranking file is empty: {path}")
    return rows


def _set_model_local_paths(cfg: EmbedFlowConfig, model_root: str | Path | None) -> EmbedFlowConfig:
    """Resolve known staged checkpoints without requiring a research checkout."""
    registry = {
        "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
        "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b",
        "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
        "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
    }
    if model_root:
        root = Path(model_root).expanduser().resolve()
        for model in (cfg.source, cfg.target):
            staged = root / registry.get(model.model, model.model)
            if staged.exists():
                model.local_path = str(staged)
    return cfg


def _direct_analysis_config(args: argparse.Namespace) -> tuple[Path, EmbedFlowConfig]:
    """Create a reusable config for the direct ``analyze`` UX."""
    if not all((args.documents, args.index, args.source_model, args.target_model, args.probe_queries)):
        raise ValueError("direct analyze requires --documents, --index, --source-model, --target-model, and --probe-queries")
    output_dir = Path(args.output_dir or ".").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "embedflow.analysis.yaml"
    source = hydrate_research_contract(ModelConfig(str(args.source_model)), project_root=Path(__file__).resolve().parents[1])
    target = hydrate_research_contract(ModelConfig(str(args.target_model)), project_root=Path(__file__).resolve().parents[1])
    index_value = str(args.index)
    qdrant_url = index_value if args.backend == "qdrant" and "://" in index_value else None
    index_path = index_value if qdrant_url else str(Path(index_value).expanduser().resolve())
    cfg = EmbedFlowConfig(
        source=source,
        target=target,
        index=IndexConfig(backend=args.backend, path=index_path, url=qdrant_url, collection=args.collection,
                          vector_name=args.vector_name, api_key_env=args.api_key_env, metric=args.metric,
                          ids=str(Path(args.index_ids).expanduser().resolve()) if args.index_ids else None),
        documents=DocumentsConfig(path=str(Path(args.documents).expanduser().resolve())),
        migration=MigrationConfig(candidate_depth="auto", kmax_probe=int(args.kmax or 500), probe_queries=int(args.limit or 100)),
        cache=CacheConfig(path=str(output_dir / "embedflow_cache")),
        probe=ProbeConfig(queries=str(Path(args.probe_queries).expanduser().resolve()), kmax=int(args.kmax or 500),
                          seed=int(args.seed if args.seed is not None else 42), limit=args.limit),
        telemetry=TelemetryConfig(latency_log=str(output_dir / "logs" / "latency.jsonl")),
        state_path=str(output_dir / "embedflow_state.json"),
        dashboard_title=f"EmbedFlow — {source.model} → {target.model}",
    )
    _set_model_local_paths(cfg, args.model_root)
    save_config(cfg, config_path)
    return config_path, cfg


def _analysis_summary(result: dict[str, Any], *, config: EmbedFlowConfig, output: Path) -> None:
    print("EmbedFlow Migration Analysis")
    print("-" * 50)
    print(f"Source model:\n  {config.source.model}")
    print(f"Target model:\n  {config.target.model}")
    print(f"Corpus:\n  {DocumentStore(config.documents.path, config.documents.id_field, config.documents.text_field).size():,} documents")
    print(f"Diagnostic:\n  {str(result.get('diagnostic', 'UNKNOWN')).upper()}")
    print(f"Recommended initial candidate depth:\n  K = {result.get('recommended_k', 'n/a')}")
    diagnostic = str(result.get("diagnostic", "UNKNOWN")).upper()
    tail = {"SAFE": "stable", "EXPAND": "still changing", "UNSAFE_OR_UNCERTAIN": "uncertain"}.get(diagnostic, "not established")
    print(f"Finite-tail behavior:\n  {tail}")
    print("ANN health:\n  UNKNOWN (unless an exact/reference audit was supplied)")
    print("Suggested strategy:\n  PROGRESSIVE" if diagnostic == "SAFE" else "Suggested strategy:\n  PROGRESSIVE_WITH_REVIEW")
    print("Warning:\n  SAFE is an empirical diagnostic, not a guarantee.")
    print(f"\nSaved probe: {output}")


def _print_registry_match(match: Any, *, heading: str = "Known evidence", reused: bool = False) -> None:
    """Render registry evidence without turning prior evidence into a verdict."""
    if not match.records:
        print(f"{heading}: none")
        return
    print("\n" + "-" * 50)
    print(heading)
    print("-" * 50)
    print(f"Registry evidence match: {match.level}")
    print(f"Exact source contract: {'yes' if match.exact_source_contract else 'no'}")
    print(f"Exact target contract: {'yes' if match.exact_target_contract else 'no'}")
    print(f"Exact corpus: {'yes' if match.exact_corpus else 'no'}")
    print("Previously studied:")
    for dataset in match.prior_datasets:
        print(f"  {dataset}")
    print("Priority probe depths: " + ", ".join(str(k) for k in match.recommended_k) if match.recommended_k else "Priority probe depths: unavailable")
    if match.level == MATCH_EXACT:
        if reused:
            print("REUSING canonical registry values for this exact match; the current probe still runs and is recorded separately.")
        else:
            print("Existing canonical benchmark available. Values may be reused only with --use-registry.")
    else:
        print("These results are prior evidence, not a compatibility decision for the current corpus.")
        print("Recommended next action: run a finite-tail probe on this corpus.")
    for row in match.records:
        data = row.raw
        dataset = data["dataset"]
        size = dataset.get("corpus_size")
        size_text = f"{int(size):,}" if size is not None else "size unavailable"
        print(f"\n  {dataset['name']} ({size_text} documents):")
        if data.get("candidate_gap"):
            display = ", ".join(f"G({k})={float(v):.5f}" for k, v in data["candidate_gap"].items())
            print(f"    {display}")
        print(f"    observed K*: {data.get('observed_migration_depth') if data.get('observed_migration_depth') is not None else 'unavailable'}")
        print(f"    CI-certified K*: {data.get('ci_certified_migration_depth') if data.get('ci_certified_migration_depth') is not None else 'unavailable'}")


def _registry_values(match: Any) -> list[dict[str, Any]]:
    """Serialize canonical values that ``--use-registry`` actually reuses."""
    values: list[dict[str, Any]] = []
    for row in match.records:
        data = row.to_dict()
        values.append({
            "evidence_id": row.evidence_id,
            "dataset": data.get("dataset", {}).get("name"),
            "candidate_gap": data.get("candidate_gap", {}),
            "containment": data.get("containment", {}),
            "source_quality": data.get("source_quality"),
            "native_target_quality": data.get("native_target_quality"),
            "restricted_target_quality": data.get("restricted_target_quality", {}),
            "observed_migration_depth": data.get("observed_migration_depth"),
            "ci_certified_migration_depth": data.get("ci_certified_migration_depth"),
            "epsilon": data.get("epsilon"),
            "provenance": data.get("provenance", {}),
        })
    return values


def cmd_registry_list(args: argparse.Namespace) -> int:
    records = load_evidence()
    if args.json:
        _json([row.to_dict() for row in records])
        return 0
    print("SOURCE                              TARGET                         DATASET                                  SIZE")
    print("-" * 116)
    for row in records:
        dataset = row.dataset
        size = dataset.get("corpus_size")
        size_text = f"{int(size):,}" if size is not None else "-"
        print(f"{row.source.get('canonical_model_id','')[:34]:34}  {row.target.get('canonical_model_id','')[:29]:29}  {str(dataset.get('name',''))[:40]:40}  {size_text:>10}")
    print(f"\n{len(records)} core evidence rows; use `embedflow registry show` for curves and provenance.")
    return 0


def cmd_registry_show(args: argparse.Namespace) -> int:
    match = match_evidence(source_model=args.source, target_model=args.target, records=load_evidence())
    if not match.records:
        print("No registry evidence found for this source-target transition.")
        return 0
    _print_registry_match(match, heading=f"Evidence for {args.source} -> {args.target}")
    if args.json:
        _json([row.to_dict() for row in match.records])
    return 0


def cmd_registry_match(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    corpus_fp = args.corpus_fingerprint
    if not corpus_fp and args.corpus:
        from .registry.loader import dataset_fingerprint
        corpus_fp = dataset_fingerprint(args.corpus, id_field=cfg.documents.id_field, text_field=cfg.documents.text_field)
    docs_size = None
    if args.corpus:
        docs_size = DocumentStore(args.corpus, cfg.documents.id_field, cfg.documents.text_field).size()
    else:
        try:
            docs_size = DocumentStore(cfg.documents.path, cfg.documents.id_field, cfg.documents.text_field).size()
        except (FileNotFoundError, ValueError):
            pass
    match = match_config(cfg, corpus_fingerprint=corpus_fp, corpus_name=args.corpus_name, corpus_size=docs_size)
    if args.json:
        _json(match.to_dict())
    else:
        _print_registry_match(match, heading="Registry evidence match")
    return 0


def cmd_registry_verify(args: argparse.Namespace) -> int:
    result = verify_registry(provenance_root=getattr(args, "research_root", None))
    if args.json:
        _json(result)
    else:
        print("REGISTRY VERIFICATION: " + ("PASS" if result["ok"] else "FAIL"))
        print(f"Core evidence rows: {result['record_count']}")
        print(f"Benchmark profiles: {result['profile_count']}")
        print(f"Retained artifacts checked: {result.get('provenance_artifacts_checked', 0)}")
        print(f"Semantic values checked: {result.get('semantic_values_checked', 0)}")
        for message in result.get("warnings", []):
            print(f"Warning: {message}")
        for message in result.get("errors", []):
            print(f"Error: {message}")
    return 0 if result["ok"] else 2


def cmd_benchmark_profiles_list(args: argparse.Namespace) -> int:
    profiles = load_benchmark_profiles()
    if args.json:
        _json([profile.to_dict() for profile in profiles])
        return 0
    print("PROFILE                                             KIND                         STATUS")
    print("-" * 106)
    for profile in profiles:
        print(f"{profile.profile_id[:50]:50}  {str(profile.raw.get('kind',''))[:28]:28}  {profile.raw.get('status','')}")
    print("\nProfiles are measured workload-specific evidence, not universal latency or throughput claims.")
    return 0


def _save_default_config(path: Path, args: argparse.Namespace) -> None:
    source = hydrate_research_contract(ModelConfig(args.source_model or "embedflow/demo-source", dimension=args.dimension or 64), path.parent)
    target = hydrate_research_contract(ModelConfig(args.target_model or "embedflow/demo-target", dimension=args.dimension or 64), path.parent)
    cfg = EmbedFlowConfig(source=source, target=target,
                          index=IndexConfig(backend=args.backend, path=args.index or "./legacy.index"),
                          documents=DocumentsConfig(path=args.documents or "./documents.jsonl"),
                          cache=CacheConfig(path=args.cache or "./embedflow_cache"))
    save_config(cfg, path); print(f"wrote {path}")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if not path.exists():
        if not args.source_model and not args.target_model and not args.documents and not args.index:
            if sys.stdin.isatty():
                print("EmbedFlow initialization")
                args.backend = "qdrant" if input("Existing index backend [1] FAISS / [2] Qdrant (1): ").strip() == "2" else "faiss"
                args.source_model = input("Source model: ").strip() or "embedflow/demo-source"
                args.target_model = input("Target model: ").strip() or "embedflow/demo-target"
                args.documents = input("Corpus JSONL path (./documents.jsonl): ").strip() or "./documents.jsonl"
                args.index = input("Legacy index path (./legacy.index): ").strip() or "./legacy.index"
                args.cache = input("Target cache path (./embedflow_cache): ").strip() or "./embedflow_cache"
            _save_default_config(path, args); return 0
        _save_default_config(path, args)
    cfg = load_config(path)
    docs = load_documents(cfg)
    print(f"loaded {docs.size():,} documents from {cfg.documents.path}")
    if cfg.index.backend == "faiss" and not Path(cfg.index.path).exists():
        if not args.build_index: raise FileNotFoundError(f"legacy FAISS index missing: {cfg.index.path}; pass --build-index to create it")
        model = load_embedding_model(cfg.source, model_root=path.parent / "models", device=args.device, demo=args.demo)
        try: build_faiss_from_documents(cfg, model, docs); print(f"built legacy index at {cfg.index.path}")
        finally: model.close()
    # Ensure dimensions and index metadata are checked before the user serves.
    engine = open_engine(path, device=args.device, demo=args.demo, start_worker=False)
    try:
        if args.queries:
            print("Testing candidate compatibility…")
            result = run_probe(engine.source_model, engine.target_model, engine.source_index, docs,
                               _load_queries(args.queries), kmax=args.kmax or cfg.migration.kmax_probe,
                               seed=42, limit=cfg.migration.probe_queries)
            save_probe(result, Path(cfg.state_path).with_name("probe_result.json"))
            print(f"Result: {result['diagnostic']}; Recommended candidate depth: K={result['recommended_k']}")
        print("configuration validated; run `embedflow analyze` then `embedflow serve`")
        _json(engine.plan.to_dict())
    finally: engine.close()
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """One-command setup for an existing FAISS or Qdrant index."""
    from .migration.facade import migrate

    session = None
    try:
        session = migrate(
            index=args.index,
            old_model=args.old_model,
            new_model=args.new_model,
            documents=args.documents,
            backend=args.backend,
            index_url=args.index_url,
            collection=args.collection,
            vector_name=args.vector_name,
            api_key_env=args.api_key_env,
            metric=args.metric,
            model_root=args.model_root,
            device=_normalize_device(args.device),
            cache_path=args.cache,
            state_path=args.state,
            config_path=args.config,
            candidate_depth=args.candidate_depth,
            kmax_probe=args.kmax_probe,
            max_sync_misses=args.max_sync_misses,
            background_batch_size=args.background_batch_size,
            probe_queries=args.probe_queries,
            probe_limit=args.probe_limit,
            start_worker=not args.no_worker,
        )
        print("EmbedFlow migration ready")
        print(f"source: {session.config.source.model}")
        print(f"target: {session.config.target.model}")
        print(f"index: {session.config.index.backend} ({session.config.index.path})")
        print(f"candidate depth: K={session.plan.candidate_depth}")
        print(f"diagnostic: {session.plan.diagnostic}")
        if session.config_path:
            print(f"configuration: {session.config_path}")
        if args.no_serve:
            _json(session.status())
            return 0
        print(f"serving at http://{args.host}:{args.port} (dashboard: /)")
        session.serve(host=args.host, port=args.port, log_level=args.log_level)
        return 0
    finally:
        if session is not None:
            session.close()


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        cfg = load_config(config_path)
        query_path = args.probe_queries or getattr(args, "queries", None) or cfg.probe.queries
        if not query_path:
            raise ValueError("analyze requires --probe-queries/--queries or probe.queries in the config")
    else:
        config_path, cfg = _direct_analysis_config(args)
        query_path = args.probe_queries
    docs = load_documents(cfg)
    registry_match = match_config(
        cfg,
        corpus_fingerprint=getattr(args, "corpus_fingerprint", None),
        corpus_name=getattr(args, "corpus_name", None),
        corpus_size=docs.size(),
        records=load_evidence(),
    )
    use_registry = bool(getattr(args, "use_registry", False))
    _print_registry_match(registry_match, reused=use_registry)
    if use_registry and registry_match.level != MATCH_EXACT:
        raise ValueError("--use-registry requires an EXACT REGISTRY MATCH; prior/related evidence cannot be reused as a result")
    engine = open_engine(config_path, device=_normalize_device(args.device), demo=args.demo, start_worker=False)
    try:
        queries = _load_queries(query_path)
        result = run_probe(engine.source_model, engine.target_model, engine.source_index, docs, queries,
                           kmax=args.kmax or cfg.probe.kmax or cfg.migration.kmax_probe,
                           seed=args.seed if args.seed is not None else cfg.probe.seed,
                           limit=args.limit if args.limit is not None else cfg.probe.limit)
        output_dir = Path(args.output_dir or Path(cfg.state_path).parent).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out = Path(args.output or output_dir / "probe_result.json")
        save_probe(result, out)
        report = {
            "schema_version": "0.1",
            "source_model": cfg.source.model,
            "target_model": cfg.target.model,
            "corpus_documents": docs.size(),
            "diagnostic": result["diagnostic"],
            "recommended_initial_k": result.get("recommended_k"),
            "observed_k_epsilon": None,
            "epsilon": 0.01,
            "ann_status": "UNKNOWN",
            "recommendation": "Progressive migration is a reasonable candidate for further deployment validation." if str(result.get("diagnostic", "")).upper() == "SAFE" else "Further validation is required before relying on progressive migration.",
            "native_target_index_used": False,
            "finite_tail_behavior": {"SAFE": "stable", "EXPAND": "still changing", "UNSAFE_OR_UNCERTAIN": "uncertain"}.get(str(result.get("diagnostic", "UNKNOWN")).upper(), "not established"),
            "t2_warning": result.get("warning", "T2-v1 is an empirical finite-tail diagnostic, not a compatibility guarantee."),
            "probe": result,
            "candidate_gap_curve": [],
            "registry": registry_match.to_dict(),
            "registry_reused": bool(use_registry and registry_match.level == MATCH_EXACT),
            "registry_reuse": {
                "used": bool(use_registry and registry_match.level == MATCH_EXACT),
                "evidence_ids": [row.evidence_id for row in registry_match.records] if use_registry else [],
                "reused_fields": ["candidate_gap", "containment", "native_target_quality", "source_quality", "observed_migration_depth", "ci_certified_migration_depth"] if use_registry else [],
                "note": "Canonical values are prior measurements; the current-corpus probe remains the deployment analysis." if use_registry else "Canonical rows were displayed but not reused.",
            },
            "registry_reused_values": _registry_values(registry_match) if use_registry else [],
            "registry_evidence": [row.to_dict() for row in registry_match.records],
            "limitations": [
                "No native target index or qrels were used; this is Mode B deployment analysis.",
                "Recommended initial K is not observed K*.",
                "ANN fidelity is UNKNOWN until an exact/reference source comparison is supplied.",
            ],
        }
        write_report(output_dir / "migration_report.json", report)
        (output_dir / "report.md").write_text(report_markdown(report))
        _analysis_summary(result, config=cfg, output=out)
    finally: engine.close()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run Mode A evaluation with qrels and native target evidence."""
    cfg = load_config(args.config)
    query_path = args.queries or cfg.probe.queries
    if not query_path:
        raise ValueError("evaluate requires --queries or probe.queries in the config")
    if not args.qrels:
        raise ValueError("evaluate requires --qrels; Mode A must have relevance labels")
    docs = load_documents(cfg)
    engine = open_engine(args.config, device=_normalize_device(args.device), demo=args.demo, start_worker=False)
    evaluation_indexes: list[Any] = []
    try:
        queries = load_queries(query_path)
        qrels = load_qrels(args.qrels)
        native_rankings = _load_rankings(args.native_target_rankings) if args.native_target_rankings else None
        native_index = None
        reference_index = None
        if args.native_target_index:
            native_index = _load_evaluation_index(args.native_target_index, cfg.index.backend, cfg.index.metric, docs.documents,
                                                  engine.target_model.dimension, cfg.index.collection, cfg.index.url,
                                                  cfg.index.vector_name, cfg.index.api_key_env, cfg.index.nprobe)
            evaluation_indexes.append(native_index)
        if args.reference_index:
            reference_index = _load_evaluation_index(args.reference_index, cfg.index.backend, cfg.index.metric, docs.documents,
                                                     engine.source_model.dimension, cfg.index.collection, cfg.index.url,
                                                     cfg.index.vector_name, cfg.index.api_key_env, cfg.index.nprobe)
            evaluation_indexes.append(reference_index)
        k_values = _parse_k_values(args.k_values or ",".join(map(str, cfg.probe.k_values)))
        if native_rankings is not None:
            result = evaluate_with_native_rankings(
                source_model=engine.source_model, target_model=engine.target_model, source_index=engine.source_index,
                documents=docs, queries=queries, native_target_rankings=native_rankings, qrels=qrels,
                reference_source_index=reference_index, k_values=k_values, quality_k=args.quality_k,
                epsilon=args.epsilon, bootstrap_resamples=args.bootstrap, seed=args.seed,
                recommended_k=None,
            )
        else:
            result = evaluate_models(
                source_model=engine.source_model, target_model=engine.target_model, source_index=engine.source_index,
                documents=docs, queries=queries, qrels=qrels, native_target_index=native_index,
                reference_source_index=reference_index, k_values=k_values, quality_k=args.quality_k,
                epsilon=args.epsilon, bootstrap_resamples=args.bootstrap, seed=args.seed,
            )
        output_dir = Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
        write_report(output_dir / "results.json", result)
        (output_dir / "report.md").write_text(report_markdown(result))
        curve_rows = result.get("candidate_gap_curve", [])
        import csv
        curve_path = output_dir / "candidate_gap_curve.csv"
        with curve_path.open("w", newline="") as handle:
            fieldnames = list(curve_rows[0]) if curve_rows else ["k", "candidate_gap", "containment"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(curve_rows)
        containment_path = output_dir / "containment_curve.csv"
        containment_rows = [{key: row.get(key) for key in ("k", "containment", "queries")} for row in curve_rows]
        with containment_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["k", "containment", "queries"]); writer.writeheader(); writer.writerows(containment_rows)
        summary: dict[str, Any] = {key: value for key, value in result.items()
                                   if isinstance(value, (str, int, float, bool)) or value is None}
        for group in ("source_quality", "native_target_quality"):
            for key, value in (result.get(group) or {}).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    summary[f"{group}_{key}"] = value
        with (output_dir / "results.csv").open("w", newline="") as handle:
            fieldnames = list(summary) or ["status"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerow(summary)
        print(report_markdown(result))
        print(f"\nWrote evaluation outputs to {output_dir}")
    finally:
        for evaluation_index in evaluation_indexes:
            try:
                evaluation_index.close()
            except Exception:
                pass
        engine.close()
    return 0


def _parse_k_values(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("K values must be comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise ValueError("K values must contain positive integers")
    return values


def _load_evaluation_index(path: str, backend: str, metric: str, documents: dict[str, str], dimension: int,
                           collection: str, url: str | None, vector_name: str | None = None,
                           api_key_env: str | None = "QDRANT_API_KEY", nprobe: int | None = None):
    from .indexes import FaissIndex, NumpyIndex, QdrantIndex
    if backend == "qdrant":
        return QdrantIndex.connect(url or path, collection, dimension, documents=documents, metric=metric,
                                   vector_name=vector_name, api_key_env=api_key_env)
    try:
        return FaissIndex.load(path, metric=metric, documents=documents, nprobe=nprobe)
    except (RuntimeError, ValueError) as exc:
        try:
            return NumpyIndex.load(path, metric=metric, documents=documents)
        except Exception as fallback_exc:
            raise exc from fallback_exc


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check dependencies and a config without downloading or starting workers."""
    checks: list[dict[str, Any]] = []
    version = sys.version_info
    checks.append({"name": "python", "ok": version >= (3, 10), "detail": platform.python_version()})
    for module, label in (("numpy", "NumPy"), ("yaml", "PyYAML"), ("faiss", "FAISS"), ("torch", "PyTorch"), ("fastapi", "FastAPI"), ("qdrant_client", "qdrant-client")):
        available = importlib.util.find_spec(module) is not None
        checks.append({"name": label.lower().replace("-", "_"), "ok": available, "detail": "installed" if available else "not installed (optional where noted)"})
    checks.append({"name": "cuda", "ok": True, "detail": _cuda_detail()})
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            checks.append({"name": "config", "ok": True, "detail": str(Path(args.config).resolve())})
            checks.append({"name": "documents", "ok": Path(config.documents.path).exists(), "detail": config.documents.path})
            qdrant_endpoint = config.index.url or config.index.path
            index_exists = (Path(config.index.path).exists() if config.index.backend.lower() == "faiss"
                            else bool(config.index.url or "://" in str(qdrant_endpoint) or Path(config.index.path).exists()))
            checks.append({"name": "index", "ok": index_exists, "detail": config.index.path})
            normalization_ok = not (config.index.metric.lower() == "cosine" and
                                    (config.source.normalization.lower() != "l2" or config.target.normalization.lower() != "l2"))
            checks.append({"name": "normalization", "ok": normalization_ok,
                           "detail": f"metric={config.index.metric}; source={config.source.normalization}; target={config.target.normalization}"})
            if Path(config.documents.path).exists():
                docs = DocumentStore(config.documents.path, config.documents.id_field, config.documents.text_field)
                checks.append({"name": "document_rows", "ok": docs.size() > 0, "detail": f"{docs.size():,} rows"})
            if config.index.backend.lower() == "faiss" and Path(config.index.path).exists():
                try:
                    from .indexes import FaissIndex, NumpyIndex
                    try:
                        index = FaissIndex.load(config.index.path, ids_path=config.index.ids, metric=config.index.metric,
                                                documents=docs.documents if "docs" in locals() else None)
                    except (RuntimeError, ValueError) as exc:
                        try:
                            index = NumpyIndex.load(config.index.path, metric=config.index.metric,
                                                    documents=docs.documents if "docs" in locals() else None)
                        except Exception as fallback_exc:
                            raise exc from fallback_exc
                    dimension_ok = config.source.dimension is None or int(config.source.dimension) == int(index.dimension)
                    checks.append({"name": "index_dimension", "ok": dimension_ok,
                                   "detail": f"index={index.dimension}; configured={config.source.dimension or index.dimension}"})
                    stored = index.metadata().get("model_fingerprint")
                    expected_fingerprint = config.source.fingerprint
                    if config.source.model.startswith("embedflow/demo"):
                        expected_fingerprint = HashEmbeddingModel(config.source.model, config.source.dimension or index.dimension).fingerprint
                    fingerprint_ok = not stored or stored == expected_fingerprint
                    checks.append({"name": "index_fingerprint", "ok": fingerprint_ok,
                                   "detail": "matches source contract" if fingerprint_ok else "does not match source contract"})
                except Exception as exc:
                    checks.append({"name": "index_integrity", "ok": False, "detail": str(exc)})
            cache_path = Path(config.cache.path)
            if cache_path.exists():
                try:
                    from .cache import SQLiteVectorCache
                    cache = SQLiteVectorCache(cache_path, config.target.fingerprint, int(config.target.dimension or 0))
                    stats = cache.stats(); cache.close()
                    checks.append({"name": "cache_integrity", "ok": True,
                                   "detail": f"{stats['cached_target_vectors']:,} vectors; dimension={stats['dimension']}"})
                except Exception as exc:
                    checks.append({"name": "cache_integrity", "ok": False, "detail": str(exc)})
            else:
                checks.append({"name": "cache_integrity", "ok": True, "detail": "not initialized yet"})
            checks.append({"name": "source_fingerprint", "ok": True, "detail": config.source.fingerprint[:16]})
            checks.append({"name": "target_fingerprint", "ok": True, "detail": config.target.fingerprint[:16]})
        except Exception as exc:
            checks.append({"name": "config", "ok": False, "detail": str(exc)})
    optional_checks = {"faiss", "qdrant_client", "fastapi", "torch"}
    failed = [check for check in checks if not check["ok"] and check["name"] not in optional_checks]
    if args.json:
        _json({"checks": checks, "status": "FAIL" if failed else "PASS"})
    else:
        print("EmbedFlow doctor")
        print("-" * 50)
        for check in checks:
            mark = "PASS" if check["ok"] else "WARN" if check["name"] in optional_checks else "FAIL"
            print(f"{mark:5} {check['name']}: {check['detail']}")
        print(f"\nOverall: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def _cuda_detail() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"available ({torch.cuda.get_device_name(0)})"
        return "not available"
    except Exception as exc:
        return f"unavailable ({exc})"


def cmd_serve(args: argparse.Namespace) -> int:
    device = _normalize_device(args.device)
    engine = open_engine(args.config, device=device, demo=args.demo, start_worker=True)
    from .serving.api import create_app
    app = create_app(engine)
    try:
        import uvicorn
        print(f"EmbedFlow serving at http://{args.host}:{args.port} (dashboard: /)")
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally: engine.close()
    return 0


def _with_engine(args: argparse.Namespace): return open_engine(args.config, device=_normalize_device(args.device), demo=args.demo, start_worker=True)


def cmd_status(args: argparse.Namespace) -> int:
    engine = _with_engine(args)
    try: _json(engine.status())
    finally: engine.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = _with_engine(args)
    try: _json(engine.search(args.query, args.top_k))
    finally: engine.close()
    return 0


def cmd_prewarm(args: argparse.Namespace) -> int:
    if args.documents is not None and (isinstance(args.documents, bool) or int(args.documents) != args.documents or int(args.documents) < 1):
        raise ValueError("--documents must be a positive integer")
    if not np.isfinite(float(args.fraction)) or not 0.0 < float(args.fraction) <= 1.0:
        raise ValueError("--fraction must be finite and in (0, 1]")
    engine = _with_engine(args)
    try:
        if args.ids:
            ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        elif args.strategy == "explicit":
            raise ValueError("--strategy explicit requires --ids")
        elif args.strategy == "popular":
            counts: dict[str, int] = {}
            for row in engine.records:
                for document_id in row.get("candidate_ids", []): counts[str(document_id)] = counts.get(str(document_id), 0) + 1
            ids = [x for x, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
            if args.documents: ids = ids[:int(args.documents)]
            elif not ids: ids = list(engine.documents.documents)
        elif args.documents:
            ids = list(engine.documents.documents)[:int(args.documents)]
        else:
            rng = random.Random(args.seed); ids = list(engine.documents.documents); rng.shuffle(ids); ids = ids[:max(1, int(len(ids) * args.fraction))]
        _json(engine.prewarm(ids, asynchronous=args.async_mode))
    finally: engine.close()
    return 0


def cmd_export_target(args: argparse.Namespace) -> int:
    """Optional explicit full target backfill and index export."""
    if isinstance(args.batch_size, bool) or int(args.batch_size) != args.batch_size or int(args.batch_size) < 1:
        raise ValueError("--batch-size must be a positive integer")
    engine = _with_engine(args)
    try:
        ids = list(engine.documents.documents)
        missing = [x for x in ids if x not in engine.cache.contains(ids)]
        print(f"materializing {len(missing):,} missing target vectors")
        if missing:
            docs = engine.documents.get(missing)
            for start in range(0, len(missing), int(args.batch_size)):
                chunk = missing[start:start + int(args.batch_size)]
                engine.cache.put(chunk, engine.target_model.encode_documents([docs[x] for x in chunk], batch_size=args.batch_size))
        from .indexes import FaissIndex, NumpyIndex, QdrantIndex
        cached = engine.cache.get(ids); vectors = np.asarray([cached[x] for x in ids], dtype="float32")
        if args.backend == "faiss":
            try: out = FaissIndex.build(vectors, ids, path=args.output_index, metric=engine.cfg.index.metric, documents=engine.documents.documents,
                                        metadata={"model_fingerprint": engine.target_model.fingerprint, "model_id": engine.target_model.model_id})
            except (RuntimeError, ValueError) as exc:
                if "FAISS" not in str(exc): raise
                out = NumpyIndex.build(vectors, ids, path=args.output_index, metric=engine.cfg.index.metric, documents=engine.documents.documents,
                                        metadata={"model_fingerprint": engine.target_model.fingerprint, "model_id": engine.target_model.model_id})
        else:
            out = QdrantIndex.build(args.output_index, args.collection, vectors, ids,
                                    documents=engine.documents.documents, metric=engine.cfg.index.metric,
                                    api_key_env=engine.cfg.index.api_key_env,
                                    vector_name=engine.cfg.index.vector_name)
        try:
            print(f"exported target index: {out.metadata()}")
        finally:
            out.close()
    finally: engine.close()
    return 0


def economics_for(corpus_size: int, docs_per_second: float | None, gpu_price: float | None,
                  cached: int = 0) -> dict[str, Any]:
    try:
        corpus_int = int(corpus_size)
        corpus_exact = float(corpus_size) == corpus_int
    except (TypeError, ValueError, OverflowError):
        corpus_int, corpus_exact = 0, False
    if isinstance(corpus_size, bool) or not corpus_exact or corpus_int < 0:
        raise ValueError("corpus_size must be a non-negative integer")
    try:
        cached_int = int(cached)
        cached_exact = float(cached) == cached_int
    except (TypeError, ValueError, OverflowError):
        cached_int, cached_exact = 0, False
    if isinstance(cached, bool) or not cached_exact or cached_int < 0:
        raise ValueError("cached documents must be a non-negative integer")
    if cached_int > corpus_int:
        raise ValueError("cached documents cannot exceed corpus size")
    price_value: float | None = None
    if gpu_price is not None:
        try:
            price_value = float(gpu_price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("gpu price must be finite and non-negative") from exc
        if not np.isfinite(price_value) or price_value < 0:
            raise ValueError("gpu price must be finite and non-negative")
    throughput: float | None = None
    if docs_per_second is not None:
        try:
            throughput = float(docs_per_second)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("docs_per_second must be finite and non-negative") from exc
        if not np.isfinite(throughput) or throughput < 0:
            raise ValueError("docs_per_second must be finite and non-negative")
    if throughput is None or throughput <= 0:
        return {"status": "NEEDS_THROUGHPUT", "corpus_documents": corpus_int, "cached_documents": cached_int,
                "remaining_documents": max(0, corpus_int - cached_int),
                "note": "Supply a measured or user-provided target_docs_per_second; projections are linear estimates."}
    remaining = max(0, corpus_int - cached_int); hours = remaining / throughput / 3600.0
    result = {"status": "ESTIMATE", "corpus_documents": corpus_int, "cached_documents": cached_int,
              "remaining_documents": remaining, "target_docs_per_second": throughput,
              "estimated_full_backfill_gpu_hours": corpus_int / throughput / 3600.0,
              "estimated_remaining_gpu_hours": hours, "deferred_fraction": remaining / max(1, int(corpus_size)),
              "note": "Linear projection from supplied throughput; not a hardware guarantee."}
    if price_value is not None:
        result["estimated_full_backfill_cost"] = result["estimated_full_backfill_gpu_hours"] * price_value
        result["estimated_remaining_cost"] = hours * price_value
    return result


def cmd_economics(args: argparse.Namespace) -> int:
    if args.config:
        cfg = load_config(args.config); docs = load_documents(cfg); cached = 0
        try:
            engine = open_engine(args.config, device="cpu", demo=args.demo, start_worker=False); cached = engine.cache.stats()["cached_target_vectors"]; engine.close()
        except Exception: pass
        corpus = docs.size()
        dps = args.target_docs_per_sec if args.target_docs_per_sec is not None else cfg.economics.target_docs_per_second
        price = args.gpu_price if args.gpu_price is not None else cfg.economics.gpu_price_per_hour
    else: corpus, dps, price, cached = args.corpus_size, args.target_docs_per_sec, args.gpu_price, args.cached_documents
    result = economics_for(corpus, dps, price, cached)
    if args.json:
        _json(result)
        return 0
    print("EmbedFlow Economics")
    print("(Projection based on supplied measured throughput.)")
    print("-" * 50)
    if result.get("status") != "ESTIMATE":
        print(result.get("note", "Supply target docs per second for an estimate."))
        return 0
    full_hours = float(result["estimated_full_backfill_gpu_hours"])
    remaining_hours = float(result["estimated_remaining_gpu_hours"])
    print("Full target backfill")
    print(f"  GPU-hours: {full_hours:,.2f}")
    print(f"  estimated cost: {_format_cost(result.get('estimated_full_backfill_cost'))}")
    print(f"  estimated one-GPU wall time: {full_hours:,.2f} hours")
    print("\nCurrent target cache")
    print(f"  documents materialized: {int(result['cached_documents']):,}")
    print(f"  fraction materialized: {1.0 - float(result['deferred_fraction']):.2%}")
    print(f"  work remaining: {int(result['remaining_documents']):,} documents ({remaining_hours:,.2f} GPU-hours)")
    print(f"  upfront work deferred: {float(result['deferred_fraction']):.2%}")
    if "estimated_remaining_cost" in result:
        print(f"  estimated remaining cost: {_format_cost(result['estimated_remaining_cost'])}")
    return 0


def _format_cost(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):,.2f}"


def cmd_audit_index(args: argparse.Namespace) -> int:
    cfg = load_config(args.config); engine = open_engine(args.config, device=_normalize_device(args.device), demo=args.demo, start_worker=False)
    reference = None
    try:
        meta = engine.source_index.metadata(); result = {"candidate_compatibility": engine.plan.diagnostic, "ann_status": "UNKNOWN",
                  "index": meta, "checks": {"dimension_match": True, "metric": cfg.index.metric, "corpus_documents": engine.documents.size()},
                  "note": "ANN recall is UNKNOWN until an exact/reference index or saved reference candidates are supplied."}
        if args.reference_index and args.queries:
            from .indexes import FaissIndex, NumpyIndex
            try: reference = FaissIndex.load(args.reference_index, metric=cfg.index.metric,
                                             documents=engine.documents.documents, nprobe=cfg.index.nprobe)
            except (RuntimeError, ValueError) as exc:
                try:
                    reference = NumpyIndex.load(args.reference_index, metric=cfg.index.metric, documents=engine.documents.documents)
                except Exception as fallback_exc:
                    raise exc from fallback_exc
            refs = _load_queries(args.queries); vals = []
            for _, text in refs[:max(1, int(args.limit or len(refs)))]:
                vector = engine.source_model.encode_query(text); got = {x.document_id for x in engine.source_index.search(vector, args.k)}; exact = {x.document_id for x in reference.search(vector, args.k)}; vals.append(len(got & exact) / max(1, len(exact)))
            recall = sum(vals) / max(1, len(vals)); result["ann_status"] = "PASS" if recall >= .95 else "WARNING"; result["ann_recall_at_k"] = recall; result["queries"] = len(vals); result["k"] = args.k; result["note"] = "Recall is overlap with the supplied reference index; it is not a T2-v1 compatibility measurement."
        _json(result)
    finally:
        if reference is not None:
            try:
                reference.close()
            except Exception:
                pass
        engine.close()
    return 0


def _demo_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    docs = path / "documents.jsonl"; queries = path / "probe_queries.jsonl"
    if not docs.exists():
        topics = [("aurora", "Auroras are caused when charged particles from the solar wind interact with gases in Earth's upper atmosphere."),
                  ("coffee", "Coffee beans are roasted seeds whose flavor depends on origin, roast temperature, and brewing method."),
                  ("batteries", "Lithium-ion batteries store energy through reversible movement of lithium ions between electrodes."),
                  ("volcano", "Volcanoes form when magma rises through weaknesses in Earth's crust and erupts at the surface."),
                  ("rainbow", "A rainbow appears when sunlight is refracted, reflected, and dispersed by water droplets."),
                  ("photosynthesis", "Plants use photosynthesis to convert light, water, and carbon dioxide into chemical energy."),
                  ("ocean", "Ocean currents transport heat around the planet and influence climate and marine ecosystems."),
                  ("sleep", "Sleep supports memory consolidation, immune function, and recovery from daily activity.")]
        rows = []
        for i in range(320):
            topic, text = topics[i % len(topics)]; rows.append({"id": f"doc-{i:04d}", "text": f"{text} This is reference note {i} about {topic}."})
        docs.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        qrows = [{"id": f"q-{i}", "text": f"what explains {topic}?"} for i, (topic, _) in enumerate(topics)]
        queries.write_text("\n".join(json.dumps(x) for x in qrows) + "\n")
    cfg_path = path / "embedflow.yaml"
    if not cfg_path.exists():
        cfg = EmbedFlowConfig(source=ModelConfig("embedflow/demo-source", dimension=64), target=ModelConfig("embedflow/demo-target", dimension=64),
                              index=IndexConfig(backend="faiss", path=str(path / "legacy.index")), documents=DocumentsConfig(path=str(docs)),
                              cache=CacheConfig(path=str(path / "cache")), state_path=str(path / "state.json"), migration=MigrationConfig(candidate_depth=20, kmax_probe=50, probe_queries=8, max_sync_misses=2, background_batch_size=16))
        save_config(cfg, cfg_path)
    return cfg_path


# These are the exact frozen contracts used by the NQ research run.  Keeping
# the IDs/revisions here means the real demo cannot silently switch to a
# different model revision or Qwen instruction template.
_REAL_DEMO_MODELS = {
    "minilm_l6": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "dimension": 384,
    },
    "qwen3_0_6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "dimension": 1024,
    },
}


def _model_snapshot(repo: str, revision: str, destination: Path, download: bool) -> Path:
    """Ensure a local, revision-pinned HF snapshot exists for the real demo."""
    destination = destination.resolve()
    weight_files = list(destination.rglob("*.safetensors")) + list(destination.rglob("*.bin")) + list(destination.rglob("*.pt"))
    ready = (destination / "config.json").exists() and bool(weight_files)
    if ready:
        print(f"using cached {repo} snapshot: {destination}")
        return destination
    if not download:
        raise FileNotFoundError(
            f"model snapshot is missing or incomplete at {destination}. "
            f"Re-run real-demo with --download, or place revision {revision} there."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("real-demo downloads require huggingface_hub; install it in the active environment") from exc
    destination.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo}@{revision} to {destination} …")
    try:
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination), max_workers=8)
    except Exception as exc:
        raise RuntimeError(
            f"could not download {repo}@{revision}. Check network/Hugging Face access, "
            f"then retry with --download: {exc}"
        ) from exc
    weight_files = list(destination.rglob("*.safetensors")) + list(destination.rglob("*.bin")) + list(destination.rglob("*.pt"))
    if not (destination / "config.json").exists() or not weight_files:
        raise RuntimeError(f"download completed but the snapshot at {destination} has no config/weights")
    return destination


def _real_demo_config(path: Path, model_root: Path, download: bool) -> tuple[Path, EmbedFlowConfig]:
    """Create a tiny-corpus config using the frozen MiniLM -> Qwen 0.6B pair."""
    path.mkdir(parents=True, exist_ok=True)
    # Reuse the same compact, inspectable corpus as the offline demo.  Only the
    # embedding models change; the query path and progressive cache are real.
    _demo_dir(path)
    source_info, target_info = _REAL_DEMO_MODELS["minilm_l6"], _REAL_DEMO_MODELS["qwen3_0_6b"]
    source_dir = _model_snapshot(source_info["model"], source_info["revision"], model_root / "minilm_l6", download)
    target_dir = _model_snapshot(target_info["model"], target_info["revision"], model_root / "qwen3_0_6b", download)
    source = hydrate_research_contract(ModelConfig(**source_info), project_root=Path(__file__).resolve().parents[1])
    target = hydrate_research_contract(ModelConfig(**target_info), project_root=Path(__file__).resolve().parents[1])
    source.local_path, target.local_path = str(source_dir), str(target_dir)
    cfg = EmbedFlowConfig(
        source=source,
        target=target,
        index=IndexConfig(backend="faiss", path=str(path / "legacy.index"), metric="cosine", nprobe=64),
        documents=DocumentsConfig(path=str(path / "documents.jsonl")),
        migration=MigrationConfig(candidate_depth=20, kmax_probe=50, probe_queries=8,
                                  max_sync_misses=2, background_batch_size=16),
        cache=CacheConfig(path=str(path / "cache")),
        state_path=str(path / "state.json"),
        dashboard_title="EmbedFlow — MiniLM → Qwen3-0.6B",
    )
    cfg_path = path / "embedflow.yaml"
    save_config(cfg, cfg_path)
    with (path / "documents.jsonl").open() as handle:
        doc_count = sum(1 for line in handle if line.strip())
    (path / "corpus_provenance.json").write_text(json.dumps({
        "kind": "synthetic_topics",
        "documents": doc_count,
        "note": "Small synthetic topic corpus for a deterministic, offline functional demo.",
    }, indent=2))
    return cfg_path, cfg


def cmd_real_demo(args: argparse.Namespace) -> int:
    """Build/serve the small real-model MiniLM -> Qwen 0.6B demonstration."""
    path = Path(args.path).resolve()
    device = _normalize_device(args.device)
    model_root = Path(args.model_root or (Path(__file__).resolve().parents[1] / "models")).resolve()
    if args.corpus != "topics":
        raise ValueError("the public real demo currently supports only the bundled topics corpus")
    cfg_path, cfg = _real_demo_config(path, model_root, args.download)
    docs = load_documents(cfg)
    index_path = Path(cfg.index.path)
    if not index_path.exists():
        print(f"building the MiniLM legacy index for {docs.size():,} demo documents …")
        source_model = load_embedding_model(cfg.source, model_root=model_root, device=device, demo=False)
        try:
            build_faiss_from_documents(cfg, source_model, docs)
        finally:
            source_model.close()
        print(f"built legacy index: {index_path}")
    else:
        print(f"reusing legacy index: {index_path}")
    engine = open_engine(cfg_path, device=device, demo=False, start_worker=False)
    try:
        print("running the frozen finite-pool compatibility probe …")
        probe_queries = _load_queries(path / "probe_queries.jsonl")
        result = run_probe(engine.source_model, engine.target_model, engine.source_index, docs,
                           probe_queries, kmax=cfg.migration.kmax_probe,
                           seed=42, limit=cfg.migration.probe_queries)
        save_probe(result, path / "probe_result.json")
        print(f"T2-v1 diagnostic: {result['diagnostic']}; recommended K={result['recommended_k']}")
    finally:
        engine.close()
    print(f"Real demo ready at {cfg_path}")
    if args.no_serve:
        return 0
    return cmd_serve(argparse.Namespace(config=str(cfg_path), device=device, demo=False,
                                        host=args.host, port=args.port, log_level=args.log_level))


def cmd_demo(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve(); cfg_path = _demo_dir(path); cfg = load_config(cfg_path); docs = load_documents(cfg)
    model = HashEmbeddingModel("embedflow/demo-source", 64)
    if args.backend == "faiss":
        build_faiss_from_documents(cfg, model, docs)
    else:
        from .indexes import QdrantIndex
        vectors = model.encode_documents(list(docs.documents.values()))
        qpath = path / "qdrant"
        qdrant_index = QdrantIndex.build(str(qpath), "embedflow-demo", vectors, list(docs.documents), documents=docs.documents)
        # The local Qdrant client holds an exclusive file lock. Close the
        # builder before opening a second client through ``open_engine`` in
        # this same process; remote clients are safe and the hook is
        # idempotent there as well.
        qdrant_index.close()
        cfg.index.backend = "qdrant"; cfg.index.path = str(qpath); cfg.index.collection = "embedflow-demo"; save_config(cfg, cfg_path)
    model.close()
    engine = open_engine(cfg_path, device="cpu", demo=True, start_worker=False)
    try:
        result = run_probe(engine.source_model, engine.target_model, engine.source_index, docs, _load_queries(path / "probe_queries.jsonl"), kmax=50, seed=42)
        save_probe(result, path / "probe_result.json"); print(f"T2-v1 diagnostic: {result['diagnostic']}; recommended K={result['recommended_k']}")
    finally: engine.close()
    print(f"Demo ready at {cfg_path}")
    if args.no_serve: return 0
    return cmd_serve(argparse.Namespace(config=str(cfg_path), device="cpu", demo=True, host=args.host, port=args.port, log_level="warning"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="embedflow", description="Progressive embedding-model migration")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create or validate an EmbedFlow YAML configuration"); init.add_argument("--config", default="embedflow.yaml"); init.add_argument("--source-model"); init.add_argument("--target-model"); init.add_argument("--documents"); init.add_argument("--index"); init.add_argument("--cache"); init.add_argument("--queries", help="optional JSONL probe queries to run during initialization"); init.add_argument("--kmax", type=int); init.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss"); init.add_argument("--dimension", type=int, default=64); init.add_argument("--build-index", action="store_true"); init.add_argument("--device", default="cpu"); init.add_argument("--demo", action="store_true"); init.set_defaults(func=cmd_init)
    migrate_cmd = sub.add_parser("migrate", help="connect an existing index and start progressive migration")
    migrate_cmd.add_argument("--index", required=True, help="FAISS index path, or Qdrant path/URL")
    migrate_cmd.add_argument("--documents", required=True, help="JSONL document store with id/text fields")
    migrate_cmd.add_argument("--old-model", required=True, help="source/legacy embedding model ID or local path")
    migrate_cmd.add_argument("--new-model", required=True, help="target embedding model ID or local path")
    migrate_cmd.add_argument("--backend", choices=["faiss", "qdrant"])
    migrate_cmd.add_argument("--index-url", help="optional Qdrant URL (otherwise --index is used)")
    migrate_cmd.add_argument("--collection", default="embedflow")
    migrate_cmd.add_argument("--vector-name", help="Qdrant named-vector key, when the collection uses named vectors")
    migrate_cmd.add_argument("--api-key-env", default="QDRANT_API_KEY", help="environment variable containing a Qdrant API key")
    migrate_cmd.add_argument("--metric", choices=["cosine", "dot", "inner_product"], default="cosine")
    migrate_cmd.add_argument("--model-root", help="directory containing staged research model snapshots")
    migrate_cmd.add_argument("--config", default="./embedflow.yaml", help="where to save the generated migration config")
    migrate_cmd.add_argument("--cache", default="./embedflow_cache")
    migrate_cmd.add_argument("--state", default="./embedflow_state.json")
    migrate_cmd.add_argument("--candidate-depth", type=int, default=50)
    migrate_cmd.add_argument("--kmax-probe", type=int, default=500)
    migrate_cmd.add_argument("--max-sync-misses", type=int, default=4)
    migrate_cmd.add_argument("--background-batch-size", type=int, default=32)
    migrate_cmd.add_argument("--probe-queries", help="optional JSONL queries; runs frozen T2-v1 before serving")
    migrate_cmd.add_argument("--probe-limit", type=int)
    migrate_cmd.add_argument("--device", default=None, help="override configured model device (cpu, cuda, or gpu)")
    migrate_cmd.add_argument("--no-worker", action="store_true", help="disable background materialization worker")
    migrate_cmd.add_argument("--no-serve", action="store_true", help="validate/write config without starting the API")
    migrate_cmd.add_argument("--host", default="127.0.0.1")
    migrate_cmd.add_argument("--port", type=int, default=8000)
    migrate_cmd.add_argument("--log-level", default="info")
    migrate_cmd.set_defaults(func=cmd_migrate)
    analyze = sub.add_parser("analyze", help="run the no-target-index finite-tail/T2-v1 diagnostic")
    analyze.add_argument("--config", help="existing EmbedFlow YAML config")
    analyze.add_argument("--documents", help="JSONL document store for direct analysis")
    analyze.add_argument("--index", help="existing FAISS/Numpy index for direct analysis")
    analyze.add_argument("--index-ids", help="optional FAISS ID sidecar path (defaults to <index>.ids.json)")
    analyze.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss")
    analyze.add_argument("--metric", choices=["cosine", "dot", "inner_product"], default="cosine")
    analyze.add_argument("--collection", default="embedflow", help="Qdrant collection for direct analysis")
    analyze.add_argument("--vector-name", help="Qdrant named-vector key")
    analyze.add_argument("--api-key-env", default="QDRANT_API_KEY", help="Qdrant API-key environment variable")
    analyze.add_argument("--source-model", help="legacy/source model ID or local path")
    analyze.add_argument("--target-model", help="desired target model ID or local path")
    analyze.add_argument("--model-root", help="directory containing staged model snapshots")
    analyze.add_argument("--probe-queries", "--queries", dest="probe_queries", help="JSONL unlabeled probe queries")
    analyze.add_argument("--kmax", type=int)
    analyze.add_argument("--limit", type=int)
    analyze.add_argument("--seed", type=int, default=None)
    analyze.add_argument("--corpus-name", help="canonical registry dataset identifier, when known")
    analyze.add_argument("--corpus-fingerprint", help="precomputed dataset fingerprint for exact registry matching")
    analyze.add_argument("--use-registry", action="store_true", help="reuse canonical evidence only after an exact registry match")
    analyze.add_argument("--device", default=None, help="override configured model device (cpu, cuda, or gpu)")
    analyze.add_argument("--demo", action="store_true")
    analyze.add_argument("--output")
    analyze.add_argument("--output-dir")
    analyze.set_defaults(func=cmd_analyze)
    evaluate = sub.add_parser("evaluate", help="compute qrels/native-target candidate gaps (Mode A)")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--queries")
    evaluate.add_argument("--qrels", required=True)
    evaluate.add_argument("--native-target-index")
    evaluate.add_argument("--native-target-rankings", help="JSONL saved native target rankings")
    evaluate.add_argument("--reference-index", help="exact/reference source index for ANN fidelity")
    evaluate.add_argument("--k-values", default="10,20,50,100,200,500")
    evaluate.add_argument("--quality-k", type=int, default=10)
    evaluate.add_argument("--epsilon", type=float, default=0.01)
    evaluate.add_argument("--bootstrap", type=int, default=0, help="paired bootstrap resamples; 0 disables")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default=None, help="override configured model device (cpu, cuda, or gpu)")
    evaluate.add_argument("--demo", action="store_true")
    evaluate.add_argument("--output-dir", default="./results")
    evaluate.set_defaults(func=cmd_evaluate)
    serve = sub.add_parser("serve", help="start the FastAPI service and dashboard"); serve.add_argument("--config", default="embedflow.yaml"); serve.add_argument("--device", default=None, help="override configured model device (cpu, cuda, or gpu)"); serve.add_argument("--demo", action="store_true"); serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8000); serve.add_argument("--log-level", default="info"); serve.set_defaults(func=cmd_serve)
    command_help = {"status": "show migration, cache, queue, and latency status", "search": "search with source retrieval and target reranking", "prewarm": "materialize selected target vectors", "audit-index": "compare ANN results with an exact/reference index"}
    for name, func in (("status", cmd_status), ("search", cmd_search), ("prewarm", cmd_prewarm), ("audit-index", cmd_audit_index)):
        sp = sub.add_parser(name, help=command_help[name]); sp.add_argument("--config", default="embedflow.yaml"); sp.add_argument("--device", default=None, help="override configured model device (cpu, cuda, or gpu)"); sp.add_argument("--demo", action="store_true"); sp.set_defaults(func=func)
    sub.choices["search"].add_argument("query"); sub.choices["search"].add_argument("--top-k", type=int, default=10)
    pre = sub.choices["prewarm"]; pre.add_argument("--documents", type=int); pre.add_argument("--fraction", type=float, default=.01); pre.add_argument("--ids"); pre.add_argument("--strategy", choices=["random", "popular", "explicit"], default="random"); pre.add_argument("--seed", type=int, default=42); pre.add_argument("--async", dest="async_mode", action="store_true")
    export = sub.add_parser("export-target", help="explicitly materialize all target vectors and build a target index"); export.add_argument("--config", default="embedflow.yaml"); export.add_argument("--output-index", required=True); export.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss"); export.add_argument("--collection", default="embedflow-target"); export.add_argument("--batch-size", type=int, default=32); export.add_argument("--device", default="cpu"); export.add_argument("--demo", action="store_true"); export.set_defaults(func=cmd_export_target)
    audit = sub.choices["audit-index"]; audit.add_argument("--reference-index"); audit.add_argument("--queries"); audit.add_argument("--k", type=int, default=500); audit.add_argument("--limit", type=int)
    econ = sub.add_parser("economics", help="project backfill time and cost from supplied throughput"); econ.add_argument("--config"); econ.add_argument("--corpus-size", type=int, default=0); econ.add_argument("--cached-documents", type=int, default=0); econ.add_argument("--target-docs-per-sec", "--docs-per-second", dest="target_docs_per_sec", type=float); econ.add_argument("--gpu-price", type=float); econ.add_argument("--json", action="store_true", help="emit machine-readable JSON"); econ.add_argument("--demo", action="store_true"); econ.set_defaults(func=cmd_economics)
    demo = sub.add_parser("demo", help="run the self-contained offline progressive-migration demo"); demo.add_argument("--path", default="./examples/local_faiss_demo/runtime"); demo.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss"); demo.add_argument("--no-serve", action="store_true"); demo.add_argument("--host", default="127.0.0.1"); demo.add_argument("--port", type=int, default=8000); demo.set_defaults(func=cmd_demo)
    real = sub.add_parser("real-demo", help="small real-model MiniLM -> Qwen3-0.6B demo")
    real.add_argument("--path", default="./examples/local_faiss_demo/nq_runtime")
    real.add_argument("--corpus", choices=["topics"], default="topics",
                      help="bundled synthetic topic fixture (the public real demo corpus)")
    real.add_argument("--model-root", help="directory for revision-pinned model snapshots (default: ./models)")
    real.add_argument("--download", action="store_true", help="download missing public model snapshots from Hugging Face")
    real.add_argument("--device", default="cpu", help="cpu or cuda; use cuda when available")
    real.add_argument("--no-serve", action="store_true")
    real.add_argument("--host", default="127.0.0.1")
    real.add_argument("--port", type=int, default=8000)
    real.add_argument("--log-level", default="info")
    real.set_defaults(func=cmd_real_demo)
    doctor = sub.add_parser("doctor", help="check dependencies, paths, and configuration")
    doctor.add_argument("--config", default=None)
    doctor.add_argument("--json", action="store_true", help="emit machine-readable checks")
    doctor.set_defaults(func=cmd_doctor)
    registry = sub.add_parser("registry", help="inspect verified migration evidence shipped with EmbedFlow")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_list = registry_sub.add_parser("list", help="list core migration evidence rows")
    registry_list.add_argument("--json", action="store_true")
    registry_list.set_defaults(func=cmd_registry_list)
    registry_show = registry_sub.add_parser("show", help="show curves and provenance for a model transition")
    registry_show.add_argument("--source", required=True)
    registry_show.add_argument("--target", required=True)
    registry_show.add_argument("--json", action="store_true")
    registry_show.set_defaults(func=cmd_registry_show)
    registry_match = registry_sub.add_parser("match", help="match a YAML config against registry contracts")
    registry_match.add_argument("--config", required=True)
    registry_match.add_argument("--corpus", help="optional JSONL corpus to fingerprint")
    registry_match.add_argument("--corpus-name", help="canonical registry dataset identifier")
    registry_match.add_argument("--corpus-fingerprint", help="precomputed dataset fingerprint")
    registry_match.add_argument("--json", action="store_true")
    registry_match.set_defaults(func=cmd_registry_match)
    registry_verify = registry_sub.add_parser("verify", help="validate packaged registry schemas, checksums, and provenance")
    registry_verify.add_argument("--research-root", help="optional retained research checkout root for byte-level provenance checks")
    registry_verify.add_argument("--json", action="store_true")
    registry_verify.set_defaults(func=cmd_registry_verify)
    benchmark_profiles = sub.add_parser("benchmark-profiles", help="list measured latency/throughput profiles")
    benchmark_profiles_sub = benchmark_profiles.add_subparsers(dest="benchmark_command", required=True)
    profiles_list = benchmark_profiles_sub.add_parser("list", help="list workload-specific measured profiles")
    profiles_list.add_argument("--json", action="store_true")
    profiles_list.set_defaults(func=cmd_benchmark_profiles_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try: return int(args.func(args))
    except KeyboardInterrupt: print("interrupted", file=sys.stderr); return 130
    except (FileNotFoundError, RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
        print(f"EmbedFlow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__": raise SystemExit(main())
