"""EmbedFlow: progressive embedding-model migration for existing indexes."""

__version__ = "0.8.0"

from .config import (
    EmbedFlowConfig,
    PlannerConfig,
    PrewarmConfig,
    RuntimeConfig,
    ShadowConfig,
    ShadowTelemetryConfig,
    load_config,
)
from .prewarm import PrewarmPlan, PrewarmPlanner, PrewarmRunner, TrafficHotsetPlanner, load_prewarm_plan


def plan(*args, **kwargs):
    """Generate an advisory migration plan without changing source traffic."""
    from .planner import plan_migration
    return plan_migration(*args, **kwargs)


def migrate(*args, **kwargs):
    """Start progressive migration over an existing FAISS, Qdrant, pgvector, Pinecone, Milvus, or Weaviate index.

    Imported lazily to keep the lightweight configuration package free of
    model-serving dependencies at import time.  See ``embedflow.migration``
    for the ``MigrationSession`` type.
    """
    from .migration.facade import migrate as _migrate
    return _migrate(*args, **kwargs)


def analyze_migration(*args, **kwargs):
    """Run the leakage-safe no-target-index analysis programmatically."""
    from .analysis import analyze_migration as _analyze_migration
    return _analyze_migration(*args, **kwargs)


def prewarm_plan(telemetry, cache, **kwargs):
    """Build a bounded traffic-hotset plan through the reusable API.

    Planner-constructor options and ``plan`` options may be supplied together;
    recognized plan options are routed to :meth:`PrewarmPlanner.plan`.
    """
    plan_keys = {"since_seconds", "start", "end", "max_docs", "target_observed_coverage",
                 "max_storage_gb", "max_runtime_seconds", "docs_per_second", "gpu_hourly_cost", "strategy"}
    plan_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in plan_keys}
    return PrewarmPlanner(telemetry, cache, **kwargs).plan(**plan_kwargs)


def prewarm_run(plan, *args, **kwargs):
    """Execute a validated prewarm plan through the existing materializer."""
    run_keys = {"max_runtime_seconds", "progress"}
    run_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in run_keys}
    return PrewarmRunner(*args, **kwargs).run(plan, **run_kwargs)


__all__ = ["EmbedFlowConfig", "PlannerConfig", "PrewarmConfig", "RuntimeConfig", "ShadowConfig", "ShadowTelemetryConfig", "load_config", "migrate", "plan", "analyze_migration", "prewarm_plan", "prewarm_run", "PrewarmPlan", "PrewarmPlanner", "TrafficHotsetPlanner", "PrewarmRunner", "load_prewarm_plan", "__version__"]
