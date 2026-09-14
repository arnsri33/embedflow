"""EmbedFlow: progressive embedding-model migration for existing indexes."""

__version__ = "0.7.0"

from .config import EmbedFlowConfig, PlannerConfig, RuntimeConfig, ShadowConfig, ShadowTelemetryConfig, load_config


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


__all__ = ["EmbedFlowConfig", "PlannerConfig", "RuntimeConfig", "ShadowConfig", "ShadowTelemetryConfig", "load_config", "migrate", "plan", "analyze_migration", "__version__"]
