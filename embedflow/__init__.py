"""EmbedFlow: progressive embedding-model migration for existing indexes."""

__version__ = "0.1.1"

from .config import EmbedFlowConfig, load_config


def migrate(*args, **kwargs):
    """Start progressive migration over an existing FAISS/Qdrant index.

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


__all__ = ["EmbedFlowConfig", "load_config", "migrate", "analyze_migration", "__version__"]
