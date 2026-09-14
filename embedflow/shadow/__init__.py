"""First-class, source-authoritative Shadow Mode."""

from .models import SAFE_FAILURE_CATEGORIES, ShadowObservation
from .report import render_shadow_report
from .runner import ShadowRunner, ShadowTaskError
from .telemetry import SCHEMA_VERSION, ShadowTelemetry, index_identity_from_config, migration_fingerprint

__all__ = ["SCHEMA_VERSION", "SAFE_FAILURE_CATEGORIES", "ShadowObservation", "ShadowRunner", "ShadowTaskError", "ShadowTelemetry",
           "index_identity_from_config", "migration_fingerprint", "render_shadow_report"]
