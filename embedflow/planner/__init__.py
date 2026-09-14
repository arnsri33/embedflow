"""Advisory, evidence-aware migration planning."""

from .economics import estimate_economics, quantity
from .models import PlanResult, PlanWarning
from .planner import MigrationPlanner, load_probe_queries, plan_migration
from .rendering import render_plan

# Short alias for applications that prefer ``embedflow.planner.plan(...)``;
# the top-level ``embedflow.plan`` facade delegates to the same function.
plan = plan_migration

__all__ = [
    "MigrationPlanner",
    "PlanResult",
    "PlanWarning",
    "estimate_economics",
    "load_probe_queries",
    "plan",
    "plan_migration",
    "quantity",
    "render_plan",
]
