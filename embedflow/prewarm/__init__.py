"""Traffic-aware target-vector prewarming.

The public classes in this module deliberately sit above the source-index
adapters.  Planning reads privacy-safe Shadow Mode aggregates and the target
cache; running a plan delegates document encoding to the existing durable
materialization worker.  No source index write path is exposed here.
"""

from .models import MAX_INLINE_PREWARM_IDS, PREWARM_SCHEMA_VERSION, PrewarmPlan, PrewarmWarning
from .planner import PrewarmPlanner, TrafficHotsetPlanner
from .runner import PrewarmRunner, load_prewarm_plan, prewarm_status

__all__ = [
    "PREWARM_SCHEMA_VERSION",
    "MAX_INLINE_PREWARM_IDS",
    "PrewarmPlan",
    "PrewarmWarning",
    "PrewarmPlanner",
    "TrafficHotsetPlanner",
    "PrewarmRunner",
    "load_prewarm_plan",
    "prewarm_status",
]
