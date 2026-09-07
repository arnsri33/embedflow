"""Research-facing compatibility analysis.

The functions in this package are deliberately separate from the serving
engine.  Analysis with qrels/native target rankings is Mode A (research
evaluation); serving without a target index is Mode B (finite-tail/T2-v1
diagnostics).
"""

from .candidate_gap import CandidateGapCurve, compute_candidate_gap_curve
from .containment import candidate_containment
from .evaluate import evaluate_models, evaluate_rankings, evaluate_with_native_rankings
from .migration_depth import observed_migration_depth, recommend_initial_k
from .probe import run_finite_pool_probe
from .t2 import T2Diagnostic, diagnose_t2

__all__ = [
    "CandidateGapCurve",
    "T2Diagnostic",
    "candidate_containment",
    "compute_candidate_gap_curve",
    "diagnose_t2",
    "evaluate_models",
    "evaluate_rankings",
    "evaluate_with_native_rankings",
    "observed_migration_depth",
    "recommend_initial_k",
    "run_finite_pool_probe",
]
