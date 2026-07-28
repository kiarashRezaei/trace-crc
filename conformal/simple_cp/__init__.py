"""
Simplified conformal prediction utilities aimed at statisticians.

Modules are intentionally function-based (minimal OO) so each step of the
pipeline can be inspected or swapped independently.
"""

__all__ = [
    "data_generation",
    "empirical_risk",
    "grouping",
    "lambda_functions",
    "models",
    "MarginalConfig",
    "run_marginal",
    "ConditionalConfig",
    "run_empirical_cqr",
    "run_ltt_certified",
]
