"""
Pipeline entrypoints for the simplified CP framework.
"""

from .marginal import MarginalConfig, run_marginal
from .conditional import (
    ConditionalConfig,
    run_empirical_cqr,
    run_ltt_certified,
)

__all__ = [
    "MarginalConfig",
    "run_marginal",
    "ConditionalConfig",
    "run_empirical_cqr",
    "run_ltt_certified",
]

