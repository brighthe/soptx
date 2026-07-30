"""Compatibility integrator namespace retained for SOPTX 1.1.x."""

from soptx.fem.integrators import (
    HuZhangMixIntegrator,
    HuZhangStressIntegrator,
    JumpPenaltyIntegrator,
    LinearElasticIntegrator,
    MassIntegrator,
    SourceIntegrator,
)

__all__ = [
    "HuZhangMixIntegrator",
    "HuZhangStressIntegrator",
    "JumpPenaltyIntegrator",
    "LinearElasticIntegrator",
    "MassIntegrator",
    "SourceIntegrator",
]
