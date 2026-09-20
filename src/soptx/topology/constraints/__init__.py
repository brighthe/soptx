"""拓扑优化约束的公共入口."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exemption import (
        apply_exemption as apply_exemption,
        apply_passive_solid as apply_passive_solid,
        build_exemption_mask as build_exemption_mask,
        validate_exemption_mask as validate_exemption_mask,
    )
    from .huzhang_stress import HuZhangStressConstraint as HuZhangStressConstraint
    from .lagrange_stress import LagrangeStressConstraint as LagrangeStressConstraint
    from .stress_formulation import (
        EpsilonRelaxedStressFormulation as EpsilonRelaxedStressFormulation,
        PolynomialVanishingStressFormulation as PolynomialVanishingStressFormulation,
        StressConstraintProtocol as StressConstraintProtocol,
        StressRelaxationFormulation as StressRelaxationFormulation,
    )
    from .volume import VolumeConstraint as VolumeConstraint

_EXPORTS = {
    "apply_exemption": (".exemption", "apply_exemption"),
    "apply_passive_solid": (".exemption", "apply_passive_solid"),
    "build_exemption_mask": (".exemption", "build_exemption_mask"),
    "validate_exemption_mask": (".exemption", "validate_exemption_mask"),
    "HuZhangStressConstraint": (".huzhang_stress", "HuZhangStressConstraint"),
    "LagrangeStressConstraint": (".lagrange_stress", "LagrangeStressConstraint"),
    "StressConstraintProtocol": (".stress_formulation", "StressConstraintProtocol"),
    "StressRelaxationFormulation": (".stress_formulation", "StressRelaxationFormulation"),
    "PolynomialVanishingStressFormulation": (
        ".stress_formulation", "PolynomialVanishingStressFormulation",
    ),
    "EpsilonRelaxedStressFormulation": (
        ".stress_formulation", "EpsilonRelaxedStressFormulation",
    ),
    "VolumeConstraint": (".volume", "VolumeConstraint"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
