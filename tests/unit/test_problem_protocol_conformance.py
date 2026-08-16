"""Keep the Problem protocols honest about what the analyzers actually need.

Two layers:

1. Each maintained Problem must satisfy every analyzer protocol it supports.
   ``runtime_checkable`` validates
   member *presence* only, never signatures.
2. Every ``pde`` member an analyzer touches must be declared in its protocol.
   This is the layer that catches drift at the source: adding a new
   ``self._pde.xxx`` to an analyzer without extending the protocol fails here.

Members reached through ``getattr(self._pde, "name", None)`` are optional by
construction and deliberately out of scope for layer 2.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from soptx.core import (
    DirichletElasticityProblem,
    ElasticityProblem,
    MixedBoundaryElasticityProblem,
)
from soptx.fem.solvers import huzhang_mfem_analyzer, lagrange_fem_analyzer
from soptx.problems import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    FixedFixedBeamCenterLoad2d,
    FullMBBBeam2d,
    HalfMBBBeamRight2d,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)


LAGRANGE_PROBLEM_CLASSES = (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    FixedFixedBeamCenterLoad2d,
    FullMBBBeam2d,
    HalfMBBBeamRight2d,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)

HUZHANG_PROBLEM_CLASSES = (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    FixedFixedBeamCenterLoad2d,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)

# Reached only inside a hasattr guard in the analyzer.
HUZHANG_OPTIONAL_MEMBERS = frozenset({"set_load_region"})

# LagrangeFEMAnalyzer dispatches on boundary_type/load_type and each branch
# needs its own members.  Modelling that tagged union is follow-up work; until
# then the names live here so the guard neither misses nor over-reports.
LAGRANGE_OPTIONAL_MEMBERS = frozenset(
    {
        "adjoint_load_bc",
        "concentrate_load_bc",
        "is_adjoint_load_boundary",
        "is_concentrate_load_boundary",
        "is_neumann_boundary",
        "is_spring_boundary",
        "k_in",
        "k_out",
        "neumann_bc",
        "set_equivalent_traction",
    }
)

ANALYZER_CONTRACTS = (
    (
        huzhang_mfem_analyzer,
        MixedBoundaryElasticityProblem,
        HUZHANG_OPTIONAL_MEMBERS,
    ),
    (
        lagrange_fem_analyzer,
        DirichletElasticityProblem,
        LAGRANGE_OPTIONAL_MEMBERS,
    ),
)


def declared_members(protocol) -> set[str]:
    """Return the public members a protocol requires, inherited ones included."""
    members: set[str] = set()
    for base in protocol.__mro__:
        if base is object:
            continue
        members.update(
            name for name in vars(base) if not name.startswith("_")
        )
        members.update(
            name
            for name in getattr(base, "__annotations__", {})
            if not name.startswith("_")
        )
    return members


def accessed_pde_members(module) -> set[str]:
    """Return every attribute the module reads off ``self._pde`` or ``pde``."""
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    members: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        reads_attribute_of_self_pde = (
            isinstance(owner, ast.Attribute)
            and owner.attr == "_pde"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        )
        reads_attribute_of_local_pde = (
            isinstance(owner, ast.Name) and owner.id == "pde"
        )
        if reads_attribute_of_self_pde or reads_attribute_of_local_pde:
            members.add(node.attr)
    return members


@pytest.mark.parametrize("problem_class", LAGRANGE_PROBLEM_CLASSES)
def test_maintained_problems_satisfy_lagrange_contract(
    problem_class,
) -> None:
    problem = problem_class()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, DirichletElasticityProblem)


@pytest.mark.parametrize("problem_class", HUZHANG_PROBLEM_CLASSES)
def test_huzhang_supported_problems_satisfy_mixed_contract(
    problem_class,
) -> None:
    problem = problem_class()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, MixedBoundaryElasticityProblem)


@pytest.mark.parametrize(
    ("module", "protocol", "optional_members"),
    ANALYZER_CONTRACTS,
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_analyzers_only_use_declared_problem_members(
    module,
    protocol,
    optional_members,
) -> None:
    accessed = accessed_pde_members(module)
    assert accessed, "the AST scan found no pde access at all"

    undeclared = accessed - declared_members(protocol) - optional_members

    assert not undeclared, (
        f"{module.__name__} reads {sorted(undeclared)} off its pde, but "
        f"{protocol.__name__} does not declare them. Extend the protocol in "
        f"src/soptx/core/protocols.py, or add the name to the optional list "
        f"here when the access is guarded."
    )
