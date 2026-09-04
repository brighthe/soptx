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

from soptx.protocols import (
    AnalysisStage,
    DirichletElasticityProblem,
    ElasticityProblem,
    MixedBoundaryElasticityProblem,
)
from soptx.fem.analyzers import (
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
    huzhang_mfem_analyzer,
    lagrange_fem_analyzer,
)
from soptx.problems import (
    CantileverCorner2d,
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
    CantileverCorner2d,
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

HUZHANG_OPTIONAL_MEMBERS = frozenset()

# 伴随右端项和弹簧支承不属于物理外载荷对象, 仍由分析器单独消费.
LAGRANGE_OPTIONAL_MEMBERS = frozenset(
    {
        "adjoint_load_bc",
        "is_adjoint_load_boundary",
        "is_spring_boundary",
        "k_in",
        "k_out",
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
        f"src/soptx/protocols/__init__.py, or add the name to the optional "
        f"list here when the access is guarded."
    )


# Analyse-stage protocol: both analyzers serve the topology "analysis stage"
# (state solve + stress + adjoint).  Checked at class level (member presence)
# rather than by constructing an instance, because HuZhang's 3D constructor
# currently trips an unrelated FEALPy-space bug.
ANALYSIS_STAGE_CLASSES = (LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer)


@pytest.mark.parametrize(
    "analyzer_cls", ANALYSIS_STAGE_CLASSES,
    ids=lambda cls: cls.__name__,
)
def test_analyzers_satisfy_analysis_stage_protocol(analyzer_cls) -> None:
    members = set(getattr(AnalysisStage, "__protocol_attrs__", ()))
    missing = {name for name in members if not hasattr(analyzer_cls, name)}
    assert not missing, (
        f"{analyzer_cls.__name__} does not satisfy {AnalysisStage.__name__}; "
        f"missing {sorted(missing)}. Declare the member in the protocol or "
        f"implement it on the analyzer."
    )
