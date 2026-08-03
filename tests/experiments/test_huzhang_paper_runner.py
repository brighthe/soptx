"""Focused tests for the Hu–Zhang common evidence runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "huzhang_topopt_paper"


def _load_module(name: str, filename: str):
    path = EXPERIMENT_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_module("huzhang_paper_runner", "run.py")


def test_common_configuration_covers_fixed_matrix(runner):
    configuration = runner.load_configuration()
    assert configuration.stage == runner.EXPECTED_STAGE
    assert [case.identifier for case in configuration.cases] == [
        "forward-manufactured",
        "boundary-ablation",
        "sensitivity-checks",
        "compliance-fixed-fixed",
        "near-incompressible-bearing",
        "stress-constrained-cantilever",
        "frozen-design-verification",
    ]
    ready = [
        case.identifier
        for case in configuration.cases
        if case.execution_status == "ready"
    ]
    assert ready == ["forward-manufactured"]
    assert all(
        case.blockers
        for case in configuration.cases
        if case.execution_status == "configured"
    )
    cases = configuration.by_id()
    assert cases["forward-manufactured"].orders == (1, 2, 3, 4)
    assert cases["forward-manufactured"].levels == (1, 2, 3)
    assert cases["forward-manufactured"].mesh_families == (
        "uniform-tri",
        "irregular-star-refined",
    )
    assert cases["compliance-fixed-fixed"].orders == (2, 3)
    assert cases["compliance-fixed-fixed"].levels == (1, 2)
    bearing = cases["near-incompressible-bearing"]
    assert bearing.orders == (2, 3)
    assert bearing.parameters["poisson_ratios"] == [0.3, 0.49, 0.499, 0.4999]
    assert bearing.parameters["huzhang_endpoint_order"] == 3
    stress = cases["stress-constrained-cantilever"]
    assert stress.orders == (2, 3)
    assert stress.levels == (1, 2)
    assert stress.parameters["huzhang_fine_order"] == 3


def test_acceptance_thresholds_are_explicit(runner):
    acceptance = runner.load_configuration().acceptance
    assert acceptance == {
        "high_order_rate_deficit": 0.35,
        "relative_equilibrium_residual_max": 1.0e-8,
        "normalized_normal_trace_jump_max": 1.0e-10,
        "finite_difference_relative_error_max": 1.0e-5,
        "volume_fraction_absolute_error_max": 1.0e-3,
        "stress_constraint_violation_max": 3.0e-3,
        "frozen_design_relative_change_max": 1.0e-2,
    }


def test_configured_case_writes_blocked_manifest(runner, tmp_path):
    configuration = runner.load_configuration()
    case = configuration.by_id()["boundary-ablation"]
    manifest, exit_code = runner.run_case(configuration, case, tmp_path)
    assert exit_code == 2
    assert manifest["status"] == "blocked"
    assert manifest["provenance"]["evidence_level"] == "development"
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "history.csv").is_file()
    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["case"]["id"] == "boundary-ablation"
    assert set(persisted["artifacts"]) == {
        "history.csv",
        "metrics.csv",
        "summary.json",
    }
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in persisted["artifacts"].values()
    )
    provenance = persisted["provenance"]
    assert len(provenance["configuration"]["sha256"]) == 64
    assert provenance["configuration"]["stage"] == runner.EXPECTED_STAGE
    assert provenance["python_executable"]
    assert set(provenance["repositories"]) == {"soptx", "fealpy"}
    for repository in provenance["repositories"].values():
        assert {"revision", "dirty"} <= set(repository)
    assert set(provenance["environment"]) == {
        "conda_default_env",
        "conda_prefix",
        "virtual_env",
    }


def test_output_filename_rejects_parent_traversal(runner):
    with pytest.raises(runner.ConfigurationError, match="unsafe output"):
        runner._safe_output_name("../manifest.json")


def test_small_mixed_boundary_state_has_expected_diagnostics():
    executors = _load_module("executors", "executors.py")
    from fealpy.backend import backend_manager as bm
    from soptx.fem import HuZhangMFEMAnalyzer
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems import MixedBoundaryExponentialSineElasticity2D

    bm.set_backend("numpy")
    problem = MixedBoundaryExponentialSineElasticity2D(
        lame_lambda=1.0,
        shear_modulus=0.5,
    )
    mesh = executors._uniform_mesh(level=1)
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis="plane_strain",
        enable_logging=False,
    )
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=3,
        integration_order=8,
        use_relaxation=True,
        solve_method="scipy",
        topopt_algorithm=None,
    )
    analyzer.solve_state(rho_val=None)
    assert analyzer.relative_state_residual() <= 1.0e-8
    assert analyzer.state_matrix_symmetry_error() <= 1.0e-12
    assert bool(mesh.edgedata["essential_bc"].any())
    assert bool(mesh.edgedata["natural_bc"].any())
    assert not bool(
        (mesh.edgedata["essential_bc"] & mesh.edgedata["natural_bc"]).any()
    )


@pytest.mark.parametrize(
    "mesh_factory_name",
    ["_uniform_mesh", "_irregular_star_mesh"],
)
def test_relaxation_meshes_have_four_supported_corners(mesh_factory_name):
    executors = _load_module("executors", "executors.py")
    from fealpy.backend import backend_manager as bm
    from soptx.fem.spaces import HuZhangFESpace
    from soptx.problems import MixedBoundaryExponentialSineElasticity2D

    bm.set_backend("numpy")
    problem = MixedBoundaryExponentialSineElasticity2D(
        lame_lambda=1.0,
        shear_modulus=0.5,
    )
    mesh = getattr(executors, mesh_factory_name)(level=1)
    corners = problem.mark_corners(mesh.entity("node"))
    conforming = HuZhangFESpace(mesh=mesh, p=2, use_relaxation=False)
    relaxed = HuZhangFESpace(mesh=mesh, p=2, use_relaxation=True, corners=corners)
    assert relaxed.NCP == 4
    assert (
        relaxed.number_of_global_dofs()
        == conforming.number_of_global_dofs() + relaxed.NCP
    )
    assert relaxed.cell_to_dof().shape == conforming.cell_to_dof().shape
