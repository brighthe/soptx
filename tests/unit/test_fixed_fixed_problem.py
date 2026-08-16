from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.core import (
    DirichletElasticityProblem,
    ElasticityProblem,
    MixedBoundaryElasticityProblem,
)
from soptx.problems import FixedFixedBeamCenterLoad2d


def test_fixed_fixed_problem_satisfies_both_analyzer_contracts() -> None:
    problem = FixedFixedBeamCenterLoad2d()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, DirichletElasticityProblem)
    assert isinstance(problem, MixedBoundaryElasticityProblem)
    assert not hasattr(problem, "init_mesh")


def test_fixed_fixed_problem_shares_one_load_for_both_discretizations() -> None:
    bm.set_backend("numpy")
    problem = FixedFixedBeamCenterLoad2d()
    points = bm.array(
        [
            [0.0, 10.0],
            [160.0, 10.0],
            [80.0, 0.0],
            [79.0, 0.0],
            [80.0, 20.0],
            [80.0, 10.0],
        ]
    )

    is_dof_x, is_dof_y = problem.is_dirichlet_boundary()
    expected_fixed = np.array([True, True, False, False, False, False])
    np.testing.assert_array_equal(bm.to_numpy(is_dof_x(points)), expected_fixed)
    np.testing.assert_array_equal(bm.to_numpy(is_dof_y(points)), expected_fixed)
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_displacement_boundary(points)),
        expected_fixed,
    )

    expected_traction = np.array([False, False, True, True, True, False])
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_traction_boundary(points)),
        expected_traction,
    )
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_neumann_boundary()(points)),
        expected_traction,
    )

    expected_values = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, -3.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        bm.to_numpy(problem.traction_bc(points)),
        expected_values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        bm.to_numpy(problem.neumann_bc(points)),
        expected_values,
        rtol=0.0,
        atol=0.0,
    )
