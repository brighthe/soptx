from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.core import DirichletElasticityProblem, ElasticityProblem
from soptx.problems import HalfMBBBeamRight2d


def test_half_mbb_problem_is_mesh_independent_and_satisfies_lagrange_contract() -> None:
    problem = HalfMBBBeamRight2d()

    assert isinstance(problem, ElasticityProblem)
    assert isinstance(problem, DirichletElasticityProblem)
    assert not hasattr(problem, "init_mesh")
    assert not hasattr(problem, "get_passive_element_mask")


def test_half_mbb_problem_boundary_conditions_and_concentrated_load() -> None:
    bm.set_backend("numpy")
    problem = HalfMBBBeamRight2d(P=-2.5)
    points = bm.array([[0.0, 20.0], [60.0, 0.0], [30.0, 10.0]])

    np.testing.assert_allclose(
        bm.to_numpy(problem.body_force(points)),
        np.zeros((3, 2)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        bm.to_numpy(problem.dirichlet_bc(points)),
        np.zeros((3, 2)),
        rtol=0.0,
        atol=0.0,
    )

    dirichlet_x, dirichlet_y = problem.is_dirichlet_boundary()
    np.testing.assert_array_equal(
        bm.to_numpy(dirichlet_x(points)),
        np.array([True, False, False]),
    )
    np.testing.assert_array_equal(
        bm.to_numpy(dirichlet_y(points)),
        np.array([False, True, False]),
    )

    load_values = problem.concentrate_load_bc()
    load_boundaries = problem.is_concentrate_load_boundary()
    assert len(load_values) == len(load_boundaries) == 1
    load_mask = load_boundaries[0](points)
    np.testing.assert_array_equal(
        bm.to_numpy(load_mask),
        np.array([True, False, False]),
    )

    concentrated_vector = load_values[0](points)[load_mask]
    np.testing.assert_allclose(
        bm.to_numpy(concentrated_vector),
        np.array([[0.0, -2.5]]),
        rtol=0.0,
        atol=0.0,
    )
    assert float(bm.sum(concentrated_vector)) == problem.P
