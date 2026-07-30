from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)


def test_problem_contract_has_no_mesh_factory() -> None:
    problems = (
        ExponentialSineManufacturedElasticity2D(),
        SinusoidalPlaneStrainElasticity2D(),
        DivergenceFreePolynomialElasticity3D(),
    )
    assert all(not hasattr(problem, "init_mesh") for problem in problems)


def test_problem_output_shapes() -> None:
    cases = (
        (
            ExponentialSineManufacturedElasticity2D(),
            np.array([[0.25, 0.4], [0.6, 0.75]]),
        ),
        (
            SinusoidalPlaneStrainElasticity2D(),
            np.array([[0.25, 0.4], [0.6, 0.75]]),
        ),
        (
            DivergenceFreePolynomialElasticity3D(),
            np.array([[0.25, 0.4, 0.6], [0.6, 0.75, 0.2]]),
        ),
    )
    for problem, points in cases:
        expected_vector_shape = points.shape
        expected_gradient_shape = (
            points.shape[0],
            problem.dimension,
            problem.dimension,
        )
        assert problem.disp_solution(points).shape == expected_vector_shape
        assert problem.body_force(points).shape == expected_vector_shape
        assert problem.dirichlet_bc(points).shape == expected_vector_shape
        assert (
            problem.grad_disp_solution(points).shape
            == expected_gradient_shape
        )


def test_new_problem_values_match_pre_v2_problem_values() -> None:
    from soptx.model.linear_elasticity_2d import (
        BoxTriLagrange2dData,
        TriSolHomoDirHuZhang2d,
    )
    from soptx.model.linear_elasticity_3d import (
        PolySolPureDirLagrange3d,
    )

    bm.set_backend("numpy")
    cases = (
        (
            ExponentialSineManufacturedElasticity2D(),
            TriSolHomoDirHuZhang2d(),
            bm.array([[0.25, 0.4], [0.6, 0.75]]),
            "grad_disp_solution",
        ),
        (
            SinusoidalPlaneStrainElasticity2D(),
            BoxTriLagrange2dData(),
            bm.array([[0.25, 0.4], [0.6, 0.75]]),
            "disp_solution_gradient",
        ),
        (
            DivergenceFreePolynomialElasticity3D(),
            PolySolPureDirLagrange3d(),
            bm.array(
                [[0.25, 0.4, 0.6], [0.6, 0.75, 0.2]]
            ),
            "grad_disp_solution",
        ),
    )
    for current, legacy, points, legacy_gradient_name in cases:
        np.testing.assert_allclose(
            bm.to_numpy(current.disp_solution(points)),
            bm.to_numpy(legacy.disp_solution(points)),
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            bm.to_numpy(current.body_force(points)),
            bm.to_numpy(legacy.body_force(points)),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            bm.to_numpy(current.grad_disp_solution(points)),
            bm.to_numpy(
                getattr(legacy, legacy_gradient_name)(points)
            ),
            rtol=0.0,
            atol=1.0e-14,
        )
