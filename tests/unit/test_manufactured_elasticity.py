from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    HarmonicPoly2D,
    HarmonicPoly3D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)
from soptx.problems.loads import BodyForceLoad, BoundaryTractionLoad


@pytest.mark.parametrize(
    "dimension, problem_type", [(2, HarmonicPoly2D), (3, HarmonicPoly3D)]
)
def test_harmonic_poly_public_names(dimension, problem_type) -> None:
    """短名称为实际类名, 旧名称在各公开入口保持同一对象."""
    class_name = f"HarmonicPoly{dimension}D"
    legacy_name = f"HarmonicPolynomialElasticity{dimension}D"
    assert problem_type.__name__ == class_name
    assert type(problem_type()).__name__ == class_name
    for module_name in (
        "soptx.problems",
        "soptx.problems.elasticity",
        f"soptx.problems.elasticity.manufactured_{dimension}d",
    ):
        module = import_module(module_name)
        assert getattr(module, class_name) is problem_type
        assert getattr(module, legacy_name) is problem_type
        if module_name in ("soptx.problems", "soptx.problems.elasticity"):
            assert class_name in module.__all__
            assert legacy_name in module.__all__


def _body_force(problem, points):
    load = next(load for load in problem.loads() if isinstance(load, BodyForceLoad))
    return load.body_force(points)


def _boundary_traction(problem, points):
    load = next(
        load for load in problem.loads() if isinstance(load, BoundaryTractionLoad)
    )
    return load.traction(points)


def test_problem_contract_has_no_mesh_factory() -> None:
    problems = (
        ExponentialSineManufacturedElasticity2D(),
        MixedBoundarySinusoidalElasticity2D(),
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
            MixedBoundarySinusoidalElasticity2D(),
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
        assert _body_force(problem, points).shape == expected_vector_shape
        assert problem.dirichlet_bc(points).shape == expected_vector_shape
        assert (
            problem.grad_disp_solution(points).shape
            == expected_gradient_shape
        )


def test_mixed_boundary_sinusoidal_problem_exact_values() -> None:
    bm.set_backend("numpy")
    problem = MixedBoundarySinusoidalElasticity2D(
        lame_lambda=1.0,
        shear_modulus=0.5,
    )

    center = bm.array([[0.5, 0.5]])
    np.testing.assert_allclose(
        bm.to_numpy(problem.disp_solution(center)),
        np.array([[1.0, 1.0]]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        bm.to_numpy(problem.stress_solution(center)),
        np.zeros((1, 3)),
        rtol=0.0,
        atol=1.0e-14,
    )
    expected_divergence = np.full((1, 2), -2.5 * np.pi**2)
    np.testing.assert_allclose(
        bm.to_numpy(problem.div_stress_solution(center)),
        expected_divergence,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        bm.to_numpy(_body_force(problem, center)),
        -expected_divergence,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    left_midpoint = bm.array([[0.0, 0.5]])
    np.testing.assert_allclose(
        bm.to_numpy(problem.stress_solution(left_midpoint)),
        np.array([[2.0 * np.pi, 0.5 * np.pi, np.pi]]),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_mixed_boundary_sinusoidal_problem_boundary_partition() -> None:
    bm.set_backend("numpy")
    problem = MixedBoundarySinusoidalElasticity2D()
    edge_midpoints = bm.array(
        [[0.0, 0.4], [0.4, 0.0], [1.0, 0.4], [0.4, 1.0]]
    )

    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_displacement_boundary(edge_midpoints)),
        np.array([True, True, False, False]),
    )
    np.testing.assert_array_equal(
        bm.to_numpy(problem.is_traction_boundary(edge_midpoints)),
        np.array([False, False, True, True]),
    )
    np.testing.assert_allclose(
        bm.to_numpy(_boundary_traction(problem, edge_midpoints)),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                bm.to_numpy(problem.stress_solution(edge_midpoints))[2, :2],
                bm.to_numpy(problem.stress_solution(edge_midpoints))[3, 1:],
            ]
        ),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_mixed_boundary_sinusoidal_problem_rejects_invalid_material() -> None:
    with pytest.raises(ValueError, match="lame_lambda"):
        MixedBoundarySinusoidalElasticity2D(lame_lambda=float("inf"))
    with pytest.raises(ValueError, match="shear_modulus"):
        MixedBoundarySinusoidalElasticity2D(shear_modulus=0.0)


def test_new_problem_values_match_pre_v2_problem_values() -> None:
    from old.old_version2.model.linear_elasticity_2d import (
        BoxTriLagrange2dData,
        TriSolHomoDirHuZhang2d,
    )
    from old.old_version2.model.linear_elasticity_3d import (
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
            bm.to_numpy(_body_force(current, points)),
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
