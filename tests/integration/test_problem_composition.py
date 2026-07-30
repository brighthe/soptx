from __future__ import annotations

from fealpy.mesh import TetrahedronMesh, TriangleMesh

from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)


def test_2d_problem_material_mesh_are_explicitly_composed() -> None:
    problem = SinusoidalPlaneStrainElasticity2D()
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis="plane_strain",
    )
    mesh = TriangleMesh.from_box(list(problem.domain), nx=2, ny=2)

    assert mesh.geo_dimension() == problem.dimension
    assert material.hypothesis == problem.plane_type


def test_3d_problem_material_mesh_are_explicitly_composed() -> None:
    problem = DivergenceFreePolynomialElasticity3D()
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis="3D",
    )
    mesh = TetrahedronMesh.from_box(
        list(problem.domain),
        nx=1,
        ny=1,
        nz=1,
    )

    assert mesh.geo_dimension() == problem.dimension
    assert material.hypothesis == problem.plane_type
