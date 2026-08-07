from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fealpy.mesh import Mesh, TetrahedronMesh, TriangleMesh

from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)

try:
    import utils.contract as contract
except ImportError:
    import contract



@dataclass(frozen=True)
class MaterialSpec:
    """Dimension-specific material data independent of the PDE object."""

    hypothesis: str
    parameters: tuple[tuple[str, float], ...]

    def create(self, *, device: str) -> IsotropicLinearElasticMaterial:
        return IsotropicLinearElasticMaterial(
            hypothesis=self.hypothesis,
            device=device,
            **dict(self.parameters),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            **dict(self.parameters),
        }


@dataclass(frozen=True)
class ElasticityCase:
    """All dimension-dependent choices for one manufactured solution."""

    dimension: int
    name: str
    domain: tuple[float, ...]
    problem: object
    material: MaterialSpec
    mesh_entity_name: str

    def resolution(
        self,
        *,
        nx: int,
        ny: int,
        nz: int | None,
    ) -> tuple[int, ...]:
        if self.dimension == 2:
            if nz is not None:
                raise ValueError("--nz is only valid when --dim 3")
            return (nx, ny)
        if nz is None:
            nz = contract.DEFAULT_RESOLUTION
        return (nx, ny, nz)

    def create_mesh(self, resolution: tuple[int, ...]) -> Mesh:
        if self.dimension == 2:
            nx, ny = resolution
            return TriangleMesh.from_box(
                list(self.domain),
                nx=nx,
                ny=ny,
            )
        nx, ny, nz = resolution
        return TetrahedronMesh.from_box(
            list(self.domain),
            nx=nx,
            ny=ny,
            nz=nz,
        )

    def validate_mesh(self, mesh: Mesh) -> None:
        dimension = int(mesh.geo_dimension())
        if dimension != self.dimension:
            raise ValueError(
                f"case dimension {self.dimension} does not match "
                f"mesh dimension {dimension}"
            )
        values = self.problem.disp_solution(
            mesh.Entity("cell").barycenter()
        )
        if values.shape[-1] != self.dimension:
            raise ValueError(
                "exact displacement component count does not match "
                f"dimension {self.dimension}: shape={values.shape}"
            )

    def barycentric_coordinate(self) -> tuple[float, ...]:
        value = 1.0 / (self.dimension + 1)
        return (value,) * (self.dimension + 1)

    def partition_split_coordinate(self, axis: int = 0) -> float:
        """Midpoint of the domain along ``axis``, used to split cells."""

        if not 0 <= axis < self.dimension:
            raise ValueError(
                f"axis {axis} is out of range for dimension {self.dimension}"
            )
        low = self.domain[2 * axis]
        high = self.domain[2 * axis + 1]
        return 0.5 * (low + high)


def create_case(dimension: int) -> ElasticityCase:
    """Create one of the two supported manufactured-solution cases."""

    if dimension == 2:
        domain = (0.0, 1.0, 0.0, 1.0)
        youngs_modulus = 1.0
        poisson_ratio = 0.3
        return ElasticityCase(
            dimension=2,
            name="sinusoidal-plane-strain",
            domain=domain,
            problem=SinusoidalPlaneStrainElasticity2D(
                domain=domain,
                youngs_modulus=youngs_modulus,
                poisson_ratio=poisson_ratio,
            ),
            material=MaterialSpec(
                hypothesis="plane_strain",
                parameters=(
                    ("youngs_modulus", youngs_modulus),
                    ("poisson_ratio", poisson_ratio),
                ),
            ),
            mesh_entity_name="tri",
        )
    if dimension == 3:
        domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        lame_lambda = 1.0
        shear_modulus = 1.0
        return ElasticityCase(
            dimension=3,
            name="divergence-free-polynomial",
            domain=domain,
            problem=DivergenceFreePolynomialElasticity3D(
                domain=domain,
                lame_lambda=lame_lambda,
                shear_modulus=shear_modulus,
            ),
            material=MaterialSpec(
                hypothesis="3D",
                parameters=(
                    ("lame_lambda", lame_lambda),
                    ("shear_modulus", shear_modulus),
                ),
            ),
            mesh_entity_name="tet",
        )
    raise ValueError(
        f"dimension must be one of {contract.SUPPORTED_DIMENSIONS}, "
        f"received {dimension}"
    )
