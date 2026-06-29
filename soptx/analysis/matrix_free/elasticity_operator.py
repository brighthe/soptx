from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.functionspace.tensor_space import TensorFunctionSpace
from fealpy.typing import TensorLike

from soptx.analysis.integrators.linear_elastic_integrator import LinearElasticIntegrator
from .boundary import MatrixFreeDirichletBC


class MatrixFreeElasticityOperator:
    """Global matrix-free wrapper for the SOPTX linear elasticity integrator."""

    def __init__(self,
                 space: TensorFunctionSpace,
                 integrator: LinearElasticIntegrator,
                 rho: Optional[TensorLike] = None,
                 material_model=None,
                 dirichlet_dofs: Optional[TensorLike] = None,
                 boundary: Optional[MatrixFreeDirichletBC] = None) -> None:
        self.space = space
        self.integrator = integrator
        self.rho = rho
        self.material_model = material_model
        self.boundary = boundary or MatrixFreeDirichletBC(dirichlet_dofs)

    @property
    def shape(self) -> tuple[int, int]:
        gdof = self.space.number_of_global_dofs()
        return gdof, gdof

    def update_density(self, rho: TensorLike) -> None:
        self.rho = rho

    def gather(self, x: TensorLike) -> TensorLike:
        return x[self.space.cell_to_dof()]

    def scatter_add(self, ye: TensorLike) -> TensorLike:
        cell2dof = self.space.cell_to_dof()
        y = bm.zeros((self.space.number_of_global_dofs(),),
                     dtype=ye.dtype,
                     device=self.space.device)
        return bm.add_at(y, bm.reshape(cell2dof, (-1,)), bm.reshape(ye, (-1,)))

    def matvec(self, x: TensorLike) -> TensorLike:
        x_bc = self.boundary.apply_input(x)
        xe = self.gather(x_bc)
        ye = self.integrator.action(
            self.space,
            xe,
            coef=self.rho,
            material_model=self.material_model,
        )
        y = self.scatter_add(ye)
        return self.boundary.apply_output(y, x)

    def __matmul__(self, x: TensorLike) -> TensorLike:
        return self.matvec(x)
