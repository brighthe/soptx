"""Matrix-Free Elasticity Operators (operator.py).

Core 2D/3D Element Assembly (EA) matrix-free operator and OverlapOperator
for distributed parallel communication and reduction.
"""

from __future__ import annotations

from typing import Any, Protocol

from fealpy.backend import backend_manager as bm
from fealpy.distributed import EntityMPI
from fealpy.fem import BilinearForm
from fealpy.functionspace import TensorFunctionSpace
from fealpy.typing import TensorLike

from cases import ElasticityCase


class SupportsMatmul(Protocol):
    """Protocol for operators supporting matrix multiplication (@)."""

    def __matmul__(self, other: TensorLike) -> TensorLike: ...


class OverlapOperator:
    """Apply a local operator to equal-status overlapping DOF copies."""

    def __init__(self, local_operator: Any, dof_comm: EntityMPI) -> None:
        self.local_operator = local_operator
        self.dof_comm = dof_comm

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        references = self.dof_comm.refs(vector.shape[-1])
        consistent_vector = self.dof_comm.sync_add(vector) / references
        local_result = self.local_operator @ consistent_vector
        return self.dof_comm.sync_add(local_result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local_operator, name)


class ElasticityEAOperator:
    """Matrix-free 2D/3D elasticity operator.

    Encapsulates the element-free / matrix-free stiffness operator y = K @ x,
    where K = sum_e R_e^T K_e R_e without ever forming the global sparse matrix K.
    """

    def __init__(
        self,
        space: TensorFunctionSpace,
        case: ElasticityCase,
        degree: int = 1,
        dof_comm: EntityMPI | None = None,
    ) -> None:
        from utils.analyzer import (
            build_distributed_analyzer,
            build_serial_analyzer,
        )

        self.space = space
        self.case = case
        self.degree = degree
        self.dof_comm = dof_comm

        if dof_comm is not None:
            self.analyzer = build_distributed_analyzer(
                space, case, degree, "ea", dof_comm=dof_comm
            )
        else:
            self.analyzer = build_serial_analyzer(
                space, case, degree, "ea"
            )

        self._element_form: BilinearForm | None = None
        self._system_operator: Any = None
        self._load_vector: TensorLike | None = None
        self._prescribed: TensorLike | None = None
        self._boundary_dofs: TensorLike | None = None

    def assemble(self) -> tuple[Any, TensorLike]:
        """Assemble element stiffness form and body force vector, applying BCs."""
        raw_stiff = self.analyzer.assemble_stiff_matrix()
        raw_force = self.analyzer.assemble_body_force_vector()

        operator, load = self.analyzer.apply_bc(raw_stiff, raw_force)
        self._element_form = raw_stiff
        self._system_operator = operator
        self._load_vector = load
        self._prescribed = self.analyzer.prescribed_solution
        self._boundary_dofs = self.space.is_boundary_dof(
            threshold=self.case.problem.is_dirichlet_boundary(),
            method="interp",
        )
        return operator, load

    @property
    def system_operator(self) -> Any:
        if self._system_operator is None:
            self.assemble()
        return self._system_operator

    @property
    def load_vector(self) -> TensorLike:
        if self._load_vector is None:
            self.assemble()
        return self._load_vector

    @property
    def prescribed_solution(self) -> TensorLike:
        if self._prescribed is None:
            self.assemble()
        return self._prescribed

    @property
    def boundary_dofs(self) -> TensorLike:
        if self._boundary_dofs is None:
            self.assemble()
        return self._boundary_dofs

    def __matmul__(self, vector: TensorLike) -> TensorLike:
        """Matrix-free matrix-vector product y = K @ x."""
        return self.system_operator @ vector


def create_matrix_free_operator(
    space: TensorFunctionSpace,
    case: ElasticityCase,
    degree: int = 1,
    dof_comm: EntityMPI | None = None,
) -> ElasticityEAOperator:
    """Factory function to build a Matrix-Free Elasticity operator."""
    return ElasticityEAOperator(space, case, degree=degree, dof_comm=dof_comm)
