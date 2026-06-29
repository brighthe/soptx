from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


class MatrixFreeDirichletBC:
    """Vector-level homogeneous Dirichlet handling for matrix-free matvec."""

    def __init__(self, fixed_dofs: Optional[TensorLike] = None) -> None:
        self.fixed_dofs = fixed_dofs

    def apply_input(self, x: TensorLike) -> TensorLike:
        if self.fixed_dofs is None:
            return x

        y = bm.copy(x)
        return bm.set_at(y, self.fixed_dofs, 0.0)

    def apply_output(self, y: TensorLike, x: Optional[TensorLike] = None) -> TensorLike:
        if self.fixed_dofs is None:
            return y

        z = bm.copy(y)
        values = 0.0 if x is None else x[self.fixed_dofs]
        return bm.set_at(z, self.fixed_dofs, values)

    def apply_rhs(self, rhs: TensorLike) -> TensorLike:
        if self.fixed_dofs is None:
            return rhs

        f = bm.copy(rhs)
        return bm.set_at(f, self.fixed_dofs, 0.0)
