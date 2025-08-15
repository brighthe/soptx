from typing import Optional
from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.functionspace import FunctionSpace
from fealpy.functionspace.functional import symmetry_index
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, 
                q: Optional[int] = None, 
                lambda0: float = 1.0, 
                lambda1: float = 1.0
            ) -> None:
        """
        Parameters:
        -----------
        q : 空间次数
        lambda0: 第一个系数, 对于各向同性材料为 1/(2μ)
        lambda1: 第二个系数, 对于各向同性材料为 λ/(2μ(dλ+2μ))
        """
        super().__init__()
        self.q = q
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        c2d0  = space.cell_to_dof()
        return c2d0

    @enable_cache
    def fetch(self, space: FunctionSpace):
        p = space.p
        q = self.q if self.q else p+3

        mesh = space.mesh
        TD = mesh.top_dimension()
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        # 2D - (NC, NQ, ldof, 3), 最后一个轴表示 2D 应力对称张量的独立分量数 (σ_xx, σ_yy, σ_xy)
        phi = space.basis(bcs)

        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        if TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

        return cm, phi, trphi, ws 

    def assembly(self, space: FunctionSpace) -> TensorLike:
        mesh = space.mesh 
        TD = mesh.top_dimension()
        lambda0, lambda1 = self.lambda0, self.lambda1 
        cm, phi, trphi, ws = self.fetch(space) 

        # 获取对称张量的权重系数
        # 2D: num = [1, 2, 1]
        # 3D: num = [1, 1, 1, 2, 2, 2]
        _, num = symmetry_index(d=TD, r=2)

        # TODO 慢
        A  = lambda0*bm.einsum('q, c, cqld, cqmd, d -> clm', ws, cm, phi, phi, num)
        A -= lambda1*bm.einsum('q, c, cql, cqm -> clm', ws, cm, trphi, trphi)
        
        return A




