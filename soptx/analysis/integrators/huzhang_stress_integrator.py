from typing import Optional
from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.functionspace import FunctionSpace
from fealpy.functionspace.functional import symmetry_index
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

from soptx.utils import timer

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, 
                lambda0: float = 1.0, 
                lambda1: float = 1.0,
                coef: Optional[TensorLike] = None,
                q: Optional[int] = None, 
            ) -> None:
        super().__init__()

        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.coef = coef
        self.q = q

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
        # (NC, NQ, LDOF, NS)
        phi = space.basis(bcs)

        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        if TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

        return cm, phi, trphi, ws 

    def assembly(self, space: FunctionSpace, enable_timing: bool = True) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"应力项组装")
            next(t)

        mesh = getattr(space, 'mesh', None)
        TD = mesh.top_dimension()
        NC = mesh.number_of_cells()

        lambda0, lambda1 = self.lambda0, self.lambda1
        
        cm, phi, trphi, ws = self.fetch(space)
        NQ = phi.shape[1] 

        # 获取对称张量的权重系数
        # 2D: num = [1, 2, 1]
        # 3D: num = [1, 1, 1, 2, 2, 2]
        _, num = symmetry_index(d=TD, r=2)

        if enable_timing:
            t.send('准备时间')

        coef = self.coef 

        # TODO phi 的每个轴的计算复杂度太大了
        if coef is None:
            A  = lambda0 * bm.einsum('q, c, cqld, cqmd, d -> clm', ws, cm, phi, phi, num)
            if enable_timing:
                t.send('Einsum 求和时间 1')

            A -= lambda1 * bm.einsum('q, c, cql, cqm -> clm', ws, cm, trphi, trphi)
            if enable_timing:
                t.send('Einsum 求和时间 2')

        # 单元密度 (NC, )
        elif coef.shape ==(NC, ):
            A  = lambda0 * bm.einsum('q, c, c, cqld, cqmd, d -> clm', ws, cm, coef, phi, phi, num)
            if enable_timing:
                t.send('Einsum 求和时间 1')
            
            A -= lambda1 * bm.einsum('q, c, c, cql, cqm -> clm', ws, cm, coef, trphi, trphi)
            if enable_timing:
                t.send('Einsum 求和时间 2')

        # 节点密度 (NC, NQ)
        elif coef.shape == (NC, NQ):
            A  = lambda0 * bm.einsum('q, c, cq, cqld, cqmd, d -> clm', ws, cm, coef, phi, phi, num)
            A -= lambda1 * bm.einsum('q, c, cq, cql, cqm -> clm', ws, cm, coef, trphi, trphi)

        else:
            raise NotImplementedError
        
        if enable_timing:
            t.send('Einsum 求和时间')
            t.send(None)

        return A




