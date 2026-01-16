from typing import Optional, Union
from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.functionspace import FunctionSpace
from fealpy.decorator.variantmethod import variantmethod
from fealpy.functionspace.functional import symmetry_index
from fealpy.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

from soptx.utils import timer

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, 
                lambda0: Union[float, TensorLike] = 1.0, 
                lambda1: Union[float, TensorLike] = 1.0,
                coef: Optional[TensorLike] = None,
                q: Optional[int] = None, 
                method: Optional[str] = None
            ) -> None:
        super().__init__()

        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.coef = coef
        self.q = q

        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        c2d0  = space.cell_to_dof()
        return c2d0

    @enable_cache
    def fetch(self, space: FunctionSpace):
        p = space.p
        q = self.q if self.q else p+3

        mesh = getattr(space, 'mesh', None)
        TD = mesh.top_dimension()
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        # (NC, NQ, LDOF, NS)
        phi = space.basis(bcs)

        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        elif TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

        return cm, phi, trphi, ws 

    @variantmethod('standard')
    def assembly(self, space: FunctionSpace, enable_timing: bool = False) -> TensorLike:
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

        if coef is None:
            # TODO for 循环把维度轴分离出来, 提升效率
            LDOF = phi.shape[2]
            A = bm.zeros((NC, LDOF, LDOF), dtype=phi.dtype)
            weighted_lambda0 = self.lambda0 * num

            for i in range(phi.shape[-1]): 
                phi_comp = phi[..., i]
                w = weighted_lambda0[i]
                part = bm.einsum('q, c, cql, cqm -> clm', ws, cm, phi_comp, phi_comp)
                A += w * part
            if enable_timing:
                t.send('Einsum 求和时间 1')
            part_tr = bm.einsum('q, c, cql, cqm -> clm', ws, cm, trphi, trphi)
            A -= self.lambda1 * part_tr
            if enable_timing:
                t.send('Einsum 求和时间 2')

            # A  = lambda0 * bm.einsum('q, c, cqld, cqmd, d -> clm', ws, cm, phi, phi, num)
            # if enable_timing:
            #     t.send('Einsum 求和时间 3')

            # A -= lambda1 * bm.einsum('q, c, cql, cqm -> clm', ws, cm, trphi, trphi)
            # if enable_timing:
            #     t.send('Einsum 求和时间 3')

        # 单元密度 (NC, )
        elif coef.shape ==(NC, ):
            # TODO for 循环把维度轴分离出来, 提升效率
            LDOF = phi.shape[2]
            A = bm.zeros((NC, LDOF, LDOF), dtype=phi.dtype)
            weighted_lambda0 = self.lambda0 * num

            for i in range(phi.shape[-1]): 
                phi_comp = phi[..., i]
                w = weighted_lambda0[i]
                part = bm.einsum('q, c, c, cql, cqm -> clm', ws, cm, coef, phi_comp, phi_comp)
                A += w * part

            if enable_timing:
                t.send('Einsum 求和时间 1')

            part_tr = bm.einsum('q, c, c, cql, cqm -> clm', ws, cm, coef, trphi, trphi)
            A -= self.lambda1 * part_tr
            
            if enable_timing:
                t.send('Einsum 求和时间 2')
            
            # A  = lambda0 * bm.einsum('q, c, c, cqld, cqmd, d -> clm', ws, cm, coef, phi, phi, num)
            # if enable_timing:
            #     t.send('Einsum 求和时间 3')
            
            # A -= lambda1 * bm.einsum('q, c, c, cql, cqm -> clm', ws, cm, coef, trphi, trphi)
            # if enable_timing:
            #     t.send('Einsum 求和时间 4')

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
    
    @enable_cache
    def fetch_fast(self, space: FunctionSpace) -> TensorLike:
        p = space.p
        q = self.q if self.q else p+3

        mesh = getattr(space, 'mesh', None)
        TD = mesh.top_dimension()
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs) # (NC, NQ, LDOF, NS)

        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        elif TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

        NC = mesh.number_of_cells()
        LDOF = phi.shape[2]

        # --- 计算 M0 (剪切/自项部分) ---
        M0 = bm.zeros((NC, LDOF, LDOF), dtype=phi.dtype)
        
        _, num = symmetry_index(d=TD, r=2)

        for i in range(phi.shape[-1]): 
            phi_comp = phi[..., i]
            weight = num[i] # 纯几何权重 (1 or 2)
            
            # 纯几何积分，不乘 lambda
            part = bm.einsum('q, c, cql, cqm -> clm', ws, cm, phi_comp, phi_comp)
            M0 += weight * part

        # --- 计算 M1 (体积/耦合部分) ---
        M1 = bm.einsum('q, c, cql, cqm -> clm', ws, cm, trphi, trphi)

        return M0, M1

    @assembly.register('fast')
    def assembly(self, 
                space: FunctionSpace, 
                enable_timing: bool = False
            ) -> TensorLike:
        t = None
        if enable_timing:
            t = timer(f"应力项组装 (Fast Cached)")
            next(t)

        M0, M1 = self.fetch_fast(space)
        # A0 = self.fetch_fast(space)

        if enable_timing:
            t.send("获取 A0 (Cache Hit/Miss)")

        # 2. 获取材料系数场
        lam0 = self.lambda0
        lam1 = self.lambda1
        # coef = self.coef

        # 3. 组装 (支持广播)
        # Case A: 标量 (均匀材料)
        if isinstance(lam0, float) and isinstance(lam1, float):
            A = lam0 * M0 - lam1 * M1

        # Case B: 数组 (拓扑优化)
        else:
            # 确保维度匹配 (NC, ) -> (NC, 1, 1)
            if hasattr(lam0, 'ndim') and lam0.ndim == 1:
                lam0 = lam0[:, None, None]
            if hasattr(lam1, 'ndim') and lam1.ndim == 1:
                lam1 = lam1[:, None, None]
            
            # 线性组合
            A = lam0 * M0 - lam1 * M1

        return A 






