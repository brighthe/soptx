"""Hu-Zhang 混合有限元的应力 (柔度) 项积分子."""

from typing import Optional, Union
from soptx.backend import backend_manager as bm

from soptx.typing import TensorLike
from soptx.functionspace import FunctionSpace
from soptx.decorator.variantmethod import variantmethod
from soptx.functionspace.functional import symmetry_index
from soptx.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

from soptx.core import timer

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    """Hu-Zhang 应力空间上的柔度项 ``(A sigma, tau)_K`` 积分子.

    柔度算子取 ``A sigma = lambda0 sigma - lambda1 tr(sigma) I``. 基函数按对称张量的
    独立分量存储, 分量内积按 ``symmetry_index`` 给出的重数加权.

    Parameters
    ----------
    lambda0, lambda1 : float or TensorLike, optional
        柔度系数, 默认均为 1.0. ``'fast'`` 变体允许形状 ``(NC, )`` 的逐单元系数.
    coef : TensorLike, optional
        只由 ``'standard'`` 变体读取的密度系数, 形状 ``(NC, )`` 或 ``(NC, NQ)``.
    q : int, optional
        积分阶, 为 None (或 0) 时取 ``space.p + 3``.
    method : str, optional
        装配变体: ``'standard'`` (默认) 或 ``'fast'``; 未注册的键回落到默认变体.
    """

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
        """单元到全局自由度的映射 ``space.cell_to_dof()``."""
        c2d0  = space.cell_to_dof()
        return c2d0

    @enable_cache
    def fetch(self, space: FunctionSpace):
        """取 ``'standard'`` 变体所需的求积数据与基函数, 结果按空间缓存.

        Parameters
        ----------
        space : FunctionSpace
            Hu-Zhang 应力空间.

        Returns
        -------
        cm : TensorLike
            单元测度, 形状 ``(NC, )``.
        phi : TensorLike
            基函数值, 形状 ``(NC, NQ, ldof, NS)``, ``NS`` 为对称张量的独立分量数.
        trphi : TensorLike
            基函数的迹, 形状 ``(NC, NQ, ldof)``.
        ws : TensorLike
            积分权重, 形状 ``(NQ, )``.
        """
        p = space.p
        q = p+3 if self.q is None else self.q

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
        """``'standard'`` 变体: 逐次求积计算柔度项局部矩阵.

        Parameters
        ----------
        space : FunctionSpace
            Hu-Zhang 应力空间.
        enable_timing : bool, optional
            是否打印分段计时. 默认 False.

        Returns
        -------
        TensorLike
            形状 ``(NC, ldof, ldof)`` 的局部矩阵.

        Raises
        ------
        NotImplementedError
            ``coef`` 不是 None, 形状也不是 ``(NC, )`` 或 ``(NC, NQ)``.

        Notes
        -----
        ``coef`` 为 ``(NC, )`` 或 ``(NC, NQ)`` 时被积函数再乘以 ``coef``. 本变体中
        ``lambda0`` 与 ``lambda1`` 按标量使用.
        """
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
        """``'fast'`` 变体的缓存部分: 与材料系数无关的两块几何矩阵.

        Parameters
        ----------
        space : FunctionSpace
            Hu-Zhang 应力空间.

        Returns
        -------
        M0 : TensorLike
            按分量重数加权的 ``(phi, phi)``, 形状 ``(NC, ldof, ldof)``.
        M1 : TensorLike
            ``(tr phi, tr phi)``, 形状 ``(NC, ldof, ldof)``.
        """
        p = space.p
        q = p+3 if self.q is None else self.q

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
            
            # 纯几何积分, 不乘 lambda
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
        """``'fast'`` 变体: 由缓存的几何矩阵组合出 ``lambda0 M0 - lambda1 M1``.

        Parameters
        ----------
        space : FunctionSpace
            Hu-Zhang 应力空间.
        enable_timing : bool, optional
            是否打印分段计时. 默认 False.

        Returns
        -------
        TensorLike
            形状 ``(NC, ldof, ldof)`` 的局部矩阵.

        Notes
        -----
        ``lambda0`` 与 ``lambda1`` 为一维 ``(NC, )`` 张量时按单元广播. 本变体不读 ``coef``.
        """
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






