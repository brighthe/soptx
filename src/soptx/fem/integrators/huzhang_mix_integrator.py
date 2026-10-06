"""Hu-Zhang 混合有限元的应力-位移耦合项积分子."""

from soptx.backend import backend_manager as bm

from soptx.typing import TensorLike
from soptx.functionspace import FunctionSpace
from soptx.fem.integrator import (LinearInt, OpInt, CellInt, enable_cache)

from soptx.core import timer

class HuZhangMixIntegrator(LinearInt, OpInt, CellInt):
    """Hu-Zhang 混合元的耦合项 ``(div tau, v)_K`` 积分子.

    ``space`` 为二元组: ``space[0]`` 为位移空间 (试验空间), ``space[1]`` 为 Hu-Zhang
    应力空间 (检验空间), 对应 ``BilinearForm((space_u, space_sigma))``, 装配出
    ``(gdof_sigma, gdof_u)`` 的矩形矩阵.

    Parameters
    ----------
    q : int, optional
        积分阶, 为 None (或 0) 时取 ``space[0].p + 3``.
    """

    def __init__(self, q = None) -> None:
        super().__init__()
        self.q = q

    @enable_cache
    def to_global_dof(self, space: FunctionSpace) -> TensorLike:
        """两个空间各自的单元到全局自由度映射.

        Parameters
        ----------
        space : tuple of FunctionSpace
            ``(space_u, space_sigma)``.

        Returns
        -------
        tuple of TensorLike
            ``(space[0].cell_to_dof(), space[1].cell_to_dof())``.
        """
        c2d0  = space[0].cell_to_dof() 
        c2d1  = space[1].cell_to_dof() 

        return (c2d0, c2d1)

    @enable_cache
    def fetch(self, space: FunctionSpace):
        """取求积数据、位移基函数与应力基函数的散度, 结果按空间缓存.

        Parameters
        ----------
        space : tuple of FunctionSpace
            ``(space_u, space_sigma)``.

        Returns
        -------
        cm : TensorLike
            单元测度, 形状 ``(NC, )``.
        ws : TensorLike
            积分权重, 形状 ``(NQ, )``.
        div_phi : TensorLike
            应力基函数的散度, 形状 ``(NC, NQ, ldof_sigma, GD)``.
        psi : TensorLike
            位移基函数值, 末两轴为 ``(ldof_u, GD)``.
        """
        space0 = space[0]
        space1 = space[1]

        p = space0.p
        q = p+3 if self.q is None else self.q
        mesh = space1.mesh
        qf = mesh.quadrature_formula(q, 'cell')
        cm = mesh.entity_measure('cell')

        bcs, ws = qf.get_quadrature_points_and_weights()

        psi = space0.basis(bcs)
        div_phi = space1.div_basis(bcs)

        return cm, ws, div_phi, psi
    
    @enable_cache
    def assembly(self, space: FunctionSpace, enable_timing: bool = False) -> TensorLike:
        """计算耦合项局部矩阵 ``int_K div(tau_l) . v_m dx``.

        Parameters
        ----------
        space : tuple of FunctionSpace
            ``(space_u, space_sigma)``, 两者须在同一网格上.
        enable_timing : bool, optional
            是否打印分段计时. 默认 False.

        Returns
        -------
        TensorLike
            形状 ``(NC, ldof_sigma, ldof_u)`` 的局部矩阵.
        """
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space "

        t = None
        if enable_timing:
            t = timer(f"mix assembly")
            next(t)

        cm, ws, div_phi, psi = self.fetch(space)

        if enable_timing:
            t.send('2')

        res = bm.einsum('q, c, cqld, cqmd -> clm', ws, cm, div_phi, psi)

        if enable_timing:
            t.send('3')
            t.send(None)

        return res



