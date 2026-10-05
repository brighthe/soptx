
"""质量型双线性型 ``(c u, v)`` 的单元积分子."""

from typing import Optional

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike, Index, _S, CoefLike

from soptx.mesh import HomogeneousMesh
from soptx.functionspace.space import FunctionSpace as _FS
from soptx.fem.coef import process_coef_func
from soptx.fem.functional import bilinear_integral
from soptx.fem.integrator import (
                                LinearInt, OpInt, CellInt,
                                enable_cache
                            )

class MassIntegrator(LinearInt, OpInt, CellInt):
    """单元上的质量型双线性型 ``(c u, v)_K`` 积分子.

    Parameters
    ----------
    coef : CoefLike, optional
        系数, 经 ``process_coef_func`` 求值. 默认 None (视为 1).
    q : int, optional
        积分阶, 默认 ``space.p + 3``.
    index : Index, optional
        参与积分的单元子集, 默认全体.
    """

    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, 
                index: Index=_S) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """单元到全局自由度的映射 ``space.cell_to_dof()``.

        Notes
        -----
        返回全体单元的映射, 不按 ``index`` 截取.
        """
        return space.cell_to_dof()

    @enable_cache
    def fetch(self, space: _FS):
        """取求积数据与基函数值, 结果按空间缓存.

        Parameters
        ----------
        space : FunctionSpace
            函数空间, 须定义在齐次网格上.

        Returns
        -------
        bcs : TensorLike
            积分点的重心坐标.
        ws : TensorLike
            积分权重, 形状 ``(NQ, )``.
        phi : TensorLike
            ``index`` 单元上的基函数值 ``space.basis(bcs, index=index)``.
        cm : TensorLike
            ``index`` 单元的测度.
        index : Index
            参与积分的单元子集.

        Raises
        ------
        RuntimeError
            网格不是 ``HomogeneousMesh``.
        """
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The MassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        
        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        """计算单元质量矩阵.

        Parameters
        ----------
        space : FunctionSpace
            试验与检验共用的函数空间.

        Returns
        -------
        TensorLike
            形状 ``(NC, ldof, ldof)`` 的局部矩阵, ``NC`` 为 ``index`` 选中的单元数,
            由 ``bilinear_integral`` 计算.
        """
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)

        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(phi, phi, ws, cm, val)


