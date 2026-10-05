"""源项线性型 ``(f, v)`` 的单元积分子."""

from typing import Optional

from soptx.typing import TensorLike, Index, _S, CoefLike
from soptx.backend import backend_manager as bm
from soptx.mesh import HomogeneousMesh
from soptx.functionspace.space import FunctionSpace as _FS
from soptx.fem.coef import process_coef_func
from soptx.fem.functional import linear_integral
from soptx.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache

class SourceIntegrator(LinearInt, SrcInt, CellInt):
    """单元上的源项线性型 ``(f, v)_K`` 积分子.

    Parameters
    ----------
    source : CoefLike, optional
        源项, 经 ``process_coef_func`` 求值. 默认 None (视为 1).
    q : int, optional
        积分阶, 默认 ``space.p + 3``.
    index : Index, optional
        参与积分的单元子集, 默认全体.
    batched : bool, optional
        源项是否带批量维, 透传给 ``linear_integral``. 默认 False.
    """

    def __init__(self, source: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """``index`` 单元到全局自由度的映射, 形状 ``(NC, ldof)``."""
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        """取求积数据与基函数值, 结果按空间缓存.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间, 须定义在齐次网格上.

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
            raise RuntimeError("The SourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)

        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        """计算单元载荷向量.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间.

        Returns
        -------
        TensorLike
            ``linear_integral`` 的结果, 首两轴为 ``(NC, ldof)``, ``NC`` 为 ``index``
            选中的单元数; ``batched`` 为 True 时首轴为批量维.
        """
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
 
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return linear_integral(phi, ws, cm, val, batched=self.batched)
