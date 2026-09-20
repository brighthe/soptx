# -*- coding: utf-8 -*-
"""单元限制算子 (element restriction).

本模块只实现矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 G 一项: L 向量 (rank 局部自由度向量, 含 ghost) 与 E 向量 (单元自由度向量)
之间的聚集与散加. 它只由单元-自由度映射 ``cell2dof`` 定义, 与所解的物理问题和所用
的装配层级都无关, 因此 EA / PA / UA 三个层级共用同一个实例.

与 libCEED 的 ``CeedElemRestriction`` 对应. P (L 向量与 T 向量之间的跨 rank 归约)
不在本模块内: 它由 ``soptx.fem.distributed`` 的通信层负责, 本模块的两个方法都是纯
本地操作, 不含任何通信.
"""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


class ElementRestriction:
    """单元限制算子 G: L 向量 <-> E 向量.

    Parameters
    ----------
    cell2dof : (NC, ldof) 的单元-自由度映射. 必须来自实际参与装配的积分子
        (``integrator.to_global_dof(space)``), 而不是 ``space.cell_to_dof()``:
        积分子可能带有 ``_index`` 子集, 两者不一定相同.
    global_dofs : L 向量长度, 即本 rank 的全局自由度数.

    Notes
    -----
    多列右端项按自由度维在前的 (gdof, B) 布局处理, 与 ``soptx.solvers`` 里迭代解法
    的规范布局一致 (``CG`` 在 ``batch_first=True`` 时先转成该布局再进迭代), 也与
    ``ConstrainedOperator`` 用布尔掩码取边界行的写法一致. FEALPy
    ``BilinearForm.__matmul__`` 在这里按 (B, gdof) 布局写, 散加却用了
    ``bm.index_add`` 的默认 ``axis=0``, 两处不自洽, 多列下必然越界; 本类不照抄.
    """

    def __init__(self, cell2dof: TensorLike, global_dofs: int) -> None:
        self._cell2dof = cell2dof
        self._global_dofs = int(global_dofs)

    @property
    def cell2dof(self) -> TensorLike:
        """单元-自由度映射, 形状 (NC, ldof)"""
        return self._cell2dof

    @property
    def n_cells(self) -> int:
        """单元数 NC"""
        return int(self._cell2dof.shape[0])

    @property
    def local_dofs(self) -> int:
        """单元内局部自由度数 ldof"""
        return int(self._cell2dof.shape[1])

    @property
    def global_dofs(self) -> int:
        """L 向量长度 (本 rank 全局自由度数)"""
        return self._global_dofs

    def gather(self, x_L: TensorLike) -> TensorLike:
        """L 向量到 E 向量: x_E = G x_L.

        Parameters
        ----------
        x_L : (gdof, ) 或 (gdof, B) 的自由度向量, 批量维在后.

        Returns
        -------
        x_E : (NC, ldof) 或 (NC, ldof, B) 的单元自由度向量.

        Notes
        -----
        (NC, ldof) 的整数下标作用在第 0 维上, 尾部的批量维原样带过来, 单列与多列
        因此走同一条语句, 不分支.
        """
        return x_L[self._cell2dof]

    def scatter_add(self, y_E: TensorLike) -> TensorLike:
        """E 向量到 L 向量: y_L = G^T y_E, 重复自由度上累加.

        Parameters
        ----------
        y_E : (NC, ldof) 或 (NC, ldof, B) 的单元自由度向量, 批量维在后.

        Returns
        -------
        y_L : (gdof, ) 或 (gdof, B) 的自由度向量.

        Notes
        -----
        结果是未归约的 L 向量: 落在 rank 交界上的自由度只含本 rank 的贡献, 跨 rank
        求和由调用方 (P) 完成. 这与右端项的处理是同一个约定.

        批量维在后, 散加的目标轴恒为第 0 维, 正是 ``bm.index_add`` 的默认 ``axis``,
        单列与多列因此共用一条语句.
        """
        batch_shape = tuple(y_E.shape[2:])

        y_L = bm.zeros((self._global_dofs, ) + batch_shape, **bm.context(y_E))

        return bm.index_add(y_L,
                            bm.reshape(self._cell2dof, (-1, )),
                            bm.reshape(y_E, (-1, ) + batch_shape))

    def persistent_bytes(self) -> int:
        """常驻内存字节数, 只有 cell2dof 一项"""
        return int(self._cell2dof.nbytes)

    def __repr__(self) -> str:
        return (f"ElementRestriction(n_cells={self.n_cells}, "
                f"local_dofs={self.local_dofs}, global_dofs={self.global_dofs})")
