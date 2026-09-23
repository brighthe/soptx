# -*- coding: utf-8 -*-
"""单元限制算子 (element restriction).

本模块只实现矩阵自由算子分解

    A = P^T G^T B^T D B G P

中的 G 一项: L 向量 (rank 局部自由度向量, 含 ghost) 与 E 向量 (单元自由度向量)
之间的聚集与散加. 它只由单元-自由度映射 ``cell2dof`` 定义, 与所解的物理问题无关.

E 向量的布局由 ``cell2dof`` 的形状决定, 两种各有其用户, 都由 ``from_integrator``
按 ``layout`` 参数构造:

- (NC, ldof): 扁平布局 (``layout='flat'``), 单元内自由度按张量空间的 ``dof_priority``
  排列, 与 FEALPy 积分子给出的单元矩阵同序. EA 用这一种, 以便直接乘常驻的 K_e.
- (NC, ldof, GD): 分量布局 (``layout='component'``, 默认), ``[c, i, d]`` 为标量基函数
  i 的第 d 个分量, 与空间的自由度排序无关. PA 与 UA 用这一种, 基函数算子 B 只认它.

自由度排序因此只在这里出现一次: 分量布局的 ``cell2dof`` 在构造时按 ``dof_priority``
一次重排好, 此后聚集用一次整数下标就直接交出规范布局, 不必在 B 里再重排一遍. 这与
MFEM 的 ``ElementRestriction`` 一致: 那里不论空间按 byNODES 还是 byVDIM 排序, E 向量
都是 (ldof, vdim, NE) 的固定布局.

与 libCEED 的 ``CeedElemRestriction`` 对应. P (L 向量与 T 向量之间的跨 rank 归约)
不在本模块内: 它由 ``soptx.fem.distributed`` 的通信层负责, 本模块的两个方法都是纯
本地操作, 不含任何通信.
"""

from fealpy.backend import backend_manager as bm
from fealpy.functionspace.utils import flatten_indices
from fealpy.typing import TensorLike


class ElementRestriction:
    """单元限制算子 G: L 向量 <-> E 向量.

    Parameters
    ----------
    cell2dof : (NC, ldof) 或 (NC, ldof, GD) 的单元-自由度映射, 形状即 E 向量的布局
        (见模块 docstring). 必须来自实际参与装配的积分子
        (``integrator.to_global_dof(space)``), 而不是 ``space.cell_to_dof()``:
        积分子可能带有 ``_index`` 子集, 两者不一定相同.
    global_dofs : L 向量长度, 即本 rank 的全局自由度数.

    Notes
    -----
    多列右端项按自由度维在前的 (gdof, NB) 布局处理, 与 ``soptx.solvers`` 里迭代解法
    的规范布局一致 (``CG`` 在 ``batch_first=True`` 时先转成该布局再进迭代), 也与
    ``ConstrainedOperator`` 用布尔掩码取边界行的写法一致. FEALPy
    ``BilinearForm.__matmul__`` 在这里按 (NB, gdof) 布局写, 散加却用了
    ``bm.index_add`` 的默认 ``axis=0``, 两处不自洽, 多列下必然越界; 本类不照抄.
    """

    def __init__(self, cell2dof: TensorLike, global_dofs: int) -> None:
        if cell2dof.ndim not in (2, 3):
            raise ValueError(
                "cell2dof 必须是 (NC, ldof) 或 (NC, ldof, GD) 的数组, 得到 "
                f"shape={tuple(cell2dof.shape)}"
            )

        self._cell2dof = cell2dof
        self._global_dofs = int(global_dofs)

    @classmethod
    def from_integrator(cls,
                        integrator,
                        space,
                        layout: str = 'component',
                    ) -> "ElementRestriction":
        """由积分子与张量函数空间构造 G.

        Parameters
        ----------
        integrator : 实际参与装配的积分子, 提供单元-自由度映射.
        space : 该双线性型所在的张量函数空间, 提供自由度排序与 L 向量长度.
        layout : E 向量布局, 取 ``'component'`` (默认) 或 ``'flat'``, 见模块 docstring.

        Returns
        -------
        ElementRestriction
            ``cell2dof`` 形状为 (NC, ldof, GD) (分量布局) 或 (NC, ldof * GD) (扁平
            布局) 的单元限制算子.

        Notes
        -----
        cell2dof 取自积分子而不是 ``space.cell_to_dof()``: 积分子可能带 ``_index``
        子集, 两者不一定相同. EA, PA 与 UA 都由本方法构造 G, 免得同一段取法在多处各
        写一遍而日后只改了其中一处. 两种布局取的是同一份 cell2dof, 只差最后是否做
        单元内置换.

        扁平布局原样使用积分子给出的 cell2dof. 分量布局的重排用 FEALPy 的
        ``flatten_indices``, 与 ``TensorFunctionSpace`` 生成张量自由度编号走的是同一
        套约定: ``perm[i, d]`` 是标量自由度 i 的第 d 个分量在扁平单元向量里的槽位, 于
        是 ``cell2dof[:, perm]`` 一步得到 (NC, ldof, GD).
        """
        if layout not in ('component', 'flat'):
            raise ValueError(f"layout 必须是 'component' 或 'flat', 得到 {layout!r}")

        cell2dof = integrator.to_global_dof(space)
        global_dofs = space.number_of_global_dofs()

        if layout == 'flat':
            return cls(cell2dof=cell2dof, global_dofs=global_dofs)

        local_dofs = int(space.scalar_space.number_of_local_dofs())
        n_components = int(space.dof_numel)
        perm = flatten_indices((local_dofs, n_components),
                            (1, 0) if space.dof_priority else (0, 1))

        return cls(cell2dof=cell2dof[:, perm], global_dofs=global_dofs)

    @property
    def cell2dof(self) -> TensorLike:
        """单元-自由度映射, 形状 (NC, ldof) 或 (NC, ldof, GD)"""
        return self._cell2dof

    @property
    def n_cells(self) -> int:
        """单元数 NC"""
        return int(self._cell2dof.shape[0])

    @property
    def local_dofs(self) -> int:
        """单元局部自由度数: 扁平布局为全部分量的总数, 分量布局为标量基函数个数"""
        return int(self._cell2dof.shape[1])

    @property
    def global_dofs(self) -> int:
        """L 向量长度 (本 rank 全局自由度数)"""
        return self._global_dofs

    def gather(self, x_L: TensorLike) -> TensorLike:
        """L 向量到 E 向量: x_E = G x_L.

        Parameters
        ----------
        x_L : (gdof, ) 或 (gdof, NB) 的自由度向量, 批量维在后.

        Returns
        -------
        x_E : 形状为 ``cell2dof.shape`` 再接上 x_L 的批量维, 即 (NC, ldof[, GD][, NB]).

        Notes
        -----
        整数下标作用在第 0 维上, 尾部的批量维原样带过来, 单列与多列因此走同一条语
        句, 不分支; 扁平与分量两种布局同理.
        """
        return x_L[self._cell2dof]

    def scatter_add(self, y_E: TensorLike) -> TensorLike:
        """E 向量到 L 向量: y_L = G^T y_E, 重复自由度上累加.

        Parameters
        ----------
        y_E : 布局与 ``gather`` 的输出相同的单元自由度向量, 批量维在后.

        Returns
        -------
        y_L : (gdof, ) 或 (gdof, NB) 的自由度向量.

        Notes
        -----
        结果是未归约的 L 向量: 落在 rank 交界上的自由度只含本 rank 的贡献, 跨 rank
        求和由调用方 (P) 完成. 这与右端项的处理是同一个约定.

        批量维在后, 散加的目标轴恒为第 0 维, 正是 ``bm.index_add`` 的默认 ``axis``,
        单列与多列因此共用一条语句.
        """
        batch_shape = tuple(y_E.shape[self._cell2dof.ndim:])

        y_L = bm.zeros((self._global_dofs, ) + batch_shape, **bm.context(y_E))

        return bm.index_add(y_L,
                            bm.reshape(self._cell2dof, (-1, )),
                            bm.reshape(y_E, (-1, ) + batch_shape))

    def persistent_bytes(self) -> int:
        """常驻内存字节数, 只有 cell2dof 一项.

        Returns
        -------
        total : 常驻内存字节数.
        """
        return int(self._cell2dof.nbytes)

    def __repr__(self) -> str:
        return (f"ElementRestriction(n_cells={self.n_cells}, "
                f"local_dofs={self.local_dofs}, global_dofs={self.global_dofs})")
