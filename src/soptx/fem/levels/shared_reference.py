# -*- coding: utf-8 -*-
"""共享参考单元矩阵的单元装配 (shared-reference element assembly).

标准 EA 为每个单元常驻一份 K_e. 单元密度下各单元的本构只差一个标量 s_e
(如 E(rho_e) / E_0), K_e = s_e K_e^0, 于是只需常驻参考单元矩阵与逐单元标量, 算子
作用写成

    y = sum_k sum_{e in C_k} s_e G_e^T K_k^0 G_e x

参考单元矩阵的份数 N_k 有两种取法:

- N_k = NC (逐单元参考): 每个单元自成一类, K_k^0 即 K_e^0, 对任意网格成立. 分析器
  在单元密度下的 'ea' 层级取这一种, K_e^0 与敏度共用同一份缓存.
- N_k < NC (共享参考): 在平移类结构化网格上, 同一平移类的单元只差平移, 共用一份
  K_k^0. 平移类取自 ``from_box`` 的单元编号约定 (见 ``soptx.mesh.structured_box``):
  每个格子剖出的 N_k 个单元连续编号, 故 k(e) = e mod N_k, 前 N_k 个单元即各类代表.
  类归属由编号现算, 不常驻.

本类不在层级注册表中, 不能经 ``create_level`` 选中: 它只接受单元密度, 且 N_k < NC 时
的平移类前提无法在运行时廉价校验. 分析器在单元密度下直接构造 N_k = NC 的实例, 其余
调用方在确认网格满足约定后显式构造.
"""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.fem.kernels import ElementRestriction

from .base import AssemblyLevelExtension


class SharedReferenceElementAssembly(AssemblyLevelExtension):
    """共享参考 EA: 常驻 {K_k^0} 与 {s_e}, 作用时走 gather-按类单元作用-缩放-scatter-add.

    Parameters
    ----------
    space : 该双线性型所在的函数空间.
    restriction : 扁平布局的单元限制算子 G, 覆盖全体单元; 其 ``cell2dof`` 的单元内
        自由度顺序必须与 ``reference_matrices`` 的行列顺序一致.
    reference_matrices : (N_k, ldof * GD, ldof * GD) 的参考单元矩阵, 第 k 个是平移类 k
        的代表单元 (即单元 k) 在 s_e = 1 时的单元矩阵. N_k 由其第一维给出; 取 N_k = NC
        时即逐单元的 K_e^0. 本类只持有引用, 不复制.
    scale : (NC, ) 的逐单元标量 s_e; 可为 None, 此时取全 1. 本类复制一份持有.

    Raises
    ------
    ValueError
        ``restriction`` 不是扁平布局, ``reference_matrices`` 形状与之不符, NC 不是
        N_k 的整数倍, 或 ``scale`` 形状不是 (NC, ).

    Notes
    -----
    N_k < NC 时正确性依赖单元编号约定: 单元 e 与单元 e mod N_k 只差平移, 且局部顶点
    顺序相同. ``from_box`` 生成、未经重编号的网格满足它; 其余网格本类照常计算但结果
    是错的, 构造时不做几何比对. N_k = NC 时 e mod N_k = e, 约定平凡成立.

    与标准 EA 的结果只差舍入, 不逐位相同: N_k < NC 时同类单元的节点坐标由 ``linspace``
    给出, 只在舍入意义下相等; N_k = NC 时 s_e 在单元作用之后才乘, 与积进 K_e 的次序不同.

    与标准 EA 共用 G 的 gather / scatter_add, 单元作用改为把 E 向量按 (格子, 类)
    重排后与 K_k^0 做批量小矩阵乘; 重排是视图, 不复制.
    """

    level = 'ea'

    def __init__(self,
                space,
                restriction: ElementRestriction,
                reference_matrices: TensorLike,
                *,
                scale: Optional[TensorLike] = None,
            ) -> None:
        if restriction.cell2dof.ndim != 2:
            raise ValueError("restriction 必须是扁平布局 (NC, ldof * GD), "
                             f"得到 cell2dof.shape={tuple(restriction.cell2dof.shape)}")

        n_cells = restriction.n_cells
        local_dofs = restriction.local_dofs
        if (reference_matrices.ndim != 3
                or tuple(reference_matrices.shape[1:]) != (local_dofs, local_dofs)):
            raise ValueError(f"reference_matrices 须为 (N_k, {local_dofs}, {local_dofs}), "
                             f"得到 {tuple(reference_matrices.shape)}")

        num_classes = int(reference_matrices.shape[0])
        if num_classes == 0 or n_cells % num_classes != 0:
            raise ValueError(f"单元数 {n_cells} 不是平移类数 {num_classes} 的整数倍")

        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._reference_matrices = reference_matrices
        self._scale = self._owned_scale(scale)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def reference_matrices(self) -> TensorLike:
        """常驻的参考单元矩阵 K_k^0, 形状 (N_k, ldof * GD, ldof * GD)"""
        return self._reference_matrices

    @property
    def scale(self) -> TensorLike:
        """常驻的逐单元标量 s_e, 形状 (NC, )"""
        return self._scale

    @property
    def num_classes(self) -> int:
        """平移类数 N_k"""
        return int(self._reference_matrices.shape[0])

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = sum_e s_e G_e^T K_{k(e)}^0 G_e x.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, NB) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        x_E = self._restriction.gather(x)                        # (NC, L[, NB])
        batch_shape = tuple(x_E.shape[2:])
        n_cells, local_dofs = x_E.shape[0], x_E.shape[1]
        num_classes = self.num_classes

        # 单元 e = g * N_k + k, 按 (格子, 类) 重排后第 1 维即 k(e)
        x_G = bm.reshape(x_E, (n_cells // num_classes, num_classes, local_dofs) + batch_shape)
        y_G = bm.einsum('kij, gkj... -> gki...', self._reference_matrices, x_G)
        y_E = bm.reshape(y_G, (n_cells, local_dofs) + batch_shape)
        y_E = y_E * bm.reshape(self._scale, (n_cells, ) + (1, ) * (1 + len(batch_shape)))

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角: 每类取 K_k^0 的对角, 乘 s_e 后按 cell2dof 散加.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        n_cells = self._restriction.n_cells
        num_classes = self.num_classes
        diag_k = bm.einsum('kii -> ki', self._reference_matrices)   # (N_k, L)

        scale_G = bm.reshape(self._scale, (n_cells // num_classes, num_classes, 1))
        diag_E = bm.reshape(scale_G * diag_k[None, :, :], (n_cells, -1))

        return self._restriction.scatter_add(diag_E)

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新逐单元标量 s_e, K_k^0 与拓扑 (G) 都不变.

        Parameters
        ----------
        coef : 相对刚度系数, None 表示恒为 1, 否则须为 (NC, ) 的单元系数.

        Raises
        ------
        ValueError
            ``coef`` 不是 None 或 (NC, ): 逐点或逐单元本构的系数会打破类内共享,
            需改用标准 EA.
        """
        self._scale = self._owned_scale(coef)

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 参考单元矩阵、逐单元标量与 cell2dof.

        Returns
        -------
        total : 常驻内存字节数.
        """
        return (int(self._reference_matrices.nbytes) + int(self._scale.nbytes)
                + self._restriction.persistent_bytes())

    def _owned_scale(self, coef: Optional[TensorLike]) -> TensorLike:
        """把 None 或 (NC, ) 的系数化成本类持有的 s_e 副本."""
        n_cells = self._restriction.n_cells
        reference = self._reference_matrices
        if coef is None:
            return bm.ones(n_cells, dtype=reference.dtype, device=bm.get_device(reference))
        if tuple(coef.shape) != (n_cells, ):
            raise ValueError(f"共享参考 EA 只接受 None 或 ({n_cells}, ) 的单元系数, "
                             f"得到 shape={tuple(coef.shape)}")

        return bm.copy(coef)
