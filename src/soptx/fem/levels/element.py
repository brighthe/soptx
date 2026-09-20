# -*- coding: utf-8 -*-
"""单元装配层级 (element assembly, EA).

常驻单元矩阵 {K_e}, 不求和成全局稀疏矩阵. 作用一次算子的代价是
gather -> 逐单元小矩阵乘 -> scatter-add, 即

    y = G^T K_e G x

与 MFEM 的 ``EABilinearFormExtension`` 对应.
"""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.fem.kernels import ElementRestriction

from .base import AssemblyLevelExtension
from .registry import register_level


@register_level('ea')
class ElementAssembly(AssemblyLevelExtension):
    """EA 层级: 常驻 {K_e}, 作用时走 gather-单元作用-scatter-add.

    Parameters
    ----------
    space : 该双线性型所在的函数空间.
    restriction : 单元限制算子 G, 其 ``cell2dof`` 必须与 ``K_e`` 的单元顺序一致.
    K_e : (NC, ldof, ldof) 的单元矩阵.
    const_integrator : 算出 ``K_e`` 的 const 积分子, 可为 None. 单元矩阵与 cell2dof
        都在其中, 外部内存与耗时统计工具直接读它, 故随层级一起持有.

    Notes
    -----
    单列右端项下 ``__matmul__`` 的 einsum 下标与 FEALPy ``BilinearForm.__matmul__``
    在单积分子情形下完全相同, 因此结果逐位一致. 这条等价性依赖 "只有一个积分子":
    多积分子时 FEALPy 逐组散加, 浮点求和次序与本实现不同.

    多列右端项下两者不再对应: 本实现按 (gdof, B) 布局, FEALPy 按 (B, gdof), 理由见
    ``ElementRestriction``.
    """

    level = 'ea'

    def __init__(self,
                space,
                restriction: ElementRestriction,
                K_e: TensorLike,
                const_integrator=None,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._K_e = K_e
        self._const_integrator = const_integrator

    @classmethod
    def build(cls, space, integrator, pattern=None, **kwargs) -> "ElementAssembly":
        """预先算出并缓存单元矩阵, 不求和成全局矩阵.

        Parameters
        ----------
        space : 该双线性型所在的函数空间.
        integrator : 积分子, 由 ``const`` 一次性算出 {K_e}.
        pattern : 仅为与 FA 的构造签名对齐而接受, EA 没有 CSR 骨架.
        """
        # const 预先算出单元矩阵, 之后每次 matvec 不再重复积分
        const_integrator = integrator.const(space)

        # cell2dof 取自积分子而不是空间: 积分子可能带 _index 子集, 两者不一定相同
        restriction = ElementRestriction(
                            cell2dof=const_integrator.to_global_dof(space),
                            global_dofs=space.number_of_global_dofs()
                        )

        return cls(space=space,
                restriction=restriction,
                K_e=const_integrator.value,
                const_integrator=const_integrator)

    @property
    def const_integrator(self):
        """算出 {K_e} 的 const 积分子, 供外部工具复用"""
        return self._const_integrator

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def element_matrices(self) -> TensorLike:
        """常驻的单元矩阵, 形状 (NC, ldof, ldof)"""
        return self._K_e

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T K_e G x.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, B) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        x_E = self._restriction.gather(x)
        y_E = bm.einsum('cij, cj... -> ci...', self._K_e, x_E)

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角: 逐单元取小矩阵对角再按 cell2dof 散加.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        diag_e = bm.einsum('cii -> ci', self._K_e)

        return self._restriction.scatter_add(diag_e)

    def update(self, K_e: TensorLike) -> None:
        """替换常驻的单元矩阵, 拓扑 (G) 不变"""
        self._K_e = K_e

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 单元矩阵加 cell2dof"""
        return int(self._K_e.nbytes) + self._restriction.persistent_bytes()
