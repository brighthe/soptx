# -*- coding: utf-8 -*-
"""单元装配层级 (element assembly, EA).

常驻单元矩阵 {K_e}, 不求和成全局稀疏矩阵. 作用一次算子的代价是
gather -> 逐单元小矩阵乘 -> scatter-add, 即

    y = G^T K_e G x

与 MFEM 的 ``EABilinearFormExtension`` 对应.

这是 EA 的一般形式, 对系数形状没有要求. 单元密度 (NC, ) 下 K_e = s_e K_e^0, 另存
K_e 是冗余的, 分析器改用常驻 {K_e^0} 与 {s_e} 的 ``SharedReferenceElementAssembly``
(N_k = NC), 见 ``shared_reference``.
"""

from typing import Optional

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike

from soptx.fem.kernels import ElementRestriction

from .base import AssemblyLevelExtension
from .registry import register_level


@register_level('ea')
class ElementAssembly(AssemblyLevelExtension):
    """EA 层级: 常驻 {K_e}, 作用时走 gather-单元作用-scatter-add.

    Parameters
    ----------
    space : 该双线性型所在的函数空间.
    restriction : 扁平布局的单元限制算子 G, 其 ``cell2dof`` 必须与
        ``element_matrices`` 的单元顺序及单元内自由度顺序一致.
    element_matrices : (NC, ldof * GD, ldof * GD) 的单元矩阵, ldof 为标量空间的单元
        自由度数.
    integrator : 算出单元矩阵的积分子; 可为 None. ``update`` 靠它按新系数重新积分.

    Notes
    -----
    单列右端项下 ``__matmul__`` 的 einsum 下标与 FEALPy ``BilinearForm.__matmul__``
    在单积分子情形下完全相同, 因此结果逐位一致. 这条等价性依赖 "只有一个积分子":
    多积分子时 FEALPy 逐组散加, 浮点求和次序与本实现不同.

    多列右端项下两者不再对应: 本实现按 (gdof, NB) 布局, FEALPy 按 (NB, gdof), 理由见
    ``ElementRestriction``.
    """

    level = 'ea'

    def __init__(self,
                space,
                restriction: ElementRestriction,
                element_matrices: TensorLike,
                *,
                integrator=None,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._element_matrices = element_matrices
        self._integrator = integrator

    @classmethod
    def build(cls, space, integrator, pattern=None, **kwargs) -> "ElementAssembly":
        """预先算出并缓存单元矩阵, 不求和成全局矩阵.

        Parameters
        ----------
        space : 该双线性型所在的函数空间.
        integrator : 积分子, 由其 ``assembly`` 一次性算出 {K_e}.
        pattern : 仅为与 FA 的构造签名对齐而接受, EA 没有 CSR 骨架; 可为 None.

        Returns
        -------
        ElementAssembly
            构建好的 EA 算子实例.

        Notes
        -----
        K_e 就是 ``integrator.assembly(space)``, 与 FA 阶段 1 逐位相同.
        """
        # 预先算出单元矩阵并常驻, 之后每次 matvec 不再重复积分
        element_matrices = integrator.assembly(space)

        # 扁平布局, 与 K_e 的行列同序
        restriction = ElementRestriction.from_integrator(integrator, space, layout='flat')

        return cls(space=space,
                restriction=restriction,
                element_matrices=element_matrices,
                integrator=integrator)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def element_matrices(self) -> TensorLike:
        """常驻的单元矩阵, 形状 (NC, ldof * GD, ldof * GD)"""
        return self._element_matrices

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T K_e G x.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, NB) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        x_E = self._restriction.gather(x)
        y_E = bm.einsum('cij, cj... -> ci...', self._element_matrices, x_E)

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角: 逐单元取小矩阵对角再按 cell2dof 散加.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        diag_e = bm.einsum('cii -> ci', self._element_matrices)

        return self._restriction.scatter_add(diag_e)

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新单元矩阵, 拓扑 (G) 不变.

        Parameters
        ----------
        coef : 相对刚度系数, 形状约定同 ``LinearElasticIntegrator.coef``, None 表示
            恒为 1. 一律按新系数重新积分.

        Raises
        ------
        RuntimeError
            构造时没有给出积分子.

        Notes
        -----
        新旧 K_e 在替换完成前同时常驻, 峰值与 setup 相当. 单元密度下只需换 s_e 的
        写法见 ``SharedReferenceElementAssembly``.
        """
        if self._integrator is None:
            raise RuntimeError("EA 层级没有积分子, 无法按新系数重新积分单元矩阵")
        self._integrator.coef = coef
        self._element_matrices = self._integrator.assembly(self.spaces[0])

    def set_element_matrices(self, element_matrices: TensorLike) -> None:
        """整块替换常驻的单元矩阵, 拓扑 (G) 不变.

        Parameters
        ----------
        element_matrices : 新的单元矩阵, 形状与原来相同.

        Notes
        -----
        只换引用, 不复制; 新 K_e 如何得到由调用方负责. 供走查与内存实测直接演示
        不同的更新写法, 优化主流程走 ``update``.
        """
        self._element_matrices = element_matrices

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 单元矩阵加 cell2dof.

        Returns
        -------
        total : 常驻内存字节数.
        """
        return int(self._element_matrices.nbytes) + self._restriction.persistent_bytes()
