# -*- coding: utf-8 -*-
"""单元装配层级 (element assembly, EA).

常驻单元矩阵 {K_e}, 不求和成全局稀疏矩阵. 作用一次算子的代价是
gather -> 逐单元小矩阵乘 -> scatter-add, 即

    y = G^T K_e G x

与 MFEM 的 ``EABilinearFormExtension`` 对应.
"""

from typing import Optional

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
    restriction : 扁平布局的单元限制算子 G, 其 ``cell2dof`` 必须与
        ``element_matrices`` 的单元顺序及单元内自由度顺序一致.
    element_matrices : (NC, ldof * GD, ldof * GD) 的单元矩阵, ldof 为标量空间的单元
        自由度数.
    reference_matrices : 基准单元矩阵 K_e^0, 即相对刚度系数全为 1 时的单元矩阵, 形状同
        ``element_matrices``; 可为 None. 给出时单元密度下的 ``update`` 由它逐单元缩放并
        原地写回, 不重新积分. 本类只持有引用, 不复制, 所有者是调用方.
    integrator : 算出单元矩阵的积分子; 可为 None. ``update`` 无法由 K_e^0 缩放时靠它
        重新积分.

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
                reference_matrices: Optional[TensorLike] = None,
                integrator=None,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._element_matrices = element_matrices
        self._reference_matrices = reference_matrices
        self._integrator = integrator

    @classmethod
    def build(cls,
            space,
            integrator,
            pattern=None,
            *,
            reference_matrices: Optional[TensorLike] = None,
            **kwargs,
        ) -> "ElementAssembly":
        """预先算出并缓存单元矩阵, 不求和成全局矩阵.

        Parameters
        ----------
        space : 该双线性型所在的函数空间.
        integrator : 积分子, 由其 ``assembly`` 一次性算出 {K_e}.
        pattern : 仅为与 FA 的构造签名对齐而接受, EA 没有 CSR 骨架; 可为 None.
        reference_matrices : 基准单元矩阵 K_e^0; 可为 None. 给出且积分子的系数为单元
            密度 (NC, ) 时, K_e 由它逐单元缩放得到, 不再积分.

        Returns
        -------
        ElementAssembly
            构建好的 EA 算子实例.

        Notes
        -----
        不给 ``reference_matrices`` 时 K_e 就是 ``integrator.assembly(space)``, 与
        FA 阶段 1 逐位相同. 给出时 K_e 与重新积分只差舍入, 因为 K_e^0 可能出自另一种
        assembly 方法.
        """
        coef = integrator.coef
        if _is_cell_scalar(coef, reference_matrices):
            # 单元密度: K_e = coef_e K_e^0, 只分配 K_e 这一份
            element_matrices = _scale_into(bm.empty_like(reference_matrices),
                                        reference_matrices, coef)
        else:
            # 预先算出单元矩阵并常驻, 之后每次 matvec 不再重复积分
            element_matrices = integrator.assembly(space)

        # 扁平布局, 与 K_e 的行列同序
        restriction = ElementRestriction.from_integrator(integrator, space, layout='flat')

        return cls(space=space,
                restriction=restriction,
                element_matrices=element_matrices,
                reference_matrices=reference_matrices,
                integrator=integrator)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def element_matrices(self) -> TensorLike:
        """常驻的单元矩阵, 形状 (NC, ldof * GD, ldof * GD)"""
        return self._element_matrices

    @property
    def reference_matrices(self) -> Optional[TensorLike]:
        """基准单元矩阵 K_e^0 的引用, 未给出时为 None"""
        return self._reference_matrices

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
        coef : 相对刚度系数, 形状约定同 ``LinearElasticIntegrator.coef``. None 表示
            恒为 1; (NC, ) 为单元密度; 其余形状 (逐点, 多分辨率, 逐单元本构) 均重新积分.

        Raises
        ------
        RuntimeError
            需要重新积分, 但构造时没有给出积分子.

        Notes
        -----
        有 K_e^0 且系数为 None 或 (NC, ) 时, 由 K_e^0 逐单元缩放后原地写回 K_e, 不分配
        新的单元矩阵; 后端不支持原地写 (如 jax) 时退化为新分配一份, 结果不变. 其余
        情形重新积分, 新旧 K_e 在替换完成前同时常驻, 峰值与 setup 相当.
        """
        reference = self._reference_matrices
        if reference is not None and coef is None:
            coef = bm.ones(reference.shape[0], dtype=reference.dtype,
                        device=bm.get_device(reference))

        if _is_cell_scalar(coef, reference):
            self._element_matrices = _scale_into(self._element_matrices, reference, coef)
            return

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

        Notes
        -----
        K_e^0 只是引用, 所有者 (如分析器的敏度缓存) 另行计数, 这里不计.
        """
        return int(self._element_matrices.nbytes) + self._restriction.persistent_bytes()


def _is_cell_scalar(coef: Optional[TensorLike],
                    reference: Optional[TensorLike]) -> bool:
    """判断能否由 K_e^0 逐单元缩放: 有 K_e^0 且系数为 (NC, ) 的单元密度"""
    if reference is None or coef is None:
        return False

    return tuple(coef.shape) == (reference.shape[0], )


def _scale_into(out: TensorLike, reference: TensorLike, coef: TensorLike) -> TensorLike:
    """计算 coef_e K_e^0 并原地写入 out.

    Parameters
    ----------
    out : 目标单元矩阵, 形状同 ``reference``.
    reference : 基准单元矩阵 K_e^0.
    coef : (NC, ) 的单元系数.

    Returns
    -------
    result : 写好的单元矩阵. 后端支持 ``out=`` 时就是 ``out`` 本身, 否则为新数组.
    """
    factor = coef[:, None, None]
    try:
        return bm.multiply(reference, factor, out=out)
    except TypeError:
        return bm.multiply(reference, factor)
