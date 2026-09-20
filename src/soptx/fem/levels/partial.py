# -*- coding: utf-8 -*-
"""部分装配层级 (partial assembly, PA).

不常驻单元矩阵, 只常驻积分点上的几何与材料数据. 作用一次算子的代价是

    y = G^T B^T D B G x

其中 B 由 ``DofToQuad`` 给出 (参考梯度加几何因子), D 由 ``QFunction`` 给出 (逐点的
本构与积分权重). 与 MFEM 的 ``PABilinearFormExtension`` 对应.

相对 EA 的取舍
--------------
存储
    EA 每单元存 (GD*ldof)^2 个数, PA 每单元只存 NQ 个几何因子矩阵加 NQ 个标量, 即
    O(NQ * GD^2), 与 ldof 无关. 默认 q = p + 3 下二维四边形 p >= 2 起 PA 更省, 三维
    六面体同样在 p = 2 处越过, 且差距随阶数迅速拉开.
运算
    EA 每单元一次 (GD*ldof)^2 的稠密矩阵向量乘; PA 是两次张量收缩加逐点作用, 单元
    上的浮点次数更多但访存更少, 高阶下总体更快. 本实现按稠密 einsum 写, 没有做
    张量积单元的和分解 (sum factorization), 因此这里只兑现存储优势.
更新
    随设计变量更新时 EA 要重算全部单元矩阵, PA 只重算每积分点一个标量.

本层级目前只认线弹性: ``build`` 从 ``LinearElasticIntegrator`` 取材料与积分阶. 换
方程时只需另写一个 ``QFunction`` 并在 ``build`` 里分派, ``__matmul__`` 与
``diagonal`` 不用动.
"""

from typing import Optional

from fealpy.typing import TensorLike

from soptx.fem.kernels import DofToQuad, ElementRestriction, LinearElasticQFunction

from ._quadrature import quadrature_geometry
from .base import AssemblyLevelExtension
from .registry import register_level


@register_level('pa')
class PartialAssembly(AssemblyLevelExtension):
    """PA 层级: 常驻积分点数据, 作用时走 gather-B-D-B^T-scatter-add.

    Parameters
    ----------
    space : 该双线性型所在的张量函数空间.
    restriction : 单元限制算子 G, 其 ``cell2dof`` 必须与积分点数据的单元顺序一致.
    dof_to_quad : B 算子.
    qfunction : 逐点算子 D.

    Notes
    -----
    ``__matmul__`` 与 EA 在数学上等价, 但浮点求和次序不同 (EA 先把 D 与两侧基函数
    缩成单元矩阵再作用, PA 逐积分点作用), 因此两者只到舍入误差一致, 不逐位相同.
    """

    level = 'pa'

    def __init__(self,
                space,
                restriction: ElementRestriction,
                dof_to_quad: DofToQuad,
                qfunction: LinearElasticQFunction,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._dof_to_quad = dof_to_quad
        self._qfunction = qfunction

    @classmethod
    def build(cls, space, integrator, pattern=None, **kwargs) -> "PartialAssembly":
        """算出并缓存积分点上的几何与材料数据, 不组装任何单元矩阵.

        Parameters
        ----------
        space : 该双线性型所在的张量函数空间.
        integrator : ``LinearElasticIntegrator``, 提供材料, 积分阶与单元子集.
        pattern : 仅为与 FA 的构造签名对齐而接受, PA 没有 CSR 骨架.

        Notes
        -----
        几何量与积分权重的构造下沉到 ``_quadrature.quadrature_geometry``,
        与 UA 层级共用: 两者喂给 B 和 D 的输入必须是字面上同一串运算, UA 对 PA 的
        逐位相等才成立.
        """
        geometry = quadrature_geometry(space, integrator)

        dof_to_quad = DofToQuad(grad_ref=geometry.grad_ref,
                            jacobi_inverse=geometry.jacobi_inverse,
                            dof_permutation=geometry.dof_permutation)

        qfunction = LinearElasticQFunction(
                            elastic_matrix=integrator.material.elastic_matrix()[0, 0],
                            weighted_measure=geometry.weighted_measure,
                            coef=integrator.coef)

        # cell2dof 取自积分子而不是空间: 积分子可能带 index 子集, 两者不一定相同
        restriction = ElementRestriction(
                            cell2dof=integrator.to_global_dof(space),
                            global_dofs=space.number_of_global_dofs()
                        )

        return cls(space=space,
                restriction=restriction,
                dof_to_quad=dof_to_quad,
                qfunction=qfunction)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def dof_to_quad(self) -> DofToQuad:
        """B 算子"""
        return self._dof_to_quad

    @property
    def qfunction(self) -> LinearElasticQFunction:
        """逐点算子 D"""
        return self._qfunction

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T B^T D B G x.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, B) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        x_E = self._restriction.gather(x)
        grad_u = self._dof_to_quad.gradient(x_E)
        y_E = self._dof_to_quad.gradient_transpose(self._qfunction(grad_u))

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角.

        对角元不能由 ``__matmul__`` 作用单位向量得到 (那要 gdof 次作用), 这里走闭式:
        先把物理基函数梯度展开成 (NC, NQ, ldof, GD) 的临时量, 再由逐点算子按分量缩成
        单元对角. 展开的量与 EA 常驻的单元矩阵同阶但只活一次调用, 而取对角每次求解
        只做一遍, 不在 matvec 的热路径上.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        basis_gradients = self._dof_to_quad.basis_gradients()
        diag_canonical = self._qfunction.diagonal(basis_gradients)
        diag_e = self._dof_to_quad.from_canonical(diag_canonical)

        return self._restriction.scatter_add(diag_e)

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新逐点系数.

        几何因子与参考梯度都不动, 只重算 (NC, NQ) 个标量.
        """
        self._qfunction.update(coef)

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 几何因子, 逐点标量因子, 加 cell2dof"""
        return (self._dof_to_quad.persistent_bytes()
                + self._qfunction.persistent_bytes()
                + self._restriction.persistent_bytes())
