# -*- coding: utf-8 -*-
"""无装配层级 (unassembled, UA / NONE).

什么都不常驻: 既没有全局矩阵, 没有单元矩阵, 也没有积分点数据. 每次作用算子时从
网格与材料现算几何因子与积分权重, 走完与 PA 相同的数据流后即丢弃

    y = G^T B^T D B G x

与 MFEM 的 ``AssemblyLevel::NONE`` 对应.

相对 PA 的取舍
--------------
存储
    PA 每单元常驻 O(NQ * GD^2) 的几何因子加 NQ 个标量, UA 一个都不存. 常驻量降到与
    网格规模无关的 O(1) (参考梯度那一份也是现算的), 只剩 cell2dof -- 见下面关于
    cell2dof 的说明.
运算
    每次 matvec 多付一次 NC * NQ 个 GD x GD 矩阵求逆加一次行列式. 这是 UA 唯一的
    代价来源, 也是它在本仓库里只用于取证、不用于生产的原因.
更新
    随设计变量更新时不需要重算任何常驻数据, 因为没有常驻数据; 系数直接留在本对象
    上, 下次作用时现场喂给 ``QFunction``.

关于 cell2dof
-------------
``ElementRestriction`` 仍然常驻. cell2dof 是函数空间的自由度编号而不是算子的常驻
数据, 每次 apply 重建它只是把索引数组重算一遍, 不产生任何信息; MFEM 的
``AssemblyLevel::NONE`` 同样依赖 ``FiniteElementSpace`` 已有的 restriction. 层级分
类法里 UA 那一栏说的 "只留几何与材料", 指的是不留算子的数值数据, 不是连网格拓扑
也要重建.

正确性判据
----------
UA 与 PA 逐位相同, 不只是吻合到舍入: 两者的几何量都出自
``_quadrature.quadrature_geometry``, 之后走的是同一个 ``DofToQuad`` 与同一个
``LinearElasticQFunction``, 浮点运算次序字面上一致, 区别只在算的时机. 因此验收取
``==`` 而不是相对误差阈值 -- 出现任何非零差异都说明实现走岔了, 而不是精度问题.
"""

from typing import Optional

from fealpy.typing import TensorLike

from soptx.fem.kernels import DofToQuad, ElementRestriction, LinearElasticQFunction

from ._quadrature import quadrature_geometry
from .base import AssemblyLevelExtension
from .registry import register_level


@register_level('ua')
class UnassembledAssembly(AssemblyLevelExtension):
    """UA 层级: 不常驻任何算子数据, 每次作用现算几何与材料.

    Parameters
    ----------
    space : 该双线性型所在的张量函数空间.
    integrator : ``LinearElasticIntegrator``, 每次作用时从它重取材料与积分阶.
    restriction : 单元限制算子 G, 其 ``cell2dof`` 必须与积分点数据的单元顺序一致.
    coef : 相对密度系数, 可为 None, (NC, ) 或 (NC, NQ).

    Notes
    -----
    ``integrator`` 是被持有的引用而不是拷贝. 它本身只带材料对象与积分阶这些元数据,
    不带 (NC, ...) 量级的数组, 因此不构成常驻开销; 但这也意味着外部改了积分子的材料,
    下一次作用就会跟着变 -- 这与 UA "什么都不缓存" 的语义一致, 不是缺陷.
    """

    level = 'ua'

    def __init__(self,
                space,
                integrator,
                restriction: ElementRestriction,
                coef: Optional[TensorLike] = None,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._integrator = integrator
        self._restriction = restriction
        self._coef = coef

    @classmethod
    def build(cls, space, integrator, pattern=None, **kwargs) -> "UnassembledAssembly":
        """只记下空间、积分子与自由度编号, 不算任何几何量.

        Parameters
        ----------
        space : 该双线性型所在的张量函数空间.
        integrator : ``LinearElasticIntegrator``, 提供材料, 积分阶与单元子集.
        pattern : 仅为与 FA 的构造签名对齐而接受, UA 没有 CSR 骨架.

        Notes
        -----
        与其余三个层级不同, 这里的 ``build`` 不做任何数值工作, 因此构造耗时与网格
        规模无关. 层级之间比装配耗时时这一格恒为近零, 代价全部转移到 matvec 上.
        """
        # 类型校验借 quadrature_geometry 那一份, 这里先手工挡一道, 免得错误的积分子
        # 一直拖到第一次 matvec 才暴露
        from soptx.fem.integrators import LinearElasticIntegrator

        if not isinstance(integrator, LinearElasticIntegrator):
            raise TypeError(
                "UA 层级目前只支持 LinearElasticIntegrator, 得到 "
                f"{type(integrator).__name__}"
            )

        # cell2dof 取自积分子而不是空间: 积分子可能带 index 子集, 两者不一定相同
        restriction = ElementRestriction(
                            cell2dof=integrator.to_global_dof(space),
                            global_dofs=space.number_of_global_dofs()
                        )

        return cls(space=space,
                integrator=integrator,
                restriction=restriction,
                coef=integrator.coef)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def integrator(self):
        """每次作用时重取材料与积分阶的积分子"""
        return self._integrator

    @property
    def coef(self) -> Optional[TensorLike]:
        """当前的相对密度系数"""
        return self._coef

    def _kernels(self):
        """现算 B 与 D.

        Returns
        -------
        dof_to_quad : 本次作用用的 B 算子.
        qfunction : 本次作用用的逐点算子 D.

        Notes
        -----
        与 ``PartialAssembly.build`` 里构造这两者的代码逐行相同, 差别只在它在 build
        里调一次, 这里每次作用调一次.
        """
        space = self.spaces[0]
        geometry = quadrature_geometry(space, self._integrator)

        dof_to_quad = DofToQuad(grad_ref=geometry.grad_ref,
                            jacobi_inverse=geometry.jacobi_inverse,
                            dof_permutation=geometry.dof_permutation)

        qfunction = LinearElasticQFunction(
                            elastic_matrix=self._integrator.material.elastic_matrix()[0, 0],
                            weighted_measure=geometry.weighted_measure,
                            coef=self._coef)

        return dof_to_quad, qfunction

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T B^T D B G x, B 与 D 现算现丢.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, B) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        dof_to_quad, qfunction = self._kernels()

        x_E = self._restriction.gather(x)
        grad_u = dof_to_quad.gradient(x_E)
        y_E = dof_to_quad.gradient_transpose(qfunction(grad_u))

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角, 走与 PA 相同的闭式.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        dof_to_quad, qfunction = self._kernels()

        basis_gradients = dof_to_quad.basis_gradients()
        diag_canonical = qfunction.diagonal(basis_gradients)
        diag_e = dof_to_quad.from_canonical(diag_canonical)

        return self._restriction.scatter_add(diag_e)

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新相对密度系数.

        没有常驻数据可改, 只把系数换掉, 下次作用时现场生效.
        """
        self._coef = coef

    def persistent_bytes(self) -> int:
        """常驻内存字节数: 只有 cell2dof.

        几何因子与逐点标量因子都不常驻, 这正是 UA 相对 PA 的全部差别.
        """
        return self._restriction.persistent_bytes()
