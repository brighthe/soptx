# -*- coding: utf-8 -*-
"""无装配层级 (unassembled, UA / NONE).

什么都不常驻: 既没有全局矩阵, 没有单元矩阵, 也没有积分点数据. 每次作用算子时从
网格与材料现算几何因子与积分权重, 走完与 PA 相同的数据流后即丢弃

    y = G^T B^T D B G x

与 MFEM 的 ``AssemblyLevel::NONE`` 对应.

相对 PA 的取舍
--------------
存储
    PA 每单元常驻 O(NQ * GD^2) 的几何因子加 NQ 个标量, UA 一个都不存. 本对象常驻的只有
    cell2dof (见下面关于 cell2dof 的说明) 与 build 阶段的参考基, 两者都不带 NC, 于是常驻
    量与网格规模无关. 参考基计在本层级名下而不是摘出去, 是为了让 PA 与 UA 的常驻量之差
    恰好等于 setup 段的那两项 (J^{-1} 与逐点标量), 差额里不掺 build 的项.
运算
    每次 matvec 多付一次 NC * NQ 个 GD x GD 矩阵求逆加一次行列式. 这是 UA 唯一的
    代价来源, 也是它在本仓库里只用于取证、不用于生产的原因. 参考基不在这条路径上: 它在
    ``build`` 里算定并持有, 每次作用原样传给 PA, 因此 UA 与 PA 的 matvec 差额是干净的
    setup 差额. 这一条由参数传递保证, 不依赖 ``ReferenceBasis.build`` 的缓存是否命中.
更新
    随设计变量更新时不需要重算任何常驻数据, 因为没有常驻数据; 系数直接留在本对象
    上, 下次作用时现场喂给 ``LinearElasticQFunction``.

关于 cell2dof
-------------
``ElementRestriction`` 仍然常驻. cell2dof 是函数空间的自由度编号而不是算子的常驻
数据, 每次 apply 重建它只是把索引数组重算一遍, 不产生任何信息; MFEM 的
``AssemblyLevel::NONE`` 同样依赖 ``FiniteElementSpace`` 已有的 restriction. 层级分
类法里 UA 那一栏说的 "只留几何与材料", 指的是不留算子的数值数据, 不是连网格拓扑
也要重建.

正确性判据
----------
UA 与 PA 逐位相同, 不只是吻合到舍入: 本层级每次作用现调一次
``PartialAssembly.build`` 造出一个 PA 算子, 作用完即弃 -- 内核构造与数据流走的都是
PA 的那一份代码, 两个层级唯一的差别是内核常驻还是现造. 因此验收取 ``==`` 而不是相对
误差阈值, 且这条判据只测 "常驻" 这一个变量: 本层级不另写一遍数据流, 也就没有第二条
可能与 PA 岔开的路径.
"""

from typing import Optional

from fealpy.typing import TensorLike

from soptx.fem.kernels import ElementRestriction, ReferenceBasis

from .base import AssemblyLevelExtension
from .partial import PartialAssembly
from .registry import register_level


@register_level('ua')
class UnassembledAssembly(AssemblyLevelExtension):
    """UA 层级: 不常驻任何算子数据, 每次作用现算几何与材料.

    Parameters
    ----------
    space : 该双线性型所在的张量函数空间.
    integrator : ``LinearElasticIntegrator``, 每次作用时从它重取材料与积分阶.
    restriction : 单元限制算子 G, 其 ``cell2dof`` 必须与积分点数据的单元顺序一致,
        且为 (NC, ldof, GD) 的分量布局.
    reference_basis : build 阶段的产物, 即参考单元上的基函数. 构造时 build 一次并持
        有, 每次作用原样交给 PA, 因此本层级每次作用重算的只有 setup 段.
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
                reference_basis: ReferenceBasis,
                coef: Optional[TensorLike] = None,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._integrator = integrator
        self._restriction = restriction
        self._reference_basis = reference_basis
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
        与其余三个层级不同, 这里的 ``build`` 不碰任何带 NC 的量: 只取一遍自由度编号,
        再走一次 build 阶段的参考基构造 (形状 (NQ, ldof, TD), 且通常命中
        ``ReferenceBasis.build`` 的缓存). 因此构造耗时与网格规模无关, 层级之间比装配耗时时
        这一格恒为近零, 代价全部转移到 matvec 上.

        参考基放在这里造而不是留给每次作用, 正是 build 与 setup 的分界: build 的产物
        与网格几何无关, 一次算定即可; UA "什么都不缓存" 指的是不缓存 setup 段的逐单元
        数据, 不包括 build 段.

        积分子的类型校验不在这里做: 本层级把构造整个委托给 ``PartialAssembly.build``,
        校验由那一份负责, 因此不支持的积分子在第一次作用时才报错. 这是 "只有一处判定"
        的代价, 换来的是两个层级不会各自维护一份会走岔的类型白名单.
        """
        restriction = ElementRestriction.from_integrator(integrator, space)

        scalar_space = space.scalar_space
        reference_basis = ReferenceBasis.build(
                            scalar_space=scalar_space,
                            q=integrator.quadrature_order(space))

        return cls(space=space,
                integrator=integrator,
                restriction=restriction,
                reference_basis=reference_basis,
                coef=integrator.coef)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def reference_basis(self) -> ReferenceBasis:
        """build 阶段的产物: 参考单元上的 B, 构造时算定, 不随作用次数重算"""
        return self._reference_basis

    @property
    def integrator(self):
        """每次作用时重取材料与积分阶的积分子"""
        return self._integrator

    @property
    def coef(self) -> Optional[TensorLike]:
        """当前的相对密度系数"""
        return self._coef

    def _operator(self) -> PartialAssembly:
        """现造一个 PA 算子, 用完即弃.

        Returns
        -------
        PartialAssembly
            用本层级当前系数构建的 PA 算子.

        Notes
        -----
        构造与作用整个委托给 PA 而不是复制它的代码, 因此两个层级走的是字面同一串浮点
        运算, 差别只在 PA 把 setup 段的结果算一次并常驻, 这里每次作用算一次即弃.
        build 段的参考基按 ``reference_basis`` 传进去, PA 那边据此跳过 build 段, 故这里
        重算的只有逐单元几何与逐点标量.

        ``build`` 用的是积分子上的系数, 而本层级的系数可能已被 ``update`` 换过, 故随后
        补一次 ``update``. 这不破坏逐位相等: ``LinearElasticQFunction.update`` 是拿存下的
        weighted_measure 重算逐点标量, 与 ``__init__`` 里走的是同一句.
        """
        operator = PartialAssembly.build(self.spaces[0], self._integrator,
                                reference_basis=self._reference_basis)
        operator.update(self._coef)

        return operator

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T B^T D B G x, 内核现造现丢.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, NB) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        return self._operator() @ x

    def diagonal(self) -> TensorLike:
        """取算子对角, 走与 PA 完全相同的那一份闭式.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.
        """
        return self._operator().diagonal()

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新相对密度系数.

        没有常驻数据可改, 只把系数换掉, 下次作用时现场生效.
        """
        self._coef = coef

    def persistent_bytes(self) -> int:
        """常驻内存字节数: cell2dof 与 build 阶段的参考基, 两项都不带 NC.

        Returns
        -------
        total : 常驻内存字节数.

        Notes
        -----
        几何因子与逐点标量因子都不常驻, 这正是 UA 相对 PA 的全部差别: 与 PA 的
        ``persistent_bytes`` 逐项相减, 差额恰好是 ``GeometricFactors`` 与 ``LinearElasticQFunction``
        两项, 即 setup 段的常驻量. 参考基在两边都计一次, 相减即抵消; 把它从 UA 这边摘掉
        反而会让差额多出一项与层级无关的常数.

        两个层级共用同一份参考基实例 (UA 每次作用把它传给 PA), 因此这里按值计入是口径
        上的记法, 不是真的各占一份内存.
        """
        return (self._restriction.persistent_bytes()
                + self._reference_basis.persistent_bytes())
