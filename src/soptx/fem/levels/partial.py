# -*- coding: utf-8 -*-
"""部分装配层级 (partial assembly, PA).

不常驻单元矩阵, 只常驻积分点上的几何与材料数据. 作用一次算子的代价是

    y = G^T B^T D B G x

其中基函数算子 B 由 ``ReferenceBasis`` (参考单元上的基函数) 与 ``GeometricFactors``
(每单元 J^{-1}) 合给 (收缩见 ``kernels.gradients``; B 不是应变位移矩阵, 记号约定见
``soptx.fem.kernels``), D 由 ``weighted_stress`` 给出 (逐点的本构与积分权重, 数组由
``LinearElasticQFunction`` 持有). 与 MFEM 的 ``PABilinearFormExtension`` 对应, 参考量
与逐单元几何量的分置也与 MFEM 一致.

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
    PA 只重算每积分点一个标量. 单元密度 (NC, ) 下 EA 由实体单元矩阵逐单元缩放即可,
    不必重新积分; 逐点密度 (NC, NQ) 下 EA 要重新积分全部单元矩阵, PA 的代价不变.

本层级目前只认线弹性: ``build`` 从 ``LinearElasticIntegrator`` 取材料与积分阶, 逐点
算子写死为 ``weighted_stress`` 与 ``weighted_stress_diagonal``, 所需数组由
``LinearElasticQFunction`` 持有. 换方程时 ``build`` 与这两处调用一起换.

UA 层级把构造与作用整个委托给本层级: 它每次作用现调一次 ``build``, 用完即弃. 于是两
个层级从内核构造到数据流走的都是字面同一串浮点运算, 唯一的差别是内核常驻还是现造 --
"UA 与 PA 逐位相同" 这条判据因此只测这一个变量, 不掺第二条可能分岔的路径.

与三段划分的术语对应
--------------------
矩阵自由的通行划分把构造期分成两段: build 只依赖单元形状, 阶次与积分阶, 产出参考单元
上的 B, 不随装配层级变化; setup 依赖网格几何, 产出逐单元的 J^{-1} 与逐点权重. 本仓库
里只有 PA 与 UA 走 B, FA 与 EA 走显式装配的那条路, 因此这份 B 的共享面是 PA 与 UA
两级. libCEED
的 ``CeedBasis`` 与 MFEM 的 ``FiniteElementCollection`` 靠构造时不吃网格把 build 在类型
上独立出来, FEALPy 的 ``LagrangeFESpace`` 绑网格, 拿不到那个保证.

soptx 把两段分在两处: build 段是 ``ReferenceBasis.build``, 它只吃 (标量空间, q), 按
(单元形状, p, q) 缓存在进程级; setup 段是本类 ``build`` 方法的后半段. 之所以还叫一个
``build`` 方法, 是因为四个层级的构造入口要与 ``create_level`` 的签名一致 -- 不传
``reference_basis`` 时它顺次做完两段, 传了就只做 setup 段. 想按阶段分开调的调用方 (UA,
走查脚本, 多重网格) 走后一支:

    ref_basis = ReferenceBasis.build(space.scalar_space, q)                  # build
    pa = PartialAssembly.build(space, integrator, reference_basis=ref_basis)  # setup

张量空间的自由度排序不进 build 段: 它由 ``ElementRestriction.from_integrator`` 在
setup 段换成 (NC, ldof, GD) 的分量布局, 参考基只认标量基函数.

因此 "阶段分离" 由参数传递保证, 不依赖缓存是否命中; 缓存只是让不分开调的那一支也不重算.
读计时与内存口径时按这两段切分, 不按方法名.
"""

from typing import Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import SimplexMesh
from fealpy.typing import TensorLike, _S

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import (
    ElementRestriction,
    GeometricFactors,
    LinearElasticQFunction,
    ReferenceBasis,
    physical_basis_gradients,
    physical_gradient,
    physical_gradient_transpose,
    weighted_stress,
    weighted_stress_diagonal,
)

from .base import AssemblyLevelExtension
from .registry import register_level


def quadrature_geometry(ctx) -> Tuple[TensorLike, TensorLike]:
    """由积分上下文求 PA setup 段的两份逐单元几何量.

    Jacobi 矩阵只算一次, 同时交给几何因子与积分权重两处用.

    Parameters
    ----------
    ctx : ``LinearElasticIntegrator.fetch_context`` 返回的积分上下文, 提供网格,
        单元子集, 积分点, 积分权重与单元测度.

    Returns
    -------
    jacobi_inverse : (NC, NQ, TD, GD) 的 Jacobi 矩阵逆, 交给 ``GeometricFactors``.
    weighted_measure : (NC, NQ) 的积分权重乘 |J|, 交给 ``LinearElasticQFunction``.

    Notes
    -----
    积分权重按网格类型分两支, 与 ``LinearElasticIntegrator.assembly`` 完全一致:
    单纯形网格的重心坐标求积权重之和为 1 而不是参考单元的测度, 故那一支用单元测度
    cm 而非 |J|. 若改用 |J|, 三角形会差 2 倍, 四面体差 6 倍.
    """
    mesh, bcs, ws = ctx.mesh, ctx.bcs, ctx.ws

    # entity_view 的 index 用 None 表示全体
    geo_index = None if ctx.index is _S else ctx.index
    jacobi = mesh.entity_view('cell').jacobi_matrix(bcs, index=geo_index)

    # inv 后下标变成 (NC, NQ, TD, GD), 即 d xi_r / d x_b
    jacobi_inverse = bm.linalg.inv(jacobi)

    if isinstance(mesh, SimplexMesh):
        weighted_measure = ws[None, :] * ctx.cell_measure[:, None]
    else:
        weighted_measure = ws[None, :] * bm.abs(bm.linalg.det(jacobi))

    return jacobi_inverse, weighted_measure


@register_level('pa')
class PartialAssembly(AssemblyLevelExtension):
    """PA 层级: 常驻积分点数据, 作用时走 gather-B-D-B^T-scatter-add.

    Parameters
    ----------
    space : 该双线性型所在的张量函数空间.
    restriction : 单元限制算子 G, 其 ``cell2dof`` 必须与积分点数据的单元顺序一致,
        且为 (NC, ldof, GD) 的分量布局.
    reference_basis : 参考单元上的基函数, 即基函数算子 B 的参考部分, 常驻量与网格规模
        无关.
    geometric_factors : 基函数算子 B 的逐单元几何部分, 即 J^{-1}.
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
                reference_basis: ReferenceBasis,
                geometric_factors: GeometricFactors,
                qfunction: LinearElasticQFunction,
            ) -> None:
        gdof = restriction.global_dofs
        super().__init__(spaces=(space, ), shape=(gdof, gdof))
        self._restriction = restriction
        self._reference_basis = reference_basis
        self._geometric_factors = geometric_factors
        self._qfunction = qfunction

    @classmethod
    def build(cls,
                space,
                integrator,
                pattern=None,
                *,
                reference_basis: Optional[ReferenceBasis] = None,
                **kwargs,
            ) -> "PartialAssembly":
        """算出并缓存积分点上的几何与材料数据, 不组装任何单元矩阵.

        Parameters
        ----------
        space : 该双线性型所在的张量函数空间.
        integrator : ``LinearElasticIntegrator``, 提供材料, 积分阶与单元子集.
        pattern : 仅为与 FA 的构造签名对齐而接受, PA 没有 CSR 骨架, 默认为 None.
        reference_basis : 已经 build 好的参考单元基函数. 传入则跳过 build 段, 只做
            setup; 留空则由本方法自行 build. ``create_level`` 走的是留空那一支.

        Returns
        -------
        PartialAssembly
            构建好的 PA 算子实例.

        Raises
        ------
        TypeError
            积分子不是 ``LinearElasticIntegrator``.
        NotImplementedError
            拓扑维数与几何维数不等 (嵌入流形网格).

        Notes
        -----
        方法体分 build 与 setup 两段, 由注释标出: build 段只依赖单元形状, p 与 q;
        setup 段依赖网格几何, 每张网格算一次. 两段能合在一个方法里, 是因为
        ``create_level`` 要求四个层级共用一个构造入口签名; 要把它们分开调, 传
        ``reference_basis`` 即可, 那时本方法只剩 setup 段.

        UA 层级每次作用都调一次本方法再把算子丢掉, 因此两个层级走的是字面同一串浮点
        运算, "UA 与 PA 逐位相同" 是结构上的保证而不是抄写的巧合. UA 在自己的 ``build``
        里把参考基 build 一次并持有, 每次作用把它传进来, 因此那条路径上重算的只有 setup
        段的逐单元几何 -- 这正是 UA 相对 PA 的那一个变量, 参考基不混在里面. 这一条由参数
        传递保证, 不依赖 ``ReferenceBasis.build`` 的缓存是否命中; 缓存只是加速.

        积分公式与单元测度取自 ``integrator.fetch_context``, 与显式装配的各条
        ``fetch_*`` 路径同出一处, 因此积分阶不会两边各自演化.

        setup 段的逐单元几何由 ``quadrature_geometry`` 给出, 手工拼装 PA 的调用方 (如
        走查脚本) 调的是同一个函数, 两条路径共用一份实现.
        """
        if not isinstance(integrator, LinearElasticIntegrator):
            raise TypeError(
                "PA / UA 层级目前只支持 LinearElasticIntegrator, 得到 "
                f"{type(integrator).__name__}"
            )

        ctx = integrator.fetch_context(space)
        scalar_space, mesh = ctx.scalar_space, ctx.mesh

        geo_dimension = int(mesh.geo_dimension())
        top_dimension = int(mesh.top_dimension())
        if geo_dimension != top_dimension:
            raise NotImplementedError(
                f"该层级要求 Jacobi 矩阵可逆, 但拓扑维数 {top_dimension} 与几何维数 "
                f"{geo_dimension} 不等 (嵌入流形网格). 这类网格上 B 要走伪逆, 与本层级"
                "的存储假设不同, 留到需要时单独实现"
            )

        # ---- build: 只依赖单元形状, p 与 q, 与网格几何无关 ----
        # 调用方已经 build 过就直接用那一份, 本方法退化成纯 setup; 否则就地 build,
        # 由 ReferenceBasis.build 按上述键缓存, 换一张同类型同阶次的网格也不重算
        if reference_basis is None:
            reference_basis = ReferenceBasis.build(scalar_space=scalar_space, q=ctx.q)

        # ---- setup: 依赖网格几何与当前设计变量, 每张网格算一次 ----
        jacobi_inverse, weighted_measure = quadrature_geometry(ctx)

        geometric_factors = GeometricFactors(jacobi_inverse=jacobi_inverse)

        qfunction = LinearElasticQFunction(
                            elastic_matrix=integrator.material.elastic_matrix()[0, 0],
                            weighted_measure=weighted_measure,
                            coef=integrator.coef)

        # 分量布局的 G: 张量空间的自由度排序在这里一次换好, B 只认 (NC, ldof, GD)
        restriction = ElementRestriction.from_integrator(integrator, space)

        return cls(space=space,
                restriction=restriction,
                reference_basis=reference_basis,
                geometric_factors=geometric_factors,
                qfunction=qfunction)

    @property
    def restriction(self) -> ElementRestriction:
        """单元限制算子 G"""
        return self._restriction

    @property
    def reference_basis(self) -> ReferenceBasis:
        """基函数算子 B 的参考部分: 参考单元上的基函数"""
        return self._reference_basis

    @property
    def geometric_factors(self) -> GeometricFactors:
        """基函数算子 B 的逐单元几何部分: J^{-1}"""
        return self._geometric_factors

    @property
    def qfunction(self) -> LinearElasticQFunction:
        """逐点算子 D"""
        return self._qfunction

    def __matmul__(self, x: TensorLike) -> TensorLike:
        """算子作用 y = G^T B^T D B G x.

        Parameters
        ----------
        x : (gdof, ) 或 (gdof, NB) 的 L 向量, 多列时批量维在后.

        Returns
        -------
        y : 与 x 同形状的未归约 L 向量.
        """
        reference_grad = self._reference_basis.grad
        jacobi_inverse = self._geometric_factors.jacobi_inverse
        qfunction = self._qfunction

        x_E = self._restriction.gather(x)
        grad_u = physical_gradient(x_E, reference_grad=reference_grad,
                                jacobi_inverse=jacobi_inverse)
        s_Q = weighted_stress(grad_u, weighted_coef=qfunction.weighted_coef,
                            elastic_matrix=qfunction.elastic_matrix,
                            strain_map=qfunction.strain_map)
        y_E = physical_gradient_transpose(s_Q, reference_grad=reference_grad,
                                        jacobi_inverse=jacobi_inverse)

        return self._restriction.scatter_add(y_E)

    def diagonal(self) -> TensorLike:
        """取算子对角.

        Returns
        -------
        diag : (gdof, ) 的未归约 L 向量.

        Notes
        -----
        对角元不能由 ``__matmul__`` 作用单位向量得到 (那要 gdof 次作用), 这里走闭式:
        先把物理基函数梯度展开成 (NC, NQ, ldof, GD) 的临时量, 再由逐点算子按分量缩成
        单元对角. 展开的量与 EA 常驻的单元矩阵同阶但只活一次调用, 而取对角每次求解
        只做一遍, 不在 matvec 的热路径上.
        """
        basis_gradients = physical_basis_gradients(
                    reference_grad=self._reference_basis.grad,
                    jacobi_inverse=self._geometric_factors.jacobi_inverse)

        diag_E = weighted_stress_diagonal(
                    basis_gradients,
                    weighted_coef=self._qfunction.weighted_coef,
                    quadratic_form=self._qfunction.quadratic_form)

        # 单元对角已是 (NC, ldof, GD) 的分量布局, 与 G 同序, 直接散加
        return self._restriction.scatter_add(diag_E)

    def update(self, coef: Optional[TensorLike]) -> None:
        """随设计变量更新逐点系数.

        Parameters
        ----------
        coef : 逐点材料系数张量.

        Notes
        -----
        几何因子与参考梯度都不动, 只重算 (NC, NQ) 个标量.
        """
        self._qfunction.update(coef)

    def persistent_bytes(self) -> int:
        """常驻内存字节数.

        Returns
        -------
        total : 常驻内存字节总数.

        Notes
        -----
        四项分别来自 ``ReferenceBasis`` (参考梯度, 不带 NC),
        ``GeometricFactors`` (J^{-1}), ``LinearElasticQFunction`` (逐点标量因子) 与
        ``ElementRestriction`` (cell2dof). 后三项带 NC, 第一项不带 -- 要单独看
        某一项时直接调对应内核的 ``persistent_bytes``.
        """
        return (self._reference_basis.persistent_bytes()
                + self._geometric_factors.persistent_bytes()
                + self._qfunction.persistent_bytes()
                + self._restriction.persistent_bytes())
