# -*- coding: utf-8 -*-
"""参考单元上的基函数 (reference basis).

本模块持有矩阵自由算子分解

    A = P^T G^T B^T D B G P

中基函数算子 B 的参考部分: 参考单元上的基函数在积分点处的取值. 对位移元而言, B 交
出的是各积分点处的物理梯度 grad u, 其转置把与 grad u 共轭的量送回单元自由度; 这两个
作用写在 ``gradients`` 模块, 只吃数组 ``grad``, 本类只是持有者.

常驻的是参考单元上的基函数梯度加每单元的几何因子, 而不是每单元的物理梯度:

    grad_x phi = grad_xi phi . J^{-1}

参考梯度 (NQ, ldof, TD) 与单元无关, 全网格共享一份, 由本模块持有; 每单元的 J^{-1}
(NC, NQ, TD, GD) 由 ``GeometricFactors`` 持有. 于是每单元常驻量是 O(NQ * GD^2), 不带
ldof 维 -- 这正是 PA 相对 EA (每单元 O((GD*ldof)^2)) 的存储优势所在, 阶数越高差距越
大. 若改存每单元物理梯度 (NC, NQ, ldof, GD), 常驻量重新带上 ldof 维, 在任何阶数下都
不会比 EA 更省.

与 MFEM 的 ``DofToQuad`` 及 libCEED 的 ``CeedBasis`` 对应, 逐单元几何量同样另置一处
(见 ``geometric_factors``). 积分权重与 Jacobi 行列式也不在本类内: 它们是逐点算子 D
的一部分, 由 ``LinearElasticQFunction`` 持有. 张量空间的自由度排序 (``dof_priority``) 同样不在本
类内: 那是 L 向量的布局, 由 ``ElementRestriction`` 在聚集时换成 (ldof, GD) 的规范布
局, 本类只认标量基函数.

build 阶段
----------
本类持有的参考梯度只由单元形状, 阶次 p 与积分阶 q 决定, 与网格实例, 单元数以及节点
坐标都无关. 按矩阵自由的三段划分, 这正是 build 阶段的产物: 它在 setup (逐单元几何)
与 apply (每次 matvec) 之前一次算定, 装配层级不影响它, 换一张同类型同阶次的网格也不
必重算.

``build`` 因此按上述键把实例缓存在进程级的 ``_BUILD_CACHE`` 里: 多重网格的各层,
PA / UA 两个层级, 以及 UA 每次作用时现造的那个 PA 算子, 命中同一份实例. 本类是不可
变的 (数组构造后不再写), 共享实例没有别名风险.

libCEED 的 ``CeedBasis`` 与 MFEM 的 ``FiniteElementCollection`` 靠 "构造时不吃网格"
在类型上保证这件事. FEALPy 的 ``LagrangeFESpace`` 绑网格, 本类拿不到那个保证, 只能
靠上面这把键把它在运行期还原; 键漏一维就是静默取到错的基函数, 故键里宁可多带
backend / device / dtype 三项冗余.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


# build 阶段的进程级缓存: 键 -> ReferenceBasis. 见模块 docstring 的 "build 阶段" 一节.
_BUILD_CACHE: "OrderedDict[Tuple[Any, ...], 'ReferenceBasis']" = OrderedDict()
_BUILD_CACHE_LOCK = Lock()
_BUILD_CACHE_MAXSIZE = 32


class ReferenceBasis:
    """参考单元上的基函数在积分点处的取值, 全网格共享一份.

    目前只持有梯度 ``grad``; 需要基函数值 (如质量矩阵) 时在此添加 ``value``.

    Parameters
    ----------
    grad : (NQ, ldof, TD) 的参考单元基函数梯度, ``[q, i, r]`` 为 d phi_i / d xi_r
        在第 q 个积分点处的值.

    Notes
    -----
    本类不持有任何带 NC 维的数组, 因此常驻量与网格规模无关. 逐单元的 J^{-1} 由
    ``GeometricFactors`` 持有, 两者都是纯数据; 用到它们的收缩在 ``gradients`` 模块,
    由 ``PartialAssembly`` 取出 ``grad`` 与 ``jacobi_inverse`` 两个数组交给它.

    这里只有标量基函数: 位移的 GD 个分量共用同一组基函数, 分量下标 d 由单元自由度
    向量的 (NC, ldof, GD) 布局带着, 不进本类.
    """

    def __init__(self, grad: TensorLike) -> None:
        if grad.ndim != 3:
            raise ValueError(
                "grad 必须是 (NQ, ldof, TD) 的三维数组, 得到 "
                f"shape={tuple(grad.shape)}"
            )

        self._grad = grad
        self._n_quad = int(grad.shape[0])
        self._local_dofs = int(grad.shape[1])
        self._top_dim = int(grad.shape[2])

    @classmethod
    def build(cls, scalar_space, q: int) -> "ReferenceBasis":
        """build 阶段: 造出参考单元上的基函数, 并按 (单元形状, p, q) 缓存.

        Parameters
        ----------
        scalar_space : 标量拉格朗日空间, 提供阶次 p 与参考梯度; 只用到它的单元形状
            与阶次, 不用网格几何.
        q : 积分阶, 取自 ``integrator.quadrature_order(space)``.

        Returns
        -------
        ReferenceBasis
            参考单元上的基函数, 可能是缓存里已有的那一份.

        Notes
        -----
        代价与网格规模无关: 积分点取在参考单元上, 参考梯度形状为 (NQ, ldof, TD),
        两者都不带 NC. 因此本方法可以放在 UA 每次作用的路径上而不改变该层级
        "常驻与作用代价均与常驻数据无关" 的口径.

        ``grad_basis`` 不传 ``index``: 参考梯度与取哪些单元无关, 传子集与传全体
        逐位相同 (已在 tri / quad / tet / hex x p = 1, 2, 3 上逐位核对). 不传, 是
        为了让这条调用在形式上也不依赖单元子集, 免得日后有人把 ``index`` 当成键的
        一维而漏进缓存.
        """
        mesh = scalar_space.mesh
        p = int(scalar_space.p)
        q = int(q)

        # 参考单元上的求积公式, 与网格几何无关
        bcs, ws = mesh.quadrature_formula(q).get_quadrature_points_and_weights()

        # 单纯形的 bcs 是数组, 张量积单元的是数组元组, 两者都不可哈希, 故键里放
        # 决定它的 (单元类型, q) 而不是 bcs 本身
        key = (type(mesh), p, q,
                bm.backend_name, str(bm.get_device(ws)), str(ws.dtype))

        with _BUILD_CACHE_LOCK:
            hit = _BUILD_CACHE.get(key)
            if hit is not None:
                _BUILD_CACHE.move_to_end(key)

                return hit

        instance = cls(grad=scalar_space.grad_basis(bcs, variable='u'))

        with _BUILD_CACHE_LOCK:
            _BUILD_CACHE[key] = instance
            _BUILD_CACHE.move_to_end(key)
            if len(_BUILD_CACHE) > _BUILD_CACHE_MAXSIZE:
                _BUILD_CACHE.popitem(last=False)

        return instance

    @property
    def grad(self) -> TensorLike:
        """积分点处的参考基函数梯度, 形状 (NQ, ldof, TD)"""
        return self._grad

    @property
    def n_quad(self) -> int:
        """每单元积分点数 NQ"""
        return self._n_quad

    @property
    def local_dofs(self) -> int:
        """标量空间的单元局部自由度数 ldof"""
        return self._local_dofs

    @property
    def top_dimension(self) -> int:
        """拓扑维数 TD"""
        return self._top_dim

    def persistent_bytes(self) -> int:
        """常驻内存字节数, 与单元数无关.

        参考梯度是全网格共享的一份; 带 NC 维的 J^{-1} 记在
        ``GeometricFactors.persistent_bytes`` 名下.
        """
        return int(self._grad.nbytes)

    def __repr__(self) -> str:
        return (f"ReferenceBasis(n_quad={self._n_quad}, "
                f"local_dofs={self._local_dofs}, top_dim={self._top_dim})")


def clear_build_cache() -> None:
    """清空 build 阶段缓存.

    只为测试与内存诊断而设: 正常使用中缓存里只有几份 (NQ, ldof, TD) 的小数组, 不需要
    手动清理.
    """
    with _BUILD_CACHE_LOCK:
        _BUILD_CACHE.clear()


def build_cache_keys() -> Tuple[Tuple[Any, ...], ...]:
    """当前缓存里的键, 按插入顺序返回.

    Returns
    -------
    keys : 键的不可变快照, 供测试断言缓存是否命中.
    """
    with _BUILD_CACHE_LOCK:
        return tuple(_BUILD_CACHE)
