# 移植自 brighthe/fealpy ``fealpy/mesh/schema/entity_schema.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""不可变网格实体 Schema 值的约定.

一个 :class:`EntitySchema` 值描述一个具体的参考实体: 稳定的身份、完整的局部节点布局、
允许的定向、几何插值, 以及可选的参考基函数能力. Schema 值不拥有网格连接、物理坐标
或有限元自由度.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import ClassVar, Concatenate, Literal, ParamSpec, TYPE_CHECKING, cast

from ...backend import Tensor, Index
from ..storage import EntityContext
from .descriptor import CanonicalValue, SchemaDescriptor

if TYPE_CHECKING:
    from ...quadrature import Quadrature
    from .local_entity import LocalEntityGroup

__all__ = ["EntitySchema"]

P = ParamSpec("P")


class EntitySchema:
    """一个不可变、参数化的参考实体约定.

    Schema 的 Python 类标识一个结构族, 每个实例标识该族中一个规范化的成员. 例如
    ``LagrangeTriangleSchema(p=1)`` 与 ``LagrangeTriangleSchema(p=2)`` 的 Python 类型相同,
    但描述符、ID、局部节点布局与几何基函数都不同.

    具体的 Schema 值不可变且按值比较. 调用方须以相等性、:attr:`descriptor` 或 :attr:`id`
    判断身份, 不得依赖 Python 对象身份. Schema 不含任何 ``MeshBlock`` 状态, 也不含与后端、
    设备或数据类型相关的张量缓存.
    """

    __slots__ = ()

    type_id: ClassVar[str]
    schema_version: ClassVar[int]
    descriptor_parameter_names: ClassVar[tuple[str, ...]]
    name: ClassVar[str]
    top_dim: ClassVar[int]
    OFace: ClassVar[Mapping[str, tuple[tuple[int, ...], ...]]] = MappingProxyType({})
    SFace: ClassVar[Mapping[str, tuple[tuple[int, ...], ...]]] = MappingProxyType({})
    orientation: ClassVar[tuple[tuple[int, ...], ...]] = ()
    ccw: ClassVar[tuple[int, ...] | None] = None
    ref_measure: ClassVar[float | None] = None

    @property
    def descriptor(self) -> SchemaDescriptor:
        """由规范化字段导出的规范描述符.

        参数名与顺序遵循所登记 Schema 类型的 ``descriptor_parameter_names``. 返回值不可变,
        足以经 ``SCHEMA_RESOLVER`` 重建出相等的 Schema.
        """
        parameters = tuple(
            (name, cast(CanonicalValue, getattr(self, name)))
            for name in self.descriptor_parameter_names
        )
        return SchemaDescriptor(
            type_id=self.type_id,
            schema_version=self.schema_version,
            parameters=parameters,
        )

    @property
    def id(self) -> str:
        """本具体 Schema 确定且可逆的 ID.

        ID 编码了 :attr:`descriptor`, 与模块路径、进程内登记顺序以及 Python 对象身份无关.
        """
        return self.descriptor.to_id()

    def number_of_vertices(self) -> int:
        """拓扑顶点骨架的大小.

        该数与插值次数无关, 对高阶几何 Schema 可小于 :meth:`number_of_nodes`.
        """
        raise NotImplementedError()

    def number_of_nodes(self) -> int:
        """本 Schema 完整局部节点布局的宽度.

        布局包含顶点及几何基函数用到的全部高阶插值节点, 即绑定到本具体 Schema 的实体分区
        所需的连接宽度.
        """
        raise NotImplementedError()

    def validate_connectivity_counts(self, counts: Tensor) -> None:
        """校验变长分区中各实体的连接数.

        定长 Schema 对现有的变长分区调用方保持过渡期的宽松行为; 变长 Schema 以其结构上的
        基数约定重写本钩子.
        """

    def local_vertices(self) -> tuple[int, ...]:
        """顶点在完整局部节点布局中的列位置.

        结果索引的是局部连接列, 不含网格块全局节点编号.
        """
        raise NotImplementedError()

    def local_entity_groups(self, top_dim: int) -> tuple[LocalEntityGroup, ...]:
        """描述某一拓扑维数的局部子实体.

        每个返回的组把一个不可变子 Schema 绑定到若干行父局部节点列下标上, 每行按子实体完整
        的规范节点顺序. 同一维数可有多个组, 如三棱柱的三角形面与四边形面.

        零维时各组覆盖父实体完整布局中的每个节点, 而不只是拓扑顶点; 只需顶点骨架时用
        :meth:`local_vertices`.

        Parameters
        ----------
        top_dim : int
            目标维数, 取值于闭区间 ``[0, self.top_dim]``.

        Returns
        -------
        tuple of LocalEntityGroup
            不可变的同类局部子实体组.

        Raises
        ------
        TypeError
            ``top_dim`` 不是普通整数.
        ValueError
            ``top_dim`` 超出支持的区间.
        """
        raise NotImplementedError()

    def vertex_permutations(self) -> tuple[tuple[int, ...], ...]:
        """本具体 Schema 支持的顶点自同构.

        这些是保持参数化节点集不变的参考实体对称, 并不表示任意顶点顺序都是合法的定向.
        """
        raise NotImplementedError()

    def node_permutation(
        self,
        vertex_permutation: tuple[int, ...],
    ) -> tuple[int, ...]:
        """把一个支持的顶点自同构提升到全部局部节点列.

        ``result[i]`` 为施加该顶点置换后局部节点 ``i`` 所在的列. 结果是
        ``range(number_of_nodes())`` 上的双射, 列约定与几何基函数及正次数 Lagrange 基函数接口
        相同.

        Raises
        ------
        TypeError
            置换不是由普通整数组成的元组.
        ValueError
            本具体 Schema 不支持该置换.
        """
        raise NotImplementedError()

    ### [实体拓扑] ###

    @classmethod
    def local_entity(
        cls,
        tgt_name: str,
        /,
        indexing: Literal["o", "s"] = "o",
    ) -> tuple[tuple[int, ...], ...]:
        """目标实体的局部编号.

        Parameters
        ----------
        tgt_name : str
            目标实体名.
        indexing : {'o', 's'}, optional
            编号方式, 默认 ``'o'``.

        Returns
        -------
        tuple of tuple of int
            不可变的局部实体编号.
        """
        raise NotImplementedError()

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """分区中的实体个数."""
        raise NotImplementedError()

    ### [多重指标] ###

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """实体的多重指标, 每个顶点一列."""
        raise NotImplementedError()

    @classmethod
    def multi_index_vertex_columns(cls) -> tuple[int, ...] | None:
        """把 ``multi_index`` 的列映射到局部顶点编号.

        ``multi_index(..., tensorprod=True)`` 每个顶点输出一列, 但张量积 Schema 按因子嵌套顺序
        编号这些列, 未必与 ``local_vertices()``、``_node_keys()`` 及局部实体表所用的顶点编号
        一致. 返回置换 ``columns``, 使 ``multi_index`` 的第 ``s`` 列是顶点 ``columns[s]`` 的重心
        权重; 两种约定一致时返回 None.
        """
        return None

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """多重指标的个数."""
        return int(cls.multi_index(order, internal=internal).shape[0])

    ### [几何计算] ###

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """计算实体的重心."""
        raise NotImplementedError()

    @classmethod
    def barycentric[**P, R](
        cls,
        ctx: EntityContext,
        func: Callable[Concatenate[Tensor, P], R],
        index: Index | None
    ) -> Callable[Concatenate[Tensor | tuple[Tensor, ...], P], R]:
        """把函数从直角坐标变换到重心坐标."""
        raise NotImplementedError()

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """把重心坐标转换为物理点."""
        raise NotImplementedError()

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """实体的几何维数."""
        raise NotImplementedError()

    def grad_shape_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """几何基函数对重心坐标变量的导数, 形状 ``(Q, Lg, B)``.

        ``Lg`` 按完整的几何局部节点顺序, ``B`` 按公开的参考因子顺序拼接各重心坐标分量;
        每个因子的重心坐标视为独立变量.

        Raises
        ------
        NotImplementedError
            本 Schema 没有独立重心坐标梯度的约定.
        """
        raise NotImplementedError()

    def grad_shape_function_cartesian(
        self,
        ctx: EntityContext,
        bcs: Tensor | tuple[Tensor, ...],
        *,
        index: Index | None = None
    ) -> Tensor:
        """几何基函数在物理坐标下的导数.

        参考实体到物理实体的映射由几何基函数与全部几何节点决定. 结果的局部基函数轴按完整
        的局部节点顺序.

        Raises
        ------
        NotImplementedError
            本 Schema 或当前嵌入不提供该物理坐标运算.
        """
        raise NotImplementedError()

    def grad_shape_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """几何基函数在参考坐标下的导数, 形状 ``(Q, Lg, R)``.

        ``Lg`` 为 :meth:`number_of_nodes`, ``R`` 为参考维数. 张量积实体的参考分量按具体
        Schema 所述的公开因子顺序.
        """
        raise NotImplementedError()

    def lagrange_basis_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """在本参考族上计算独立的 Lagrange 基函数.

        这是供有限元空间使用的参考实体能力, 不改变 Schema 的几何次数、描述符、身份或局部
        节点数. 标量单纯形族要求非负整数 ``p``; 张量积族接受对所有因子统一的非负整数, 或按
        公开因子顺序给出的元组.

        单纯形的 ``bcs`` 为一个 ``(Q, V)`` 重心坐标张量, 张量积为各因子一个张量的元组, 张量积
        的点按笛卡尔积求值. 结果形状为 ``(Q, Lp)``; 正次数时基函数轴按几何次数为 ``p`` 的同族
        成员的完整局部节点顺序. 输出保持输入的浮点类型与设备.

        Raises
        ------
        TypeError
            坐标容器或次数的类型不合法.
        ValueError
            坐标形状或次数取值不合法.
        NotImplementedError
            本 Schema 族不提供任意次 Lagrange 参考基函数.
        """
        raise NotImplementedError()

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """``p`` 次基函数对重心坐标变量的导数, 形状 ``(Q, Lp, B)``.

        基函数轴的顺序同 :meth:`lagrange_basis_function`; 重心坐标分量按公开的参考因子顺序
        拼接, 视为独立变量.
        """
        raise NotImplementedError()

    def grad_lagrange_basis_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """``p`` 次基函数在参考坐标下的导数, 形状 ``(Q, Lp, R)``.

        基函数轴的顺序同 :meth:`lagrange_basis_function`, 参考坐标轴按公开的因子顺序.
        """
        raise NotImplementedError()

    @classmethod
    def integral(
        cls,
        ctx: EntityContext,
        func: Callable[[Tensor], Tensor] | Callable[[tuple[Tensor, ...]], Tensor],
        q: int, index: Index | None
    ) -> Tensor:
        """重心坐标函数的积分."""
        raise NotImplementedError()

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """参考实体到物理实体变换的 Jacobi 矩阵.

        Parameters
        ----------
        ctx : EntityContext
            含网格块与实体分区的上下文.
        bcs : tuple of Tensor
            求值点的重心坐标, 形状 ``(NQ, num_bc)``.
        index : Index or None
            分区中的实体编号, None 表示全部实体.

        Returns
        -------
        Tensor
            Jacobi 矩阵, 形状 ``(NC, NQ, GD, ref_dim)``: ``NC`` 为实体数, ``NQ`` 为点数, ``GD`` 为
            几何维数, ``ref_dim`` 为参考坐标数 (通常等于实体的拓扑维数).
        """
        raise NotImplementedError()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """计算实体的测度."""
        raise NotImplementedError()

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """计算实体的法向量."""
        raise NotImplementedError()

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device = None) -> "Quadrature":
        """实体上的积分公式."""
        raise NotImplementedError()

    def shape_function(self, bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
        """计算本具体 Schema 的几何插值基函数.

        与 :meth:`lagrange_basis_function` 不同, 本方法没有次数参数, 几何次数由不可变的 Schema
        值确定. 结果形状为 ``(Q, Lg)``, ``Lg`` 为 :meth:`number_of_nodes`, 末轴按实体分区连接
        所用的完整局部节点布局. 输出保持输入的浮点类型与设备.
        """
        raise NotImplementedError()

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """计算实体的切向量."""
        raise NotImplementedError()


def _freeze_local_entities(
    entities: Mapping[str, Iterable[Iterable[int]]],
) -> Mapping[str, tuple[tuple[int, ...], ...]]:
    """构造只读、深度不可变的局部实体映射."""
    return MappingProxyType(
        {
            name: tuple(tuple(local) for local in local_entities)
            for name, local_entities in entities.items()
        }
    )
