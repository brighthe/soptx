# 移植自 brighthe/fealpy ``fealpy/mesh/schema/classic/base.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""经典 Schema 的公共基类: 固定参考形状的完整节点布局、单纯形族与张量积族的基函数实现."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from typing import Any, Callable, ClassVar, Concatenate, Literal, overload

from ....backend import bm, Tensor, Index
from ...reference_basis import (
    simplex_lagrange_basis,
    simplex_lagrange_grad_barycentric,
    simplex_lagrange_grad_reference,
    tensor_product_lagrange_basis,
    tensor_product_lagrange_grad_barycentric,
    tensor_product_lagrange_grad_reference,
)
from ...storage import EntityContext
from ..entity_schema import EntitySchema
from ..local_entity import LocalEntityGroup

_NodeKey = tuple[Fraction, ...]
_VertexMap = tuple[int, ...]
_EntityDefinition = tuple["ShapedEntitySchema", tuple[_VertexMap, ...]]


class ShapedEntitySchema(EntitySchema):
    """固定参考形状的完整节点布局规则的共享基类.

    具体子类给出精确的有理参考节点键与子实体定义; 本基类把它们转换为拓扑优先的公开
    布局, 在连接、局部组、定向与正次数基函数列之间一致地使用.
    """

    __slots__ = ()

    _vertex_count: ClassVar[int]

    def _raw_node_keys(self) -> tuple[_NodeKey, ...]:
        raise NotImplementedError()

    def _lagrange_kernel_node_keys(self) -> tuple[_NodeKey, ...]:
        """按共享内核未置换的基函数顺序返回节点键."""
        return self._raw_node_keys()

    def _layout_entity_definitions(self) -> tuple[_EntityDefinition, ...]:
        return ()

    def _local_entity_definitions(
        self,
        top_dim: int,
    ) -> tuple[_EntityDefinition, ...]:
        return ()

    def _candidate_vertex_permutations(self) -> tuple[_VertexMap, ...]:
        raise NotImplementedError()

    @staticmethod
    def _map_child_key(
        child_key: _NodeKey,
        vertex_map: _VertexMap,
        parent_vertex_count: int,
    ) -> _NodeKey:
        mapped = [Fraction(0) for _ in range(parent_vertex_count)]
        for child_vertex, parent_vertex in enumerate(vertex_map):
            mapped[parent_vertex] += child_key[child_vertex]
        return tuple(mapped)

    def _node_keys(self) -> tuple[_NodeKey, ...]:
        raw_keys = self._raw_node_keys()
        raw_key_set = set(raw_keys)
        if len(raw_key_set) != len(raw_keys):
            raise RuntimeError(f"{type(self).__name__} generated duplicate nodes")

        zero = Fraction(0)
        one = Fraction(1)
        ordered: list[_NodeKey] = [
            tuple(one if i == vertex else zero for i in range(self._vertex_count))
            for vertex in range(self._vertex_count)
        ]
        ordered_set = set(ordered)

        for child_schema, vertex_maps in self._layout_entity_definitions():
            child_keys = child_schema._node_keys()
            for vertex_map in vertex_maps:
                for child_key in child_keys:
                    if not all(weight != 0 for weight in child_key):
                        continue
                    mapped = self._map_child_key(
                        child_key,
                        vertex_map,
                        self._vertex_count,
                    )
                    if mapped not in raw_key_set:
                        raise RuntimeError(
                            f"{type(self).__name__} local entity generated "
                            "a node outside its reference-node set"
                        )
                    if mapped not in ordered_set:
                        ordered.append(mapped)
                        ordered_set.add(mapped)

        ordered.extend(key for key in raw_keys if key not in ordered_set)
        if len(ordered) != len(raw_keys) or set(ordered) != raw_key_set:
            raise RuntimeError(
                f"{type(self).__name__} local-node layout is not bijective"
            )
        return tuple(ordered)

    def number_of_vertices(self) -> int:
        """本族顶点骨架的固定大小."""
        return self._vertex_count

    def number_of_nodes(self) -> int:
        """本几何次数下的完整连接宽度."""
        return len(self._node_keys())

    def local_vertices(self) -> tuple[int, ...]:
        """顶点在完整局部节点连接中的列."""
        one = Fraction(1)
        zero = Fraction(0)
        lookup = {key: index for index, key in enumerate(self._node_keys())}
        return tuple(
            lookup[
                tuple(
                    one if i == vertex else zero
                    for i in range(self._vertex_count)
                )
            ]
            for vertex in range(self._vertex_count)
        )

    def local_entity_groups(self, top_dim: int) -> tuple[LocalEntityGroup, ...]:
        """按相等的具体 Schema 分组返回子实体的完整布局."""
        if type(top_dim) is not int:
            raise TypeError("top_dim must be a plain integer")
        if top_dim < 0 or top_dim > self.top_dim:
            raise ValueError(
                f"top_dim must be in [0, {self.top_dim}], got {top_dim}"
            )

        if top_dim == 0:
            from .node import NodeSchema

            return (
                LocalEntityGroup(
                    schema=NodeSchema(),
                    local_node_indices=tuple(
                        (index,) for index in range(self.number_of_nodes())
                    ),
                ),
            )
        if top_dim == self.top_dim:
            return (
                LocalEntityGroup(
                    schema=self,
                    local_node_indices=(tuple(range(self.number_of_nodes())),),
                ),
            )

        parent_keys = self._node_keys()
        lookup = {key: index for index, key in enumerate(parent_keys)}
        groups: list[LocalEntityGroup] = []
        for child_schema, vertex_maps in self._local_entity_definitions(top_dim):
            rows = tuple(
                tuple(
                    lookup[
                        self._map_child_key(
                            child_key,
                            vertex_map,
                            self._vertex_count,
                        )
                    ]
                    for child_key in child_schema._node_keys()
                )
                for vertex_map in vertex_maps
            )
            group = LocalEntityGroup(child_schema, rows)
            if any(index >= len(parent_keys) for row in rows for index in row):
                raise RuntimeError("local-node index exceeds the parent layout")
            if group.schema.top_dim != top_dim:
                raise RuntimeError(
                    "local entity group has the wrong topological dimension"
                )
            groups.append(group)
        return tuple(groups)

    def vertex_permutations(self) -> tuple[_VertexMap, ...]:
        """保持本参数化节点集不变的自同构."""
        node_key_set = set(self._node_keys())
        supported: list[_VertexMap] = []
        for candidate in self._candidate_vertex_permutations():
            if len(candidate) != self._vertex_count:
                raise RuntimeError("invalid candidate vertex permutation width")
            if set(candidate) != set(range(self._vertex_count)):
                raise RuntimeError("candidate vertex permutation is not bijective")
            if all(
                self._permute_node_key(key, candidate) in node_key_set
                for key in node_key_set
            ):
                supported.append(candidate)
        return tuple(supported)

    @staticmethod
    def _permute_node_key(
        key: _NodeKey,
        vertex_permutation: _VertexMap,
    ) -> _NodeKey:
        mapped = [Fraction(0) for _ in key]
        for vertex, original_vertex in enumerate(vertex_permutation):
            mapped[original_vertex] = key[vertex]
        return tuple(mapped)

    def node_permutation(
        self,
        vertex_permutation: tuple[int, ...],
    ) -> tuple[int, ...]:
        """把支持的顶点置换提升到完整的局部节点."""
        if type(vertex_permutation) is not tuple:
            raise TypeError("vertex_permutation must be a tuple")
        if any(type(index) is not int for index in vertex_permutation):
            raise TypeError("vertex_permutation must contain plain integers")
        if vertex_permutation not in self.vertex_permutations():
            raise ValueError(
                f"unsupported vertex permutation {vertex_permutation!r} "
                f"for {self.id}"
            )

        node_keys = self._node_keys()
        lookup = {key: index for index, key in enumerate(node_keys)}
        return tuple(
            lookup[self._permute_node_key(key, vertex_permutation)]
            for key in node_keys
        )

    @classmethod
    def local_entity(
        cls,
        tgt_name: str,
        /,
        indexing: Literal["o", "s"] = "o",
    ) -> tuple[tuple[int, ...], ...]:
        """按名返回局部子实体表.

        ``indexing='o'`` 取有向表 ``OFace``, ``'s'`` 取排序表 ``SFace``.

        Raises
        ------
        ValueError
            ``indexing`` 取值非法, 或本 Schema 没有定义该子实体.
        """
        if indexing == "o":
            if tgt_name in cls.OFace:
                return cls.OFace[tgt_name]
        elif indexing == "s":
            if tgt_name in cls.SFace:
                return cls.SFace[tgt_name]
        else:
            raise ValueError(f"indexing must be 'o' or 's', got {indexing!r}")
        raise ValueError(f"local entity {tgt_name!r} is not defined for {cls.name!r}")

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """分区中的实体个数."""
        return ctx.sector.indices.shape[0]

    ### [几何计算] ###

    @classmethod
    def barycentric[**P, R](
        cls,
        ctx: EntityContext,
        func: Callable[Concatenate[Tensor, P], R],
        index: Index | None
    ) -> Callable[Concatenate[Tensor | tuple[Tensor, ...], P], R]:
        """把直角坐标函数包装为重心坐标函数: 求值时先把重心坐标映射为物理点."""
        from functools import wraps
        from ....decorator import barycentric
        @wraps(func)
        @barycentric
        def wrapper(bcs: Tensor | tuple[Tensor, ...], *args, **kwargs) -> R:
            """把重心坐标映射为物理点后调用原函数."""
            if not isinstance(bcs, tuple):
                bcs = (bcs,)
            points = cls.bc_to_point(ctx, bcs, index) # [NC, NQ, GD]
            return func(points, *args, **kwargs)
        return wrapper

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """网格块节点坐标的几何维数."""
        return int(ctx.block.positions.shape[1])

    @classmethod
    def integral(cls, ctx: EntityContext, func: Callable[[Tensor], Tensor], q: int, index: Index | None) -> Tensor:
        """重心坐标函数的积分."""
        quadrature = cls.quadrature_formula(q)
        bcs, ws = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)

        if not getattr(func, "coordtype", None) == "barycentric":
            func = cls.barycentric(ctx, func, index)
        values = func(bcs) # type: ignore
        J = cls.jacobi_matrix(ctx, bcs, index)
        det = bm.linalg.det # type: ignore

        if J.shape[-2] == J.shape[-1]:
            factor = bm.abs(det(J)) # [NC, NQ]
        else:
            # 非方阵的 Jacobi 矩阵取 J^T * J 行列式的平方根
            JTJ = bm.einsum("...ji, ...jk -> ...ik", J, J) # [NC, NQ, ref_dim, ref_dim]
            factor = bm.sqrt(det(JTJ)) # [NC, NQ]

        if cls.ref_measure is not None:
            factor = cls.ref_measure * factor

        return bm.einsum("cq, q, cq... -> c...", factor, ws, values)


def _normalize_scalar_geometry_order(p: object, name: str) -> int:
    if type(p) is not int:
        raise TypeError(f"{name} p must be an integer, got {type(p).__name__}")
    if p < 1:
        raise ValueError(f"{name} p must be positive, got {p}")
    return p


def _normalize_tensor_geometry_order(
    p: object,
    factor_count: int,
    name: str,
) -> tuple[int, ...]:
    if type(p) is int:
        values = (p,) * factor_count
    elif type(p) is tuple:
        values = p
    else:
        raise TypeError(
            f"{name} p must be an integer or a tuple of integers, "
            f"got {type(p).__name__}"
        )

    if len(values) != factor_count:
        raise ValueError(
            f"{name} p must contain {factor_count} orders, got {len(values)}"
        )
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} p must contain only integers, got {values!r}")
    if any(value < 1 for value in values):
        raise ValueError(
            f"{name} p must contain only positive orders, got {values!r}"
        )
    return values


def _normalize_scalar_basis_order(p: object, name: str) -> int:
    if type(p) is not int:
        raise TypeError(
            f"{name} basis p must be an integer, got {type(p).__name__}"
        )
    if p < 0:
        raise ValueError(f"{name} basis p must be non-negative, got {p}")
    return p


def _normalize_tensor_basis_order(
    p: object,
    factor_count: int,
    name: str,
) -> tuple[int, ...]:
    if type(p) is int:
        values = (p,) * factor_count
    elif type(p) is tuple:
        values = p
    else:
        raise TypeError(
            f"{name} basis p must be an integer or a tuple of integers, "
            f"got {type(p).__name__}"
        )
    if len(values) != factor_count:
        raise ValueError(
            f"{name} basis p must contain {factor_count} orders, got {len(values)}"
        )
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} basis p must contain only integers, got {values!r}")
    if any(value < 0 for value in values):
        raise ValueError(
            f"{name} basis p must contain only non-negative orders, got {values!r}"
        )
    return values


@dataclass(frozen=True, slots=True)
class _ScalarOrderSchema(ShapedEntitySchema):
    """几何次数为标量 ``p`` 的单纯形族 Schema 实现.

    几何次数为正的普通整数; 独立 Lagrange 基函数的次数为非负普通整数, 不改变 ``p``.
    输入为形状 ``(Q, V)`` 的重心坐标张量, 或只含该张量的单元素元组.
    """

    p: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "p",
            _normalize_scalar_geometry_order(self.p, type(self).__name__),
        )

    def __hash__(self) -> int:
        return hash((type(self), self.descriptor))

    def _lagrange_basis_permutation(self, p: int) -> tuple[int, ...] | None:
        if p == 0:
            return None
        basis_schema = type(self)(p)
        source = basis_schema._lagrange_kernel_node_keys()
        lookup = {key: index for index, key in enumerate(source)}
        return tuple(lookup[key] for key in basis_schema._node_keys())

    def _lagrange_bcs(self, bcs: Tensor | tuple[Tensor, ...], name: str) -> Tensor:
        if isinstance(bcs, tuple):
            if len(bcs) != 1:
                raise ValueError(
                    f"{name} expects one barycentric tensor, got {len(bcs)}"
                )
            bcs = bcs[0]
        if int(bcs.shape[-1]) != self._vertex_count:
            raise ValueError(
                f"{name} expects last dimension {self._vertex_count}, "
                f"got {bcs.shape[-1]}"
            )
        return bcs

    def shape_function(self, bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
        """计算 ``self.p`` 次几何基函数.

        结果形状为 ``(Q, number_of_nodes())``, 按拓扑优先的完整局部节点顺序.
        """
        return self.lagrange_basis_function(bcs, self.p)

    def grad_shape_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """几何基函数的重心坐标梯度, 形状 ``(Q, Lg, V)``."""
        return self.grad_lagrange_basis_function_barycentric(bcs, self.p)

    def grad_shape_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """几何基函数的参考坐标梯度, 形状 ``(Q, Lg, V - 1)``."""
        return self.grad_lagrange_basis_function_reference(bcs, self.p)

    def lagrange_basis_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """按拓扑优先的列顺序计算 ``p`` 次基函数.

        结果形状为 ``(Q, Lp)``, 保持坐标张量的浮点类型与设备; ``p=0`` 返回一列常数基函数.

        Raises
        ------
        TypeError
            ``p`` 不是普通整数.
        ValueError
            ``p`` 为负, 或重心坐标的形状与参考单纯形不符.
        """
        p = _normalize_scalar_basis_order(p, type(self).__name__)
        bc = self._lagrange_bcs(bcs, "lagrange_basis_function")
        return simplex_lagrange_basis(
            bc,
            p,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(p),
        )

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """``p`` 次基函数的重心坐标梯度, 形状 ``(Q, Lp, V)``."""
        p = _normalize_scalar_basis_order(p, type(self).__name__)
        bc = self._lagrange_bcs(
            bcs,
            "grad_lagrange_basis_function_barycentric",
        )
        return simplex_lagrange_grad_barycentric(
            bc,
            p,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(p),
        )

    def grad_lagrange_basis_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int,
    ) -> Tensor:
        """``p`` 次基函数的参考坐标梯度, 形状 ``(Q, Lp, V - 1)``."""
        p = _normalize_scalar_basis_order(p, type(self).__name__)
        bc = self._lagrange_bcs(
            bcs,
            "grad_lagrange_basis_function_reference",
        )
        return simplex_lagrange_grad_reference(
            bc,
            p,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(p),
        )


@dataclass(frozen=True, slots=True)
class _TensorProductOrderSchema(ShapedEntitySchema):
    """由有序的单纯形参考因子组成的 Schema 实现.

    标量几何次数对每个因子重复使用, 否则把 ``p`` 规范化为按具体 Schema 公开因子顺序的
    正整数元组; 独立基函数的次数还可以含 0. 输入坐标总是元组, 每个公开因子一个
    ``(Qf, Vf)`` 重心坐标张量, 按其笛卡尔积求值.
    """

    factor_count: ClassVar[int]
    p: int | tuple[int, ...] = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "p",
            _normalize_tensor_geometry_order(
                self.p,
                self.factor_count,
                type(self).__name__,
            ),
        )

    def __hash__(self) -> int:
        return hash((type(self), self.descriptor))

    def _lagrange_kernel_factor_order(self) -> tuple[int, ...]:
        """把共享内核的因子位置映射到公开的参考因子."""
        return tuple(range(self.factor_count))

    @classmethod
    def multi_index_vertex_columns(cls) -> tuple[int, ...] | None:
        """``multi_index`` 列到局部顶点编号的置换, 即具体 Schema 的 ``_tp_to_contract``; 未定义时 (如三棱柱) 返回 None."""
        # multi_index 按因子嵌套顺序给列编号, 与 bc_to_point 在张量收缩前对顶点
        # 做的重排是同一个置换, 因此直接复用具体 Schema 已有的 _tp_to_contract.
        # 三棱柱两套编号本就一致, 不定义该常量, 这里返回 None.
        return getattr(cls, "_tp_to_contract", None)

    def _lagrange_basis_permutation(
        self,
        p: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        if any(order == 0 for order in p):
            return None
        basis_schema = type(self)(p)
        source = basis_schema._lagrange_kernel_node_keys()
        lookup = {key: index for index, key in enumerate(source)}
        return tuple(lookup[key] for key in basis_schema._node_keys())

    def _lagrange_inputs(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
        name: str,
    ) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
        if not isinstance(bcs, tuple):
            raise TypeError(f"{name} expects a tuple of barycentric tensors")
        if len(bcs) != self.factor_count:
            raise ValueError(
                f"{name} expects {self.factor_count} barycentric tensors, "
                f"got {len(bcs)}"
            )
        order = _normalize_tensor_basis_order(
            p,
            self.factor_count,
            type(self).__name__,
        )
        factor_order = self._lagrange_kernel_factor_order()
        return (
            tuple(bcs[factor] for factor in factor_order),
            tuple(order[factor] for factor in factor_order),
        )

    def _restore_gradient_factor_order(
        self,
        values: Tensor,
        bcs: tuple[Tensor, ...],
        *,
        reference: bool,
    ) -> Tensor:
        factor_order = self._lagrange_kernel_factor_order()
        if factor_order == tuple(range(self.factor_count)):
            return values
        blocks: dict[int, tuple[int, ...]] = {}
        offset = 0
        for factor in factor_order:
            width = int(bcs[factor].shape[-1]) - int(reference)
            blocks[factor] = tuple(range(offset, offset + width))
            offset += width
        component_order = tuple(
            component
            for factor in range(self.factor_count)
            for component in blocks[factor]
        )
        indices = bm.asarray(
            component_order,
            dtype=bm.int64,
            device=bm.get_device(values),
        )
        return values[..., indices]

    def shape_function(self, bcs: tuple[Tensor, ...]) -> Tensor:
        """在笛卡尔积点上计算几何基函数.

        结果形状为 ``(prod(Qf), number_of_nodes())``, 按拓扑优先的完整局部节点顺序.
        """
        return self.lagrange_basis_function(bcs, self.p)

    def grad_shape_function_barycentric(
        self,
        bcs: tuple[Tensor, ...],
    ) -> Tensor:
        """按公开的重心坐标因子顺序返回几何基函数的梯度."""
        return self.grad_lagrange_basis_function_barycentric(bcs, self.p)

    def grad_shape_function_reference(
        self,
        bcs: tuple[Tensor, ...],
    ) -> Tensor:
        """按公开的参考因子顺序返回几何基函数的梯度."""
        return self.grad_lagrange_basis_function_reference(bcs, self.p)

    def lagrange_basis_function(
        self,
        bcs: tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """计算独立的张量积 Lagrange 基函数.

        标量 ``p`` 对每个因子重复使用, 元组则为每个公开因子给出一个非负次数. 结果形状为
        ``(prod(Qf), Lp)``, 保持输入的浮点类型与设备. 正次数时各列遵循相等几何 Schema 的
        完整局部节点布局; 某因子次数为 0 时没有完整的顶点骨架, 采用内核文档所述的规范乘积
        顺序.

        Raises
        ------
        TypeError
            坐标不是元组, 或 ``p`` 的类型不合法.
        ValueError
            因子个数、坐标形状或次数取值不合法.
        """
        kernel_bcs, kernel_order = self._lagrange_inputs(
            bcs,
            p,
            "lagrange_basis_function",
        )
        public_order = _normalize_tensor_basis_order(
            p,
            self.factor_count,
            type(self).__name__,
        )
        return tensor_product_lagrange_basis(
            kernel_bcs,
            kernel_order,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(public_order),
        )

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """按公开的重心坐标因子顺序返回基函数梯度, 形状 ``(prod(Qf), Lp, sum(Vf))``."""
        kernel_bcs, kernel_order = self._lagrange_inputs(
            bcs,
            p,
            "grad_lagrange_basis_function_barycentric",
        )
        public_order = _normalize_tensor_basis_order(
            p,
            self.factor_count,
            type(self).__name__,
        )
        values = tensor_product_lagrange_grad_barycentric(
            kernel_bcs,
            kernel_order,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(public_order),
        )
        return self._restore_gradient_factor_order(
            values,
            bcs,
            reference=False,
        )

    def grad_lagrange_basis_function_reference(
        self,
        bcs: tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """按公开的参考因子顺序返回基函数梯度, 形状 ``(prod(Qf), Lp, sum(Vf - 1))``."""
        kernel_bcs, kernel_order = self._lagrange_inputs(
            bcs,
            p,
            "grad_lagrange_basis_function_reference",
        )
        public_order = _normalize_tensor_basis_order(
            p,
            self.factor_count,
            type(self).__name__,
        )
        values = tensor_product_lagrange_grad_reference(
            kernel_bcs,
            kernel_order,
            descriptor=self.descriptor,
            permutation=self._lagrange_basis_permutation(public_order),
        )
        return self._restore_gradient_factor_order(
            values,
            bcs,
            reference=True,
        )


def _compositions_desc(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 1:
        return ((total,),)
    return tuple(
        (first,) + tail
        for first in range(total, -1, -1)
        for tail in _compositions_desc(total - first, length - 1)
    )


def _simplex_node_keys(p: int, vertex_count: int) -> tuple[_NodeKey, ...]:
    return tuple(
        tuple(Fraction(value, p) for value in composition)
        for composition in _compositions_desc(p, vertex_count)
    )


def _quadrilateral_node_keys(p: tuple[int, int]) -> tuple[_NodeKey, ...]:
    px, py = p
    result: list[_NodeKey] = []
    for iy in range(py + 1):
        y = Fraction(iy, py)
        for ix in range(px + 1):
            x = Fraction(ix, px)
            result.append(
                ((1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y)
            )
    return tuple(result)


def _hexahedron_node_keys(p: tuple[int, int, int]) -> tuple[_NodeKey, ...]:
    px, py, pz = p
    result: list[_NodeKey] = []
    for iz in range(pz + 1):
        z = Fraction(iz, pz)
        for iy in range(py + 1):
            y = Fraction(iy, py)
            for ix in range(px + 1):
                x = Fraction(ix, px)
                result.append(
                    (
                        (1 - x) * (1 - y) * (1 - z),
                        x * (1 - y) * (1 - z),
                        x * y * (1 - z),
                        (1 - x) * y * (1 - z),
                        (1 - x) * (1 - y) * z,
                        x * (1 - y) * z,
                        x * y * z,
                        (1 - x) * y * z,
                    )
                )
    return tuple(result)


def _prism_node_keys(p: tuple[int, int]) -> tuple[_NodeKey, ...]:
    triangle_order, interval_order = p
    triangle_keys = _simplex_node_keys(triangle_order, 3)
    result: list[_NodeKey] = []
    for iz in range(interval_order + 1):
        z = Fraction(iz, interval_order)
        for a, b, c in triangle_keys:
            result.append(
                (
                    a * (1 - z),
                    b * (1 - z),
                    c * (1 - z),
                    a * z,
                    b * z,
                    c * z,
                )
            )
    return tuple(result)


def _simplex_vertex_permutations(vertex_count: int) -> tuple[_VertexMap, ...]:
    return tuple(permutations(range(vertex_count)))


def _quadrilateral_vertex_permutations() -> tuple[_VertexMap, ...]:
    return (
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 3, 2),
        (2, 1, 0, 3),
        (3, 2, 1, 0),
    )


def _hexahedron_vertex_permutations() -> tuple[_VertexMap, ...]:
    vertices = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    lookup = {coordinate: index for index, coordinate in enumerate(vertices)}
    result: list[_VertexMap] = []
    for axes in permutations(range(3)):
        for flips in product((0, 1), repeat=3):
            candidate = tuple(
                lookup[
                    tuple(
                        1 - coordinate[axes[axis]]
                        if flips[axis]
                        else coordinate[axes[axis]]
                        for axis in range(3)
                    )
                ]
                for coordinate in vertices
            )
            if candidate not in result:
                result.append(candidate)
    return tuple(result)


def _prism_vertex_permutations() -> tuple[_VertexMap, ...]:
    result: list[_VertexMap] = []
    for triangle_permutation in permutations(range(3)):
        bottom = tuple(triangle_permutation) + tuple(
            vertex + 3 for vertex in triangle_permutation
        )
        top = tuple(vertex + 3 for vertex in triangle_permutation) + tuple(
            triangle_permutation
        )
        result.extend((bottom, top))
    return tuple(result)


def _group_entity_definitions(
    definitions: tuple[tuple[ShapedEntitySchema, _VertexMap], ...],
) -> tuple[_EntityDefinition, ...]:
    grouped: list[tuple[ShapedEntitySchema, list[_VertexMap]]] = []
    for schema, vertex_map in definitions:
        for grouped_schema, vertex_maps in grouped:
            if grouped_schema == schema:
                vertex_maps.append(vertex_map)
                break
        else:
            grouped.append((schema, [vertex_map]))
    return tuple((schema, tuple(maps)) for schema, maps in grouped)


@overload
def _require_bcs_tuple(bcs: Any, name: str) -> tuple[Tensor, ...]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[1]) -> tuple[Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[2]) -> tuple[Tensor, Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[3]) -> tuple[Tensor, Tensor, Tensor]: ...
def _require_bcs_tuple(bcs: Any, name: str, n: int | None = None) -> tuple[Tensor, ...]:
    if not isinstance(bcs, tuple):
        raise TypeError(f"{name} expects barycentric coordinates as a tuple of tensors, got {type(bcs).__name__}")

    if n is None:
        return bcs

    if len(bcs) == n:
        pass
    elif len(bcs) == 1:
        bcs = (bcs[0],) * n
    else:
        raise ValueError(f"{name} expects {n} barycentric tensors, got {len(bcs)}")

    return bcs


@overload
def _require_order_tuple(p: Any, name: str) -> tuple[int, ...]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[1]) -> tuple[int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[2]) -> tuple[int, int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[3]) -> tuple[int, int, int]: ...
def _require_order_tuple(p: Any, name: str, n: int | None = None) -> tuple[int, ...]:
    """确保多项式次数为整数元组.

    Parameters
    ----------
    p : tuple of int
        输入的多项式次数.
    name : str
        报错信息中使用的函数名.
    n : int or None
        期望的次数个数; None 时不检查. 为整数时元组长度须为 1 或 ``n``, 长度为 1 表示
        所有方向使用同一次数, 返回重复 ``n`` 次的元组.

    Returns
    -------
    tuple of int
        次数元组, ``n`` 不为 None 时长度为 ``n``.
    """
    if not isinstance(p, tuple):
        raise TypeError(f"{name} expects polynomial degrees as a tuple of integers, got {type(p).__name__}")

    if not all(isinstance(pi, int) for pi in p):
        raise TypeError(f"{name} expects polynomial degrees as integers, got {p}")

    if not all(pi >= 0 for pi in p):
        raise ValueError(f"{name} expects non-negative polynomial degrees, got {p}")

    if n is None:
        return p

    if len(p) == n:
        pass
    elif len(p) == 1:
        p = (p[0],) * n
    else:
        raise ValueError(f"{name} expects {n} polynomial degrees, got {len(p)}")

    return p
