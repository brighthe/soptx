# 移植自 brighthe/fealpy ``fealpy/mesh/view/entity_view.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""绑定在实体分区上的同类实体视图.

``EntityView`` 为 :class:`MeshBlock` 中的一个 ``EntitySector`` 提供几何、积分、插值与
关系的访问. 本模块还保留私有的 ``_legacy_*`` 基函数顺序适配器, 供经典 ``MeshView`` 的
历史形函数约定使用; 它们不是公开的基函数接口, 新的函数空间代码应直接调用
``EntitySchema`` 或 ``EntityView`` 的接口.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, final, Literal, ParamSpec, TYPE_CHECKING

from ...backend import bm, Tensor, Index
from ..schema.entity_schema import EntitySchema
from ..schema.classic.base import _TensorProductOrderSchema
from ..storage import EntityContext

if TYPE_CHECKING:
    from ..storage import MeshBlock, EntitySector, Relation
    from ..topology.boundary import BoundaryInfo

__all__ = ["EntityView"]

P = ParamSpec("P")


def _small_square_det(matrix: Tensor) -> Tensor:
    """最后两维为 1x1, 2x2 或 3x3 方阵时, 按展开式逐元素计算行列式.

    Parameters
    ----------
    matrix : 形状 ``(..., n, n)``, ``n`` 取 1, 2 或 3.

    Returns
    -------
    det : 形状 ``(...)`` 的行列式.
    """
    a = matrix
    n = int(a.shape[-1])
    if n == 1:
        return a[..., 0, 0]
    if n == 2:
        return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    return (a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
            - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
            + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]))


def _normalized_legacy_order(
    schema: EntitySchema,
    p: int | tuple[int, ...],
) -> int | tuple[int, ...]:
    """为私有的经典基函数列适配器规范化次数参数."""
    factor_count = getattr(schema, "factor_count", None)
    if factor_count is None:
        if type(p) is int:
            return p
        if type(p) is tuple and len(p) == 1:
            return p[0]
        return p
    if type(p) is int:
        return (p,) * factor_count
    return p


def _legacy_basis_indices(
    schema: EntitySchema,
    p: int | tuple[int, ...],
) -> Tensor | None:
    """把新的局部节点基函数顺序映射回经典视图的顺序."""
    permutation_getter = getattr(schema, "_lagrange_basis_permutation", None)
    if permutation_getter is None:
        return None

    order = _normalized_legacy_order(schema, p)
    permutation = permutation_getter(order)
    if permutation is None:
        return None

    # 张量积单元 (四边形/六面体/三棱柱) 不做这次反置换.
    #
    # lagrange_basis_function 已经把基函数排成 _node_keys() 序, 而
    # cell_to_ipoint 给出的局部自由度也正是这个序; 再套一次逆置换会把列序
    # 打回张量积枚举序 (四边形 p=1 即 (0,0),(1,0),(0,1),(1,1)), 与自由度
    # 错位. 后果是静默的: 残差照样收敛, 只有观测收敛阶会塌掉 (Q1 由 2.0
    # 掉到 0.11).
    #
    # 原判据只豁免 prism / hexahedron 且要求各方向阶次全为 1, 于是
    # 六面体 p=1 侥幸正确, 四边形所有阶次以及 p >= 2 的六面体/三棱柱全错.
    # 单纯形的 _lagrange_basis_permutation 是恒等置换, 反置换对它无害,
    # 因此这里只需按张量积与否豁免, 不再看阶次.
    if isinstance(schema, _TensorProductOrderSchema):
        return None

    inverse = [0] * len(permutation)
    for local_index, kernel_index in enumerate(permutation):
        inverse[kernel_index] = local_index
    return bm.asarray(inverse, dtype=bm.int64)


def _restore_legacy_basis_order(
    values: Tensor,
    schema: EntitySchema,
    p: int | tuple[int, ...],
    *,
    gradient: bool,
) -> Tensor:
    """施加私有的 FEALPy 风格视图基函数列约定."""
    indices = _legacy_basis_indices(schema, p)
    if indices is None:
        return values
    indices = bm.device_put(indices, bm.get_device(values))
    if gradient:
        return values[..., indices, :]
    return values[..., indices]


def _pyramid_geometry_order(p: int | tuple[int, ...]) -> bool:
    """判断过渡期的四棱锥视图请求是否为几何 p=1."""
    return p == 1 or p == (1,)


@final
@dataclass(slots=True)
class EntityView:
    """一个同类网格实体分区的视图.

    ``EntityView`` 以稳定的分区 id 把网格块绑定到一个分区上; 它不另存一份
    :class:`EntitySector` 的状态, :attr:`sector` 与 :attr:`schema` 每次访问都从网格块中
    解析.

    Attributes
    ----------
    block : MeshBlock
        含坐标、分区与关系的网格存储块.
    sector_id : str
        经 ``block.sectors`` 解析的稳定分区 id.
    """
    block: MeshBlock
    sector_id: str

    def __post_init__(self) -> None:
        """dataclass 初始化后校验绑定的分区 id."""
        if type(self.sector_id) is not str or not self.sector_id:
            raise TypeError("EntityView.sector_id must be a non-empty string")
        self.block.get_sector(self.sector_id)

    @property
    def sector(self) -> EntitySector:
        """从网格块中取得绑定的 :class:`EntitySector`."""
        return self.block.get_sector(self.sector_id)

    @property
    def schema(self) -> EntitySchema:
        """分区绑定的具体、不可变的 Schema 值."""
        return self.sector.schema

    def __len__(self) -> int:
        """本分区的实体个数."""
        return self.schema.size(self.context())

    def context(self) -> EntityContext:
        """本实体视图的 Schema 计算上下文.

        Returns
        -------
        EntityContext
            含网格块与当前实体分区的轻量容器.
        """
        return EntityContext(self.block, self.sector)

    def _selected_connectivity(self, index: Index | None) -> Tensor:
        """返回选定的分区连接数组, 保留显式的实体轴."""
        connectivity = self.sector.indices
        if index is None:
            return connectivity
        selected = connectivity[index]
        if len(selected.shape) == 1:
            selected = bm.reshape(selected, (1, -1))
        return selected

    def _default_geometry_quadrature_order(self) -> int:
        """返回确定的默认几何积分阶."""
        order = getattr(self.schema, "p", None)
        if order is None:
            return 1
        if isinstance(order, tuple):
            order = max(order)
        return max(int(order), 1) + 1

    def _reference_center_bcs(self) -> tuple[Tensor, ...]:
        """返回参考实体规范中心的重心坐标输入."""
        if self.schema.type_id == "lagrange_pyramid":
            center = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
            return (center, center, center)
        quadrature = self.schema.quadrature_formula(1)
        bcs, _ = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        return bcs

    # 用户接口

    def barycentric[**P, R](self, func: Callable[Concatenate[Tensor, P], R], /, *, index: Index | None = None):
        """把直角坐标函数包装为重心坐标函数.

        Parameters
        ----------
        func : callable
            首个位置参数为物理坐标张量的函数, 坐标的形状由实体的 Schema 决定.
        index : Index, optional
            实体子集.
            把重心坐标转换为物理点时使用.

        Returns
        -------
        callable
            其余参数与 ``func`` 相同、首个位置参数为重心坐标张量 (张量积实体为重心坐标
            张量的元组) 的函数.
        """
        return self.schema.barycentric(self.context(), func, index)

    def barycenter(self, *, index: Index | None = None) -> Tensor:
        """选定实体的映射后参考中心.

        参考中心经与 :meth:`bc_to_point` 相同的全节点几何插值映射. 对仿射实体它等于通常的
        顶点平均; 对弯曲实体则不是按测度加权的形心.

        Parameters
        ----------
        index : Index, optional
            实体子集. 为 None 时使用本分区的全部实体.

        Returns
        -------
        Tensor
            重心坐标, 形状 ``(NE, GD)`` 或对应子集的形状, ``GD`` 为几何维数.
        """
        if self.sector.indptr is not None:
            return self.schema.barycenter(self.context(), index)
        bcs = self._reference_center_bcs()
        points = self.bc_to_point(bcs, index=index)
        return bm.reshape(points, (points.shape[0], points.shape[-1]))

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """把重心坐标映射为物理坐标.

        映射使用完整的几何局部节点布局, 因此弯曲的高阶实体用全部几何节点计算, 而不只是
        顶点骨架.

        Parameters
        ----------
        bc : Tensor or tuple of Tensor
            重心坐标: 单纯形实体为一个张量, 张量积实体可为各因子一个张量.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            物理点; 常见单纯形实体选定实体与积分点后的形状为 ``(NE, NQ, GD)``.

        Raises
        ------
        TypeError
            坐标容器类型不合法.
        ValueError
            重心坐标的形状与绑定的 Schema 不符.
        """
        if not isinstance(bc, tuple):
            bc = (bc,)
        values = self.schema.shape_function(bc)
        points = self.block.positions[self._selected_connectivity(index)]
        return bm.einsum("qi,cij->cqj", values, points)

    def boundary(self) -> "BoundaryInfo":
        """推断本实体分区的边界信息.

        Returns
        -------
        BoundaryInfo
            ``mask`` 为标记边界实体的布尔张量, ``index`` 为边界实体的编号.
        """
        from ..topology.boundary import BoundaryInferencer

        if self.block._cache_boundary_info is None:
            self.block._cache_boundary_info = {}
        return BoundaryInferencer.infer_entity(
            self.block,
            self.sector_id,
            self.block._cache_boundary_info,
        )

    def del_attribute(self, name: str) -> None:
        """删除实体分区上的用户属性.

        Parameters
        ----------
        name : str
            属性名.

        Raises
        ------
        KeyError
            ``sector.attributes`` 中没有 ``name``.
        """
        if name in self.sector.attributes:
            del self.sector.attributes[name]
        else:
            raise KeyError(f"Attribute '{name}' not found in sector attributes.")

    def error(
        self,
        f1: Callable[..., Tensor],
        f2: Callable[..., Tensor],
        /,
        power: float = 2.0,
        q: int = 3,
        *,
        cell_axis: bool = False,
        index: Index | None = None
    ) -> Tensor:
        """计算两个函数之差在实体上的积分误差范数.

        未标记为重心坐标的函数先经 :meth:`barycentric` 包装. 计算的量为
        ``(integral(abs(f1 - f2)**power))**(1/power)``.

        Parameters
        ----------
        f1, f2 : callable
            两个函数.
        power : float, optional
            范数的幂次, 默认 2.0.
        q : int, optional
            积分阶, 默认 3.
        cell_axis : bool, optional
            为 True 时返回每个选定实体上的误差, 否则返回全部选定实体上的总误差.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            标量总误差; ``cell_axis`` 为 True 时为逐实体误差.
        """
        from ...decorator import barycentric
        if not getattr(f1, "coordtype", None) == "barycentric":
            f1 = self.barycentric(f1, index=index)
        if not getattr(f2, "coordtype", None) == "barycentric":
            f2 = self.barycentric(f2, index=index)
        @barycentric
        def integrand(bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
            """被积函数 ``|f1 - f2| ** power``."""
            v1 = f1(bcs)
            v2 = f2(bcs)
            return bm.abs(v1 - v2) ** power

        if cell_axis:
            return self.integral(integrand, q=q, index=index) ** (1.0 / power)
        return bm.sum(self.integral(integrand, q=q, index=index)) ** (1.0 / power)

    def geo_dimension(self) -> int:
        """嵌入空间的几何维数."""
        return self.schema.geo_dimension(self.context())

    def get_attribute(self, name: str) -> Any:
        """返回实体分区上存储的用户属性.

        Parameters
        ----------
        name : str
            属性名.

        Returns
        -------
        Any
            属性值; 属性不存在时为 None.
        """
        return self.sector.attributes.get(name)

    def global_permutations(
        self,
        name_or_topdim: str | int | EntityView,
        idx: int = 0,
        indexing: Literal["o", "s"] = "o",
    ) -> Tensor:
        """返回子实体从局部到全局定向的置换.

        Parameters
        ----------
        name_or_topdim : str, int or EntityView
            目标子实体的 schema 名、拓扑维数或显式的 ``EntityView``.
        idx : int, optional
            ``name_or_topdim`` 选中的维数或实体类型有多个分区时, 目标分区的序号. 默认 0.

        Returns
        -------
        Tensor
            整数张量: 前导轴依次枚举源实体及其局部目标实体, 末轴存全局定向诱导的顶点置换.
        """
        if indexing != "o":
            raise NotImplementedError(
                "only indexing='o' is supported by explicit-sector orientation"
            )

        if not isinstance(name_or_topdim, EntityView):
            raise TypeError("global_permutations expects an EntityView")
        tgt = name_or_topdim

        from ...mesh.schema.utils import argpermute
        from ...mesh.topology.relation import resolve_relation

        groups = self.schema.local_entity_groups(tgt.top_dimension())
        candidates = [group for group in groups if group.schema == tgt.schema]
        if not candidates:
            raise ValueError(
                f"target sector {tgt.sector_id!r} is not a local subentity "
                f"of {self.sector_id!r}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous local subentity groups for target "
                f"{tgt.sector_id!r}"
            )

        child_vertices = tgt.schema.local_vertices()
        local_face_vertices = [
            tuple(row[vertex] for vertex in child_vertices)
            for row in candidates[0].local_node_indices
        ]
        local_face = bm.asarray(local_face_vertices, dtype=bm.int64)
        parent_vertices = self.indices[:, local_face]

        target_indices = tgt.indices[:, child_vertices]
        cell_to_target = resolve_relation(
            self.block,
            self.sector_id,
            tgt.sector_id,
        ).tgt_indices

        return argpermute(
            parent_vertices,
            target_indices[cell_to_target],
            dtype=bm.uint8,
        )

    def grad_lambda(
        self,
        bcs: tuple[Tensor, ...] | None = None,
        index: Index | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """重心坐标函数的梯度.

        Parameters
        ----------
        bcs : tuple of Tensor, optional
            求值点的重心坐标; 给出时梯度可沿积分点轴广播.
        index : Index, optional
            实体子集.
        ref : bool, optional
            为 True 时返回参考实体上的梯度, 否则 (默认) 返回物理坐标下的梯度.

        Returns
        -------
        Tensor
            重心坐标的梯度; 不给 ``bcs`` 时, 物理梯度的典型形状为 ``(NE, NV, GD)``, 参考梯度为
            ``(NE, NV, NR)``.
        """
        return self.schema.grad_lambda(self.context(), index, bcs=bcs, ref=ref) # type: ignore

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """本分区几何形函数的参考梯度.

        几何次数由绑定的 Schema 值确定, 本方法不接受独立的解次数. 结果形状为
        ``(Q, Lg, R)``, ``Lg`` 为 Schema 的完整局部节点数, ``R`` 为参考维数.
        """
        return self.schema.grad_shape_function_reference(bcs)

    def grad_shape_function_cartesian(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        *,
        index: Index | None = None,
    ) -> Tensor:
        """几何形函数在物理坐标下的梯度, 形状 ``(NE, Q, Lg, GD)``.

        由参考梯度经当前全节点 Jacobi 矩阵与参考度量映射得到, 因此弯曲几何不会被静默地
        替换为只用顶点的仿射公式.

        Parameters
        ----------
        bcs : Tensor or tuple of Tensor
            求值点的重心坐标: 单纯形实体为一个张量, 张量积实体为各因子一个张量.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            物理坐标下的梯度, 形状 ``(NE, Q, Lg, GD)``.
        """
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        reference_gradient = self.schema.grad_shape_function_reference(bcs)
        jacobian = self.jacobi_matrix(bcs, index=index)
        metric = bm.einsum("cqdr,cqds->cqrs", jacobian, jacobian)
        metric_inverse = bm.linalg.inv(metric)
        return bm.einsum(
            "cqdr,cqrs,qis->cqid",
            jacobian,
            metric_inverse,
            reference_gradient,
        )

    def _legacy_grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal["b", "u", "x"] = "u",
        mi = None,
    ) -> Tensor:
        """私有的经典任意次梯度入口.

        为经典 ``MeshView`` 的方法保留 FEALPy 历史上 ``grad_shape_function(p)`` 的列约定.
        新代码应直接使用 :meth:`EntityView.grad_shape_function` 或
        ``EntitySchema.grad_lagrange_basis_function_reference``.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if variables == "b":
            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                return self.schema.grad_shape_function_barycentric(bcs)
            grad = self.schema.grad_lagrange_basis_function_barycentric(
                bcs,
                p,
            )
            return _restore_legacy_basis_order(
                grad,
                self.schema,
                p,
                gradient=True,
            )
        elif variables == "u":
            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                grad = self.schema.grad_shape_function_reference(bcs)
            else:
                grad = self.schema.grad_lagrange_basis_function_reference(
                    bcs,
                    p,
                )
            return _restore_legacy_basis_order(
                grad,
                self.schema,
                p,
                gradient=True,
            )
        elif variables == "x":
            from ..mapping import piola_transform_covariant

            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                grad_ref = self.schema.grad_shape_function_reference(bcs)
            else:
                grad_ref = self.schema.grad_lagrange_basis_function_reference(
                    bcs,
                    p,
                )
            grad_ref = _restore_legacy_basis_order(
                grad_ref,
                self.schema,
                p,
                gradient=True,
            )[None, ...]
            J = self.schema.jacobi_matrix(
                self.context(),
                bcs,
                index,
            )[..., None, :, :]
            return piola_transform_covariant(grad_ref, J)
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    @property
    def indices(self) -> Tensor:
        """本实体分区的连接数组."""
        return getattr(self.sector, "indices")

    @property
    def indptr(self) -> Tensor | None:
        """变长连接的偏移量; 定长分区为 None."""
        return self.sector.indptr

    def integral(
        self,
        func: Callable[..., Tensor],
        /,
        q: int = 3,
        *,
        index: Index | None = None
    ) -> Tensor:
        """在选定实体上积分重心坐标函数.

        Parameters
        ----------
        func : callable
            在重心坐标积分点处求值的函数.
        q : int, optional
            积分阶, 默认 3.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            积分值; 标量被积函数通常在调用方归约之前每个选定实体一个值.
        """
        return self.schema.integral(self.context(), func, q, index)

    def jacobi_matrix(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """参考实体到物理实体映射的 Jacobi 矩阵.

        由几何参考梯度与全部几何节点计算:

        ``J[c, q, d, r] = sum_i X[c, i, d] * dphi_ref[q, i, r]``

        Parameters
        ----------
        bcs : Tensor or tuple of Tensor
            求值点的重心坐标: 单纯形实体为一个张量, 张量积实体为各因子一个张量.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            Jacobi 张量, 通常形状为 ``(NE, NQ, GD, ref_dim)``, ``ref_dim`` 为参考维数.
        """
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        gradients = self.schema.grad_shape_function_reference(bcs)
        points = self.block.positions[self._selected_connectivity(index)]
        return bm.einsum("cij,qir->cqjr", points, gradients)

    def metric_density(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """参考实体到物理实体映射的无符号度量密度.

        参考维数 ``r > 0`` 时为 ``sqrt(det(J^T J))``; 零维参考实体按 Schema 的零维约定, 每个
        采样点返回一个密度. ``J`` 为不超过 3 阶的方阵 (参考维数等于几何维数, 如体单元或平面
        上的面单元) 时两者相等, 改按 ``|det J|`` 的展开式直接计算: 免去构造 ``J^T J`` 与逐个
        小矩阵做 LU 分解的 ``det``, 结果只差舍入.

        Parameters
        ----------
        bcs : Tensor or tuple of Tensor
            求值点的重心坐标: 单纯形实体为一个张量, 张量积实体为各因子一个张量.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            度量密度, 形状 ``(NE, NQ)``.
        """
        jacobian = self.jacobi_matrix(bcs, index=index)
        ref_dim = int(jacobian.shape[-1])
        if ref_dim == 0:
            return bm.ones(jacobian.shape[:2], dtype=jacobian.dtype, device=bm.get_device(jacobian))
        if ref_dim == int(jacobian.shape[-2]) and ref_dim <= 3:
            return bm.abs(_small_square_det(jacobian))
        metric = bm.einsum("cqdr,cqds->cqrs", jacobian, jacobian)
        return bm.sqrt(bm.linalg.det(metric))

    def measure(self, *, index: Index | None = None, q: int | None = None) -> Tensor:
        """选定实体的物理测度.

        由无符号度量密度在参考实体上积分得到. 省略 ``q`` 时使用 Schema 确定的默认几何积分
        阶; 对弯曲几何可提高 ``q`` 检查收敛.

        Parameters
        ----------
        index : Index, optional
            实体子集.
        q : int, optional
            几何积分阶; 为 None 时按默认策略.

        Returns
        -------
        Tensor
            每个选定实体一个测度: 边为长度, 面为面积, 体为体积.
        """
        if self.sector.indptr is not None:
            return self.schema.measure(self.context(), index)  # type: ignore[attr-defined]
        if q is None:
            q = self._default_geometry_quadrature_order()
        quadrature = self.schema.quadrature_formula(q)
        bcs, weights = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        density = self.metric_density(bcs, index=index)
        measure = bm.einsum("cq,q->c", density, weights)
        ref_measure = getattr(self.schema, "ref_measure", None)
        if ref_measure is not None:
            measure = measure * ref_measure
        return measure

    def multi_index_matrix(self, order: int | tuple[int, ...], *, internal: bool = False, tensorprod: bool = True):
        """本实体类型的插值多重指标.

        Parameters
        ----------
        order : int or tuple of int
            多项式次数, 或张量积各方向的次数.
        internal : bool, optional
            为 True 时只返回内部插值点的多重指标. 默认 False.
        tensorprod : bool, optional
            为 True (默认) 时使用插值工具所期望的张量积顺序.

        Returns
        -------
        Tensor
            整数张量, 每行对应一个局部插值点, 每列对应一个局部顶点或张量积坐标.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.multi_index(order, internal=internal, tensorprod=tensorprod)

    def normal(self, bcs: Tensor | tuple[Tensor, ...] | None = None, *, index: Index | None = None) -> Tensor:
        """选定实体的内蕴法向量.

        给出 ``bcs`` 时由全节点 Jacobi 矩阵逐点计算法向; 不给时对仿射分区保留历史上的便捷
        行为.

        Parameters
        ----------
        bcs : Tensor or tuple of Tensor, optional
            求值点的重心坐标.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            法向量; 常见形状为不给 ``bcs`` 时 ``(NE, NN, GD)``, 给出时 ``(NE, Q, NN, GD)``,
            ``NN`` 为法方向个数.

        Raises
        ------
        NotImplementedError
            对约定尚未确定的高余维实体请求逐点法标架.
        """
        if bcs is None:
            return self.schema.normal(self.context(), index)
        if not isinstance(bcs, tuple):
            bcs = (bcs,)

        tangent = self.tangent(bcs, index=index)
        gd = self.geo_dimension()
        td = self.top_dimension()
        if gd == td:
            return bm.zeros(
                tangent.shape[:2] + (0, gd),
                dtype=tangent.dtype,
                device=bm.get_device(tangent),
            )
        if gd == td + 1:
            if td == 1:
                edge = tangent[..., 0, :]
                normal = bm.stack([edge[..., 1], -edge[..., 0]], axis=-1)
                return bm.expand_dims(normal, axis=2)
            if td == 2:
                normal = bm.cross(
                    tangent[..., 0, :],
                    tangent[..., 1, :],
                    axis=-1,
                )
                return bm.expand_dims(normal, axis=2)
        raise NotImplementedError(
            f"pointwise normal frame is not frozen for codimension "
            f"{gd - td} sector {self.sector_id!r}"
        )

    def num_multi_index(self, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        """局部插值多重指标的个数.

        Parameters
        ----------
        order : int or tuple of int
            多项式次数, 或张量积各方向的次数.
        internal : bool, optional
            为 True 时只计内部插值点. 默认 False.

        Returns
        -------
        int
            所求次数下局部多重指标的个数.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.num_multi_index(order, internal=internal)

    def quadrature_formula(self, q: int = 3, qtype: str = "legendre"):
        """参考实体上的积分公式.

        Parameters
        ----------
        q : int, optional
            积分阶, 默认 3.
        qtype : str, optional
            积分公式族, 默认 ``"legendre"``.

        Returns
        -------
        Quadrature
            由实体 Schema 提供的积分公式对象.
        """
        return self.schema.quadrature_formula(q, qtype)

    def set_attribute(self, name: str, value: Any) -> None:
        """在实体分区上设置用户属性.

        Parameters
        ----------
        name : str
            属性名.
        value : Any
            属性值.
        """
        self.sector.attributes[name] = value

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """本分区的几何形函数值.

        输出遵循具体 Schema 的完整局部节点布局, 末轴长度等于 ``schema.number_of_nodes()``,
        也等于 ``indices`` 的连接宽度. 不接受独立的解次数.
        """
        return self.schema.shape_function(bcs)

    def _legacy_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi = None,
    ) -> Tensor:
        """私有的经典任意次基函数入口.

        为经典 ``MeshView`` 的方法保留 FEALPy 历史上 ``shape_function(p)`` 的列约定. 新代码
        应直接使用 :meth:`EntityView.shape_function` 或 ``EntitySchema.lagrange_basis_function``.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if (
            self.schema.type_id == "lagrange_pyramid"
            and _pyramid_geometry_order(p)
        ):
            val = self.schema.shape_function(bcs)
        else:
            val = self.schema.lagrange_basis_function(bcs, p)
        val = _restore_legacy_basis_order(
            val,
            self.schema,
            p,
            gradient=False,
        )
        if variables == "u":
            return val
        elif variables == "x":
            return val[None, ...] # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    def size(self) -> int:
        """本分区的实体个数."""
        return self.schema.size(self.context())

    def tangent(self, bcs: Tensor | tuple[Tensor, ...] | None = None, *, index: Index | None = None) -> Tensor:
        """选定实体的切向量.

        给出 ``bcs`` 时切标架为逐点的全节点 Jacobi 矩阵, 参考轴转置到最后; 不给时对仿射
        分区保留历史上的便捷行为.

        Parameters
        ----------
        bcs : Tensor or tuple of Tensor, optional
            求值点的重心坐标.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            切向量; 常见形状为不给 ``bcs`` 时 ``(NE, TD, GD)``, 给出时 ``(NE, Q, TD, GD)``.
        """
        if bcs is None:
            return self.schema.tangent(self.context(), index)
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        jacobian = self.jacobi_matrix(bcs, index=index)
        return bm.swapaxes(jacobian, -1, -2)

    def to(self, target: int | str | EntityView, idx: int = 0, /) -> Relation:
        """返回本实体分区到目标分区的关系.

        Parameters
        ----------
        target : int, str or EntityView
            目标实体选择器: 拓扑维数、实体类型名或另一个 ``EntityView``.
        idx : int, optional
            ``target`` 选中的维数或实体类型有多个分区时, 目标分区的序号. 默认 0.

        Returns
        -------
        Relation
            含源与目标索引的关系对象, 带数组、COO 等表示的转换工具.
        """
        from ..topology.relation import resolve_relation

        if isinstance(target, EntityView):
            tgt = target.sector_id
        elif isinstance(target, str) and target in self.block.sectors:
            tgt = target
        elif isinstance(target, int):
            top_dim = target + self.top_dimension() + 1 if target < 0 else target
            names = [
                sector.id
                for sector in self.block.sectors.values()
                if sector.schema.top_dim == top_dim
            ]
            if idx >= len(names):
                raise IndexError(f"no target sector at index {idx} for top_dim {top_dim}")
            tgt = names[idx]
        else:
            raise TypeError("EntityView.to expects an EntityView, sector id, or int")
        return resolve_relation(self.block, self.sector_id, tgt)

    def to_ipoint(self, order: int, index: Index | None = None) -> Tensor:
        """实体到全局插值点编号的映射.

        Parameters
        ----------
        order : int
            插值次数.
        index : Index, optional
            实体子集.

        Returns
        -------
        Tensor
            整数张量, 形状 ``(NE, NIP)`` 或对应子集的形状, ``NIP`` 为本实体类型上的局部
            插值点数.
        """
        from ..ipoints import to_ipoint
        from .mesh_view import MeshView
        root_id = self.sector.source_cell_sector_id or self.sector_id
        mapping = to_ipoint(
            MeshView(self.block, cell_sector_id=root_id),
            self,
            order,
        )
        return mapping if index is None else mapping[index]

    def top_dimension(self) -> int:
        """本实体类型的拓扑维数."""
        return self.schema.top_dim
