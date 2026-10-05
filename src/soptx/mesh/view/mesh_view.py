# 移植自 brighthe/fealpy ``fealpy/mesh/view/mesh_view.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""锚定在根单元上的单网格块视图.

``MeshView`` 是单个 ``MeshBlock`` 上的标准计算上下文. 经典的 ``shape_function``、
``grad_shape_function`` 等方法有意保留历史调用约定, 并委托给 ``EntityView`` 的私有
基函数适配器; 这些方法属于经典兼容约定, 而非通用的函数空间协议.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, overload, Self

from ...backend import bm, Tensor
from ..schema import registry as _Reg
from ..storage import MeshBlock
from .entity_view import EntityView


class AdjointRelation(NamedTuple):
    """异类形式的关联关系: 源索引 (同类关系为 None) 与目标索引."""
    src_indices: Tensor | None
    tgt_indices: Tensor


@dataclass
class MeshView:
    """锚定在根单元上的单网格块视图.

    视图锚定在一个 ``root_cell_sector_id`` 上, 其拓扑维数与角色解析都来自该锚点, 而不是
    整个网格块中的最高维数.

    Attributes
    ----------
    block : MeshBlock
        本视图共享的离散存储块.
    cell_sector_id : str or None
        作为锚点的根单元分区 id. 为 None 且网格块恰有一个根单元分区时自动推断;
        多根网格块须显式给出.
    """
    block: MeshBlock
    cell_sector_id: str | None = None

    def __post_init__(self) -> None:
        """校验给出的或推断出的锚点分区 id."""
        if (
            self.cell_sector_id is not None
            and self.cell_sector_id not in self.block.root_cell_sector_ids
        ):
            raise ValueError(
                f"cell_sector_id {self.cell_sector_id!r} is not a root "
                "cell sector"
            )
        if (
            self.cell_sector_id is None
            and len(self.block.root_cell_sector_ids) > 1
        ):
            raise ValueError(
                "MeshView requires an explicit cell_sector_id for a "
                "multi-root MeshBlock"
            )

    def _anchor_sector_id(self) -> str | None:
        """返回本视图唯一的根单元分区 id."""
        if self.cell_sector_id is not None:
            if self.cell_sector_id not in self.block.root_cell_sector_ids:
                raise ValueError(
                    f"cell_sector_id {self.cell_sector_id!r} is not a root "
                    "cell sector"
                )
            return self.cell_sector_id

        roots = self.block.root_cell_sector_ids
        if len(roots) == 1:
            return roots[0]
        if len(roots) > 1:
            raise ValueError(
                "MeshView requires an explicit cell_sector_id for a "
                "multi-root MeshBlock"
            )
        return None

    def _visible_sector_ids(self) -> list[str]:
        """返回本视图可见的分区 id 的逻辑闭包."""
        ids: list[str] = []
        if "node" in self.block.sectors:
            ids.append("node")

        anchor = self._anchor_sector_id()
        if anchor is not None:
            ids.append(anchor)
            ids.extend(
                sector.id
                for sector in self.block.sectors.values()
                if sector.source_cell_sector_id == anchor
                and sector.id != anchor
            )
        else:
            ids.extend(
                sector.id
                for sector in self.block.sectors.values()
                if sector.id != "node"
            )
        return list(dict.fromkeys(ids))

    def _entity_name(self, name_or_topdim: str | int, idx: int) -> str:
        """在视图闭包内解析旧式的实体选择器."""
        visible = self._visible_sector_ids()
        if isinstance(name_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        elif isinstance(name_or_topdim, str):
            if ":" in name_or_topdim:
                topdim, idx = _Reg.string_to_etype_and_idx(
                    name_or_topdim,
                    self.top_dimension(),
                )
            elif name_or_topdim in visible:
                return name_or_topdim
            else:
                topdim = _Reg.etype_to_topdim(
                    name_or_topdim,
                    self.top_dimension(),
                )
        else:
            raise TypeError(
                f"Expected str or int for name_or_topdim, "
                f"got {type(name_or_topdim)}"
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        return names[idx]

    ## 实体取值

    def entity_views(self, etype_or_topdim: str | int, /) -> list[EntityView]:
        """返回某个角色或拓扑维数对应的全部实体视图.

        与 :meth:`entity_view` 不同, 本方法遇到歧义不报错, 返回所有匹配的分区.
        """
        visible = self._visible_sector_ids()
        if isinstance(etype_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                etype_or_topdim,
                self.top_dimension(),
            )
        else:
            if not isinstance(etype_or_topdim, str):
                raise TypeError(
                    "Expected str or int for etype_or_topdim, "
                    f"got {type(etype_or_topdim).__name__}"
                )
            if etype_or_topdim in visible:
                return [EntityView(self.block, etype_or_topdim)]
            topdim = _Reg.etype_to_topdim(
                etype_or_topdim,
                self.top_dimension(),
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        return [EntityView(self.block, name) for name in names]

    def entity_view(self, name_or_topdim: str | int, /) -> EntityView:
        """按无歧义的实体选择器返回一个 :class:`EntityView`.

        具体的分区 id 直接返回; ``"cell"``、``"face"``、``"edge"``、``"node"`` 等角色
        相对当前锚点解析. 一个角色对应多个分区时报错, 而不是静默地选其中之一.

        Parameters
        ----------
        name_or_topdim : str or int
            具体分区 id、角色名, 或相对锚点的拓扑维数.

        Returns
        -------
        EntityView
            选中的实体视图.

        Raises
        ------
        TypeError
            选择器类型不受支持.
        ValueError
            角色未知, 或在本视图中对应零个或多个分区.
        """
        visible = self._visible_sector_ids()
        if isinstance(name_or_topdim, str) and name_or_topdim in visible:
            return EntityView(self.block, name_or_topdim)

        if isinstance(name_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        elif isinstance(name_or_topdim, str):
            topdim = _Reg.etype_to_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        else:
            raise TypeError(
                f"Expected str or int for name_or_topdim, "
                f"got {type(name_or_topdim)}"
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        if not names:
            raise ValueError(
                f"no entity sector resolves role {name_or_topdim!r} "
                "in this view"
            )
        if len(names) > 1:
            raise ValueError(
                f"role {name_or_topdim!r} is ambiguous in this view; "
                f"available sectors: {names}"
            )
        return EntityView(self.block, names[0])

    def entity(self, name_or_topdim: str | int, /) -> Tensor:
        """以经典 FEALPy 布局返回实体数据.

        节点角色返回坐标张量, 其余角色返回整数连接数组.

        Parameters
        ----------
        name_or_topdim : str or int
            具体分区 id、角色名, 或相对锚点的拓扑维数.

        Returns
        -------
        Tensor
            节点为坐标 ``(NN, GD)``, 其余实体为连接数组 ``(NE, NVE)``.
        """
        if name_or_topdim in {"node", "Node", "NODE", 0}:
            return self.block.positions
        return self.entity_view(name_or_topdim).indices

    @property
    def cell_view(self) -> EntityView:
        """返回锚定的单元实体视图."""
        return self.entity_view("cell")

    @property
    def face_view(self) -> EntityView:
        """返回余维 1 的面实体视图.

        Raises
        ------
        ValueError
            面角色在本视图中有歧义.
        """
        return self.entity_view("face")

    @property
    def edge_view(self) -> EntityView:
        """返回边实体视图.

        Raises
        ------
        ValueError
            边角色在本视图中有歧义.
        """
        return self.entity_view("edge")

    @property
    def node_view(self) -> EntityView:
        """返回网格块全局的节点实体视图."""
        return self.entity_view("node")

    @property
    def cell(self) -> Tensor:
        """单元的连接数组."""
        return self.entity("cell")

    @property
    def face(self) -> Tensor:
        """面的连接数组."""
        return self.entity("face")

    @property
    def edge(self) -> Tensor:
        """边的连接数组."""
        return self.entity("edge")

    @property
    def node(self) -> Tensor:
        """节点坐标."""
        return self.entity("node")

    ## 判断

    def is_simplex_mesh(self) -> bool:
        """是否为单纯形网格."""
        for name in self._visible_sector_ids():
            schema_name = self.block.get_sector(name).schema.name
            if schema_name not in {"node", "edge", "tri", "tet"}:
                return False
        return True

    def is_tensor_mesh(self) -> bool:
        """是否为张量积网格."""
        for name in self._visible_sector_ids():
            schema_name = self.block.get_sector(name).schema.name
            if schema_name not in {"node", "edge", "quad", "hex"}:
                return False
        return True

    def is_elemental(self, entity_name: str | None = None, /) -> bool:
        """网格是否只有一种根实体类型."""
        anchor = self._anchor_sector_id()
        is_one_root = anchor is not None
        if entity_name is None:
            return is_one_root
        return is_one_root and anchor == entity_name

    ## 其他取值

    def geo_dimension(self) -> int:
        """网格的几何维数."""
        return int(self.block.positions.shape[1])

    def top_dimension(self) -> int:
        """由锚点单元的 Schema 给出的拓扑维数.

        Returns
        -------
        int
            锚点单元的拓扑维数; 视图没有根单元锚点 (如只有节点的旧式网格块) 时为 -1.
        """
        anchor = self._anchor_sector_id()
        if anchor is None:
            return -1
        return self.block.get_sector(anchor).schema.top_dim

    @property
    def ftype(self):
        """节点坐标的浮点类型."""
        return self.block.positions.dtype

    @property
    def itype(self):
        """规范节点分区使用的整数类型."""
        node = self.block.sectors.get("node")
        if node is not None:
            return node.indices.dtype
        return bm.int32

    @property
    def device(self):
        """节点坐标所在的后端设备 (若可得)."""
        return getattr(self.block.positions, "device", None)

    # [形函数]

    def shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """经过渡的基函数接口计算单元形函数."""
        return self.entity_views(-1)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    cell_shape_function = shape_function

    def face_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """经过渡的基函数接口计算面形函数."""
        return self.entity_views(-2)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def edge_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """经过渡的基函数接口计算边形函数."""
        return self.entity_views(1)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def grad_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """经过渡的基函数接口计算单元形函数的梯度."""
        return self.entity_views(-1)[0]._legacy_grad_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    # [计数与插值]

    def number_of_cells(self) -> int:
        """最高维实体的总数."""
        return sum(sec.size() for sec in self.entity_views(-1))

    def number_of_faces(self) -> int:
        """余维 1 实体的总数."""
        return sum(sec.size() for sec in self.entity_views(-2))

    def number_of_edges(self) -> int:
        """边的总数."""
        return sum(sec.size() for sec in self.entity_views(1))

    def number_of_nodes(self) -> int:
        """节点的总数."""
        return sum(sec.size() for sec in self.entity_views(0))

    def number_of_global_ipoints(self, p) -> int:
        """``p`` 次全局插值点的个数."""
        total = 0
        for name in self._visible_sector_ids():
            sector_view = self.entity_view(name)
            if name == "node":
                cell_view = self.cell_view
                local_vertices = cell_view.schema.local_vertices()
                vertex_ids = bm.reshape(
                    cell_view.indices[:, local_vertices],
                    (-1,),
                )
                vertex_count = int(bm.unique(vertex_ids).shape[0])
                total += sector_view.num_multi_index(
                    p,
                    internal=True,
                ) * vertex_count
            else:
                total += sector_view.num_multi_index(
                    p,
                    internal=True,
                ) * sector_view.size()
        return total

    def number_of_local_ipoints(self, p, iptype="cell") -> int:
        """一个实体上局部插值点的个数."""
        return self.entity_view(iptype).num_multi_index(p)

    def multi_index_matrix(self, p, etype="cell"):
        """某类实体的局部多重指标矩阵."""
        return self.entity_view(etype).multi_index_matrix(p)

    def quadrature_formula(self, q, name_or_topdim="cell", qtype="legendre"):
        """选定实体类型上的积分公式."""
        return self.entity_view(name_or_topdim).quadrature_formula(q, qtype)

    @property
    def localEdge(self):
        """单元参考实体的局部边到顶点表."""
        cell_sec = self.entity_views(-1)[0]
        edge_sec = self.entity_views(1)[0]
        data = cell_sec.schema.local_entity(edge_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    @property
    def localFace(self):
        """单元参考实体的局部面到顶点表."""
        cell_sec = self.entity_views(-1)[0]
        face_sec = self.entity_views(-2)[0]
        data = cell_sec.schema.local_entity(face_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    def cell_to_ipoint(self, p, *, index=None):
        """单元到全局插值点编号的映射."""
        from ..ipoints import to_ipoint
        view = self.entity_views(-1)[0]
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def face_to_ipoint(self, p, *, index=None):
        """面到全局插值点编号的映射."""
        from ..ipoints import to_ipoint
        view = self.entity_views(-2)[0]
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def edge_to_ipoint(self, p, *, index=None):
        """边到全局插值点编号的映射."""
        from ..ipoints import to_ipoint
        view = self.edge_view
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def interpolation_points(self, p, entity=None, index=None):
        """全局插值点的坐标."""
        from ..ipoints import ipoints

        visible = self._visible_sector_ids()
        names: list[str] = []

        def _sector_ids_by_topdim(top_dim: int) -> list[str]:
            return [
                name
                for name in visible
                if self.block.get_sector(name).schema.top_dim == top_dim
            ]

        if entity is None:
            for top_dim in range(self.top_dimension() + 1):
                names.extend(_sector_ids_by_topdim(top_dim))
        else:
            selectors = (
                list(entity)
                if isinstance(entity, Iterable) and not isinstance(entity, str)
                else [entity]
            )
            for selector in selectors:
                if isinstance(selector, int):
                    top_dim = _Reg.ensure_positive_topdim(
                        selector,
                        self.top_dimension(),
                    )
                    names.extend(_sector_ids_by_topdim(top_dim))
                else:
                    names.append(self.entity_view(selector).sector_id)

        ips = ipoints(self, p, names)
        return ips if index is None else ips[index, :]

    # [拓扑]

    def cell_to_face(self, src_id=0, dst_id=0):
        """单元到面的连接映射."""
        cell = self.entity_views("cell")[src_id]
        face = self.entity_views("face")[dst_id]
        return cell.to(face).as_array()

    def cell_to_edge(self, src_id=0, dst_id=0):
        """单元到边的连接映射."""
        cell = self.entity_views("cell")[src_id]
        edge = self.entity_views("edge")[dst_id]
        return cell.to(edge).as_array()

    def face_to_edge(self, src_id=0, dst_id=0):
        """面到边的连接映射."""
        face = self.entity_views("face")[src_id]
        edge = self.entity_views("edge")[dst_id]
        return face.to(edge).as_array()

    def face_to_cell(self, src_id=0, dst_id=0):
        """面与单元的相邻关系, 形状 ``(NF, 4)``.

        各列依次为第一个与最后一个相邻单元 (由关系的 ``unique`` 给出), 以及面在这两个
        单元中的局部序号; 边界面的两个单元相同.
        """
        face = self.entity_views("face")[src_id]
        cell = self.entity_views("cell")[dst_id]
        rel = face.to(cell)
        data = rel.unique
        lidx = rel.local_index
        return bm.stack([data.first, data.last, lidx.floc, lidx.lloc], axis=1)

    def cell_to_edge_sign(self):
        """单元各局部边的定向是否与全局边一致 (局部顶点对等于全局边的顶点对时为 True)."""
        cell_sec = self.entity_views(-1)[0]
        edge_sec = self.entity_views(1)[0]
        c2e = self.cell_to_edge()
        local_pair = cell_sec.indices[:, self.localEdge]
        global_pair = edge_sec.indices[c2e]
        return bm.all(local_pair == global_pair, axis=-1)

    def face_to_edge_sign(self):
        """面各局部边的定向是否与全局边一致."""
        face_sec = self.entity_views(-2)[0]
        edge_sec = self.entity_views(1)[0]
        f2e = self.face_to_edge()
        sign = bm.zeros((face_sec.indices.shape[0], 3), dtype=bm.bool)
        local_f2e = face_sec.schema.local_entity("edge")
        n = [item[0] for item in local_f2e]
        for i in range(len(n)):
            sign[:, i] = face_sec.indices[:, n[i]] == edge_sec.indices[f2e[:, i], 0]
        return sign

    def boundary_cell_flag(self):
        """标记边界单元的布尔掩码."""
        return self.entity_view(-1).boundary().mask

    def boundary_face_flag(self):
        """标记边界面的布尔掩码."""
        return self.entity_view(-2).boundary().mask

    def boundary_edge_flag(self):
        """标记边界边的布尔掩码."""
        return self.entity_view(1).boundary().mask

    def boundary_node_flag(self):
        """标记边界节点的布尔掩码."""
        return self.entity_view(0).boundary().mask

    def boundary_cell_index(self):
        """边界单元的全局编号."""
        return self.entity_view(-1).boundary().index

    def boundary_face_index(self):
        """边界面的全局编号."""
        return self.entity_view(-2).boundary().index

    def boundary_edge_index(self):
        """边界边的全局编号."""
        return self.entity_view(1).boundary().index

    def boundary_node_index(self):
        """边界节点的全局编号."""
        return self.entity_view(0).boundary().index

    # [几何]

    def bc_to_point(self, bc, *, index=None):
        """把重心坐标转换为物理坐标."""
        if not isinstance(bc, tuple):
            bc = (bc,)
        top = sum(b.shape[1] - 1 for b in bc)
        return self.entity_views(top)[0].bc_to_point(bc, index=index)

    def entity_barycenter(self, name_or_topdim, /, *, index=None):
        """选定实体的重心."""
        return self.entity_view(name_or_topdim).barycenter(index=index)

    def entity_measure(self, name_or_topdim, /, *, index=None):
        """选定实体的测度."""
        return self.entity_view(name_or_topdim).measure(index=index)

    def edge_tangent(self, *, index=None):
        """边的切向量."""
        return self.entity_views(1)[0].tangent(index=index)[:, 0, :]

    def edge_unit_tangent(self, *, index=None):
        """边的单位切向量."""
        tangent = self.entity_views(1)[0].tangent(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(tangent, axis=1, keepdims=True)
        return tangent / norm

    def face_normal(self, *, index=None):
        """面的法向量."""
        return self.entity_views(self.top_dimension() - 1)[0].normal(index=index)[:, 0, :]

    def face_unit_normal(self, *, index=None):
        """面的单位法向量."""
        normal = self.entity_views(self.top_dimension() - 1)[0].normal(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(normal, axis=1, keepdims=True)
        return normal / norm

    def grad_lambda(self, index=None, TD=None):
        """重心坐标函数的梯度."""
        if TD is None:
            TD = self.top_dimension()
        if TD == self.top_dimension():
            block = self.entity_views(self.top_dimension())[0]
        elif TD == self.top_dimension() - 1:
            block = self.entity_views(self.top_dimension() - 1)[0]
        elif TD == 1:
            block = self.entity_views(1)[0]
        else:
            raise ValueError(f"Unsupported top dimension: {TD}")
        return block.grad_lambda(index=index)

    def grad_face_lambda(self, index=None):
        """面上重心坐标函数的梯度."""
        return self.grad_lambda(index=index, TD=self.top_dimension() - 1)

    def error(self, f1, f2, /, power=2.0, q=3, *, cell_axis=False, index=None):
        """计算两个函数之差在网格上的范数."""
        return self.entity_views(-1)[0].error(
            f1, f2, power=power, q=q, cell_axis=cell_axis, index=index
        )

    # 修改 (原地)

    def construct(self, exclude: list[str] | None = None) -> Self:
        """构造网格拓扑, 可排除某些实体类型."""
        from ..topology.builder import TopologyBuilder
        TopologyBuilder.construct(self.block, exclude=exclude)
        return self

    def uniform_refine(self, n: int = 1, **kwargs):
        """对网格做若干次一致加密.

        Parameters
        ----------
        n : int, optional
            加密次数, 默认 1.
        **kwargs
            传给加密过程的附加参数.

        Raises
        ------
        NotImplementedError
            网格块有多个根分区.
        """
        if len(self.block.root_cell_sector_ids) > 1:
            raise NotImplementedError(
                "MeshView.uniform_refine() does not support a multi-root "
                "MeshBlock"
            )

        from ..transform import uniform_refine
        return uniform_refine(
            self.block,
            n=n,
            root_id=self._anchor_sector_id(),
            **kwargs,
        )
