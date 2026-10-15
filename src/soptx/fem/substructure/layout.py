"""规则同构子结构的整体布局、有限元上下文与自由度映射."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.functionspace.utils import to_tensor_dof
from soptx.mesh import HexahedronMesh, QuadrangleMesh

from soptx.materials import IsotropicLinearElasticMaterial

from .mesh import resolve_material_hypothesis

class StructuredSubstructureLayout:
    """管理规则子结构整体布局及其局部到全局映射.

    该对象统一保存整体尺寸、子结构划分、全尺度有限元上下文、自由度编号和
    结构化场重排规则, 不负责缩聚刚度装配或线性求解.

    Parameters
    ----------
    full_mesh : HexahedronMesh or QuadrangleMesh, optional
        已创建的规则细网格. 尺寸、剖分与编号须与布局一致, 校验后复用原对象.
        未传入时保留惰性创建行为. 网格接入后不得修改其坐标或连接关系.
    tensor_space : TensorFunctionSpace, optional
        已有连续 Lagrange 位移空间, 须采用 ``shape=(-1, dim)``.
        复用其标量空间与网格; 同时给出 full_mesh 时须为同一对象.
        未显式指定 degree 或 p 时采用该空间的次数, 无空间时缺省为 1.

    Notes
    -----
    全局自由度计数和坐标由 FEM 空间提供. 已接入的空间与网格应保持不变;
    GlobalAssembler 的兼容替换入口会同步清理计数与映射缓存.
    """

    def __init__(
        self,
        domain_size: Union[Tuple[float, ...], float],
        n_sub: Union[Tuple[int, ...], float, int],
        n_fine: Union[Tuple[int, ...], int],
        *args: Any,
        degree: Optional[int] = None,
        p: Optional[int] = None,
        E_base: float = 1.0,
        nu: float = 0.3,
        hypothesis: Optional[str] = None,
        full_mesh: Optional[Any] = None,
        tensor_space: Optional[TensorFunctionSpace] = None,
    ) -> None:
        """
        构造规则子结构布局.

        参数:
            domain_size: 各方向的求解域尺寸元组, 或 2D/3D 标量形式的第一个分量.
            n_sub: 各方向的子结构数元组, 或标量形式的第一个分量.
            n_fine: 单个子结构在各方向的细单元数元组, 或标量形式的第一个分量.
            *args: 标量形式下的其余尺寸, 子结构数和细单元数, 之后可选跟
                ``E_base`` 与 ``nu``; 元组形式下可选跟 ``E_base`` 与 ``nu``.
            degree: 位移插值次数. 未给出时采用 tensor_space.p, 无空间时为 1.
            p: ``degree`` 的别名, 显式给出时优先于 ``degree``.
            E_base: 实体材料的杨氏模量.
            nu: 泊松比.
            hypothesis: 二维为 plane_stress 或 plane_strain, 默认前者; 三维为 3D 或 None.

        异常:
            ValueError: 当参数无法解析为 2D 或 3D 的尺寸, 子结构数与细单元数
                三元组时, 或 ``degree`` 非正时抛出.

        说明:
            求解域固定为以原点为下角的长方体 ``[0, Lx] x [0, Ly] (x [0, Lz])``.
            元组形式与标量形式在 2D 与 3D 下都可用; 标量形式按 ``尺寸, 子结构数,
            细单元数`` 的分组依次给出各方向分量.
        """
        all_args = (domain_size, n_sub, n_fine) + args
        parsed = self._parse_layout(all_args, E_base, nu)
        self.domain_size, self.n_sub, self.n_fine, self.E_base, self.nu = parsed
        self.dim: int = len(self.domain_size)
        self.hypothesis = resolve_material_hypothesis(self.dim, hypothesis)

        if tensor_space is not None and not isinstance(tensor_space, TensorFunctionSpace):
            raise TypeError("tensor_space 必须为 TensorFunctionSpace.")
        requested_degree = p if p is not None else degree
        if requested_degree is None:
            requested_degree = tensor_space.p if tensor_space is not None else 1
        self.degree: int = int(requested_degree)
        if self.degree <= 0:
            raise ValueError(f"degree 必须为正整数; 当前为 {self.degree}.")
        self.p: int = self.degree

        self.total_fine: Tuple[int, ...] = tuple(
            self.n_sub[d] * self.n_fine[d] for d in range(self.dim)
        )
        self.n_full_nodes: Tuple[int, ...] = tuple(
            n * self.degree + 1 for n in self.total_fine
        )
        # 单个子结构在各方向的物理尺寸, 用于由 box_span 反解子结构位置.
        self._sub_size: Tuple[float, ...] = tuple(
            self.domain_size[d] / self.n_sub[d] for d in range(self.dim)
        )

        # 属性别名
        self.Lx = self.domain_size[0]
        self.Ly = self.domain_size[1]
        self.n_sub_x = self.n_sub[0]
        self.n_sub_y = self.n_sub[1]
        self.n_fine_x = self.n_fine[0]
        self.n_fine_y = self.n_fine[1]
        self.total_fine_x = self.total_fine[0]
        self.total_fine_y = self.total_fine[1]
        self.n_full_nodes_x: int = self.n_full_nodes[0]
        self.n_full_nodes_y: int = self.n_full_nodes[1]
        if self.dim == 3:
            self.Lz = self.domain_size[2]
            self.n_sub_z = self.n_sub[2]
            self.n_fine_z = self.n_fine[2]
            self.total_fine_z = self.total_fine[2]
            self.n_full_nodes_z: int = self.n_full_nodes[2]

        # 格点规模用于布局校验; 实际自由度数首次查询 FEM 空间后缓存.
        self._total_full_nodes: Optional[int] = None
        self._total_full_dofs: Optional[int] = None

        self._full_mesh: Optional[Any] = None
        self._material: Optional[Any] = None
        self._sspace_full: Optional[LagrangeFESpace] = None
        self._space_full: Optional[TensorFunctionSpace] = None
        self._node_of_grid: Optional[Any] = None
        # 子结构全局自由度按 (位置, 参考子结构) 缓存; 同一子结构会被多个方法重复查询.
        self._dof_cache: dict = {}
        if tensor_space is not None:
            if full_mesh is not None and full_mesh is not tensor_space.mesh:
                raise ValueError("full_mesh 必须与 tensor_space.mesh 为同一对象.")
            self._set_space_full(tensor_space)
        elif full_mesh is not None:
            self._validate_full_mesh(full_mesh)
            self._full_mesh = full_mesh

    def _invalidate_dof_caches(self) -> None:
        """空间或网格替换后清理计数与布局映射缓存."""
        self._total_full_nodes = None
        self._total_full_dofs = None
        self._node_of_grid = None
        self._dof_cache.clear()

    def _set_space_full(self, space: Optional[TensorFunctionSpace]) -> None:
        """校验并接入已有位移空间, None 清空空间并保留网格."""
        if space is not None:
            if not isinstance(space, TensorFunctionSpace):
                raise TypeError("位移空间必须为 TensorFunctionSpace.")
            scalar = space.scalar_space
            if not isinstance(scalar, LagrangeFESpace) or scalar.ctype != 'C':
                raise ValueError("子结构要求连续 Lagrange 标量空间.")
            if int(space.p) != self.degree:
                raise ValueError("位移空间次数与布局 degree 不一致.")
            if tuple(space.shape) != (-1, self.dim):
                raise ValueError("位移空间须采用 shape=(-1, dim) 的节点分量交错排列.")
            self._validate_full_mesh(space.mesh)
            self._full_mesh = space.mesh
            self._sspace_full = scalar
        else:
            self._sspace_full = None
        self._space_full = space
        self._invalidate_dof_caches()

    def _set_sspace_full(self, space: Optional[LagrangeFESpace]) -> None:
        """兼容标量空间替换, 同步重建张量包装并清理映射缓存."""
        if space is not None and not isinstance(space, LagrangeFESpace):
            raise TypeError("标量空间必须为 LagrangeFESpace.")
        self._set_space_full(
            None if space is None else TensorFunctionSpace(space, shape=(-1, self.dim))
        )

    def _set_full_mesh(self, mesh: Optional[Any]) -> None:
        """兼容网格替换, 清空依赖原网格的空间和映射缓存."""
        if mesh is not None:
            self._validate_full_mesh(mesh)
        self._full_mesh = mesh
        self._set_space_full(None)

    @property
    def total_full_nodes(self) -> int:
        """FEM 标量自由度总数, 首次查询时校验规则格点规模并缓存."""
        if self._total_full_nodes is None:
            count = int(self.sspace_full.number_of_global_dofs())
            expected = 1
            for n in self.n_full_nodes:
                expected *= n
            if count != expected:
                raise ValueError("FEM 标量自由度数与规则格点规模不一致.")
            self._total_full_nodes = count
        return self._total_full_nodes

    @property
    def total_full_dofs(self) -> int:
        """FEM 位移自由度总数, 首次查询后缓存."""
        if self._total_full_dofs is None:
            expected = self.dim * self.total_full_nodes
            count = int(self.space_full.number_of_global_dofs())
            if count != expected:
                raise ValueError("FEM 位移自由度数与布局维数及标量空间不一致.")
            self._total_full_dofs = count
        return self._total_full_dofs

    def _validate_full_mesh(self, mesh: Any) -> None:
        """校验外部细网格的几何、拓扑与规则编号.

        Parameters
        ----------
        mesh : HexahedronMesh or QuadrangleMesh
            原点起始的规则细网格, 须与布局尺寸及剖分一致.

        Raises
        ------
        ValueError
            网格类型、坐标或连接关系与布局不一致.

        Notes
        -----
        复用网格生成器的原始数组校验编号, 不构造第二份网格及面边拓扑.
        """
        from soptx.mesh.generation import Box2d, Box3d

        mesh_type = QuadrangleMesh if self.dim == 2 else HexahedronMesh
        if not isinstance(mesh, mesh_type):
            raise ValueError('full_mesh 类型与子结构布局维数不一致')
        box = [v for length in self.domain_size for v in (0.0, length)]
        generator = Box2d if self.dim == 2 else Box3d
        nodes, cells = generator(box, *self.total_fine, device=mesh.device).initialize()
        order = [0, 1, 3, 2] if self.dim == 2 else [0, 1, 3, 2, 4, 5, 7, 6]
        actual_nodes = mesh.entity('node')
        actual_cells = mesh.entity('cell')
        if actual_nodes.shape != nodes.shape or not bool(
            bm.allclose(actual_nodes, nodes, rtol=1.0e-12, atol=1.0e-12)
        ):
            raise ValueError('full_mesh 坐标或节点编号与布局尺寸及剖分不一致')
        if actual_cells.shape != cells.shape or not bool(bm.all(actual_cells == cells[:, order])):
            raise ValueError('full_mesh 单元连接或编号与布局不一致')

    @staticmethod
    def _parse_layout(
        all_args: Tuple[Any, ...],
        E_base: float,
        nu: float,
    ) -> Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[int, ...], float, float]:
        """解析求解域尺寸, 子结构数, 细单元数与可选的材料参数.

        参数:
            all_args: ``__init__`` 收到的全部位置参数.
            E_base: 关键字形式给出的杨氏模量默认值.
            nu: 关键字形式给出的泊松比默认值.

        返回:
            (domain_size, n_sub, n_fine, E_base, nu): 三个长度相同的布局元组和
                解析后的材料参数.

        异常:
            ValueError: 当维数不是 2 或 3, 标量形式的位置参数不足, 或存在非正的
                尺寸与计数时抛出.
        """
        if isinstance(all_args[0], (tuple, list)):
            domain_size = tuple(float(v) for v in all_args[0])
            n_sub = tuple(int(v) for v in all_args[1])
            n_fine = tuple(int(v) for v in all_args[2])
            rest = all_args[3:]
        else:
            # 标量形式: 尺寸, 子结构数, 细单元数各占 dim 个位置参数.
            n_scalar = len(all_args)
            dim = 3 if n_scalar >= 9 else 2
            if n_scalar < 3 * dim:
                raise ValueError(
                    f"标量形式的 {dim}D 布局需要 {3 * dim} 个位置参数; "
                    f"当前只有 {n_scalar} 个."
                )
            domain_size = tuple(float(v) for v in all_args[0:dim])
            n_sub = tuple(int(v) for v in all_args[dim:2 * dim])
            n_fine = tuple(int(v) for v in all_args[2 * dim:3 * dim])
            rest = all_args[3 * dim:]

        if len(rest) > 0:
            E_base = float(rest[0])
        if len(rest) > 1:
            nu = float(rest[1])

        dim = len(domain_size)
        if dim not in (2, 3):
            raise ValueError(f"仅支持 2D 与 3D 求解域; 当前维数为 {dim}.")
        if len(n_sub) != dim or len(n_fine) != dim:
            raise ValueError(
                f"domain_size, n_sub 与 n_fine 的长度必须一致; "
                f"当前分别为 {dim}, {len(n_sub)}, {len(n_fine)}."
            )
        if any(s <= 0.0 for s in domain_size):
            raise ValueError(f"domain_size 的各分量必须为正; 当前为 {domain_size}.")
        if any(n <= 0 for n in n_sub) or any(n <= 0 for n in n_fine):
            raise ValueError(
                f"n_sub 与 n_fine 的各分量必须为正整数; "
                f"当前为 {n_sub} 与 {n_fine}."
            )
        return domain_size, n_sub, n_fine, E_base, nu

    ### 全尺度有限元对象 (按需构造) ###

    @property
    def full_mesh(self) -> Any:
        """全尺度细网格; 优先复用传入对象, 未传入时首次访问构造."""
        if self._full_mesh is None:
            box = [
                coord
                for d in range(self.dim)
                for coord in (0.0, self.domain_size[d])
            ]
            if self.dim == 2:
                self._full_mesh = QuadrangleMesh.from_box(
                    box=box, nx=self.total_fine[0], ny=self.total_fine[1]
                )
            else:
                self._full_mesh = HexahedronMesh.from_box(
                    box=box,
                    nx=self.total_fine[0],
                    ny=self.total_fine[1],
                    nz=self.total_fine[2],
                )
        return self._full_mesh

    @property
    def material(self) -> Any:
        """全尺度各向同性线弹性材料; 首次访问时构造."""
        if self._material is None:
            if self.dim == 2:
                self._material = IsotropicLinearElasticMaterial(
                    youngs_modulus=self.E_base,
                    poisson_ratio=self.nu,
                    hypothesis=self.hypothesis,
                )
            else:
                self._material = IsotropicLinearElasticMaterial(
                    youngs_modulus=self.E_base, poisson_ratio=self.nu
                )
        return self._material

    @property
    def sspace_full(self) -> LagrangeFESpace:
        """全尺度标量拉格朗日空间; 首次访问时构造."""
        if self._sspace_full is None:
            self._sspace_full = LagrangeFESpace(
                self.full_mesh, p=self.degree, ctype='C'
            )
        return self._sspace_full

    def node_coordinates(self, node_indices: Optional[Any] = None) -> Any:
        """按全局标量自由度编号返回 FEM 插值点坐标.

        Parameters
        ----------
        node_indices : array_like, optional
            全局标量自由度编号. 省略时返回全部插值点.

        Returns
        -------
        coordinates : array_like
            节点坐标, 形状为 ``(N, dim)``.

        Notes
        -----
        所有次数均复用标量空间的插值点接口, 首次访问可能创建全尺度空间.
        """
        points = self.sspace_full.interpolation_points()
        if node_indices is None:
            return points
        indices = bm.asarray(
            node_indices, dtype=bm.int64, device=self.sspace_full.device
        )
        return points[indices]

    @property
    def space_full(self) -> TensorFunctionSpace:
        """全尺度张量位移空间; 首次访问时构造."""
        if self._space_full is None:
            self._space_full = TensorFunctionSpace(
                self.sspace_full, shape=(-1, self.dim)
            )
        return self._space_full

    ### 全局自由度映射 ###

    def _node_of_grid_index(self) -> Any:
        """返回规则格点的 C 序编号到全局标量自由度编号的映射.

        Returns
        -------
        node_of_grid : array_like
            形状为 ``(total_full_nodes,)`` 的整数置换.

        Raises
        ------
        RuntimeError
            高阶插值点数量、坐标或格点覆盖与规则布局不一致.

        Notes
        -----
        Q1 保留恒等映射. 高阶插值点按实际有限元空间编号, 通过坐标反解
        规则格点下标后构造逆向置换. 显式设置的非空缓存仍优先使用.
        """
        if self._node_of_grid is not None:
            return self._node_of_grid

        if self.degree == 1:
            self._node_of_grid = bm.arange(self.total_full_nodes, dtype=bm.int64)
            return self._node_of_grid

        points = self.sspace_full.interpolation_points()
        if tuple(points.shape) != (self.total_full_nodes, self.dim):
            raise RuntimeError("全局插值点数量或维数与规则布局不一致.")
        if not bool(bm.all(bm.isfinite(points))):
            raise RuntimeError("全局插值点坐标包含非有限值.")

        device = self.full_mesh.device
        linear_index = bm.zeros(
            (self.total_full_nodes,), dtype=bm.int64, device=device
        )
        for d in range(self.dim):
            step = self.domain_size[d] / (self.total_fine[d] * self.degree)
            grid_coordinate = points[:, d] / step
            rounded = bm.round(grid_coordinate)
            if not bool(bm.all(bm.abs(grid_coordinate - rounded) <= 1.0e-6)):
                raise RuntimeError("全局插值点未落在预期的规则高阶格点上.")
            if not bool(bm.all(
                (rounded >= 0) & (rounded < self.n_full_nodes[d])
            )):
                raise RuntimeError("全局插值点的规则格点下标越界.")
            grid_index = bm.astype(rounded, bm.int64)
            linear_index = linear_index * self.n_full_nodes[d] + grid_index

        node_of_grid = bm.argsort(linear_index)
        expected = bm.arange(self.total_full_nodes, dtype=bm.int64, device=device)
        if not bool(bm.all(linear_index[node_of_grid] == expected)):
            raise RuntimeError("全局插值点未能完整且唯一地覆盖规则高阶格点.")

        self._node_of_grid = node_of_grid
        return self._node_of_grid

    def get_substructure_global_dofs(self, *args: Any) -> Any:
        """
        将子结构的局部自由度映射到全尺度网格的全局自由度.

        参数:
            *args: ``(sub_pos, sub_mesh)``, 2D 的 ``(sx, sy, sub_mesh)`` 或 3D 的
                ``(sx, sy, sz, sub_mesh)``.

        返回:
            sub_global_dofs: 形状 ``(n_dof,)`` 的全局自由度编号, 第 ``j`` 项对应
                子结构的第 ``j`` 个局部自由度.

        异常:
            ValueError: 当子结构位置越界时抛出.

        说明:
            子结构节点的规则格点下标加上位置偏移后, 经全局格点到有限元编号
            的映射转换为全局节点编号. Q1 使用恒等映射, 高阶使用实际插值点
            生成的置换. 结果按位置与参考子结构缓存.
        """
        if len(args) == 2:
            sub_pos, sub_mesh = args[0], args[1]
        elif len(args) == 3:
            sub_pos, sub_mesh = (args[0], args[1]), args[2]
        else:
            sub_pos, sub_mesh = (args[0], args[1], args[2]), args[3]

        pos = tuple(int(p) for p in sub_pos)
        for d in range(self.dim):
            if not 0 <= pos[d] <= self.n_sub[d] - 1:
                raise ValueError(
                    f"子结构位置 {pos} 的第 {d} 个分量越界; "
                    f"该方向共有 {self.n_sub[d]} 个子结构."
                )

        prototype = getattr(sub_mesh, 'prototype', sub_mesh)
        key = (pos, id(prototype))
        cached = self._dof_cache.get(key)
        if cached is not None:
            return cached[1]

        node_of_grid = self._node_of_grid_index()
        sub_degree = getattr(prototype, 'degree', getattr(prototype, 'p', self.degree))
        offset = bm.asarray(
            [pos[d] * self.n_fine[d] * sub_degree for d in range(self.dim)], dtype=bm.int64
        )
        grid_index = sub_mesh.node_grid_index + offset

        linear_index: Any = bm.zeros(
            (grid_index.shape[0],), dtype=bm.int64
        )
        for d in range(self.dim):
            linear_index = linear_index * self.n_full_nodes[d] + grid_index[:, d]
        global_nodes = node_of_grid[linear_index]

        # 与 TensorFunctionSpace 共用标量到张量编号的展开规则.
        space = self.space_full
        sub_global_dofs = to_tensor_dof(
            global_nodes[None, :], space.dof_numel,
            self.total_full_nodes, space.dof_priority,
        )[0]
        # 同时持有参考子结构的引用, 避免它被回收后 id 复用导致缓存串号.
        self._dof_cache[key] = (prototype, sub_global_dofs)
        return sub_global_dofs

    def _substructure_positions(self, sub_meshes: Sequence[Any]) -> List[Tuple[int, ...]]:
        """由各子结构的 ``box_span`` 反解它们在子结构网格中的整数位置.

        参数:
            sub_meshes: 子结构列表.

        返回:
            positions: 每个子结构的整数位置元组, 与 ``sub_meshes`` 同序.

        异常:
            ValueError: 当子结构维数与求解域不符, 位置越界, 或两个子结构落在同一
                位置时抛出.

        说明:
            位置来自子结构自身记录的几何跨度, 因此装配不要求 ``sub_meshes`` 按
            任何特定次序排列.
        """
        positions: List[Tuple[int, ...]] = []
        occupied: dict = {}
        for idx, sub_mesh in enumerate(sub_meshes):
            span = sub_mesh.box_span
            if len(span) != self.dim:
                raise ValueError(
                    f"第 {idx} 个子结构是 {len(span)}D, 与 {self.dim}D 求解域不符."
                )
            pos = tuple(
                int(round(span[d][0] / self._sub_size[d])) for d in range(self.dim)
            )
            for d in range(self.dim):
                if not 0 <= pos[d] <= self.n_sub[d] - 1:
                    raise ValueError(
                        f"第 {idx} 个子结构的跨度 {tuple(span)} 反解出位置 {pos}, "
                        f"第 {d} 个分量越界; 该方向共有 {self.n_sub[d]} 个子结构."
                    )
            if pos in occupied:
                raise ValueError(
                    f"第 {idx} 个子结构与第 {occupied[pos]} 个落在同一位置 {pos}."
                )
            occupied[pos] = idx
            positions.append(pos)
        return positions

    def substructure_positions(
        self,
        sub_meshes: Sequence[Any],
    ) -> Tuple[Tuple[int, ...], ...]:
        """返回各子结构在规则分块网格中的位置.

        返回顺序与 ``sub_meshes`` 一致. 该公共只读接口供分析器构造局部到全局
        位移映射, 避免调用方依赖私有的 ``_substructure_positions``.
        """
        return tuple(self._substructure_positions(sub_meshes))

    def interface_indices(
        self,
        sub_meshes: Sequence[Any],
        interface_global_dofs: Any,
    ) -> Any:
        """给出各子结构接口自由度在接口系统中的编号.

        参数:
            sub_meshes: 子结构列表.
            interface_global_dofs: 升序排列的接口全局自由度.

        返回:
            b_interface: 形状 ``(B, n_b)`` 的接口编号, 第 ``s`` 行按第 ``s`` 个
                子结构的 ``b_dofs`` 顺序排列.

        异常:
            ValueError: 当各子结构的接口自由度数不一致时抛出.
        """
        n_b = int(sub_meshes[0].n_b)
        rows = []
        for pos, sub_mesh in zip(self._substructure_positions(sub_meshes), sub_meshes):
            if int(sub_mesh.n_b) != n_b:
                raise ValueError(
                    "批量装配要求全部子结构同构; "
                    f"接口自由度数出现 {n_b} 与 {int(sub_mesh.n_b)} 两种."
                )
            b_global = self.get_substructure_global_dofs(pos, sub_mesh)[sub_mesh.b_dofs]
            # interface_global_dofs 升序, 二分定位取代全局到接口的字典反查.
            rows.append(bm.searchsorted(interface_global_dofs, b_global))
        return bm.stack(rows, axis=0)

    def build_interface_dofs(self, sub_meshes: List[Any]) -> Any:
        """
        收集全部子结构接口自由度的全局编号, 构成接口系统的自由度集合.

        参数:
            sub_meshes: 子结构列表.

        返回:
            interface_global_dofs: 升序去重后的接口全局自由度, 形状
                ``(n_interface,)``.
        """
        b_global_all = []
        for pos, sub_mesh in zip(self._substructure_positions(sub_meshes), sub_meshes):
            sub_global_dofs = self.get_substructure_global_dofs(pos, sub_mesh)
            b_global_all.append(sub_global_dofs[sub_mesh.b_dofs])
        return bm.unique(bm.concat(b_global_all))

    @property
    def n_macro_nodes_per_dir(self) -> Tuple[int, ...]:
        """宏观粗网格各方向节点数: n_sub[d] + 1."""
        return tuple(n + 1 for n in self.n_sub)

    @property
    def total_macro_nodes(self) -> int:
        """宏观粗网格总节点数."""
        n = 1
        for count in self.n_macro_nodes_per_dir:
            n *= count
        return n

    @property
    def total_macro_dofs(self) -> int:
        """宏观粗网格总自由度数: dim * total_macro_nodes."""
        return self.dim * self.total_macro_nodes

    def macro_node_coordinates(self, node_indices: Optional[Any] = None) -> Any:
        """解析计算宏观粗网格节点的物理坐标, 形状 ``(N, dim)``."""
        if node_indices is None:
            node_indices = bm.arange(self.total_macro_nodes, dtype=bm.int64)
        else:
            node_indices = bm.asarray(node_indices, dtype=bm.int64)

        if self.dim == 2:
            ny1 = self.n_macro_nodes_per_dir[1]
            ix = node_indices // ny1
            iy = node_indices % ny1
            hx = self.domain_size[0] / self.n_sub[0]
            hy = self.domain_size[1] / self.n_sub[1]
            x = bm.astype(ix, bm.float64) * hx
            y = bm.astype(iy, bm.float64) * hy
            return bm.stack([x, y], axis=-1)
        else:
            ny1 = self.n_macro_nodes_per_dir[1]
            nz1 = self.n_macro_nodes_per_dir[2]
            nyz = ny1 * nz1
            ix = node_indices // nyz
            rem = node_indices % nyz
            iy = rem // nz1
            iz = rem % nz1
            hx = self.domain_size[0] / self.n_sub[0]
            hy = self.domain_size[1] / self.n_sub[1]
            hz = self.domain_size[2] / self.n_sub[2]
            x = bm.astype(ix, bm.float64) * hx
            y = bm.astype(iy, bm.float64) * hy
            z = bm.astype(iz, bm.float64) * hz
            return bm.stack([x, y, z], axis=-1)

    def macro_corner_indices(self, sub_meshes: Sequence[Any]) -> Any:
        """给出各子结构角节点在宏观粗网格系统中的全局自由度编号, 形状 ``(B, dim * 2**dim)``.

        参数:
            sub_meshes: 子结构列表.
        """
        positions = self._substructure_positions(sub_meshes)
        rows = []

        if self.dim == 2:
            ny1 = self.n_macro_nodes_per_dir[1]
            for sx, sy in positions:
                c_nodes = [
                    sx * ny1 + sy,
                    (sx + 1) * ny1 + sy,
                    (sx + 1) * ny1 + (sy + 1),
                    sx * ny1 + (sy + 1),
                ]
                dofs = []
                for node in c_nodes:
                    for k in range(2):
                        dofs.append(2 * node + k)
                rows.append(bm.asarray(dofs, dtype=bm.int64))
        else:
            ny1 = self.n_macro_nodes_per_dir[1]
            nz1 = self.n_macro_nodes_per_dir[2]
            nyz = ny1 * nz1
            for sx, sy, sz in positions:
                c_nodes = [
                    sx * nyz + sy * nz1 + sz,
                    sx * nyz + sy * nz1 + (sz + 1),
                    sx * nyz + (sy + 1) * nz1 + sz,
                    sx * nyz + (sy + 1) * nz1 + (sz + 1),
                    (sx + 1) * nyz + sy * nz1 + sz,
                    (sx + 1) * nyz + sy * nz1 + (sz + 1),
                    (sx + 1) * nyz + (sy + 1) * nz1 + sz,
                    (sx + 1) * nyz + (sy + 1) * nz1 + (sz + 1),
                ]
                dofs = []
                for node in c_nodes:
                    for k in range(3):
                        dofs.append(3 * node + k)
                rows.append(bm.asarray(dofs, dtype=bm.int64))

        return bm.stack(rows, axis=0)

    def project_global_vector(self, interface_dofs: TensorLike, global_vector: Any) -> Any:
        """提取全局向量的完整接口分量.

        Parameters
        ----------
        interface_dofs : TensorLike
            完整细网格接口的全局自由度编号, 升序且无重复, 形状 (n_interface,).
        global_vector : array_like
            全局自由度向量, 长度为 total_full_dofs.

        Returns
        -------
        interface_vector : TensorLike
            与 interface_dofs 顺序一致的接口分量, 形状 (n_interface,).

        Raises
        ------
        ValueError
            全局向量长度与全局自由度数不一致.
        """
        vector = bm.asarray(global_vector, dtype=bm.float64)
        if len(vector) != self.total_full_dofs:
            raise ValueError(
                f"global_vector 的长度必须等于全局自由度数 {self.total_full_dofs}; "
                f"当前为 {len(vector)}."
            )
        return vector[interface_dofs]

    def project_global_dofs(self, interface_dofs: TensorLike, global_dofs: Any) -> Any:
        """把全局自由度集合转换为接口系统中的编号.

        Parameters
        ----------
        interface_dofs : TensorLike
            完整细网格接口的全局自由度编号, 升序且无重复, 形状 (n_interface,).
        global_dofs : array_like
            待转换的全局自由度编号, 不在接口上的分量被丢弃.

        Returns
        -------
        local_interface_dofs : TensorLike
            升序去重的接口系统编号, 即 interface_dofs 中的位置.
        """
        dofs = bm.asarray(global_dofs, dtype=bm.int64)
        on_interface = bm.isin(dofs, interface_dofs)
        return bm.unique(bm.searchsorted(interface_dofs, dofs[on_interface]))

    def to_node_grid(self, node_values: Any) -> Any:
        """把按全局节点编号排列的标量场重排为结构化网格形状.

        参数:
            node_values: 按全局节点编号排列的标量场, 形状
                ``(total_full_nodes,)``. 位移的某个分量可由 ``U[k::dim]`` 取出.

        返回:
            grid: 形状为 ``n_full_nodes`` 的结构化标量场, 下标按各方向的整数
                网格位置排列.

        异常:
            ValueError: 当 ``node_values`` 长度与全局节点数不一致时抛出.

        说明:
            按规则格点到全局标量自由度编号的映射提取数值再重排.
            Q1 使用恒等映射, 高阶使用实际插值点生成的置换.
        """
        values = bm.asarray(node_values)
        if len(values) != self.total_full_nodes:
            raise ValueError(
                f"node_values 的长度必须等于全局节点数 {self.total_full_nodes}; "
                f"当前为 {len(values)}."
            )
        return bm.reshape(values[self._node_of_grid_index()], self.n_full_nodes)

    def split_global_cell_field(self, global_field: Any) -> Any:
        """把全局结构化 cell 场拆成 block-major 子结构局部网格场.

        参数:
            global_field: 形状为 ``total_fine`` 的结构化场, 或其 C 序展平向量.

        返回:
            sub_fields: 形状 ``(B, *n_fine)`` 的局部网格场. ``B`` 按 x 优先
                字典序排列, 即 2D 下 ``sub_id = sx * n_sub_y + sy``, 3D 下
                ``sub_id = (sx * n_sub_y + sy) * n_sub_z + sz``.

        异常:
            ValueError: 当输入形状既不是 ``total_fine`` 也不是对应展平向量时抛出.
        """
        field = bm.asarray(global_field)
        n_total = 1
        for count in self.total_fine:
            n_total *= count

        if tuple(field.shape) == self.total_fine:
            grid = field
        elif field.ndim == 1 and field.shape[0] == n_total:
            grid = bm.reshape(field, self.total_fine)
        else:
            raise ValueError(
                f"全局 cell 场形状 {tuple(field.shape)} 无法解释: "
                f"应为 {self.total_fine} 或 ({n_total},)."
            )

        if self.dim == 2:
            blocked = bm.reshape(
                grid,
                (self.n_sub[0], self.n_fine[0],
                 self.n_sub[1], self.n_fine[1]),
            )
            blocked = bm.permute_dims(blocked, (0, 2, 1, 3))
        else:
            blocked = bm.reshape(
                grid,
                (
                    self.n_sub[0], self.n_fine[0],
                    self.n_sub[1], self.n_fine[1],
                    self.n_sub[2], self.n_fine[2],
                ),
            )
            blocked = bm.permute_dims(blocked, (0, 2, 4, 1, 3, 5))

        n_sub_total = 1
        for count in self.n_sub:
            n_sub_total *= count
        return bm.reshape(blocked, (n_sub_total,) + self.n_fine)

    def merge_substructure_cell_field(self, sub_fields: Any) -> Any:
        """把 block-major 子结构局部网格场拼接为全局结构化 cell 场.

        参数:
            sub_fields: 形状 ``(B, *n_fine)`` 的局部网格场. 子结构必须按 x 优先
                字典序排列; 该顺序与 ``split_global_cell_field`` 返回值一致.

        返回:
            global_field: 形状为 ``total_fine`` 的全局场.

        异常:
            ValueError: 当输入形状不是 ``(B, *n_fine)`` 时抛出.
        """
        field = bm.asarray(sub_fields)
        n_sub_total = 1
        for count in self.n_sub:
            n_sub_total *= count
        expected = (n_sub_total,) + self.n_fine
        if tuple(field.shape) != expected:
            raise ValueError(
                f"子结构 cell 场形状 {tuple(field.shape)} 必须为 {expected}."
            )

        if self.dim == 2:
            field = bm.reshape(
                field,
                (self.n_sub[0], self.n_sub[1], self.n_fine[0], self.n_fine[1]),
            )
            field = bm.permute_dims(field, (0, 2, 1, 3))
            return bm.reshape(field, self.total_fine)

        field = bm.reshape(
            field,
            (
                self.n_sub[0], self.n_sub[1], self.n_sub[2],
                self.n_fine[0], self.n_fine[1], self.n_fine[2],
            ),
        )
        field = bm.permute_dims(field, (0, 3, 1, 4, 2, 5))
        return bm.reshape(field, self.total_fine)

    def reconstruct_global_field(self, sub_fields: Union[List[Any], Any]) -> Any:
        """把各子结构局部网格场拼接为全局结构化 cell 场.

        说明:
            这是 ``merge_substructure_cell_field`` 的兼容入口. 新代码应使用名称
            更明确的成对 API ``split_global_cell_field`` 与
            ``merge_substructure_cell_field``.
        """
        return self.merge_substructure_cell_field(sub_fields)

    # 向后兼容别名
    reconstruct_full_density = reconstruct_global_field
    assemble_full_density = reconstruct_global_field
