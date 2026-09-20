"""全局接口系统装配.

负责规则排布的同构子结构的自由度映射, 缩聚刚度散加与全场位移恢复. 装配器
只表达装配结果, 不携带载荷, 边界条件或求解策略, 这些由调用方编排.

子结构在全局网格中的位置由其 ``box_span`` 反解得到, 全局节点编号由节点坐标
反解得到, 因此装配不依赖子结构列表的排列次序, 也不依赖网格生成器的节点编号
约定. 缩聚结果既可以按子结构逐个给出, 也可以按批量前导维 ``B`` 一次给出.
"""

from dataclasses import dataclass
import hashlib
from typing import Tuple, List, Union, Any, Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.sparse import coo_matrix

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor, CSRTensor
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.fem.matrix.csr_pattern import (
    CSRPattern,
    assemble_csr_chunks,
    build_csr_pattern_from_dofmap,
)

from .streaming import TraceStiffnessBatch
from .traces import FullTraceBasis, LinearCornerTraceBasis, TraceBasis


@dataclass(frozen=True)
class InterfaceSystem:
    """缩聚后的接口刚度矩阵及其全局自由度映射.

    该对象只表达装配结果, 不携带载荷, 边界条件或求解策略.

    属性:
        stiffness: 接口刚度矩阵, 形状 ``(n_interface, n_interface)``. 采用
            FEALPy ``CSRTensor`` 格式, 供 ``soptx.solvers.spsolve`` 直接求解.
        global_dofs: 接口自由度对应的全局自由度编号, 升序排列, 形状
            ``(n_interface,)``. 升序是契约的一部分, 全局到接口的反查依赖
            二分定位而非字典.
    """

    stiffness: CSRTensor
    global_dofs: Any


class GlobalAssembler:
    """
    组装子结构缩聚后的全局接口系统, 并管理接口与全局自由度映射.

    载荷, 边界条件, 线性求解和内部位移恢复的流程编排由调用方负责, 避免将具体
    工程物理模型或求解策略耦合到装配器.

    全尺度细网格及其函数空间按需构造: 只查询自由度规模不会触发建网格, 只有需要
    全局节点编号或全尺度有限元对象时才实际构造.
    """

    def __init__(
        self,
        domain_size: Union[Tuple[float, ...], float],
        n_sub: Union[Tuple[int, ...], float, int],
        n_fine: Union[Tuple[int, ...], int],
        *args: Any,
        degree: int = 1,
        p: Optional[int] = None,
        E_base: float = 1.0,
        nu: float = 0.3,
    ) -> None:
        """
        构造全局接口系统装配器.

        参数:
            domain_size: 各方向的求解域尺寸元组, 或 2D/3D 标量形式的第一个分量.
            n_sub: 各方向的子结构数元组, 或标量形式的第一个分量.
            n_fine: 单个子结构在各方向的细单元数元组, 或标量形式的第一个分量.
            *args: 标量形式下的其余尺寸, 子结构数和细单元数, 之后可选跟
                ``E_base`` 与 ``nu``; 元组形式下可选跟 ``E_base`` 与 ``nu``.
            degree: 有限元位移空间的多项式插值次数, 缺省为 ``1``.
            p: ``degree`` 的别名, 显式给出时优先于 ``degree``.
            E_base: 实体材料的杨氏模量.
            nu: 泊松比.

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

        self.degree: int = int(p if p is not None else degree)
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

        # 自由度规模由结构化划分直接算出, 不依赖全尺度网格.
        n_nodes = 1
        for n in self.n_full_nodes:
            n_nodes *= n
        self.total_full_nodes: int = n_nodes
        self.total_full_dofs: int = self.dim * n_nodes

        self._full_mesh: Optional[Any] = None
        self._material: Optional[Any] = None
        self._sspace_full: Optional[LagrangeFESpace] = None
        self._space_full: Optional[TensorFunctionSpace] = None
        self._node_of_grid: Optional[Any] = None
        # 子结构全局自由度按 (位置, 参考子结构) 缓存; 同一子结构会被多个方法重复查询.
        self._dof_cache: dict = {}
        self._interface_pattern_cache: Optional[Tuple[Any, CSRPattern]] = None

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
        """全尺度细网格; 首次访问时构造."""
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
                    hypothesis='plane_stress',
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
        """解析计算结构化全网格节点的物理坐标, 形状 ``(N, dim)``.

        说明:
            直接基于网格步长解析计算, 彻底避免为了查询坐标而构造全尺度网格与空间.
        """
        if node_indices is None:
            node_indices = bm.arange(self.total_full_nodes, dtype=bm.int64)
        else:
            node_indices = bm.asarray(node_indices, dtype=bm.int64)

        if self.dim == 2:
            ny1 = self.n_full_nodes[1]
            ix = node_indices // ny1
            iy = node_indices % ny1
            hx = self.domain_size[0] / (self.total_fine[0] * self.degree)
            hy = self.domain_size[1] / (self.total_fine[1] * self.degree)
            x = bm.astype(ix, bm.float64) * hx
            y = bm.astype(iy, bm.float64) * hy
            return bm.stack([x, y], axis=-1)
        else:
            ny1 = self.n_full_nodes[1]
            nz1 = self.n_full_nodes[2]
            nyz = ny1 * nz1
            ix = node_indices // nyz
            rem = node_indices % nyz
            iy = rem // nz1
            iz = rem % nz1
            hx = self.domain_size[0] / (self.total_fine[0] * self.degree)
            hy = self.domain_size[1] / (self.total_fine[1] * self.degree)
            hz = self.domain_size[2] / (self.total_fine[2] * self.degree)
            x = bm.astype(ix, bm.float64) * hx
            y = bm.astype(iy, bm.float64) * hy
            z = bm.astype(iz, bm.float64) * hz
            return bm.stack([x, y, z], axis=-1)

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
        """返回结构化高阶网格下标到全局节点编号的映射, 形状 ``(total_full_nodes,)``.

        说明:
            FEALPy 标准结构化网格节点按 C 序字典序排布 (x 优先, y 次之, z 内层),
            节点下标与线性编号自然恒等, 直接由 ``bm.arange`` 给出, 彻底避免构造全尺度
            网格与插值坐标所造成的数十 GB 内存开销.
        """
        if self._node_of_grid is not None:
            return self._node_of_grid

        self._node_of_grid = bm.arange(self.total_full_nodes, dtype=bm.int64)
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
            映射由子结构自身的节点结构化下标加上子结构偏移得到, 再经全网格的
            下标反查表转成全局节点编号, 不依赖任何编号次序约定. 结果按位置与
            参考子结构缓存.
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

        # 局部自由度按 dim * node + k 排布, 展平后与局部自由度编号逐项对应.
        components = bm.arange(self.dim, dtype=bm.int64)
        sub_global_dofs = bm.reshape(
            self.dim * global_nodes[:, None] + components[None, :], (-1,)
        )
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

    def assemble_macro_system(
        self,
        sub_meshes: List[Any],
        Ks_macro_batch: Any,
    ) -> InterfaceSystem:
        """装配 Huang 2023 线性边界变形降维后的宏观粗网格系统 (Huang 2023 式 16).

        参数:
            sub_meshes: 子结构列表.
            Ks_macro_batch: 降维缩聚刚度矩阵, 形状 ``(B, l, l)``, 其中 ``l = dim * 2**dim``
                (3D 为 24, 2D 为 8).

        返回:
            system: 宏观刚度矩阵与全局宏观自由度映射.
        """
        c_macro = self.macro_corner_indices(sub_meshes)
        n_batch, l_dim = c_macro.shape
        shape = (n_batch, l_dim, l_dim)

        rows = bm.reshape(bm.broadcast_to(c_macro[:, :, None], shape), (-1,))
        cols = bm.reshape(bm.broadcast_to(c_macro[:, None, :], shape), (-1,))
        vals = bm.reshape(Ks_macro_batch, (-1,))

        indices = bm.stack([rows, cols], axis=0)
        coo = COOTensor(
            indices=indices,
            values=vals,
            spshape=(self.total_macro_dofs, self.total_macro_dofs),
        )
        K_macro = coo.coalesce().tocsr()
        macro_global_dofs = bm.arange(self.total_macro_dofs, dtype=bm.int64)

        return InterfaceSystem(stiffness=K_macro, global_dofs=macro_global_dofs)

    def build_linear_corner_projection(
        self,
        sub_meshes: Sequence[Any],
        interface_system: InterfaceSystem,
        trace_basis: Any,
    ) -> Any:
        """构造全局角点线性迹投影 ``u_interface = P q``.

        参数:
            sub_meshes: 参与接口装配的子结构.
            interface_system: 完整接口系统或至少具有 ``global_dofs`` 属性的视图.
            trace_basis: 局部角点迹基, 其矩阵把局部宏观角点自由度映射为
                完整局部边界位移.

        返回:
            scipy.sparse.csr_matrix: 形状为
                ``(n_interface_dofs, total_macro_dofs)`` 的全局投影 ``P``.

        异常:
            ValueError: 当局部维度不匹配, 投影未覆盖完整接口, 或相邻子结构
                在共享接口上给出不一致的插值时抛出.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")

        interface_dofs = bm.asarray(interface_system.global_dofs, dtype=bm.int64)
        boundary = np.asarray(
            bm.to_numpy(self.interface_indices(sub_meshes, interface_dofs)),
            dtype=np.int64,
        )
        corners = np.asarray(
            bm.to_numpy(self.macro_corner_indices(sub_meshes)), dtype=np.int64
        )
        local = np.asarray(bm.to_numpy(trace_basis.matrix), dtype=np.float64)
        if local.shape != (boundary.shape[1], corners.shape[1]):
            raise ValueError(
                "trace_basis.matrix 的形状必须为 "
                f"({boundary.shape[1]}, {corners.shape[1]}); 当前为 {local.shape}."
            )

        row, col = np.nonzero(local)
        n_batch, n_boundary = boundary.shape
        candidates = coo_matrix(
            (
                np.tile(local[row, col], n_batch),
                (
                    (np.arange(n_batch)[:, None] * n_boundary + row).ravel(),
                    corners[:, col].ravel(),
                ),
            ),
            shape=(n_batch * n_boundary, self.total_macro_dofs),
        ).tocsr()
        global_rows, first = np.unique(boundary.ravel(), return_index=True)
        if not np.array_equal(global_rows, np.arange(len(interface_dofs))):
            raise ValueError("角点线性迹投影未覆盖完整接口.")
        projection = candidates[first].tocsr()
        difference = candidates - projection[boundary.ravel()]
        if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-12:
            raise ValueError("相邻子结构在共享接口上给出了不一致的角点插值.")
        return projection

    def assemble_macro_system_batches(
        self,
        sub_meshes: Sequence[Any],
        stiffness_batches: Iterable[Any],
    ) -> InterfaceSystem:
        """由连续的迹空间刚度批次流式装配宏观粗网格系统.

        参数:
            sub_meshes: 按批次编号排列的全部子结构.
            stiffness_batches: 连续覆盖全部子结构的批次迭代器. 每项提供
                ``start``, ``end`` 和 ``stiffness`` 属性, 其中刚度形状为
                ``(end - start, l, l)``.

        返回:
            system: 宏观刚度矩阵与全局宏观自由度映射.

        异常:
            ValueError: 当子结构为空, 批次区间不连续, 未完整覆盖子结构, 或
                批次刚度形状与宏观角点自由度不一致时抛出.

        说明:
            每批先独立生成规范 CSR 矩阵, 再与当前全局 CSR 做稀疏加法. 因此
            峰值内存由当前全局稀疏矩阵和单个批次决定, 不保存完整的
            ``K_trace_batch`` 或全量未合并 COO 三元组.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")

        c_macro = self.macro_corner_indices(sub_meshes)
        n_sub_total, n_trace = c_macro.shape
        next_start = 0
        K_macro: Optional[CSRTensor] = None

        for batch in stiffness_batches:
            start = int(batch.start)
            end = int(batch.end)
            stiffness = bm.asarray(batch.stiffness)

            if start != next_start or end <= start or end > n_sub_total:
                raise ValueError(
                    "stiffness_batches 必须以连续半开区间完整覆盖子结构; "
                    f"期望 start={next_start}, 当前区间为 [{start}, {end})."
                )

            expected_shape = (end - start, n_trace, n_trace)
            if tuple(stiffness.shape) != expected_shape:
                raise ValueError(
                    f"批次刚度形状必须为 {expected_shape}; "
                    f"当前为 {tuple(stiffness.shape)}."
                )

            indices_chunk = c_macro[start:end]
            shape = expected_shape
            rows = bm.reshape(
                bm.broadcast_to(indices_chunk[:, :, None], shape),
                (-1,),
            )
            cols = bm.reshape(
                bm.broadcast_to(indices_chunk[:, None, :], shape),
                (-1,),
            )
            values = bm.reshape(stiffness, (-1,))
            coo = COOTensor(
                indices=bm.stack([rows, cols], axis=0),
                values=values,
                spshape=(self.total_macro_dofs, self.total_macro_dofs),
            )
            K_chunk = coo.coalesce().tocsr()
            K_macro = K_chunk if K_macro is None else K_macro.add(K_chunk)
            next_start = end

        if next_start != n_sub_total or K_macro is None:
            raise ValueError(
                "stiffness_batches 未完整覆盖全部子结构; "
                f"已覆盖 {next_start}, 总数为 {n_sub_total}."
            )

        return InterfaceSystem(
            stiffness=K_macro,
            global_dofs=bm.arange(self.total_macro_dofs, dtype=bm.int64),
        )

    def assemble_trace_system(
        self,
        sub_meshes: Sequence[Any],
        condensors: Any,
        *,
        trace_basis: TraceBasis,
        chunk_size: Optional[int] = None,
    ) -> InterfaceSystem:
        """按指定迹空间装配全局子结构系统.

        Parameters
        ----------
        sub_meshes : sequence
            参与装配的全部子结构.
        condensors : Any
            已缩聚的批量结果、逐块结果, 或提供 get_chunk_stiffness 的流式对象.
        trace_basis : TraceBasis
            当前支持 FullTraceBasis 和 LinearCornerTraceBasis.
        chunk_size : int, optional
            每批装配的子结构数. full_trace 委托给接口 pattern 路径;
            linear_corner 在指定该参数或使用流式缩聚结果时按批投影.

        Returns
        -------
        InterfaceSystem
            full_trace 返回完整接口系统, linear_corner 返回宏观角点迹系统.

        Raises
        ------
        TypeError
            当 trace_basis 不是当前具有明确全局映射的迹基类型时抛出.
        ValueError
            当子结构为空、迹基维度不匹配或 chunk_size 非正时抛出.

        Notes
        -----
        full_trace 直接复用完整接口装配, 不对缩聚刚度执行恒等投影;
        linear_corner 先计算 T^T K_s T, 再装配宏观角点系统.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数; 当前为 {chunk_size}.")
        if not isinstance(
            trace_basis, (FullTraceBasis, LinearCornerTraceBasis)
        ):
            raise TypeError(
                "trace_basis 当前仅支持 FullTraceBasis 或 "
                "LinearCornerTraceBasis; "
                f"当前为 {type(trace_basis).__name__}."
            )

        n_sub_total = len(sub_meshes)
        n_b = int(sub_meshes[0].n_b)
        if trace_basis.n_boundary_dofs != n_b:
            raise ValueError(
                "trace_basis 的完整接口自由度数必须与子结构 n_b 一致; "
                f"当前为 {trace_basis.n_boundary_dofs} 与 {n_b}."
            )

        if isinstance(trace_basis, FullTraceBasis):
            return self.assemble_interface_system(
                list(sub_meshes), condensors, chunk_size=chunk_size
            )

        K_s_batch, _ = self.normalize_condensors(
            condensors, n_sub_total, n_b
        )
        if K_s_batch is not None and chunk_size is None:
            return self.assemble_macro_system(
                list(sub_meshes), trace_basis.project_stiffness(K_s_batch)
            )

        step = min(
            64 if chunk_size is None else chunk_size,
            n_sub_total,
        )

        def projected_batches() -> Iterable[TraceStiffnessBatch]:
            """逐批读取完整 Schur 刚度并投影到角点迹空间."""
            for start in range(0, n_sub_total, step):
                stop = min(start + step, n_sub_total)
                stiffness = (
                    condensors.get_chunk_stiffness(start, stop)
                    if K_s_batch is None
                    else K_s_batch[start:stop]
                )
                yield TraceStiffnessBatch(
                    start=start,
                    end=stop,
                    stiffness=trace_basis.project_stiffness(stiffness),
                )

        return self.assemble_macro_system_batches(
            sub_meshes, projected_batches()
        )

    @staticmethod
    def normalize_condensors(
        condensors: Any,
        n_sub_total: int,
        n_b: int,
    ) -> Tuple[Any, Callable[[Any], Any]]:
        """把逐个或批量给出的缩聚结果统一成批量形式.

        参数:
            condensors: 缩聚器列表, 单个旧式批量缩聚器, 或无状态
                ``LocalReductionBatchResult``. 刚度批量形状为
                ``(B, n_b, n_b)``; 旧式二维 ``K_s`` 表示全部子结构共用同一结果.
            n_sub_total: 子结构总数 ``B``.
            n_b: 单个子结构的接口自由度数.

        返回:
            (K_s_batch, recover): ``K_s_batch`` 形状为 ``(B, n_b, n_b)``;
                ``recover`` 把形状 ``(B, n_b)`` 的接口位移映射为形状 ``(B, n_i)``
                的内部位移.

        异常:
            ValueError: 当缩聚器数量或 ``K_s`` 形状与子结构不匹配时抛出.
            RuntimeError: 当任一缩聚器尚未调用 ``condense`` 时抛出.
        """
        if isinstance(condensors, (list, tuple)):
            if len(condensors) != n_sub_total:
                raise ValueError(
                    f"sub_meshes 与 condensors 的数量必须一致; "
                    f"当前为 {n_sub_total} 与 {len(condensors)}."
                )
            blocks = []
            for idx, condensor in enumerate(condensors):
                if condensor.K_s is None:
                    raise RuntimeError(
                        f"第 {idx} 个 condensor 必须在全局装配前完成 condense()."
                    )
                blocks.append(condensor.K_s)
            K_s_batch = bm.stack(blocks, axis=0)

            def recover(u_b_batch: Any) -> Any:
                """逐个子结构恢复内部位移后堆叠."""
                return bm.stack(
                    [c.recover(u_b_batch[i]) for i, c in enumerate(condensors)],
                    axis=0,
                )
        elif hasattr(condensors, "stiffness") and hasattr(condensors, "recover"):
            # ``LocalReductionBatchResult`` 是新无状态缩聚契约. 直接消费结果
            # 快照, 避免先回填旧 condensor 的 K_s/N 可变属性.
            K_s_batch = condensors.stiffness

            def recover(u_b_batch: Any) -> Any:
                """由无状态批量缩聚结果恢复内部位移."""
                return condensors.recover(u_b_batch)

        else:
            condensor = condensors
            if getattr(condensor, "K_s", None) is None:
                if hasattr(condensor, "get_chunk_stiffness"):
                    # 流式容器模式: 不在内存中持有全局全量张量, 按需由 get_chunk_stiffness 提供
                    def recover(u_b_batch: Any) -> Any:
                        return condensor.recover(u_b_batch)
                    return None, recover
                raise RuntimeError("condensor 必须在全局装配前完成 condense().")
            K_s_batch = condensor.K_s
            if K_s_batch.ndim == 2:
                # 单个缩聚结果广播到全部子结构, 对应各子结构密度完全相同的情形.
                K_s_batch = bm.broadcast_to(
                    K_s_batch[None, ...], (n_sub_total,) + tuple(K_s_batch.shape)
                )

            def recover(u_b_batch: Any) -> Any:
                """批量缩聚器的 recover 沿前导维广播, 一次完成全部子结构."""
                return condensor.recover(u_b_batch)

        if K_s_batch is not None:
            if K_s_batch.ndim != 3 or tuple(K_s_batch.shape) != (n_sub_total, n_b, n_b):
                raise ValueError(
                    f"K_s 的批量形状必须为 ({n_sub_total}, {n_b}, {n_b}); "
                    f"当前为 {tuple(K_s_batch.shape)}."
                )
        return K_s_batch, recover

    ### 装配与投影 ###

    def _prepare_interface_pattern(
        self,
        b_interface: Any,
        n_interface: int,
        dtype: Any,
    ) -> CSRPattern:
        """校验节点优先向量映射, 并构建或复用接口 CSR 符号模式."""
        mapping = np.asarray(bm.to_numpy(b_interface), dtype=np.int64)
        if n_interface % self.dim != 0 or mapping.shape[1] % self.dim != 0:
            raise ValueError("接口自由度数必须按空间分量完整分组.")

        grouped = mapping.reshape(mapping.shape[0], -1, self.dim)
        base = grouped[:, :, 0]
        expected = base[:, :, None] + np.arange(self.dim, dtype=np.int64)
        if np.any(base % self.dim) or not np.array_equal(grouped, expected):
            raise ValueError(
                "接口映射必须采用节点优先排列: dof = dim * node + component."
            )

        scalar_mapping = b_interface[:, ::self.dim] // self.dim
        scalar_np = np.ascontiguousarray(base // self.dim)
        digest = hashlib.blake2b(
            memoryview(scalar_np).cast("B"), digest_size=16
        ).digest()
        key = (
            tuple(scalar_np.shape),
            n_interface,
            self.dim,
            digest,
            bm.backend_name,
            str(getattr(b_interface, "device", "cpu")),
            str(dtype),
        )
        cached = self._interface_pattern_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        pattern = build_csr_pattern_from_dofmap(
            scalar_mapping,
            n_interface,
            dof_numel=self.dim,
            dof_priority=False,
            device=getattr(b_interface, "device", None),
            dtype=dtype,
            allocate_buffer=False,
        )
        self._interface_pattern_cache = (key, pattern)
        return pattern

    @staticmethod
    def _assemble_interface_values(
        pattern: CSRPattern,
        K_s_batch: Any,
        condensors: Any,
        n_batch: int,
        step: int,
    ) -> CSRTensor:
        """把完整或流式缩聚刚度按批次累加到独立数值缓冲区."""
        def chunks() -> Iterable[Tuple[int, Any]]:
            for start in range(0, n_batch, step):
                stop = min(start + step, n_batch)
                if K_s_batch is None:
                    values = condensors.get_chunk_stiffness(start, stop)
                else:
                    values = K_s_batch[start:stop]
                yield start, values
                del values

        return assemble_csr_chunks(chunks(), pattern)

    def assemble_interface_system(
        self,
        sub_meshes: List[Any],
        condensors: Any,
        *,
        chunk_size: Optional[int] = None,
    ) -> InterfaceSystem:
        """以缓存的 CSR pattern 散加各子结构缩聚刚度.

        Parameters
        ----------
        sub_meshes : list
            子结构列表, 边界自由度按节点优先分量顺序排列.
        condensors : Any
            已缩聚的批量结果、逐块结果, 或提供 get_chunk_stiffness 的对象.
        chunk_size : int, optional
            每次数值累加的子结构数. 不限制首次符号构建的内存.

        Returns
        -------
        InterfaceSystem
            接口 CSR 矩阵与自由度映射. 数值缓冲独立, 符号结构可复用.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数; 当前为 {chunk_size}.")

        interface_global_dofs = self.build_interface_dofs(sub_meshes)
        n_interface = int(len(interface_global_dofs))
        n_b = int(sub_meshes[0].n_b)
        b_interface = self.interface_indices(sub_meshes, interface_global_dofs)
        K_s_batch, _ = self.normalize_condensors(
            condensors, len(sub_meshes), n_b
        )
        dtype = (
            getattr(K_s_batch, "dtype", None)
            if K_s_batch is not None
            else bm.float64
        )
        pattern = self._prepare_interface_pattern(
            b_interface, n_interface, dtype
        )
        n_batch = len(sub_meshes)
        step = n_batch if chunk_size is None else min(chunk_size, n_batch)
        K_global = self._assemble_interface_values(
            pattern, K_s_batch, condensors, n_batch, step
        )
        return InterfaceSystem(
            stiffness=K_global,
            global_dofs=interface_global_dofs,
        )

    def project_global_vector(self, system: InterfaceSystem, global_vector: Any) -> Any:
        """把全局自由度向量投影为接口系统的右端项.

        参数:
            system: 接口系统.
            global_vector: 全局自由度向量, 长度为 ``total_full_dofs``.

        返回:
            interface_vector: 接口自由度上的分量, 形状 ``(n_interface,)``.

        异常:
            ValueError: 当 ``global_vector`` 长度与全局自由度数不一致时抛出.
        """
        vector = bm.asarray(global_vector, dtype=bm.float64)
        if len(vector) != self.total_full_dofs:
            raise ValueError(
                f"global_vector 的长度必须等于全局自由度数 {self.total_full_dofs}; "
                f"当前为 {len(vector)}."
            )
        return vector[system.global_dofs]

    def project_global_dofs(self, system: InterfaceSystem, global_dofs: Any) -> Any:
        """把全局自由度集合投影为接口自由度集合.

        参数:
            system: 接口系统.
            global_dofs: 待投影的全局自由度编号, 不在接口上的分量被丢弃.

        返回:
            interface_dofs: 升序去重的接口自由度编号.
        """
        dofs = bm.asarray(global_dofs, dtype=bm.int64)
        on_interface = bm.isin(dofs, system.global_dofs)
        return bm.unique(bm.searchsorted(system.global_dofs, dofs[on_interface]))

    def recover_full_displacement(
        self,
        sub_meshes: List[Any],
        condensors: Any,
        system: InterfaceSystem,
        interface_displacement: Any,
    ) -> Any:
        """由接口位移和局部缩聚结果恢复完整的全局位移向量.

        参数:
            sub_meshes: 子结构列表.
            condensors: 缩聚器列表或单个批量缩聚器, 形状约定见
                ``normalize_condensors``.
            system: 接口系统.
            interface_displacement: 接口自由度上的位移, 形状 ``(n_interface,)``.

        返回:
            U_full: 全局位移向量, 形状 ``(total_full_dofs,)``.

        异常:
            ValueError: 当子结构与缩聚器不匹配, 或位移长度与接口自由度数不一致时
                抛出.

        说明:
            内部自由度不被任何两个子结构共享, 因此写回时不存在重复索引;
            接口自由度直接由 ``interface_displacement`` 一次写入.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        u_b = bm.asarray(interface_displacement, dtype=bm.float64)
        if len(u_b) != len(system.global_dofs):
            raise ValueError(
                f"interface_displacement 的长度必须等于接口自由度数 "
                f"{len(system.global_dofs)}; 当前为 {len(u_b)}."
            )

        n_b = int(sub_meshes[0].n_b)
        b_interface = self.interface_indices(sub_meshes, system.global_dofs)
        _, recover = self.normalize_condensors(condensors, len(sub_meshes), n_b)

        U_full: Any = bm.zeros((self.total_full_dofs,), dtype=bm.float64)
        U_full = bm.set_at(U_full, system.global_dofs, u_b)

        # (B, n_b) -> (B, n_i): 批量缩聚器一次完成, 缩聚器列表逐个恢复后堆叠.
        u_sub_i = recover(u_b[b_interface])

        i_global = bm.stack(
            [
                self.get_substructure_global_dofs(pos, sub_mesh)[sub_mesh.i_dofs]
                for pos, sub_mesh in zip(
                    self._substructure_positions(sub_meshes), sub_meshes
                )
            ],
            axis=0,
        )
        return bm.set_at(
            U_full, bm.reshape(i_global, (-1,)), bm.reshape(u_sub_i, (-1,))
        )

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
            重排依据由节点坐标反解出的结构化下标, 不假定网格生成器的节点编号
            次序, 因此可直接用于云图绘制.
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
