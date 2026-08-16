"""全局接口系统装配.

负责规则排布的同构子结构的自由度映射, 缩聚刚度散加与全场位移恢复. 装配器
只表达装配结果, 不携带载荷, 边界条件或求解策略, 这些由调用方编排.

子结构在全局网格中的位置由其 ``box_span`` 反解得到, 全局节点编号由节点坐标
反解得到, 因此装配不依赖子结构列表的排列次序, 也不依赖网格生成器的节点编号
约定. 缩聚结果既可以按子结构逐个给出, 也可以按批量前导维 ``B`` 一次给出.
"""

from dataclasses import dataclass
from typing import Tuple, List, Union, Any, Callable, Optional, Sequence

import scipy.sparse as sp

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.materials import IsotropicLinearElasticMaterial


@dataclass(frozen=True)
class InterfaceSystem:
    """缩聚后的接口刚度矩阵及其全局自由度映射.

    该对象只表达装配结果, 不携带载荷, 边界条件或求解策略.

    属性:
        stiffness: 接口刚度矩阵, 形状 ``(n_interface, n_interface)``. 采用
            ``scipy`` 稀疏格式, 供 ``scipy.sparse.linalg`` 直接求解.
        global_dofs: 接口自由度对应的全局自由度编号, 升序排列, 形状
            ``(n_interface,)``. 升序是契约的一部分, 全局到接口的反查依赖
            二分定位而非字典.
    """

    stiffness: sp.csr_matrix
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
            E_base: 实体材料的杨氏模量.
            nu: 泊松比.

        异常:
            ValueError: 当参数无法解析为 2D 或 3D 的尺寸, 子结构数与细单元数
                三元组时抛出.

        说明:
            求解域固定为以原点为下角的长方体 ``[0, Lx] x [0, Ly] (x [0, Lz])``.
            元组形式与标量形式在 2D 与 3D 下都可用; 标量形式按 ``尺寸, 子结构数,
            细单元数`` 的分组依次给出各方向分量.
        """
        all_args = (domain_size, n_sub, n_fine) + args
        parsed = self._parse_layout(all_args, E_base, nu)
        self.domain_size, self.n_sub, self.n_fine, self.E_base, self.nu = parsed
        self.dim: int = len(self.domain_size)

        self.total_fine: Tuple[int, ...] = tuple(
            self.n_sub[d] * self.n_fine[d] for d in range(self.dim)
        )
        self.n_full_nodes: Tuple[int, ...] = tuple(n + 1 for n in self.total_fine)
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
            self._sspace_full = LagrangeFESpace(self.full_mesh, p=1, ctype='C')
        return self._sspace_full

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
        """返回结构化网格下标到全局节点编号的映射, 形状 ``(total_full_nodes,)``.

        返回:
            node_of_grid: 以 C 序线性化的结构化下标为索引, 给出对应的全局节点编号.

        异常:
            RuntimeError: 当全网格节点无法与结构化下标一一对应时抛出.

        说明:
            由节点坐标反解下标而不假定网格生成器的编号次序; 结果缓存复用.
        """
        if self._node_of_grid is not None:
            return self._node_of_grid

        node = self.full_mesh.entity('node')
        linear_index: Any = bm.zeros((self.total_full_nodes,), dtype=bm.int64)
        for d in range(self.dim):
            h_d = self.domain_size[d] / self.total_fine[d]
            idx_d = bm.astype(bm.round(node[:, d] / h_d), bm.int64)
            idx_d = bm.clip(idx_d, 0, self.total_fine[d])
            linear_index = linear_index * self.n_full_nodes[d] + idx_d

        arange_nodes = bm.arange(self.total_full_nodes, dtype=bm.int64)
        if not bool(bm.all(bm.sort(linear_index) == arange_nodes)):
            raise RuntimeError(
                "全网格节点未能与结构化下标一一对应, 无法建立子结构到全局的映射."
            )

        inverse: Any = bm.zeros((self.total_full_nodes,), dtype=bm.int64)
        self._node_of_grid = bm.set_at(inverse, linear_index, arange_nodes)
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
        offset = bm.asarray(
            [pos[d] * self.n_fine[d] for d in range(self.dim)], dtype=bm.int64
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

    def _interface_indices(
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

    ### 缩聚结果归一化 ###

    @staticmethod
    def _normalize_condensors(
        condensors: Any,
        n_sub_total: int,
        n_b: int,
    ) -> Tuple[Any, Callable[[Any], Any]]:
        """把逐个或批量给出的缩聚结果统一成批量形式.

        参数:
            condensors: 缩聚器列表, 每个的 ``K_s`` 形状为 ``(n_b, n_b)``; 或单个
                缩聚器, 其 ``K_s`` 形状为 ``(B, n_b, n_b)`` 或 ``(n_b, n_b)``,
                后者表示全部子结构共用同一缩聚结果.
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
        else:
            condensor = condensors
            if condensor.K_s is None:
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

        if K_s_batch.ndim != 3 or tuple(K_s_batch.shape) != (n_sub_total, n_b, n_b):
            raise ValueError(
                f"K_s 的批量形状必须为 ({n_sub_total}, {n_b}, {n_b}); "
                f"当前为 {tuple(K_s_batch.shape)}."
            )
        return K_s_batch, recover

    ### 装配与投影 ###

    def assemble_interface_system(
        self,
        sub_meshes: List[Any],
        condensors: Any,
        *,
        chunk_size: Optional[int] = None,
    ) -> InterfaceSystem:
        """
        对各子结构的缩聚刚度矩阵执行散加, 构造全局接口刚度矩阵.

        参数:
            sub_meshes: 子结构列表.
            condensors: 缩聚器列表或单个批量缩聚器, 形状约定见
                ``_normalize_condensors``.
            chunk_size: 每批散加的子结构数. 为 ``None`` 时一次散加全部子结构.
                散加需要 ``B * n_b**2`` 量级的行列索引与数值缓冲, 子结构数很大时
                用该参数限制峰值内存.

        返回:
            system: 接口刚度矩阵及其全局自由度映射.

        异常:
            ValueError: 当子结构与缩聚器不匹配或 ``chunk_size`` 非正时抛出.

        说明:
            不施加载荷, 边界条件, 也不调用线性求解器.

            散加通过 COO 三元组完成, 重复项由 ``coo_matrix`` 转 CSR 时求和, 取代
            逐元素累加. 数值在此处经 ``bm.to_numpy`` 转出到 ``scipy``, 这是流程中
            与第三方求解库对接的边界.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数; 当前为 {chunk_size}.")

        interface_global_dofs = self.build_interface_dofs(sub_meshes)
        n_interface = int(len(interface_global_dofs))
        n_b = int(sub_meshes[0].n_b)

        b_interface = self._interface_indices(sub_meshes, interface_global_dofs)
        K_s_batch, _ = self._normalize_condensors(condensors, len(sub_meshes), n_b)

        n_batch = len(sub_meshes)
        step = n_batch if chunk_size is None else min(chunk_size, n_batch)
        K_global = sp.csr_matrix((n_interface, n_interface), dtype=float)
        for start in range(0, n_batch, step):
            sl = slice(start, start + step)
            idx = b_interface[sl]
            n_chunk = int(idx.shape[0])
            shape = (n_chunk, n_b, n_b)
            rows = bm.to_numpy(bm.broadcast_to(idx[:, :, None], shape)).reshape(-1)
            cols = bm.to_numpy(bm.broadcast_to(idx[:, None, :], shape)).reshape(-1)
            vals = bm.to_numpy(K_s_batch[sl]).reshape(-1)
            K_global = K_global + sp.coo_matrix(
                (vals, (rows, cols)), shape=(n_interface, n_interface)
            ).tocsr()

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
                ``_normalize_condensors``.
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
        b_interface = self._interface_indices(sub_meshes, system.global_dofs)
        _, recover = self._normalize_condensors(condensors, len(sub_meshes), n_b)

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

    def reconstruct_global_field(self, sub_fields: Union[List[Any], Any]) -> Any:
        """把各子结构的局部数据场拼接为全局连续数据场.

        参数:
            sub_fields: 各子结构的局部场, 每个形状为 ``n_fine``. 必须按 x 优先的
                字典序排列, 即 2D 下 ``sub_id = sx * n_sub_y + sy``, 3D 下
                ``sub_id = (sx * n_sub_y + sy) * n_sub_z + sz``.

        返回:
            global_field: 形状为 ``total_fine`` 的全局场.

        说明:
            本方法只收到数据场, 无从反解子结构位置, 因此次序契约由调用方保证;
            涉及网格的接口一律由 ``box_span`` 反解位置, 不依赖列表次序.
        """
        field = bm.asarray(sub_fields)
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

    # 向后兼容别名
    reconstruct_full_density = reconstruct_global_field
    assemble_full_density = reconstruct_global_field
