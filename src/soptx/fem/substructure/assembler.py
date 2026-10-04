"""全局接口系统装配.

基于规则同构子结构布局完成缩聚刚度散加; 布局映射与位移恢复由独立模块提供. 装配器
只表达装配结果, 不携带载荷, 边界条件或求解策略, 这些由调用方编排.

子结构在全局网格中的位置由其 ``box_span`` 反解得到, 全局节点编号由节点坐标
反解得到, 因此装配不依赖子结构列表的排列次序, 也不依赖网格生成器的节点编号
约定. 缩聚结果既可以按子结构逐个给出, 也可以按批量前导维 ``B`` 一次给出.
"""

from dataclasses import dataclass
import hashlib
from typing import (
    Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
from scipy.sparse import coo_matrix

from soptx.backend import backend_manager as bm

from soptx.sparse import COOTensor, CSRTensor
from soptx.fem.matrix.csr_pattern import (
    CSRPattern,
    assemble_csr_chunks,
    build_csr_pattern_from_dofmap,
)

from .layout import (
    HasGlobalDofs,
    InterfaceDofsView,
    StructuredSubstructureLayout,
)
from .recovery import recover_full_displacement
from .reduction_adapter import normalize_local_reduction
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
    """基于结构化布局装配子结构缩聚后的全局接口系统.

    ``layout`` 保存整体几何、有限元上下文、自由度映射和场重排规则; 本类只
    保存接口装配的符号缓存并实现 full_trace、linear_corner 与统一 trace 装配.
    旧构造参数和布局查询入口继续转发到 ``layout``.
    """

    def __init__(
        self,
        domain_size: Union[StructuredSubstructureLayout, Tuple[float, ...], float],
        n_sub: Optional[Union[Tuple[int, ...], float, int]] = None,
        n_fine: Optional[Union[Tuple[int, ...], int]] = None,
        *args: Any,
        degree: int = 1,
        p: Optional[int] = None,
        E_base: float = 1.0,
        nu: float = 0.3,
        hypothesis: Optional[str] = None,
    ) -> None:
        """创建装配器或复用已有结构化布局.

        Parameters
        ----------
        domain_size : StructuredSubstructureLayout, tuple or float
            已构造布局, 或旧接口中的整体尺寸.
        n_sub, n_fine, args, degree, p, E_base, nu, hypothesis
            使用旧接口创建布局时原样传给 ``StructuredSubstructureLayout``.
        """
        if isinstance(domain_size, StructuredSubstructureLayout):
            if n_sub is not None or n_fine is not None or args:
                raise TypeError("传入 layout 时不能再指定 n_sub、n_fine 或位置参数.")
            if (
                degree != 1 or p is not None or E_base != 1.0
                or nu != 0.3 or hypothesis is not None
            ):
                raise TypeError("传入 layout 时不能重复指定布局或材料参数.")
            self.layout = domain_size
        else:
            if n_sub is None or n_fine is None:
                raise TypeError("旧构造接口必须同时给出 n_sub 与 n_fine.")
            self.layout = StructuredSubstructureLayout(
                domain_size,
                n_sub,
                n_fine,
                *args,
                degree=degree,
                p=p,
                E_base=E_base,
                nu=nu,
                hypothesis=hypothesis,
            )
        self._interface_pattern_cache: Optional[Tuple[Any, CSRPattern]] = None

    def __getattr__(self, name: str) -> Any:
        """把旧的布局属性与方法读取转发到统一 layout."""
        layout = self.__dict__.get("layout")
        if layout is None:
            raise AttributeError(name)
        return getattr(layout, name)

    @property
    def _full_mesh(self) -> Any:
        """兼容旧代码直接读取或替换全尺度网格缓存."""
        return self.layout._full_mesh

    @_full_mesh.setter
    def _full_mesh(self, value: Any) -> None:
        self.layout._full_mesh = value

    @property
    def _material(self) -> Any:
        """兼容旧代码直接读取或替换材料缓存."""
        return self.layout._material

    @_material.setter
    def _material(self, value: Any) -> None:
        self.layout._material = value

    @property
    def _sspace_full(self) -> Any:
        """兼容旧代码直接读取或替换标量空间缓存."""
        return self.layout._sspace_full

    @_sspace_full.setter
    def _sspace_full(self, value: Any) -> None:
        self.layout._sspace_full = value

    @property
    def _space_full(self) -> Any:
        """兼容旧代码直接读取或替换位移空间缓存."""
        return self.layout._space_full

    @_space_full.setter
    def _space_full(self, value: Any) -> None:
        self.layout._space_full = value

    @property
    def _node_of_grid(self) -> Any:
        """兼容旧代码直接读取或替换节点编号缓存."""
        return self.layout._node_of_grid

    @_node_of_grid.setter
    def _node_of_grid(self, value: Any) -> None:
        self.layout._node_of_grid = value

    @property
    def _dof_cache(self) -> Any:
        """兼容旧代码直接读取或替换自由度映射缓存."""
        return self.layout._dof_cache

    @_dof_cache.setter
    def _dof_cache(self, value: Any) -> None:
        self.layout._dof_cache = value

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
        interface_system: HasGlobalDofs,
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

        K_s_batch, _ = normalize_local_reduction(
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
        """兼容旧入口; 新代码使用 ``normalize_local_reduction``."""
        return normalize_local_reduction(condensors, n_sub_total, n_b)
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
        K_s_batch, _ = normalize_local_reduction(
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

    def recover_full_displacement(
        self,
        sub_meshes: List[Any],
        condensors: Any,
        system: HasGlobalDofs,
        interface_displacement: Any,
    ) -> Any:
        """兼容旧入口; 委托独立全局位移恢复函数."""
        return recover_full_displacement(
            self.layout,
            sub_meshes,
            condensors,
            system,
            interface_displacement,
        )
