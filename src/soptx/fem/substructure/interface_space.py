"""接口空间: 迹基 Psi, 全局编号 A_q^j 与全局迹映射 P_q 的统一载体.

概念文档 §3.1 与 §3.2 分别给出完整接口与角点接口的组装与求解, 两者只在局部
迹基 ``Psi^j``, 全局接口坐标 ``Q``, 局部提取矩阵 ``A_q^j`` 与全局迹映射 ``P_q``
的取值上不同: 完整接口取 ``Psi = I``, ``Q = U_Gamma``, ``A_q = A_b``,
``P_q = I``; 角点接口取 ``Psi = L``, ``Q = U_C``, ``A_q = A_c``, ``P_q = P``.
本模块把这组量收成一个对象, 让 ``full_trace`` 与 ``linear_corner`` 成为同一
类型的两个实例, 调用方不再按接口空间种类分叉.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity

from soptx.backend import backend_manager as bm
from soptx.sparse import CSRTensor

from soptx.fem.matrix.csr_pattern import (
    CSRPattern,
    assemble_csr_chunks,
    build_csr_pattern_from_dofmap,
)

from .layout import InterfaceDofsView
from .problem_adapter import (
    project_problem_conditions_to_interface_system,
    project_problem_conditions_to_macro_system,
)
from .traces import FullTraceBasis, LinearCornerTraceBasis, TraceBasis


@dataclass(frozen=True)
class InterfaceSystem:
    """缩聚后的接口刚度矩阵及其全局自由度映射.

    该对象只表达装配结果, 不携带载荷, 边界条件或求解策略.

    Attributes
    ----------
    stiffness : CSRTensor
        接口刚度矩阵 ``K_Q``, 形状 ``(N_q, N_q)``, 供 ``soptx.solvers.spsolve``
        直接求解.
    global_dofs : TensorLike
        ``Q`` 各分量对应的全局自由度编号, 升序排列, 形状 ``(N_q,)``. 升序是
        契约的一部分, 全局到接口的反查依赖二分定位而非字典.
    """

    stiffness: CSRTensor
    global_dofs: Any


def build_interface_pattern(
    local_dofs: Any,
    n_global: int,
    *,
    dof_numel: int = 1,
    dtype: Any = None,
) -> CSRPattern:
    """由 ``A_q^j`` 的编号形式构建全局接口刚度的 CSR 符号模式.

    Parameters
    ----------
    local_dofs : TensorLike
        形状 ``(M, n_q)``, 第 ``j`` 行为该子结构迹自由度在 ``Q`` 中的编号, 即
        普通有限元组装中的 ``cell2dof``.
    n_global : int
        全局接口坐标数 ``N_q``.
    dof_numel : int, optional
        每个节点的分量数. 大于 1 时要求 ``local_dofs`` 为节点优先排列
        ``dof = dof_numel * node + k``, 符号模式按标量节点映射构建后再按分量
        展开, 中间数组规模缩小 ``dof_numel**2`` 倍; 缺省 1 为通用的完整自由度
        映射.
    dtype : dtype, optional
        数值缓冲区的数据类型, 缺省 ``bm.float64``.

    Returns
    -------
    CSRPattern
        不预分配数值缓冲区的符号模式, 可在同一 ``local_dofs`` 上反复组装.

    Raises
    ------
    ValueError
        ``dof_numel`` 大于 1 但 ``local_dofs`` 不是节点优先排列.
    """
    local = bm.asarray(local_dofs, dtype=bm.int64)
    GD = int(dof_numel)
    device = getattr(local_dofs, "device", None)
    dtype = bm.float64 if dtype is None else dtype
    if GD == 1:
        return build_csr_pattern_from_dofmap(
            local, int(n_global), dof_numel=1,
            device=device, dtype=dtype, allocate_buffer=False,
        )

    mapping = np.asarray(bm.to_numpy(local), dtype=np.int64)
    if int(n_global) % GD != 0 or mapping.shape[1] % GD != 0:
        raise ValueError("接口自由度数必须按空间分量完整分组.")
    grouped = mapping.reshape(mapping.shape[0], -1, GD)
    base = grouped[:, :, 0]
    expected = base[:, :, None] + np.arange(GD, dtype=np.int64)
    if np.any(base % GD) or not np.array_equal(grouped, expected):
        raise ValueError(
            "local_dofs 必须采用节点优先排列: dof = dof_numel * node + k."
        )
    return build_csr_pattern_from_dofmap(
        local[:, ::GD] // GD,
        int(n_global),
        dof_numel=GD,
        dof_priority=False,
        device=device,
        dtype=dtype,
        allocate_buffer=False,
    )


def assemble_interface_stiffness(
    local_dofs: Any,
    n_global: int,
    stiffness_batches: Iterable[Any],
    *,
    pattern: Optional[CSRPattern] = None,
    dof_numel: int = 1,
) -> CSRTensor:
    """按普通有限元方式组装 ``K_Q = sum_j (A_q^j)^T K_r^j A_q^j``.

    完整接口下即式 (3.3) 的 ``K_Gamma``, 角点接口下即式 (3.8) 的 ``K_C``.

    Parameters
    ----------
    local_dofs : TensorLike
        形状 ``(M, n_q)`` 的 ``A_q^j`` 编号形式.
    n_global : int
        全局接口坐标数 ``N_q``.
    stiffness_batches : iterable
        连续覆盖全部子结构的批次迭代器, 每项提供 ``start``, ``end`` 与形状
        ``(end - start, n_q, n_q)`` 的 ``stiffness``.
    pattern : CSRPattern, optional
        预先由 ``build_interface_pattern`` 构建的符号模式; 省略时现场构建.
    dof_numel : int, optional
        现场构建模式时传给 ``build_interface_pattern`` 的每节点分量数.

    Returns
    -------
    CSRTensor
        形状 ``(N_q, N_q)`` 的全局接口刚度.

    Raises
    ------
    ValueError
        批次刚度形状与 ``n_q`` 不一致, 批次区间不连续或未完整覆盖子结构.

    Notes
    -----
    子结构充当单元, ``K_r^j`` 充当单元刚度, ``local_dofs`` 充当 ``cell2dof``,
    其余交给 ``soptx.fem.matrix.csr_pattern`` 的通用 dofmap 组装. 每个批次
    只向数值缓冲区累加, 峰值内存由缓冲区与单个批次决定.
    """
    local = bm.asarray(local_dofs, dtype=bm.int64)
    n_q = int(local.shape[1])
    if pattern is None:
        pattern = build_interface_pattern(local, n_global, dof_numel=dof_numel)

    def chunks() -> Iterator[Tuple[int, Any]]:
        for batch in stiffness_batches:
            start = int(batch.start)
            end = int(batch.end)
            stiffness = bm.asarray(batch.stiffness)
            expected_shape = (end - start, n_q, n_q)
            if tuple(stiffness.shape) != expected_shape:
                raise ValueError(
                    f"批次刚度形状必须为 {expected_shape}; "
                    f"当前为 {tuple(stiffness.shape)}."
                )
            yield start, stiffness
            del stiffness

    return assemble_csr_chunks(chunks(), pattern)


def linear_corner_global_map(
    layout: Any,
    sub_meshes: Sequence[Any],
    interface_global_dofs: Any,
    trace_basis: TraceBasis,
) -> csr_matrix:
    """构造角点线性迹的全局迹映射 ``U_Gamma = P U_C``, 对应式 (3.7).

    Parameters
    ----------
    layout : Any
        提供 ``interface_indices``, ``macro_corner_indices`` 与
        ``total_macro_dofs`` 的布局或装配器.
    sub_meshes : sequence
        参与接口装配的子结构.
    interface_global_dofs : TensorLike
        升序排列的完整接口全局自由度.
    trace_basis : TraceBasis
        局部角点迹基, 其矩阵把局部宏观角点自由度映射为完整局部边界位移.

    Returns
    -------
    scipy.sparse.csr_matrix
        形状为 ``(n_interface_dofs, total_macro_dofs)`` 的全局投影 ``P``.

    Raises
    ------
    ValueError
        局部维度不匹配, 投影未覆盖完整接口, 或相邻子结构在共享接口上给出
        不一致的插值.
    """
    if not sub_meshes:
        raise ValueError("sub_meshes 不能为空.")

    interface_dofs = bm.asarray(interface_global_dofs, dtype=bm.int64)
    boundary = np.asarray(
        bm.to_numpy(layout.interface_indices(sub_meshes, interface_dofs)),
        dtype=np.int64,
    )
    corners = np.asarray(
        bm.to_numpy(layout.macro_corner_indices(sub_meshes)), dtype=np.int64
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
        shape=(n_batch * n_boundary, layout.total_macro_dofs),
    ).tocsr()
    global_rows, first = np.unique(boundary.ravel(), return_index=True)
    if not np.array_equal(global_rows, np.arange(len(interface_dofs))):
        raise ValueError("角点线性迹投影未覆盖完整接口.")
    projection = candidates[first].tocsr()
    difference = candidates - projection[boundary.ravel()]
    if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-12:
        raise ValueError("相邻子结构在共享接口上给出了不一致的角点插值.")
    return projection


@dataclass(frozen=True, eq=False)
class InterfaceSpace:
    """一种接口空间在给定子结构排列上的实例.

    Attributes
    ----------
    name : str
        ``"full_trace"`` 或 ``"linear_corner"``.
    trace_basis : TraceBasis
        局部迹基 ``Psi``, 形状 ``(n_b, n_q)``.
    local_dofs : TensorLike
        局部提取矩阵 ``A_q^j`` 的编号形式, 形状 ``(M, n_q)``: 第 ``j`` 行给出
        该子结构 ``n_q`` 个迹自由度在全局坐标 ``Q`` 中的编号.
    global_dofs : TensorLike
        全局坐标 ``Q`` 各分量的编号, 形状 ``(N_q,)``. ``full_trace`` 为升序的
        完整接口全局自由度; ``linear_corner`` 为宏观粗网格自由度
        ``arange(total_macro_dofs)``.
    assembler : GlobalAssembler
        构造本空间所用的装配器.
    sub_meshes : sequence
        本空间绑定的子结构排列, ``local_dofs`` 的行序与之一致.

    Notes
    -----
    ``linear_corner`` 对应概念文档 §3.2: ``Psi = L``, ``Q = U_C``, ``A_q^j`` 即
    式 (3.6) 的 ``A_c^j``, 由 ``macro_corner_indices`` 给出. ``full_trace`` 对应
    §3.1: ``Psi = I``, ``Q = U_Gamma``, ``A_q^j`` 即式 (3.1) 的 ``A_b^j``, 由
    ``interface_indices`` 给出.
    """

    name: str
    trace_basis: TraceBasis
    local_dofs: Any
    global_dofs: Any
    assembler: Any
    sub_meshes: Sequence[Any]
    _pattern: Optional[CSRPattern] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        local = bm.asarray(self.local_dofs, dtype=bm.int64)
        if local.ndim != 2:
            raise ValueError(
                f"local_dofs 必须为二维 (M, n_q); 当前 ndim={local.ndim}."
            )
        if int(local.shape[0]) != len(self.sub_meshes):
            raise ValueError(
                f"local_dofs 的行数必须等于子结构数 {len(self.sub_meshes)}; "
                f"当前为 {int(local.shape[0])}."
            )
        if int(local.shape[1]) != self.trace_basis.n_trace_dofs:
            raise ValueError(
                "迹自由度编号与迹基维数不一致: "
                f"{int(local.shape[1])} vs {self.trace_basis.n_trace_dofs}."
            )
        n_b = int(self.sub_meshes[0].n_b)
        if self.trace_basis.n_boundary_dofs != n_b:
            raise ValueError(
                "trace_basis 的完整接口自由度数必须与子结构 n_b 一致; "
                f"当前为 {self.trace_basis.n_boundary_dofs} 与 {n_b}."
            )
        if int(bm.max(local)) >= self.n_global:
            raise ValueError(
                f"local_dofs 含有超出 N_q = {self.n_global} 的编号."
            )
        object.__setattr__(self, "local_dofs", local)
        object.__setattr__(
            self, "global_dofs", bm.asarray(self.global_dofs, dtype=bm.int64)
        )

    ### 维数 ###

    @property
    def n_global(self) -> int:
        """全局接口坐标数 ``N_q``."""
        return int(len(self.global_dofs))

    @property
    def n_trace_dofs(self) -> int:
        """单子结构迹自由度数 ``n_q``."""
        return self.trace_basis.n_trace_dofs

    @property
    def n_boundary_dofs(self) -> int:
        """单子结构完整接口自由度数 ``n_b``."""
        return self.trace_basis.n_boundary_dofs

    @property
    def n_substructures(self) -> int:
        """子结构数 ``M``."""
        return int(self.local_dofs.shape[0])

    ### §2.2 迹降阶 ###

    def project_stiffness(self, stiffness: Any) -> Any:
        """计算 ``K_r = Psi^T K_s Psi``.

        Parameters
        ----------
        stiffness : TensorLike
            完整接口缩聚刚度, 形状 ``(..., n_b, n_b)``.

        Returns
        -------
        TensorLike
            迹空间刚度, 形状 ``(..., n_q, n_q)``.
        """
        return self.trace_basis.project_stiffness(stiffness)

    ### 全局接口刚度的组装: 式 (3.3) 与 (3.8) ###

    @property
    def pattern(self) -> CSRPattern:
        """全局接口刚度的 CSR 符号模式, 只依赖 ``local_dofs``, 首次访问时构建并缓存."""
        if self._pattern is None:
            object.__setattr__(
                self,
                "_pattern",
                build_interface_pattern(
                    self.local_dofs,
                    self.n_global,
                    dof_numel=int(self.assembler.dim),
                ),
            )
        return self._pattern

    def assemble(self, stiffness_batches: Iterable[Any]) -> InterfaceSystem:
        """由迹空间刚度批次组装 ``K_Q = sum_j (A_q^j)^T K_r^j A_q^j``.

        完整接口下即式 (3.3) 的 ``K_Gamma``, 角点接口下即式 (3.8) 的 ``K_C``.

        Parameters
        ----------
        stiffness_batches : iterable
            连续覆盖全部子结构的批次迭代器, 每项提供 ``start``, ``end`` 与
            形状 ``(end - start, n_q, n_q)`` 的 ``stiffness``.

        Returns
        -------
        InterfaceSystem
            以 ``global_dofs`` 为行列编号的接口 CSR 系统.

        Notes
        -----
        两种接口空间共用同一份普通有限元式的 dofmap 组装, 见
        ``assemble_interface_stiffness``; 符号模式缓存在本对象上, 密度更新后
        反复组装只重填数值.
        """
        stiffness = assemble_interface_stiffness(
            self.local_dofs,
            self.n_global,
            stiffness_batches,
            pattern=self.pattern,
        )
        return InterfaceSystem(stiffness=stiffness, global_dofs=self.global_dofs)

    ### 全局接口载荷与支承约束: 式 (3.4), (3.9), (3.10) ###

    def project_conditions(self, problem: Any) -> Tuple[Any, Any]:
        """把 problem 契约投影为全局接口载荷 ``F_Q`` 与约束自由度集 ``D``.

        Parameters
        ----------
        problem : Any
            满足 soptx 弹性问题契约的对象.

        Returns
        -------
        load : TensorLike
            全局接口载荷, 形状 ``(N_q,)``: 完整接口下为式 (3.4) 的 ``F_Gamma``,
            角点接口下为直接落在宏观节点上的载荷.
        fixed_dofs : TensorLike
            ``Q`` 中受齐次位移约束的分量编号, 升序, 即 §3.1 的集合 ``D`` 或其
            宏观节点对应.

        Raises
        ------
        ValueError
            ``full_trace`` 下载荷或约束落在子结构内部自由度上. 静力缩聚未
            缩聚载荷, 完整接口系统无法表达这类条件.
        """
        if self.name == "linear_corner":
            return project_problem_conditions_to_macro_system(
                problem, self.assembler
            )
        conditions = project_problem_conditions_to_interface_system(
            problem,
            self.assembler,
            InterfaceDofsView(global_dofs=self.global_dofs),
        )
        return conditions.interface_force, conditions.interface_fixed_dofs

    def interface_conditions(self, problem: Any) -> Tuple[Any, Any]:
        """把 problem 契约投影到完整接口 ``U_Gamma``, 返回 ``(F_Gamma, D)``.

        Parameters
        ----------
        problem : Any
            满足 soptx 弹性问题契约的对象.

        Returns
        -------
        interface_force : TensorLike
            完整接口载荷 ``F_Gamma``, 形状 ``(n_interface,)``, 式 (3.4).
        fixed_dofs : TensorLike
            受齐次位移约束的完整接口自由度集 ``D``, 为接口编号, 升序, 见 §3.1.

        Raises
        ------
        ValueError
            载荷或约束落在子结构内部自由度上.

        Notes
        -----
        与接口空间种类无关, 总是投影到细网格的完整接口. ``full_trace`` 下与
        ``project_conditions`` 相同; ``linear_corner`` 下给出的是细网格接口上的
        条件, 供 ``constrained_conditions`` 经 ``P_q`` 转到 ``Q``, 而
        ``project_conditions`` 给出的是直接落在宏观节点上的条件.
        """
        interface_dofs = self.assembler.build_interface_dofs(list(self.sub_meshes))
        conditions = project_problem_conditions_to_interface_system(
            problem,
            self.assembler,
            InterfaceDofsView(global_dofs=interface_dofs),
        )
        return conditions.interface_force, conditions.interface_fixed_dofs

    def constrained_conditions(self, problem: Any) -> Tuple[Any, csr_matrix]:
        """按概念文档 §3.2 的形式给出 ``(F_Q, C_D)``.

        Parameters
        ----------
        problem : Any
            满足 soptx 弹性问题契约的对象.

        Returns
        -------
        load : TensorLike
            全局接口载荷 ``F_Q = P_q^T F_Gamma``, 形状 ``(N_q,)``. 由式 (3.7)
            它等于式 (3.9) 的 ``F_C``; 完整接口下退化为式 (3.4) 的 ``F_Gamma``.
        constraints : scipy.sparse.csr_matrix
            约束矩阵 ``C_D = P_q[D, :]``, 形状 ``(|D|, N_q)``, 约束方程
            ``C_D Q = 0`` 即式 (3.10) 的齐次情形.

        Notes
        -----
        载荷与支承先投影到完整接口, 再经全局迹映射 ``P_q`` 转到 ``Q``. 这样
        支承不落在角点上时 ``linear_corner`` 仍可表达, 由
        ``solve_constrained_system`` 以式 (3.11) 的乘子系统或消元处理冗余与
        一般线性约束. ``full_trace`` 下 ``P_q = I``, ``C_D`` 退化为 ``D`` 对应
        的单位阵行, 求解即式 (3.5) 的消元.
        """
        interface_force, fixed_dofs = self.interface_conditions(problem)
        P = self.global_trace_map()
        load = bm.asarray(
            P.T @ np.asarray(bm.to_numpy(interface_force), dtype=np.float64),
            dtype=bm.float64,
        )
        rows = np.asarray(bm.to_numpy(fixed_dofs), dtype=np.int64)
        constraints = P[rows].tocsr()
        return load, constraints

    ### 全局迹映射: 式 (3.7) ###

    def global_trace_map(self) -> csr_matrix:
        """全局迹映射 ``P_q``, 满足 ``U_Gamma = P_q Q``, 式 (3.7).

        Returns
        -------
        scipy.sparse.csr_matrix
            形状 ``(n_interface, N_q)``. ``full_trace`` 为单位阵;
            ``linear_corner`` 由各子结构的 ``L`` 拼装并校验共享接口一致.
        """
        if self.name == "linear_corner":
            interface_dofs = self.assembler.build_interface_dofs(
                list(self.sub_meshes)
            )
            return linear_corner_global_map(
                self.assembler, self.sub_meshes, interface_dofs, self.trace_basis
            )
        return identity(self.n_global, dtype=np.float64, format="csr")

    ### 位移提取: 式 (3.1), (3.6), 供 §3.3 恢复使用 ###

    def trace_displacement(self, global_displacement: Any) -> Any:
        """提取各子结构的迹位移 ``q^j = A_q^j Q``.

        Parameters
        ----------
        global_displacement : TensorLike
            全局接口位移 ``Q``, 形状 ``(N_q,)``.

        Returns
        -------
        TensorLike
            形状 ``(M, n_q)``.

        Raises
        ------
        ValueError
            ``global_displacement`` 不是长度为 ``N_q`` 的一维向量.
        """
        u = bm.asarray(global_displacement)
        if u.ndim != 1 or int(u.shape[0]) != self.n_global:
            raise ValueError(
                f"global_displacement 的形状必须为 ({self.n_global},); "
                f"当前为 {tuple(u.shape)}."
            )
        return u[self.local_dofs]

    def boundary_displacement(self, global_displacement: Any) -> Any:
        """提取各子结构的完整边界位移 ``u_b^j = Psi A_q^j Q``.

        Parameters
        ----------
        global_displacement : TensorLike
            全局接口位移 ``Q``, 形状 ``(N_q,)``.

        Returns
        -------
        TensorLike
            形状 ``(M, n_b)``.
        """
        return self.trace_basis.expand_displacement(
            self.trace_displacement(global_displacement)
        )

    ### 构造 ###

    @classmethod
    def from_trace_basis(
        cls,
        trace_basis: TraceBasis,
        assembler: Any,
        sub_meshes: Sequence[Any],
    ) -> "InterfaceSpace":
        """由迹基类型选定全局编号, 构造对应的接口空间.

        Parameters
        ----------
        trace_basis : TraceBasis
            ``FullTraceBasis`` 或 ``LinearCornerTraceBasis`` 的实例.
        assembler : GlobalAssembler
            子结构排列所在的装配器.
        sub_meshes : sequence
            子结构列表.

        Returns
        -------
        InterfaceSpace

        Raises
        ------
        ValueError
            ``sub_meshes`` 为空.
        TypeError
            ``trace_basis`` 不是具有明确全局映射的迹基类型.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")
        if isinstance(trace_basis, FullTraceBasis):
            global_dofs = assembler.build_interface_dofs(list(sub_meshes))
            local_dofs = assembler.interface_indices(sub_meshes, global_dofs)
            return cls(
                name="full_trace",
                trace_basis=trace_basis,
                local_dofs=local_dofs,
                global_dofs=global_dofs,
                assembler=assembler,
                sub_meshes=tuple(sub_meshes),
            )
        if isinstance(trace_basis, LinearCornerTraceBasis):
            return cls(
                name="linear_corner",
                trace_basis=trace_basis,
                local_dofs=assembler.macro_corner_indices(sub_meshes),
                global_dofs=bm.arange(
                    int(assembler.total_macro_dofs), dtype=bm.int64
                ),
                assembler=assembler,
                sub_meshes=tuple(sub_meshes),
            )
        raise TypeError(
            "trace_basis 当前仅支持 FullTraceBasis 或 LinearCornerTraceBasis; "
            f"当前为 {type(trace_basis).__name__}."
        )


INTERFACE_SPACE_KINDS = ("full_trace", "linear_corner")


def build_interface_space(
    kind: str,
    assembler: Any,
    sub_meshes: Sequence[Any],
    prototype: Any,
) -> InterfaceSpace:
    """按名称构造接口空间.

    Parameters
    ----------
    kind : str
        ``"full_trace"`` 或 ``"linear_corner"``.
    assembler : GlobalAssembler
        子结构排列所在的装配器.
    sub_meshes : sequence
        子结构列表.
    prototype : SubstructurePrototype
        参考子结构, 提供迹基的几何构造.

    Returns
    -------
    InterfaceSpace

    Raises
    ------
    ValueError
        ``kind`` 不在 ``INTERFACE_SPACE_KINDS`` 中.
    """
    if kind == "full_trace":
        trace_basis: TraceBasis = FullTraceBasis.from_prototype(prototype)
    elif kind == "linear_corner":
        trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
    else:
        raise ValueError(
            f"未知的接口空间: {kind!r}; 可选 {INTERFACE_SPACE_KINDS}."
        )
    return InterfaceSpace.from_trace_basis(trace_basis, assembler, sub_meshes)


__all__ = [
    "INTERFACE_SPACE_KINDS",
    "InterfaceSpace",
    "InterfaceSystem",
    "assemble_interface_stiffness",
    "build_interface_space",
    "build_interface_pattern",
    "linear_corner_global_map",
]
