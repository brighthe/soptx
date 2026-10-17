"""子结构局部缩聚与迹投影的流式编排."""

from __future__ import annotations

from time import perf_counter
from dataclasses import dataclass
from typing import Any, Iterator

from soptx.backend import backend_manager as bm

from .reductions import ExactSchurReduction
from .traces import TraceBasis


@dataclass(frozen=True)
class LocalCondensationBatch:
    """一个连续子结构批次在完整接口上的精确静力缩聚结果.

    Attributes
    ----------
    start : int
        批次在展平子结构序列中的起始编号.
    end : int
        批次在展平子结构序列中的结束编号, 不包含该位置.
    stiffness : TensorLike
        完整接口缩聚刚度 ``K_s = K_bb - K_bi K_ii^{-1} K_ib``, 形状
        ``(end - start, n_b, n_b)``.
    recovery : TensorLike
        内部位移恢复矩阵 ``T_full = -K_ii^{-1} K_ib``, 形状
        ``(end - start, n_i, n_b)``.
    """

    start: int
    end: int
    stiffness: Any
    recovery: Any


@dataclass(frozen=True)
class TraceStiffnessBatch:
    """一个连续子结构批次在指定迹空间上的缩聚刚度.

    Attributes
    ----------
    start : int
        批次在展平子结构序列中的起始编号.
    end : int
        批次在展平子结构序列中的结束编号, 不包含该位置.
    stiffness : TensorLike
        迹空间缩聚刚度, 形状 ``(end - start, n_trace, n_trace)``.
    """

    start: int
    end: int
    stiffness: Any


@dataclass(frozen=True)
class InternalDisplacementBatch:
    """一个连续子结构批次由迹位移恢复出的边界与内部位移.

    Attributes
    ----------
    start : int
        批次在展平子结构序列中的起始编号.
    end : int
        批次在展平子结构序列中的结束编号, 不包含该位置.
    boundary : TensorLike
        完整接口位移 ``u_b = Psi q``, 形状 ``(end - start, n_b)``.
    internal : TensorLike
        内部位移 ``u_i = T_full u_b``, 形状 ``(end - start, n_i)``.
    """

    start: int
    end: int
    boundary: Any
    internal: Any


@dataclass(frozen=True)
class ElementStrainEnergyBatch:
    """一个连续子结构批次的单位刚度单元应变能.

    Attributes
    ----------
    start : int
        批次在展平子结构序列中的起始编号.
    end : int
        批次在展平子结构序列中的结束编号, 不包含该位置.
    energy : TensorLike
        按参考子结构 FE cell 编号排列的单位刚度单元应变能, 形状
        ``(end - start, n_cells)``.
    """

    start: int
    end: int
    energy: Any


def iter_exact_condensation_batches(
    prototype: Any,
    density: Any,
    *,
    chunk_size: int,
) -> Iterator[LocalCondensationBatch]:
    """流式执行局部刚度装配与精确 Schur 静力缩聚.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构共享的参考子结构.
    density : TensorLike
        局部子结构密度批次, 形状约定见 ``SubstructurePrototype.to_cell_density``.
    chunk_size : int
        单次处理的最大子结构数, 必须为正整数.

    Yields
    ------
    LocalCondensationBatch
        当前连续批次的完整接口缩聚刚度与恢复矩阵.

    Raises
    ------
    ValueError
        ``chunk_size`` 非正, 由局部装配抛出.

    Notes
    -----
    每个批次先由密度装配局部刚度 ``K_local``, 再按精确 Schur 补得到 ``K_s``
    与 ``T_full``. ``K_local`` 只在当前批次生命周期内存在, 不随结果产出. 本函数
    不施加接口迹投影, 迹投影由 ``iter_exact_trace_stiffness_batches`` 在其之上
    完成.
    """
    yield from iter_condensation_from_stiffness_batches(
        prototype.iter_local_stiffness_batches(density, chunk_size=chunk_size),
        i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
    )


def iter_condensation_from_stiffness_batches(
    stiffness_batches: Any, *, i_dofs: Any, b_dofs: Any,
) -> Iterator[LocalCondensationBatch]:
    """消费局部刚度批次, 执行完整边界上的 Exact Schur 缩聚.

    Parameters
    ----------
    stiffness_batches : iterable
        从 0 开始连续的 (start, end, K_local), 矩阵形状为
        (end - start, n_dof, n_dof). 完整覆盖由下游组装或恢复校验.
    i_dofs, b_dofs : array_like
        内部和边界自由度编号, 须完整划分局部自由度.

    Yields
    ------
    LocalCondensationBatch
        完整边界缩聚刚度及恢复矩阵, 不跨批缓存.

    Raises
    ------
    ValueError
        批次不连续、区间为空或矩阵形状不符.
    """
    reduction = ExactSchurReduction(i_dofs, b_dofs)
    n_dof = len(i_dofs) + len(b_dofs)
    expected_start = 0
    for start, end, local_stiffness in stiffness_batches:
        if (start != expected_start or end <= start
                or tuple(local_stiffness.shape) != (end - start, n_dof, n_dof)):
            raise ValueError("局部刚度批次的区间或形状不符.")
        result = reduction.reduce_many(local_stiffness)
        yield LocalCondensationBatch(
            start=start,
            end=end,
            stiffness=result.stiffness,
            recovery=result.recovery,
        )
        expected_start = end


def iter_exact_trace_stiffness_batches(
    prototype: Any,
    density: Any,
    trace_basis: TraceBasis,
    *,
    chunk_size: int,
) -> Iterator[TraceStiffnessBatch]:
    """流式执行局部刚度装配, Exact Schur 缩聚和迹投影.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构共享的参考子结构.
    density : TensorLike
        局部子结构密度批次, 形状约定见 ``SubstructurePrototype.to_cell_density``.
    trace_basis : TraceBasis
        从迹自由度到完整接口自由度的线性映射.
    chunk_size : int
        单次处理的最大子结构数, 必须为正整数.

    Yields
    ------
    TraceStiffnessBatch
        当前连续批次的迹空间缩聚刚度.

    Raises
    ------
    ValueError
        迹基的完整接口自由度数与参考子结构不一致.

    Notes
    -----
    每个批次严格执行有限元 Exact Schur 补, 不使用同质判定, 阈值分类或 PIML
    代理. 完整局部刚度, 完整接口 Schur 矩阵和恢复矩阵均只在当前批次生命周期
    内存在; 跨批次只向调用方传递投影后的迹空间刚度.
    """
    if trace_basis.n_boundary_dofs != prototype.n_b:
        raise ValueError(
            "trace_basis 的完整接口自由度数必须与 prototype.n_b 一致; "
            f"当前为 {trace_basis.n_boundary_dofs} 与 {prototype.n_b}."
        )

    yield from iter_trace_stiffness_from_local_batches(
        prototype.iter_local_stiffness_batches(density, chunk_size=chunk_size),
        trace_basis, i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
    )


def iter_trace_stiffness_from_local_batches(
    stiffness_batches: Any, trace_basis: TraceBasis, *, i_dofs: Any, b_dofs: Any,
) -> Iterator[TraceStiffnessBatch]:
    """消费局部刚度批次, 完成完整 Schur 缩聚与迹投影.

    Parameters
    ----------
    stiffness_batches : iterable
        连续的 (start, end, K_local) 批次.
    trace_basis : TraceBasis
        完整边界与迹空间之间的映射.
    i_dofs, b_dofs : array_like
        内部和边界自由度编号.

    Yields
    ------
    TraceStiffnessBatch
        可直接交给 InterfaceSpace.assemble 的矩阵批次.

    Raises
    ------
    ValueError
        迹基与边界自由度数不一致, 或局部矩阵批次不合法.
    """
    if trace_basis.n_boundary_dofs != len(b_dofs):
        raise ValueError("迹基边界自由度数与子结构不一致.")
    for batch in iter_condensation_from_stiffness_batches(
        stiffness_batches, i_dofs=i_dofs, b_dofs=b_dofs,
    ):
        yield TraceStiffnessBatch(
            start=batch.start,
            end=batch.end,
            stiffness=trace_basis.project_stiffness(batch.stiffness),
        )


def assemble_exact_interface_system(
    prototype: Any,
    density: Any,
    interface_space: Any,
    *,
    chunk_size: int,
) -> Any:
    """由局部密度一次完成精确缩聚, 迹降阶与全局接口刚度组装.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构共享的参考子结构.
    density : TensorLike
        局部子结构密度批次, 形状约定见 ``SubstructurePrototype.to_cell_density``.
    interface_space : InterfaceSpace
        接口空间, 提供迹基 ``Psi`` 与全局编号 ``A_q^j``.
    chunk_size : int
        单次处理的最大子结构数, 必须为正整数.

    Returns
    -------
    InterfaceSystem
        全局接口刚度 ``K_Q`` 及其自由度编号.

    Raises
    ------
    ValueError
        ``chunk_size`` 非正, 或迹基与参考子结构的接口自由度数不一致.

    Notes
    -----
    调用即执行, 不返回惰性迭代器. 子结构按 ``chunk_size`` 分块, 每块依次:

    1. 局部装配 ``K^j = sum_e coef(rho_e) K_e``, 形状 ``(b, n_dof, n_dof)``;
    2. 静力缩聚 ``K_s^j = K_bb - K_bi K_ii^{-1} K_ib``, 形状 ``(b, n_b, n_b)``;
    3. 迹降阶 ``K_r^j = Psi^T K_s^j Psi``, 形状 ``(b, n_q, n_q)``;
    4. 散加 ``K_Q += (A_q^j)^T K_r^j A_q^j``.

    前三步的中间量只在当前分块内存在, 跨块只保留 ``K_Q`` 的 CSR 数值缓冲区.
    等价于 ``interface_space.assemble(iter_exact_trace_stiffness_batches(...))``,
    完整接口下得到式 (3.3) 的 ``K_Gamma``, 角点接口下得到式 (3.8) 的 ``K_C``.
    """
    batches = iter_exact_trace_stiffness_batches(
        prototype,
        density,
        interface_space.trace_basis,
        chunk_size=chunk_size,
    )
    return interface_space.assemble(batches)


def assemble_corner_system_with_recovery(
    prototype: Any, density: Any, interface_space: Any, *, chunk_size: int,
    timings: Any = None,
) -> tuple[Any, list]:
    """直接装配角点缩聚系统并缓存当前密度的内部恢复矩阵.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构原型.
    density : TensorLike
        当前物理密度.
    interface_space : InterfaceSpace
        linear_corner 接口空间.
    chunk_size : int
        每批子结构数.

    timings : dict, optional
        写入 CPU 墙钟分项秒数及 local_condensation_total 汇总.
        汇总不参与分项累加, global_csr 包含编排差额.

    Returns
    -------
    system : InterfaceSystem
        角点接口刚度.
    recovery_batches : list
        (start, end, T_corner) 列表, T_corner = -solve(K_ii, K_ib P).
        不保存完整 Schur 补, 仅在装配成功后交给分析器.
    """
    return assemble_corner_from_stiffness_batches(
        prototype.iter_local_stiffness_batches(density, chunk_size=chunk_size),
        interface_space, i_dofs=prototype.i_dofs, b_dofs=prototype.b_dofs,
        timings=timings,
    )


def assemble_corner_from_stiffness_batches(
    stiffness_batches: Any, interface_space: Any, *, i_dofs: Any, b_dofs: Any,
    timings: Any = None,
) -> tuple[Any, list]:
    """消费局部刚度批次, 直接组装角点系统并保存恢复矩阵.

    Parameters
    ----------
    stiffness_batches : iterable
        从 0 开始连续的 (start, end, K_local) 批次.
    interface_space : InterfaceSpace
        linear_corner 接口空间.
    i_dofs, b_dofs : array_like
        内部和边界自由度编号.
    timings : dict, optional
        写入分项墙钟时间. local_stiffness 统计获取上游批次的时间,
        已在调用前生成矩阵时不包含其生成成本. block_projection、local_solve
        和 condensation_assembly 分别统计分块投影、内部求解与缩聚刚度合成;
        local_condensation_total 是三项之和, 不可再作为独立分项累加.

    Returns
    -------
    system : InterfaceSystem
        全局角点接口刚度.
    recovery_batches : list
        (start, end, T_corner), 满足 u_i = T_corner q.

    Raises
    ------
    ValueError
        接口空间不是 linear_corner, 或批次区间及矩阵形状不符.
    """
    measured = {key: 0.0 for key in
                ("local_stiffness", "block_projection", "local_solve",
                 "condensation_assembly", "global_csr")}
    total_started = perf_counter()
    basis = interface_space.trace_basis
    if basis.name != "linear_corner":
        raise ValueError("直接角点缩聚仅支持 linear_corner.")
    i = bm.asarray(i_dofs, dtype=bm.int64)
    b = bm.asarray(b_dofs, dtype=bm.int64)
    if basis.n_boundary_dofs != len(b):
        raise ValueError('迹基边界自由度数与子结构不一致.')
    P = basis.matrix
    recovery_batches = []

    def batches():
        """逐批投影边界块, 只求解角点数目的右端项."""
        iterator = iter(stiffness_batches)
        expected_start = 0
        n_dof = len(i) + len(b)
        while True:
            started = perf_counter()
            try:
                start, end, local = next(iterator)
            except StopIteration:
                measured["local_stiffness"] += perf_counter() - started
                break
            measured["local_stiffness"] += perf_counter() - started
            started = perf_counter()
            if (start != expected_start or end <= start
                    or tuple(local.shape) != (end - start, n_dof, n_dof)):
                raise ValueError("局部刚度批次的区间或形状不符.")
            expected_start = end
            K_ii = local[..., i[:, None], i]
            B = local[..., i[:, None], b] @ P
            K_cc = bm.matrix_transpose(P) @ local[..., b[:, None], b] @ P
            measured["block_projection"] += perf_counter() - started
            started = perf_counter()
            if len(i):
                recovery = -bm.linalg.solve(K_ii, B)
            else:
                recovery = B
            measured["local_solve"] += perf_counter() - started
            started = perf_counter()
            stiffness = K_cc + bm.matrix_transpose(B) @ recovery
            recovery_batches.append((start, end, recovery))
            measured["condensation_assembly"] += perf_counter() - started
            yield TraceStiffnessBatch(start, end, stiffness)

    system = interface_space.assemble(batches())
    # 生成器内部计算已单独累计, 差额含 CSR 模式构建、散加和编排开销.
    measured["global_csr"] = max(0.0, perf_counter() - total_started - sum(measured.values()))
    # 汇总在互斥分项结算后添加, 避免从 global_csr 中重复扣减.
    measured["local_condensation_total"] = sum(
        measured[key] for key in ("block_projection", "local_solve", "condensation_assembly")
    )
    if timings is not None:
        timings.update(measured)
    return system, recovery_batches


def iter_exact_internal_displacement_batches(
    prototype: Any,
    density: Any,
    trace_displacement: Any,
    trace_basis: TraceBasis,
    *,
    chunk_size: int,
    recovery_batches: Any = None,
) -> Iterator[InternalDisplacementBatch]:
    """流式恢复各子结构的边界与内部位移, 对应概念文档 §3.3.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构共享的参考子结构.
    density : TensorLike
        局部子结构密度批次, 形状约定见 ``SubstructurePrototype.to_cell_density``.
    trace_displacement : TensorLike
        各子结构的迹自由度位移 ``q^j``, 形状 ``(n_substructure, n_trace)``.
    trace_basis : TraceBasis
        从迹自由度到完整接口自由度的线性映射 ``Psi``.
    chunk_size : int
        单次处理的最大子结构数, 必须为正整数.

    recovery_batches : list, optional
        与当前密度和接口系统对应的角点恢复缓存, 由装配函数生成.

    Yields
    ------
    InternalDisplacementBatch
        当前连续批次的 ``u_b^j`` 与 ``u_i^j``.

    Raises
    ------
    ValueError
        迹基, 密度批量或迹位移形状不一致.

    Notes
    -----
    提供 recovery_batches 时直接使用当前装配缓存的角点恢复矩阵, 不重新缩聚.
    未提供缓存时, 本方法按批重新装配局部刚度并执行 Exact Schur, 用该批
    的 ``T_full`` 计算 ``u_i = T_full Psi q``. 恢复矩阵不跨批保存, 以重算换
    内存. 单元应变能与全尺度位移向量都建在此恢复接口之上.
    """
    if trace_basis.n_boundary_dofs != prototype.n_b:
        raise ValueError(
            "trace_basis 的完整接口自由度数必须与 prototype.n_b 一致; "
            f"当前为 {trace_basis.n_boundary_dofs} 与 {prototype.n_b}."
        )

    rho_cells = prototype.to_cell_density(density)
    n_substructure = int(bm.reshape(rho_cells, (-1, prototype.n_cells)).shape[0])
    displacement = bm.asarray(trace_displacement)
    expected_shape = (n_substructure, trace_basis.n_trace_dofs)
    if tuple(displacement.shape) != expected_shape:
        raise ValueError(
            f"trace_displacement 形状必须为 {expected_shape}; "
            f"当前为 {tuple(displacement.shape)}."
        )

    if recovery_batches is not None:
        yield from iter_displacement_from_recovery_batches(
            recovery_batches, displacement, trace_basis,
            n_internal=len(prototype.i_dofs), recovery_on_trace=True,
        )
        return

    recovery = (
        (batch.start, batch.end, batch.recovery)
        for batch in iter_exact_condensation_batches(
            prototype, rho_cells, chunk_size=chunk_size,
        )
    )
    yield from iter_displacement_from_recovery_batches(
        recovery, displacement, trace_basis,
        n_internal=len(prototype.i_dofs), recovery_on_trace=False,
    )


def iter_displacement_from_recovery_batches(
    recovery_batches: Any, trace_displacement: Any, trace_basis: TraceBasis,
    *, n_internal: int, recovery_on_trace: bool,
) -> Iterator[InternalDisplacementBatch]:
    """仅由恢复矩阵批次与迹位移计算边界和内部位移.

    Parameters
    ----------
    recovery_batches : iterable
        连续覆盖全部子结构的 (start, end, T) 批次.
    trace_displacement : TensorLike
        形状为 (n_substructure, n_trace) 的局部迹位移.
    trace_basis : TraceBasis
        用于计算 u_b = Psi q 的迹基.
    n_internal : int
        每个子结构的内部自由度数.
    recovery_on_trace : bool
        True 时 T 的形状为 (b, n_internal, n_trace), 计算 T q;
        False 时形状为 (b, n_internal, n_boundary), 计算 T u_b.

    Yields
    ------
    InternalDisplacementBatch
        可交给全场位移拼接接口的局部位移批次.

    Raises
    ------
    ValueError
        位移或恢复矩阵形状不符, 批次不连续或覆盖不完整.

    Notes
    -----
    不读取密度、材料或局部刚度, 不执行缩聚. 显式区分恢复矩阵的输入空间,
    保留完整边界恢复的 T (Psi q) 运算顺序, 不额外形成 T Psi.
    """
    displacement = bm.asarray(trace_displacement)
    if (displacement.ndim != 2
            or displacement.shape[1] != trace_basis.n_trace_dofs or n_internal < 0):
        raise ValueError("迹位移形状或内部自由度数不符.")
    n_substructure = displacement.shape[0]
    n_input = (trace_basis.n_trace_dofs if recovery_on_trace
               else trace_basis.n_boundary_dofs)
    expected_start = 0
    for start, end, recovery in recovery_batches:
        if (start != expected_start or end <= start or end > n_substructure
                or tuple(recovery.shape) != (end - start, n_internal, n_input)):
            raise ValueError("恢复缓存的批次或形状与当前接口系统不一致.")
        q = displacement[start:end]
        u_boundary = trace_basis.expand_displacement(q)
        u_internal = bm.einsum(
            "...ij,...j->...i", recovery, q if recovery_on_trace else u_boundary,
        )
        yield InternalDisplacementBatch(
            start=start,
            end=end,
            boundary=u_boundary,
            internal=u_internal,
        )
        expected_start = end
    if expected_start != n_substructure:
        raise ValueError("恢复缓存未覆盖全部子结构.")


def iter_exact_element_energy_batches(
    prototype: Any,
    density: Any,
    trace_displacement: Any,
    trace_basis: TraceBasis,
    *,
    chunk_size: int,
) -> Iterator[ElementStrainEnergyBatch]:
    """流式恢复内部位移并计算单位刚度单元应变能.

    Parameters
    ----------
    prototype : SubstructurePrototype
        同构子结构共享的参考子结构.
    density : TensorLike
        局部子结构密度批次, 形状约定见 ``SubstructurePrototype.to_cell_density``.
    trace_displacement : TensorLike
        各子结构的迹自由度位移, 形状 ``(n_substructure, n_trace)``.
    trace_basis : TraceBasis
        从迹自由度到完整接口自由度的线性映射.
    chunk_size : int
        单次处理的最大子结构数, 必须为正整数.

    Yields
    ------
    ElementStrainEnergyBatch
        当前连续批次的单位刚度单元应变能.

    Raises
    ------
    ValueError
        迹基, 密度批量或迹位移形状不一致.

    Notes
    -----
    建在 ``iter_exact_internal_displacement_batches`` 之上: 每批拼出局部位移
    后计算 ``u_e^T K_0 u_e``. 返回的是与 SIMP 插值系数解耦的单位刚度能量, 供
    柔顺度灵敏度计算复用. 位移批次用完即释放.
    """
    K0 = prototype.KE_unit[0]
    n_total = prototype.n_total_dofs
    for batch in iter_exact_internal_displacement_batches(
        prototype,
        density,
        trace_displacement,
        trace_basis,
        chunk_size=chunk_size,
    ):
        u_local = bm.zeros((batch.end - batch.start, n_total), dtype=bm.float64)
        u_local = bm.set_at(u_local, (slice(None), prototype.b_dofs), batch.boundary)
        u_local = bm.set_at(u_local, (slice(None), prototype.i_dofs), batch.internal)
        u_element = u_local[:, prototype.cell2dof]
        energy = bm.sum((u_element @ K0) * u_element, axis=-1)
        yield ElementStrainEnergyBatch(
            start=batch.start,
            end=batch.end,
            energy=energy,
        )
