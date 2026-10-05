"""子结构局部缩聚与迹投影的流式编排."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from fealpy.backend import backend_manager as bm

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
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    for start, end, local_stiffness in prototype.iter_local_stiffness_batches(
        density,
        chunk_size=chunk_size,
    ):
        result = reduction.reduce_many(local_stiffness)
        yield LocalCondensationBatch(
            start=start,
            end=end,
            stiffness=result.stiffness,
            recovery=result.recovery,
        )


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

    for batch in iter_exact_condensation_batches(
        prototype,
        density,
        chunk_size=chunk_size,
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


def iter_exact_internal_displacement_batches(
    prototype: Any,
    density: Any,
    trace_displacement: Any,
    trace_basis: TraceBasis,
    *,
    chunk_size: int,
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
    全局接口系统求解后, 本方法按批重新装配局部刚度并执行 Exact Schur, 用该批
    的 ``T_full`` 计算 ``u_i = T_full Psi q``. 恢复矩阵不跨批保存, 以重算换
    内存. 这是流式路线下位移恢复的唯一实现, 单元应变能与全尺度位移向量都建在
    它之上.
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

    for batch in iter_exact_condensation_batches(
        prototype,
        rho_cells,
        chunk_size=chunk_size,
    ):
        u_boundary = trace_basis.expand_displacement(
            displacement[batch.start:batch.end]
        )
        u_internal = bm.einsum("...ij,...j->...i", batch.recovery, u_boundary)
        yield InternalDisplacementBatch(
            start=batch.start,
            end=batch.end,
            boundary=u_boundary,
            internal=u_internal,
        )


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
