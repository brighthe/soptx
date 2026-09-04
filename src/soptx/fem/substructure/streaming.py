"""子结构局部缩聚与迹投影的流式编排."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from fealpy.backend import backend_manager as bm

from .reductions import ExactSchurReduction
from .traces import TraceBasis


@dataclass(frozen=True)
class TraceStiffnessBatch:
    """一个连续子结构批次在指定迹空间上的缩聚刚度.

    属性:
        start: 批次在展平子结构序列中的起始编号.
        end: 批次在展平子结构序列中的结束编号, 不包含该位置.
        stiffness: 迹空间缩聚刚度, 形状
            ``(end - start, n_trace, n_trace)``.
    """

    start: int
    end: int
    stiffness: Any


@dataclass(frozen=True)
class ElementStrainEnergyBatch:
    """一个连续子结构批次的单位刚度单元应变能.

    属性:
        start: 批次在展平子结构序列中的起始编号.
        end: 批次在展平子结构序列中的结束编号, 不包含该位置.
        energy: 按参考子结构 FE cell 编号排列的单位刚度单元应变能,
            形状 ``(end - start, n_cells)``.
    """

    start: int
    end: int
    energy: Any


def iter_exact_trace_stiffness_batches(
    prototype: Any,
    density: Any,
    trace_basis: TraceBasis,
    *,
    chunk_size: int,
) -> Iterator[TraceStiffnessBatch]:
    """流式执行局部刚度装配、Exact Schur 缩聚和迹投影.

    参数:
        prototype: 同构子结构共享的 ``SubstructurePrototype``.
        density: 局部子结构密度批次, 形状约定见
            ``SubstructurePrototype.to_cell_density``.
        trace_basis: 从迹自由度到完整接口自由度的线性映射.
        chunk_size: 单次处理的最大子结构数, 必须为正整数.

    生成:
        TraceStiffnessBatch: 当前连续批次的迹空间缩聚刚度.

    异常:
        ValueError: 当迹基的完整接口自由度数与参考子结构不一致时抛出.

    说明:
        每个批次严格执行有限元 Exact Schur 补, 不使用同质判定、阈值分类或
        PIML 代理. 完整局部刚度、完整接口 Schur 矩阵和恢复矩阵均只在当前
        批次生命周期内存在; 跨批次只向调用方传递投影后的迹空间刚度.
    """
    if trace_basis.n_boundary_dofs != prototype.n_b:
        raise ValueError(
            "trace_basis 的完整接口自由度数必须与 prototype.n_b 一致; "
            f"当前为 {trace_basis.n_boundary_dofs} 与 {prototype.n_b}."
        )

    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    for start, end, local_stiffness in prototype.iter_local_stiffness_batches(
        density,
        chunk_size=chunk_size,
    ):
        result = reduction.reduce_many(local_stiffness)
        projected = trace_basis.project_stiffness(result.stiffness)
        yield TraceStiffnessBatch(
            start=start,
            end=end,
            stiffness=projected,
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

    参数:
        prototype: 同构子结构共享的 ``SubstructurePrototype``.
        density: 局部子结构密度批次, 形状约定见
            ``SubstructurePrototype.to_cell_density``.
        trace_displacement: 各子结构的迹自由度位移, 形状
            ``(n_substructure, n_trace)``.
        trace_basis: 从迹自由度到完整接口自由度的线性映射.
        chunk_size: 单次处理的最大子结构数, 必须为正整数.

    生成:
        ElementStrainEnergyBatch: 当前连续批次的单位刚度单元应变能.

    异常:
        ValueError: 当迹基、密度批量或迹位移形状不一致时抛出.

    说明:
        全局迹系统求解后, 本方法按批重新装配局部刚度并执行 Exact Schur,
        使用该批恢复矩阵计算内部位移. ``recovery``, ``u_local`` 和单元位移
        均只在当前批次生命周期内存在. 返回的是与 SIMP 插值系数解耦的
        ``u_e^T K_0 u_e``, 供柔顺度灵敏度计算复用.
    """
    if trace_basis.n_boundary_dofs != prototype.n_b:
        raise ValueError(
            "trace_basis 的完整接口自由度数必须与 prototype.n_b 一致; "
            f"当前为 {trace_basis.n_boundary_dofs} 与 {prototype.n_b}."
        )

    rho_cells = prototype.to_cell_density(density)
    rho_flat = bm.reshape(rho_cells, (-1, prototype.n_cells))
    displacement = bm.asarray(trace_displacement)
    expected_shape = (rho_flat.shape[0], trace_basis.n_trace_dofs)
    if tuple(displacement.shape) != expected_shape:
        raise ValueError(
            f"trace_displacement 形状必须为 {expected_shape}; "
            f"当前为 {tuple(displacement.shape)}."
        )

    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)
    K0 = prototype.KE_unit[0]
    for start, end, local_stiffness in prototype.iter_local_stiffness_batches(
        rho_flat,
        chunk_size=chunk_size,
    ):
        result = reduction.reduce_many(local_stiffness)
        u_boundary = trace_basis.expand_displacement(displacement[start:end])
        u_internal = result.recover(u_boundary)

        u_local = bm.zeros(
            (end - start, prototype.n_total_dofs),
            dtype=local_stiffness.dtype,
        )
        u_local = bm.set_at(
            u_local,
            (slice(None), prototype.b_dofs),
            u_boundary,
        )
        u_local = bm.set_at(
            u_local,
            (slice(None), prototype.i_dofs),
            u_internal,
        )
        u_element = u_local[:, prototype.cell2dof]
        energy = bm.sum((u_element @ K0) * u_element, axis=-1)

        yield ElementStrainEnergyBatch(
            start=start,
            end=end,
            energy=energy,
        )
