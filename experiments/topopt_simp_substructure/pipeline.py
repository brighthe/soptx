# -*- coding: utf-8 -*-
"""soptx 公共组件组装 (接口迹 trace 参数化的精确子结构缩聚分析链).

本模块只负责两件事:

1. ``build_components``: 由工况构造 problem / GlobalAssembler / 子结构原型 /
   接口迹基, 并把与密度无关的量 (接口自由度编号、载荷、固定自由度) 一次算好;
2. ``solve_forward``: 给定全场单元密度, 完成 "局部装配 -> 精确 Schur 缩聚 ->
   接口系统装配与求解 -> 细观位移恢复 -> 单元应变能" 的一次正问题求解。

两条接口迹在局部张量生命周期、全局系统装配与自由度编号上分叉:

* ``full_trace``    保留全部接口自由度, 走 ``assemble_interface_system``;
* ``linear_corner`` 角点线性迹, 按 ``chunk_size`` 流式执行 Exact Schur、迹投影、
  宏观角点系统散加及求解后的内部位移恢复和单元应变能计算。

两条路径使用相同的 Exact Schur、恢复和能量公式, 只改变计算次序与迹空间。
优化循环、滤波与 OC 更新在 ``run.py``, 本模块不含任何优化状态。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from fealpy.backend import backend_manager as bm

CURRENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from soptx.fem.substructure import (  # noqa: E402
    ExactSchurReduction,
    FullTraceBasis,
    GlobalAssembler,
    LinearCornerTraceBasis,
    build_substructures,
    iter_exact_element_energy_batches,
    iter_exact_trace_stiffness_batches,
    project_problem_conditions_to_full_system,
    project_problem_conditions_to_macro_system,
    solve_interface_system,
)
from soptx.problems.elasticity import (  # noqa: E402
    CantileverCorner2d,
    FullMBBBeam2d,
    FullMBBBeam3d,
)

from config import TopOptCase


class _BatchCondensor:
    """把批量缩聚刚度适配成 ``GlobalAssembler`` 期望的缩聚器契约.

    ``normalize_condensors`` 只需要 ``K_s`` 形状为 ``(B, n_b, n_b)``;
    ``recover`` 仅在 ``recover_full_displacement`` 中使用, 这里一并提供,
    使本适配器可直接交给装配器的任一接口。
    """

    def __init__(self, result: Any) -> None:
        self.K_s = result.stiffness
        self._result = result

    def recover(self, boundary_displacement: Any) -> Any:
        return self._result.recover(boundary_displacement)


@dataclass
class CaseContext:
    """一个工况的全部与密度无关的分析对象."""

    case: TopOptCase
    problem: Any
    assembler: Any
    prototype: Any
    sub_meshes: Any
    trace_basis: Any
    reduction: Any
    # 几何与网格
    domain_size: Tuple[float, ...]
    n_elem_grid: Tuple[int, ...]
    n_elem_total: int
    h_grid: Tuple[float, ...]
    n_sub_total: int
    # 接口系统的自由度与条件 (随 trace 不同而不同, 但都与密度无关)
    load: Any
    fixed_dofs: Any
    trace_indices: Any          # (B, n_trace): 各子结构迹自由度在全局迹向量中的编号
    n_global_trace_dofs: int


@dataclass
class ForwardResult:
    """一次正问题求解的结果."""

    compliance: float
    energy: Any                 # (n_elem_total,) 各单元应变能 u_e^T K0 u_e
    cond_time_ms: float         # 局部数值缩聚与全局系统组装耗时 (ms)
    solve_time_ms: float        # 全局界面线性系统求解耗时 (ms)
    recover_time_ms: float      # 内部细观位移恢复与能量计算耗时 (ms)
    total_time_ms: float        # 正问题总耗时 (ms)


def _build_problem(case: TopOptCase) -> Tuple[Any, Tuple[float, ...]]:
    """构造弹性问题与物理域尺寸."""
    prob_name = getattr(case, "problem", "")
    if not prob_name:
        prob_name = "CantileverCorner2d" if case.dim == 2 else "FullMBBBeam3d"

    if prob_name == "CantileverCorner2d":
        problem = CantileverCorner2d(
            domain=case.domain, P=case.p_load, E=case.emax, nu=case.nu
        )
    elif prob_name == "FullMBBBeam2d":
        problem = FullMBBBeam2d(
            domain=case.domain, P=case.p_load, E=case.emax, nu=case.nu
        )
    elif prob_name == "FullMBBBeam3d":
        problem = FullMBBBeam3d(
            domain=case.domain, P=case.p_load, E=case.emax, nu=case.nu
        )
    else:
        raise ValueError(f"不支持的问题模型: {prob_name}")

    domain_size = tuple(
        case.domain[2 * d + 1] - case.domain[2 * d] for d in range(case.dim)
    )
    return problem, domain_size


def _build_trace_basis(case: TopOptCase, prototype: Any) -> Any:
    if case.trace == "full_trace":
        return FullTraceBasis.from_prototype(prototype)
    if case.trace == "linear_corner":
        return LinearCornerTraceBasis.from_prototype(prototype)
    raise ValueError(f"未知的接口迹: {case.trace}")


def _project_conditions_to_interface(
    problem: Any,
    assembler: Any,
    sub_meshes: Any,
    case: TopOptCase,
) -> Tuple[Any, Any, Any, int]:
    """把 problem 契约投影到 ``full_trace`` 的全部接口自由度.

    与 ``project_problem_conditions_to_macro_system`` 的区别是: 宏观版本投影到
    角点粗网格自由度, 这里投影到细网格上全部子结构边界自由度。载荷或位移约束
    一旦落在子结构内部自由度上, 完整接口系统无法表达, 此时直接报错而不是静默
    丢弃。

    返回 ``(f_interface, fixed_interface, b_interface, n_interface)``。
    """
    force_full, fixed_full = project_problem_conditions_to_full_system(
        problem,
        assembler,
    )

    interface_global_dofs = assembler.build_interface_dofs(sub_meshes)
    b_interface = assembler.interface_indices(sub_meshes, interface_global_dofs)

    interface_np = bm.to_numpy(interface_global_dofs)
    force_np = bm.to_numpy(force_full)

    on_interface = np.zeros(force_np.shape[0], dtype=bool)
    on_interface[interface_np] = True
    stray_load = np.nonzero((~on_interface) & (force_np != 0.0))[0]
    if stray_load.size > 0:
        raise ValueError(
            "存在落在子结构内部自由度上的等效节点载荷, 完整接口系统无法表达: "
            f"全局自由度 {stray_load[:8].tolist()} (共 {stray_load.size} 个). "
            "请调整子结构划分使载荷位置位于子结构边界上."
        )

    fixed_np = bm.to_numpy(fixed_full)
    positions = np.searchsorted(interface_np, fixed_np)
    positions = np.clip(positions, 0, max(interface_np.shape[0] - 1, 0))
    matched = interface_np[positions] == fixed_np
    if not bool(np.all(matched)):
        stray_fixed = fixed_np[~matched]
        raise ValueError(
            "存在落在子结构内部自由度上的位移约束, 完整接口系统无法表达: "
            f"全局自由度 {stray_fixed[:8].tolist()} (共 {stray_fixed.size} 个)."
        )

    f_interface = force_full[interface_global_dofs]
    fixed_interface = bm.asarray(positions, dtype=bm.int64)
    return (
        f_interface,
        fixed_interface,
        b_interface,
        int(interface_np.shape[0]),
    )


def build_components(case: TopOptCase) -> CaseContext:
    """构造工况的全部分析组件, 并预计算与密度无关的接口条件."""
    problem, domain_size = _build_problem(case)

    assembler = GlobalAssembler(
        domain_size, case.n_sub, case.n_fine, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, _positions = build_substructures(
        assembler,
        integration_order=case.integration_order,
    )
    prototype.rho_min = case.emin
    prototype.penal = case.simp_penalty

    trace_basis = _build_trace_basis(case, prototype)
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)

    n_elem_grid = tuple(
        case.n_sub[d] * case.n_fine[d] for d in range(case.dim)
    )
    h_grid = tuple(domain_size[d] / n_elem_grid[d] for d in range(case.dim))

    if case.trace == "full_trace":
        load, fixed_dofs, trace_indices, n_global = (
            _project_conditions_to_interface(
                problem, assembler, sub_meshes, case
            )
        )
    else:
        load, fixed_dofs = project_problem_conditions_to_macro_system(
            problem, assembler
        )
        trace_indices = assembler.macro_corner_indices(sub_meshes)
        n_global = int(assembler.total_macro_dofs)

    if int(trace_indices.shape[1]) != trace_basis.n_trace_dofs:
        raise ValueError(
            "迹自由度编号与迹基维数不一致: "
            f"{int(trace_indices.shape[1])} vs {trace_basis.n_trace_dofs}."
        )

    return CaseContext(
        case=case,
        problem=problem,
        assembler=assembler,
        prototype=prototype,
        sub_meshes=sub_meshes,
        trace_basis=trace_basis,
        reduction=reduction,
        domain_size=domain_size,
        n_elem_grid=n_elem_grid,
        n_elem_total=int(np.prod(n_elem_grid)),
        h_grid=h_grid,
        n_sub_total=len(sub_meshes),
        load=load,
        fixed_dofs=fixed_dofs,
        trace_indices=trace_indices,
        n_global_trace_dofs=n_global,
    )


def solve_forward(ctx: CaseContext, rho: Any) -> ForwardResult:
    """给定全场单元密度, 完成一次精确缩聚正问题求解并返回单元应变能与分项耗时."""
    t_start = time.perf_counter()

    # (a) 全场密度 -> 子结构局部密度
    rho_subs_grid = ctx.assembler.split_global_cell_field(rho)

    # (b) 接口系统装配与求解
    t_cond_start = time.perf_counter()
    if ctx.case.trace == "full_trace":
        # 完整接口门禁暂时保留原批量 Exact 路径. chunk_size 只限制局部装配
        # 临时缓冲, 尚不等价于端到端流式完整接口装配.
        rho_subs_cell = ctx.prototype.grid_to_cell_field(rho_subs_grid)
        K_local_batch = ctx.prototype.assemble_local_stiffness_batch(
            rho_subs_cell,
            chunk_size=ctx.case.chunk_size,
        )
        result = ctx.reduction.reduce_many(K_local_batch, rho_subs_grid)
        condensor = _BatchCondensor(result)
        system = ctx.assembler.assemble_interface_system(
            ctx.sub_meshes,
            condensor,
            chunk_size=ctx.case.chunk_size,
        )
        t_cond_ms = (time.perf_counter() - t_cond_start) * 1000.0

        t_solve_start = time.perf_counter()
        u_global_trace = solve_interface_system(
            system, ctx.load, ctx.fixed_dofs
        )
        t_solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

        t_rec_start = time.perf_counter()
        u_trace_batch = u_global_trace[ctx.trace_indices]
        u_b_batch = ctx.trace_basis.expand_displacement(u_trace_batch)
        u_i_batch = result.recover(u_b_batch)
        u_local_batch = bm.zeros(
            (ctx.n_sub_total, ctx.prototype.n_total_dofs), dtype=bm.float64
        )
        u_local_batch = bm.set_at(
            u_local_batch, (slice(None), ctx.prototype.b_dofs), u_b_batch
        )
        u_local_batch = bm.set_at(
            u_local_batch, (slice(None), ctx.prototype.i_dofs), u_i_batch
        )
        K0 = ctx.prototype.KE_unit[0]
        u_elem = u_local_batch[:, ctx.prototype.cell2dof]
        energy_sub_cell = bm.sum((u_elem @ K0) * u_elem, axis=-1)
        t_recover_ms = (time.perf_counter() - t_rec_start) * 1000.0
    else:
        stiffness_batches = iter_exact_trace_stiffness_batches(
            ctx.prototype,
            rho_subs_grid,
            ctx.trace_basis,
            chunk_size=ctx.case.chunk_size,
        )
        system = ctx.assembler.assemble_macro_system_batches(
            ctx.sub_meshes,
            stiffness_batches,
        )
        t_cond_ms = (time.perf_counter() - t_cond_start) * 1000.0

        t_solve_start = time.perf_counter()
        u_global_trace = solve_interface_system(
            system, ctx.load, ctx.fixed_dofs
        )
        t_solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

        t_rec_start = time.perf_counter()
        u_trace_batch = u_global_trace[ctx.trace_indices]
        # 只保存最终必要的 (B, NC) 能量场; 局部刚度、恢复矩阵和位移均逐批释放.
        energy_sub_cell = bm.zeros(
            (ctx.n_sub_total, ctx.prototype.n_cells),
            dtype=bm.float64,
        )
        for batch in iter_exact_element_energy_batches(
            ctx.prototype,
            rho_subs_grid,
            u_trace_batch,
            ctx.trace_basis,
            chunk_size=ctx.case.chunk_size,
        ):
            energy_sub_cell = bm.set_at(
                energy_sub_cell,
                (slice(batch.start, batch.end), slice(None)),
                batch.energy,
            )
        t_recover_ms = (time.perf_counter() - t_rec_start) * 1000.0

    energy_sub_grid = ctx.prototype.cell_to_grid_field(energy_sub_cell)
    energy_global_grid = ctx.assembler.merge_substructure_cell_field(
        energy_sub_grid
    )
    energy_flat = bm.reshape(energy_global_grid, (-1,))

    compliance = float(bm.dot(ctx.load, u_global_trace))
    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    return ForwardResult(
        compliance=compliance,
        energy=energy_flat,
        cond_time_ms=t_cond_ms,
        solve_time_ms=t_solve_ms,
        recover_time_ms=t_recover_ms,
        total_time_ms=total_time_ms,
    )


def compliance_sensitivity(ctx: CaseContext, rho: Any, energy: Any) -> Any:
    """modified SIMP 下的柔顺度灵敏度 ``dC/drho`` (未滤波)."""
    penal = ctx.case.simp_penalty
    dcoef = penal * (rho ** (penal - 1.0))
    if ctx.prototype.rho_min != 0.0:
        dcoef = (1.0 - ctx.prototype.rho_min) * dcoef
    return -dcoef * energy
