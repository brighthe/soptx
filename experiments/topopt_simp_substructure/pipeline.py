# -*- coding: utf-8 -*-
"""soptx 公共组件组装 (接口迹 trace 参数化的精确子结构缩聚分析链).

本模块只负责两件事:

1. ``build_components``: 由工况构造 problem / GlobalAssembler / 子结构原型 /
   接口迹基, 并把与密度无关的量 (接口自由度编号、载荷、固定自由度) 一次算好;
2. ``solve_forward``: 给定全场单元密度, 完成 "局部装配 -> 精确 Schur 缩聚 ->
   接口系统装配与求解 -> 细观位移恢复 -> 单元应变能" 的一次正问题求解。

两种接口迹 (``full_trace`` 与 ``linear_corner``) 走同一条流式路线: 按
``chunk_size`` 流式执行 Exact Schur、迹投影、全局接口散加, 求解后再按块重新
缩聚以恢复内部位移并计算单元应变能。接口空间相关的量 (迹基 ``Psi``、全局
编号 ``A_q^j``、``N_q``、载荷与约束投影、装配入口) 统一由 ``InterfaceSpace``
提供, 本模块不含任何按接口迹种类的分叉。

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

from soptx.backend import backend_manager as bm

CURRENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from soptx.fem.substructure import (  # noqa: E402
    ExactSchurReduction,
    GlobalAssembler,
    build_interface_space,
    build_substructures,
    iter_exact_element_energy_batches,
    iter_exact_trace_stiffness_batches,
    solve_interface_system,
)
from soptx.problems.elasticity import (  # noqa: E402
    CantileverCorner2d,
    FullMBBBeam2d,
    FullMBBBeam3d,
)

from config import TopOptCase


@dataclass
class CaseContext:
    """一个工况的全部与密度无关的分析对象."""

    case: TopOptCase
    problem: Any
    assembler: Any
    prototype: Any
    sub_meshes: Any
    interface_space: Any        # InterfaceSpace: 迹基 Psi, 全局编号 A_q^j 与 N_q
    reduction: Any
    # 几何与网格
    domain_size: Tuple[float, ...]
    n_elem_grid: Tuple[int, ...]
    n_elem_total: int
    h_grid: Tuple[float, ...]
    n_sub_total: int
    # 接口系统上的载荷与约束 (由 interface_space 投影, 与密度无关)
    load: Any
    fixed_dofs: Any


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

    interface_space = build_interface_space(
        kind=case.trace,
        assembler=assembler,
        sub_meshes=sub_meshes,
        prototype=prototype,
    )
    reduction = ExactSchurReduction(prototype.i_dofs, prototype.b_dofs)

    n_elem_grid = tuple(
        case.n_sub[d] * case.n_fine[d] for d in range(case.dim)
    )
    h_grid = tuple(domain_size[d] / n_elem_grid[d] for d in range(case.dim))

    load, fixed_dofs = interface_space.project_conditions(problem)

    return CaseContext(
        case=case,
        problem=problem,
        assembler=assembler,
        prototype=prototype,
        sub_meshes=sub_meshes,
        interface_space=interface_space,
        reduction=reduction,
        domain_size=domain_size,
        n_elem_grid=n_elem_grid,
        n_elem_total=int(np.prod(n_elem_grid)),
        h_grid=h_grid,
        n_sub_total=len(sub_meshes),
        load=load,
        fixed_dofs=fixed_dofs,
    )


def solve_forward(ctx: CaseContext, rho: Any) -> ForwardResult:
    """给定全场单元密度, 完成一次精确缩聚正问题求解并返回单元应变能与分项耗时."""
    t_start = time.perf_counter()

    # (a) 全场密度 -> 子结构局部密度
    rho_subs_grid = ctx.assembler.split_global_cell_field(rho)

    # (b) 局部缩聚, 迹投影与全局接口装配: 两种接口空间走同一条流式路线,
    #     差别全部封装在 ctx.interface_space 中.
    t_cond_start = time.perf_counter()
    stiffness_batches = iter_exact_trace_stiffness_batches(
        ctx.prototype,
        rho_subs_grid,
        ctx.interface_space.trace_basis,
        chunk_size=ctx.case.chunk_size,
    )
    system = ctx.interface_space.assemble(stiffness_batches)
    t_cond_ms = (time.perf_counter() - t_cond_start) * 1000.0

    # (c) 全局接口求解
    t_solve_start = time.perf_counter()
    u_global_trace = solve_interface_system(system, ctx.load, ctx.fixed_dofs)
    t_solve_ms = (time.perf_counter() - t_solve_start) * 1000.0

    # (d) 流式位移恢复与单元应变能: 按块重新装配并缩聚, 恢复矩阵不跨块保存.
    #     只保存最终必要的 (B, NC) 能量场.
    t_rec_start = time.perf_counter()
    u_trace_batch = ctx.interface_space.trace_displacement(u_global_trace)
    energy_sub_cell = bm.zeros(
        (ctx.n_sub_total, ctx.prototype.n_cells),
        dtype=bm.float64,
    )
    for batch in iter_exact_element_energy_batches(
        ctx.prototype,
        rho_subs_grid,
        u_trace_batch,
        ctx.interface_space.trace_basis,
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
