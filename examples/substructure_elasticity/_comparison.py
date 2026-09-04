"""精确子结构静力缩聚的共享比较与一致性验证实现.

在同一网格、材料、密度、载荷和支承下比较全结构 FA、完整接口 Exact Schur
与角点线性迹 Exact Schur. 只有完整接口路径对 FA 使用 1e-11 等价性门禁;
角点路径报告近似误差, 以降阶平衡、约束、恢复及能量一致性验收.
三维角点约束若涉及多个角点, 保留原约束并用稀疏乘子系统求解.
默认三路径独立装配和消元, 单次耗时仅供辅助判断. 性能比较通过公开脚本
选择两条或三条路径, 每条路径的每个样本均在独立新进程中串行执行.

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析, 与从哪个目录发起命令无关;
传相对路径时按当前工作目录解析, 可能落到 ``.gitignore`` 覆盖范围之外.

``--n-sub`` 和 ``--n-fine`` 分别指定各方向的子结构数与每块单元数,
各接收 ``dim`` 个正整数. 不指定时沿用当前维度的默认规模.
一致性结果名包含维度及划分, 相同规模重跑覆盖旧文件; 性能证据另含
密度模式和 UTC 时间戳, 不覆盖旧结果.
"""

import json
import time
import hashlib
import os
import platform
import warnings
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import numpy as np
from scipy.linalg import qr
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, norm as sparse_norm, spsolve as scipy_spsolve
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.problems.elasticity import HalfMBBBeamRight2d, FullMBBBeam3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    LinearCornerTraceBasis,
    SubstructureMesh,
    SubstructurePrototype,
    solve_interface_system,
)


from examples.substructure_elasticity._common import (
    CONSISTENCY_TOLERANCE,
    DEFAULT_DENSITY,
    DEFAULT_REPEAT,
    DEFAULT_WARMUP,
    RELATIVE_ERROR_TOLERANCE,
    print_table,
    resolve_grid,
)


def result_filename(
    dimension: str, n_sub: Sequence[int], n_fine: Sequence[int]
) -> str:
    """按维度、子结构划分与局部单元划分生成结果文件名."""
    sub = "x".join(str(n) for n in n_sub)
    fine = "x".join(str(n) for n in n_fine)
    return f"lagrange_comparison_{dimension.lower()}_sub-{sub}_fine-{fine}.json"


### 验收 ###

def validate_and_write_result(result: Dict[str, Any], output_dir: str | None) -> None:
    """分别验收完整接口等价性与角点路径一致性, 再落盘证据.

    参数:
        result: 单个算例的全部统计指标.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.

    异常:
        AssertionError: 当柔度或位移的相对误差超出 ``RELATIVE_ERROR_TOLERANCE``
            时抛出.
    """
    for key, label in (
        ("compliance_relative_error", "compliance"),
        ("displacement_relative_error", "displacement"),
    ):
        if not np.isfinite(result[key]) or result[key] > RELATIVE_ERROR_TOLERANCE:
            raise AssertionError(
                f"{result['dimension']} {label} relative error "
                f"{result[key]:.4e} exceeds {RELATIVE_ERROR_TOLERANCE:.1e}"
            )

    corner = result["linear_corner"]
    for key in (
        "equilibrium_relative_residual", "constraint_relative_residual",
        "recovered_support_relative_error", "internal_relative_residual",
        "energy_relative_error", "load_work_relative_error",
        "galerkin_relative_error", "compliance_order_violation",
    ):
        value = corner[key]
        if not np.isfinite(value) or value > CONSISTENCY_TOLERANCE:
            raise AssertionError(
                f"linear_corner {key}={value:.4e} 超过一致性阈值 "
                f"{CONSISTENCY_TOLERANCE:.1e}; 此门禁不是相对 FA 的近似误差门禁."
            )
    for key in ("compliance_relative_error", "displacement_relative_error"):
        if not np.isfinite(corner[key]):
            raise AssertionError(f"linear_corner {key} 不是有限值.")
    full_residual = result["full_trace_equilibrium_relative_residual"]
    if not np.isfinite(full_residual) or full_residual > CONSISTENCY_TOLERANCE:
        raise AssertionError(f"full_trace 自由接口平衡残差异常: {full_residual:.4e}")
    result["validation"] = {
        "full_trace_equivalence": "PASS",
        "linear_corner_consistency": "PASS",
        "linear_corner_fa_error": "reported_only",
    }

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / result_filename(
            result["dimension"], result["n_sub"], result["n_fine"]
        )
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"[证据] 验收通过, 结果已写入: {target}")


### 子结构构造与密度场 ###

def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh], List[Tuple[int, ...]]]:
    """按装配器的布局铺开全部子结构, 共享同一个参考子结构.

    参数:
        assembler: 已构造的全局装配器, 提供求解域尺寸与子结构划分.

    返回:
        (prototype, sub_meshes, positions): 共享的参考子结构, 子结构列表, 以及
            各子结构在子结构网格中的整数位置, 三者按 x 优先的字典序同序排列.

    说明:
        字典序是 ``reconstruct_global_field`` 的次序契约; 涉及网格的装配接口则
        由 ``box_span`` 反解位置, 不依赖列表次序.
    """
    dim = assembler.dim
    sub_size = tuple(
        assembler.domain_size[d] / assembler.n_sub[d] for d in range(dim)
    )
    prototype = SubstructurePrototype(
        sub_size, assembler.n_fine, assembler.E_base, assembler.nu
    )

    grid = (
        [(sx, sy) for sx in range(assembler.n_sub[0]) for sy in range(assembler.n_sub[1])]
        if dim == 2
        else [
            (sx, sy, sz)
            for sx in range(assembler.n_sub[0])
            for sy in range(assembler.n_sub[1])
            for sz in range(assembler.n_sub[2])
        ]
    )
    sub_meshes: List[SubstructureMesh] = []
    for sub_id, pos in enumerate(grid):
        spans = tuple(
            (pos[d] * sub_size[d], (pos[d] + 1) * sub_size[d]) for d in range(dim)
        )
        sub_meshes.append(
            SubstructureMesh(
                sub_id, *spans, *assembler.n_fine,
                assembler.E_base, assembler.nu, prototype=prototype,
            )
        )
    return prototype, sub_meshes, grid


def make_density_fields(
    sub_meshes: Sequence[SubstructureMesh],
    domain_size: Sequence[float],
) -> Any:
    """按子结构中心坐标生成一批平滑变化的局部密度场.

    参数:
        sub_meshes: 子结构列表.
        domain_size: 各方向的求解域尺寸.

    返回:
        density: 形状 ``(B, *n_fine)`` 的批量密度场, 前导维 ``B`` 与
            ``sub_meshes`` 同序; 每个子结构内部密度均匀.
    """
    dim = len(domain_size)
    centers = bm.asarray(
        [
            [(sm.box_span[d][0] + sm.box_span[d][1]) / 2.0 for d in range(dim)]
            for sm in sub_meshes
        ],
        dtype=bm.float64,
    )
    scaled = bm.pi * centers / bm.asarray(domain_size, dtype=bm.float64)
    modulation = bm.sin(scaled[:, 0]) * bm.cos(scaled[:, 1])
    if dim == 3:
        modulation = modulation * bm.sin(scaled[:, 2])
    rho = 0.7 + 0.3 * modulation

    n_fine = tuple(sub_meshes[0].n_fine)
    rho = bm.reshape(rho, (len(sub_meshes),) + (1,) * dim)
    return bm.broadcast_to(rho, (len(sub_meshes),) + n_fine)


def build_corner_projection(
    assembler: GlobalAssembler,
    sub_meshes: List[Any],
    interface_dofs: Any,
    trace_basis: Any,
) -> Any:
    """构造角点位移到去重完整接口位移的稀疏插值矩阵.

    共享接口行只取一次, 并检查相邻子结构给出的插值是否相同.
    """
    boundary = np.asarray(
        bm.to_numpy(assembler.interface_indices(sub_meshes, interface_dofs)),
        dtype=np.int64,
    )
    corners = np.asarray(
        bm.to_numpy(assembler.macro_corner_indices(sub_meshes)), dtype=np.int64
    )
    local = np.asarray(bm.to_numpy(trace_basis.matrix), dtype=np.float64)
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
        shape=(n_batch * n_boundary, assembler.total_macro_dofs),
    ).tocsr()
    global_rows, first = np.unique(boundary.ravel(), return_index=True)
    if not np.array_equal(global_rows, np.arange(len(interface_dofs))):
        raise ValueError("角点插值未覆盖完整接口.")
    projection = candidates[first].tocsr()
    difference = candidates - projection[boundary.ravel()]
    if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-12:
        raise ValueError("相邻子结构在共享接口上给出了不一致的角点插值.")
    return projection


def solve_corner_system(
    system: Any, force: Any, constraints: Any
) -> Tuple[Any, int, float, float, str]:
    """求解齐次约束下的角点系统, 不将混合约束近似为角点固定.

    单自由度约束复用公共直接求解器; 其余约束选取独立行后求解稀疏
    Lagrange 乘子系统. QR 仅作用于约束中实际出现的角点列.
    返回位移、独立约束数、平衡残差、约束残差与求解方式.
    """
    stiffness = system.stiffness.to_scipy().tocsr()
    constraints = constraints.copy().tocsr()
    constraints.eliminate_zeros()
    counts = np.diff(constraints.indptr)
    if np.any(counts == 0):
        raise ValueError("原始固定自由度在角点空间中没有对应约束.")
    force = np.asarray(force, dtype=np.float64)
    if np.all(counts == 1):
        fixed = np.unique(constraints.indices)
        q = np.asarray(bm.to_numpy(solve_interface_system(
            system, bm.asarray(force), bm.asarray(fixed, dtype=bm.int64)
        )))
        rank = len(fixed)
        free = np.setdiff1d(np.arange(len(q)), fixed)
        residual = (stiffness @ q - force)[free]
        scale = max(float(np.linalg.norm(force[free])), np.finfo(float).tiny)
        mode = "固定角点消元"
    else:
        active = np.unique(constraints.indices)
        _, triangular, pivots = qr(
            constraints[:, active].toarray().T, mode="economic", pivoting=True
        )
        diagonal = np.abs(np.diag(triangular))
        threshold = (
            max(constraints.shape[0], len(active))
            * np.finfo(float).eps * (diagonal.max() if diagonal.size else 0.0)
        )
        rank = int(np.count_nonzero(diagonal > threshold))
        independent = constraints[pivots[:rank]]
        saddle = bmat(
            [[stiffness, independent.T], [independent, None]], format="csc"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solved = scipy_spsolve(saddle, np.concatenate((force, np.zeros(rank))))
        if not np.all(np.isfinite(solved)):
            raise AssertionError("角点乘子系统产生非有限解.")
        q, multipliers = solved[:len(force)], solved[len(force):]
        residual = stiffness @ q - force + independent.T @ multipliers
        scale = max(float(np.linalg.norm(force)), np.finfo(float).tiny)
        mode = "一般线性约束 / 稀疏乘子系统"
    if not np.all(np.isfinite(q)):
        raise AssertionError("角点系统产生非有限位移, 请检查支承与矩阵可解性.")
    balance = float(np.linalg.norm(residual)) / scale
    constraint_error = float(np.linalg.norm(constraints @ q)) / max(
        float(np.linalg.norm(q)), np.finfo(float).tiny
    )
    return bm.asarray(q, dtype=bm.float64), rank, balance, constraint_error, mode


def analyze_linear_corner(
    assembler: GlobalAssembler,
    prototype: SubstructurePrototype,
    sub_meshes: List[Any],
    density: Any,
    full_force: Any,
    full_fixed: Any,
    *,
    peak_memory_reader: Callable[[], int] | None = None,
) -> Tuple[Any, Dict[str, Any], Any, Any]:
    """独立完成角点路径的装配、精确缩聚、条件投影、求解与全场恢复.

    校验用局部能量与内部平衡统计不计入分析耗时; 不复用完整接口路径的刚度.
    """
    phases: Dict[str, float] = {}
    start = time.perf_counter()
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
    phases["setup"] = time.perf_counter() - start
    tick = time.perf_counter()
    local_stiffness = prototype.assemble_local_stiffness_batch(density)
    phases["assembly"] = time.perf_counter() - tick
    tick = time.perf_counter()
    condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor.condense(local_stiffness)
    phases["condensation"] = time.perf_counter() - tick
    tick = time.perf_counter()
    macro = assembler.assemble_macro_system(
        sub_meshes, trace_basis.project_stiffness(condensor.K_s)
    )
    interface_dofs = assembler.build_interface_dofs(sub_meshes)
    # 仅恢复时使用完整接口编号, 不装配另一个完整接口刚度.
    interface_view = SimpleNamespace(global_dofs=interface_dofs)
    projection = build_corner_projection(
        assembler, sub_meshes, interface_dofs, trace_basis
    )
    phases["interface_assembly"] = time.perf_counter() - tick
    tick = time.perf_counter()
    interface_force = np.asarray(bm.to_numpy(
        assembler.project_global_vector(interface_view, full_force)
    ))
    fixed_interface = np.asarray(bm.to_numpy(
        assembler.project_global_dofs(interface_view, full_fixed)
    ), dtype=np.int64)
    # 以原完整载荷的虚功为准; 不在宏观角点上重新生成物理载荷.
    force = projection.T @ interface_force
    constraints = projection[fixed_interface]
    q, rank, balance, constraint_error, mode = solve_corner_system(
        macro, force, constraints
    )
    phases["conditions_solve"] = time.perf_counter() - tick
    tick = time.perf_counter()
    interface_u = bm.asarray(projection @ bm.to_numpy(q), dtype=bm.float64)
    displacement = assembler.recover_full_displacement(
        sub_meshes, condensor, interface_view, interface_u
    )
    phases["recovery"] = time.perf_counter() - tick
    phases["total"] = time.perf_counter() - start
    peak_rss = None if peak_memory_reader is None else peak_memory_reader()
    if peak_rss is not None and (not isinstance(peak_rss, int) or peak_rss <= 0):
        raise AssertionError("峰值 RSS 必须为正整数字节数.")

    boundary_indices = assembler.interface_indices(sub_meshes, interface_dofs)
    u_b = interface_u[boundary_indices]
    u_i = condensor.recover(u_b)
    local_u = bm.zeros(
        (len(sub_meshes), prototype.n_total_dofs), dtype=bm.float64
    )
    local_u = bm.set_at(local_u, (slice(None), prototype.b_dofs), u_b)
    local_u = bm.set_at(local_u, (slice(None), prototype.i_dofs), u_i)
    local_force = bm.einsum("bij,bj->bi", local_stiffness, local_u)
    energy = float(bm.sum(local_u * local_force))
    compliance = float(bm.sum(full_force * displacement))
    projected_work = float(np.dot(force, bm.to_numpy(q)))
    response_scale = max(abs(compliance), np.finfo(float).tiny)
    internal_residual = float(bm.linalg.norm(local_force[:, prototype.i_dofs]))
    internal_scale = max(float(bm.linalg.norm(local_force)), np.finfo(float).tiny)
    fixed_error = float(bm.linalg.norm(displacement[full_fixed])) / max(
        float(bm.linalg.norm(displacement)), np.finfo(float).tiny
    )
    # Galerkin 恒等式属于实现一致性诊断, 不计入路径耗时和峰值 RSS.
    full_system = assembler.assemble_interface_system(sub_meshes, condensor)
    projected_stiffness = projection.T @ full_system.stiffness.to_scipy().tocsr() @ projection
    corner_stiffness = macro.stiffness.to_scipy().tocsr()
    galerkin_error = float(sparse_norm(
        corner_stiffness - projected_stiffness
    )) / max(float(sparse_norm(projected_stiffness)), np.finfo(float).tiny)
    stats = {
        "trace_dofs": assembler.total_macro_dofs,
        "free_dofs": assembler.total_macro_dofs - rank,
        "constraint_rank": rank,
        "local_trace_dofs": trace_basis.n_trace_dofs,
        "compliance": compliance,
        "seconds": phases["total"],
        "phase_seconds": phases,
        "memory_peak_rss_bytes": peak_rss,
        "equilibrium_relative_residual": balance,
        "constraint_relative_residual": constraint_error,
        "recovered_support_relative_error": fixed_error,
        "internal_relative_residual": internal_residual / internal_scale,
        "energy_relative_error": abs(energy - compliance) / response_scale,
        "load_work_relative_error": abs(projected_work - compliance) / response_scale,
        "galerkin_relative_error": galerkin_error,
        "constraint_solver": mode,
    }
    return displacement, stats, projection, corner_stiffness


LINEAR_CORNER_CONSISTENCY_KEYS = (
    "equilibrium_relative_residual",
    "constraint_relative_residual",
    "recovered_support_relative_error",
    "internal_relative_residual",
    "energy_relative_error",
    "load_work_relative_error",
    "galerkin_relative_error",
)


def validate_linear_corner_consistency(record: Dict[str, Any]) -> None:
    """验收角点迹离散自身的一致性, 不把相对 FA 的近似误差作为门禁."""
    for key in LINEAR_CORNER_CONSISTENCY_KEYS:
        value = record[key]
        if not np.isfinite(value) or value > CONSISTENCY_TOLERANCE:
            raise AssertionError(
                f"linear_corner {key}={value:.4e} 超过一致性阈值 "
                f"{CONSISTENCY_TOLERANCE:.1e}."
            )


### 基准主体 ###

def make_mbb_problem(dim: int) -> Any:
    """构造三路径比较和重复计时共用的 MBB 梁问题."""
    if dim == 2:
        return HalfMBBBeamRight2d(
            domain=(0.0, 60.0, 0.0, 20.0), P=-1.0, E=1.0, nu=0.3,
        )
    if dim == 3:
        return FullMBBBeam3d(
            domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0), P=-1.0, E=1.0, nu=0.3,
        )
    raise ValueError("dim 必须为 2 或 3.")


def make_fa_analyzer(mesh: Any, pde: Any, material: Any) -> LagrangeFEMAnalyzer:
    """构造一阶 SIMP 全装配分析器, 不复用数值刚度或分解."""
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=pde, material=material, space_degree=1,
        assembly_method='standard', operator_level='fa', solve_method='scipy',
        topopt_algorithm='density_based',
        interpolation_scheme=MaterialInterpolationScheme(
            density_location='element', interpolation_method='simp',
            options={'penalty_factor': 3.0, 'stress_penalty_factor': 1.0},
        ),
        enable_logging=False,
    )


def make_cell_density(shape: Sequence[int], mode: str) -> Any:
    """按细单元中心采样密度, 同一细网格上的结果与子结构划分无关.

    Parameters
    ----------
    shape : sequence of int
        各方向细单元数, 采用结构化场 C 序.
    mode : str
        cell 为固定光滑函数, uniform 为均匀密度 0.7.

    Returns
    -------
    TensorLike
        形状为 shape 的单元密度场.
    """
    if mode == 'uniform':
        return bm.full(tuple(shape), 0.7, dtype=bm.float64)
    if mode != 'cell':
        raise ValueError("性能模式的 density 必须为 cell 或 uniform.")
    coords = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in shape], indexing='ij')
    modulation = np.sin(np.pi * coords[0]) * np.cos(np.pi * coords[1])
    if len(shape) == 3:
        modulation *= np.sin(np.pi * coords[2])
    return bm.asarray(0.7 + 0.3 * modulation, dtype=bm.float64)


def timed_full_analysis(
    route: str, pde: Any, reference: GlobalAssembler, rho_global: Any,
    full_force: Any, fixed_global: Any, *,
    peak_memory_reader: Callable[[], int] | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """重新装配并求解一条路径, 返回位移与计时及平衡指标.

    Parameters
    ----------
    route : str
        fa、full_trace 或 linear_corner.
    pde : object
        同源物理问题.
    reference : GlobalAssembler
        提供共享细网格、材料与划分, 不提供数值刚度.
    rho_global, full_force, fixed_global : TensorLike
        不随重复次数改变的细网格密度、完整载荷和固定自由度.
    peak_memory_reader : callable or None
        在分析结束且诊断尚未开始时读取进程峰值 RSS, 返回字节数.

    Returns
    -------
    displacement : TensorLike
        完整位移向量.
    record : dict
        分项时间、自由度、柔顺度、自由系统残差与支承误差.
    """
    phases: Dict[str, float] = {}
    start = time.perf_counter()
    if route == 'fa':
        analyzer = make_fa_analyzer(reference.full_mesh, pde, reference.material)
        phases['setup'] = time.perf_counter() - start
        tick = time.perf_counter()
        stiffness = analyzer.assemble_stiff_matrix(
            rho_val=bm.reshape(rho_global, (-1,))
        ).tocsr()
        system = SimpleNamespace(
            stiffness=stiffness,
            global_dofs=bm.arange(reference.total_full_dofs, dtype=bm.int64),
        )
        phases['assembly'] = time.perf_counter() - tick
        tick = time.perf_counter()
        load, fixed = full_force, fixed_global
        # FA 也提取自由子矩阵, 与 Schur 使用相同的约束处理和直接求解器.
        reduced_u = solve_interface_system(system, load, fixed, solver='scipy')
        displacement = reduced_u
        phases['conditions_solve'] = time.perf_counter() - tick
    elif route == 'full_trace':
        # 每次新建映射与原型, 将本路径独有初始化计入, 不跨次复用 Schur 补.
        assembler = GlobalAssembler(
            reference.domain_size, reference.n_sub, reference.n_fine,
            E_base=pde.E, nu=pde.nu,
        )
        prototype, sub_meshes, _ = build_substructures(assembler)
        density = assembler.split_global_cell_field(rho_global)
        phases['setup'] = time.perf_counter() - start
        tick = time.perf_counter()
        local_stiffness = prototype.assemble_local_stiffness_batch(density)
        phases['assembly'] = time.perf_counter() - tick
        tick = time.perf_counter()
        condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
        condensor.condense(local_stiffness)
        phases['condensation'] = time.perf_counter() - tick
        tick = time.perf_counter()
        system = assembler.assemble_interface_system(sub_meshes, condensor)
        phases['interface_assembly'] = time.perf_counter() - tick
        tick = time.perf_counter()
        load = assembler.project_global_vector(system, full_force)
        fixed = assembler.project_global_dofs(system, fixed_global)
        reduced_u = solve_interface_system(system, load, fixed, solver='scipy')
        phases['conditions_solve'] = time.perf_counter() - tick
        tick = time.perf_counter()
        displacement = assembler.recover_full_displacement(
            sub_meshes, condensor, system, reduced_u
        )
        phases['recovery'] = time.perf_counter() - tick
    elif route == 'linear_corner':
        # 角点迹保留精确局部 Schur 缩聚, 仅接口迹空间采用线性角点插值.
        assembler = GlobalAssembler(
            reference.domain_size, reference.n_sub, reference.n_fine,
            E_base=pde.E, nu=pde.nu,
        )
        prototype, sub_meshes, _ = build_substructures(assembler)
        density = assembler.split_global_cell_field(rho_global)
        outer_setup = time.perf_counter() - start
        displacement, corner, _, _ = analyze_linear_corner(
            assembler, prototype, sub_meshes, density, full_force, fixed_global,
            peak_memory_reader=peak_memory_reader,
        )
        phases = dict(corner['phase_seconds'])
        phases['setup'] += outer_setup
        phases['total'] += outer_setup
        record = {
            'seconds': phases,
            'compliance': corner['compliance'],
            'system_dofs': corner['trace_dofs'],
            'free_dofs': corner['free_dofs'],
            'equilibrium_relative_residual': corner['equilibrium_relative_residual'],
            'support_relative_error': corner['recovered_support_relative_error'],
            'memory_peak_rss_bytes': corner['memory_peak_rss_bytes'],
            'constraint_rank': corner['constraint_rank'],
            'local_trace_dofs': corner['local_trace_dofs'],
            'constraint_solver': corner['constraint_solver'],
            **{key: corner[key] for key in LINEAR_CORNER_CONSISTENCY_KEYS},
        }
        validate_linear_corner_consistency(record)
        return displacement, record
    else:
        raise ValueError(f"未知计时路径: {route}")
    phases['total'] = time.perf_counter() - start
    # 必须在残差数组、跨路径比较和结果序列化之前读取.
    peak_rss = None if peak_memory_reader is None else peak_memory_reader()
    if peak_rss is not None and (not isinstance(peak_rss, int) or peak_rss <= 0):
        raise AssertionError("峰值 RSS 必须为正整数字节数.")

    # 以下诊断与统计全部在计时区间之外.
    u = np.asarray(bm.to_numpy(displacement)).reshape(-1)
    q = np.asarray(bm.to_numpy(reduced_u)).reshape(-1)
    f = np.asarray(bm.to_numpy(load)).reshape(-1)
    fixed_np = np.asarray(bm.to_numpy(fixed), dtype=np.int64)
    free = np.setdiff1d(np.arange(len(q)), fixed_np)
    residual = (system.stiffness.to_scipy() @ q - f)[free]
    residual_error = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(f[free])), np.finfo(float).tiny
    )
    support_error = float(np.linalg.norm(u[bm.to_numpy(fixed_global)])) / max(
        float(np.linalg.norm(u)), np.finfo(float).tiny
    )
    compliance = float(np.dot(bm.to_numpy(full_force), u))
    if not np.all(np.isfinite(u)) or not np.isfinite(compliance) or compliance <= 0:
        raise AssertionError(f"{route} 产生非有限位移或非正柔顺度.")
    for label, value in (('equilibrium', residual_error), ('support', support_error)):
        if not np.isfinite(value) or value > CONSISTENCY_TOLERANCE:
            raise AssertionError(f"{route} {label}={value:.4e} 超过一致性阈值.")
    if any(not np.isfinite(value) or value < 0 for value in phases.values()) or phases['total'] <= 0:
        raise AssertionError(f"{route} 计时记录非法.")
    return displacement, {
        'seconds': phases, 'compliance': compliance, 'system_dofs': len(q),
        'free_dofs': len(free), 'equilibrium_relative_residual': residual_error,
        'support_relative_error': support_error, 'memory_peak_rss_bytes': peak_rss,
    }


def timing_statistics(values: Sequence[float]) -> Dict[str, float]:
    """汇总重复测量, 四分位点采用 NumPy 默认线性插值."""
    data = np.asarray(values, dtype=float)
    return {
        'median': float(np.median(data)), 'q25': float(np.quantile(data, 0.25)),
        'q75': float(np.quantile(data, 0.75)), 'min': float(np.min(data)),
        'max': float(np.max(data)),
    }


def array_fingerprint(array: Any) -> str:
    """计算连续数组的数据指纹, 避免额外创建同样大小的 bytes 副本."""
    data = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()


def prepare_performance_problem(
    dim: int, n_sub: Sequence[int], n_fine: Sequence[int], density_mode: str,
) -> tuple:
    """在每个工作进程中独立准备同源问题及输入指纹, 不装配刚度."""
    bm.set_backend('numpy')
    shared_start = time.perf_counter()
    pde = make_mbb_problem(dim)
    domain_size = tuple(pde.domain[2*d+1] - pde.domain[2*d] for d in range(dim))
    reference = GlobalAssembler(domain_size, n_sub, n_fine, E_base=pde.E, nu=pde.nu)
    rho_global = make_cell_density(reference.total_fine, density_mode)
    conditions = make_fa_analyzer(reference.full_mesh, pde, reference.material)
    full_force = conditions.assemble_external_load()
    prescribed, fixed_mask = conditions.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc, threshold=cast(Any, pde.is_dirichlet_boundary()), method='interp',
    )
    fixed_global = bm.nonzero(fixed_mask)[0]
    if np.any(bm.to_numpy(prescribed)):
        raise ValueError("性能算例仅支持齐次 Dirichlet 约束.")
    # Q1 规则网格节点: 至少一个坐标落在子结构分割面上才属于保留接口.
    coords = np.asarray(bm.to_numpy(reference.full_mesh.entity('node')))
    scaled = coords / (np.asarray(domain_size) / np.asarray(n_sub))
    boundary_nodes = np.any(np.isclose(scaled, np.rint(scaled), rtol=0., atol=1e-10), axis=1)
    retained_mask = np.repeat(boundary_nodes, dim)
    force_np = np.asarray(bm.to_numpy(full_force)).reshape(-1)
    if not np.all(np.isfinite(force_np)) or not np.any(force_np):
        raise ValueError("载荷必须有限且非零.")
    if np.any(force_np[~retained_mask]) or not np.all(retained_mask[bm.to_numpy(fixed_global)]):
        raise ValueError("当前划分含内部载荷或内部支承, 不满足本示例的缩聚前提.")
    del conditions
    density_np = np.asarray(bm.to_numpy(rho_global), dtype='<f8', order='C')
    metadata = {
        'dimension': f'{dim}D', 'problem': type(pde).__name__, 'domain': list(pde.domain),
        'n_sub': list(n_sub), 'n_fine': list(n_fine), 'total_fine': list(reference.total_fine),
        'full_dofs': reference.total_full_dofs, 'density_mode': density_mode,
        'density_sha256': array_fingerprint(density_np),
        'load_sha256': array_fingerprint(force_np),
        'fixed_dofs_sha256': array_fingerprint(bm.to_numpy(fixed_global)),
        'nodes_sha256': array_fingerprint(coords),
        'displacement_shape': [reference.total_full_dofs], 'displacement_dtype': 'float64',
        'material': {'E': pde.E, 'nu': pde.nu, 'hypothesis': reference.material.hypothesis, 'penalty': 3.0},
        'load': {'P': pde.P, 'resultant': force_np.reshape(-1, dim).sum(axis=0).tolist()},
        'fixed_dofs': int(len(fixed_global)), 'degree': 1, 'backend': 'numpy', 'solver': 'scipy',
        'density_min': float(density_np.min()), 'density_max': float(density_np.max()),
        'density_mean': float(density_np.mean()),
    }
    preparation_seconds = time.perf_counter() - shared_start
    return pde, reference, rho_global, full_force, fixed_global, metadata, preparation_seconds


def performance_environment() -> Dict[str, Any]:
    """记录当前工作进程的依赖版本、线程环境及已加载的 BLAS 线程池."""
    environment = {
        'platform': platform.platform(), 'python': platform.python_version(),
        'numpy': np.__version__, 'processor': platform.processor(),
        'logical_cpu_count': os.cpu_count(),
        'thread_environment': {key: os.environ.get(key) for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
        )},
    }
    for package in ('scipy', 'fealpy', 'soptx'):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        environment['threadpools'] = None
    else:
        environment['threadpools'] = sorted(threadpool_info(), key=lambda item: item.get('filepath', ''))
    return environment


def print_performance_summary(result: Dict[str, Any]) -> None:
    """按正确性、耗时、内存、结论的顺序汇总任意两条或三条路径."""
    routes = tuple(result['routes'])
    samples = result['samples']
    checked = result['warmup_samples'] + samples
    rows = [('正确性', *routes)]
    rows.append((
        '柔顺度 (最后一次)',
        *(f"{samples[-1]['paths'][route]['compliance']:.8f}" for route in routes),
    ))
    for route in routes[1:]:
        errors = [row['relative_errors_to_fa'][route] for row in checked]
        rows.append((
            f'{route} 最大位移相对差 / FA',
            *(('--' if current != route else f"{max(item['displacement'] for item in errors):.2e}")
              for current in routes),
        ))
        rows.append((
            f'{route} 最大柔顺度相对差 / FA',
            *(('--' if current != route else f"{max(item['compliance'] for item in errors):.2e}")
              for current in routes),
        ))
    if 'linear_corner' in routes:
        maximum = max(
            row['paths']['linear_corner'][key]
            for row in checked for key in LINEAR_CORNER_CONSISTENCY_KEYS
        )
        rows.append((
            'linear_corner 最大一致性残差',
            *(('--' if route != 'linear_corner' else f'{maximum:.2e}') for route in routes),
        ))
    print()
    print_table(rows)
    validations = []
    if 'full_trace' in routes:
        validations.append(
            f"full_trace 对 FA 的位移与柔顺度相对差 <= "
            f"{result['relative_error_tolerance']:.1e}"
        )
    if 'linear_corner' in routes:
        validations.append(
            f"linear_corner 自身一致性残差 <= {result['consistency_tolerance']:.1e}"
        )
    print(f"验收：PASS, {'; '.join(validations)}.")

    stats = result['statistics_seconds']
    rows = [('耗时 (s, 正式测量中位数)', *routes)]
    rows.append((
        '问题准备',
        *(f"{result['statistics_preparation_seconds'][route]['median']:.4f}" for route in routes),
    ))
    for phase, label in (
        ('setup', '初始化'),
        ('assembly', '全局/局部刚度装配'),
        ('condensation', '子结构内部消元'),
        ('interface_assembly', '接口矩阵映射与装配'),
        ('conditions_solve', '边界条件处理与直接求解'),
        ('recovery', '完整位移恢复'),
        ('total', '总分析时间'),
    ):
        rows.append((
            label,
            *(('--' if phase not in stats[route]
               else f"{stats[route][phase]['median']:.4f}") for route in routes),
        ))
    rows.append((
        '准备+分析',
        *(f"{result['statistics_preparation_and_analysis_seconds'][route]['median']:.4f}"
          for route in routes),
    ))
    print()
    print_table(rows)
    print()
    print_table([
        ('内存与规模', *routes),
        ('峰值 RSS (MiB)', *(f"{result['statistics_peak_rss_bytes'][route]['median'] / 2**20:.2f}"
                              for route in routes)),
        ('独立求解自由度', *(str(samples[-1]['paths'][route]['free_dofs']) for route in routes)),
    ])

    conclusions = []
    for route in routes[1:]:
        comparison = result['comparisons_to_fa'][route]
        speedup = comparison['preparation_and_analysis_speedup_ratio_of_medians']
        saving = comparison['peak_memory_saving_fraction']
        if f'{speedup:.2f}' == '1.00':
            time_text = '耗时比约为 1.00'
        elif speedup > 1:
            time_text = f'加速 {speedup:.2f} 倍'
        else:
            time_text = f'耗时为 FA 的 {1 / speedup:.2f} 倍'
        if f'{abs(saving):.1%}' == '0.0%':
            memory_text = '峰值内存变化约为 0.0%'
        elif saving > 0:
            memory_text = f'峰值内存节省 {saving:.1%}'
        else:
            memory_text = f'峰值内存增加 {-saving:.1%}'
        accuracy = '与 FA 精度一致' if route == 'full_trace' else '自身一致性通过, 对 FA 为近似解'
        conclusions.append(f'{route}: {accuracy}, {time_text}, {memory_text}')
    print("口径：峰值 RSS 含依赖加载、准备及分析; 分项中位数之和不必等于总时间中位数.")
    if result['repeat'] == 1:
        print("提醒：本次仅正式测量 1 组, 不能形成稳定性能结论.")
    print("\n结论：" + "; ".join(conclusions) + ".")
    print("      时间比按准备+分析的中位数计算, 结论仅适用于本次配置.")


def _run_route_comparison(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    warmup: int = DEFAULT_WARMUP, repeat: int = DEFAULT_REPEAT,
    density_mode: str = DEFAULT_DENSITY,
    routes: Sequence[str] = ('fa', 'full_trace'),
    comparison_name: str = 'full_trace 与 FA 比较',
    schema_version: str = 'full-trace-performance-v2',
    output_tag: str = 'full-trace',
) -> Dict[str, Any]:
    """在独立进程中串行比较指定路径的时间、峰值 RSS 与正确性.

    Parameters
    ----------
    dim : int
        空间维数, 2 或 3.
    output_dir : str or None
        全部验收通过后的证据目录; None 时不写最终 JSON.
    n_sub, n_fine : sequence of int or None
        子结构划分与每块细单元数.
    warmup, repeat : int
        独立进程试运行对数与正式测量对数, 试运行不纳入统计.
        每个样本均为新进程, 不会继承前次 BLAS 初始化或矩阵缓存.
    density_mode : str
        cell 或 uniform, 两者均与子结构划分无关.

    Returns
    -------
    dict
        每个样本的环境、输入指纹、计时、峰值 RSS、精度及汇总.
    """
    from examples.substructure_elasticity._performance_process import peak_rss_bytes, run_worker

    n_sub, n_fine = resolve_grid(dim, n_sub, n_fine)
    routes = tuple(routes)
    supported_routes = {'fa', 'full_trace', 'linear_corner'}
    if (not routes or routes[0] != 'fa' or len(set(routes)) != len(routes)
            or not set(routes) <= supported_routes):
        raise ValueError(
            "性能比较 routes 必须以 fa 开头, 且由 fa/full_trace/linear_corner 唯一组成."
        )
    for name, value, lower in (('warmup', warmup, 0), ('repeat', repeat, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} 必须为 >= {lower} 的整数.")
    if density_mode not in ('cell', 'uniform'):
        raise ValueError("density 必须为 cell 或 uniform.")
    # 提前确认平台支持; 父进程自身的读数不纳入测量.
    peak_rss_bytes()
    pde = make_mbb_problem(dim)
    grid = tuple(sub * fine for sub, fine in zip(n_sub, n_fine))
    print(f"comparison   {comparison_name}")
    print(f"problem      {type(pde).__name__}, Q1, scipy, float64")
    print(f"mesh         {'x'.join(map(str, grid))}, {int(np.prod(grid))} 单元, "
          f"{dim * int(np.prod(np.asarray(grid) + 1))} 自由度")
    print(f"substructure n-sub={'x'.join(map(str, n_sub))} ({int(np.prod(n_sub))} 块), "
          f"n-fine={'x'.join(map(str, n_fine))}")
    print(f"routes       {' / '.join(routes)}")
    print(f"measurement  试运行 {warmup} 组, 正式测量 {repeat} 组, 独立进程, density={density_mode}")
    print(flush=True)

    samples, warmup_samples = [], []
    metadata, environment = None, None
    config = dict(dim=dim, n_sub=list(n_sub), n_fine=list(n_fine), density_mode=density_mode)
    for index in range(warmup + repeat):
        shift = index % len(routes)
        order = routes[shift:] + routes[:shift]
        label = f"试运行 {index+1}/{warmup}" if index < warmup else f"测量 {index-warmup+1}/{repeat}"
        print(f"{label}  ", end="", flush=True)
        records = {}
        with TemporaryDirectory(prefix="soptx-substructure-") as temporary:
            directory = Path(temporary)
            for route in order:
                records[route] = run_worker({**config, 'route': route}, directory / route)
                current = records[route]
                if metadata is None:
                    metadata, environment = current['problem_data'], current['environment']
                if current['problem_data'] != metadata:
                    raise AssertionError(f"{label} {route} 的网格、密度、载荷或约束元数据不一致.")
                if current['environment'] != environment:
                    raise AssertionError(f"{label} {route} 的依赖或线程环境不一致, 不能比较性能.")
                peak = current['memory_peak_rss_bytes']
                if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
                    raise AssertionError(f"{label} {route} 的峰值 RSS 非法.")
                seconds = current['preparation_seconds']
                if not np.isfinite(seconds) or seconds < 0:
                    raise AssertionError(f"{label} {route} 的问题准备时间非法.")
            # 全部工作进程均退出后才读位移, 避免父进程数组影响后续路径峰值.
            displacements = {
                route: np.load(directory / route / 'displacement.npy', allow_pickle=False)
                for route in routes
            }
            for route, displacement in displacements.items():
                if (list(displacement.shape) != metadata['displacement_shape']
                        or str(displacement.dtype) != metadata['displacement_dtype']
                        or not np.all(np.isfinite(displacement))):
                    raise AssertionError(f"{label} {route} 的位移形状、类型或有限性检查失败.")
            relative_errors = {}
            for route in routes[1:]:
                relative_errors[route] = {
                    'displacement': float(np.linalg.norm(
                        displacements[route] - displacements['fa']
                    )) / max(float(np.linalg.norm(displacements['fa'])), np.finfo(float).tiny),
                    'compliance': abs(
                        records[route]['compliance'] - records['fa']['compliance']
                    ) / abs(records['fa']['compliance']),
                }
            # 清除全部数组引用再启动下一对工作进程; 临时文件随上下文退出清理.
            del displacement, displacements
        for route, errors in relative_errors.items():
            if not all(np.isfinite(value) for value in errors.values()):
                raise AssertionError(f"{label} {route} 对 FA 的近似误差不是有限值.")
            if route == 'full_trace' and max(errors.values()) > RELATIVE_ERROR_TOLERANCE:
                raise AssertionError(
                    f"{label} full_trace 等价性不通过: "
                    f"位移={errors['displacement']:.4e}, 柔顺度={errors['compliance']:.4e}"
                )
            if route == 'linear_corner':
                validate_linear_corner_consistency(records[route])
        record = {
            'index': index+1, 'order': list(order), 'paths': records,
            'relative_errors_to_fa': relative_errors,
            'validation': 'PASS',
        }
        if 'full_trace' in relative_errors:
            record.update(
                displacement_relative_error=relative_errors['full_trace']['displacement'],
                compliance_relative_error=relative_errors['full_trace']['compliance'],
                speedup=records['fa']['seconds']['total'] / records['full_trace']['seconds']['total'],
            )
        (warmup_samples if index < warmup else samples).append(record)
        print("PASS", flush=True)

    stats = {
        route: {phase: timing_statistics([row['paths'][route]['seconds'][phase] for row in samples])
                for phase in samples[0]['paths'][route]['seconds']}
        for route in routes
    }
    memory_stats = {
        route: timing_statistics([row['paths'][route]['memory_peak_rss_bytes'] for row in samples])
        for route in routes
    }
    preparation_stats = {
        route: timing_statistics([row['paths'][route]['preparation_seconds'] for row in samples])
        for route in routes
    }
    combined_stats = {
        route: timing_statistics([
            row['paths'][route]['preparation_seconds'] + row['paths'][route]['seconds']['total']
            for row in samples
        ]) for route in routes
    }
    comparisons_to_fa = {
        route: {
            'analysis_speedup_ratio_of_medians': (
                stats['fa']['total']['median'] / stats[route]['total']['median']
            ),
            'preparation_and_analysis_speedup_ratio_of_medians': (
                combined_stats['fa']['median'] / combined_stats[route]['median']
            ),
            'peak_memory_ratio_of_medians': (
                memory_stats['fa']['median'] / memory_stats[route]['median']
            ),
            'peak_memory_saving_fraction': (
                1.0 - memory_stats[route]['median'] / memory_stats['fa']['median']
            ),
        }
        for route in routes[1:]
    }
    validation = {
        'input_and_environment_consistency': 'PASS',
        'timing_stability': 'reported_only',
        'memory_stability': 'reported_only',
    }
    if 'full_trace' in routes:
        validation['full_trace_equivalence'] = 'PASS'
    if 'linear_corner' in routes:
        validation.update(
            linear_corner_consistency='PASS',
            linear_corner_fa_error='reported_only',
        )
    result = {
        **metadata, 'schema_version': schema_version, 'routes': list(routes),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'environment': environment,
        'execution_mode': 'fresh_subprocess_per_sample_serial',
        'timing_scope': 'Fresh route setup, assembly, condensation, constraints/solve and recovery; per-worker problem preparation reported separately. Interpreter startup/imports, diagnostics and transfer excluded.',
        'memory_scope': 'Linux VmHWM peak RSS from process exec through preparation and full analysis, sampled before diagnostics and serialization. Includes imports and native allocations; excludes parent and peer processes.',
        'memory_method': '/proc/self/status:VmHWM', 'memory_unit': 'bytes',
        'warmup_scope': 'Discarded independent-process trial pairs; no BLAS/cache warmup carries into later processes.',
        'warmup': warmup, 'repeat': repeat, 'warmup_samples': warmup_samples, 'samples': samples,
        'statistics_seconds': stats,
        'statistics_preparation_seconds': preparation_stats,
        'statistics_preparation_and_analysis_seconds': combined_stats,
        'statistics_peak_rss_bytes': memory_stats,
        'comparisons_to_fa': comparisons_to_fa,
        'relative_error_tolerance': RELATIVE_ERROR_TOLERANCE,
        'consistency_tolerance': CONSISTENCY_TOLERANCE,
        'validation': validation,
    }
    # 保留完整接口旧结果的常用顶层字段, 便于既有分析脚本读取.
    if 'full_trace' in routes:
        full = comparisons_to_fa['full_trace']
        result.update(
            speedup_ratio_of_medians=full['analysis_speedup_ratio_of_medians'],
            preparation_and_analysis_speedup_ratio_of_medians=(
                full['preparation_and_analysis_speedup_ratio_of_medians']
            ),
            peak_memory_ratio_of_medians=full['peak_memory_ratio_of_medians'],
            peak_memory_saving_fraction=full['peak_memory_saving_fraction'],
            paired_speedup=timing_statistics([row['speedup'] for row in samples]),
        )
    print_performance_summary(result)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stem = result_filename(f'{dim}D', n_sub, n_fine).removesuffix('.json')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = output / f'{stem}_{output_tag}_{density_mode}_{stamp}.json'
        # 独占创建, 不覆盖已有研究证据.
        with target.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        default_output = Path(__file__).resolve().parent / 'outputs'
        display_path = Path('outputs') / target.name if output.resolve() == default_output else target.resolve()
        print(f"详细结果：{display_path}")
    return result


def run_full_trace_comparison(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    warmup: int = DEFAULT_WARMUP, repeat: int = DEFAULT_REPEAT,
    density_mode: str = DEFAULT_DENSITY,
) -> Dict[str, Any]:
    """独立进程重复比较 FA 与 full_trace 的精度、时间和峰值内存."""
    return _run_route_comparison(
        dim, output_dir, n_sub=n_sub, n_fine=n_fine,
        warmup=warmup, repeat=repeat, density_mode=density_mode,
        routes=('fa', 'full_trace'), comparison_name='full_trace 与 FA 比较',
        schema_version='full-trace-performance-v2', output_tag='full-trace',
    )


def run_linear_corner_comparison(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    warmup: int = DEFAULT_WARMUP, repeat: int = DEFAULT_REPEAT,
    density_mode: str = DEFAULT_DENSITY,
) -> Dict[str, Any]:
    """独立进程比较 linear_corner 与 FA, 仅报告对 FA 的近似误差."""
    return _run_route_comparison(
        dim, output_dir, n_sub=n_sub, n_fine=n_fine,
        warmup=warmup, repeat=repeat, density_mode=density_mode,
        routes=('fa', 'linear_corner'), comparison_name='linear_corner 与 FA 比较',
        schema_version='linear-corner-performance-v1', output_tag='linear-corner',
    )


def run_three_path_comparison(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    warmup: int = DEFAULT_WARMUP, repeat: int = DEFAULT_REPEAT,
    density_mode: str = DEFAULT_DENSITY,
) -> Dict[str, Any]:
    """在同一配置下独立进程比较 FA、full_trace 与 linear_corner."""
    return _run_route_comparison(
        dim, output_dir, n_sub=n_sub, n_fine=n_fine,
        warmup=warmup, repeat=repeat, density_mode=density_mode,
        routes=('fa', 'full_trace', 'linear_corner'),
        comparison_name='FA / full_trace / linear_corner 三路径比较',
        schema_version='substructure-three-path-performance-v1',
        output_tag='three-paths',
    )


def run_linear_corner_consistency(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    density_mode: str = DEFAULT_DENSITY,
) -> Dict[str, Any]:
    """单独验收 linear_corner 的投影、平衡、约束、恢复和能量一致性."""
    n_sub, n_fine = resolve_grid(dim, n_sub, n_fine)
    pde, reference, density, force, fixed, metadata, preparation_seconds = (
        prepare_performance_problem(dim, n_sub, n_fine, density_mode)
    )
    _, record = timed_full_analysis(
        'linear_corner', pde, reference, density, force, fixed,
    )
    validate_linear_corner_consistency(record)
    result = {
        **metadata,
        'schema_version': 'linear-corner-consistency-v1',
        'route': 'linear_corner',
        'preparation_seconds': preparation_seconds,
        'analysis': record,
        'consistency_tolerance': CONSISTENCY_TOLERANCE,
        'validation': {'linear_corner_consistency': 'PASS'},
    }
    grid = 'x'.join(map(str, metadata['total_fine']))
    print("verification  linear_corner 自身一致性")
    print(f"problem       {metadata['problem']}, Q1, scipy, float64")
    print(f"mesh          {grid}, {int(np.prod(metadata['total_fine']))} 单元, "
          f"{metadata['full_dofs']} 自由度")
    print(f"substructure  n-sub={'x'.join(map(str, n_sub))} ({int(np.prod(n_sub))} 块), "
          f"n-fine={'x'.join(map(str, n_fine))}")
    print()
    print_table([
        ('一致性指标', 'linear_corner'),
        ('Galerkin 矩阵相对误差', f"{record['galerkin_relative_error']:.2e}"),
        ('系统平衡相对残差', f"{record['equilibrium_relative_residual']:.2e}"),
        ('线性约束相对残差', f"{record['constraint_relative_residual']:.2e}"),
        ('支承恢复相对误差', f"{record['recovered_support_relative_error']:.2e}"),
        ('内部平衡相对残差', f"{record['internal_relative_residual']:.2e}"),
        ('能量相对误差', f"{record['energy_relative_error']:.2e}"),
        ('载荷虚功相对误差', f"{record['load_work_relative_error']:.2e}"),
    ])
    print(f"\n结论：PASS, 所有一致性指标 <= {CONSISTENCY_TOLERANCE:.1e}; "
          "本验证不以相对 FA 的近似误差作为门禁.")
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        sub = 'x'.join(map(str, n_sub))
        fine = 'x'.join(map(str, n_fine))
        target = output / f'linear_corner_consistency_{dim}d_sub-{sub}_fine-{fine}.json'
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        default_output = Path(__file__).resolve().parent / 'outputs'
        display_path = Path('outputs') / target.name if output.resolve() == default_output else target.resolve()
        print(f"详细结果：{display_path}")
    return result
