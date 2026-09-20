"""验证角点线性迹投影及降阶系统的一致性.

在同一网格、材料、密度、载荷和支承下验证角点线性迹 (linear_corner) 降阶模型
在 Galerkin 平衡、线性约束、支承恢复、内部平衡及能量虚功守恒上的内部代数一致性.
本脚本不以相对 FA 全结构解的近似误差作为门禁, 专门验收降阶代数系统的自洽性.

使用方法:
    # 2D MBB 梁角点一致性验证 (默认 6x2 子结构, 每块 5x5 细网格)
    python examples/substructure_elasticity/verify_linear_corner_consistency.py --problem HalfMBBBeamRight2d

    # 3D MBB 梁角点一致性验证 (默认 6x2x2 子结构, 每块 4x4x4 细网格)
    python examples/substructure_elasticity/verify_linear_corner_consistency.py --problem FullMBBBeam3d
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

import numpy as np
from scipy.sparse.linalg import norm as sparse_norm

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
    build_substructures as _build_substructures,
    solve_constrained_system,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

DEFAULT_DENSITY = "cell"
CONSISTENCY_TOLERANCE = 1.0e-9
PROBLEMS = {"HalfMBBBeamRight2d": 2, "FullMBBBeam3d": 3}
LINEAR_CORNER_CONSISTENCY_KEYS = (
    "equilibrium_relative_residual",
    "constraint_relative_residual",
    "recovered_support_relative_error",
    "internal_relative_residual",
    "energy_relative_error",
    "load_work_relative_error",
    "galerkin_relative_error",
)


def resolve_grid(
    dim: int,
    n_sub: Sequence[int] | None,
    n_fine: Sequence[int] | None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """补齐并校验子结构划分, 在创建网格之前拒绝非法规模."""
    if dim not in (2, 3):
        raise ValueError("dim 必须为 2 或 3.")
    default_sub, default_fine = (
        ((6, 2), (5, 5)) if dim == 2 else ((6, 2, 2), (4, 4, 4))
    )
    sub = tuple(default_sub if n_sub is None else n_sub)
    fine = tuple(default_fine if n_fine is None else n_fine)
    for name, values in (("--n-sub", sub), ("--n-fine", fine)):
        if len(values) != dim:
            raise ValueError(f"{name} 在 dim={dim} 时必须提供 {dim} 个整数.")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in values):
            raise ValueError(f"{name} 的各项必须为正整数.")
    return sub, fine


def display_width(text: str) -> int:
    """计算字符串的终端显示宽度, 东亚全角字符按两列计."""
    return sum(2 if unicodedata.east_asian_width(char) in ("F", "W") else 1 for char in text)


def print_table(rows: Sequence[Sequence[str]]) -> None:
    """按终端显示宽度输出紧凑表格."""
    widths = [max(display_width(row[col]) for row in rows) for col in range(len(rows[0]))]
    for row in rows:
        print("  ".join(
            value + " " * (widths[col] - display_width(value))
            for col, value in enumerate(row)
        ).rstrip())


def array_fingerprint(array: Any) -> str:
    """计算连续数组的数据指纹, 避免额外创建同样大小的 bytes 副本."""
    data = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh], List[Tuple[int, ...]]]:
    """复用核心子结构构造, 并保留既有实验导入入口."""
    return _build_substructures(assembler)


def build_corner_projection(
    assembler: GlobalAssembler,
    sub_meshes: List[Any],
    interface_dofs: Any,
    trace_basis: Any,
) -> Any:
    """复用核心角点投影, 并保留既有实验导入入口."""
    interface_view = SimpleNamespace(global_dofs=interface_dofs)
    return assembler.build_linear_corner_projection(
        sub_meshes, interface_view, trace_basis
    )


def solve_corner_system(
    system: Any, force: Any, constraints: Any
) -> Tuple[Any, int, float, float, str]:
    """复用核心一般线性约束求解, 并保留既有实验导入入口."""
    result = solve_constrained_system(system, force, constraints)
    return (
        result.displacement,
        result.constraint_rank,
        result.equilibrium_relative_residual,
        result.constraint_relative_residual,
        result.mode,
    )


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
    """独立完成角点路径的装配、精确缩聚、条件投影、求解与全场恢复."""
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
        "stiffness_dtype": str(bm.to_numpy(local_stiffness).dtype),
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


def validate_linear_corner_consistency(record: Dict[str, Any]) -> None:
    """验收角点迹离散自身的一致性, 不把相对 FA 的近似误差作为门禁."""
    for key in LINEAR_CORNER_CONSISTENCY_KEYS:
        value = record[key]
        if not np.isfinite(value) or value > CONSISTENCY_TOLERANCE:
            raise AssertionError(
                f"linear_corner {key}={value:.4e} 超过一致性阈值 "
                f"{CONSISTENCY_TOLERANCE:.1e}."
            )


def make_mbb_problem(dim: int) -> Any:
    """构造 MBB 梁问题."""
    if dim == 2:
        return HalfMBBBeamRight2d(
            domain=(0.0, 60.0, 0.0, 20.0), P=-1.0, E=1.0, nu=0.3,
        )
    if dim == 3:
        return FullMBBBeam3d(
            domain=(0.0, 6.0, 0.0, 1.0, 0.0, 1.0), P=-1.0, E=1.0, nu=0.3,
        )
    raise ValueError("dim 必须为 2 或 3.")


def make_fa_analyzer(mesh: Any, pde: Any, material: Any, solve_method: str = "scipy") -> LagrangeFEMAnalyzer:
    """构造一阶 SIMP 全装配分析器, 不复用数值刚度或分解."""
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=pde, material=material, space_degree=1,
        assembly_method="standard", operator_level="fa", solve_method=solve_method,
        topopt_algorithm="density_based",
        interpolation_scheme=MaterialInterpolationScheme(
            density_location="element", interpolation_method="simp",
            options={"penalty_factor": 3.0, "stress_penalty_factor": 1.0},
        ),
        enable_logging=False,
    )


def make_cell_density(shape: Sequence[int], mode: str) -> Any:
    """按细单元中心采样密度, 同一细网格上的结果与子结构划分无关."""
    if mode == "uniform":
        return bm.full(tuple(shape), 0.7, dtype=bm.float64)
    if mode != "cell":
        raise ValueError("density 必须为 cell 或 uniform.")
    coords = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in shape], indexing="ij")
    modulation = np.sin(np.pi * coords[0]) * np.cos(np.pi * coords[1])
    if len(shape) == 3:
        modulation *= np.sin(np.pi * coords[2])
    return bm.asarray(0.7 + 0.3 * modulation, dtype=bm.float64)


def prepare_consistency_problem(
    dim: int, n_sub: Sequence[int], n_fine: Sequence[int], density_mode: str,
    solve_method: str = "scipy",
) -> tuple:
    """准备同源问题及输入指纹, 不装配刚度."""
    bm.set_backend("numpy")
    shared_start = time.perf_counter()
    pde = make_mbb_problem(dim)
    domain_size = tuple(pde.domain[2*d+1] - pde.domain[2*d] for d in range(dim))
    reference = GlobalAssembler(domain_size, n_sub, n_fine, E_base=pde.E, nu=pde.nu)
    rho_global = make_cell_density(reference.total_fine, density_mode)
    conditions = make_fa_analyzer(reference.full_mesh, pde, reference.material, solve_method=solve_method)
    full_force = conditions.assemble_external_load()
    prescribed, fixed_mask = conditions.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc, threshold=cast(Any, pde.is_dirichlet_boundary()), method="interp",
    )
    fixed_global = bm.nonzero(fixed_mask)[0]
    if np.any(bm.to_numpy(prescribed)):
        raise ValueError("性能算例仅支持齐次 Dirichlet 约束.")
    # Q1 规则网格节点: 至少一个坐标落在子结构分割面上才属于保留接口.
    coords = np.asarray(bm.to_numpy(reference.full_mesh.entity("node")))
    scaled = coords / (np.asarray(domain_size) / np.asarray(n_sub))
    boundary_nodes = np.any(np.isclose(scaled, np.rint(scaled), rtol=0., atol=1e-10), axis=1)
    retained_mask = np.repeat(boundary_nodes, dim)
    force_np = np.asarray(bm.to_numpy(full_force)).reshape(-1)
    if not np.all(np.isfinite(force_np)) or not np.any(force_np):
        raise ValueError("载荷必须有限且非零.")
    if np.any(force_np[~retained_mask]) or not np.all(retained_mask[bm.to_numpy(fixed_global)]):
        raise ValueError("当前划分含内部载荷或内部支承, 不满足本示例的缩聚前提.")
    del conditions
    density_np = np.asarray(bm.to_numpy(rho_global), dtype="<f8", order="C")
    metadata = {
        "dimension": f"{dim}D", "problem": type(pde).__name__, "domain": list(pde.domain),
        "n_sub": list(n_sub), "n_fine": list(n_fine), "total_fine": list(reference.total_fine),
        "full_dofs": reference.total_full_dofs, "density_mode": density_mode,
        "density_sha256": array_fingerprint(density_np),
        "load_sha256": array_fingerprint(force_np),
        "fixed_dofs_sha256": array_fingerprint(bm.to_numpy(fixed_global)),
        "nodes_sha256": array_fingerprint(coords),
        "displacement_shape": [reference.total_full_dofs], "displacement_dtype": "float64",
        "material": {"E": pde.E, "nu": pde.nu, "hypothesis": reference.material.hypothesis, "penalty": 3.0},
        "load": {"P": pde.P, "resultant": force_np.reshape(-1, dim).sum(axis=0).tolist()},
        "fixed_dofs": int(len(fixed_global)), "degree": 1, "backend": "numpy", "solver": solve_method,
        "density_min": float(density_np.min()), "density_max": float(density_np.max()),
        "density_mean": float(density_np.mean()),
    }
    preparation_seconds = time.perf_counter() - shared_start
    return pde, reference, rho_global, full_force, fixed_global, metadata, preparation_seconds


def run_linear_corner_consistency(
    dim: int, output_dir: str | None = None, *,
    n_sub: Sequence[int] | None = None, n_fine: Sequence[int] | None = None,
    density_mode: str = DEFAULT_DENSITY,
) -> Dict[str, Any]:
    """单独验收 linear_corner 的投影、平衡、约束、恢复和能量一致性."""
    n_sub, n_fine = resolve_grid(dim, n_sub, n_fine)
    pde, reference, density, force, fixed, metadata, preparation_seconds = (
        prepare_consistency_problem(dim, n_sub, n_fine, density_mode)
    )
    analysis_start = time.perf_counter()
    assembler = GlobalAssembler(
        reference.domain_size, reference.n_sub, reference.n_fine,
        E_base=pde.E, nu=pde.nu,
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    local_density = assembler.split_global_cell_field(density)
    outer_setup = time.perf_counter() - analysis_start
    displacement, corner, _, _ = analyze_linear_corner(
        assembler, prototype, sub_meshes, local_density, force, fixed,
    )
    phases = dict(corner["phase_seconds"])
    phases["setup"] += outer_setup
    phases["total"] += outer_setup
    record = {
        "seconds": phases,
        "compliance": corner["compliance"],
        "system_dofs": corner["trace_dofs"],
        "free_dofs": corner["free_dofs"],
        "equilibrium_relative_residual": corner["equilibrium_relative_residual"],
        "support_relative_error": corner["recovered_support_relative_error"],
        "memory_peak_rss_bytes": corner["memory_peak_rss_bytes"],
        "constraint_rank": corner["constraint_rank"],
        "local_trace_dofs": corner["local_trace_dofs"],
        "constraint_solver": corner["constraint_solver"],
        **{key: corner[key] for key in LINEAR_CORNER_CONSISTENCY_KEYS},
    }
    validate_linear_corner_consistency(record)
    result = {
        **metadata,
        "schema_version": "linear-corner-consistency-v1",
        "route": "linear_corner",
        "preparation_seconds": preparation_seconds,
        "analysis": record,
        "consistency_tolerance": CONSISTENCY_TOLERANCE,
        "validation": {"linear_corner_consistency": "PASS"},
    }
    grid = "x".join(map(str, metadata["total_fine"]))
    print("verification  linear_corner 自身一致性")
    dtype_values = {
        "coordinates": str(bm.to_numpy(assembler.full_mesh.entity("node")).dtype),
        "stiffness": corner["stiffness_dtype"],
        "displacement": str(bm.to_numpy(displacement).dtype),
    }
    dtype_text = (
        next(iter(dtype_values.values())) if len(set(dtype_values.values())) == 1
        else ", ".join(f"{name}={value}" for name, value in dtype_values.items())
    )
    print(f"problem       {metadata['problem']}")
    print(f"mesh          {type(assembler.full_mesh).__name__}, {grid}, "
          f"{int(np.prod(metadata['total_fine']))} 单元")
    print(f"space         Q{assembler.degree}, {metadata['full_dofs']} 自由度")
    print(f"dtype         {dtype_text}")
    print(f"solver        {metadata['solver']}")
    print(f"substructure  n-sub={'x'.join(map(str, n_sub))} ({int(np.prod(n_sub))} 块), "
          f"n-fine={'x'.join(map(str, n_fine))}")
    print()
    print_table([
        ("一致性指标", "linear_corner"),
        ("Galerkin 矩阵相对误差", f"{record['galerkin_relative_error']:.2e}"),
        ("系统平衡相对残差", f"{record['equilibrium_relative_residual']:.2e}"),
        ("线性约束相对残差", f"{record['constraint_relative_residual']:.2e}"),
        ("支承恢复相对误差", f"{record['recovered_support_relative_error']:.2e}"),
        ("内部平衡相对残差", f"{record['internal_relative_residual']:.2e}"),
        ("能量相对误差", f"{record['energy_relative_error']:.2e}"),
        ("载荷虚功相对误差", f"{record['load_work_relative_error']:.2e}"),
    ])
    print(f"\n结论：PASS, 所有一致性指标 <= {CONSISTENCY_TOLERANCE:.1e}; "
          "本验证不以相对 FA 的近似误差作为门禁.")
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        sub = "x".join(map(str, n_sub))
        fine = "x".join(map(str, n_fine))
        target = output / f"linear_corner_consistency_{dim}d_sub-{sub}_fine-{fine}.json"
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        default_output = Path(__file__).resolve().parent / "outputs"
        display_path = Path("outputs") / target.name if output.resolve() == default_output else target.resolve()
        print(f"详细结果：{display_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证角点线性迹投影及降阶系统的内部一致性.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--problem", choices=tuple(PROBLEMS), default="HalfMBBBeamRight2d",
        help="MBB 梁 Problem; 同时确定计算维度. 默认 HalfMBBBeamRight2d.",
    )
    parser.add_argument(
        "--n-sub", type=int, nargs="+", metavar="N",
        help="各方向子结构数; 省略时 2D 为 6 2, 3D 为 6 2 2.",
    )
    parser.add_argument(
        "--n-fine", type=int, nargs="+", metavar="N",
        help="每个子结构各方向的有限元单元数; 省略时 2D 为 5 5, 3D 为 4 4 4.",
    )
    parser.add_argument(
        "--density", choices=("cell", "uniform"), default=DEFAULT_DENSITY,
        help="单元密度场类型. 默认 cell.",
    )
    parser.add_argument(
        "--output-dir", default=str(_SCRIPT_DIR / "outputs"),
        help="JSON 结果目录. 默认脚本同级 outputs/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_linear_corner_consistency(
        PROBLEMS[args.problem], args.output_dir,
        n_sub=args.n_sub, n_fine=args.n_fine, density_mode=args.density,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
