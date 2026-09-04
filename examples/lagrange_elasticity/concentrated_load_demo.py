"""集中力 (点载荷) 工程基准算例.

与 ``manufactured_convergence_demo.py`` 的分工:

* ``manufactured_convergence_demo.py`` 走制造解: 有精确解, 判据是 L2 观测收敛阶
  + 真相对残差, 回答"离散是否正确";
* 本文件走工程基准: 没有解析解, 判据是真相对残差 + 载荷等效性, 回答
  "集中力载荷路径是否被正确装配"。

为什么没有 L2 收敛阶判据: 二维点载荷在作用点处应力奇异, 位移解光滑性低于
制造解, 理论收敛阶也随之降低, 沿用制造解的 1.5 门槛会误报。集中力路径的
正确性由两条无歧义判据守住:

* 真相对残差 ``||K u - F|| / ||F||`` —— 线性系统确实解开了;
* 载荷等效性 ``|sum(F_sigmah) - P|`` —— 等效节点力装配没有丢力/多分力。
  这一步必须测量 ``apply_bc`` 覆盖 Dirichlet 自由度之前的 traction 向量:
  若载荷落在被强加的自由度上, 残差依然为 0, 而载荷会被静默吞掉, 只有
  载荷和校验能抓到这种失效。

全部门禁在 ``run_concentrated_load_benchmark`` 内以运行时断言实现: 任一项不达标
即抛 ``AssertionError`` 且不写任何文件, 全部通过才落盘 JSON 证据。证据缺省写入
本文件同目录的 ``outputs/``, 由 ``--output-dir`` 改写。契约与实测证据见同目录
``results_analysis.md``。

运行::

    # 2D: 对称半域 MBB 梁 (默认)
    python examples/lagrange_elasticity/concentrated_load_demo.py
    python examples/lagrange_elasticity/concentrated_load_demo.py --levels 4
    python examples/lagrange_elasticity/concentrated_load_demo.py --mesh-type tri
    python examples/lagrange_elasticity/concentrated_load_demo.py --solver cg --rtol 1e-12
    python examples/lagrange_elasticity/concentrated_load_demo.py --save-vtu

    # 3D: 完整全域 MBB 梁 (FullMBBBeam3d, 默认 120x20x20)
    python examples/lagrange_elasticity/concentrated_load_demo.py --dim 3 --levels 1
    python examples/lagrange_elasticity/concentrated_load_demo.py --dim 3 --nx 30 --ny 10 --nz 10
    # 3D: 对称半域 MBB 梁 (HalfMBBBeamRight3d)
    python examples/lagrange_elasticity/concentrated_load_demo.py --dim 3 --problem mbb-half-3d --levels 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Literal

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TriangleMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeam3d, HalfMBBBeamRight2d, HalfMBBBeamRight3d
from soptx.postprocess.vtk_export import export_vtu


# ---------------------------------------------------------------------------
# 集中力算例注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcentratedLoadProblemEntry:
    """集中力工程基准算例的元数据。"""

    name: str
    label: str
    factory: Callable[..., Any]
    default_domain: tuple[float, ...]
    default_nx: int
    default_ny: int
    material_extra: dict[str, Any] | None = None
    load_attr: str = "P"
    default_nz: int | None = None


PROBLEM_REGISTRY: dict[str, ConcentratedLoadProblemEntry] = {
    "mbb-half": ConcentratedLoadProblemEntry(
        name="mbb-half",
        label="MBB 梁对称右半域 (2D)",
        factory=lambda **kw: HalfMBBBeamRight2d(
            domain=kw.pop("domain", (0.0, 60.0, 0.0, 20.0)),
            P=kw.pop("P", -1.0),
            E=kw.pop("E", 1.0),
            nu=kw.pop("nu", 0.3),
            plane_type=kw.pop("plane_type", "plane_stress"),
            **kw,
        ),
        default_domain=(0.0, 60.0, 0.0, 20.0),
        default_nx=60,
        default_ny=20,
    ),
    "mbb-half-3d": ConcentratedLoadProblemEntry(
        name="mbb-half-3d",
        label="MBB 梁对称右半域 (3D)",
        factory=lambda **kw: HalfMBBBeamRight3d(
            domain=kw.pop("domain", (0.0, 60.0, 0.0, 20.0, 0.0, 20.0)),
            P=kw.pop("P", -1.0),
            E=kw.pop("E", 1.0),
            nu=kw.pop("nu", 0.3),
            **kw,
        ),
        default_domain=(0.0, 60.0, 0.0, 20.0, 0.0, 20.0),
        default_nx=60,
        default_ny=20,
        default_nz=20,
    ),
    "mbb-full-3d": ConcentratedLoadProblemEntry(
        name="mbb-full-3d",
        label="MBB 梁完整全域 (3D)",
        factory=lambda **kw: FullMBBBeam3d(
            domain=kw.pop("domain", (0.0, 120.0, 0.0, 20.0, 0.0, 20.0)),
            P=kw.pop("P", -1.0),
            E=kw.pop("E", 1.0),
            nu=kw.pop("nu", 0.3),
            **kw,
        ),
        default_domain=(0.0, 120.0, 0.0, 20.0, 0.0, 20.0),
        default_nx=120,
        default_ny=20,
        default_nz=20,
    ),
}


# 与 manufactured_convergence_demo.py / tools/matrix_free_evidence/contract.py 的残差门禁保持一致
RESIDUAL_TOLERANCE = 1.0e-10
# 施加节点力总和与 P 的偏差门禁
LOAD_TOLERANCE = 1.0e-10

# 分母里出现范数时的下限
NORM_FLOOR = 1.0e-30

# 与 LagrangeFEMAnalyzer 的 solve_method 形参取值域保持一致, 避免传入未支持的求解器名
SolverName = Literal["scipy", "mumps", "cg"]

DIRECT_SOLVERS: tuple[SolverName, ...] = ("scipy", "mumps")
ITERATIVE_SOLVERS: tuple[SolverName, ...] = ("cg",)


def create_problem_and_material(
    entry: ConcentratedLoadProblemEntry,
    domain_override: tuple[float, ...] | None = None,
) -> tuple[Any, IsotropicLinearElasticMaterial]:
    """根据注册表条目创建问题实例和对应的材料对象。"""

    domain = domain_override if domain_override is not None else entry.default_domain
    problem = entry.factory(domain=domain)

    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        enable_logging=False,
    )
    return problem, material


def create_mesh(problem, mesh_type: str, nx: int, ny: int, nz: int | None = None):
    """显式创建问题离散网格，保持 Problem 与 Mesh 分离。"""

    if problem.dimension == 2:
        constructor = {"quad": QuadrangleMesh, "tri": TriangleMesh}[mesh_type]
        return constructor.from_box(box=list(problem.domain), nx=nx, ny=ny)
    else:  # 3D
        if mesh_type != "hex":
            raise ValueError(f"3D 仅支持 hex 网格，不支持 {mesh_type}")
        if nz is None:
            raise ValueError("3D 问题必须提供 nz 参数")
        return HexahedronMesh.from_box(box=list(problem.domain), nx=nx, ny=ny, nz=nz)


def solve_one_level(
    problem,
    material,
    entry: ConcentratedLoadProblemEntry,
    mesh_type: str,
    nx: int,
    ny: int,
    degree: int,
    solver: SolverName,
    solver_options: dict[str, Any],
    nz: int | None = None,
) -> dict:
    """在一层网格上求解，返回残差、载荷和与柔顺度等诊断量。"""

    mesh = create_mesh(problem, mesh_type, nx, ny, nz)
    integration_order = degree + 3

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=degree,
        integration_order=integration_order,
        operator_level="fa",
        solve_method=solver,
        topopt_algorithm=None,
        enable_logging=False,
    )

    # 计时覆盖装配到求解的完整链路, 不含网格生成与载荷和校验
    started = time.perf_counter()

    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()
    K, F = analyzer.apply_bc(K0, F0)

    uh = analyzer.tensor_space.function()
    _, solver_info = analyzer.solve_system(K, F, uh, **solver_options)

    elapsed = time.perf_counter() - started

    displacement = bm.asarray(uh)
    residual_norm = float(np.linalg.norm(np.asarray(K @ displacement - F)))
    load_norm = float(np.linalg.norm(np.asarray(F)))
    traction = analyzer._assemble_traction_load(adjoint=False)
    load_sum = float(bm.sum(bm.asarray(traction)))
    compliance = float(np.asarray(displacement) @ np.asarray(F))

    applied_load = getattr(problem, entry.load_attr)

    result = {
        "nx": nx,
        "ny": ny,
        "mesh_size": float(problem.domain[1] / nx),
        "cells": int(mesh.number_of_cells()),
        "dofs": int(analyzer.tensor_space.number_of_global_dofs()),
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        "load_sum": load_sum,
        "load_error": abs(load_sum - applied_load),
        "compliance": compliance,
        "seconds": float(elapsed),
        # 直接解法不返回迭代信息, 保留 None 而不是补 0/True 这类看似合规的占位值
        "niter": None if solver_info.get("niter") is None else int(solver_info["niter"]),
        "converged": (
            None if solver_info.get("converged") is None
            else bool(solver_info["converged"])
        ),
    }
    if problem.dimension == 3:
        result["nz"] = nz
    return result


def report(rows: list[dict], solver: str) -> None:
    """打印结果表."""

    iterative = solver in ITERATIVE_SOLVERS
    is_3d = "nz" in rows[0] if rows else False

    if is_3d:
        header = (
            f"{'nx':>5} {'ny':>5} {'nz':>5} {'cells':>9} {'gdof':>9} {'h':>9} "
            f"{'residual':>11} {'load_sum':>10} {'load_err':>10} "
            f"{'compliance':>12} {'sec':>8}"
        )
    else:
        header = (
            f"{'nx':>5} {'cells':>9} {'gdof':>9} {'h':>9} "
            f"{'residual':>11} {'load_sum':>10} {'load_err':>10} "
            f"{'compliance':>12} {'sec':>8}"
        )
    if iterative:
        header += f" {'niter':>7} {'conv':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if is_3d:
            line = (
                f"{row['nx']:>5} {row['ny']:>5} {row['nz']:>5} {row['cells']:>9} {row['dofs']:>9} "
                f"{row['mesh_size']:>9.4f} {row['residual']:>11.2e} "
                f"{row['load_sum']:>10.6f} {row['load_error']:>10.2e} "
                f"{row['compliance']:>12.6e} {row['seconds']:>8.4f}"
            )
        else:
            line = (
                f"{row['nx']:>5} {row['cells']:>9} {row['dofs']:>9} "
                f"{row['mesh_size']:>9.4f} {row['residual']:>11.2e} "
                f"{row['load_sum']:>10.6f} {row['load_error']:>10.2e} "
                f"{row['compliance']:>12.6e} {row['seconds']:>8.4f}"
            )
        if iterative:
            line += f" {row['niter']:>7} {str(row['converged']):>6}"
        print(line)


def solver_unavailable_reason(solver: str) -> str | None:
    """求解器后端不可用时返回原因, 可用则返回 None."""

    if solver != "mumps":
        return None

    try:
        import_module("mumps")
    except Exception as exc:
        return (
            f"求解器 'mumps' 不可用 ({type(exc).__name__}: {exc}); "
            "该后端需要 PyMUMPS 包 (pip install pymumps) 及系统 MUMPS 库。"
            "请改用 --solver scipy 或 --solver cg."
        )
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="集中力载荷路径的工程基准算例",
    )
    parser.add_argument(
        "--dim", type=int, choices=[2, 3], default=2,
        help="问题维度: 2D 或 3D (默认 2)",
    )
    parser.add_argument(
        "--problem", default=None,
        help="集中力工程基准算例; 若不指定则自动选择 (2D/3D 默认)",
    )
    parser.add_argument(
        "--nx", type=int, default=None,
        help="最粗档的 x 方向单元数 (默认由算例决定)",
    )
    parser.add_argument(
        "--ny", type=int, default=None,
        help="最粗档的 y 方向单元数 (默认由算例决定)",
    )
    parser.add_argument(
        "--nz", type=int, default=None,
        help="最粗档的 z 方向单元数 (3D 问题专用, 默认由算例决定)",
    )
    parser.add_argument(
        "--levels", type=int, default=3,
        help="加密层数, 每层 nx/ny/nz 加倍 (默认 3)",
    )
    parser.add_argument(
        "--mesh-type", choices=("quad", "tri"), default="quad",
        help="网格类型 (默认 quad)",
    )
    parser.add_argument(
        "--degree", type=int, default=1,
        help="位移空间次数 (默认 1)",
    )
    parser.add_argument(
        "--solver", choices=DIRECT_SOLVERS + ITERATIVE_SOLVERS,
        default="scipy",
        help="求解器 (默认 scipy); mumps 需要 PyMUMPS 包",
    )
    parser.add_argument(
        "--rtol", type=float, default=1.0e-12,
        help="cg 相对收敛容差 (默认 1e-12)",
    )
    parser.add_argument(
        "--atol", type=float, default=1.0e-12,
        help="cg 绝对收敛容差 (默认 1e-12)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=5000,
        help="cg 最大迭代步数 (默认 5000)",
    )
    parser.add_argument(
        "--save-vtu", action="store_true",
        help="导出最密层网格的位移场为 VTU 文件 (ParaView 可视化)",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).with_name("outputs")),
        help="验收通过后写入 JSON 证据的目录 (默认为本文件同级 outputs/)",
    )
    return parser.parse_args()


def run_concentrated_load_benchmark(
    dim: int = 2,
    problem_name: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    mesh_type: str = "quad",
    levels: int = 3,
    degree: int = 1,
    solver: SolverName = "scipy",
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    maxiter: int = 5000,
    save_vtu: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """在多层加密网格上运行集中力载荷路径的验收基准.

    参数:
        dim: 问题维度, ``2`` 或 ``3``; 只在未显式给出 ``problem_name`` 时用于选缺省算例.
        problem_name: ``PROBLEM_REGISTRY`` 中的算例名; 为 ``None`` 时按 ``dim`` 取缺省.
        nx: 最粗档 x 方向单元数; 为 ``None`` 时取算例缺省.
        ny: 最粗档 y 方向单元数; 为 ``None`` 时取算例缺省.
        nz: 最粗档 z 方向单元数, 仅 3D 使用; 为 ``None`` 时取算例缺省.
        mesh_type: 2D 网格类型 ``quad`` 或 ``tri``; 3D 强制为 ``hex``.
        levels: 加密层数, 每层各方向单元数加倍, 至少为 ``1``.
        degree: 位移空间多项式次数, 必须为正整数.
        solver: 求解器名, 见 ``DIRECT_SOLVERS`` 与 ``ITERATIVE_SOLVERS``.
        rtol: ``cg`` 相对收敛容差, 只对迭代解法生效.
        atol: ``cg`` 绝对收敛容差, 只对迭代解法生效.
        maxiter: ``cg`` 最大迭代步数, 只对迭代解法生效.
        save_vtu: 是否把最密层位移场导出为 VTU.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.

    返回:
        summary: 逐层残差、载荷等效性、柔顺度与判定结果的汇总记录.

    异常:
        ValueError: 当次数、层数、算例名或网格类型非法时抛出.
        AssertionError: 当任一门禁不达标时抛出; 此时不写任何 JSON 文件.
    """
    if degree < 1:
        raise ValueError(f"degree 必须为正整数; 收到 degree={degree}.")
    if levels < 1:
        raise ValueError(f"levels 至少为 1; 收到 levels={levels}.")

    # 未指定算例时按维度取缺省: 3D 问题的 default_nz 非 None
    if problem_name is None:
        problem_name = "mbb-full-3d" if dim == 3 else "mbb-half"
    if problem_name not in PROBLEM_REGISTRY:
        raise ValueError(
            f"未登记的算例 '{problem_name}'; "
            f"可选 {'/'.join(PROBLEM_REGISTRY)}."
        )

    entry = PROBLEM_REGISTRY[problem_name]
    is_3d = entry.default_nz is not None

    if is_3d:
        mesh_type = "hex"
    elif mesh_type not in ("quad", "tri"):
        raise ValueError(f"2D 问题只支持 quad/tri 网格, 不支持 {mesh_type}.")

    nx = nx if nx is not None else entry.default_nx
    ny = ny if ny is not None else entry.default_ny
    nz = nz if nz is not None else entry.default_nz

    bm.set_backend("numpy")

    problem, material = create_problem_and_material(entry)

    iterative = solver in ITERATIVE_SOLVERS
    solver_options = (
        {"rtol": rtol, "atol": atol, "maxiter": maxiter} if iterative else {}
    )

    fealpy_file = import_module("fealpy").__file__
    if fealpy_file is None:
        raise RuntimeError("无法确定当前导入的 FEALPy 模块文件路径.")
    fealpy_path = str(Path(fealpy_file).resolve().parents[1])
    print(f"FEALPy: {fealpy_path}")
    print(
        f"问题={entry.label}, 网格={mesh_type}, "
        f"空间次数={degree}, 求解器={solver}"
    )
    if iterative:
        print(f"cg 参数: rtol={rtol:.1e}, atol={atol:.1e}, maxiter={maxiter}")

    rows = []
    for level in range(levels):
        solve_kwargs = {
            "problem": problem,
            "material": material,
            "entry": entry,
            "mesh_type": mesh_type,
            "nx": nx * 2**level,
            "ny": ny * 2**level,
            "degree": degree,
            "solver": solver,
            "solver_options": solver_options,
        }
        if is_3d:
            solve_kwargs["nz"] = nz * 2**level
        rows.append(solve_one_level(**solve_kwargs))

    report(rows, solver)

    vtu_path = None
    if save_vtu:
        vtu_dir = Path(__file__).resolve().parent / "outputs" / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        finest = rows[-1]
        finest_nx = finest["nx"]
        finest_ny = finest["ny"]
        finest_nz = finest.get("nz")

        # 逐层求解不保留位移场, 导出时在最密层上重解一次
        finest_mesh = create_mesh(problem, mesh_type, finest_nx, finest_ny, finest_nz)
        finest_analyzer = LagrangeFEMAnalyzer(
            disp_mesh=finest_mesh,
            pde=problem,
            material=material,
            space_degree=degree,
            integration_order=degree + 3,
            operator_level="fa",
            solve_method=solver,
            topopt_algorithm=None,
            enable_logging=False,
        )
        K0_f = finest_analyzer.assemble_stiff_matrix()
        F0_f = finest_analyzer.assemble_body_force_vector()
        K_f, F_f = finest_analyzer.apply_bc(K0_f, F0_f)
        uh_f = finest_analyzer.tensor_space.function()
        _, _ = finest_analyzer.solve_system(K_f, F_f, uh_f, **solver_options)
        disp_array = np.asarray(bm.asarray(uh_f))
        disp_f = disp_array.reshape(-1, problem.dimension)

        vtu_stem_parts = [
            problem_name, f"p{degree}", mesh_type, f"{finest_nx}x{finest_ny}"
        ]
        if is_3d:
            vtu_stem_parts.append(f"x{finest_nz}")
        vtu_stem = "_".join(vtu_stem_parts)
        vtu_path = str(vtu_dir / vtu_stem)
        export_vtu(finest_mesh, disp_f, vtu_path)
        print(f"\nVTU 已导出: {vtu_path}.vtu")

    applied_load = getattr(problem, entry.load_attr)
    residual_max = max(row["residual"] for row in rows)
    load_error_max = max(row["load_error"] for row in rows)
    residual_passed = residual_max <= RESIDUAL_TOLERANCE
    load_passed = load_error_max <= LOAD_TOLERANCE

    print(
        f"\n真相对残差最大值 = {residual_max:.2e} "
        f"(阈值 {RESIDUAL_TOLERANCE:.0e}) -> "
        f"{'通过' if residual_passed else '未通过'}"
    )
    print(
        f"载荷等效性最大偏差 = {load_error_max:.2e} "
        f"(阈值 {LOAD_TOLERANCE:.0e}, P = {applied_load}) -> "
        f"{'通过' if load_passed else '未通过'}"
    )

    converged = True
    if iterative:
        converged = all(bool(row["converged"]) for row in rows)
        print(f"cg 每层均收敛 -> {'通过' if converged else '未通过'}")

    failures = []
    if not residual_passed:
        failures.append(
            f"真相对残差最大值 {residual_max:.4e} 超出阈值 {RESIDUAL_TOLERANCE:.1e}"
        )
    if not load_passed:
        failures.append(
            f"载荷等效性最大偏差 {load_error_max:.4e} 超出阈值 {LOAD_TOLERANCE:.1e}"
        )
    if not converged:
        failures.append("cg 存在未收敛的层级")
    if failures:
        raise AssertionError(
            f"{entry.label} ({mesh_type} 网格 + {solver}): " + "; ".join(failures)
        )

    print(
        "\n结论: SOPTX 的拉格朗日位移元集中力载荷路径 "
        f"({mesh_type} 网格 + {solver}) 可用."
    )

    summary: dict[str, Any] = {
        "script": Path(__file__).name,
        "fealpy_path": fealpy_path,
        "case": entry.name,
        "case_label": entry.label,
        "problem": type(problem).__name__,
        "dimension": f"{problem.dimension}D",
        "domain": [float(value) for value in problem.domain],
        "mesh_type": mesh_type,
        "operator_level": "fa",
        "space_degree": degree,
        "material_hypothesis": material.hypothesis,
        "solver": solver,
        "solver_options": solver_options,
        "applied_load": float(applied_load),
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "load_tolerance": LOAD_TOLERANCE,
        "levels": rows,
        "max_residual": residual_max,
        "max_load_error": load_error_max,
        # 直接解法没有收敛标志, 记为 None 而不是伪造 True
        "all_levels_converged": converged if iterative else None,
        "vtu_path": f"{vtu_path}.vtu" if vtu_path is not None else None,
        "passed": True,
    }

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # 文件名带全部判别项: 不同算例、网格、次数与求解器的结果不能互相覆盖
        target = path / (
            f"concentrated_load_{problem_name}_{mesh_type}_p{degree}_{solver}.json"
        )
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[证据] 验收通过, 结果已写入: {target}")

    return summary


def main() -> int:
    arguments = parse_arguments()

    reason = solver_unavailable_reason(arguments.solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    try:
        run_concentrated_load_benchmark(
            dim=arguments.dim,
            problem_name=arguments.problem,
            nx=arguments.nx,
            ny=arguments.ny,
            nz=arguments.nz,
            mesh_type=arguments.mesh_type,
            levels=arguments.levels,
            degree=arguments.degree,
            solver=arguments.solver,
            rtol=arguments.rtol,
            atol=arguments.atol,
            maxiter=arguments.maxiter,
            save_vtu=arguments.save_vtu,
            output_dir=arguments.output_dir,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    except AssertionError as error:
        # 门禁未过在库调用侧是异常, 在命令行侧回落为退出码, 便于被脚本与 CI 消费
        print(f"\n结论: 集中力载荷路径存在问题 —— {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
