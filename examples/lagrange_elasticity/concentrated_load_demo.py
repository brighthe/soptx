"""集中力 (点载荷) 工程基准算例.

与 ``minimal_demo.py`` 的分工:

* ``minimal_demo.py`` 走制造解: 有精确解, 判据是 L2 观测收敛阶 + 真相对残差,
  回答"离散是否正确";
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

运行::

    python examples/lagrange_elasticity/concentrated_load_demo.py
    python examples/lagrange_elasticity/concentrated_load_demo.py --problem cantilever-corner
    python examples/lagrange_elasticity/concentrated_load_demo.py --levels 4
    python examples/lagrange_elasticity/concentrated_load_demo.py --mesh-type tri
    python examples/lagrange_elasticity/concentrated_load_demo.py --solver cg --rtol 1e-12
    python examples/lagrange_elasticity/concentrated_load_demo.py --save-vtu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TriangleMesh

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeam3d, HalfMBBBeamRight2d, HalfMBBBeamRight3d
from soptx.visualization.vtk_export import export_vtu


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


# 与 minimal_demo.py / matrix_free_elasticity/contract.py 的残差门禁保持一致
RESIDUAL_TOLERANCE = 1.0e-10
# 施加节点力总和与 P 的偏差门禁
LOAD_TOLERANCE = 1.0e-10

# 分母里出现范数时的下限
NORM_FLOOR = 1.0e-30

DIRECT_SOLVERS = ("scipy", "mumps")
ITERATIVE_SOLVERS = ("cg",)


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
    solver: str,
    solver_options: dict,
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

    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()
    K, F = analyzer.apply_bc(K0, F0)

    uh = analyzer.tensor_space.function()
    _, solver_info = analyzer.solve_system(K, F, uh, **solver_options)

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
        "mesh_size": problem.domain[1] / nx,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(analyzer.tensor_space.number_of_global_dofs()),
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        "load_sum": load_sum,
        "load_error": abs(load_sum - applied_load),
        "compliance": compliance,
        "niter": solver_info.get("niter"),
        "converged": solver_info.get("converged"),
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
            f"{'compliance':>12}"
        )
    else:
        header = (
            f"{'nx':>5} {'cells':>9} {'gdof':>9} {'h':>9} "
            f"{'residual':>11} {'load_sum':>10} {'load_err':>10} "
            f"{'compliance':>12}"
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
                f"{row['compliance']:>12.6e}"
            )
        else:
            line = (
                f"{row['nx']:>5} {row['cells']:>9} {row['dofs']:>9} "
                f"{row['mesh_size']:>9.4f} {row['residual']:>11.2e} "
                f"{row['load_sum']:>10.6f} {row['load_error']:>10.2e} "
                f"{row['compliance']:>12.6e}"
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
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.degree < 1:
        print("degree 必须为正整数", file=sys.stderr)
        return 1
    if arguments.levels < 1:
        print("levels 至少为 1", file=sys.stderr)
        return 1

    reason = solver_unavailable_reason(arguments.solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    # 根据维度确定问题和网格类型
    if arguments.problem is None:
        if arguments.dim == 2:
            problem_name = "mbb-half"
            mesh_type = arguments.mesh_type if arguments.mesh_type != "hex" else "quad"
        else:  # dim == 3
            problem_name = "mbb-full-3d"
            mesh_type = "hex"
    else:
        problem_name = arguments.problem
        mesh_type = arguments.mesh_type

    # 3D 问题必须用 hex 网格
    if arguments.dim == 3 and mesh_type != "hex":
        print(f"错误: 3D 问题只支持 hex 网格，不支持 {mesh_type}", file=sys.stderr)
        return 1

    entry = PROBLEM_REGISTRY[problem_name]
    if entry.default_domain != (0.0, 60.0, 0.0, 20.0) and not (arguments.nx or arguments.ny):
        # 对于非标准问题（如 3D），给出友好提示
        pass

    nx = arguments.nx if arguments.nx is not None else entry.default_nx
    ny = arguments.ny if arguments.ny is not None else entry.default_ny
    nz = arguments.nz if arguments.nz is not None else entry.default_nz

    problem, material = create_problem_and_material(entry)

    iterative = arguments.solver in ITERATIVE_SOLVERS
    solver_options = (
        {
            "rtol": arguments.rtol,
            "atol": arguments.atol,
            "maxiter": arguments.maxiter,
        }
        if iterative
        else {}
    )

    fealpy_root = Path(import_module("fealpy").__file__).resolve().parents[1]
    print(f"FEALPy: {fealpy_root}")
    print(
        f"问题={entry.label}, 网格={mesh_type}, "
        f"空间次数={arguments.degree}, 求解器={arguments.solver}"
    )
    if iterative:
        print(
            f"cg 参数: rtol={arguments.rtol:.1e}, atol={arguments.atol:.1e}, "
            f"maxiter={arguments.maxiter}"
        )

    rows = []
    for level in range(arguments.levels):
        solve_kwargs = {
            "problem": problem,
            "material": material,
            "entry": entry,
            "mesh_type": mesh_type,
            "nx": nx * 2**level,
            "ny": ny * 2**level,
            "degree": arguments.degree,
            "solver": arguments.solver,
            "solver_options": solver_options,
        }
        if arguments.dim == 3:
            solve_kwargs["nz"] = nz * 2**level
        rows.append(solve_one_level(**solve_kwargs))

    report(rows, arguments.solver)

    if arguments.save_vtu:
        vtu_dir = Path(__file__).resolve().parent / "outputs" / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        finest = rows[-1]
        finest_nx = finest["nx"]
        finest_ny = finest["ny"]
        finest_nz = finest.get("nz")

        vtu_mesh_kwargs = {
            "mesh_type": mesh_type,
            "nx": finest_nx,
            "ny": finest_ny,
        }
        if arguments.dim == 3:
            vtu_mesh_kwargs["nz"] = finest_nz

        finest_mesh = create_mesh(problem, mesh_type, finest_nx, finest_ny, finest_nz)
        finest_analyzer = LagrangeFEMAnalyzer(
            disp_mesh=finest_mesh,
            pde=problem,
            material=material,
            space_degree=arguments.degree,
            integration_order=arguments.degree + 3,
            operator_level="fa",
            solve_method=arguments.solver,
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

        vtu_stem_parts = [problem_name, f"p{arguments.degree}", mesh_type, f"{finest_nx}x{finest_ny}"]
        if arguments.dim == 3:
            vtu_stem_parts.append(f"x{finest_nz}")
        vtu_stem = "_".join(vtu_stem_parts)
        vtu_path = str(vtu_dir / vtu_stem)
        export_vtu(finest_mesh, disp_f, vtu_path)
        print(f"\nVTU 已导出: {vtu_path}.vtu")

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
        f"(阈值 {LOAD_TOLERANCE:.0e}, P = {getattr(problem, entry.load_attr)}) -> "
        f"{'通过' if load_passed else '未通过'}"
    )

    converged = True
    if iterative:
        converged = all(bool(row["converged"]) for row in rows)
        print(
            f"cg 每层均收敛 -> {'通过' if converged else '未通过'}"
        )

    if residual_passed and load_passed and converged:
        print(
            "\n结论: SOPTX 的拉格朗日位移元集中力载荷路径 "
            f"({arguments.mesh_type} 网格 + {arguments.solver}) 可用."
        )
        return 0

    print("\n结论: 集中力载荷路径存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
