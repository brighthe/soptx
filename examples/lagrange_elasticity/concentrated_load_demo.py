"""集中力 (点载荷) 工程基准算例: 二维 MBB 梁右半域.

与 ``minimal_demo.py`` 的分工:

* ``minimal_demo.py`` 走制造解: 有精确解, 判据是 L2 观测收敛阶 + 真相对残差,
  回答"离散是否正确";
* 本文件走工程基准: 没有解析解, 判据是真相对残差 + 载荷等效性, 回答
  "集中力载荷路径是否被正确装配"。

问题取博士论文算例 3.1 (二维 MBB 梁) 的对称右半域, 由
``soptx.model.HalfMBBBeamRight2d`` 建模: 60 x 20 mm, 左边界对称约束
(u_x = 0), 右下角滑移支座 (u_y = 0), 左上角竖直向下集中载荷 P = 1 N,
E = 1 MPa, nu = 0.3, 平面应力。

为什么没有 L2 收敛阶判据: 二维点载荷在作用点处应力奇异, 位移解光滑性低于
制造解, 理论收敛阶也随之降低, 沿用制造解的 1.5 门槛会误报。集中力路径的
正确性由两条无歧义判据守住:

* 真相对残差 ``||K u - F|| / ||F||`` —— 线性系统确实解开了;
* 载荷等效性 ``|sum(F_sigmah) - P|`` —— 等效节点力装配没有丢力/多分力。
  这一步必须测量 ``apply_bc`` 覆盖 Dirichlet 自由度之前的 traction 向量:
  若载荷落在被强加的自由度上, 残差依然为 0, 而载荷会被静默吞掉, 只有
  载荷和校验能抓到这种失效。

注意 ``soptx.model`` 已标记 deprecated (维护中的数学问题在
``soptx.problems``); 集中力工程问题尚未迁移过去, 本算例暂时直接引用模型类。

运行::

    python examples/lagrange_elasticity/concentrated_load_demo.py
    python examples/lagrange_elasticity/concentrated_load_demo.py --levels 4
    python examples/lagrange_elasticity/concentrated_load_demo.py --mesh-type tri
    python examples/lagrange_elasticity/concentrated_load_demo.py --solver cg --rtol 1e-12
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.model.mbb_beam_2d_lfem import HalfMBBBeamRight2d


# 与 minimal_demo.py / matrix_free_elasticity/contract.py 的残差门禁保持一致
RESIDUAL_TOLERANCE = 1.0e-10
# 施加节点力总和与 P 的偏差门禁
LOAD_TOLERANCE = 1.0e-10

# 分母里出现范数时的下限
NORM_FLOOR = 1.0e-30

# 网格类型到模型 init_mesh 变体名的映射
MESH_VARIANTS = {
    "quad": "uniform_quad",
    "tri": "uniform_aligned_tri",
}

DIRECT_SOLVERS = ("scipy", "mumps")
ITERATIVE_SOLVERS = ("cg",)


def create_problem_and_material():
    """论文算例 3.1 的物理模型与材料 (参数从模型属性读取, 不重复写字面量)."""

    pde = HalfMBBBeamRight2d(
        domain=[0.0, 60.0, 0.0, 20.0],
        P=-1.0,
        E=1.0,
        nu=0.3,
        plane_type="plane_stress",
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_stress",
        youngs_modulus=pde.E,
        poisson_ratio=pde.nu,
        enable_logging=False,
    )
    return pde, material


def solve_one_level(
    pde,
    material,
    mesh_type: str,
    nx: int,
    ny: int,
    degree: int,
    solver: str,
    solver_options: dict,
) -> dict:
    """在一层网格上求解, 返回残差、载荷和与柔顺度等诊断量."""

    pde.init_mesh.set(MESH_VARIANTS[mesh_type])
    mesh = pde.init_mesh(nx=nx, ny=ny)
    integration_order = degree + 3

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=pde,
        material=material,
        space_degree=degree,
        integration_order=integration_order,
        operator_level="fa",
        solve_method=solver,
        topopt_algorithm=None,
        enable_logging=False,
    )

    # 1. 装配: 全局刚度矩阵与体力右端项 (MBB 无体力, 右端项为 0)
    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()

    # 2. 边界条件: mixed + concentrated 先把等效节点力加进右端项,
    #    再做 Dirichlet 强施加 (对称消元)
    K, F = analyzer.apply_bc(K0, F0)

    # 3. 求解: 'fa' 的 K 已经过对称消元, 迭代解法从零初值起步即可
    uh = analyzer.tensor_space.function()
    _, solver_info = analyzer.solve_system(K, F, uh, **solver_options)

    displacement = bm.asarray(uh)
    residual_norm = float(np.linalg.norm(np.asarray(K @ displacement - F)))
    load_norm = float(np.linalg.norm(np.asarray(F)))

    # 载荷等效性: 测量 apply_bc 覆盖 Dirichlet 自由度之前的 traction 向量.
    # _assemble_traction_load 是分析器内部方法, 这里直接调用是因为载荷和
    # 必须在这一步测量, 没有等价的公开入口
    traction = analyzer._assemble_traction_load(adjoint=False)
    load_sum = float(bm.sum(bm.asarray(traction)))

    compliance = float(np.asarray(displacement) @ np.asarray(F))

    return {
        "nx": nx,
        "ny": ny,
        "mesh_size": 60.0 / nx,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(analyzer.tensor_space.number_of_global_dofs()),
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        "load_sum": load_sum,
        "load_error": abs(load_sum - pde.P),
        "compliance": compliance,
        "niter": solver_info.get("niter"),
        "converged": solver_info.get("converged"),
    }


def report(rows: list[dict], solver: str) -> None:
    """打印结果表."""

    iterative = solver in ITERATIVE_SOLVERS

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
        description="集中力载荷路径的工程基准算例 (二维 MBB 梁, 论文算例 3.1)",
    )
    parser.add_argument(
        "--nx", type=int, default=60,
        help="最粗档的 x 方向单元数 (默认 60, 与论文一致)",
    )
    parser.add_argument(
        "--ny", type=int, default=20,
        help="最粗档的 y 方向单元数 (默认 20, 与论文一致)",
    )
    parser.add_argument(
        "--levels", type=int, default=3,
        help="加密层数, 每层 nx/ny 加倍 (默认 3)",
    )
    parser.add_argument(
        "--mesh-type", choices=("quad", "tri"), default="quad",
        help="网格类型 (默认 quad, 与论文一致)",
    )
    parser.add_argument(
        "--degree", type=int, default=1,
        help="位移空间次数 (默认 1, 与论文一致)",
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

    pde, material = create_problem_and_material()

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
        f"问题=HalfMBBBeamRight2d (论文算例 3.1), 网格={arguments.mesh_type}, "
        f"空间次数={arguments.degree}, 求解器={arguments.solver}"
    )
    if iterative:
        print(
            f"cg 参数: rtol={arguments.rtol:.1e}, atol={arguments.atol:.1e}, "
            f"maxiter={arguments.maxiter}"
        )

    rows = []
    for level in range(arguments.levels):
        rows.append(
            solve_one_level(
                pde=pde,
                material=material,
                mesh_type=arguments.mesh_type,
                nx=arguments.nx * 2**level,
                ny=arguments.ny * 2**level,
                degree=arguments.degree,
                solver=arguments.solver,
                solver_options=solver_options,
            )
        )

    report(rows, arguments.solver)

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
        f"(阈值 {LOAD_TOLERANCE:.0e}, P = {pde.P}) -> "
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
