"""胡张混合有限元求解线弹性问题的最小可运行算例.

目的是用尽量少的代码验证一件事: 当前 ``soptx`` 命名空间下的
:class:`~soptx.fem.HuZhangMFEMAnalyzer` 能够真正解出线弹性问题, 而不是只
能构造出函数空间.

问题取 :class:`soptx.problems.MixedBoundarySinusoidalElasticity2D`, 与论文制造解
收敛验证保持一致: 单位正方形上的平面应变问题, 精确位移两个分量均为
``sin(pi*x)*sin(pi*y)``, 左边和下边施加齐次位移条件, 右边和上边施加精确应力
法向迹给出的非齐次牵引. 默认空间次数 ``p=3``, 满足二维胡张元的
``p >= d + 1``, 因此刚度矩阵是干净的鞍点结构 ``[[A, B], [B^T, 0]]``; ``p=1,2``
时分析器自动启用低阶跳量稳定化.

网格限制: 当前 SOPTX 的 Hu--Zhang 实现仅支持 simplex mesh, 即二维 triangle
和三维 tetrahedron. 本算例固定使用二维 triangle, 不支持 quadrilateral 或
hexahedral, 因而不提供 ``--mesh-type`` 选项. 这是当前软件实现的支持范围,
不是对 Hu--Zhang 方法数学理论的永久限制.

当前二维角点松弛还要求每个几何角点恰好连接两个三角形, 且两者共享一条从角点
出发的内部边. 因此本算例使用 ``triangle-checkerboard``: 从规则四边形网格按
棋盘格交替选取对角线. 这项限制属于当前松弛实现, 不是一般 Hu--Zhang 理论限制.

问题类直接取自 ``soptx.problems``, 没有任何本地适配层 —— 这本身就是算例的一
部分: 它验证维护中的 Problem 满足 ``MixedBoundaryElasticityProblem`` 契约.

判据只取无歧义的两项: 线性系统的相对平衡残差和状态矩阵的对称性缺陷, 阈值
沿用 ``experiments/huzhang_topopt_paper/cases.toml`` 里的 acceptance. 收敛阶
只打印不判定 —— 该实验目录记录的预期阶目前仍是 ``theory-audit-required``,
在理论核查完成前不应作为通过条件.

前置: SOPTX 需以 editable 方式安装 (``pip install -e .``, 见仓库 README),
这样 ``import soptx`` 直接解析到工作树的 ``src/soptx``, 脚本无需注入源码路径.

运行::

    python .\\examples\\huzhang_elasticity\\minimal_demo.py
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --degree 2
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --no-relaxation
"""

from __future__ import annotations

import argparse
from math import log2
import sys

from fealpy.backend import backend_manager as bm

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundarySinusoidalElasticity2D


# 与 cases.toml 的 acceptance 保持一致
RESIDUAL_TOLERANCE = 1.0e-8
SYMMETRY_TOLERANCE = 1.0e-12


def _as_float(value) -> float:
    return float(bm.to_numpy(value).reshape(-1)[0])


def solve_one_level(problem, material, degree: int, subdivisions: int,
                    integration_order: int, use_relaxation: bool) -> dict:
    """在一层网格上求解并返回误差与诊断量."""
    mesh = create_huzhang_checkerboard_mesh(
        box=problem.domain,
        nx=subdivisions,
        ny=subdivisions,
    )

    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=degree,
        integration_order=integration_order,
        use_relaxation=use_relaxation,
        solve_method="scipy",
        topopt_algorithm=None,
    )

    state = analyzer.solve_state(rho_val=None)
    sigmah, uh = state["stress"], state["displacement"]

    disp_error = mesh.error(
        uh, problem.disp_solution, q=integration_order
    )
    stress_error = mesh.error(
        sigmah, problem.stress_solution, q=integration_order
    )
    div_stress_error = mesh.error(
        sigmah.div_value, problem.div_stress_solution, q=integration_order
    )
    stress_hdiv_error = bm.sqrt(stress_error**2 + div_stress_error**2)

    stress_dofs = analyzer.huzhang_space.number_of_global_dofs()
    disp_dofs = analyzer.tensor_space.number_of_global_dofs()

    return {
        "subdivisions": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "total_dofs": int(stress_dofs + disp_dofs),
        "disp_error": _as_float(disp_error),
        "stress_error": _as_float(stress_error),
        "div_stress_error": _as_float(div_stress_error),
        "stress_hdiv_error": _as_float(stress_hdiv_error),
        "residual": analyzer.relative_state_residual(),
        "symmetry_error": analyzer.state_matrix_symmetry_error(),
    }


def observed_order(coarse: float, fine: float) -> float | None:
    """网格每层减半, 观测阶即误差比值的以 2 为底对数."""
    if coarse > 0.0 and fine > 0.0:
        return log2(coarse / fine)
    return None


def report(rows: list[dict]) -> None:
    header = (
        f"{'nx':>4} {'gdof':>8} {'h':>9} "
        f"{'|u-uh|_0':>11} {'|s-sh|_0':>11} "
        f"{'|div(s-sh)|_0':>14} {'|s-sh|_Hdiv':>12} "
        f"{'residual':>11} {'symmetry':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subdivisions']:>4} {row['total_dofs']:>8} "
            f"{row['mesh_size']:>9.4f} "
            f"{row['disp_error']:>11.4e} {row['stress_error']:>11.4e} "
            f"{row['div_stress_error']:>14.4e} "
            f"{row['stress_hdiv_error']:>12.4e} "
            f"{row['residual']:>11.2e} {row['symmetry_error']:>11.2e}"
        )

    if len(rows) < 2:
        return

    print("\n观测收敛阶 (仅供参考, 预期阶尚未完成理论核查, 不作为判据):")
    names = [
        ("disp_error", "|u-uh|_0    "),
        ("stress_error", "|s-sh|_0    "),
        ("div_stress_error", "|div(s-sh)|_0"),
        ("stress_hdiv_error", "|s-sh|_Hdiv "),
    ]
    for key, label in names:
        orders = []
        for coarse, fine in zip(rows[:-1], rows[1:]):
            value = observed_order(coarse[key], fine[key])
            orders.append("   n/a" if value is None else f"{value:6.2f}")
        print(f"  {label}: {' '.join(orders)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="胡张混合有限元求解线弹性问题的最小算例",
    )
    parser.add_argument(
        "--degree", type=int, default=3,
        help="应力空间次数, 二维胡张元要求 p >= 3 才无需低阶稳定化 (默认 3)",
    )
    parser.add_argument(
        "--levels", type=int, default=5,
        help="加密层数, 第 i 层网格为 2^i x 2^i (默认 5)",
    )
    parser.add_argument(
        "--relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭角点松弛 (默认启用)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.degree < 1:
        print("degree 必须为正整数", file=sys.stderr)
        return 1
    if arguments.levels < 1:
        print("levels 必须为正整数", file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    problem = MixedBoundarySinusoidalElasticity2D(
        lame_lambda=1.0,
        shear_modulus=0.5,
    )
    material = IsotropicLinearElasticMaterial(
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    integration_order = 2 * arguments.degree + 2

    print(
        f"网格=triangle-checkerboard, 问题={type(problem).__name__}, "
        f"平面假设={problem.plane_type}, "
        f"空间次数={arguments.degree}, 积分阶={integration_order}, "
        f"角点松弛={arguments.relaxation}"
    )

    rows = []
    for level in range(1, arguments.levels + 1):
        rows.append(
            solve_one_level(
                problem=problem,
                material=material,
                degree=arguments.degree,
                subdivisions=2**level,
                integration_order=integration_order,
                use_relaxation=arguments.relaxation,
            )
        )

    report(rows)

    residual_max = max(row["residual"] for row in rows)
    symmetry_max = max(row["symmetry_error"] for row in rows)
    residual_passed = residual_max <= RESIDUAL_TOLERANCE
    symmetry_passed = symmetry_max <= SYMMETRY_TOLERANCE

    print(
        f"\n相对平衡残差最大值 = {residual_max:.2e} "
        f"(阈值 {RESIDUAL_TOLERANCE:.0e}) -> "
        f"{'通过' if residual_passed else '未通过'}"
    )
    print(
        f"状态矩阵对称性缺陷最大值 = {symmetry_max:.2e} "
        f"(阈值 {SYMMETRY_TOLERANCE:.0e}) -> "
        f"{'通过' if symmetry_passed else '未通过'}"
    )

    if residual_passed and symmetry_passed:
        print("\n结论: SOPTX 的胡张混合有限元求解链在该算例上可用.")
        return 0

    print("\n结论: 求解链存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
