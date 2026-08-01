"""胡张混合有限元求解线弹性问题的最小可运行算例.

目的是用尽量少的代码验证一件事: 当前 ``soptx`` 命名空间下的
:class:`~soptx.fem.HuZhangMFEMAnalyzer` 能够真正解出线弹性问题, 而不是只
能构造出函数空间.

问题取 :class:`soptx.problems.ExponentialSineManufacturedElasticity2D`, 单位
正方形上的平面应变制造解, 全边界施加位移边界条件 (混合形式下是自然边界,
应力空间上没有本质边界条件). 默认空间次数 ``p=3``, 满足二维胡张元的
``p >= d + 1``, 因此刚度矩阵是干净的鞍点结构 ``[[A, B], [B^T, 0]]``, 不需要
低阶稳定化项.

问题类直接取自 ``soptx.problems``, 没有任何本地适配层 —— 这本身就是算例的一
部分: 它验证维护中的 Problem 满足 ``MixedBoundaryElasticityProblem`` 契约.

判据只取无歧义的两项: 线性系统的相对平衡残差和状态矩阵的对称性缺陷, 阈值
沿用 ``experiments/huzhang_topopt_paper/cases.toml`` 里的 acceptance. 收敛阶
只打印不判定 —— 该实验目录记录的预期阶目前仍是 ``theory-audit-required``,
在理论核查完成前不应作为通过条件.

运行::

    python .\\examples\\huzhang_elasticity\\minimal_demo.py
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --degree 3 --levels 4 --relaxation
"""

from __future__ import annotations

import argparse
from math import log2
from pathlib import Path
import sys

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from soptx.fem import HuZhangMFEMAnalyzer  # noqa: E402
from soptx.materials import IsotropicLinearElasticMaterial  # noqa: E402
from soptx.problems import ExponentialSineManufacturedElasticity2D  # noqa: E402


# 与 cases.toml 的 acceptance 保持一致
RESIDUAL_TOLERANCE = 1.0e-8
SYMMETRY_TOLERANCE = 1.0e-12


def _as_float(value) -> float:
    return float(bm.to_numpy(value).reshape(-1)[0])


def solve_one_level(problem, material, degree: int, subdivisions: int,
                    integration_order: int, use_relaxation: bool) -> dict:
    """在一层网格上求解并返回误差与诊断量."""
    mesh = TriangleMesh.from_box(
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
        u=uh, v=problem.disp_solution, q=integration_order
    )
    stress_error = mesh.error(
        u=sigmah, v=problem.stress_solution, q=integration_order
    )
    div_stress_error = mesh.error(
        u=sigmah.div_value, v=problem.div_stress_solution, q=integration_order
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
        "--levels", type=int, default=4,
        help="加密层数, 第 i 层网格为 2^i x 2^i (默认 4)",
    )
    parser.add_argument(
        "--relaxation", action="store_true",
        help="开启角点松弛 (默认关闭)",
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

    problem = ExponentialSineManufacturedElasticity2D(
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
        f"问题={type(problem).__name__}, 平面假设={problem.plane_type}, "
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
