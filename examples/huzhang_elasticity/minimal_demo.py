"""胡张混合有限元求解线弹性问题的最小可运行算例.

目的是用尽量少的代码验证一件事: 当前 ``soptx`` 命名空间下的
:class:`~soptx.fem.HuZhangMFEMAnalyzer` 能够真正解出线弹性问题, 而不是只
能构造出函数空间.

问题取 ``soptx.problems`` 的混合边界制造解, 与 ``examples/lagrange_elasticity``
的分工保持一致: 那里验证拉格朗日位移元, 这里验证胡张元 —— 同一组
``MixedBoundary*Elasticity2D`` 问题, 两种离散各自验证各自的收敛链. 各制造解的
完整数学定义见
`制造解文档 <../../docs/problems/manufactured-elasticity.md>`__. 实测收敛
数据、判据与诊断见 `结果分析报告 <outputs/results_analysis.md>`__.

当前 ``soptx`` 的胡张元仅支持 2D simplex 网格, 因此本算例不提供 ``--dim``
与 ``--mesh-type`` 选项, 固定使用二维 ``triangle-checkerboard``: 从规则四边形
网格按棋盘格交替选取对角线. 这是当前软件实现的支持范围, 不是对胡--张方法数学
理论的永久限制 (角点松弛要求每个几何角点恰好连接两个三角形, 且两者共享一条
从角点出发的内部边).

问题与材料直接取自 ``soptx``, 没有本地适配层 —— 这本身就是算例的一部分:
它验证维护中的 Problem 满足 ``MixedBoundaryElasticityProblem`` 契约. 可选模型
两个:

* ``mixed-sinusoidal``: 精确位移 ``u1=u2=sin(pi x) sin(pi y)``, 左边与下边
  为位移边界, 右边与上边为 traction 边界 (默认);
* ``mixed-exp-sine``: 指数/正弦制造解, 三条边为位移边界, 右边为 traction
  边界.

全 Dirichlet 制造解 (``sinusoidal``/``exp-sine``) 在理论上也能走通
``AllDisplacementBoundaryMixin`` 提供的混合形式边界接口, 但该路径未经充分
测试, 暂不开放.

判据只取无歧义的两项: 线性系统的相对平衡残差和状态矩阵的对称性缺陷, 阈值
沿用 ``experiments/huzhang_topopt_paper/cases.toml`` 里的 acceptance. 收敛阶
只打印不判定 —— 该实验目录记录的预期阶目前仍是 ``theory-audit-required``,
在理论核查完成前不应作为通过条件.

求解器限制: 胡张元的状态方程是 ``[[A, B], [B^T, 0]]`` 型对称不定鞍点系统,
不能用 CG (``solve_state`` 对 ``cg`` 直接报错), 只提供 ``scipy`` 与 ``mumps``.

前置: SOPTX 需以 editable 方式安装 (``pip install -e .``, 见仓库 README),
这样 ``import soptx`` 直接解析到工作树的 ``src/soptx``, 脚本无需注入源码路径.

运行::

    python .\\examples\\huzhang_elasticity\\minimal_demo.py
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --model mixed-exp-sine
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --degree 2
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --no-relaxation
    python .\\examples\\huzhang_elasticity\\minimal_demo.py --solver mumps
"""

from __future__ import annotations

import argparse
from importlib import import_module
from math import log2
from pathlib import Path
import sys

from fealpy.backend import backend_manager as bm

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    create_huzhang_checkerboard_mesh,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
)


# 与 cases.toml 的 acceptance 保持一致
RESIDUAL_TOLERANCE = 1.0e-8
SYMMETRY_TOLERANCE = 1.0e-12

# 各维度最粗一档的每方向单元数, 之后逐层加倍; 胡张元角点松弛要求偶数
BASE_SUBDIVISIONS = {2: 2}

# 直接解法经 fealpy.solver.spsolve 分派到对应后端; 胡张元不支持迭代解法
DIRECT_SOLVERS = ("scipy", "mumps")


def _as_float(value) -> float:
    return float(bm.to_numpy(value).reshape(-1)[0])


def _mixed_sinusoidal_2d() -> tuple:
    """u1=u2=sin(pi x) sin(pi y); Gamma_D={x=0}∪{y=0}, Gamma_N={x=1}∪{y=1}.

    与 lagrange 版同一个问题, 但这里通过 Hu--Zhang 混合形式求解: 位移边界
    ``u=0`` 是自然边界条件 (弱施加), traction 边界 ``sigma.n=t`` 才是本质
    边界条件 (强施加).
    """

    problem = MixedBoundarySinusoidalElasticity2D()
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material


def _mixed_exponential_sine_2d() -> tuple:
    """指数/正弦制造解的混合边界视角; 右边为非零 traction 边界."""

    problem = MixedBoundaryExponentialSineElasticity2D()
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material


# 胡张元当前只开放经充分测试的混合边界制造解. 材料参数一律从 problem 的
# 属性读取而不是各写一遍字面量 —— 两者不一致时不会报错, 只会让收敛阶悄悄
# 塌掉, 是这个算例最难查的错法.
PROBLEM_FACTORIES = {
    2: {
        "mixed-sinusoidal": _mixed_sinusoidal_2d,
        "mixed-exp-sine": _mixed_exponential_sine_2d,
    },
}


def create_problem_and_material(dimension: int, model: str):
    """按模型选择制造解与材料, 二者的弹性参数由 problem 属性保证一致."""

    return PROBLEM_FACTORIES[dimension][model]()


def create_mesh(domain: tuple, subdivisions: int):
    """单位区域上的一致加密 checkerboard 网格.

    ``nx``/``ny`` 必须为正偶数: 角点松弛要求每个几何角点恰好连接两个三角形,
    且两者共享一条从角点出发的内部边.
    """

    return create_huzhang_checkerboard_mesh(
        box=domain,
        nx=subdivisions,
        ny=subdivisions,
    )


def solve_one_level(
    problem,
    material,
    degree: int,
    subdivisions: int,
    integration_order: int,
    use_relaxation: bool,
    solver: str,
) -> dict:
    """在一层网格上求解, 返回误差与诊断量.

    胡张元的 ``solve_state`` 内部完成了装配、边界条件与求解三步 —— 与
    lagrange 版把三步展开不同, 这里是混合形式, 边界条件的语义相反
    (traction 强施加, 位移弱施加), 展开反而降低可读性, 所以直接调用.
    """

    mesh = create_mesh(problem.domain, subdivisions)

    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=degree,
        integration_order=integration_order,
        use_relaxation=use_relaxation,
        solve_method=solver,
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
    """打印结果表与逐层观测阶 (仅参考, 不判定)."""

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


def solver_unavailable_reason(solver: str) -> str | None:
    """求解器后端不可用时返回原因, 可用则返回 None.

    只有 ``mumps`` 需要探测: 它依赖外部 ``mumps`` 包 (PyMUMPS), 不是 fealpy
    自带. 放在入口检查, 免得装配跑完了才在求解那一步炸.
    """

    if solver != "mumps":
        return None

    try:
        import_module("mumps")
    except Exception as exc:
        return (
            f"求解器 'mumps' 不可用 ({type(exc).__name__}: {exc}); "
            "该后端需要 PyMUMPS 包 (pip install pymumps) 及系统 MUMPS 库。"
            "请改用 --solver scipy."
        )
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="胡张混合有限元求解线弹性问题的最小算例",
    )
    parser.add_argument(
        "--model",
        choices=("mixed-sinusoidal", "mixed-exp-sine"),
        default="mixed-sinusoidal",
        help="制造解模型 (默认 mixed-sinusoidal)",
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
        "--solver", choices=DIRECT_SOLVERS, default="scipy",
        help="求解器 (默认 scipy); mumps 需要 PyMUMPS 包",
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

    reason = solver_unavailable_reason(arguments.solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    dimension = 2  # 胡张元当前仅支持 2D simplex
    problem, material = create_problem_and_material(dimension, arguments.model)
    base = BASE_SUBDIVISIONS[dimension]
    integration_order = 2 * arguments.degree + 2

    # 结论依赖于哪一份 FEALPy: 官方检出与打了缺陷修复的检出版本号都是 4.0.0,
    # 只有解析路径能区分. 这里用 import_module 而不是模块级 ``import fealpy``:
    # 后者只在这一行用到, 会被 "移除未使用导入" 的工具删掉
    fealpy_root = Path(import_module("fealpy").__file__).resolve().parents[1]
    print(f"FEALPy: {fealpy_root}")
    print(
        f"维数={dimension}D, 网格=triangle-checkerboard, "
        f"问题={type(problem).__name__}, 平面假设={problem.plane_type}, "
        f"空间次数={arguments.degree}, 积分阶={integration_order}, "
        f"角点松弛={arguments.relaxation}, 求解器={arguments.solver}"
    )

    rows = []
    for level in range(1, arguments.levels + 1):
        rows.append(
            solve_one_level(
                problem=problem,
                material=material,
                degree=arguments.degree,
                subdivisions=base * 2**level,
                integration_order=integration_order,
                use_relaxation=arguments.relaxation,
                solver=arguments.solver,
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
