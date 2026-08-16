"""胡张混合有限元线弹性制造解收敛阶验证算例.

本脚本验证二维胡张混合有限元求解器在混合边界制造解下的数值收敛性.
多层网格加密统计应力与位移的 L2 误差、H(div) 误差、代数残差与鞍点系统对称性缺陷,
覆盖正弦与指数正弦制造解以及低阶跳量稳定化格式.

使用方法:
    # 默认验证: p=3, mixed-sinusoidal 模型.
    python examples/huzhang_elasticity/manufactured_convergence_demo.py

    # 低阶稳定化验证: p=2, mixed-exp-sine 模型.
    python examples/huzhang_elasticity/manufactured_convergence_demo.py \
        --model mixed-exp-sine --degree 2
"""

from __future__ import annotations

import argparse
from importlib import import_module
from math import log2
from pathlib import Path
import sys
from typing import Any, Callable, Literal, Protocol, cast

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

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
SolverName = Literal["scipy", "mumps"]


class _ErrorEvaluableMesh(Protocol):
    """声明本示例计算误差时实际需要的网格接口."""

    def error(
        self,
        u: Any,
        v: Any,
        *,
        q: int,
    ) -> TensorLike:
        """计算给定函数对的离散误差.

        参数:
            u: 数值解或插值场.
            v: 解析解函数或真解场.
            q: 高斯积分代数精度阶次.

        返回:
            标量误差张量.
        """
        ...


class _DofCountableSpace(Protocol):
    """声明本示例统计自由度时实际需要的空间接口."""

    def number_of_global_dofs(self) -> int:
        """返回全局自由度总数.

        返回:
            全局自由度数量.
        """
        ...


def _as_float(value: TensorLike | float) -> float:
    """将标量张量或浮点数转换为 Python 原生浮点数.

    参数:
        value: 标量张量或浮点数值.

    返回:
        Python 原生 ``float`` 标量.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return float(bm.to_numpy(value).reshape(-1)[0])


def _mixed_sinusoidal_2d() -> tuple[
    MixedBoundarySinusoidalElasticity2D, IsotropicLinearElasticMaterial
]:
    """构造二维正弦混合边界制造解及对应的线弹性材料.

    精确位移 ``u1=u2=sin(pi x) sin(pi y)``; 位移边界 ``Gamma_D={x=0}∪{y=0}``,
    traction 边界 ``Gamma_N={x=1}∪{y=1}``. 与 lagrange 版同一个问题, 但这里
    通过 Hu--Zhang 混合形式求解: 位移边界 ``u=0`` 是自然边界条件 (弱施加),
    traction 边界 ``sigma.n=t`` 才是本质边界条件 (强施加).

    返回:
        制造解问题实例与各项同性线弹性材料对象构成的二元组.
    """
    problem = MixedBoundarySinusoidalElasticity2D()
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material


def _mixed_exponential_sine_2d() -> tuple[
    MixedBoundaryExponentialSineElasticity2D, IsotropicLinearElasticMaterial
]:
    """构造二维指数/正弦混合边界制造解及对应的线弹性材料.

    三条边为位移边界, 右边为非零 traction 边界.

    返回:
        制造解问题实例与各项同性线弹性材料对象构成的二元组.
    """
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
PROBLEM_FACTORIES: dict[
    int,
    dict[
        str,
        Callable[
            [],
            tuple[
                MixedBoundarySinusoidalElasticity2D
                | MixedBoundaryExponentialSineElasticity2D,
                IsotropicLinearElasticMaterial,
            ],
        ],
    ],
] = {
    2: {
        "mixed-sinusoidal": _mixed_sinusoidal_2d,
        "mixed-exp-sine": _mixed_exponential_sine_2d,
    },
}


def create_problem_and_material(
    dimension: int,
    model: str,
) -> tuple[
    MixedBoundarySinusoidalElasticity2D | MixedBoundaryExponentialSineElasticity2D,
    IsotropicLinearElasticMaterial,
]:
    """按模型名称选择制造解与材料, 二者的弹性参数由 problem 属性保证一致.

    参数:
        dimension: 空间物理维数, 当前仅支持 ``2``.
        model: 制造解模型代号, 可选 ``mixed-sinusoidal`` 或 ``mixed-exp-sine``.

    返回:
        制造解问题与材料对象的二元组.
    """
    return PROBLEM_FACTORIES[dimension][model]()


def create_mesh(domain: tuple[float, ...], subdivisions: int) -> Any:
    """在给定矩形区域上生成一致加密的 checkerboard 三角形网格.

    ``nx`` 与 ``ny`` 必须为正偶数: 角点松弛要求每个几何角点恰好连接两个三角形,
    且两者共享一条从角点出发的内部边.

    参数:
        domain: 区域边界坐标 ``(xmin, xmax, ymin, ymax)``.
        subdivisions: 各坐标轴方向的单元剖分数 (必须为正偶数).

    返回:
        构建完成的 FEALPy 棋盘格三角形网格对象.
    """
    return create_huzhang_checkerboard_mesh(
        box=domain,
        nx=subdivisions,
        ny=subdivisions,
    )


def solve_one_level(
    problem: MixedBoundarySinusoidalElasticity2D
    | MixedBoundaryExponentialSineElasticity2D,
    material: IsotropicLinearElasticMaterial,
    degree: int,
    subdivisions: int,
    integration_order: int,
    use_relaxation: bool,
    solver: SolverName,
) -> dict[str, Any]:
    """在单层网格上执行 Hu--Zhang 混合有限元求解并计算误差与诊断量.

    胡张元的 ``solve_state`` 内部完成了装配、边界条件施加与线性求解三步.
    由于是混合形式, 边界条件的语义相反 (traction 强施加, 位移弱施加), 直接调用
    分析器高级求解接口保持端到端契约.

    参数:
        problem: 制造解物理问题对象.
        material: 各向同性线弹性材料对象.
        degree: 应力有限元空间次数 ``k``.
        subdivisions: 当前网格每边剖分数.
        integration_order: 数值积分代数精度阶次 ``q``.
        use_relaxation: 是否启用角点松弛.
        solver: 直接线性求解器后端名称, 如 ``scipy`` 或 ``mumps``.

    返回:
        包含网格尺寸、自由度数、各项 L2/H(div) 误差、残差与对称性指标的字典.
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

    error_mesh = cast(_ErrorEvaluableMesh, mesh)
    disp_error = error_mesh.error(
        uh, problem.disp_solution, q=integration_order
    )
    stress_error = error_mesh.error(
        sigmah, problem.stress_solution, q=integration_order
    )
    div_stress_error = error_mesh.error(
        sigmah.div_value, problem.div_stress_solution, q=integration_order
    )
    stress_hdiv_error = bm.sqrt(
        bm.add(
            bm.multiply(stress_error, stress_error),
            bm.multiply(div_stress_error, div_stress_error),
        )
    )

    stress_space = cast(_DofCountableSpace, analyzer.huzhang_space)
    displacement_space = cast(_DofCountableSpace, analyzer.tensor_space)
    stress_dofs = stress_space.number_of_global_dofs()
    disp_dofs = displacement_space.number_of_global_dofs()

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
    """计算网格二分加密下的数值观测收敛阶.

    网格尺寸每层减半, 观测阶即为误差比值以 2 为底的对数 ``log2(coarse / fine)``.

    参数:
        coarse: 粗网格下的误差值.
        fine: 细网格下的误差值.

    返回:
        若误差均为正实数则返回观测阶浮点数; 否则返回 ``None``.
    """
    if coarse > 0.0 and fine > 0.0:
        return log2(coarse / fine)
    return None


def report(rows: list[dict[str, Any]]) -> None:
    """格式化打印收敛结果数据表与逐层观测阶.

    参数:
        rows: 各加密层网格求解结果构成的字典列表.
    """
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


def solver_unavailable_reason(solver: SolverName) -> str | None:
    """探测求解器后端是否可用, 不可用时返回提示原因.

    只有 ``mumps`` 需要探测: 它依赖外部 ``mumps`` 包 (PyMUMPS), 不是 fealpy
    自带. 在入口提前检查, 避免装配完成后才在求解步骤异常退出.

    参数:
        solver: 求解器后端名称.

    返回:
        若求解器不可用返回说明字符串; 可用时返回 ``None``.
    """
    if solver != "mumps":
        return None

    try:
        import_module("mumps")
    except Exception as exc:
        return (
            f"求解器 'mumps' 不可用 ({type(exc).__name__}: {exc}); "
            "该后端需要 PyMUMPS 包 (pip install pymumps) 及系统 MUMPS 库. "
            "请改用 --solver scipy."
        )
    return None


def parse_arguments() -> argparse.Namespace:
    """解析制造解收敛测试命令行参数.

    返回:
        包含解析后命令行选项的 ``argparse.Namespace`` 对象.
    """
    parser = argparse.ArgumentParser(
        description="胡张混合有限元求解线弹性问题的制造解收敛算例.",
    )
    parser.add_argument(
        "--model",
        choices=("mixed-sinusoidal", "mixed-exp-sine"),
        default="mixed-sinusoidal",
        help="制造解模型 (默认 mixed-sinusoidal).",
    )
    parser.add_argument(
        "--degree", type=int, default=3,
        help="应力空间次数, 二维胡张元要求 p >= 3 才无需低阶稳定化 (默认 3).",
    )
    parser.add_argument(
        "--levels", type=int, default=5,
        help="加密层数, 第 i 层网格为 2^i x 2^i (默认 5).",
    )
    parser.add_argument(
        "--solver", choices=DIRECT_SOLVERS, default="scipy",
        help="求解器 (默认 scipy); mumps 需要 PyMUMPS 包.",
    )
    parser.add_argument(
        "--relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭角点松弛 (默认启用).",
    )
    return parser.parse_args()


def main() -> int:
    """执行多层网格加密求解、收敛阶统计与判据检验.

    返回:
        退出状态码, 0 表示全部判据通过, 1 表示存在未通过项或输入错误.
    """
    arguments = parse_arguments()
    if arguments.degree < 1:
        print("degree 必须为正整数.", file=sys.stderr)
        return 1
    if arguments.levels < 1:
        print("levels 必须为正整数.", file=sys.stderr)
        return 1

    solver = cast(SolverName, arguments.solver)
    reason = solver_unavailable_reason(solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    dimension = 2  # 胡张元当前仅支持 2D simplex
    problem, material = create_problem_and_material(dimension, arguments.model)
    base = BASE_SUBDIVISIONS[dimension]
    integration_order = 2 * arguments.degree + 2

    print(
        f"维数={dimension}D, 网格=triangle-checkerboard, "
        f"问题={type(problem).__name__}, 平面假设={problem.plane_type}, "
        f"空间次数={arguments.degree}, 积分阶={integration_order}, "
        f"角点松弛={arguments.relaxation}, 求解器={solver}"
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
                solver=solver,
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
