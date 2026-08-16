"""拉格朗日位移元求解线弹性问题的最小可运行算例 (CPU 串行, 全装配).

目的是用尽量少的代码走通一条完整的求解链: 装配全局刚度矩阵 -> 施加边界条件
-> 直接解 -> 与制造解比较, 并观察 L2 误差的收敛阶。二维和三维共用同一段流程,
只有网格类型、问题类和材料假设按维数选择。

与 ``examples/matrix_free_elasticity`` 的关系: 那里同时承担 MPI 重叠副本、
FA/EA 双路对照和可重放 evidence 三件事, 因此有十余个模块。本算例只保留 CPU
串行 FA 这一条主路径, 不导入那个目录的任何模块, 也不生成 evidence。想看
matrix-free 的算子层级和并行, 去读那个目录; 想看"有限元怎么把方程解出来",
读这一个文件就够。

问题类和材料类直接取自 ``soptx``, 没有本地适配层 —— 这本身就是算例的一部分:
它验证维护中的 Problem 满足 ``DirichletElasticityProblem`` 契约。

判据两项, 都无歧义:

* 真相对残差 ``||K u - F|| / ||F||`` —— 线性系统是否真的解开了;
* 最细一档的 L2 观测收敛阶 —— 离散是否正确。P1 与 Q1 元的 L2 误差理论阶都是
  2, 阈值取 1.5, 与 ``tools/matrix_free_evidence/contract.py`` 的门禁一致。

用 ``cg`` 时额外要求每一层都收敛: 迭代解法的真残差达标只说明这一次侥幸解对了,
没收敛却残差合格不能算通过。

网格类型与制造解都按维数配对: 网格 2D 是 ``tri``/``quad``、3D 是
``tet``/``hex``; 模型 2D 是 ``sinusoidal``/``exp-sine`` (全 Dirichlet) 与
``mixed-sinusoidal``/``mixed-exp-sine`` (混合边界), 3D 只有 ``divfree-poly``。
交叉组合在入口报错。各制造解的完整数学定义见
`制造解文档 <../../docs/problems/manufactured-elasticity.md>`__。

两个 ``mixed-`` 模型的右端项多一段 traction 边界积分, 走的是全 Dirichlet 模型
碰不到的面积分装配路径; 它们的强 Dirichlet 只施加在 Gamma_D 上。``quad`` 与 ``hex`` 是张量积网格, FEALPy 4.0.0 在这两条路上都出过基
函数缺陷, 见 ``docs/known-issues/fealpy-tensor-product-mesh.md``; 那里的判据落
在基函数层, 这里的收敛阶落在求解层, 两者不能互相替代。

前置: SOPTX 需以 editable 方式安装 (``pip install -e .``, 见仓库 README),
这样 ``import soptx`` 直接解析到工作树的 ``src/soptx``, 脚本里不必再改
``sys.path``。

运行::

    python .\\examples\\lagrange_elasticity\\minimal_demo.py
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --dim 3
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --dim 2 --levels 4
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --mesh-type quad
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --dim 3 --mesh-type hex
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --model exp-sine
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --model mixed-sinusoidal
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --solver cg --rtol 1e-12
"""

from __future__ import annotations

import argparse
from importlib import import_module
from math import log2
from pathlib import Path
import sys

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    MixedBoundaryExponentialSineElasticity2D,
    MixedBoundarySinusoidalElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)


# 与 tools/matrix_free_evidence/contract.py 的对应门禁保持一致
RESIDUAL_TOLERANCE = 1.0e-10
MINIMUM_L2_ORDER = 1.5

# 各维度最粗一档的每方向单元数, 之后逐层加倍
BASE_SUBDIVISIONS = {2: 8, 3: 4}

# 各维度可用的网格类型; 交叉组合 (2D 的 hex、3D 的 quad) 在入口报错
MESH_CONSTRUCTORS = {
    2: {"tri": TriangleMesh, "quad": QuadrangleMesh},
    3: {"tet": TetrahedronMesh, "hex": HexahedronMesh},
}

MESH_LABELS = {
    "tri": "triangle",
    "quad": "quadrangle",
    "tet": "tetrahedron",
    "hex": "hexahedron",
}

# 分母里出现范数时的下限
NORM_FLOOR = 1.0e-30

# 直接解法经 fealpy.solver.spsolve 分派到对应后端; cg 是迭代解法
DIRECT_SOLVERS = ("scipy", "mumps")
ITERATIVE_SOLVERS = ("cg",)


def _sinusoidal_2d() -> tuple:
    """u=(sin(pi x) sin(pi y), 0), 全 Dirichlet.

    第二个位移分量恒为零, 因此两个分量之间的耦合项基本不被激活。
    """

    domain = (0.0, 1.0, 0.0, 1.0)
    problem = SinusoidalPlaneStrainElasticity2D(domain=domain)
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        enable_logging=False,
    )
    return problem, material, domain


def _exponential_sine_2d() -> tuple:
    """u=(e^{x-y} x(1-x) y(1-y), sin(pi x) sin(pi y)), 全 Dirichlet.

    两个位移分量都非平凡, 能激活 sinusoidal 覆盖不到的耦合项。
    """

    domain = (0.0, 1.0, 0.0, 1.0)
    problem = ExponentialSineManufacturedElasticity2D(domain=domain)
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material, domain


def _divergence_free_3d() -> tuple:
    """无散多项式位移, 全 Dirichlet; 体力只依赖剪切模量."""

    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    problem = DivergenceFreePolynomialElasticity3D(domain=domain)
    material = IsotropicLinearElasticMaterial(
        hypothesis="3D",
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material, domain


def _mixed_sinusoidal_2d() -> tuple:
    """u1=u2=sin(pi x) sin(pi y); Gamma_D={x=0}∪{y=0}, Gamma_N={x=1}∪{y=1}.

    右端项因此多一段 traction 边界积分, 走的是全 Dirichlet 模型完全碰不到的
    面积分装配路径。
    """

    problem = MixedBoundarySinusoidalElasticity2D()
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material, problem.domain


def _mixed_exponential_sine_2d() -> tuple:
    """与 exp-sine 同一组精确场, 但右边改判为非零 traction 边界."""

    problem = MixedBoundaryExponentialSineElasticity2D()
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material, problem.domain


# 各维度可用的制造解。材料参数一律从 problem 的属性读取而不是各写一遍字面量
# —— 两者不一致时不会报错, 只会让收敛阶悄悄塌掉, 是这个算例最难查的错法。
PROBLEM_FACTORIES = {
    2: {
        "sinusoidal": _sinusoidal_2d,
        "exp-sine": _exponential_sine_2d,
        "mixed-sinusoidal": _mixed_sinusoidal_2d,
        "mixed-exp-sine": _mixed_exponential_sine_2d,
    },
    3: {
        "divfree-poly": _divergence_free_3d,
    },
}


def create_problem_and_material(dimension: int, model: str):
    """按模型选择制造解与材料, 二者的弹性参数由 problem 属性保证一致."""

    return PROBLEM_FACTORIES[dimension][model]()


def create_mesh(
    dimension: int,
    domain: tuple,
    subdivisions: int,
    mesh_type: str,
):
    """单位区域上的一致加密网格.

    ``quad`` 与 ``hex`` 是张量积网格, 走的基函数实现与单纯形网格完全不同 ——
    FEALPy 4.0.0 在这两条路上都出过基函数缺陷, 见
    ``docs/known-issues/fealpy-tensor-product-mesh.md``。
    """

    constructor = MESH_CONSTRUCTORS[dimension][mesh_type]

    if dimension == 2:
        return constructor.from_box(
            list(domain),
            nx=subdivisions,
            ny=subdivisions,
        )
    return constructor.from_box(
        list(domain),
        nx=subdivisions,
        ny=subdivisions,
        nz=subdivisions,
    )


def solve_one_level(
    problem,
    material,
    domain: tuple,
    dimension: int,
    degree: int,
    subdivisions: int,
    mesh_type: str,
    solver: str,
    solver_options: dict,
) -> dict:
    """在一层网格上求解, 返回误差与诊断量.

    这里不调用 ``analyzer.solve_state()``, 而是把它内部的三步展开写出来 ——
    整个算例想说明的就是这三步。
    """

    mesh = create_mesh(dimension, domain, subdivisions, mesh_type)
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

    # 1. 装配: 全局刚度矩阵与体力右端项
    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()

    # 2. 边界条件: 'fa' 走对称消元, 直接改写已装配好的矩阵; 混合边界模型还会
    #    先把 Gamma_N 的 traction 等效载荷加进右端项
    K, F = analyzer.apply_bc(K0, F0)

    # 3. 求解: 直接解法或 cg。'fa' 的 K 已经过对称消元, 迭代解法从零初值起步
    #    即可; 只有 'ea' 需要把 apply_bc 留下的 prescribed_solution 传成 x0
    uh = analyzer.tensor_space.function()
    _, solver_info = analyzer.solve_system(K, F, uh, **solver_options)

    # Function 不能直接进 @, 残差在裸数组上算
    displacement = bm.asarray(uh)
    residual_norm = float(np.linalg.norm(np.asarray(K @ displacement - F)))
    load_norm = float(np.linalg.norm(np.asarray(F)))
    l2_error = float(
        mesh.error(problem.disp_solution, uh, q=integration_order)
    )

    return {
        "subdivisions": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(analyzer.tensor_space.number_of_global_dofs()),
        "l2_error": l2_error,
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        # 直接解法没有迭代信息, 保持为 None 而不是伪造 0
        "niter": solver_info.get("niter"),
        "converged": solver_info.get("converged"),
    }


def observed_order(coarse: float, fine: float) -> float | None:
    """网格每层减半, 观测阶即误差比值的以 2 为底对数."""

    if coarse > 0.0 and fine > 0.0:
        return log2(coarse / fine)
    return None


def report(rows: list[dict], solver: str) -> list[float]:
    """打印结果表并返回逐层观测阶."""

    iterative = solver in ITERATIVE_SOLVERS

    header = (
        f"{'n':>4} {'cells':>9} {'gdof':>9} {'h':>9} "
        f"{'|u-uh|_0':>12} {'residual':>11}"
    )
    if iterative:
        header += f" {'niter':>7} {'conv':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['subdivisions']:>4} {row['cells']:>9} "
            f"{row['dofs']:>9} {row['mesh_size']:>9.4f} "
            f"{row['l2_error']:>12.4e} {row['residual']:>11.2e}"
        )
        if iterative:
            line += f" {row['niter']:>7} {str(row['converged']):>6}"
        print(line)

    orders: list[float] = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        value = observed_order(coarse["l2_error"], fine["l2_error"])
        if value is not None:
            orders.append(value)

    if orders:
        print(
            "\nL2 观测收敛阶: "
            + " ".join(f"{value:.3f}" for value in orders)
        )
    return orders


def solver_unavailable_reason(solver: str) -> str | None:
    """求解器后端不可用时返回原因, 可用则返回 None.

    只有 ``mumps`` 需要探测: 它依赖外部 ``mumps`` 包 (PyMUMPS), 不是 fealpy
    自带。放在入口检查, 免得装配跑完了才在求解那一步炸。
    """

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
        description="拉格朗日位移元求解线弹性问题的最小算例 (串行 FA)",
    )
    parser.add_argument(
        "--dim", type=int, choices=(2, 3), default=2,
        help="空间维数 (默认 2)",
    )
    parser.add_argument(
        "--degree", type=int, default=1,
        help="位移空间次数 (默认 1)",
    )
    parser.add_argument(
        "--levels", type=int, default=3,
        help="加密层数, 每层单元数加倍 (默认 3)",
    )
    parser.add_argument(
        "--mesh-type", choices=("tri", "quad", "tet", "hex"), default=None,
        help=(
            "网格类型; 2D 可选 tri/quad (默认 tri), "
            "3D 可选 tet/hex (默认 tet)"
        ),
    )
    parser.add_argument(
        "--model",
        choices=(
            "sinusoidal", "exp-sine",
            "mixed-sinusoidal", "mixed-exp-sine",
            "divfree-poly",
        ),
        default=None,
        help=(
            "制造解模型; 2D 可选 sinusoidal/exp-sine (全 Dirichlet) 与 "
            "mixed-sinusoidal/mixed-exp-sine (混合边界), 默认 sinusoidal; "
            "3D 只有 divfree-poly"
        ),
    )
    parser.add_argument(
        "--solver", choices=DIRECT_SOLVERS + ITERATIVE_SOLVERS,
        default="scipy",
        help="求解器 (默认 scipy); mumps 需要 PyMUMPS 包",
    )
    # 以下三项只对 cg 生效, 默认值与 LagrangeFEMAnalyzer.solve_system 一致
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
    if arguments.levels < 2:
        print("levels 至少为 2, 否则无法观测收敛阶", file=sys.stderr)
        return 1
    # 网格类型按维数配对, 不给就取该维数的单纯形网格
    available_mesh_types = MESH_CONSTRUCTORS[arguments.dim]
    mesh_type = arguments.mesh_type or ("tri" if arguments.dim == 2 else "tet")
    if mesh_type not in available_mesh_types:
        print(
            f"--mesh-type {mesh_type} 不适用于 {arguments.dim}D, "
            f"可选 {'/'.join(available_mesh_types)}",
            file=sys.stderr,
        )
        return 1

    # 制造解同样按维数配对
    available_models = PROBLEM_FACTORIES[arguments.dim]
    model = arguments.model or (
        "sinusoidal" if arguments.dim == 2 else "divfree-poly"
    )
    if model not in available_models:
        print(
            f"--model {model} 不适用于 {arguments.dim}D, "
            f"可选 {'/'.join(available_models)}",
            file=sys.stderr,
        )
        return 1

    reason = solver_unavailable_reason(arguments.solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    dimension = arguments.dim
    problem, material, domain = create_problem_and_material(dimension, model)
    base = BASE_SUBDIVISIONS[dimension]

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

    # 结论依赖于哪一份 FEALPy: 官方检出与打了缺陷修复的检出版本号都是 4.0.0,
    # 只有解析路径能区分。见 docs/known-issues/fealpy-tensor-product-mesh.md
    # 这里用 import_module 而不是模块级 ``import fealpy``: 后者只在这一行用到,
    # 会被 "移除未使用导入" 的工具删掉, 而删掉的后果是整个算例起不来
    fealpy_file = import_module("fealpy").__file__
    if fealpy_file is None:
        raise RuntimeError("无法确定当前导入的 FEALPy 模块文件路径.")
    print(f"FEALPy: {Path(fealpy_file).resolve().parents[1]}")
    print(
        f"维数={dimension}D, 网格={MESH_LABELS[mesh_type]}, "
        f"问题={type(problem).__name__}, 算子层级=fa, "
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
                problem=problem,
                material=material,
                domain=domain,
                dimension=dimension,
                degree=arguments.degree,
                subdivisions=base * 2**level,
                mesh_type=mesh_type,
                solver=arguments.solver,
                solver_options=solver_options,
            )
        )

    orders = report(rows, arguments.solver)

    residual_max = max(row["residual"] for row in rows)
    residual_passed = residual_max <= RESIDUAL_TOLERANCE
    final_order = orders[-1] if orders else 0.0
    order_passed = final_order >= MINIMUM_L2_ORDER
    decreasing = all(
        coarse["l2_error"] > fine["l2_error"]
        for coarse, fine in zip(rows[:-1], rows[1:])
    )

    print(
        f"\n真相对残差最大值 = {residual_max:.2e} "
        f"(阈值 {RESIDUAL_TOLERANCE:.0e}) -> "
        f"{'通过' if residual_passed else '未通过'}"
    )
    print(
        f"最细一档 L2 观测阶 = {final_order:.3f} "
        f"(阈值 {MINIMUM_L2_ORDER}) -> "
        f"{'通过' if order_passed else '未通过'}"
    )
    print(
        f"L2 误差逐层下降 -> {'通过' if decreasing else '未通过'}"
    )

    # 迭代解法多一项: 真残差达标不能代替收敛判定, 没收敛而残差碰巧合格
    # 只说明这一次侥幸, 不能作为求解链可用的证据
    converged = True
    if iterative:
        converged = all(bool(row["converged"]) for row in rows)
        print(
            f"cg 每层均收敛 -> {'通过' if converged else '未通过'}"
        )

    if residual_passed and order_passed and decreasing and converged:
        print(
            f"\n结论: SOPTX 的拉格朗日位移元串行 FA 求解链在 "
            f"{dimension}D {MESH_LABELS[mesh_type]} "
            f"网格 + {arguments.solver} 上可用."
        )
        return 0

    print("\n结论: 求解链存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
