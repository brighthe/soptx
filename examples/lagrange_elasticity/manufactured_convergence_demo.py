"""拉格朗日位移元求解线弹性问题的制造解收敛阶算例 (CPU 串行, 全装配).

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
函数缺陷, 见 ``docs/known-issues/fealpy-patches.md`` 第一节; 那里的判据落
在基函数层, 这里的收敛阶落在求解层, 两者不能互相替代。

前置: SOPTX 需以 editable 方式安装 (``pip install -e .``, 见仓库 README),
这样 ``import soptx`` 直接解析到工作树的 ``src/soptx``, 脚本里不必再改
``sys.path``。

运行::

    python examples/lagrange_elasticity/manufactured_convergence_demo.py
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --dim 3
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --dim 2 --levels 4 --base 4
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --dim 3 --levels 4
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --mesh-type quad
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --dim 3 --mesh-type hex
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --model exp-sine
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --model mixed-sinusoidal
    python examples/lagrange_elasticity/manufactured_convergence_demo.py --solver cg --rtol 1e-12

全部门禁在 ``run_manufactured_convergence_benchmark`` 内以运行时断言实现: 任一项不达标
即抛 ``AssertionError`` 且不写任何文件, 全部通过才落盘 JSON 证据。契约与实测证据见同目录
``results_analysis.md``。

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析, 与从哪个目录发起命令
无关; 传相对路径时按当前工作目录解析, 可能落到 ``.gitignore`` 覆盖范围之外。
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from math import log2
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.verification import solution_error
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

# 各维度最粗一档的每方向单元数缺省值, 之后逐层加倍; 可用 ``--base`` 覆盖。
#
# 缺省值按各维度的单档成本定: 2D 从 8 起, 三档就落在误差已进入渐近区的网格上;
# 3D 从 4 起, 因为同样的每方向剖分数在 3D 是三次方的自由度。
#
# 要把两个维度放在同一段 h 区间上并排比较 (例如图表里 2D/3D 两条链画在一起),
# 用 ``--base`` 显式对齐, 不要改这里的缺省值 —— 缺省值一动, 已归档的整套证据
# 就都不再能用缺省命令复现。
#
# 基数与层数一起决定加密序列, 因此它们随每次运行记进 JSON 的
# ``base_subdivisions`` 与 ``refinement_levels``: 文件名只区分配置
# (维数/网格/模型/次数/求解器), 不区分加密序列, 同一配置换一组序列重跑会覆盖
# 旧文件, 靠 JSON 内部这两项自述当前口径。
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

# 与 LagrangeFEMAnalyzer 的 solve_method 形参取值域保持一致, 避免传入未支持的求解器名
SolverName = Literal["scipy", "mumps", "cg"]

# 直接解法经 fealpy.solver.spsolve 分派到对应后端; cg 是迭代解法
DIRECT_SOLVERS: tuple[SolverName, ...] = ("scipy", "mumps")
ITERATIVE_SOLVERS: tuple[SolverName, ...] = ("cg",)

# 与 LagrangeFEMAnalyzer 的 assembly_method 形参取值域保持一致。三条路径描述的是
# 同一个双线性型, 只是收缩次序不同, 因此解必须逐位一致; 差别在装配期的临时数组规模。
AssemblyMethodName = Literal["standard", "voigt", "fast"]
ASSEMBLY_METHODS: tuple[AssemblyMethodName, ...] = ("standard", "voigt", "fast")


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
    ``docs/known-issues/fealpy-patches.md`` 第一节。
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
    solver: SolverName,
    solver_options: dict[str, Any],
    assembly_method: AssemblyMethodName,
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
        assembly_method=assembly_method,
        solve_method=solver,
        topopt_algorithm=None,
        enable_logging=False,
    )

    # 计时覆盖装配到求解的完整链路, 不含网格生成与误差积分
    started = time.perf_counter()

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

    elapsed = time.perf_counter() - started

    # Function 不能直接进 @, 残差在裸数组上算
    displacement = bm.asarray(uh)
    residual_norm = float(np.linalg.norm(np.asarray(K @ displacement - F)))
    load_norm = float(np.linalg.norm(np.asarray(F)))
    # 绝对与相对两个口径都记: 观测阶只看比值, 两者给出同一个数; 但跨维度、跨
    # 算例比较误差大小时, 只有除掉精确解范数的相对误差可比。定义取自
    # ``soptx.fem.verification``, 与 matrix-free 那条路径共用同一个函数, 两边的
    # 数字因此可以直接对照。
    l2_error, l2_relative = solution_error(mesh, uh, problem, degree)

    return {
        "subdivisions": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(analyzer.tensor_space.number_of_global_dofs()),
        "l2_error": l2_error,
        "l2_relative": l2_relative,
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        "seconds": float(elapsed),
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
        f"{'|u-uh|_0':>12} {'rel L2':>12} {'residual':>11} {'sec':>8}"
    )
    if iterative:
        header += f" {'niter':>7} {'conv':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['subdivisions']:>4} {row['cells']:>9} "
            f"{row['dofs']:>9} {row['mesh_size']:>9.4f} "
            f"{row['l2_error']:>12.4e} {row['l2_relative']:>12.4e} "
            f"{row['residual']:>11.2e} {row['seconds']:>8.4f}"
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
        "--base", type=int, default=None,
        help=(
            "最粗一档的每方向单元数; 默认取 BASE_SUBDIVISIONS 中该维数的值。"
            "高次元或 3D 上想把整条链整体放粗一档时用它, 不必改源码"
        ),
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
    parser.add_argument(
        "--mumps-sym", type=int, choices=(0, 1, 2), default=0,
        help=(
            "MUMPS 对称性标志, 只对 --solver mumps 生效 (默认 0); "
            "0 一般非对称, 1 对称正定, 2 一般对称。1/2 只读下三角"
        ),
    )
    parser.add_argument(
        "--assembly-method", choices=ASSEMBLY_METHODS,
        default="standard",
        help=(
            "单元刚度阵装配路径 (默认 standard); 三者是同一双线性型的不同收缩"
            "次序, 解应逐位一致, 差别在装配期临时数组规模"
        ),
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
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("outputs")),
        help="通过验收后写入 JSON 证据的目录 (默认为本脚本同级的 outputs/)",
    )
    return parser.parse_args()


def run_manufactured_convergence_benchmark(
    dim: int = 2,
    model: str | None = None,
    mesh_type: str | None = None,
    degree: int = 1,
    levels: int = 3,
    base: int | None = None,
    solver: SolverName = "scipy",
    assembly_method: AssemblyMethodName = "standard",
    mumps_sym: int = 0,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    maxiter: int = 5000,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """在多层嵌套加密网格上运行制造解验收基准.

    参数:
        dim: 空间维数, ``2`` 或 ``3``.
        model: 制造解模型名; 为 ``None`` 时取该维数的缺省模型.
        mesh_type: 网格类型; 为 ``None`` 时取该维数的单纯形网格.
        degree: 位移空间多项式次数, 必须为正整数.
        levels: 网格加密层数, 至少为 ``2`` 才能观测收敛阶.
        base: 最粗一档的每方向单元数; 为 ``None`` 时取 ``BASE_SUBDIVISIONS[dim]``.
        solver: 求解器名, 见 ``DIRECT_SOLVERS`` 与 ``ITERATIVE_SOLVERS``.
        assembly_method: 单元刚度阵的装配路径, 见 ``ASSEMBLY_METHODS``.
        mumps_sym: MUMPS 对称性标志, 只对 ``solver='mumps'`` 生效. ``0`` 按一般
            非对称矩阵分解, ``1`` 对称正定, ``2`` 一般对称.
        rtol: ``cg`` 相对收敛容差, 只对迭代解法生效.
        atol: ``cg`` 绝对收敛容差, 只对迭代解法生效.
        maxiter: ``cg`` 最大迭代步数, 只对迭代解法生效.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.

    返回:
        summary: 逐层误差、观测收敛阶、门禁阈值与判定结果的汇总记录.

    异常:
        ValueError: 当次数或层数非法, 或模型与网格类型同维数不匹配时抛出.
        AssertionError: 当任一门禁不达标时抛出; 此时不写任何文件.
    """
    if degree < 1:
        raise ValueError(f"degree 必须为正整数; 收到 degree={degree}.")
    if levels < 2:
        raise ValueError(f"levels 至少为 2, 否则无法观测收敛阶; 收到 levels={levels}.")
    if base is not None and base < 1:
        raise ValueError(f"base 必须为正整数; 收到 base={base}.")

    # 网格类型按维数配对, 不给就取该维数的单纯形网格
    available_mesh_types = MESH_CONSTRUCTORS[dim]
    mesh_type = mesh_type or ("tri" if dim == 2 else "tet")
    if mesh_type not in available_mesh_types:
        raise ValueError(
            f"mesh_type '{mesh_type}' 不适用于 {dim}D, "
            f"可选 {'/'.join(available_mesh_types)}."
        )

    # 制造解同样按维数配对
    available_models = PROBLEM_FACTORIES[dim]
    model = model or ("sinusoidal" if dim == 2 else "divfree-poly")
    if model not in available_models:
        raise ValueError(
            f"model '{model}' 不适用于 {dim}D, 可选 {'/'.join(available_models)}."
        )

    bm.set_backend("numpy")

    problem, material, domain = create_problem_and_material(dim, model)
    base = base if base is not None else BASE_SUBDIVISIONS[dim]

    iterative = solver in ITERATIVE_SOLVERS
    if iterative:
        solver_options: dict[str, Any] = {
            "rtol": rtol, "atol": atol, "maxiter": maxiter
        }
    elif solver == "mumps":
        # 只有 mumps 认这个开关; scipy 后端没有对应参数, 传了会被忽略, 干脆不传
        solver_options = {"sym": mumps_sym}
    else:
        solver_options = {}

    # 结论依赖于哪一份 FEALPy: 官方检出与打了缺陷修复的检出版本号都是 4.0.0,
    # 只有解析路径能区分。见 docs/known-issues/fealpy-patches.md 第一节
    # 这里用 import_module 而不是模块级 ``import fealpy``: 后者只在这一行用到,
    # 会被 "移除未使用导入" 的工具删掉, 而删掉的后果是整个算例起不来
    fealpy_file = import_module("fealpy").__file__
    if fealpy_file is None:
        raise RuntimeError("无法确定当前导入的 FEALPy 模块文件路径.")
    fealpy_path = str(Path(fealpy_file).resolve().parents[1])
    print(f"FEALPy: {fealpy_path}")
    print(
        f"维数={dim}D, 网格={MESH_LABELS[mesh_type]}, "
        f"问题={type(problem).__name__}, 算子层级=fa, "
        f"空间次数={degree}, 求解器={solver}, 装配方法={assembly_method}"
    )
    if solver == "mumps":
        print(f"mumps 参数: sym={mumps_sym}")
    if iterative:
        print(f"cg 参数: rtol={rtol:.1e}, atol={atol:.1e}, maxiter={maxiter}")

    rows = []
    for level in range(levels):
        rows.append(
            solve_one_level(
                problem=problem,
                material=material,
                domain=domain,
                dimension=dim,
                degree=degree,
                subdivisions=base * 2**level,
                mesh_type=mesh_type,
                solver=solver,
                solver_options=solver_options,
                assembly_method=assembly_method,
            )
        )

    orders = report(rows, solver)

    # 逐层观测阶回填到各层记录: 最粗一档没有上一层可比, 记为 None 而不是 0
    rows[0]["l2_order"] = None
    for row, order in zip(rows[1:], orders):
        row["l2_order"] = order

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
    print(f"L2 误差逐层下降 -> {'通过' if decreasing else '未通过'}")

    # 迭代解法多一项: 真残差达标不能代替收敛判定, 没收敛而残差碰巧合格
    # 只说明这一次侥幸, 不能作为求解链可用的证据
    converged = True
    if iterative:
        converged = all(bool(row["converged"]) for row in rows)
        print(f"cg 每层均收敛 -> {'通过' if converged else '未通过'}")

    failures = []
    if not residual_passed:
        failures.append(
            f"真相对残差最大值 {residual_max:.4e} 超出阈值 {RESIDUAL_TOLERANCE:.1e}"
        )
    if not order_passed:
        failures.append(
            f"最细一档 L2 观测阶 {final_order:.4f} 低于门禁阈值 {MINIMUM_L2_ORDER}"
        )
    if not decreasing:
        failures.append("L2 误差未逐层严格下降")
    if not converged:
        failures.append("cg 存在未收敛的层级")
    if failures:
        raise AssertionError(
            f"{dim}D {MESH_LABELS[mesh_type]} 网格 + {solver}: " + "; ".join(failures)
        )

    print(
        f"\n结论: SOPTX 的拉格朗日位移元串行 FA 求解链在 "
        f"{dim}D {MESH_LABELS[mesh_type]} 网格 + {solver} 上可用."
    )

    summary: dict[str, Any] = {
        "script": Path(__file__).name,
        "fealpy_path": fealpy_path,
        "dimension": f"{dim}D",
        "problem": type(problem).__name__,
        "model": model,
        "mesh_type": mesh_type,
        "mesh_label": MESH_LABELS[mesh_type],
        "operator_level": "fa",
        "assembly_method": assembly_method,
        "space_degree": degree,
        # 加密序列自述: 文件名不带这两项, 判读 JSON 时靠它们确认口径
        "base_subdivisions": base,
        "refinement_levels": levels,
        "material_hypothesis": material.hypothesis,
        "solver": solver,
        "solver_options": solver_options,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "minimum_l2_order_gate": MINIMUM_L2_ORDER,
        "theoretical_order": float(degree + 1),
        "levels": rows,
        "final_l2_order": final_order,
        "max_residual": residual_max,
        "l2_error_decreasing": decreasing,
        # 直接解法没有收敛标志, 记为 None 而不是伪造 True
        "all_levels_converged": converged if iterative else None,
        "passed": True,
    }

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # 文件名带全部判别项: 不同网格、模型、次数与求解器的结果不能互相覆盖,
        # 否则同一份 JSON 会把不同口径的历史数值混成一组证据
        # 缺省装配路径不进文件名, 已冻结的证据文件因此保持原名; 非缺省路径各占
        # 一个文件, 免得不同收缩次序的结果互相覆盖。判读时以 JSON 内的
        # ``assembly_method`` 字段为准, 文件名只是去重手段
        method_tag = "" if assembly_method == "standard" else f"_{assembly_method}"
        target = path / (
            f"manufactured_convergence_{dim}d_{mesh_type}_{model}"
            f"_p{degree}_{solver}{method_tag}.json"
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
        run_manufactured_convergence_benchmark(
            dim=arguments.dim,
            model=arguments.model,
            mesh_type=arguments.mesh_type,
            degree=arguments.degree,
            levels=arguments.levels,
            base=arguments.base,
            solver=arguments.solver,
            assembly_method=arguments.assembly_method,
            mumps_sym=arguments.mumps_sym,
            rtol=arguments.rtol,
            atol=arguments.atol,
            maxiter=arguments.maxiter,
            output_dir=arguments.output_dir,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    except AssertionError as error:
        # 门禁未过在库调用侧是异常, 在命令行侧回落为退出码, 便于被脚本与 CI 消费
        print(f"\n结论: 求解链存在问题 —— {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
