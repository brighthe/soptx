"""拉格朗日位移元求解线弹性问题的最小可运行算例 (CPU 串行, 全装配).

目的是用尽量少的代码走通一条完整的求解链: 装配全局刚度矩阵 -> 施加 Dirichlet
边界 -> 直接解 -> 与制造解比较, 并观察 L2 误差的收敛阶。二维和三维共用同一段
流程, 只有网格类型、问题类和材料假设按维数选择。

与 ``examples/matrix_free_elasticity`` 的关系: 那里同时承担 MPI 重叠副本、
FA/EA 双路对照和可重放 evidence 三件事, 因此有十余个模块。本算例只保留 CPU
串行 FA 这一条主路径, 不导入那个目录的任何模块, 也不生成 evidence。想看
matrix-free 的算子层级和并行, 去读那个目录; 想看"有限元怎么把方程解出来",
读这一个文件就够。

问题类和材料类直接取自 ``soptx``, 没有本地适配层 —— 这本身就是算例的一部分:
它验证维护中的 Problem 满足 ``DirichletElasticityProblem`` 契约。

判据两项, 都无歧义:

* 真相对残差 ``||K u - F|| / ||F||`` —— 线性系统是否真的解开了;
* 最细一档的 L2 观测收敛阶 —— 离散是否正确。P1 元的 L2 误差理论阶为 2,
  阈值取 1.5, 与 ``matrix_free_elasticity/contract.py`` 的门禁一致。

前置: SOPTX 需以 editable 方式安装 (``pip install -e .``, 见仓库 README),
这样 ``import soptx`` 直接解析到工作树的 ``src/soptx``, 脚本里不必再改
``sys.path``。

运行::

    python .\\examples\\lagrange_elasticity\\minimal_demo.py
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --dim 3
    python .\\examples\\lagrange_elasticity\\minimal_demo.py --dim 2 --levels 4
"""

from __future__ import annotations

import argparse
from math import log2
import sys

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, TriangleMesh

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)


# 与 matrix_free_elasticity/contract.py 的对应门禁保持一致
RESIDUAL_TOLERANCE = 1.0e-10
MINIMUM_L2_ORDER = 1.5

# 各维度最粗一档的每方向单元数, 之后逐层加倍
BASE_SUBDIVISIONS = {2: 8, 3: 4}

# 分母里出现范数时的下限
NORM_FLOOR = 1.0e-30


def create_problem_and_material(dimension: int):
    """按维数选择制造解与材料, 二者的弹性参数必须一致."""

    if dimension == 2:
        domain = (0.0, 1.0, 0.0, 1.0)
        youngs_modulus, poisson_ratio = 1.0, 0.3
        problem = SinusoidalPlaneStrainElasticity2D(
            domain=domain,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        material = IsotropicLinearElasticMaterial(
            hypothesis="plane_strain",
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            enable_logging=False,
        )
        return problem, material, domain

    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    lame_lambda, shear_modulus = 1.0, 1.0
    problem = DivergenceFreePolynomialElasticity3D(
        domain=domain,
        lame_lambda=lame_lambda,
        shear_modulus=shear_modulus,
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis="3D",
        lame_lambda=lame_lambda,
        shear_modulus=shear_modulus,
        enable_logging=False,
    )
    return problem, material, domain


def create_mesh(dimension: int, domain: tuple, subdivisions: int):
    """单位区域上的一致加密网格."""

    if dimension == 2:
        return TriangleMesh.from_box(
            list(domain),
            nx=subdivisions,
            ny=subdivisions,
        )
    return TetrahedronMesh.from_box(
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
) -> dict:
    """在一层网格上求解, 返回误差与诊断量.

    这里不调用 ``analyzer.solve_state()``, 而是把它内部的三步展开写出来 ——
    整个算例想说明的就是这三步。
    """

    mesh = create_mesh(dimension, domain, subdivisions)
    integration_order = degree + 3

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=degree,
        integration_order=integration_order,
        operator_level="fa",
        solve_method="scipy",
        topopt_algorithm=None,
        enable_logging=False,
    )

    # 1. 装配: 全局刚度矩阵与体力右端项
    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()

    # 2. 边界条件: 'fa' 走对称消元, 直接改写已装配好的矩阵
    K, F = analyzer.apply_bc(K0, F0)

    # 3. 求解: 串行直接解法
    uh = analyzer.tensor_space.function()
    analyzer.solve_system(K, F, uh)

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
    }


def observed_order(coarse: float, fine: float) -> float | None:
    """网格每层减半, 观测阶即误差比值的以 2 为底对数."""

    if coarse > 0.0 and fine > 0.0:
        return log2(coarse / fine)
    return None


def report(rows: list[dict]) -> list[float]:
    """打印结果表并返回逐层观测阶."""

    header = (
        f"{'n':>4} {'cells':>9} {'gdof':>9} {'h':>9} "
        f"{'|u-uh|_0':>12} {'residual':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subdivisions']:>4} {row['cells']:>9} "
            f"{row['dofs']:>9} {row['mesh_size']:>9.4f} "
            f"{row['l2_error']:>12.4e} {row['residual']:>11.2e}"
        )

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
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.degree < 1:
        print("degree 必须为正整数", file=sys.stderr)
        return 1
    if arguments.levels < 2:
        print("levels 至少为 2, 否则无法观测收敛阶", file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    dimension = arguments.dim
    problem, material, domain = create_problem_and_material(dimension)
    base = BASE_SUBDIVISIONS[dimension]

    print(
        f"维数={dimension}D, 问题={type(problem).__name__}, "
        f"算子层级=fa, 空间次数={arguments.degree}, "
        f"求解器=scipy 直接解"
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
            )
        )

    orders = report(rows)

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

    if residual_passed and order_passed and decreasing:
        print(
            f"\n结论: SOPTX 的拉格朗日位移元串行 FA 求解链在 "
            f"{dimension}D 上可用."
        )
        return 0

    print("\n结论: 求解链存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
