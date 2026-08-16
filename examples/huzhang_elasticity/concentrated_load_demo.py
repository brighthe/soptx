"""胡张混合有限元等效局部牵引工程基准算例.

本脚本验证两端固支梁在底边中点等效局部牵引下的载荷路径与结构合力守恒.
对比胡张混合有限元 (Hu--Zhang MFEM) 与拉格朗日有限元 (LFEM) 在同等多项式空间次数下的
真相对残差与支座/边界合力, 检验 P1 连续迹投影载荷的等效性与守恒性.

使用方法:
    # 默认快速验证: fixed-fixed 模型, 80x10 网格, k=2, 3 次数.
    python examples/huzhang_elasticity/concentrated_load_demo.py

    # 全阶次与精细网格验证: 160x20 网格, k=2, 3, 4 次数.
    python examples/huzhang_elasticity/concentrated_load_demo.py \
        --model fixed-fixed --nx 160 --ny 20 --degrees 2 3 4
"""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Any, Literal, cast

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.fem import (
    HuZhangMFEMAnalyzer,
    LagrangeFEMAnalyzer,
    P1TraceLoad,
    create_huzhang_checkerboard_mesh,
    project_patch_traction_to_p1_trace,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FixedFixedBeamCenterLoad2d


# 与 manufactured_convergence_demo.py / cases.toml 的 acceptance 一致
RESIDUAL_TOLERANCE = 1.0e-8

# 从解出的场反算的结构合力还要经过一次线性求解, 门禁放宽到求解精度量级
STRUCTURAL_LOAD_TOLERANCE = 1.0e-6

SUPPORTED_DEGREES = (1, 2, 3, 4)

DIRECT_SOLVERS = ("scipy", "mumps")
SolverName = Literal["scipy", "mumps"]
MethodName = Literal["lfem", "huzhang"]


def _build_fixed_fixed_problem(
    nx: int,
) -> tuple[FixedFixedBeamCenterLoad2d, P1TraceLoad]:
    """构造两端固支梁并注入底边 P1 迹投影后的共同牵引.

    贴片几何全部从物理问题自身的属性读取, 本文件不复述几何或载荷常数.

    参数:
        nx: 梁底边的网格单元剖分数.

    返回:
        包含更新后物理问题对象与 P1 迹载荷投影对象的二元组.
    """
    baseline = FixedFixedBeamCenterLoad2d()
    common_load = project_patch_traction_to_p1_trace(
        line=(baseline.domain[0], baseline.domain[1]),
        n_cells=nx,
        level=baseline.traction_level,
        patch=baseline.traction_patch,
        intensity=baseline.traction_intensity,
    )
    return FixedFixedBeamCenterLoad2d(traction=common_load), common_load


PROBLEM_FACTORIES = {
    "fixed-fixed": _build_fixed_fixed_problem,
}


def build_problem(
    model: str,
    nx: int,
) -> tuple[FixedFixedBeamCenterLoad2d, P1TraceLoad]:
    """根据模型名称构造物理问题并注入 P1 迹投影载荷.

    参数:
        model: 工程物理模型名称, 如 ``fixed-fixed``.
        nx: 梁底边的网格单元剖分数.

    返回:
        包含物理问题对象与 P1 迹载荷对象的二元组.
    """
    return PROBLEM_FACTORIES[model](nx)


def huzhang_structural_resultant(
    mesh: Any,
    space: Any,
    problem: FixedFixedBeamCenterLoad2d,
    stress: Any,
    integration_order: int,
) -> list[float]:
    r"""由解出的应力场积分计算边界合力向量 $\int_{\Gamma_N} \boldsymbol{\sigma}_h \cdot \boldsymbol{n}\,\mathrm{d}s$.

    应力空间按单元求值, 而牵引边界上的边分属不同的局部边号, 对应的单元重心
    坐标不同, 因此按局部边号分组 (三角形至多 3 组) 分别求值. 边的全局定向与
    单元局部定向可能相反, 但高斯点关于中点对称, 求和结果不受影响.

    参数:
        mesh: 离散三角形网格对象.
        space: 胡张有限元空间对象.
        problem: 两端固支梁物理问题实例.
        stress: 解出的胡张应力有限元函数.
        integration_order: 边界数值积分代数精度阶次 ``q``.

    返回:
        合力向量在各坐标轴上的投影分量列表 ``[Fx, Fy]``.

    异常:
        RuntimeError: 当牵引边界上未检索到任何边界边时抛出.
    """
    is_traction = bm.logical_and(
        mesh.boundary_edge_flag(),
        problem.is_traction_boundary(mesh.entity_barycenter("edge")),
    )
    edge_indices = bm.nonzero(is_traction)[0]
    if edge_indices.shape[0] == 0:
        raise RuntimeError("牵引边界上找不到边界边.")

    edge_measure = mesh.entity_measure("edge", index=edge_indices)
    normals = mesh.face_unit_normal()[edge_indices]

    face2cell = mesh.face_to_cell()[edge_indices]
    cells, local_edges = face2cell[:, 0], face2cell[:, 2]

    quadrature = mesh.quadrature_formula(integration_order, "face")
    bcs, weights = quadrature.get_quadrature_points_and_weights()

    resultant = bm.zeros((2,), dtype=bm.float64)
    for local in range(3):
        group = bm.nonzero(bm.equal(local_edges, local))[0]
        if group.shape[0] == 0:
            continue

        # 局部边 local 的两个端点即 localEdge[local], 其余分量为零
        cell_bcs = bm.zeros((bcs.shape[0], 3), dtype=bm.float64)
        first, second = mesh.localEdge[local]
        cell_bcs = bm.set_at(cell_bcs, (slice(None), int(first)), bcs[:, 0])
        cell_bcs = bm.set_at(cell_bcs, (slice(None), int(second)), bcs[:, 1])

        # (n, NQ, 3) 形状的 Voigt 应力 [sxx, sxy, syy], 与 boundary_interpolate 的 Case A 同一约定
        voigt = space.value(stress[:], cell_bcs, index=cells[group])
        nvec = normals[group][:, None, :]
        traction = bm.stack(
            [
                voigt[..., 0] * nvec[..., 0] + voigt[..., 1] * nvec[..., 1],
                voigt[..., 1] * nvec[..., 0] + voigt[..., 2] * nvec[..., 1],
            ],
            axis=-1,
        )
        resultant = resultant + bm.einsum(
            "e, q, eqd -> d", edge_measure[group], weights, traction
        )
    return bm.to_numpy(resultant).ravel().tolist()


def _component_sums(space: Any, vector: TensorLike) -> list[float]:
    """把张量空间上的自由度向量按坐标分量求和.

    参数:
        space: 矢量拉格朗日有限元张量空间.
        vector: 节点自由度张量.

    返回:
        按物理分量累加后的合力分量列表 ``[Fx, Fy]``.
    """
    dimension = space.mesh.geo_dimension()
    shape = (dimension, -1) if space.dof_priority else (-1, dimension)
    axis = 1 if space.dof_priority else 0
    return bm.to_numpy(bm.sum(vector.reshape(*shape), axis=axis)).ravel().tolist()


def lfem_structural_resultants(
    analyzer: Any,
    problem: FixedFixedBeamCenterLoad2d,
    displacement: Any,
) -> tuple[list[float], list[float]]:
    """计算 LFEM 支座反力合力与被强加自由度吞掉的右端项载荷.

    ``reaction`` 取未施加边界条件的刚度矩阵作用在解上、再限制到 Dirichlet
    自由度, 即支座实际承担了多少力; ``swallowed`` 取右端项落在这些自由度上的
    部分, 它随网格加密按 ``P * 0.27^(nx/2)`` 迅速消失.

    参数:
        analyzer: 拉格朗日有限元分析器实例.
        problem: 两端固支梁物理问题对象.
        displacement: 解出的位移有限元函数.

    返回:
        支座反力列表与被吞载荷列表组成的二元组.
    """
    space = analyzer.tensor_space
    is_boundary = space.is_boundary_dof(
        threshold=problem.is_dirichlet_boundary(), method="interp"
    )
    zero = bm.zeros_like(displacement[:])
    internal_force = analyzer.stiffness_matrix.matmul(displacement[:])
    reaction = bm.where(is_boundary, internal_force, zero)
    swallowed = bm.where(is_boundary, analyzer.force_vector[:], zero)
    # 反力与外载荷方向相反, 取负号后与目标 P 直接可比
    return (
        [-value for value in _component_sums(space, reaction)],
        _component_sums(space, swallowed),
    )


def lfem_relative_residual(
    analyzer: Any,
    problem: FixedFixedBeamCenterLoad2d,
    displacement: Any,
) -> float:
    """在自由节点自由度上计算 LFEM 相对代数残差 ``||K u - F|| / ||F||``.

    参数:
        analyzer: 拉格朗日有限元分析器实例.
        problem: 两端固支梁物理问题对象.
        displacement: 解出的位移有限元函数.

    返回:
        自由自由度上的相对残差标量.
    """
    space = analyzer.tensor_space
    residual = analyzer.stiffness_matrix.matmul(displacement[:]) - analyzer.force_vector
    is_boundary = space.is_boundary_dof(
        threshold=problem.is_dirichlet_boundary(), method="interp"
    )
    numerator = float(bm.linalg.norm(residual[~is_boundary]))
    denominator = max(
        float(bm.linalg.norm(analyzer.force_vector[~is_boundary])), 1.0e-30
    )
    return numerator / denominator


def solve_one(
    problem: FixedFixedBeamCenterLoad2d,
    material: Any,
    mesh: Any,
    method: MethodName,
    degree: int,
    use_relaxation: bool,
    solver: SolverName,
) -> dict[str, Any]:
    """在指定网格与配置下求解单一有限元离散链并收集指标.

    参数:
        problem: 包含 P1 投影载荷的两端固支梁物理问题.
        material: 各向同性线弹性材料对象.
        mesh: 棋盘格三角形网格对象.
        method: 离散方法名称, ``lfem`` 或 ``huzhang``.
        degree: 空间有限元次数 ``k``.
        use_relaxation: 是否启用角点松弛.
        solver: 直接线性求解器后端.

    返回:
        包含离散方法、空间次数、自由度总数、相对残差与结构合力结果的字典.
    """
    integration_order = 2 * degree + 2
    row: dict[str, Any] = {
        "method": method,
        "degree": degree,
        "integration_order": integration_order,
    }

    if method == "lfem":
        analyzer = LagrangeFEMAnalyzer(
            disp_mesh=mesh,
            pde=problem,
            material=material,
            space_degree=degree,
            integration_order=integration_order,
            assembly_method="standard",
            solve_method=solver,
            topopt_algorithm=None,
            interpolation_scheme=None,
        )
        state = analyzer.solve_state(rho_val=None)
        displacement = state["displacement"]

        reaction, swallowed = lfem_structural_resultants(analyzer, problem, displacement)
        row["dofs"] = int(analyzer.tensor_space.number_of_global_dofs())
        row["residual"] = lfem_relative_residual(analyzer, problem, displacement)
        row["structural"] = reaction[1]
        row["swallowed"] = swallowed[1]
        return row

    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=degree,
        integration_order=integration_order,
        use_relaxation=use_relaxation,
        solve_method=solver,
        topopt_algorithm=None,
        interpolation_scheme=None,
    )
    state = analyzer.solve_state(rho_val=None)
    stress = state["stress"]

    # HuZhangFESpace 是工厂类, __new__ 返回 2d/3d 实现, 静态类型上放宽为 Any
    stress_space: Any = analyzer.huzhang_space
    row["dofs"] = int(
        stress_space.number_of_global_dofs() + analyzer.tensor_space.number_of_global_dofs()
    )
    row["residual"] = float(analyzer.relative_state_residual())
    row["structural"] = huzhang_structural_resultant(
        mesh, stress_space, problem, stress, integration_order
    )[1]
    row["swallowed"] = None
    return row


def report(rows: list[dict[str, Any]], target: float) -> tuple[bool, bool]:
    """格式化打印对比表格并判定相对残差与结构合力守恒.

    参数:
        rows: 各方法求解结果字典列表.
        target: 目标总外力标量值 ``P``.

    返回:
        残差通过状态与合力守恒通过状态构成的布尔二元组.
    """
    print()
    header = (
        f"{'method':>8} {'k':>3} {'q':>3} {'gdof':>9} "
        f"{'residual':>11} {'Py(结构)':>14} {'被吞载荷':>12}"
    )
    print(header)
    print("-" * (len(header) + 6))

    residual_ok = True
    structural_ok = True
    for row in rows:
        structural_py = row["structural"]
        residual_ok = residual_ok and row["residual"] <= RESIDUAL_TOLERANCE
        structural_ok = structural_ok and abs(structural_py - target) <= (
            STRUCTURAL_LOAD_TOLERANCE * abs(target)
        )

        swallowed = row["swallowed"]
        print(
            f"{row['method']:>8} {row['degree']:>3} {row['integration_order']:>3} "
            f"{row['dofs']:>9} {row['residual']:>11.2e} {structural_py:>14.8g} "
            f"{'-' if swallowed is None else format(swallowed, '.3e'):>12}"
        )

    return residual_ok, structural_ok


def solver_unavailable_reason(solver: SolverName) -> str | None:
    """探测求解器后端是否可用, 不可用时返回提示原因.

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


def even_positive(text: str) -> int:
    """校验网格剖分数是否为正偶数.

    棋盘格剖分与角点松弛均要求每边单元数为偶数.

    参数:
        text: 命令行传入的剖分数文本.

    返回:
        转换后的正偶数整型值.

    异常:
        argparse.ArgumentTypeError: 当输入非正数或非偶数时抛出.
    """
    value = int(text)
    if value <= 0 or value % 2:
        raise argparse.ArgumentTypeError("必须为正偶数 (棋盘格剖分与角点松弛要求).")
    return value


def parse_arguments() -> argparse.Namespace:
    """解析集中力基准测试命令行参数.

    返回:
        包含解析后命令行选项的 ``argparse.Namespace`` 对象.
    """
    parser = argparse.ArgumentParser(
        description="胡张混合有限元的等效局部牵引工程基准算例.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(PROBLEM_FACTORIES.keys()),
        default="fixed-fixed",
        help="工程物理模型名称 (默认 fixed-fixed, 即两端固支中点受载梁).",
    )
    parser.add_argument(
        "--degrees", type=int, nargs="+", choices=SUPPORTED_DEGREES, default=[2, 3],
        help="比较空间次数 k, LFEM 用 p=k, 胡张元用应力空间次数 k (默认 2 3).",
    )
    parser.add_argument(
        "--nx", type=even_positive, default=80,
        help="横向单元数, 正偶数 (默认 80; cases.toml 用 160).",
    )
    parser.add_argument(
        "--ny", type=even_positive, default=10,
        help="纵向单元数, 正偶数 (默认 10; cases.toml 用 20).",
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
    """执行两端固支梁在各次数下的 LFEM 与胡张元求解及载荷守恒检验.

    返回:
        退出状态码, 0 表示全部判据通过, 1 表示存在未通过项或求解异常.
    """
    arguments = parse_arguments()

    solver = cast(SolverName, arguments.solver)
    reason = solver_unavailable_reason(solver)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1

    bm.set_backend("numpy")

    problem, common_load = build_problem(arguments.model, arguments.nx)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        hypothesis=problem.plane_type,
        enable_logging=False,
    )
    mesh: Any = create_huzhang_checkerboard_mesh(
        box=problem.domain, nx=arguments.nx, ny=arguments.ny
    )

    print(
        f"问题={type(problem).__name__} {problem.domain} {problem.plane_type}, "
        f"E={problem.E}, nu={problem.nu}, 贴片={problem.traction_patch}, "
        f"P={problem.P} N (P1 迹投影后合力={common_load.resultant():.12g})"
    )
    print(
        f"网格=triangle-checkerboard {arguments.nx}x{arguments.ny} "
        f"({int(mesh.number_of_cells())} 单元), "
        f"角点松弛={arguments.relaxation}, 求解器={solver}"
    )

    rows = []
    for degree in sorted(arguments.degrees):
        for method in ("lfem", "huzhang"):
            rows.append(
                solve_one(
                    problem=problem,
                    material=material,
                    mesh=mesh,
                    method=cast(MethodName, method),
                    degree=degree,
                    use_relaxation=arguments.relaxation,
                    solver=solver,
                )
            )

    residual_ok, structural_ok = report(rows, problem.P)

    print()
    print(
        f"真相对残差 <= {RESIDUAL_TOLERANCE:.0e} -> "
        f"{'通过' if residual_ok else '未通过'}\n"
        f"结构合力守恒 (两条离散链均等于 P={problem.P}, 相对容差 "
        f"{STRUCTURAL_LOAD_TOLERANCE:.0e}) -> "
        f"{'通过' if structural_ok else '未通过'}"
    )

    if residual_ok and structural_ok:
        print("\n结论: 两条离散链都把同一个载荷泛函正确传进了结构.")
        return 0

    print("\n结论: 载荷路径存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
