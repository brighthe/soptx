"""胡张混合有限元的 traction 载荷路径工程基准.

与 ``minimal_demo.py`` 的分工:

* ``minimal_demo.py`` 走制造解全过程: 有精确解, 判据是 L2 观测收敛阶 (仅
  参考) + 真相对残差 + 对称性缺陷, 回答"离散是否正确";
* 本文件聚焦 traction 边界载荷路径: 回答"强加的 traction 边界数据是否被
  正确翻译成结构上的力"。

与 ``examples/lagrange_elasticity/concentrated_load_demo.py`` 的对应关系:
那里验证集中力载荷路径 (载荷直接加到右端项, 判据是 ``|sum(F)-P|``); 这里
验证分布 traction 路径 —— 但胡张元是混合形式, traction 边界 ``σ·n=t`` 是
**本质边界条件**, 由 ``apply_traction_boundary_condition`` 强加在应力自由度
上, 不像拉格朗日元那样加进右端项向量. 因此没有"装配成节点力向量"这一步,
载荷等效性改用**边界合力守恒**度量:

.. math::

   F = \\int_{\\Gamma_N} \\sigma\\cdot n \\, ds

数值合力 ``F_num`` 由解出的应力场 ``σ_h`` 在 traction 边界上积分得到, 解析
合力 ``F_ref`` 由制造解的精确应力积分得到. 两者之差是应力场的**边界泛函**
(由散度定理 + div 交换图可证其精确等于位移边界上的反作用力误差), 收敛比
应力场 L2 误差更快 (``p=3`` 实测 ~4 阶、``p=4`` 实测 ~6 阶), 所以判据取
最细一层的相对差阈值 (而非机器精度), 并要求逐层递减; 合力差的值照常打印
作为判据, 不再打印它的收敛阶. 制造解同时给出精确位移与应力解, 因此输出
位移与**应力场 L2 误差**及其收敛阶 (``p=3`` 时分别实测 ~3 / ~4 阶,
``p=4`` 时 ~4 / ~5 阶; 应力阶 $h^{p+1}$ 是胡张元超收敛, 见
``outputs/results_analysis.md`` 1.1 节) 作为诊断量.

问题类直接取自 ``soptx.problems`` 的混合边界制造解, 与 ``minimal_demo.py``
同一组:

* ``mixed-sinusoidal``: 右边与上边为 traction 边界 (默认);
* ``mixed-exp-sine``: 右边为 traction 边界.

各制造解的完整数学定义见
`制造解文档 <../../docs/problems/manufactured-elasticity.md>`__。实测合力
收敛数据与判据见 `结果分析报告 <outputs/results_analysis.md>`__。

网格固定为 ``triangle-checkerboard`` (同 ``minimal_demo.py``): 当前胡张元
角点松弛要求每个几何角点恰好连接两个三角形, 且两者共享一条从角点出发的
内部边. 默认 ``degree=3``, 此时位移空间是次数 ``2`` 的不连续 Lagrange 空间,
自由度不直接对应网格节点 —— ``--save-vtu`` 导出的位移场需要先做节点插值.

运行::

    python .\\examples\\huzhang_elasticity\\concentrated_load_demo.py
    python .\\examples\\huzhang_elasticity\\concentrated_load_demo.py --problem mixed-exp-sine
    python .\\examples\\huzhang_elasticity\\concentrated_load_demo.py --levels 4
    python .\\examples\\huzhang_elasticity\\concentrated_load_demo.py --no-relaxation
    python .\\examples\\huzhang_elasticity\\concentrated_load_demo.py --save-vtu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from math import log2
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

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
from soptx.visualization.vtk_export import export_vtu


# ---------------------------------------------------------------------------
# traction 载荷算例注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TractionLoadProblemEntry:
    """traction 载荷路径工程基准算例的元数据。"""

    name: str
    label: str
    factory: Callable[..., Any]


PROBLEM_REGISTRY: dict[str, TractionLoadProblemEntry] = {
    "mixed-sinusoidal": TractionLoadProblemEntry(
        name="mixed-sinusoidal",
        label="sinusoidal 制造解 (右边+上边 traction)",
        factory=lambda: MixedBoundarySinusoidalElasticity2D(),
    ),
    "mixed-exp-sine": TractionLoadProblemEntry(
        name="mixed-exp-sine",
        label="exp-sine 制造解 (右边 traction)",
        factory=lambda: MixedBoundaryExponentialSineElasticity2D(),
    ),
}


# 真相对残差门禁与 minimal_demo.py / cases.toml 一致
RESIDUAL_TOLERANCE = 1.0e-8
# 边界合力守恒的相对差门禁 (收敛量, 取比最细层实测值宽松一档)
LOAD_TOLERANCE = 1.0e-5


def _as_float(value) -> float:
    return float(bm.to_numpy(value).reshape(-1)[0])


def _voigt_to_traction(voigt: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """二维 Voigt 应力 ``[sxx, sxy, syy]`` 与法向作用得到 ``σ·n``.

    应力张量 ``[[sxx, sxy], [sxy, syy]]`` 点乘法向 ``(nx, ny)``:
    ``σ·n = (sxx*nx + sxy*ny, sxy*nx + syy*ny)``。
    """

    return np.stack(
        [
            voigt[..., 0] * normal[..., 0] + voigt[..., 1] * normal[..., 1],
            voigt[..., 1] * normal[..., 0] + voigt[..., 2] * normal[..., 1],
        ],
        axis=-1,
    )


def traction_force_balance(
    analyzer: HuZhangMFEMAnalyzer,
    problem,
    sigmah,
    integration_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 traction 边界合力 ``F_num`` (数值) 与 ``F_ref`` (解析).

    数值合力: 用解出的应力场 ``σ_h`` 在每条 traction 边界边的高斯积分点上
    求值, 点乘法向后按边长与权重积分求和. 解析合力: 同一组高斯点上的
    ``problem.traction_bc`` (精确应力的 Voigt 向量) 作同样的变换与积分.

    胡张元应力空间 ``HuZhangFESpace2d`` 的 ``value`` 方法在子集单元上求值
    有缺陷 (``cell_to_dof(index)`` 忽略 index), 因此这里对全部单元求值再按
    边所在单元索引, 代价只是多算几个单元.
    """

    mesh = analyzer.disp_mesh
    # ``solve_state`` 返回的应力是 fealpy Function. numpy 后端的 ``bm.to_numpy``
    # 是恒等函数, 对 Function 原样返回; 转系数数组要经其 ``__array__`` 协议
    sigmah_dof = np.asarray(sigmah)

    bc_edge = mesh.entity_barycenter("edge")
    is_traction = problem.is_traction_boundary(bc_edge)
    bdedge = np.asarray(mesh.boundary_edge_flag() & is_traction).astype(bool)
    if not bool(bdedge.any()):
        raise RuntimeError("traction 边界上找不到边界边")

    edges = np.asarray(mesh.entity("edge"))[bdedge]
    node = np.asarray(mesh.entity("node"))
    e2c = np.asarray(mesh.face_to_cell()[bdedge])
    cids = e2c[:, 0].astype(int)
    loc = e2c[:, 2].astype(int)          # 边在单元中的局部边号
    norms = np.asarray(mesh.face_unit_normal()[bdedge])
    lens = np.asarray(mesh.entity_measure("edge")[bdedge])
    nbf = len(cids)

    # 三角形局部边: 边 e 的对面顶点是 e, 边两端是另外两个顶点
    other = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    qf = mesh.quadrature_formula(integration_order, "edge")
    bcs_edge, ws = qf.get_quadrature_points_and_weights()
    bcs_edge_np = np.asarray(bcs_edge)
    ws_np = np.asarray(ws)
    nq = len(bcs_edge_np)

    # 边高斯点 (1D 重心) -> 单元重心坐标 (3 分量)
    bc_tri = np.zeros((nbf, nq, 3))
    for b in range(nbf):
        i, j = other[int(loc[b])]
        bc_tri[b, :, i] = bcs_edge_np[:, 0]
        bc_tri[b, :, j] = bcs_edge_np[:, 1]

    # 对全部单元求值, 再按 (单元, 高斯点) 配对索引
    bc_flat = bm.tensor(bc_tri.reshape(-1, 3), dtype=bm.float64)
    val_all = np.asarray(analyzer.huzhang_space.value(sigmah_dof, bc_flat))
    row_idx = np.repeat(cids, nq)
    col_idx = np.arange(nbf * nq)
    voigt_num = val_all[row_idx, col_idx].reshape(nbf, nq, 3)

    F_num = np.einsum(
        "q, b, bqk -> k",
        ws_np, lens, _voigt_to_traction(voigt_num, norms[:, None, :]),
    )

    # 解析合力: 同一组高斯点的精确应力
    edge_pts = node[edges]
    gauss_pts = (
        bcs_edge_np[:, 0][None, :, None] * edge_pts[:, 0][:, None, :]
        + bcs_edge_np[:, 1][None, :, None] * edge_pts[:, 1][:, None, :]
    )
    voigt_ref = np.asarray(
        problem.traction_bc(gauss_pts.reshape(-1, 2))
    ).reshape(nbf, nq, 3)
    F_ref = np.einsum(
        "q, b, bqk -> k",
        ws_np, lens, _voigt_to_traction(voigt_ref, norms[:, None, :]),
    )

    return F_num, F_ref


def extract_nodal_displacement(uh, tensor_space, mesh) -> np.ndarray:
    """把不连续 Lagrange 位移场插值到网格节点 (按共享单元平均).

    位移空间是 ``TensorFunctionSpace(LagrangeFESpace(p=degree-1, ctype='D'))``,
    自由度是单元局部的, 不直接对应网格节点. 这里在每个单元的三个顶点重心
    坐标处求值, 再把共享同一节点的多个单元的取值平均 —— 对连续物理量这是
    标准的跨单元平均后处理.
    """

    # 同 traction_force_balance: Function 转系数数组经 __array__, 不经过
    # bm.to_numpy (numpy 后端对非数组对象是恒等函数)
    uh_dof = np.asarray(uh)
    nodes = np.asarray(mesh.entity("node"))
    cells = np.asarray(mesh.entity("cell"))
    n_nodes = nodes.shape[0]

    bc_vertices = bm.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bm.float64)
    val = np.asarray(tensor_space.value(uh_dof, bc_vertices))  # (NC, 3, GD)

    displacement = np.zeros((n_nodes, val.shape[-1]))
    counts = np.zeros(n_nodes)
    for c in range(cells.shape[0]):
        for local in range(3):
            displacement[cells[c, local]] += val[c, local]
            counts[cells[c, local]] += 1
    displacement /= counts[:, None]
    return displacement


def create_problem_and_material(
    entry: TractionLoadProblemEntry,
) -> tuple[Any, IsotropicLinearElasticMaterial]:
    """根据注册表条目创建问题实例和对应的材料对象."""

    problem = entry.factory()
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    return problem, material


def solve_one_level(
    problem,
    material,
    degree: int,
    subdivisions: int,
    integration_order: int,
    use_relaxation: bool,
) -> dict:
    """在一层网格上求解, 返回真相对残差与边界合力相对差."""

    mesh = create_huzhang_checkerboard_mesh(
        box=problem.domain, nx=subdivisions, ny=subdivisions,
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
    F_num, F_ref = traction_force_balance(
        analyzer, problem, state["stress"], integration_order,
    )
    load_error = float(
        np.linalg.norm(F_num - F_ref) / max(np.linalg.norm(F_ref), 1.0e-30)
    )
    disp_error = mesh.error(
        state["displacement"], problem.disp_solution, q=integration_order
    )
    stress_error = mesh.error(
        state["stress"], problem.stress_solution, q=integration_order
    )

    stress_dofs = analyzer.huzhang_space.number_of_global_dofs()
    disp_dofs = analyzer.tensor_space.number_of_global_dofs()

    return {
        "subdivisions": subdivisions,
        "mesh_size": 1.0 / subdivisions,
        "total_dofs": int(stress_dofs + disp_dofs),
        "disp_error": _as_float(disp_error),
        "stress_error": _as_float(stress_error),
        "residual": analyzer.relative_state_residual(),
        "load_error": load_error,
    }


def observed_order(coarse: float, fine: float) -> float | None:
    """网格每层减半, 观测阶即误差比值的以 2 为底对数."""
    if coarse > 0.0 and fine > 0.0:
        return log2(coarse / fine)
    return None


def _order_or_none(rows: list[dict], index: int, key: str) -> str:
    """第 index 行相对上一层的观测收敛阶, 首行或非法时为 ``n/a``."""
    if index == 0:
        return "   n/a"
    value = observed_order(rows[index - 1][key], rows[index][key])
    return "   n/a" if value is None else f"{value:6.2f}"


def report(rows: list[dict]) -> None:
    """打印结果表, 含位移/应力场 L2 误差与收敛阶, 并给出边界合力判据值."""

    header = (
        f"{'nx':>4} {'gdof':>8} {'h':>9} "
        f"{'|u-uh|_0':>11} {'u_order':>8} "
        f"{'|s-sh|_0':>11} {'s_order':>8} "
        f"{'residual':>11} {'load_error':>11}"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows):
        u_order = _order_or_none(rows, index, "disp_error")
        s_order = _order_or_none(rows, index, "stress_error")
        print(
            f"{row['subdivisions']:>4} {row['total_dofs']:>8} "
            f"{row['mesh_size']:>9.4f} "
            f"{row['disp_error']:>11.4e} {u_order:>8} "
            f"{row['stress_error']:>11.4e} {s_order:>8} "
            f"{row['residual']:>11.2e} {row['load_error']:>11.2e}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="胡张混合有限元的 traction 载荷路径工程基准",
    )
    parser.add_argument(
        "--problem", choices=list(PROBLEM_REGISTRY), default="mixed-sinusoidal",
        help="制造解问题 (默认 mixed-sinusoidal)",
    )
    parser.add_argument(
        "--degree", type=int, default=3,
        help="应力空间次数, 二维胡张元要求 p >= 3 才无需低阶稳定化 (默认 3)",
    )
    parser.add_argument(
        "--levels", type=int, default=3,
        help="加密层数, 第 i 层网格为 2^i x 2^i (默认 3)",
    )
    parser.add_argument(
        "--relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭角点松弛 (默认启用)",
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

    bm.set_backend("numpy")

    entry = PROBLEM_REGISTRY[arguments.problem]
    problem, material = create_problem_and_material(entry)
    integration_order = 2 * arguments.degree + 2

    # 结论依赖于哪一份 FEALPy: 官方检出与打了缺陷修复的检出版本号都是 4.0.0,
    # 只有解析路径能区分. 这里用 import_module 而不是模块级 ``import fealpy``:
    # 后者只在这一行用到, 会被 "移除未使用导入" 的工具删掉
    fealpy_root = Path(import_module("fealpy").__file__).resolve().parents[1]
    print(f"FEALPy: {fealpy_root}")
    print(
        f"问题={entry.label}, 网格=triangle-checkerboard, "
        f"平面假设={problem.plane_type}, 空间次数={arguments.degree}, "
        f"角点松弛={arguments.relaxation}, 求解器=scipy"
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
    residual_passed = residual_max <= RESIDUAL_TOLERANCE

    # 合力相对差按应力逼近阶收敛: 最粗层 h=0.5 时逼近误差可达 1e-3, 这是有限元
    # 逼近误差而非路径失效, 所以判据看最细一层的逼近精度, 并以逐层递减作为
    # 收敛趋势 (若 traction 数据被强加错, 数值合力不会随加密逼近解析值)
    finest_load_error = rows[-1]["load_error"]
    load_passed = finest_load_error <= LOAD_TOLERANCE
    decreasing = all(
        coarse["load_error"] > fine["load_error"]
        for coarse, fine in zip(rows[:-1], rows[1:])
    )

    print(
        f"\n真相对残差最大值 = {residual_max:.2e} "
        f"(阈值 {RESIDUAL_TOLERANCE:.0e}) -> "
        f"{'通过' if residual_passed else '未通过'}"
    )
    print(
        f"traction 边界合力相对差 (最细层 nx={rows[-1]['subdivisions']}) "
        f"= {finest_load_error:.2e} "
        f"(阈值 {LOAD_TOLERANCE:.0e}) -> "
        f"{'通过' if load_passed else '未通过'}"
    )
    print(
        f"合力相对差逐层递减 -> {'通过' if decreasing else '未通过'}"
    )

    if arguments.save_vtu:
        finest = rows[-1]
        finest_nx = finest["subdivisions"]
        finest_mesh = create_huzhang_checkerboard_mesh(
            box=problem.domain, nx=finest_nx, ny=finest_nx,
        )
        finest_analyzer = HuZhangMFEMAnalyzer(
            disp_mesh=finest_mesh,
            pde=problem,
            material=material,
            interpolation_scheme=None,
            space_degree=arguments.degree,
            integration_order=integration_order,
            use_relaxation=arguments.relaxation,
            solve_method="scipy",
            topopt_algorithm=None,
        )
        state = finest_analyzer.solve_state(rho_val=None)
        disp = extract_nodal_displacement(
            state["displacement"], finest_analyzer.tensor_space, finest_mesh
        )

        vtu_dir = Path(__file__).resolve().parent / "outputs" / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        vtu_stem = (
            f"{arguments.problem}_p{arguments.degree}_"
            f"tri_{finest_nx}x{finest_nx}"
        )
        vtu_path = str(vtu_dir / vtu_stem)
        export_vtu(finest_mesh, disp, vtu_path)
        print(f"\nVTU 已导出: {vtu_path}.vtu")

    if residual_passed and load_passed and decreasing:
        print(
            "\n结论: SOPTX 的胡张混合有限元 traction 载荷路径在该算例上可用."
        )
        return 0

    print("\n结论: traction 载荷路径存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
