"""前向制造解收敛阶验证 (论文 5.1 节 / 表 5.1 与表 5.2).

自包含脚本: 算例参数即下方常量, 与论文 5.1 节逐项对应.

- 表 5.1: k = 3, 4, 原生格式 (k >= GD + 1, 不加稳定化);
- 表 5.2: k = 1, 2, 矩阵跳量稳定化 (alpha = mu / L0^2, 即论文 gamma_0 = 1).

一次运行跑完全部阶次, 整份结果出自同一份代码, 只盖一个溯源戳记; 同时写出表 5.1 /
5.2 的 Markdown 并回显. 结果写入运行目录 ``outputs/manufactured-convergence/<label>/``
(不入库, 同名目录直接覆盖): 数据 ``manufactured_convergence.json``, 表格 ``table5_1.md`` /
``table5_2.md`` (只在对应阶次都跑了时写出); 确认后经 ``sync_results.py`` 固化到入库的
``results/``.

    python run_manufactured.py                       # 论文口径 (不松弛), 全部阶次, 目录 relaxation-off
    python run_manufactured.py --relaxation          # 开启角点松弛的对照, 目录 relaxation-on
    python run_manufactured.py --degree 2            # 只跑单个阶次, 同样落盘
"""

from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path
from typing import Literal

from soptx.backend import backend_manager as bm
from soptx.fem import HuZhangMFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import TriangleMesh
from soptx.problems import MixedBoundarySinusoidalElasticity2D

from analysis import provenance

CASE_ID = "manufactured-convergence"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / CASE_ID
RESULT_FILE = "manufactured_convergence.json"

# 论文 5.1 节的算例设置
LAME_LAMBDA = 1.0
SHEAR_MODULUS = 0.5
SUBDIVISIONS = (4, 8, 16, 32, 64)
# 角点松弛缺省值, 命令行 --relaxation / --no-relaxation 可覆盖. 论文口径不松弛: 唯一的牵引--牵引
# 角点 (1, 1) 处两边牵引来自同一光滑对称应力, 满足 Hu--Ma (2021) 4.1 节的一致性条件
USE_RELAXATION = False
SOLVER = "mumps"
# 阶次 -> 稳定化; k >= 3 时分析器走原生鞍点装配, 只能写 none
STABILIZATION: dict[int, Literal["none", "matrix_jump"]] = {
    1: "matrix_jump", 2: "matrix_jump", 3: "none", 4: "none"}
PAPER_TABLES = {
    "table5_1": ((3, 4), "**表 5.1**  高阶 Hu–Zhang 混合有限元 ($k=3,4$) 制造解收敛误差与观测阶"),
    "table5_2": ((1, 2), "**表 5.2**  低阶跳量稳定化 Hu–Zhang 混合有限元 ($k=1,2$) 制造解收敛误差与观测阶"),
}
ERROR_KEYS = ("disp_l2_error", "stress_l2_error", "stress_hdiv_error")
# 运行目录名模板 (填 relaxation)
RUN_LABEL = "relaxation-{relaxation}"
TABLE_HEADER = (
    "| $k$ | $n_x$ | 总自由度 | $\\|\\boldsymbol{u}-\\boldsymbol{u}_h\\|_0$ | 观测阶 "
    "| $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_0$ | 观测阶 "
    "| $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_{H(\\mathrm{div})}$ | 观测阶 |"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析要跑的阶次与角点松弛开关, 并据此确定运行目录; 其余设置均为模块顶部常量."""
    parser = argparse.ArgumentParser(description="制造解前向收敛阶验证 (论文表 5.1 / 5.2)")
    parser.add_argument(
        "--degree", type=int, action="append", dest="degrees", choices=sorted(STABILIZATION),
        help="只跑指定阶次 (可重复); 缺省全部阶次",
    )
    parser.add_argument(
        "--relaxation", action=argparse.BooleanOptionalAction, default=USE_RELAXATION,
        help=f"Hu--Zhang 角点松弛开关 (缺省 {USE_RELAXATION})",
    )
    args = parser.parse_args(argv)
    args.degrees = args.degrees or sorted(STABILIZATION)
    args.label = RUN_LABEL.format(relaxation="on" if args.relaxation else "off")
    args.output = OUTPUT_DIR / args.label
    return args


def main(argv: list[str] | None = None) -> int:
    """逐阶次逐级网格求解 -> 三项误差与观测阶 -> 论文表格 -> 落盘."""
    args = parse_args(argv)
    bm.set_backend("numpy")

    # 记下当前代码的 git 版本, 用于追溯结果由哪版代码算出
    stamp = provenance.run_stamp()
    degrees, relaxation, label, output = args.degrees, args.relaxation, args.label, args.output
    print(f"[run] {label}: 角点松弛={relaxation}, 结果 -> {output}", flush=True)

    # 1. 逐阶次逐级网格求解, 记录自由度、三项误差与求解诊断
    results: dict[str, list] = {}
    for degree in degrees:
        print(f"\n>>> k = {degree} ({STABILIZATION[degree]}), n_x = {list(SUBDIVISIONS)}")
        rows = []
        for subdivisions in SUBDIVISIONS:
            problem = MixedBoundarySinusoidalElasticity2D(
                lame_lambda=LAME_LAMBDA, shear_modulus=SHEAR_MODULUS)
            material = IsotropicLinearElasticMaterial(
                hypothesis=problem.plane_type, lame_lambda=problem.lam, shear_modulus=problem.mu)
            mesh = TriangleMesh.from_checkerboard_box(
                box=problem.domain, nx=subdivisions, ny=subdivisions)
            q = 2 * degree + 2
            analyzer = HuZhangMFEMAnalyzer(
                disp_mesh=mesh, pde=problem, material=material, interpolation_scheme=None,
                space_degree=degree, integration_order=q, use_relaxation=relaxation,
                solve_method=SOLVER, topopt_algorithm=None, stabilization=STABILIZATION[degree])
            state = analyzer.solve_state(rho_val=None)
            sigmah, uh = state["stress"], state["displacement"]

            # mesh.error 可能返回零维张量, 统一取成 float
            as_float = lambda value: float(bm.to_numpy(value).reshape(-1)[0])  # noqa: E731
            disp_error = as_float(mesh.error(uh, problem.disp_solution, q=q))
            stress_error = as_float(mesh.error(sigmah, problem.stress_solution, q=q))
            div_error = as_float(mesh.error(sigmah.div_value, problem.div_stress_solution, q=q))
            stress_dofs = int(analyzer.huzhang_space.number_of_global_dofs())
            disp_dofs = int(analyzer.tensor_space.number_of_global_dofs())
            row = {
                "nx": subdivisions,
                "total_dofs": stress_dofs + disp_dofs,
                "stress_dofs": stress_dofs,
                "disp_dofs": disp_dofs,
                "disp_l2_error": disp_error,
                "stress_l2_error": stress_error,
                "div_stress_l2_error": div_error,
                "stress_hdiv_error": (stress_error**2 + div_error**2) ** 0.5,
                "relative_residual": analyzer.relative_state_residual(),
                "symmetry_error": analyzer.state_matrix_symmetry_error(),
            }
            # 观测阶按相邻两级误差之比 log2(e_h / e_{h/2}), 首级记 None
            for key in ERROR_KEYS:
                previous = rows[-1][key] if rows else 0.0
                row[f"{key}_order"] = (
                    log2(previous / row[key]) if previous > 0.0 and row[key] > 0.0 else None)
            rows.append(row)
            fmt = lambda key: (f"{row[key]:.4e}" + ("" if row[f"{key}_order"] is None  # noqa: E731
                                                     else f" ({row[f'{key}_order']:.2f})"))
            print(f"  [n_x={subdivisions:2d}] DOF={row['total_dofs']:7d} | "
                  f"u={fmt('disp_l2_error')} | s={fmt('stress_l2_error')} | "
                  f"s_Hdiv={fmt('stress_hdiv_error')}")
        results[str(degree)] = rows

    # 2. 论文表格 (Markdown): 只排阶次都跑齐的表
    tables: dict[str, str] = {}
    for name, (table_degrees, title) in PAPER_TABLES.items():
        if not set(table_degrees) <= set(degrees):
            continue
        lines = [title, "", TABLE_HEADER, "|:---:" * 9 + "|"]
        for degree in table_degrees:
            for index, row in enumerate(results[str(degree)]):
                cells = [f"**{degree}**" if index == 0 else "", str(row["nx"]),
                         f"{row['total_dofs']:,}".replace(",", " ")]
                for key in ERROR_KEYS:
                    base, exponent = f"{row[key]:.4e}".split("e")
                    order = row[f"{key}_order"]
                    cells += [f"${base}\\times10^{{{int(exponent)}}}$",
                              "—" if order is None else f"{order:.2f}"]
                lines.append("| " + " | ".join(cells) + " |")
        tables[name] = "\n".join(lines)
        print("\n" + tables[name])

    # 3. 落盘: 数据 (含设置与溯源) 与表格写运行目录
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": stamp,
        "settings": {
            "lame_lambda": LAME_LAMBDA,
            "shear_modulus": SHEAR_MODULUS,
            "plane_type": "plane_strain",
            "subdivisions": list(SUBDIVISIONS),
            "use_relaxation": relaxation,
            "solver": SOLVER,
            "stabilization": {str(k): v for k, v in STABILIZATION.items()},
            "degrees": degrees,
        },
        "results": results,
    }
    (output / RESULT_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, markdown in tables.items():
        (output / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\n[OK] 结果写入 {output} (数据 {RESULT_FILE}, 表格 {', '.join(tables) or '无'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
