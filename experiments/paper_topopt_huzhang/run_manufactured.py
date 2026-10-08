"""前向制造解收敛阶验证 (论文 5.1 节 / 表 5.1 与表 5.2).

自包含脚本: 算例参数即下方常量, 与论文 5.1 节逐项对应.

- 表 5.1: k = 3, 4, 原生格式 (k >= GD + 1, 不加稳定化);
- 表 5.2: k = 1, 2, 矩阵跳量稳定化 (alpha = mu / L0^2, 即论文 gamma_0 = 1).

一次运行跑完全部阶次, 整份结果出自同一份代码, 只盖一个溯源戳记; 同时写出表 5.1 /
5.2 的 Markdown 并回显. 结果写入入库的论文证据目录: 数据
``results/manufactured-convergence/manufactured_convergence.json``, 表格
``results/tables/table5_1.md`` / ``table5_2.md`` (与第 5.2 节各表同处); 溯源戳记
如实记录工作区状态, 是否可复现以 ``provenance.reproducible`` 为准.

用法::

    python run_manufactured.py                # 论文口径, 全部阶次
    python run_manufactured.py --degree 2     # 调试单个阶次, 只回显不落盘
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from math import log2
from pathlib import Path
from typing import Any

from soptx.backend import backend_manager as bm

from soptx.fem import HuZhangMFEMAnalyzer
from soptx.mesh import create_huzhang_checkerboard_mesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import MixedBoundarySinusoidalElasticity2D

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = RESULTS_DIR / "manufactured-convergence"
TABLE_DIR = RESULTS_DIR / "tables"
RESULT_FILE = "manufactured_convergence.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 论文 5.1 节的算例设置
LAME_LAMBDA = 1.0
SHEAR_MODULUS = 0.5
SUBDIVISIONS = (4, 8, 16, 32, 64)
USE_RELAXATION = True
SOLVER = "mumps"
# 阶次 -> 稳定化; k >= 3 时分析器走原生鞍点装配, 只能写 none
STABILIZATION = {1: "matrix_jump", 2: "matrix_jump", 3: "none", 4: "none"}
PAPER_TABLES = {
    "table5_1": ((3, 4), "**表 5.1**  高阶 Hu–Zhang 混合有限元 ($k=3,4$) 制造解收敛误差与观测阶"),
    "table5_2": ((1, 2), "**表 5.2**  低阶跳量稳定化 Hu–Zhang 混合有限元 ($k=1,2$) 制造解收敛误差与观测阶"),
}
ERROR_KEYS = ("disp_l2_error", "stress_l2_error", "stress_hdiv_error")


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(bm.to_numpy(value).reshape(-1)[0])


def solve_one_level(degree: int, subdivisions: int) -> dict[str, Any]:
    """在一级网格上求解并计算三项误差.

    Parameters
    ----------
    degree : int
        Hu-Zhang 应力空间次数 k.
    subdivisions : int
        每个坐标方向的等分数 n_x.

    Returns
    -------
    dict[str, Any]
        该级网格的自由度、误差与求解诊断.
    """
    problem = MixedBoundarySinusoidalElasticity2D(
        lame_lambda=LAME_LAMBDA, shear_modulus=SHEAR_MODULUS
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis=problem.plane_type,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        enable_logging=False,
    )
    mesh = create_huzhang_checkerboard_mesh(
        box=problem.domain, nx=subdivisions, ny=subdivisions
    )
    q = 2 * degree + 2
    analyzer = HuZhangMFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=None,
        space_degree=degree,
        integration_order=q,
        use_relaxation=USE_RELAXATION,
        solve_method=SOLVER,
        topopt_algorithm=None,
        stabilization=STABILIZATION[degree],
    )
    state = analyzer.solve_state(rho_val=None)
    sigmah, uh = state["stress"], state["displacement"]

    disp_error = _as_float(mesh.error(uh, problem.disp_solution, q=q))
    stress_error = _as_float(mesh.error(sigmah, problem.stress_solution, q=q))
    div_error = _as_float(
        mesh.error(sigmah.div_value, problem.div_stress_solution, q=q)
    )
    stress_dofs = int(analyzer.huzhang_space.number_of_global_dofs())
    disp_dofs = int(analyzer.tensor_space.number_of_global_dofs())
    return {
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


def add_observed_orders(rows: list[dict[str, Any]]) -> None:
    """按相邻两级误差之比 log2(e_h / e_{h/2}) 原地补上观测阶, 首级记 None."""
    for index, row in enumerate(rows):
        for key in ERROR_KEYS:
            previous = rows[index - 1][key] if index else 0.0
            row[f"{key}_order"] = (
                log2(previous / row[key]) if previous > 0.0 and row[key] > 0.0 else None
            )


def _git_state(path: Path) -> dict[str, Any]:
    """返回 path 所在 Git 工作副本的 revision 与 dirty 状态; 取不到时记 None."""

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments], capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def provenance() -> dict[str, Any]:
    """本次运行的溯源戳记: soptx 的 revision/dirty.

    数值代码 (含自 FEALPy 移植的部分) 全部位于本仓库, 工作区干净才记 ``reproducible = True``.
    """
    soptx = _git_state(REPOSITORY_ROOT)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **soptx,
        "reproducible": soptx["git_dirty"] is False,
    }


def _format_sci(value: float) -> str:
    base, exponent = f"{value:.4e}".split("e")
    return f"${base}\\times10^{{{int(exponent)}}}$"


def table_markdown(results: dict[str, list], degrees: tuple[int, ...], title: str) -> str:
    """把若干阶次的逐级记录排成一张论文收敛表 (Markdown)."""
    header = (
        "| $k$ | $n_x$ | 总自由度 | $\\|\\boldsymbol{u}-\\boldsymbol{u}_h\\|_0$ | 观测阶 "
        "| $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_0$ | 观测阶 "
        "| $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_{H(\\mathrm{div})}$ | 观测阶 |"
    )
    lines = [title, "", header, "|:---:" * 9 + "|"]
    for degree in degrees:
        for index, row in enumerate(results.get(str(degree), [])):
            cells = [f"**{degree}**" if index == 0 else "", str(row["nx"]),
                     f"{row['total_dofs']:,}".replace(",", " ")]
            for key in ERROR_KEYS:
                order = row[f"{key}_order"]
                cells += [_format_sci(row[key]), "—" if order is None else f"{order:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="制造解前向收敛阶验证 (论文表 5.1 / 5.2)")
    parser.add_argument(
        "--degree", type=int, action="append", dest="degrees", choices=sorted(STABILIZATION),
        help="只跑指定阶次 (可重复); 结果只回显, 不落盘",
    )
    args = parser.parse_args(argv)
    degrees = args.degrees or sorted(STABILIZATION)

    results: dict[str, list] = {}
    for degree in degrees:
        print(f"\n>>> k = {degree} ({STABILIZATION[degree]}), n_x = {list(SUBDIVISIONS)}")
        rows = []
        for subdivisions in SUBDIVISIONS:
            row = solve_one_level(degree, subdivisions)
            rows.append(row)
            print(
                f"  [n_x={subdivisions:2d}] DOF={row['total_dofs']:7d} | "
                f"u={row['disp_l2_error']:.4e} | s={row['stress_l2_error']:.4e} | "
                f"s_Hdiv={row['stress_hdiv_error']:.4e}"
            )
        add_observed_orders(rows)
        results[str(degree)] = rows

    tables = {
        name: table_markdown(results, table_degrees, title)
        for name, (table_degrees, title) in PAPER_TABLES.items()
        if set(table_degrees) <= set(degrees)
    }
    for markdown in tables.values():
        print("\n" + markdown)
    if args.degrees:
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": provenance(),
        "settings": {
            "lame_lambda": LAME_LAMBDA,
            "shear_modulus": SHEAR_MODULUS,
            "plane_type": "plane_strain",
            "subdivisions": list(SUBDIVISIONS),
            "use_relaxation": USE_RELAXATION,
            "solver": SOLVER,
            "stabilization": {str(k): v for k, v in STABILIZATION.items()},
        },
        "results": results,
    }
    (DATA_DIR / RESULT_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, markdown in tables.items():
        (TABLE_DIR / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\n[OK] 数据写入 {DATA_DIR}, 表格写入 {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
