"""论文表 5.1 / 5.2 的生成.

从制造解收敛产物 ``outputs/manufactured_convergence/summary.json`` 重算两张收敛表,
由 ``compare.py table`` 调用, 写入 ``outputs/tables/`` 并回显 Markdown.

命名沿用 ``experiments/matrix_free_capability/report.py``: 本层只出论文表报, 与
``metrics.py`` (重分析出来的校验数字) 并列. 原为 make_tables.py, 2026-09-01 并入;
成图底座原也在本文件前半段, 2026-09-03 下沉至 ``plots/_base.py``.
"""

from __future__ import annotations

import json

import config

SUMMARY_PATH = config.OUTPUT_DIR / "manufactured_convergence" / "summary.json"
TABLES_DIR = config.OUTPUT_DIR / "tables"


def format_order(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}"


def format_sci(val: float) -> str:
    s = f"{val:.4e}"
    base, exp = s.split("e")
    exp_int = int(exp)
    return f"${base}\\times10^{{{exp_int}}}$"


def generate_table_5_1_md(data: dict) -> str:
    """生成表 5.1: 高阶 Hu-Zhang 混合有限元 (k=3,4) 制造解收敛误差与观测阶."""
    lines = [
        "**表 5.1**  高阶 Hu–Zhang 混合有限元 ($k=3,4$) 制造解收敛误差与观测阶",
        "",
        "| $k$ | $nx$ | 全局 DOF | $h$ | $\\|\\boldsymbol{u}-\\boldsymbol{u}_h\\|_0$ | 观测阶 | $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_0$ | 观测阶 | $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_{H(\\mathrm{div})}$ | 观测阶 |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for deg in ["3", "4"]:
        if deg not in data:
            continue
        rows = data[deg]
        for i, r in enumerate(rows):
            k_str = f"**{deg}**" if i == 0 else ""
            dof_str = f"{r['total_dofs']:,}".replace(",", " ")
            u_err = format_sci(r["disp_l2_error"])
            u_ord = format_order(r["disp_l2_error_order"])
            s_err = format_sci(r["stress_l2_error"])
            s_ord = format_order(r["stress_l2_error_order"])
            h_err = format_sci(r["stress_hdiv_error"])
            h_ord = format_order(r["stress_hdiv_error_order"])
            lines.append(
                f"| {k_str} | {r['nx']} | {dof_str} | {r['mesh_size']:.4f} | {u_err} | {u_ord} | {s_err} | {s_ord} | {h_err} | {h_ord} |"
            )
    return "\n".join(lines)


def generate_table_5_2_md(data: dict) -> str:
    """生成表 5.2: 低阶跳量稳定化 Hu-Zhang 混合有限元 (k=1,2) 制造解收敛误差与观测阶."""
    lines = [
        "**表 5.2**  低阶跳量稳定化 Hu–Zhang 混合有限元 ($k=1,2$) 制造解收敛误差与观测阶",
        "",
        "| $k$ | $nx$ | 全局 DOF | $h$ | $\\|\\boldsymbol{u}-\\boldsymbol{u}_h\\|_0$ | 观测阶 | $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_0$ | 观测阶 | $\\|\\boldsymbol{\\sigma}-\\boldsymbol{\\sigma}_h\\|_{H(\\mathrm{div})}$ | 观测阶 |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for deg in ["1", "2"]:
        if deg not in data:
            continue
        rows = data[deg]
        for i, r in enumerate(rows):
            k_str = f"**{deg}**" if i == 0 else ""
            dof_str = f"{r['total_dofs']:,}".replace(",", " ")
            u_err = format_sci(r["disp_l2_error"])
            u_ord = format_order(r["disp_l2_error_order"])
            s_err = format_sci(r["stress_l2_error"])
            s_ord = format_order(r["stress_l2_error_order"])
            h_err = format_sci(r["stress_hdiv_error"])
            h_ord = format_order(r["stress_hdiv_error_order"])
            lines.append(
                f"| {k_str} | {r['nx']} | {dof_str} | {r['mesh_size']:.4f} | {u_err} | {u_ord} | {s_err} | {s_ord} | {h_err} | {h_ord} |"
            )
    return "\n".join(lines)


def main() -> None:
    if not SUMMARY_PATH.exists():
        print(f"[Error] 数据文件未找到: {SUMMARY_PATH}")
        print("请先运行: python run.py --case manufactured-native")
        return

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    t51_md = generate_table_5_1_md(data)
    t52_md = generate_table_5_2_md(data)

    (TABLES_DIR / "table5_1.md").write_text(t51_md, encoding="utf-8")
    (TABLES_DIR / "table5_2.md").write_text(t52_md, encoding="utf-8")

    print("================ 表 5.1 (Markdown) ================")
    print(t51_md)
    print("\n================ 表 5.2 (Markdown) ================")
    print(t52_md)
    print(f"\n[OK] 表格文件已写入: {TABLES_DIR}")
