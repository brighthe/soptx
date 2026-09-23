"""不同泊松比下三种离散的拓扑构型对比 (论文图 5.6).

产物 case ``bearing-topologies``: 3x2 六格, 三排依次 LFEM p=1 / LFEM p=2 / HZMFEM k=2,
左列可压缩基准组右列近不可压实验组, 同一离散的两种材料左右并置, 六格的运行目录见
REQUIRED_RUNS。

输出: outputs/figures/bearing_topologies.png, 自动同步至 papers/huzhang-topopt/figures/

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from ._base import (
    academic_rcparams,
    compliance_label,
    load_density,
    resolve_run_dir,
    save_figure,
)

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
# 可压缩基准组与近不可压实验组按 cases.toml 的口径是两条 case (nu 属 A 问题层, 改它
# 等于换题目), 而本图正是要把两组并排, 故 SOURCE_CASE 是元组; REQUIRED_RUNS 的每项
# 相应写全 <case>/<run>, 否则同名的 analyzer-*__order-* 分不清属于哪一组。
SOURCE_CASE = ("bearing-compressible", "bearing-incompressible")
REQUIRED_RUNS = (
    "bearing-compressible/analyzer-lfem__order-1",
    "bearing-incompressible/analyzer-lfem__order-1",
    "bearing-compressible/analyzer-lfem__order-2",
    "bearing-incompressible/analyzer-lfem__order-2",
    "bearing-compressible/analyzer-huzhang__order-2",
    "bearing-incompressible/analyzer-huzhang__order-2",
)

OUTPUTS_ROOT = config.OUTPUT_DIR

# 面板顺序与 REQUIRED_RUNS 逐项对齐, 按 subplots 的行优先展开: 每排一种离散
# (LFEM p=1 / LFEM p=2 / HZMFEM k=2), 排内左可压缩右近不可压。下面 zip 起来即成
# PANELS: run 路径只在 REQUIRED_RUNS 里写一遍, 不在本文件出现第二处, 免得改一处漏一
# 处。阶次标签 LFEM 用 p, HZMFEM 用 k。
PANEL_LABELS = [
    ("(a)", "LFEM", "p=1", r"\nu_0 = 0.30"),
    ("(b)", "LFEM", "p=1", r"\nu_0 = 0.4999"),
    ("(c)", "LFEM", "p=2", r"\nu_0 = 0.30"),
    ("(d)", "LFEM", "p=2", r"\nu_0 = 0.4999"),
    ("(e)", "HZMFEM", "k=2", r"\nu_0 = 0.30"),
    ("(f)", "HZMFEM", "k=2", r"\nu_0 = 0.4999"),
]
PANELS = [
    (*labels, *run.split("/", 1))
    for labels, run in zip(PANEL_LABELS, REQUIRED_RUNS)
]


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 7.4), dpi=300)
    axes = axes.flat

    for idx, (tag, method, order_label, nu_label, case_id, folder_name) in enumerate(PANELS):
        # 本图缺数据时按设计画占位面板, 故用 resolve 而非 require
        run_dir = resolve_run_dir(OUTPUTS_ROOT / case_id, folder_name)
        vtu_path = run_dir / "density_final.vtu" if run_dir else None
        ax = axes[idx]

        if vtu_path is None or not vtu_path.is_file():
            ax.text(
                0.5, 0.5, f"{case_id}/{folder_name}\n(Pending calculation)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray"
            )
            ax.set_title(f"{tag} {method} (${order_label}$, ${nu_label}$)", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        pts, conn, rho = load_density(vtu_path)
        tri = Triangulation(pts[:, 0], pts[:, 1], triangles=conn)

        # 标准拓扑黑白映射 (rho=1 实体为黑, rho=0 空洞为白)
        ax.tripcolor(tri, rho, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 40)

        # 边框美化
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.8)

        c_str = compliance_label(run_dir)
        ax.set_title(f"{tag} {method} (${order_label}$, ${nu_label}${c_str})", fontsize=11, fontweight="bold", pad=5)

    # 紧凑优雅排版 (完全无 Colorbar 遮挡)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.08, hspace=0.28)

    save_figure(fig, "bearing_topologies", formats=("png", "pdf", "eps"))
    plt.close(fig)


if __name__ == "__main__":
    main()

