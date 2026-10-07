"""不同泊松比下三种离散的拓扑构型对比 (论文图 5.6).

产物 case ``bearing-topologies``: 3x2 六格, 三排依次 LFEM p=1 / LFEM p=2 / HZMFEM k=2,
左列可压缩基准组右列近不可压实验组, 同一离散的两种材料左右并置, 六格的运行目录见
REQUIRED_RUNS.

输出: results/figures/bearing_topologies.png, 自动同步至 papers/huzhang-topopt/figures/

2026-09-28 起按版心尺寸出图: 图宽取 CICP 版心 150 mm (5.9 in), 论文里以 ``width=\\textwidth``
原尺寸嵌入, 面板标题字号即纸面字号; 高度按 3 行 120x40 面板加 3 行标题算出. 面板标题只写
编号、方法、阶次与泊松比, 柔顺度数值由论文表 (优化所得柔顺度列) 给出. 密度场是分片常数,
矢量输出会显出三角形接缝, 故只出 600 dpi 的 PNG.

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from ._base import (
    load_density,
    paper_rcparams,
    resolve_run_dir,
    save_figure,
)

paper_rcparams()

# 版心宽 150 mm; 高度 = 3 行面板 (每行约 0.95 in) + 3 行 9 pt 标题 + 行距.
FIG_SIZE_IN = (5.9, 3.55)
TITLE_PT = 9.0
DPI = 600

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
# 两组材料是同一算例 results/bearing/ 下的两个子目录 nu-0.3 / nu-0.4999; REQUIRED_RUNS 的
# 每项写全 <组>/<run>, 否则同名的 analyzer-*__order-* 分不清属于哪一组.
SOURCE_CASE = "bearing"
REQUIRED_RUNS = (
    "nu-0.3/analyzer-lfem__order-1",
    "nu-0.4999/analyzer-lfem__order-1",
    "nu-0.3/analyzer-lfem__order-2",
    "nu-0.4999/analyzer-lfem__order-2",
    "nu-0.3/analyzer-huzhang__order-2",
    "nu-0.4999/analyzer-huzhang__order-2",
)

OUTPUTS_ROOT = config.OUTPUT_DIR / SOURCE_CASE

# 面板顺序与 REQUIRED_RUNS 逐项对齐, 按 subplots 的行优先展开: 每排一种离散
# (LFEM p=1 / LFEM p=2 / HZMFEM k=2), 排内左可压缩右近不可压. 下面 zip 起来即成
# PANELS: run 路径只在 REQUIRED_RUNS 里写一遍, 不在本文件出现第二处, 免得改一处漏一
# 处. 阶次标签 LFEM 用 p, HZMFEM 用 k.
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
    fig, axes = plt.subplots(3, 2, figsize=FIG_SIZE_IN, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.06)
    axes = axes.flat

    for idx, (tag, method, order_label, nu_label, group, folder_name) in enumerate(PANELS):
        # 本图缺数据时按设计画占位面板, 故用 resolve 而非 require
        run_dir = resolve_run_dir(OUTPUTS_ROOT / group, folder_name)
        vtu_path = run_dir / "density_final.vtu" if run_dir else None
        ax = axes[idx]

        if vtu_path is None or not vtu_path.is_file():
            ax.text(
                0.5, 0.5, f"{group}/{folder_name}\n(Pending calculation)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray"
            )
            ax.set_title(f"{tag} {method}, ${order_label}$, ${nu_label}$", fontsize=TITLE_PT, pad=2.5)
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
            spine.set_linewidth(0.6)

        ax.set_title(f"{tag} {method}, ${order_label}$, ${nu_label}$", fontsize=TITLE_PT, pad=2.5)

    save_figure(fig, "bearing_topologies", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()

