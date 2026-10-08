"""固支梁各方法各阶次最终拓扑构型对比 (论文图 5.2).

产物 case ``compliance-topology``: 3 行 2 列, 左半域算得的密度按对称镜像成全域::

  Row 1: (a) LFEM p=2   | (b) HZMFEM k=2
  Row 2: (c) LFEM p=3   | (d) HZMFEM k=3
  Row 3: (e) LFEM p=4   | (f) HZMFEM k=4

输出: results/figures/compliance_topology.png, 自动同步至 papers/huzhang-topopt/figures/

2026-09-28 起按版心尺寸出图: 图宽取 CICP 版心 150 mm (5.9 in), 论文里以
``width=\\textwidth`` 原尺寸嵌入, 面板标题字号即纸面字号; 高度按 3 行 160x20 面板加
3 行标题算出, 不再留大片空白. 面板标题只写编号、方法与阶次, 柔顺度数值由论文表给出.
密度场是分片常数, 矢量输出在多数阅读器里会显出三角形接缝, 故只出 600 dpi 的 PNG.

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from ._base import (
    load_density,
    mirror_half_beam,
    paper_rcparams,
    require_run_dir,
    save_figure,
)

paper_rcparams()

# ---- 自描述元数据: plot.py 用 ast 静态解析读走, 不 import 本模块 ----
# 这张图吃哪个算例的哪几次运行, 本就是绘图代码自己的事实, 故写在模块身上而
# 不另立注册表; 两个常量都保持字面量, ast.literal_eval 才读得动.
SOURCE_CASE = "compliance-fixed-fixed-half"
REQUIRED_RUNS = (
    "analyzer-lfem__order-2", "analyzer-huzhang__order-2",
    "analyzer-lfem__order-3", "analyzer-huzhang__order-3",
    "analyzer-lfem__order-4", "analyzer-huzhang__order-4",
)

OUTPUTS = config.OUTPUT_DIR / SOURCE_CASE

# 3 行 2 列: 行是阶次 k=2,3,4, 列是方法 (左 LFEM / 右 HZMFEM). run 目录只在
# REQUIRED_RUNS 里写一遍, 面板按阅读顺序取, 同一个目录名不在本文件出现两处.
PANELS = [
    [
        (
            method,
            order,
            REQUIRED_RUNS[2 * row + column],
            f"({'abcdef'[2 * row + column]})",
        )
        for column, method in enumerate(("LFEM", "HZMFEM"))
    ]
    for row, order in enumerate((2, 3, 4))
]

# 版心宽 150 mm; 高度 = 3 行面板 (每行约 0.36 in) + 3 行 9 pt 标题 + 行距.
FIG_SIZE_IN = (5.9, 1.85)
TITLE_PT = 9.0
DPI = 600


def main():
    fig, axes = plt.subplots(3, 2, figsize=FIG_SIZE_IN, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.06)

    for row_idx, row in enumerate(PANELS):
        for col_idx, (method, k, folder, tag) in enumerate(row):
            ax = axes[row_idx, col_idx]
            run_dir = require_run_dir(OUTPUTS, folder)
            print(f"  {tag} {method} k={k}: {run_dir.relative_to(config.OUTPUT_DIR)}")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("compliance_domain") != "half":
                raise ValueError(
                    f"{run_dir}: summary.json 未明确采用半域柔顺度口径, "
                    "请先核查并迁移旧摘要, 或用 run_fixed_fixed.py 重新运行."
                )
            pts, conn, rho = load_density(run_dir / "density_final.vtu")

            pts_full, conn_full, rho_full = mirror_half_beam(pts, conn, rho)
            tri = Triangulation(pts_full[:, 0], pts_full[:, 1], triangles=conn_full)

            # 标准拓扑黑白映射 (rho=1 实体为黑, rho=0 空洞为白)
            ax.tripcolor(tri, rho_full, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0)
            ax.set_aspect("equal")
            ax.set_xlim(0, 160)
            ax.set_ylim(0, 20)

            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.6)

            ax.set_xticks([])
            ax.set_yticks([])

            # 阶次记号随方法走: LFEM 的 p 是位移阶, HZMFEM 的 k 是应力阶 (与题注一致).
            symbol = "p" if method == "LFEM" else "k"
            ax.set_title(f"{tag} {method}, ${symbol} = {k}$", fontsize=TITLE_PT, pad=2.5)

    save_figure(fig, "compliance_topology", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
