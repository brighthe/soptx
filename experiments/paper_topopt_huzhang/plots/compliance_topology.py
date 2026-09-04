"""固支梁各方法各阶次最终拓扑构型对比 (论文图 5.2).

产物 case ``compliance-topology``: 3 行 2 列, 左半域算得的密度按对称镜像成全域::

  Row 1: (a) LFEM k=2   | (b) HZMFEM k=2
  Row 2: (c) LFEM k=3   | (d) HZMFEM k=3
  Row 3: (e) LFEM k=4   | (f) HZMFEM k=4

输出: outputs/figures/compliance_topology.png, 自动同步至 papers/figures/

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响。
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from ._base import (
    academic_rcparams,
    load_density,
    mirror_half_beam,
    require_run_dir,
    save_figure,
)

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
# 这张图吃哪个算例的哪几次运行, 本就是绘图代码自己的事实, 故写在模块身上而
# 不另立注册表; 两个常量都保持字面量, ast.literal_eval 才读得动。
SOURCE_CASE = "compliance-fixed-fixed-half"
REQUIRED_RUNS = (
    "analyzer-lfem__order-2", "analyzer-huzhang__order-2",
    "analyzer-lfem__order-3", "analyzer-huzhang__order-3",
    "analyzer-lfem__order-4", "analyzer-huzhang__order-4",
)

OUTPUTS = config.OUTPUT_DIR / SOURCE_CASE

# 3 行 2 列: 行是阶次 k=2,3,4, 列是方法 (左 LFEM / 右 HZMFEM). run 目录只在
# REQUIRED_RUNS 里写一遍, 面板按阅读顺序取, 同一个目录名不在本文件出现两处。
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


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 5.2), dpi=300)

    for row_idx, row in enumerate(PANELS):
        for col_idx, (method, k, folder, tag) in enumerate(row):
            ax = axes[row_idx, col_idx]
            run_dir = require_run_dir(OUTPUTS, folder)
            print(f"  {tag} {method} k={k}: {run_dir.relative_to(config.OUTPUT_DIR)}")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("compliance_domain") != "half":
                raise ValueError(
                    f"{run_dir}: summary.json 未明确采用半域柔顺度口径, "
                    "请先核查并迁移旧摘要, 或使用当前 driver 重新运行."
                )
            # 与下方镜像拓扑采用相同的完整结构展示口径, 不回写原始摘要.
            full_compliance = 2.0 * float(summary["compliance"])
            pts, conn, rho = load_density(run_dir / "density_final.vtu")

            pts_full, conn_full, rho_full = mirror_half_beam(pts, conn, rho)
            tri = Triangulation(pts_full[:, 0], pts_full[:, 1], triangles=conn_full)

            # 标准拓扑黑白映射 (rho=1 实体为黑, rho=0 空洞为白)
            ax.tripcolor(tri, rho_full, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0)
            ax.set_aspect("equal")
            ax.set_xlim(0, 160)
            ax.set_ylim(0, 20)

            # 边框美化与标题
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.8)

            ax.set_xticks([])
            ax.set_yticks([])

            method_name = "LFEM" if method == "LFEM" else "HZMFEM"
            ax.set_title(
                f"{tag} {method_name} ($k = {k}$), full $C = {full_compliance:.2f}$",
                fontsize=11, fontweight="bold", pad=4,
            )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04, wspace=0.06, hspace=0.35)

    save_figure(fig, "compliance_topology")
    plt.close(fig)


if __name__ == "__main__":
    main()

