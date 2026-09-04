"""固支梁 k=1 与 k=2 拓扑构型对比 (补充图, 草稿未引用).

产物 case ``compliance-k1-comparison``: 2x2 四格, 上排 k=1 下排 k=2, 左 LFEM 右 HZMFEM,
运行目录见 REQUIRED_RUNS。

输出: outputs/figures/compliance_k1_comparison.png, 自动同步至 papers/figures/
"""

from __future__ import annotations

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
SOURCE_CASE = "compliance-fixed-fixed-half"
REQUIRED_RUNS = (
    "analyzer-lfem__order-1", "analyzer-huzhang__order-1",
    "analyzer-lfem__order-2", "analyzer-huzhang__order-2",
)

OUTPUTS_ROOT = config.OUTPUT_DIR / SOURCE_CASE

# 面板顺序与 REQUIRED_RUNS 逐项对齐; run 目录名只在 REQUIRED_RUNS 里写一遍。
CASE_LABELS = [
    # Row 1: k = 1
    ("LFEM", 1, "(a)", "$C = 30.59$"),
    ("HZMFEM", 1, "(b)", "$C = 43.19$ (Severe Distortion)"),
    # Row 2: k = 2
    ("LFEM", 2, "(c)", "$C = 31.94$"),
    ("HZMFEM", 2, "(d)", "$C = 33.07$ (98.8% Match)"),
]
CASES = [
    (method, k, run, tag, note)
    for (method, k, tag, note), run in zip(CASE_LABELS, REQUIRED_RUNS)
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 3.8), dpi=300)
    axes = axes.flat

    for idx, (method, k, folder, tag, c_note) in enumerate(CASES):
        run_dir = require_run_dir(OUTPUTS_ROOT, folder)
        print(f"  {tag} {method} k={k}: {run_dir.relative_to(config.OUTPUT_DIR)}")
        ax = axes[idx]
        pts, conn, rho = load_density(run_dir / "density_final.vtu")

        pts_full, conn_full, rho_full = mirror_half_beam(pts, conn, rho)
        tri = Triangulation(pts_full[:, 0], pts_full[:, 1], triangles=conn_full)

        # 标准黑白灰度映射 (rho=1 实体为黑, rho=0 空洞为白)
        ax.tripcolor(tri, rho_full, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 20)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.8)

        method_str = "LFEM" if method == "LFEM" else "HZMFEM"
        ax.set_title(f"{tag} {method_str} ($k = {k}$, {c_note})", fontsize=10.5, fontweight="bold", pad=4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.06, hspace=0.35)

    save_figure(fig, "compliance_k1_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
