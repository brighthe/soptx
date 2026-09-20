"""近不可压极限下 HZMFEM 高阶 k=3/4 拓扑构型对比 (补充图, 草稿未引用).

产物 case ``bearing-highorder-topologies``: 左右两格 k=3/4, 运行目录见 REQUIRED_RUNS。

输出: outputs/figures/bearing_highorder_topologies.png, 自动同步至 papers/huzhang-topopt/figures/
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from ._base import (
    chinese_font,
    compliance_label,
    load_density,
    resolve_run_dir,
    save_figure,
)

plt.rcParams["axes.unicode_minus"] = False
ZH = chinese_font()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "bearing-incompressible"
REQUIRED_RUNS = (
    "analyzer-huzhang__order-3",
    "analyzer-huzhang__order-4",
)

OUTPUTS_ROOT = config.OUTPUT_DIR

# 面板顺序与 REQUIRED_RUNS 逐项对齐; run 目录名只在 REQUIRED_RUNS 里写一遍。
PANEL_LABELS = [
    ("HZMFEM", 3, r"$\nu_0 = 0.4999$"),
    ("HZMFEM", 4, r"$\nu_0 = 0.4999$"),
]
PANELS = [
    (*labels, SOURCE_CASE, run) for labels, run in zip(PANEL_LABELS, REQUIRED_RUNS)
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.0), dpi=200)
    axes = axes.flat

    for idx, (method, k, nu_label, case_id, folder_name) in enumerate(PANELS):
        # 本图缺数据时按设计画占位面板, 故用 resolve 而非 require
        run_dir = resolve_run_dir(OUTPUTS_ROOT / case_id, folder_name)
        vtu_path = run_dir / "density_final.vtu" if run_dir else None
        ax = axes[idx]
        if vtu_path is None or not vtu_path.is_file():
            ax.text(0.5, 0.5, f"{case_id}/{folder_name}\n(Waiting for calculation)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(f"({chr(97 + idx)}) {method} $k={k}$ ({nu_label})", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        pts, conn, rho = load_density(vtu_path)
        tri = Triangulation(pts[:, 0], pts[:, 1], triangles=conn)
        ax.tripcolor(tri, rho, shading="flat", cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 40)

        c_str = compliance_label(run_dir)
        ax.set_title(f"({chr(97 + idx)}) {method} $k={k}$, {nu_label}{c_str}", fontsize=12)

    sm = matplotlib.cm.ScalarMappable(cmap="Greys", norm=matplotlib.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=list(axes), orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(r"Density $\rho$")

    fig.suptitle(
        "高阶胡张混合法在近不可压缩极限下的最终拓扑构型对比",
        fontsize=14,
        y=0.98,
        fontproperties=ZH,
    )
    fig.subplots_adjust(left=0.03, right=0.96, top=0.88, bottom=0.05, wspace=0.08)
    save_figure(fig, "bearing_highorder_topologies", formats=("png", "pdf", "eps"))
    plt.close(fig)


if __name__ == "__main__":
    main()
