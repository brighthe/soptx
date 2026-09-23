"""全域实体时柔顺度相对偏差随泊松比的变化 (存档; 论文图 5.5 已改用 bearing_solid_h_convergence 的 h 收敛图).

作废原因: 参考值取 120x40 上的 HZMFEM $k=4$, 其自身仍偏高 0.3% 至 0.4%, 与被评估离散的偏差同量级;
固定网格扫 $\\nu$ 也区分不了 "$p=2$ 不闭锁" 与 "闭锁被细网格掩盖"。模块保留可复现旧图。

产物 case ``bearing-solid-locking``: 单幅折线图, 横轴按 $0.5 - \\nu_0$ 取对数刻度但刻度
标签直接写 $\\nu_0$ 六档 (0.3 … 0.49999), 越靠右越接近不可压缩; 纵轴是各离散在全域实体
($\\rho_e \\equiv 1$, 无材料插值) 上的柔顺度相对 HZMFEM $k=4$ 的偏差 (%)。三条曲线为
LFEM $p=1$ / LFEM $p=2$ / HZMFEM $k=2$ (跳量稳定化), 与论文表 5.5 的参赛离散同一批;
LFEM $p=4$ 只做参考, 不画。

数据不重解方程, 只读 ``bearing_reanalysis.py`` 落盘的
``outputs/bearing-incompressible/postprocess/frozen_reanalysis.json`` 里的 ``solid_sweep``
块; 缺该块即报错, 请先运行 ``compare.py bearing-reanalysis``。

输出: outputs/figures/bearing_solid_locking.{png,pdf,eps}, 自动同步至
papers/huzhang-topopt/figures/。

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响。
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from ._base import academic_rcparams, chinese_font, save_figure

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "bearing-incompressible"
REQUIRED_RUNS = ("postprocess/frozen_reanalysis.json",)

# 与 bearing_reanalysis.DESIGNS / REFERENCE_LABEL 同口径; 不 import 那个模块, 它在模块级
# 引 pipeline 与 FEALPy, 画图不该为此把求解栈拉起来。json 里缺任一键即报错。
REFERENCE_LABEL = "huzhang-4"
SERIES = (
    # (json 键, 图例, 颜色, 线型, 标记)
    ("lfem-1", "LFEM $p=1$", "#d62728", "-", "o"),
    ("lfem-2", "LFEM $p=2$", "#1f77b4", "--", "s"),
    ("huzhang-2", "HZMFEM $k=2$", "#2ca02c", "-.", "^"),
)


def load_solid_sweep() -> list[dict]:
    path = config.OUTPUT_DIR / SOURCE_CASE / REQUIRED_RUNS[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("solid_sweep")
    if not block or block.get("design") != "solid":
        raise RuntimeError(f"{path} 无 solid_sweep 块; 请先运行 compare.py bearing-reanalysis")
    rows = block["rows"]
    needed = {REFERENCE_LABEL, *(key for key, *_ in SERIES)}
    for row in rows:
        missing = needed - set(row["compliance"])
        if missing:
            raise RuntimeError(f"nu={row['poisson_ratio']} 缺离散 {sorted(missing)}")
    return rows


def main() -> None:
    rows = load_solid_sweep()
    nu = np.array([row["poisson_ratio"] for row in rows], dtype=float)
    order = np.argsort(nu)  # 升序: 0.3 在前, 0.49999 在后
    nu = nu[order]
    rows = [rows[i] for i in order]
    x = 0.5 - nu  # 对数横轴的真实坐标

    ZH = chinese_font()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    ax.axhline(0.0, color="#888888", lw=0.9, ls=":", zorder=1)
    for key, label, color, ls, marker in SERIES:
        dev = np.array([
            100.0 * (row["compliance"][key] / row["compliance"][REFERENCE_LABEL] - 1.0)
            for row in rows
        ])
        ax.plot(x, dev, color=color, lw=1.8, ls=ls, marker=marker, ms=5.5,
                mec="white", mew=0.7, label=label, zorder=3)
        # 只给自锁的那条标端点数值: 另两条全程贴零轴, 数值写在正文
        if key == "lfem-1":
            # 首点标在点的右下方, 末点标在点的左侧, 两处都落在坐标框内
            ax.annotate(f"{dev[0]:+.1f}%", (x[0], dev[0]), textcoords="offset points",
                        xytext=(8, -13), ha="left", fontsize=9, color=color)
            ax.annotate(f"{dev[-1]:+.1f}%", (x[-1], dev[-1]), textcoords="offset points",
                        xytext=(-9, -4), ha="right", fontsize=9, color=color)

    ax.set_xscale("log")
    ax.set_xlim(x.max() * 1.8, x.min() / 1.8)  # 反转: 左 0.3, 右 0.49999, 两端留白
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in nu], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("泊松比 $\\nu_0$", fontsize=11, fontproperties=ZH)
    ax.set_ylabel("柔顺度相对偏差 (%)", fontsize=11, fontproperties=ZH)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(-45.0, 5.0)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.14)
    save_figure(fig, "bearing_solid_locking", formats=("png", "pdf", "eps"))
    plt.close(fig)


if __name__ == "__main__":
    main()
