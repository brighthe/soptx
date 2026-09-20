# -*- coding: utf-8 -*-
"""悬臂梁应力约束的最大表观应力比迭代历史 (论文图 5.9).

产物 case ``stress-max-ratio-history``: 只吃运行目录的 history.json
(``scalar_histories`` 里的 ``volfrac`` 与 ``max_apparent_stress_ratio``), 不依赖
postprocess/ 下的 npz, 故 ``REQUIRED_RUNS`` 为空 —— 运行目录名由 ``driver._run_label``
按字段名排序生成, 注册缺省值一变目录名就变, 改由 ``metrics.resolve_run_dir`` 按注册
口径核对 summary 后解析。

与图 5.8(a)(c) 的分工: 那里画停止准则直接判的 \\(g_{\\max}\\), 看的是内层怎么把违反量
压到容差线上; 这里画物理量 \\(\\sigma_{\\max}/\\bar{\\sigma}\\), 看的是最大应力相对许用值
的逼近过程。两者并非同一条曲线换个刻度: \\(g_e\\) 是表观应力比与 \\(\\eta(\\bar{\\rho}_e)\\)
之差, 控制点落在灰度单元上时分母小、差值小, 对数轴会把百分之几的应力波动放大成几十
倍的峰谷比, 故两个视角都留着。

三块面板取 5.2.3 节比较的三条离散 (LFEM p=2、HZMFEM k=2、HZMFEM k=3), 纵轴刻度统一,
可直接横向比对峰值与贴合许用线的时刻。

配色与排版同 ``stress_convergence`` (图 5.8), 见该模块 docstring。

输出 png/pdf/eps 三种格式至 papers/huzhang-topopt/figures 与本地 outputs/figures.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._base import academic_rcparams, chinese_font, save_figure

academic_rcparams()

COLOR_PRIMARY = "#1f77b4"    # 体积分数 V_f
COLOR_SECONDARY = "#2ca02c"  # 最大表观应力比 sigma_max / sbar
LEGEND_STYLE = dict(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = ()

# 三条离散与面板标题: 顺序即论文里 (a)(b)(c) 的顺序
ARMS = (
    ("lfem", 2, "(a) 标准位移法 ($p = 2$)"),
    ("huzhang", 2, "(b) 胡张混合法 ($k = 2$)"),
    ("huzhang", 3, "(c) 胡张混合法 ($k = 3$)"),
)


def main() -> None:
    ZH = chinese_font()

    # 1. 加载数据
    from metrics import resolve_run_dir  # 延迟导入: metrics 会拉起 fealpy 求解栈

    def load_history(method, order):
        run_dir = resolve_run_dir(method, order, announce=False)[0]
        sh = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))[
            "scalar_histories"
        ]
        return (np.asarray(sh["volfrac"], dtype=float),
                np.asarray(sh["max_apparent_stress_ratio"], dtype=float))

    series = [(load_history(method, order), title) for method, order, title in ARMS]

    # 2. 三块面板共用的右轴范围: 统一刻度才能横向比峰值
    all_ratio = np.concatenate([ratio for (_, ratio), _ in series])
    lo = float(all_ratio.min()) - 0.03
    hi = float(all_ratio.max()) + 0.05

    # 3. 单块面板
    def plot_panel(ax, volfrac, ratio, title):
        it = np.arange(1, len(volfrac) + 1)

        ln1, = ax.plot(it, volfrac, color=COLOR_PRIMARY, lw=1.8,
                       label="体积分数 $V_f$")
        ax.set_xlabel("迭代步", fontsize=11, fontproperties=ZH)
        ax.set_ylabel("体积分数 $V_f$", fontsize=11, fontproperties=ZH,
                      color=COLOR_PRIMARY)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylim(0.2, 1.05)

        ax2 = ax.twinx()
        ln2, = ax2.plot(it, ratio, color=COLOR_SECONDARY, lw=1.6, ls="--",
                        label="最大表观应力比 $\\sigma_{\\max}^{\\mathrm{app}}/\\bar{\\sigma}$")
        # 许用水平取 1: 约束右端 eta(rho) 在实体单元上趋于 1, 该线即可行边界
        ln3 = ax2.axhline(1.0, color=COLOR_SECONDARY, ls=":", lw=1.0, alpha=0.7,
                          label="许用应力水平")
        ax2.set_ylabel("最大表观应力比 $\\sigma_{\\max}^{\\mathrm{app}}/\\bar{\\sigma}$",
                       fontsize=11, fontproperties=ZH, color=COLOR_SECONDARY)
        ax2.tick_params(axis="y", labelsize=9)
        ax2.set_ylim(lo, hi)

        ax.legend(handles=[ln1, ln2, ln3], loc="upper right", prop=ZH,
                  **LEGEND_STYLE)
        ax.set_title(title, fontsize=11, fontproperties=ZH, y=-0.26)
        ax.grid(True, ls=":", alpha=0.5)

    # 4. 绘制 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8), dpi=300)
    for ax, ((volfrac, ratio), title) in zip(axes, series):
        plot_panel(ax, volfrac, ratio, title)
    fig.subplots_adjust(wspace=0.45, bottom=0.24)

    # 5. 输出 png/pdf/eps 三种格式
    save_figure(fig, "stress_max_ratio_history", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
