# -*- coding: utf-8 -*-
"""悬臂梁应力约束收敛历史与主应力分布 (论文图 5.8).

产物 case ``stress-convergence``: 依赖两类产物, 都列在 REQUIRED_RUNS 里 ——
- 收敛历史取两个运行目录的 history.json (scalar_histories.volfrac 与 .max_von_mises);
- 主应力分布取 postprocess/ 下的 npz (sig1/sig2/solid_mask), 由 ``compare.py export``
  从 density_final.vtu 冻结求解导出。

输出 png/pdf/eps 三种格式至 papers/figures 与本地 outputs/figures.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ._base import chinese_font, save_figure

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "analyzer-lfem__order-2",
    "analyzer-huzhang__order-2",
    "postprocess/fig_data_lfem-k2.npz",
    "postprocess/fig_data_huzhang-k2.npz",
)


def main() -> None:
    BASE = config.OUTPUT_DIR / SOURCE_CASE

    # 1. 加载数据
    def load_history(method, order):
        run_dir = BASE / f"analyzer-{method}__order-{order}"
        with open(run_dir / "history.json") as f:
            h = json.load(f)
        sh = h["scalar_histories"]
        return np.asarray(sh["volfrac"], dtype=float), np.asarray(sh["max_von_mises"], dtype=float)

    lf_vf, lf_vm = load_history("lfem", 2)
    hz_vf, hz_vm = load_history("huzhang", 2)

    lf_npz = np.load(BASE / "postprocess" / "fig_data_lfem-k2.npz")
    hz_npz = np.load(BASE / "postprocess" / "fig_data_huzhang-k2.npz")

    # 2. 中文字体设置
    ZH = chinese_font()

    # 3. 绘图辅助
    def plot_convergence(ax, vf, vm, title):
        it = np.arange(1, len(vf) + 1)
        ln1, = ax.plot(it, vf, color="tab:blue", lw=1.6, label="体积分数 $V_f$")
        ax.set_xlabel("迭代步", fontsize=10, fontproperties=ZH)
        ax.set_ylabel("体积分数 $V_f$", fontsize=10, fontproperties=ZH, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0.2, 1.05)

        ax2 = ax.twinx()
        ln2, = ax2.plot(it, vm, color="tab:red", lw=1.6,
                        label="最大归一化 von Mises 应力")
        ax2.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.7)
        ax2.set_ylabel("最大归一化 von Mises 应力", fontsize=10, fontproperties=ZH, color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=9)
        ax2.set_ylim(0.6, max(2.0, float(vm.max()) * 1.05))

        ax.legend(handles=[ln1, ln2], loc="upper right", fontsize=9, prop=ZH, framealpha=0.9)
        ax.set_title(title, fontsize=11, fontproperties=ZH, y=-0.30)
        ax.grid(alpha=0.25, lw=0.5)

    def plot_yield_surface(ax, npz, title):
        sig1 = npz["sig1"]
        sig2 = npz["sig2"]
        solid = npz["solid_mask"].astype(bool)

        # von Mises 屈服面: sig1^2 - sig1*sig2 + sig2^2 = 1 (参数化椭圆)
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ct, st = np.cos(th), np.sin(th)
        scale = 1.0 / np.sqrt(ct ** 2 - ct * st + st ** 2)
        ax.plot(ct * scale, st * scale, color="tab:red", lw=1.6, label="von Mises 屈服面")

        ax.scatter(sig1[~solid], sig2[~solid], s=4, c="0.75", alpha=0.5, linewidths=0,
                   label="空洞单元 ($\\rho \\leq 0.5$)")
        ax.scatter(sig1[solid], sig2[solid], s=5, c="tab:blue", alpha=0.6, linewidths=0,
                   label="实体单元 ($\\rho > 0.5$)")

        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xlabel("归一化第一主应力 $\\bar{\\sigma}_1$", fontsize=10, fontproperties=ZH)
        ax.set_ylabel("归一化第二主应力 $\\bar{\\sigma}_2$", fontsize=10, fontproperties=ZH)
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper left", fontsize=8, prop=ZH, framealpha=0.9)
        ax.set_title(title, fontsize=11, fontproperties=ZH, y=-0.36)
        ax.grid(alpha=0.25, lw=0.5)

    # 4. 绘制 2x2 四宫格
    fig = plt.figure(figsize=(12.0, 9.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.32, hspace=0.55)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_convergence(ax_a, lf_vf, lf_vm, "(a) 标准位移法 ($k = 2$) 收敛历史")
    plot_yield_surface(ax_b, lf_npz, "(b) 标准位移法 ($k = 2$) 主应力分布")
    plot_convergence(ax_c, hz_vf, hz_vm, "(c) 胡张混合法 ($k = 2$) 收敛历史")
    plot_yield_surface(ax_d, hz_npz, "(d) 胡张混合法 ($k = 2$) 主应力分布")

    # 5. 输出 png/pdf/eps 三种格式
    save_figure(fig, "stress_convergence", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
