# -*- coding: utf-8 -*-
"""悬臂梁应力约束收敛历史与主应力分布 (论文图 5.8).

产物 case ``stress-convergence``: 依赖两类产物 ——
- 主应力分布取 postprocess/ 下的 npz (sig1/sig2/solid_mask), 由 ``compare.py export``
  从 density_final.vtu 冻结求解导出, 列在 REQUIRED_RUNS 里;
- 收敛历史取运行目录的 history.json (scalar_histories.volfrac 与
  .max_relative_violation)。运行目录名由 ``driver._run_label`` 按字段名排序生成, 注册
  缺省值一变目录名就变, 故不写进 REQUIRED_RUNS, 改由 ``metrics.resolve_run_dir`` 按
  注册口径逐项核对 summary 后解析 —— 与 export 认的是同一批目录。

右轴画最大局部约束值 \\(g_{\\max}\\) 而非最大 von Mises 应力: 停止准则判的就是
\\(g_{\\max} \\le \\delta_g\\), 画它才看得出内层是怎么把违反量压到容差线上的; 参考线
\\(\\delta_g\\) 取自 summary 记录的 ``relative_stress_tolerance``, 不另写常数。

配色与排版: 调用 ``_base.academic_rcparams`` 取与 ``compliance_convergence`` (图 5.3)
一致的学术口径 (stix 数学字体、正常负号、网格点线、浅框图例); 曲线色沿用本图早期定稿
的红绿蓝三色, 见下方常量注释. 中文文本一律靠 ``fontproperties=ZH`` 逐处指定, 不依赖
rcParams 的字族.

输出 png/pdf/eps 三种格式至 papers/huzhang-topopt/figures 与本地 outputs/figures.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from ._base import academic_rcparams, chinese_font, save_figure

academic_rcparams()

# 配色沿用本图早期定稿的红绿蓝三色: 左轴体积分数走蓝实线, 右轴违反量走绿虚线 (虚
# 实分工让两条曲线在黑白打印下仍可区分), 屈服面走红。参考线与坐标零线取对应曲线的
# 浅色点线。
COLOR_PRIMARY = "#1f77b4"    # 体积分数 V_f
COLOR_SECONDARY = "#2ca02c"  # 最大局部约束值 g_max
COLOR_YIELD = "#d62728"      # von Mises 许用应力边界
COLOR_REFERENCE = "#aaaaaa"  # 坐标零线
STRESS_CMAP = "jet"          # 主应力点按归一化 von Mises 应力上色
LEGEND_STYLE = dict(fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/lfem_constraint-apparent/fig_data_lfem-k2.npz",
    "postprocess/lfem_constraint-apparent/fig_data_huzhang-k2.npz",
)


def main() -> None:
    BASE = config.OUTPUT_DIR / SOURCE_CASE

    # 1. 加载数据
    from metrics import resolve_run_dir  # 延迟导入: metrics 会拉起 fealpy 求解栈

    def load_history(method, order):
        run_dir = resolve_run_dir(method, order, announce=False)[0]
        with open(run_dir / "history.json") as f:
            h = json.load(f)
        sh = h["scalar_histories"]
        with open(run_dir / "summary.json") as f:
            tol = float(json.load(f)["relative_stress_tolerance"])
        return (np.asarray(sh["volfrac"], dtype=float),
                np.asarray(sh["max_relative_violation"], dtype=float),
                tol)

    lf_vf, lf_g, lf_tol = load_history("lfem", 2)
    hz_vf, hz_g, hz_tol = load_history("huzhang", 2)

    postprocess = BASE / "postprocess" / "lfem_constraint-apparent"
    lf_npz = np.load(postprocess / "fig_data_lfem-k2.npz")
    hz_npz = np.load(postprocess / "fig_data_huzhang-k2.npz")

    # 2. 中文字体设置
    ZH = chinese_font()

    # 3. 绘图辅助
    def plot_convergence(ax, vf, gmax, tol, title):
        it = np.arange(1, len(vf) + 1)
        ln1, = ax.plot(it, vf, color=COLOR_PRIMARY, lw=1.8, label="体积分数 $V_f$")
        ax.set_xlabel("迭代步", fontsize=11, fontproperties=ZH)
        ax.set_ylabel("体积分数 $V_f$", fontsize=11, fontproperties=ZH,
                      color=COLOR_PRIMARY)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylim(0.2, 1.05)

        ax2 = ax.twinx()
        ln2, = ax2.plot(it, gmax, color=COLOR_SECONDARY, lw=1.6, ls="--",
                        label="最大局部约束值 $g_{\\max}$")
        ln3 = ax2.axhline(tol, color=COLOR_SECONDARY, ls=":", lw=1.0, alpha=0.7,
                          label=f"应力容差 $\\delta_g = {tol:g}$")
        ax2.set_yscale("log")
        ax2.set_ylabel("最大局部约束值 $g_{\\max}$", fontsize=11,
                       fontproperties=ZH, color=COLOR_SECONDARY)
        ax2.tick_params(axis="y", labelsize=9)
        ax2.set_ylim(min(tol, float(gmax.min())) * 0.5, float(gmax.max()) * 2.0)

        ax.legend(handles=[ln1, ln2, ln3], loc="upper right", prop=ZH,
                  **LEGEND_STYLE)
        ax.set_title(title, fontsize=11, fontproperties=ZH, y=-0.30)
        ax.grid(True, ls=":", alpha=0.5)

    def plot_yield_surface(ax, npz, title):
        # 只画实体单元: 空洞单元的表观应力被 m_E 压到接近零, 全堆在原点, 既盖不住
        # 信息也会把色标下端占满。
        solid = npz["solid_mask"].astype(bool)
        sig1 = npz["sig1"][solid]
        sig2 = npz["sig2"][solid]
        vm = npz["vm"][solid]

        # von Mises 许用应力边界: sig1^2 - sig1*sig2 + sig2^2 = 1 (参数化椭圆)
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ct, st = np.cos(th), np.sin(th)
        scale = 1.0 / np.sqrt(ct ** 2 - ct * st + st ** 2)
        yield_line, = ax.plot(ct * scale, st * scale, color=COLOR_YIELD, lw=1.6,
                              label="von Mises 许用应力边界")

        # 按归一化 von Mises 应力上色: 屈服面是该量的等值线 1, 点的颜色由冷到暖即
        # 沿径向逼近屈服面, 色标固定在 [0, 1] 两图才可直接对比。
        sc = ax.scatter(sig1, sig2, c=vm, cmap=STRESS_CMAP, vmin=0.0, vmax=1.0,
                        s=5, alpha=0.85, linewidths=0)

        # 图例里的散点代理: 散点本身按色标取色, 直接给它 label 会让图例随机挑一个
        # 颜色, 故另造一个取色标中段的空数据句柄, 只用来说明"这些点是什么"。
        marker_proxy = Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                              markerfacecolor=plt.get_cmap(STRESS_CMAP)(0.75),
                              markeredgecolor="none",
                              label=r"表观应力点（$\overline{\rho}_e>0.5$）")

        ax.axhline(0.0, color=COLOR_REFERENCE, lw=0.8, ls=":")
        ax.axvline(0.0, color=COLOR_REFERENCE, lw=0.8, ls=":")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("归一化第一表观主应力 $\\bar{\\sigma}_1$", fontsize=11, fontproperties=ZH)
        ax.set_ylabel("归一化第二表观主应力 $\\bar{\\sigma}_2$", fontsize=11, fontproperties=ZH)
        ax.tick_params(labelsize=9)
        ax.legend(handles=[yield_line, marker_proxy], loc="upper left", prop=ZH,
                  **LEGEND_STYLE)
        ax.set_title(title, fontsize=11, fontproperties=ZH, y=-0.36)
        ax.grid(True, ls=":", alpha=0.5)

        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0.0, 1.0, 6))
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("表观 von Mises 应力比 $\\sigma_{\\mathrm{vm}} / \\bar{\\sigma}$",
                       fontsize=10, fontproperties=ZH)

    # 4. 绘制 2x2 四宫格
    fig = plt.figure(figsize=(12.0, 9.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.32, hspace=0.55)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_convergence(ax_a, lf_vf, lf_g, lf_tol, "(a) 标准位移法 ($p = 2$) 收敛历史")
    plot_yield_surface(ax_b, lf_npz, "(b) 标准位移法 ($p = 2$) 主应力空间内的单元应力分布")
    plot_convergence(ax_c, hz_vf, hz_g, hz_tol, "(c) 胡张混合法 ($k = 2$) 收敛历史")
    plot_yield_surface(ax_d, hz_npz, "(d) 胡张混合法 ($k = 2$) 主应力空间内的单元应力分布")

    # 5. 输出 png/pdf/eps 三种格式
    save_figure(fig, "stress_convergence", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
