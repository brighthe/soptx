# -*- coding: utf-8 -*-
"""三次离散下悬臂梁应力约束优化的收敛历史与主应力分布 (论文图 5.9).

产物 case ``stress-cubic-convergence``: 与图 5.8 同构, 2x2 四宫格 ——
- (a)(c) 收敛历史: 只读两次优化运行目录下的 history.json 与 summary.json, 不重解方程。
  运行目录按 driver 的 _run_label 命名, 取 pad 1.5 mm 且按判据集合验收 (solid_thr-0.5)
  的两组; 本模块启动时核对 summary 记录了 ``acceptance_solid_threshold``, 缺失即报错,
  免得静默画出旧口径的历史。
- (b)(d) 主应力空间内的单元应力分布: 取 postprocess/ 下的 npz
  (sig1/sig2/vm/solid_mask/pad_mask), 由 ``compare.py export --run lfem-k3 --run
  huzhang-k3`` 从 density_final.vtu 冻结重分析导出; export 认的运行目录与本模块的
  REQUIRED_RUNS 前两项是同一批 (metrics.resolve_run_dir 按注册口径解析)。散点只画
  判据集合 E_acc: solid_mask (rho > 0.5) 剔除 pad_mask (被动实体区, rho 固定 1 但不
  施加约束)。

右轴画 ``max_relative_violation_solid``, 即判据集合上的 g_max, 停止准则判的就是它
<= delta_g; 全域 (含灰度单元) 的 ``max_relative_violation`` 不画, 正文亦不引.
参考线取 summary 记录的 ``relative_stress_tolerance``, 不另写常数。

输出 PDF 矢量与 600 dpi PNG 至 papers/huzhang-topopt/figures 与本地 outputs/figures, 不再出 EPS。

2026-09-28 起按版心尺寸出图: 图宽取 CICP 版心 150 mm (5.9 in), 论文里以 ``width=\\textwidth``
原尺寸嵌入, 面板标题字号即纸面字号; 字体走 ``paper_rcparams`` 的 Palatino 口径, 标题与轴名
改英文 (投稿稿用)。面板标题只写编号、方法与阶次, 体积分数等数值由正文给出。图例放在各面板内:
收敛历史三项单列放右上角 (框底在左轴 0.57 之上、框左缘在第 160 步之后, 该段 f_V 已低于 0.38,
g_max 低于 0.15, 不压数据; 图例挂在 twin 上, 免得被后画的 g_max 曲线盖住), 只写符号, 全名在
同色轴名上; 主应力面两项放左上角, 纵轴上限由 1.5 抬到 1.9 留出图例带 (屈服椭圆顶 1.155 在框底
1.20 之下, 满应力点全在 sigma_1 >= sigma_2 半平面), 色标改挂 inset_axes 以与非正方形面板等高。
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from ._base import paper_rcparams, require_run_dir, save_figure

paper_rcparams()

# 版心宽 150 mm; 左列收敛历史 (双纵轴), 右列主应力平面 (正方形 + 色标), 两行.
FIG_SIZE_IN = (5.9, 5.0)
TITLE_PT = 9.5
LABEL_PT = 9.0
TICK_PT = 8.0
DPI = 600

# 配色与图 5.8 一致: 体积分数走蓝实线, 违反量走绿虚线, 容差线取同色点线, 屈服面走红.
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#2ca02c"
COLOR_YIELD = "#d62728"
COLOR_REFERENCE = "#aaaaaa"
STRESS_CMAP = "jet"
LEGEND_STYLE = dict(fontsize=7.5, framealpha=0.95, edgecolor="#cccccc",
                    borderpad=0.4, labelspacing=0.3, handlelength=2.0)

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "analyzer-lfem__lfem_constraint-apparent__load_pad_radius-1.5__order-3__solid_thr-0.5",
    "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-3__solid_thr-0.5",
    "postprocess/lfem_constraint-apparent/fig_data_lfem-k3.npz",
    "postprocess/lfem_constraint-apparent/fig_data_huzhang-k3.npz",
)


def load_history(base, folder: str):
    """读一次运行的体积分数、最大局部约束值与应力容差.

    Parameters
    ----------
    base : Path
        算例产物根目录.
    folder : str
        运行目录名, 见 REQUIRED_RUNS.

    Returns
    -------
    tuple
        ``(volfrac, g_max, delta_g)``, 前两者为逐 MMA 更新的一维数组.

    Raises
    ------
    RuntimeError
        该运行未按判据集合验收 (summary 无 acceptance_solid_threshold).
    """
    run_dir = require_run_dir(base, folder)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("acceptance_solid_threshold") is None:
        raise RuntimeError(f"{folder}: summary 无 acceptance_solid_threshold, 不是判据集合验收的运行.")

    histories = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    scalars = histories["scalar_histories"]
    return (
        np.asarray(scalars["volfrac"], dtype=float),
        np.asarray(scalars["max_relative_violation_solid"], dtype=float),
        float(summary["relative_stress_tolerance"]),
    )


def load_fields(base, relpath: str):
    """读冻结重分析导出的主应力场; 缺 pad_mask 即报错, 免得把被动实体区画进去."""
    path = base / relpath
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}; 请先运行 compare.py export --run lfem-k3 --run huzhang-k3")
    with np.load(path) as npz:
        if "pad_mask" not in npz.files:
            raise RuntimeError(f"{path} 无 pad_mask, 是旧口径导出; 请重新 export.")
        return {key: np.asarray(npz[key])
                for key in ("sig1", "sig2", "vm", "solid_mask", "pad_mask")}


def acceptance_mask(fields) -> np.ndarray:
    """判据集合 E_acc = 实体单元 (rho > 0.5) 剔除被动实体区."""
    return fields["solid_mask"].astype(bool) & ~fields["pad_mask"].astype(bool)


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE
    lf_vf, lf_g, lf_tol = load_history(base, REQUIRED_RUNS[0])
    hz_vf, hz_g, hz_tol = load_history(base, REQUIRED_RUNS[1])
    lf_fields = load_fields(base, REQUIRED_RUNS[2])
    hz_fields = load_fields(base, REQUIRED_RUNS[3])

    def plot_convergence(axes, volfrac, g_max, tolerance, title):
        iterations = np.arange(1, len(volfrac) + 1)
        line_volume, = axes.plot(iterations, volfrac, color=COLOR_PRIMARY, lw=1.2,
                                 label="$f_V$")
        axes.set_xlabel("Iteration", fontsize=LABEL_PT)
        axes.set_ylabel("Volume fraction $f_V$", fontsize=LABEL_PT, color=COLOR_PRIMARY)
        axes.tick_params(axis="both", labelsize=TICK_PT)
        axes.set_ylim(0.30, 0.70)

        twin = axes.twinx()
        line_violation, = twin.plot(iterations, g_max, color=COLOR_SECONDARY, lw=1.1,
                                    ls="--", label="$g_{\\max}$")
        line_tolerance = twin.axhline(tolerance, color=COLOR_SECONDARY, ls=":", lw=0.9,
                                      alpha=0.7, label=f"$\\delta_g = {tolerance:g}$")
        twin.set_yscale("log")
        twin.set_ylabel("Max. local constraint $g_{\\max}$", fontsize=LABEL_PT,
                        color=COLOR_SECONDARY)
        twin.tick_params(axis="y", labelsize=TICK_PT)
        twin.set_ylim(min(tolerance, float(g_max.min())) * 0.5,
                      float(g_max.max()) * 2.0)

        twin.legend(handles=[line_volume, line_violation, line_tolerance], loc="upper right",
                    **{**LEGEND_STYLE, "handlelength": 1.6})
        axes.set_title(title, fontsize=TITLE_PT, pad=4)
        axes.grid(True, ls=":", lw=0.5, alpha=0.5)

    def plot_yield_surface(axes, fields, title):
        # 只画判据集合: 空洞单元的表观应力被 m_E 压到接近零, 全堆在原点; 被动实体区
        # rho 固定为 1 但不受约束, 也不属于判据集合, 一并剔除.
        keep = acceptance_mask(fields)
        sig1 = fields["sig1"][keep]
        sig2 = fields["sig2"][keep]
        vm = fields["vm"][keep]

        # von Mises 许用应力边界: sig1^2 - sig1*sig2 + sig2^2 = 1 (参数化椭圆)
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ct, st = np.cos(th), np.sin(th)
        scale = 1.0 / np.sqrt(ct ** 2 - ct * st + st ** 2)
        yield_line, = axes.plot(ct * scale, st * scale, color=COLOR_YIELD, lw=1.1,
                                label="$\\sigma_{\\mathrm{vm}} = \\bar{\\sigma}$")

        # 色标固定在 [0, 1], 两图可直接对比.
        sc = axes.scatter(sig1, sig2, c=vm, cmap=STRESS_CMAP, vmin=0.0, vmax=1.0,
                          s=2.5, alpha=0.85, linewidths=0)

        marker_proxy = Line2D([], [], linestyle="none", marker="o", markersize=3.5,
                              markerfacecolor=plt.get_cmap(STRESS_CMAP)(0.75),
                              markeredgecolor="none",
                              label=r"Solid elements ($\mathcal{E}_{\mathrm{acc}}$)")

        axes.axhline(0.0, color=COLOR_REFERENCE, lw=0.6, ls=":")
        axes.axvline(0.0, color=COLOR_REFERENCE, lw=0.6, ls=":")
        axes.set_xlim(-1.5, 1.5)
        axes.set_ylim(-1.5, 1.9)
        axes.set_aspect("equal")
        axes.legend(handles=[yield_line, marker_proxy], loc="upper left",
                    **{**LEGEND_STYLE, "fontsize": 7.0, "handlelength": 1.2,
                       "borderpad": 0.3})
        axes.set_xlabel(r"$\sigma_1 / \bar{\sigma}$", fontsize=LABEL_PT)
        axes.set_ylabel(r"$\sigma_2 / \bar{\sigma}$", fontsize=LABEL_PT)
        axes.tick_params(labelsize=TICK_PT)
        axes.set_title(title, fontsize=TITLE_PT, pad=4)
        axes.grid(True, ls=":", lw=0.5, alpha=0.5)

        cax = axes.inset_axes([1.05, 0.0, 0.05, 1.0])
        cbar = axes.figure.colorbar(sc, cax=cax)
        cbar.set_ticks(np.linspace(0.0, 1.0, 6))
        cbar.ax.tick_params(labelsize=TICK_PT)
        cbar.set_label(r"$\sigma_{\mathrm{vm}} / \bar{\sigma}$", fontsize=LABEL_PT)

    # MMA 更新次数由正文给出, 标题不写.
    fig = plt.figure(figsize=FIG_SIZE_IN)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.55, hspace=0.42,
                          left=0.10, right=0.93, top=0.95, bottom=0.08)

    plot_convergence(fig.add_subplot(gs[0, 0]), lf_vf, lf_g, lf_tol,
                     "(a) LFEM, $p = 3$")
    plot_yield_surface(fig.add_subplot(gs[0, 1]), lf_fields, "(b) LFEM, $p = 3$")
    plot_convergence(fig.add_subplot(gs[1, 0]), hz_vf, hz_g, hz_tol,
                     "(c) HZMFEM, $k = 3$")
    plot_yield_surface(fig.add_subplot(gs[1, 1]), hz_fields, "(d) HZMFEM, $k = 3$")

    save_figure(fig, "stress_cubic_convergence", formats=("pdf", "png"), dpi=DPI)


if __name__ == "__main__":
    main()
