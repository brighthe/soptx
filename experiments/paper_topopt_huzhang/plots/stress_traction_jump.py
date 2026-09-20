# -*- coding: utf-8 -*-
"""冻结构型上的法向牵引剖面与绝对跳量-余量对比 (论文图 5.14).

产物 case ``stress-traction-jump``: 依赖 postprocess/discretization_probe/ 下的两份
fields.npz, 由 ``compare.py discretization-probe`` 产出, 不重解方程。

本图回答 5.2.3(new) 的核心问题: 位移元下"单元 e 的应力"不唯一, 其取值依赖于从哪一侧
单元读取; 这一不唯一性的量级与停止容差 delta_g、与判决余量 -g_e 相比如何。四幅面板:

- (a) 剖面: 在位移法 p=3 构型的上缘杆件内取一列竖直内边 (法向沿 x, 中点 y 落在
  PROFILE_BAND, 两侧单元均在实体带), 对每条边分别从 x- 侧与 x+ 侧单元读取边平均的
  法向牵引 sigma_nn / sigma_bar (LFEM p=3). 杆件内部两侧读数在图上重合, 差在 1e-3
  量级; 单看剖面看不出不唯一性, 故 (b) 画差本身.
- (b) 同一边链上两侧读数之差 |t_n^+ - t_n^-| / sigma_bar, 对数纵轴, LFEM p=3、p=4
  与 HZMFEM k=3 并列, 并标 delta_g 线. 位移法的差与 delta_g 同量级, 混合法在舍入.
  剖面数据来自 npz 的 ``trace__<disc>`` (探针在同序求积点上算得, 已除以许用应力),
  边平均用 ``trace_weights__<disc>`` 加权.
- (c)(d) 累积分布: 逐单元绝对跳量 ``A_e = max_{F in dT_e, F 为内边} ||[[sigma n]]||_rms(F) / sigma_bar``
  (npz 的 ``absjump__<disc>``, 由 edge_jump 的 ``cell_rms_jump`` 归一) 与 delta_g、
  可行余量 ``-g_e`` 对照. 余量与占比一律取同一条离散 (LFEM p=3) 的读数: 跳量是 p=3
  读数的不唯一性, 余量也须是 p=3 读数距约束边界的距离, 二者才可比; 在混合法构型上
  这意味着余量来自 p=3 的重评价而非构型自身的 k=3 读数.
  不用相对跳量 J_F = ||[[sigma n]]|| / ||{sigma n}||: 杆件内水平边的法向牵引
  (~sigma_yy) 本身接近零, J_F 被分母放大到 O(1) 而绝对跳量只有 1e-2, 相对量与
  余量不在同一尺度, 不能对照. 实体带内 m_E ~ 1 故 eta ~ 1, ``-g_e`` 即
  1 - sigma_vm / sigma_bar.

统计一律限于实体带 (physical density > SOLID_THRESHOLD 且不在被动实体区): 被动实体区
不施加约束, 其余量无判决意义 (掩码取 npz 的 ``pad_mask``).

输出 png/pdf/eps 三种格式至 papers/huzhang-topopt/figures 与本地 outputs/figures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from ._base import academic_rcparams, chinese_font, save_figure

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "postprocess/discretization_probe/lfem-3-pad-solid__fields.npz",
    "postprocess/discretization_probe/huzhang-3-pad-solid__fields.npz",
)

PROBE_DIR = "postprocess/discretization_probe"
SOLID_THRESHOLD = 0.9
# 停止准则容差, 与 discretization_probe.DELTA_G / cases.toml 的 stress_tolerance 同值.
DELTA_G = 5.0e-3
# 剖面边链: 上缘杆件占 y in [36, 40], x in [0, 46]; 取中点 y 落在杆件中部一行的竖直内边.
PROFILE_BAND = (38.0, 39.0)
# 剖面所比较的位移离散: 余量与占比都按它读.
PROFILE_LABEL = "lfem-3"
# 混合法的跳量在 1e-16 量级: (b) 按真值画, 纵轴下界 1e-17; (c)(d) 截到 CDF_FLOOR 画成
# 一条竖线, 横轴左界再低半个量级, 使竖线不与坐标轴重合而被挡掉.
DIFF_FLOOR = 1e-17
CDF_FLOOR = 1e-4
CDF_XMIN = 3e-5
CDF_XMAX = 3e0
LFEM_LABELS = ("lfem-1", "lfem-2", "lfem-3", "lfem-4")
LFEM_COLORS = ("#9e9e9e", "#5a5a5a", "#1f77b4", "#ff7f0e")
COLOR_HEADROOM = "#d62728"
COLOR_DELTA = "#7f7f7f"
COLOR_HUZHANG = "#2ca02c"
COLOR_MINUS = "#1f77b4"
COLOR_PLUS = "#ff7f0e"
LEGEND_STYLE = dict(fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")


def load_design(base, tag: str):
    """读一个冻结构型的密度、掩码、逐单元绝对跳量/余量与逐边剖面数据.

    Parameters
    ----------
    base : Path
        算例产物根目录.
    tag : str
        探针产物前缀, 如 ``lfem-3-pad-solid``.

    Returns
    -------
    dict
        ``solid`` 实体带掩码; ``headroom`` 为 PROFILE_LABEL 读数下的 ``-g_e``;
        ``jump`` 标签到逐单元绝对跳量 ``A_e`` 的映射; ``fields`` 为原始 npz (剖面用).
    """
    fields = np.load(base / PROBE_DIR / f"{tag}__fields.npz")
    needed = ("face_index", f"trace__{PROFILE_LABEL}", f"absjump__{PROFILE_LABEL}")
    if any(key not in fields.files for key in needed):
        raise RuntimeError(
            f"{tag}: npz 缺逐边剖面或绝对跳量 (face_index / trace__* / absjump__*), "
            "是旧口径探针; 请重新运行 compare.py discretization-probe."
        )
    rho = fields["density"]
    jumps = {key: fields[f"absjump__{key}"] for key in (*LFEM_LABELS, "huzhang-3")}
    return {
        "rho": rho,
        "barycenter": fields["barycenter"],
        "solid": (rho > SOLID_THRESHOLD) & ~fields["pad_mask"].astype(bool),
        "headroom": -fields[f"g__{PROFILE_LABEL}"],
        "jump": jumps,
        "fields": fields,
    }


def profile_edges(design) -> np.ndarray:
    """选出剖面边链: 法向沿 x、中点 y 在 PROFILE_BAND 内、两侧均为实体带的内边, 按 x 排序."""
    fields = design["fields"]
    normal = fields["face_normal"]
    midpoint = fields["face_midpoint"]
    cells = fields["face_cells"]
    solid = design["solid"]
    keep = (np.abs(normal[:, 0]) > 0.99)
    keep &= (midpoint[:, 1] >= PROFILE_BAND[0]) & (midpoint[:, 1] <= PROFILE_BAND[1])
    keep &= solid[cells[:, 0]] & solid[cells[:, 1]]
    selected = np.flatnonzero(keep)
    if selected.size == 0:
        raise RuntimeError("剖面边链为空: PROFILE_BAND 与构型不匹配.")
    return selected[np.argsort(midpoint[selected, 0])]


def side_readings(design, label: str, edges: np.ndarray):
    """对边链上每条边给出 x- 侧与 x+ 侧单元读到的边平均法向牵引 sigma_nn / sigma_bar.

    Returns
    -------
    x : ndarray, shape (n,)
        边中点横坐标.
    minus, plus : ndarray, shape (n,)
        两侧读数; sigma_nn = n^T sigma n 与法向取向无关, 故无需处理 n 的符号.
    """
    fields = design["fields"]
    trace = fields[f"trace__{label}"][edges]            # (n, 2, NQ, 2)
    weights = fields[f"trace_weights__{label}"]
    normal_component = (trace[..., 0] * weights[None, None, :]).sum(-1) / weights.sum()
    cells = fields["face_cells"][edges]
    x_cells = design["barycenter"][cells, 0]              # (n, 2)
    minus_is_first = x_cells[:, 0] < x_cells[:, 1]
    minus = np.where(minus_is_first, normal_component[:, 0], normal_component[:, 1])
    plus = np.where(minus_is_first, normal_component[:, 1], normal_component[:, 0])
    x = fields["face_midpoint"][edges, 0]
    return x, minus, plus


def plot_profile(axes, design, label: str, edges: np.ndarray, title, font):
    """画一条离散在剖面边链上的两侧法向牵引读数."""
    x, minus, plus = side_readings(design, label, edges)
    axes.vlines(x, np.minimum(minus, plus), np.maximum(minus, plus),
                color="#888888", lw=1.0, zorder=1)
    h_minus, = axes.plot(x, minus, ls="none", marker="o", ms=5.0, color=COLOR_MINUS,
                         label="从 $x^-$ 侧单元读取")
    h_plus, = axes.plot(x, plus, ls="none", marker="s", ms=4.5, mfc="none", mew=1.3,
                        color=COLOR_PLUS, label="从 $x^+$ 侧单元读取")
    axes.axhline(1.0, color="#d62728", lw=1.0, ls="--")
    axes.axhline(-1.0, color="#d62728", lw=1.0, ls="--")
    axes.axhline(0.0, color="#aaaaaa", lw=0.8, ls=":")
    diff = np.abs(plus - minus)
    stat = Line2D([], [], linestyle="none",
                  label=(f"两侧之差中位 {np.median(diff):.1e}，最大 {diff.max():.1e}"))
    axes.legend(handles=[h_minus, h_plus, stat], loc="lower left", prop=font,
                **LEGEND_STYLE)
    axes.set_xlabel("边中点横坐标 $x$ (mm)", fontsize=11, fontproperties=font)
    axes.set_ylabel(r"边平均法向牵引 $\bar t_n / \bar\sigma$", fontsize=11,
                    fontproperties=font)
    axes.tick_params(labelsize=9)
    axes.grid(True, ls=":", alpha=0.5)
    axes.set_title(title, fontsize=11, fontproperties=font, y=-0.30)


def plot_side_difference(axes, design, edges: np.ndarray, title, font):
    """画同一边链上两侧读数之差 |t_n^+ - t_n^-| / sigma_bar, 对数纵轴, 三条离散并列."""
    handles = []
    series = (
        ("lfem-3", COLOR_MINUS, "o", 5.0, "LFEM $p = 3$"),
        ("lfem-4", COLOR_PLUS, "s", 4.5, "LFEM $p = 4$"),
        ("huzhang-3", COLOR_HUZHANG, "^", 5.0, "HZMFEM $k = 3$"),
    )
    for label, color, marker, size, text in series:
        x, minus, plus = side_readings(design, label, edges)
        raw = np.abs(plus - minus)
        # 混合法的差多为精确零, 截到 DIFF_FLOOR 才能上对数轴; 图例报截断前的最大值.
        diff = np.clip(raw, DIFF_FLOOR, None)
        stat = (f"最大 {raw.max():.1e}" if np.median(raw) < DIFF_FLOOR
                else f"中位 {np.median(raw):.1e}，最大 {raw.max():.1e}")
        line, = axes.plot(x, diff, ls="-", lw=0.8, marker=marker, ms=size, color=color,
                          mfc="none" if marker == "s" else color, mew=1.2,
                          label=f"{text}（{stat}）")
        handles.append(line)
    line = axes.axhline(DELTA_G, color=COLOR_DELTA, lw=1.4, ls="--",
                        label=f"停止容差 $\\delta_g = {DELTA_G:g}$")
    handles.append(line)
    axes.set_yscale("log")
    axes.set_ylim(DIFF_FLOOR / 3, 1e0)
    axes.set_xlabel("边中点横坐标 $x$ (mm)", fontsize=11, fontproperties=font)
    axes.set_ylabel(r"两侧读数之差 $|\bar t_n^{+} - \bar t_n^{-}| / \bar\sigma$",
                    fontsize=11, fontproperties=font)
    axes.tick_params(labelsize=9)
    axes.grid(True, ls=":", alpha=0.5, which="major")
    axes.legend(handles=handles, loc="center right", prop=font, **LEGEND_STYLE)
    axes.set_title(title, fontsize=11, fontproperties=font, y=-0.30)


def plot_cdf(axes, design, title, font):
    """画实体带上逐单元绝对跳量 A_e 与余量 -g_e 的经验累积分布, 并标 delta_g.

    图例句柄返回给调用方合成 (c)(d) 共用的图底图例; 逐面板的占比写进标题第二行,
    避免图例或文字框压住左上角的曲线与两条竖线.
    """
    solid = design["solid"]
    headroom = design["headroom"][solid]
    handles = []

    for label, color in zip(LFEM_LABELS, LFEM_COLORS):
        values = np.clip(design["jump"][label][solid], CDF_FLOOR, None)
        order = np.sort(values)
        fraction = np.arange(1, order.size + 1) / order.size
        line, = axes.plot(order, fraction, color=color, lw=1.6,
                          label=f"$A_e$（LFEM $p = {label[-1]}$）")
        handles.append(line)

    values = np.clip(design["jump"]["huzhang-3"][solid], CDF_FLOOR, None)
    order = np.sort(values)
    line, = axes.plot(order, np.arange(1, order.size + 1) / order.size,
                      color=COLOR_HUZHANG, lw=2.0,
                      label="$A_e$（HZMFEM $k = 3$，机器零）")
    handles.append(line)

    order = np.sort(np.clip(headroom, CDF_FLOOR, None))
    line, = axes.plot(order, np.arange(1, order.size + 1) / order.size,
                      color=COLOR_HEADROOM, lw=2.2, ls="--",
                      label="可行余量 $-g_e$（LFEM $p = 3$ 读数）")
    handles.append(line)

    line = axes.axvline(DELTA_G, color=COLOR_DELTA, lw=1.4, ls="--",
                        label=f"停止容差 $\\delta_g = {DELTA_G:g}$")
    handles.append(line)

    # 直读指标: 同一离散 (p=3) 下, 跳量大于停止容差、大于自身余量的实体单元占比.
    jump = design["jump"][PROFILE_LABEL][solid]
    over_delta = float(np.mean(jump > DELTA_G))
    over_headroom = float(np.mean(jump > headroom))
    subtitle = (f"$p = 3$ 读数：$A_e > \\delta_g$ 占 {over_delta * 100:.0f}%，"
                f"$A_e > -g_e$ 占 {over_headroom * 100:.1f}%")

    axes.set_xscale("log")
    axes.set_xlim(CDF_XMIN, CDF_XMAX)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel(r"逐单元绝对牵引跳量 $A_e$ 与可行余量 $-g_e$（单位 $\bar\sigma$）",
                    fontsize=11, fontproperties=font)
    axes.set_ylabel("实体单元累积占比", fontsize=11, fontproperties=font)
    axes.tick_params(labelsize=9)
    axes.grid(True, ls=":", alpha=0.5)
    axes.set_title(f"{title}\n{subtitle}", fontsize=11, fontproperties=font, y=-0.40)
    return handles


def main() -> None:
    base = config.OUTPUT_DIR / SOURCE_CASE
    lfem = load_design(base, "lfem-3-pad-solid")
    huzhang = load_design(base, "huzhang-3-pad-solid")
    edges = profile_edges(lfem)

    ZH = chinese_font()

    fig = plt.figure(figsize=(12.5, 11.0), dpi=300)
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.58,
                          left=0.07, right=0.98, top=0.97, bottom=0.235)

    plot_profile(fig.add_subplot(gs[0, 0]), lfem, PROFILE_LABEL, edges,
                 f"(a) 上缘杆件边链上的两侧法向牵引读数（LFEM $p = 3$，{edges.size} 条边）",
                 ZH)
    plot_side_difference(fig.add_subplot(gs[0, 1]), lfem, edges,
                         "(b) 同一边链上两侧读数之差与停止容差的对比", ZH)

    handles = plot_cdf(fig.add_subplot(gs[1, 0]), lfem,
                       "(c) 位移法 ($p = 3$) 构型上实体带的累积分布", ZH)
    plot_cdf(fig.add_subplot(gs[1, 1]), huzhang,
             "(d) 混合法 ($k = 3$) 构型上实体带的累积分布", ZH)
    fig.legend(handles=handles, loc="lower center", ncol=4, prop=ZH,
               bbox_to_anchor=(0.5, 0.01), **LEGEND_STYLE)

    save_figure(fig, "stress_traction_jump", formats=("png", "pdf", "eps"))


if __name__ == "__main__":
    main()
