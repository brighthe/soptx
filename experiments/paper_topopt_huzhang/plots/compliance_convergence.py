"""固支梁柔顺度收敛历史对比 (论文图 5.3).

产物 case ``compliance-convergence``: 左右两幅并排, 每幅一种离散方法, 三条柔顺度曲线
按阶次着色, 另叠一条体积分数虚线看约束是否锁在 0.40::

  (a) LFEM   p=2,3,4
  (b) HZMFEM k=2,3,4

主轴的量级由前 20 步的瞬态定 (184 -> 32), 末值之间只差 2% 上下, 在主轴里分不开, 因此
右下角另开一个 inset 单框收敛段; 主轴与 inset 的纵向量程都取自全部六次运行, 两幅子图
共用一套刻度, 否则 (a)(b) 横着对出来的差是缩放差而不是柔顺度差。

输出: outputs/figures/ 下三张 —— 合并图 compliance_convergence 与两幅单图
compliance_convergence_{lfem,hzmfem}, 均自动同步至 papers/figures/

论文图号只写在首行括注里: 排版改号时改这一处, case id 与命令行都不受影响。
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import config
from ._base import academic_rcparams, require_run_dir, save_figure

academic_rcparams()

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
# 这张图吃哪个算例的哪几次运行, 本就是绘图代码自己的事实, 故写在模块身上而
# 不另立注册表; 两个常量都保持字面量, ast.literal_eval 才读得动。
SOURCE_CASE = "compliance-fixed-fixed-half"
REQUIRED_RUNS = (
    "analyzer-lfem__order-2", "analyzer-lfem__order-3", "analyzer-lfem__order-4",
    "analyzer-huzhang__order-2", "analyzer-huzhang__order-3", "analyzer-huzhang__order-4",
)

OUTPUTS = config.OUTPUT_DIR / SOURCE_CASE

# REQUIRED_RUNS 前半段是 LFEM 后半段是 HZMFEM, 两幅子图各取一段配上阶次。目录名
# 只在 REQUIRED_RUNS 里写一遍, 不在本文件出现第二处。
ORDERS = (2, 3, 4)
LFEM_RUNS = list(zip(REQUIRED_RUNS[: len(ORDERS)], ORDERS))
HZ_RUNS = list(zip(REQUIRED_RUNS[len(ORDERS) :], ORDERS))

# 阶次记号随方法走, 与 run.py 的 ORDER_SYMBOLS 同口径: Hu--Zhang 的 k 是应力空间次数
# (位移阶为 k-1), LFEM 的 p 是位移阶。两幅子图统一写成 k 会把 LFEM 的位移阶读成应力阶,
# 图上同样标 2 的两条线就不是同一个量了。
ORDER_SYMBOLS = {"huzhang": "k", "lfem": "p"}

# 经典高对比学术配色; linestyle 与颜色同时区分阶次, 三条线在收敛段几乎重合, 只靠
# 颜色的话哪条在上层纯看绘制顺序。
COLORS = ["#d62728", "#1f77b4", "#2ca02c"]  # 阶次 2: 红色, 3: 蓝色, 4: 绿色
LINESTYLES = ["-", "--", ":"]

# 横轴上界两幅子图共用 (最长的一次运行 222 步), 不随子图各自的最大迭代数浮动。
XMAX = 230
# inset 从第 100 步框起: 六次运行最短的一条 152 步, 此后全在收敛段。
INSET_XMIN = 100
# inset 在主轴里的位置 (axes 坐标 x0,y0,w,h): 避开左上的瞬态、右上的图例与中部的 V_f 线。
INSET_BOUNDS = (0.35, 0.12, 0.60, 0.30)

_HISTORY_CACHE: dict[str, dict] = {}


def load_history(folder: str) -> dict:
    """读一次运行的 history.json; 主轴/inset/量程计算要各读一遍, 故缓存."""
    if folder not in _HISTORY_CACHE:
        run_dir = require_run_dir(OUTPUTS, folder)
        print(f"  {folder}: {run_dir.relative_to(config.OUTPUT_DIR)}")
        with open(run_dir / "history.json", encoding="utf-8") as f:
            _HISTORY_CACHE[folder] = json.load(f)
    return _HISTORY_CACHE[folder]


def compliance(folder: str) -> np.ndarray:
    """该次运行的完整结构柔顺度历史; 产物存的是半域, 在这里 x2 折算."""
    return np.array(load_history(folder)["scalar_histories"]["compliance"]) * 2.0


def padded_limits(lo: float, hi: float, margin: float) -> tuple[float, float]:
    """把 [lo, hi] 按跨度比例向两侧留白, 曲线不贴着框边跑."""
    span = hi - lo
    return lo - margin * span, hi + margin * span


def main_ylim() -> tuple[float, float]:
    """主轴纵向量程贴住全部六次运行的全程."""
    curves = [compliance(folder) for folder in REQUIRED_RUNS]
    return padded_limits(
        min(float(c.min()) for c in curves), max(float(c.max()) for c in curves), 0.05
    )


def inset_ylim() -> tuple[float, float]:
    """inset 纵向量程只贴收敛段, 同样取全部六次运行.

    LFEM 三阶次的末值散布 (0.13) 比 HZMFEM (0.75) 小近一个量级, 两幅 inset 各自缩放
    会把这个量级差本身抹平, 看上去像两种方法对阶次一样敏感。
    """
    tails = [compliance(folder)[INSET_XMIN - 1 :] for folder in REQUIRED_RUNS]
    return padded_limits(
        min(float(t.min()) for t in tails), max(float(t.max()) for t in tails), 0.08
    )


def plot_single_method(
    ax: plt.Axes,
    runs: list[tuple[str, int]],
    title: str,
    method: str,
) -> None:
    """绘制单种离散方法的宏观收敛曲线 (主轴 + 收敛段 inset)."""
    ax2 = ax.twinx()
    symbol = ORDER_SYMBOLS.get(method, "p")
    axins = ax.inset_axes(INSET_BOUNDS)

    # 1. 绘制柔顺度收敛曲线 (左 Y 轴), 同一条曲线同时进主轴与 inset
    for (folder, order), color, ls in zip(runs, COLORS, LINESTYLES):
        c = compliance(folder)
        it = np.arange(1, len(c) + 1)
        for target, lw in ((ax, 1.8), (axins, 1.4)):
            target.plot(it, c, color=color, lw=lw, ls=ls)
            # 末步端点: 三次运行迭代数不同 (152~222), 不标端点的话曲线尾部只剩「上层
            # 画完露出下层」, 看着像柔顺度分级下跌, 其实是绘制顺序造成的假象。
            target.plot(
                it[-1], c[-1], marker="o", ms=4.5, color=color,
                mec="white", mew=0.7, zorder=5, clip_on=False,
            )

    # 2. 绘制体积分数曲线 (右 Y 轴，灰色虚线)
    # 三个阶次各画一条: 它们全程都在 0.400 +- 3e-4 内, 在 0.38~0.42 的量程上完全重合,
    # 视觉上仍是一条线, 但线长跟着该子图最长的一次运行走。只画首条 (阶次 2) 的话,
    # 线会在那一次的末步断掉 —— HZMFEM 的 k=2 只跑 182 步, 看着就比 LFEM 短一截,
    # 而图例是通用的 $V_f$, 读者无从知道断口是哪来的。
    for folder, _ in runs:
        v = np.array(load_history(folder)["scalar_histories"]["volfrac"])
        it0 = np.arange(1, len(v) + 1)
        ax2.plot(it0, v, color="#777777", lw=1.3, ls="--", alpha=0.9)
    ax2.axhline(0.40, color="#aaaaaa", lw=0.8, ls=":")

    ax.set_xlabel("Iteration step", fontsize=11)
    ax.set_ylabel("Full-structure compliance $C$", fontsize=11)
    ax2.set_ylabel("Volume fraction $V_f$", fontsize=11, color="#555555")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(*main_ylim())
    ax2.set_ylim(0.38, 0.42)
    ax.grid(True, ls=":", alpha=0.5)

    # 3. 收敛段 inset: 主轴里被压成一条线的末值差, 在这里才分得开
    axins.set_xlim(INSET_XMIN, XMAX)
    axins.set_ylim(*inset_ylim())
    axins.grid(True, ls=":", alpha=0.5)
    axins.tick_params(labelsize=7.5)
    axins.set_title("Converged branch", fontsize=8, pad=3)
    ax.indicate_inset_zoom(axins, edgecolor="#555555", lw=0.8, alpha=0.85)

    # 4. 优雅图例设置 (放置在右上角空白区域，主次分明)
    handles = [
        Line2D([], [], color=c, lw=1.8, ls=ls) for c, ls in zip(COLORS, LINESTYLES)
    ]
    labels = [fr"${symbol}={order}$" for _, order in runs]
    handles += [Line2D([], [], color="#777777", lw=1.3, ls="--")]
    labels += [r"$V_f$ (Target 0.40)"]

    ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        fontsize=9.5,
        framealpha=0.95,
        edgecolor="#cccccc",
    )


def make_single(title: str, runs: list[tuple[str, int]], method: str, stem: str) -> None:
    """生成单张独立高质量收敛曲线图."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    plot_single_method(ax, runs, title, method)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.12)
    save_figure(fig, stem)
    plt.close(fig)


def make_combined() -> None:
    """生成一张 1 行 2 列并排横幅大图."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=300)
    plot_single_method(ax1, LFEM_RUNS, "(a) Standard Displacement Method (LFEM)", "lfem")
    plot_single_method(ax2, HZ_RUNS, "(b) Hu–Zhang Mixed Method (HZMFEM)", "huzhang")

    fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.12, wspace=0.28)
    save_figure(fig, "compliance_convergence")
    plt.close(fig)


def main():
    make_single(
        "(a) Standard Displacement Method (LFEM)", LFEM_RUNS, "lfem",
        "compliance_convergence_lfem",
    )
    make_single(
        "(b) Hu–Zhang Mixed Method (HZMFEM)", HZ_RUNS, "huzhang",
        "compliance_convergence_hzmfem",
    )
    make_combined()
    print("[OK] 收敛曲线三张图重新生成并同步完成")


if __name__ == "__main__":
    main()
