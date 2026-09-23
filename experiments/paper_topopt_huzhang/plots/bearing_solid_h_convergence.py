"""全域实体时柔顺度误差随网格尺寸的收敛 (论文图 5.5, Bruggi 2016 5.1 节图 6 式闭锁考察).

产物 case ``bearing-solid-h-convergence``: 左右两幅并排, (a) $\\nu_0 = 0.3$, (b) $\\nu_0 = 0.4999$;
横轴单元尺寸 $h$ (30x10 … 240x80 四级, 对数刻度), 纵轴各离散在全域实体 ($\\rho_e \\equiv 1$,
无材料插值) 上的柔顺度相对误差绝对值 (%, 对数刻度)。四条曲线 LFEM $p=1$ / LFEM $p=2$ /
HZMFEM $k=2$ (跳量稳定化) / HZMFEM $k=4$ (原生, 最细一级未跑)。位移元从下方逼近 (误差为负),
混合元从上方逼近 (误差为正), 图上只画绝对值, 符号写在图注。

参考值不是任一离散在某网格上的值: 120x40 上 $k=4$ 自身仍高出真值 0.3% 至 0.4%, 与被评估
离散的误差同量级。这里取 $k=4$ 前三级按逐级差等比递减外推的极限 (Aitken delta^2, 按实测差比,
不假定收敛阶), 与 ``bearing_h_locking_probe.py`` 终端表同一公式; 改用其余序列外推, 参考值变化
不超过 0.02%, 即其不确定度, 因此 $p=2$ 在 $\\nu_0 = 0.3$ 最细两级的点 (0.016%, 0.013%) 不可信。

数据不重解方程, 只读 ``bearing_h_locking_probe.py`` 落盘的
``outputs/bearing-incompressible/postprocess/solid_h_sweep.json``; 缺文件即报错。

输出: outputs/figures/bearing_solid_h_convergence.{png,pdf,eps}, 自动同步至
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
REQUIRED_RUNS = ("postprocess/solid_h_sweep.json",)

REFERENCE_LABEL = "huzhang-4"
DOMAIN_LENGTH = 120.0  # mm; h = 120 / nx
SERIES = (
    # (json 键, 图例, 颜色, 线型, 标记)
    ("lfem-1", "LFEM $p=1$", "#d62728", "-", "o"),
    ("lfem-2", "LFEM $p=2$", "#1f77b4", "--", "s"),
    ("huzhang-2", "HZMFEM $k=2$", "#2ca02c", "-.", "^"),
    ("huzhang-4", "HZMFEM $k=4$", "#9467bd", ":", "D"),
)


def load_rows() -> list[dict]:
    path = config.OUTPUT_DIR / SOURCE_CASE / REQUIRED_RUNS[0]
    if not path.is_file():
        raise RuntimeError(f"缺少 {path}; 请先运行 compare.py bearing-h-locking")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("design") != "solid":
        raise RuntimeError(f"{path} 不是全实体域扫描产物")
    return data["rows"]


def extrapolated_reference(block: list[dict]) -> float:
    """k=4 前三级 (由粗到细) 按逐级差等比递减外推的极限 (Aitken delta^2); 与 bearing_h_locking_probe 同一公式."""
    c1, c2, c3 = (block[i]["compliance"][REFERENCE_LABEL] for i in range(3))
    ratio = (c2 - c3) / (c1 - c2)
    return c3 - (c2 - c3) * ratio / (1.0 - ratio)


def main() -> None:
    rows = load_rows()
    nus = sorted({row["poisson_ratio"] for row in rows})
    if len(nus) != 2:
        raise RuntimeError(f"期望两档 nu, 实得 {nus}")

    ZH = chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=300, sharey=True)
    panel = ("(a)", "(b)")

    for ax, tag, nu in zip(axes, panel, nus):
        block = sorted((r for r in rows if r["poisson_ratio"] == nu), key=lambda r: r["nx"])  # 由粗到细
        reference = extrapolated_reference(block)
        for key, label, color, ls, marker in SERIES:
            pts = [(DOMAIN_LENGTH / r["nx"], 100.0 * abs(r["compliance"][key] / reference - 1.0))
                   for r in block if key in r["compliance"]]
            h = np.array([p[0] for p in pts])
            err = np.array([p[1] for p in pts])
            ax.loglog(h, err, color=color, lw=1.8, ls=ls, marker=marker, ms=5.5,
                      mec="white", mew=0.7, label=label, zorder=3)
        # 一阶参考斜率: 放在右下, 不压任何曲线
        h0, h1 = 1.6, 3.2
        e0 = 0.008
        ax.loglog([h0, h1, h1, h0], [e0, e0 * 2.0, e0, e0], color="#555555", lw=0.9, zorder=2)
        ax.text(h1 * 1.05, e0 * 1.3, "1", fontsize=8.5, color="#555555", va="center")

        ax.set_xscale("log")
        ax.set_xticks([0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["0.5", "1", "2", "4"], fontsize=9)
        ax.minorticks_off()
        ax.set_xlim(0.38, 5.2)
        ax.set_xlabel("单元尺寸 $h$ (mm)", fontsize=11, fontproperties=ZH)
        ax.grid(True, which="major", ls=":", alpha=0.5)
        ax.set_title(f"{tag} $\\nu_0 = {nu:g}$", fontsize=11, pad=6)
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylim(5e-3, 100.0)
    axes[0].set_ylabel("柔顺度相对误差 $|\\Delta C / C|$ (%)", fontsize=11, fontproperties=ZH)
    axes[0].legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.15, wspace=0.08)
    save_figure(fig, "bearing_solid_h_convergence", formats=("png", "pdf", "eps"))
    plt.close(fig)


if __name__ == "__main__":
    main()
