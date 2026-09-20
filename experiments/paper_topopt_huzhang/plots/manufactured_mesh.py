"""制造解算例的棋盘格三角剖分示意 (论文 5.1 节, 图号待定).

产物 case ``manufactured-mesh``: 单幅, 在单位正方形上画出网格序列的最粗一级
``nx = ny = 4``。

剖分取 ``triangle-checkerboard``: 每个四边形按 ``(i+j)`` 的奇偶交替取对角线, 偶数取
``/``, 奇数取 ``\\`` (见 ``soptx.mesh.create_huzhang_checkerboard_mesh``)。序列其余四级
(8 / 16 / 32 / 64) 按同一规则加密, 画最粗一级即可看清规则。

输出: outputs/figures/manufactured_mesh.png, 自动同步至 papers/huzhang-topopt/figures/

本模块不读任何运行产物 (``REQUIRED_RUNS`` 为空): 剖分规则是纯几何事实, 由网格生成器
自身决定, 与是否跑过收敛验证无关。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation

from soptx.mesh import create_huzhang_checkerboard_mesh

from ._base import save_figure

# ---- 自描述元数据: compare.py 用 ast 静态解析读走, 不 import 本模块 ----
# REQUIRED_RUNS 留空, 见模块 docstring。
SOURCE_CASE = "manufactured-native"
REQUIRED_RUNS = ()

# 与 2.1 节的问题定义一致: 单位正方形; 只画序列最粗一级
BOX = (0.0, 1.0, 0.0, 1.0)
LEVEL = 4

_EDGE_COLOR = "#2b2b2b"
_FILL_COLOR = "#e8eef6"


def _draw(ax, n: int) -> None:
    """在 ``ax`` 上画 nx = ny = n 的棋盘格剖分."""
    mesh = create_huzhang_checkerboard_mesh(box=list(BOX), nx=n, ny=n)
    node = mesh.entity("node")
    cell = mesh.entity("cell")
    triangulation = Triangulation(node[:, 0], node[:, 1], cell)

    xmin, xmax, ymin, ymax = BOX
    # 网格铺满整个矩形域, 故底色用一块矩形即可, 不必按单元逐个上色
    ax.add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor=_FILL_COLOR, edgecolor="none", zorder=1,
        )
    )
    ax.triplot(triangulation, color=_EDGE_COLOR, linewidth=0.8, zorder=2)

    ax.set_xlim(xmin - 0.03, xmax + 0.03)
    ax.set_ylim(ymin - 0.03, ymax + 0.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    _draw(ax, LEVEL)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, "manufactured_mesh")
    plt.close(fig)


if __name__ == "__main__":
    main()
