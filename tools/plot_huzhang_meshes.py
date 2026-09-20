"""生成 ``docs/fem/assets/`` 下的胡张元结构网格示意图.

图形直接由 :mod:`soptx.mesh.structured_triangle` 的生成器构造出的网格绘制,
而不是手工摆放线段, 因此剖分规则一旦改动, 重跑本脚本即可让插图跟上代码。

两种剖分共用同一套视觉约定: 灰色三角形边、四个几何角点标绿点并注 "2 tri"
(角点松弛要求的两单元拓扑), 镜像剖分额外画红色虚线中缝并把两个翻转的四边形
底色标蓝。

Examples
--------
::

    python tools/plot_huzhang_meshes.py --mesh checkerboard
    python tools/plot_huzhang_meshes.py --mesh symmetric --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fealpy.backend import backend_manager as bm  # noqa: E402

from soptx.mesh import (  # noqa: E402
    create_huzhang_checkerboard_mesh,
    create_huzhang_symmetric_single_diagonal_mesh,
)

ASSETS_DIR = REPO_ROOT / "docs" / "fem" / "assets"

# 取 1.5 x 1.0 配 6 x 4, 使单元为正方形 (hx = hy = 0.25), 三角形不被拉长
BOX = (0.0, 1.5, 0.0, 1.0)
NX, NY = 6, 4

EDGE_COLOR = "#4d4d4d"
EDGE_WIDTH = 0.9
CORNER_COLOR = "#1a7a1a"
HIGHLIGHT_COLOR = "#cfe2f0"
MIRROR_COLOR = "#d62728"


def _corner_label_offsets(box):
    """返回四个几何角点的坐标及其标注文字的对齐方式.

    Parameters
    ----------
    box : tuple of float
        ``(xmin, xmax, ymin, ymax)``。

    Returns
    -------
    list of tuple
        每项为 ``(x, y, dx, dy, ha, va)``, 后四者用于把 "2 tri" 推到角点外侧。
    """
    xmin, xmax, ymin, ymax = box
    span = min(xmax - xmin, ymax - ymin)
    dx = dy = 0.06 * span
    return [
        (xmin, ymin, -dx, -dy, "right", "top"),
        (xmax, ymin, +dx, -dy, "left", "top"),
        (xmin, ymax, -dx, +dy, "right", "bottom"),
        (xmax, ymax, +dx, +dy, "left", "bottom"),
    ]


def _highlighted_cells(node, cell, quads):
    """挑出重心落在指定四边形内的三角形.

    Parameters
    ----------
    node : ndarray
        结点坐标, 形状 ``(NN, 2)``。
    cell : ndarray
        单元结点编号, 形状 ``(NC, 3)``。
    quads : iterable of tuple
        四边形的 ``(ix, iy)`` 下标。

    Returns
    -------
    ndarray
        命中的单元下标。
    """
    centroid = node[cell].mean(axis=1)
    hx, hy = (BOX[1] - BOX[0]) / NX, (BOX[3] - BOX[2]) / NY
    hit = np.zeros(cell.shape[0], dtype=bool)
    for ix, iy in quads:
        x0, x1 = BOX[0] + ix * hx, BOX[0] + (ix + 1) * hx
        y0, y1 = BOX[2] + iy * hy, BOX[2] + (iy + 1) * hy
        hit |= (
            (centroid[:, 0] > x0) & (centroid[:, 0] < x1)
            & (centroid[:, 1] > y0) & (centroid[:, 1] < y1)
        )
    return np.flatnonzero(hit)


def _draw_on(ax, mesh, *, highlight_quads=(), mirror_line=False):
    """按共用视觉约定把一张网格示意图画到指定 axes 上.

    图内不写标题, 剖分规则由文档正文的表格承担。

    Parameters
    ----------
    ax : Axes
        目标坐标轴。
    mesh : TriangleMesh
        待绘制的三角网格。
    highlight_quads : iterable of tuple, optional
        需要标蓝底色的四边形 ``(ix, iy)`` 下标。
    mirror_line : bool, optional
        是否画竖直镜像中缝。
    """
    node = bm.to_numpy(mesh.entity("node"))
    cell = bm.to_numpy(mesh.entity("cell"))

    if highlight_quads:
        # 只把命中的三角形画成多边形; tripcolor 走 colormap 无法表达"其余透明"
        picked = _highlighted_cells(node, cell, highlight_quads)
        ax.add_collection(PolyCollection(
            [node[cell[c]] for c in picked],
            facecolors=HIGHLIGHT_COLOR, edgecolors="none", zorder=0,
        ))

    ax.triplot(node[:, 0], node[:, 1], cell, color=EDGE_COLOR, linewidth=EDGE_WIDTH)

    if mirror_line:
        xmid = 0.5 * (BOX[0] + BOX[1])
        pad = 0.08 * (BOX[3] - BOX[2])
        ax.plot(
            [xmid, xmid], [BOX[2] - pad, BOX[3] + pad],
            color=MIRROR_COLOR, linestyle="--", linewidth=1.2, zorder=3,
        )

    for x, y, dx, dy, ha, va in _corner_label_offsets(BOX):
        ax.plot(x, y, "o", color=CORNER_COLOR, markersize=8, zorder=4)
        ax.text(
            x + dx, y + dy, "2 tri",
            color=CORNER_COLOR, fontsize=11, fontweight="bold",
            ha=ha, va=va,
        )

    ax.set_aspect("equal")
    ax.margins(0.14)
    ax.axis("off")


def _draw(mesh, **kwargs):
    """单独成图地绘制一张网格示意图.

    Returns
    -------
    Figure
        绘制完成的 matplotlib 图对象。
    """
    fig, ax = plt.subplots(figsize=(7.7, 5.4), dpi=100)
    _draw_on(ax, mesh, **kwargs)
    fig.tight_layout()
    return fig


def _checkerboard_panel():
    """返回棋盘格剖分的 (网格, 绘制选项)."""
    mesh = create_huzhang_checkerboard_mesh(box=BOX, nx=NX, ny=NY)
    return mesh, {}


def _symmetric_panel():
    """返回镜像对称单向对角剖分的 (网格, 绘制选项)."""
    mesh = create_huzhang_symmetric_single_diagonal_mesh(box=BOX, nx=NX, ny=NY)
    opts = dict(
        highlight_quads=((0, NY - 1), (NX - 1, NY - 1)),
        mirror_line=True,
    )
    return mesh, opts


def build_checkerboard():
    """单独绘制棋盘格交替对角剖分."""
    mesh, opts = _checkerboard_panel()
    return _draw(mesh, **opts)


def build_symmetric():
    """单独绘制镜像对称单向对角剖分."""
    mesh, opts = _symmetric_panel()
    return _draw(mesh, **opts)


def build_both():
    """把两种剖分并排画在同一张图上, 供文档单行展示."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=140)
    for ax, panel in zip(axes, (_checkerboard_panel, _symmetric_panel)):
        mesh, opts = panel()
        _draw_on(ax, mesh, **opts)
    fig.tight_layout()
    return fig


FIGURES = {
    "checkerboard": (build_checkerboard, "checkerboard-mesh-6x4.png"),
    "symmetric": (build_symmetric, "single-diagonal-symmetric-mesh-6x4.png"),
    "both": (build_both, "corner-relaxation-meshes.png"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mesh", choices=sorted(FIGURES), required=True,
        help="要绘制的剖分方式.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="输出路径; 缺省写入 docs/fem/assets/ 下的约定文件名.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="允许覆盖已存在的图片 (插图不在 git 跟踪内, 覆盖不可撤销).",
    )
    args = parser.parse_args()

    bm.set_backend("numpy")
    builder, default_name = FIGURES[args.mesh]
    out = args.out or (ASSETS_DIR / default_name)

    if out.exists() and not args.force:
        parser.error(f"{out} 已存在; 确认要覆盖再加 --force.")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig = builder()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
