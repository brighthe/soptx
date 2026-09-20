# -*- coding: utf-8 -*-
"""四族网格的剖分示意图.

画 results_analysis.md 第 2.2 节那四条加密序列各自的最粗一档: quad / tri 取
``n = 4``, hex 取 ``n = 4``, tet 取 ``n = 2``, 与该节表格的起点逐项对齐。加密序列
其余各档按同一规则二分, 画最粗一档即可看清剖分规则。

本脚本不读任何运行产物: 剖分规则是纯几何事实, 由 ``.from_box`` 自身决定, 与是否
跑过收敛链无关。

输出: figure_data/mesh_families.svg 与同名 .png。

运行::

    ~/miniconda3/envs/ihpcm/bin/python experiments/assembly_level_consistency/plot_meshes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

_FIGURE_DIR = Path(__file__).resolve().parent / "figure_data"

# 与 paper_topopt_huzhang/plots/manufactured_mesh.py 同一套配色
_EDGE_COLOR = "#2b2b2b"
_FILL_COLOR = "#e8eef6"

# (标签, 网格类, 最粗一档的 n, 几何维数); n 取值见 results_analysis.md 2.2 节
_FAMILIES = (
    ("quad", QuadrangleMesh, 4, 2),
    ("tri", TriangleMesh, 4, 2),
    ("hex", HexahedronMesh, 4, 3),
    ("tet", TetrahedronMesh, 2, 3),
)


def _build(mesh_cls, n: int, gd: int):
    """按几何维数取 from_box 的参数个数."""
    if gd == 2:
        return mesh_cls.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
    return mesh_cls.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)


def _segments(mesh, gd: int) -> np.ndarray:
    """取网格全部边的端点坐标, 形状 (NE, 2, gd)."""
    node = np.asarray(mesh.entity("node"))[:, :gd]
    edge = np.asarray(mesh.entity("edge"))
    return node[edge]


def _draw_2d(ax, mesh, label: str, n: int) -> None:
    # 网格铺满整个单位正方形, 底色用一块矩形即可, 不必逐单元上色
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0,
                           facecolor=_FILL_COLOR, edgecolor="none", zorder=1))
    ax.add_collection(
        LineCollection(_segments(mesh, 2), colors=_EDGE_COLOR, linewidths=0.8, zorder=2)
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(_caption(label, n, mesh), fontsize=9, pad=6)


def _draw_3d(ax, mesh, label: str, n: int) -> None:
    ax.add_collection3d(
        Line3DCollection(_segments(mesh, 3), colors=_EDGE_COLOR,
                         linewidths=0.5, alpha=0.75)
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title(_caption(label, n, mesh), fontsize=9, pad=0)


def _caption(label: str, n: int, mesh) -> str:
    return f"{label}  $n={n}$\n{mesh.number_of_cells():,d} cells"


def main() -> None:
    fig = plt.figure(figsize=(11.0, 3.1))
    for i, (label, mesh_cls, n, gd) in enumerate(_FAMILIES, start=1):
        mesh = _build(mesh_cls, n, gd)
        if gd == 2:
            ax = fig.add_subplot(1, 4, i)
            _draw_2d(ax, mesh, label, n)
        else:
            ax = fig.add_subplot(1, 4, i, projection="3d")
            _draw_3d(ax, mesh, label, n)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)

    _FIGURE_DIR.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        path = _FIGURE_DIR / f"mesh_families.{suffix}"
        fig.savefig(path, dpi=200, format=suffix)
        print(f"已写出 {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
