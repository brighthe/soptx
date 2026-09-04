# -*- coding: utf-8 -*-
"""把 SIMP 最终密度场渲染为灰阶 3D 构型图（申请书画图用）.

管线：密度场(阈值 0.5) -> VTK Marching Cubes 等值面 -> 三角面片 ->
matplotlib ``Poly3DCollection`` 灰阶渲染。只借 VTK 做几何提取，不做 VTK 渲染，
因此无头（无 X/GL）环境也能稳定出图；灰阶印刷风格与申请书其余图片一致。

用法:
  python examples/topopt_platform/render_topology.py \
      --density outputs/cal729/density_final.npy --nx 120 --ny 60 --nz 30 \
      --out /tmp/topo_preview.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import vtk
from vtk.util.numpy_support import numpy_to_vtk

SURFACE = "#ffffff"
INK = "#0b0b0b"
FACE = "#a6a6a6"
EDGE = "#6b6b6b"

VIEW_ELEV = 20.0
VIEW_AZIM = -62.0

FONT_CANDIDATES = [
    ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyhbd.ttc"),
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"),
    ("/mnt/c/Windows/Fonts/simhei.ttf", None),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
     "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
]


def setup_font():
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    for regular, bold in FONT_CANDIDATES:
        if not os.path.exists(regular):
            continue
        try:
            fm.fontManager.addfont(regular)
            name = fm.FontProperties(fname=regular).get_name()
        except Exception:
            continue
        if bold and os.path.exists(bold):
            try:
                fm.fontManager.addfont(bold)
            except Exception:
                pass
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        print(f"[font] 使用中文字体: {name}")
        return
    print("[font] 警告: 未找到中文字体, 中文将显示为方块")


def _trim_white(path: str, pad: int = 8) -> None:
    """裁掉图片四周的纯白边.

    ``mpl_toolkits.mplot3d`` 按 ``box_aspect`` 的外接立方体预留画布, 细长设计域
    (如 15:5:1) 会留下大片空白; ``bbox_inches="tight"`` 只贴合坐标轴而非实际几何,
    故按像素找出非白包围盒再裁一次. 图片被就地覆盖.

    单张图按自身内容裁四边. 初始构型与优化构型要并排对比时改用 :func:`trim_pair`,
    两张共用同一裁剪框, 否则优化构型因有孔洞而比设计域窄一截, 并排会错位.

    参数:
        path: 待裁剪的图片路径.
        pad: 裁剪后四周保留的空白像素数.
    """
    img = plt.imread(path)
    rgb = img[..., :3] if img.ndim == 3 else img
    if rgb.dtype.kind == "f":
        ink = (rgb < 0.99).any(axis=-1)
    else:
        ink = (rgb < 252).any(axis=-1)
    if not ink.any():
        return
    box = _ink_box(img, pad)
    _apply_box(path, img, box)


def _ink_box(img: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    """图片中非白像素的包围盒, 四周外扩 ``pad`` 像素 -> ``(r0, r1, c0, c1)``."""
    rgb = img[..., :3] if img.ndim == 3 else img
    if rgb.dtype.kind == "f":
        ink = (rgb < 0.99).any(axis=-1)
    else:
        ink = (rgb < 252).any(axis=-1)
    if not ink.any():
        return 0, img.shape[0], 0, img.shape[1]
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    return (max(int(rows[0]) - pad, 0), min(int(rows[-1]) + 1 + pad, img.shape[0]),
            max(int(cols[0]) - pad, 0), min(int(cols[-1]) + 1 + pad, img.shape[1]))


def _apply_box(path: str, img: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, r1, c0, c1 = box
    plt.imsave(path, img[r0:r1, c0:c1])
    print(f"[trim] {Path(path).name}: {img.shape[1]}x{img.shape[0]} "
          f"-> {c1 - c0}x{r1 - r0}")


def trim_pair(path_a: str, path_b: str, pad: int = 8) -> None:
    """按两张图非白包围盒的并集同时裁剪, 使二者裁后尺寸完全一致.

    并排放进同一张图的初始构型与优化构型必须共用裁剪框: 各自按内容裁会得到不同宽
    度, ``imshow`` 保持长宽比后两个面板的构型大小与基线都对不齐.

    参数:
        path_a: 第一张图片路径, 就地覆盖.
        path_b: 第二张图片路径, 就地覆盖.
        pad: 裁剪后四周保留的空白像素数.
    """
    img_a, img_b = plt.imread(path_a), plt.imread(path_b)
    if img_a.shape[:2] != img_b.shape[:2]:
        raise ValueError(f"两张底图画布尺寸不一致, 无法共用裁剪框: "
                         f"{img_a.shape[:2]} vs {img_b.shape[:2]}")
    ba, bb = _ink_box(img_a, pad), _ink_box(img_b, pad)
    box = (min(ba[0], bb[0]), max(ba[1], bb[1]),
           min(ba[2], bb[2]), max(ba[3], bb[3]))
    _apply_box(path_a, img_a, box)
    _apply_box(path_b, img_b, box)


def density_to_triangles(density: np.ndarray, Lx: float, Ly: float, Lz: float,
                         threshold: float = 0.5):
    """Marching Cubes 提取等值面三角面片 -> (verts(N,3), faces(M,3)).

    密度是单元中心量, 而 ``vtkImageData`` 的标量按节点解释, 故先在六个面各补一层
    零密度: 补零后等值面在设计域边界闭合(否则构型的外表面被切开成敞口壳), 且节点
    坐标原点后移半个单元, 使单元中心与节点严格对齐, 等值面范围回到 ``[0, L]``.
    """
    nx, ny, nz = density.shape
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
    padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
    padded[1:-1, 1:-1, 1:-1] = density

    img = vtk.vtkImageData()
    img.SetDimensions(nx + 2, ny + 2, nz + 2)
    img.SetSpacing(hx, hy, hz)
    img.SetOrigin(-0.5 * hx, -0.5 * hy, -0.5 * hz)
    arr = numpy_to_vtk(padded.ravel(order="F"), deep=True)  # vtk 用 Fortran 序
    arr.SetName("rho")
    img.GetPointData().SetScalars(arr)

    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(img)
    mc.SetValue(0, threshold)
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.Update()
    poly = mc.GetOutput()

    verts = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
    polys = poly.GetPolys()
    cell_arr = polys.GetData()
    n_tri = polys.GetNumberOfCells()
    tri = np.empty((n_tri, 3), dtype=np.int64)
    off = 0
    for i in range(n_tri):
        conn = cell_arr.GetValue(off)  # 单元顶点数（应为 3）
        tri[i] = [cell_arr.GetValue(off + 1 + j) for j in range(3)]
        off += conn + 1
    return verts, tri


def _to_plot_axes(pts: np.ndarray) -> np.ndarray:
    """物理坐标 (x, y, z) -> 绘图坐标 (x, z, y).

    梁高 ``y`` 同时也是载荷方向, 必须落在 ``mplot3d`` 的竖直轴上, 否则悬臂梁被画
    成一块平放的板, 载荷箭头指向进深方向而无法辨认弯曲方向. 厚度 ``z`` 改走进深轴,
    取向与博士论文图 3.6 一致.

    参数:
        pts: 物理坐标, 形状 ``(..., 3)``.

    返回:
        绘图坐标, 形状 ``(..., 3)``.
    """
    return pts[..., [0, 2, 1]]


def _shade_faces(verts: np.ndarray, tri: np.ndarray, *,
                 lo: float = 0.28, hi: float = 0.84) -> np.ndarray:
    """按面法向做 Lambert 灰阶着色 -> ``(n_tri, 4)`` RGBA.

    不用 ``Poly3DCollection(shade=True)``: 它把 Lambert 系数直接乘到基色上, 背光
    面趋近纯黑, 该取向下细长桁架会整体糊成死黑, 提亮基色也救不回来. 这里把灰度显
    式压进 ``[lo, hi]``, 保证灰阶印刷时最暗的面仍与背景和相邻杆件可分辨.

    等值面法向朝内朝外并存, 故取 ``|n·L|`` 双面着色, 避免半数面片突然发黑.

    参数:
        verts: 等值面顶点物理坐标, 形状 ``(n_vert, 3)``.
        tri: 三角面片顶点索引, 形状 ``(n_tri, 3)``.
        lo: 最暗面的灰度.
        hi: 最亮面的灰度.

    返回:
        逐面 RGBA 颜色, 形状 ``(n_tri, 4)``.
    """
    v = verts[tri]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30

    el, az = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    # 光源跟随相机: 先在绘图坐标系求视线方向, 再换回物理坐标系与法向配对.
    cam_plot = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    light = cam_plot[[0, 2, 1]]
    light /= np.linalg.norm(light)

    g = lo + (hi - lo) * np.abs(n @ light)
    return np.stack([g, g, g, np.ones_like(g)], axis=1)


def _annotate(ax, Lx, Ly, Lz):
    """载荷箭头与固支标注（论文算例 3.3：x=0 固支，右端面底边 -y 均布线载荷）。"""
    # 绘图坐标下载荷沿竖直轴向下, 起点取自由端底边的厚度中面.
    ax.quiver(Lx, Lz / 2, 0.0, 0, 0, -0.26 * Ly, color=INK,
              arrow_length_ratio=0.30, linewidth=2.2)
    ax.text(Lx + 0.035 * Lx, Lz / 2, -0.20 * Ly, "$F$", fontsize=13, color=INK,
            ha="left", va="center")
    ax.text(0.0, Lz / 2, Ly * 1.10, "固支端", fontsize=12, color=INK,
            ha="center", va="bottom")


def _setup_axes(ax, Lx, Ly, Lz):
    ax.set_box_aspect((Lx, Lz, Ly))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Lz)
    # 竖直轴下探到域外, 给自由端的载荷箭头留出位置, 否则箭头被 clip 掉.
    ax.set_zlim(-0.30 * Ly, Ly)
    ax.set_axis_off()


def render_domain_box(Lx: float, Ly: float, Lz: float, out: str, *,
                      title: str = "", figsize=(6.0, 3.2), trim: bool = True) -> None:
    """初始构型：设计域长方体（均匀材料分布的初始设计）. """
    verts = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                      [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    fig = plt.figure(figsize=figsize, dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    fc = np.full((6, 4), [0.78, 0.78, 0.78, 1.0])
    ec = np.full((6, 4), [0.45, 0.45, 0.45, 1.0])
    pc = Poly3DCollection([_to_plot_axes(verts[f]) for f in faces], facecolors=fc,
                          edgecolors=ec, linewidth=1.2, shade=False)
    ax.add_collection3d(pc)
    _setup_axes(ax, Lx, Ly, Lz)
    _annotate(ax, Lx, Ly, Lz)
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=-2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.08)
    plt.close(fig)
    if trim:
        _trim_white(out)
    print(f"[out] {out}")


def render(density: np.ndarray, Lx: float, Ly: float, Lz: float, out: str,
           *, threshold: float = 0.5, title: str = "", figsize=(7.0, 3.6),
           trim: bool = True) -> None:
    verts, tri = density_to_triangles(density, Lx, Ly, Lz, threshold)

    fig = plt.figure(figsize=figsize, dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    fc = _shade_faces(verts, tri)
    # 百万单元网格的等值面有数十万个三角, 逐面描边会在表面叠成网点噪声, 面数大时
    # 只靠法向明暗表达形体.
    dense = len(tri) > 20000
    ec = "none" if dense else np.full((len(tri), 4), [0.42, 0.42, 0.42, 1.0])
    pc = Poly3DCollection(_to_plot_axes(verts[tri]), facecolors=fc, edgecolors=ec,
                          linewidth=0.0 if dense else 0.1, shade=False)
    ax.add_collection3d(pc)

    _setup_axes(ax, Lx, Ly, Lz)
    _annotate(ax, Lx, Ly, Lz)
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=-2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE,
                pad_inches=0.08)
    plt.close(fig)
    if trim:
        _trim_white(out)
    print(f"[out] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SIMP 密度场灰阶 3D 渲染")
    parser.add_argument("--density", type=str, default=None,
                        help="密度场 .npy（--box 时无需）")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--Lx", type=float, default=2.0)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--Lz", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--figsize", type=str, default="7.0x3.6",
                        help="matplotlib 画布尺寸 WxH")
    parser.add_argument("--box", action="store_true",
                        help="渲染初始构型(设计域长方体)，忽略 --density")
    parser.add_argument("--pair", type=str, default=None,
                        help="成对模式: 本参数为初始构型输出路径, --out 为优化构型"
                             "输出路径; 两图共用裁剪框, 可直接并排对比")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    w, h = (float(v) for v in args.figsize.split("x"))
    setup_font()
    if args.pair:
        density = np.load(args.density)
        assert density.shape == (args.nx, args.ny, args.nz), (
            f"密度场形状 {density.shape} 与 --nx/--ny/--nz 不符")
        render_domain_box(args.Lx, args.Ly, args.Lz, args.pair,
                          figsize=(w, h), trim=False)
        render(density, args.Lx, args.Ly, args.Lz, args.out,
               threshold=args.threshold, figsize=(w, h), trim=False)
        trim_pair(args.pair, args.out)
        return 0
    if args.box:
        render_domain_box(args.Lx, args.Ly, args.Lz, args.out,
                          title=args.title, figsize=(w, h))
        return 0
    density = np.load(args.density)
    assert density.shape == (args.nx, args.ny, args.nz), (
        f"密度场形状 {density.shape} 与 --nx/--ny/--nz 不符")
    render(density, args.Lx, args.Ly, args.Lz, args.out,
           threshold=args.threshold, title=args.title, figsize=(w, h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
