"""
SOPTX 工程基准算例物理示意图生成脚本
========================================
自动生成高品质 (300 DPI) 出版级结构物理工程示意图：
  1. HalfMBBBeamRight2d (docs/problems/images/mbb-beam-half-domain.png)
  2. FullMBBBeam3d (docs/problems/images/mbb-beam-3d-half-domain.png)

作者: Liang He & Antigravity Assistant
日期: 2026-08-07
"""

import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compile_tikz_to_png(tex_file: str, output_png: str) -> bool:
    """尝试使用 pdflatex + pdftoppm 直接将 TikZ .tex 源码编译为高分辨率 PNG"""
    pdflatex_bin = shutil.which("pdflatex")
    pdftoppm_bin = shutil.which("pdftoppm")
    if not (pdflatex_bin and pdftoppm_bin):
        return False

    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_basename = os.path.basename(tex_file)
    name_without_ext = os.path.splitext(tex_basename)[0]

    try:
        res = subprocess.run(
            [pdflatex_bin, "-interaction=nonstopmode", tex_basename],
            cwd=tex_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        pdf_path = os.path.join(tex_dir, f"{name_without_ext}.pdf")
        if res.returncode != 0 or not os.path.exists(pdf_path):
            print(f"⚠️ pdflatex 编译 {tex_basename} 失败, 将使用 matplotlib 回退生成。")
            return False

        output_prefix = os.path.splitext(output_png)[0]
        res_ppm = subprocess.run(
            [pdftoppm_bin, "-png", "-r", "300", "-singlefile", pdf_path, output_prefix],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res_ppm.returncode == 0 and os.path.exists(output_png):
            print(f"✅ 已直接从 TikZ 源码 {tex_basename} 编译生成超高清晰度 PNG: {output_png}")
            return True
    except Exception as e:
        print(f"⚠️ TikZ 编译过程出现异常 ({e}), 将使用 matplotlib 回退生成。")

    return False


def generate_2d_mbb_diagram(output_path: str) -> None:
    """生成 2D 对称半 MBB 梁 (HalfMBBBeamRight2d) 的 TikZ 风格学术插图 PNG"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm"
    })

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    Lx, Ly = 60.0, 20.0

    # 1. 主设计域矩形 (TikZ 风格浅蓝与淡紫高质感)
    rect = patches.FancyBboxPatch(
        (0, 0), Lx, Ly,
        boxstyle="square,pad=0.0",
        facecolor='#F0F7FF',
        edgecolor='#1E40AF',
        linewidth=2,
        alpha=0.95,
        zorder=2
    )
    ax.add_patch(rect)

    # 网格背景线 (TikZ 风格细虚线)
    nx, ny = 12, 4
    for x in np.linspace(0, Lx, nx + 1):
        ax.plot([x, x], [0, Ly], color='#CBD5E1', linestyle=':', linewidth=0.8, zorder=3)
    for y in np.linspace(0, Ly, ny + 1):
        ax.plot([0, Lx], [y, y], color='#CBD5E1', linestyle=':', linewidth=0.8, zorder=3)

    # 2. 左边界对称约束 (u_x = 0): TikZ 经典双斜线画法
    ax.plot([0, 0], [0, Ly], color='#DC2626', linestyle='--', linewidth=3.5, zorder=4)
    for y in np.linspace(3, Ly - 3, 5):
        ax.plot([-1.2, -0.2], [y - 0.8, y + 0.8], color='#DC2626', linewidth=1.5, zorder=4)
        ax.plot([-1.8, -0.8], [y - 0.8, y + 0.8], color='#DC2626', linewidth=1.5, zorder=4)

    # 3. 右下角滚轴支座 (u_y = 0): TikZ 经典支撑三角形与滚轴
    triangle = patches.Polygon(
        [[Lx, 0], [Lx - 2.0, -3.5], [Lx + 2.0, -3.5]],
        closed=True, facecolor='#16A34A', edgecolor='#15803D', linewidth=1.5, zorder=4
    )
    ax.add_patch(triangle)
    ax.plot([Lx - 2.5, Lx + 2.5], [-4.2, -4.2], color='#15803D', linewidth=2, zorder=4)
    ax.scatter([Lx - 1.2, Lx + 1.2], [-3.85, -3.85], s=25, color='white', edgecolor='#15803D', linewidth=1.2, zorder=5)

    # 4. 左上角向下集中荷载 P = -1.0 N (TikZ 粗红色箭号)
    arrow_start = (0, Ly + 9.0)
    arrow_end = (0, Ly + 0.5)
    ax.annotate(
        '', xy=arrow_end, xytext=arrow_start,
        arrowprops=dict(facecolor='#B91C1C', edgecolor='#7F1D1D', width=4, headwidth=11, headlength=11),
        zorder=5
    )
    ax.text(0, Ly + 10.5, r'$P = -1.0\text{ N}$', color='#B91C1C', fontsize=13, fontweight='bold', ha='center', zorder=5)

    # 5. 标注说明与文本
    ax.text(Lx / 2, Ly / 2, r'\textbf{Design Domain }$\Omega$', color='#1E3A8A', fontsize=14, ha='center', va='center', zorder=5)
    ax.text(-3.0, Ly / 2, r'\textbf{Symmetry BC }($u_x = 0$)', color='#B91C1C', fontsize=11, ha='right', va='center', rotation=90, zorder=5)
    ax.text(Lx + 1.0, -5.5, r'\textbf{Roller Support }($u_y = 0$)', color='#15803D', fontsize=11, ha='center', zorder=5)

    # 尺寸线标注
    ax.annotate('', xy=(0, -2), xytext=(Lx, -2), arrowprops=dict(arrowstyle='<->', color='#334155', lw=1.2))
    ax.text(Lx / 2, -4.0, f'$L_x = {Lx:.0f}\\text{{ mm}}$', color='#334155', fontsize=11, ha='center')

    ax.annotate('', xy=(Lx + 2, 0), xytext=(Lx + 2, Ly), arrowprops=dict(arrowstyle='<->', color='#334155', lw=1.2))
    ax.text(Lx + 4.0, Ly / 2, f'$L_y = {Ly:.0f}\\text{{ mm}}$', color='#334155', fontsize=11, va='center', rotation=270)

    # 6. 样式调整
    ax.set_xlim(-6, Lx + 8)
    ax.set_ylim(-7, Ly + 13)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("TikZ Rendered: HalfMBBBeamRight2d Boundary Conditions", fontsize=14, pad=10, color='#0F172A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已成功生成 2D TikZ 风格渲染图: {output_path}")


def generate_3d_mbb_diagram(output_path: str) -> None:
    """生成完整 3D MBB 梁 (FullMBBBeam3d) 的 TikZ 风格学术插图 PNG"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm"
    })

    fig = plt.figure(figsize=(9, 5.5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 呈现用的几何比例 (6:1:1 比例对齐 Huang2023 图 5)
    Lx, Ly, Lz = 6.0, 1.0, 1.0

    # 顶点坐标 (X 为长度, Y 为宽度, Z 为高度)
    nodes = np.array([
        [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],       # 底面 z=0
        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]    # 顶面 z=Lz
    ])

    # 体单元六个面
    faces = [
        [nodes[0], nodes[1], nodes[2], nodes[3]], # 底面 z=0
        [nodes[4], nodes[5], nodes[6], nodes[7]], # 顶面 z=Lz
        [nodes[0], nodes[1], nodes[5], nodes[4]], # 前面 y=0
        [nodes[2], nodes[3], nodes[7], nodes[6]], # 后面 y=Ly
        [nodes[0], nodes[3], nodes[7], nodes[4]], # 左面 x=0
        [nodes[1], nodes[2], nodes[6], nodes[5]]  # 右面 x=Lx
    ]

    # 1. 绘制半透明 TikZ 风格实体框
    poly3d = Poly3DCollection(faces, facecolors='#F0F7FF', linewidths=1.5, edgecolors='#1E40AF', alpha=0.4)
    ax.add_collection3d(poly3d)

    # 2. 左底线 (x=0, z=0) 铰支座 (u_x=0, u_y=0) 高亮
    ax.plot([0, 0], [0, Ly], [0, 0], color='#DC2626', linewidth=4.5, zorder=6, label='Pin Support Line ($u_x=0, u_y=0$)')

    # 3. 右底线 (x=Lx, z=0) 滚轴支座 (u_y=0) 高亮
    ax.plot([Lx, Lx], [0, Ly], [0, 0], color='#16A34A', linewidth=4.5, zorder=6, label='Roller Support Line ($u_y=0$)')

    # 3.5. uz=0 底面中心线 (代码: y=0, z=Lz/2, 沿 x 长度) —— 防止刚体运动
    #      数据轴 Y=宽度=Ly/2, Z=高度=0
    ax.plot([0, Lx], [Ly / 2, Ly / 2], [0, 0], color='#1D4ED8', linewidth=2.2,
            linestyle='--', zorder=6, label='Centerline $u_z=0$')

    # 4. 顶面中心点 (代码: x=Lx/2, y=Ly, z=Lz/2) 集中载荷 P = -1.0 N, 沿 -y 方向 (即 -Z)
    #    数据轴 Y=宽度=Ly/2, Z=高度=顶面
    top_center = [Lx / 2, Ly / 2, Lz]
    ax.quiver(top_center[0], top_center[1], top_center[2] + 0.65, 0, 0, -0.6, color='#B91C1C', lw=4.0, arrow_length_ratio=0.35, zorder=8)
    ax.scatter([top_center[0]], [top_center[1]], [top_center[2]], color='#B91C1C', s=55, zorder=9)
    ax.text(top_center[0], top_center[1], top_center[2] + 0.75, r'$P = -1.0\text{ N}$ (Top Center Load)', color='#B91C1C', fontsize=11, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEE2E2', edgecolor='#B91C1C', alpha=0.95))

    # 5. 标注说明文字 (附带清晰白色遮罩框，防止网格重叠遮挡)
    ax.text(-0.5, 0, 0, 'Pin Support Line\n($u_x=0, u_y=0$)', color='#B91C1C', fontsize=9.5, fontweight='bold', ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#B91C1C', alpha=0.95))
    ax.text(Lx + 0.5, 0, 0, 'Roller Support Line\n($u_y=0$)', color='#15803D', fontsize=9.5, fontweight='bold', ha='left', va='top', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#15803D', alpha=0.95))
    ax.text(Lx / 2, Ly / 2, Lz / 2, r'\textbf{Full 3D MBB Beam Domain }$\Omega$', color='#1E3A8A', fontsize=11, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.25', facecolor='#F0F9FF', edgecolor='#1E40AF', alpha=0.9))

    # 尺寸标注说明 (标签按代码坐标系标注, 与 TikZ 图及 FullMBBBeam3d 一致):
    #   matplotlib 的 Y 轴(渲染为纵深) = 代码 z (宽度); Z 轴(渲染为竖直) = 代码 y (高度)。
    ax.set_xlabel(r'$X$ (Length $L_x = 120\text{ mm} \;\vert\; 6$)', fontsize=9.5, labelpad=6)
    ax.set_ylabel(r'$Z$ (Width $L_z = 20\text{ mm} \;\vert\; 1$)', fontsize=9.5, labelpad=6)
    ax.set_zlabel(r'$Y$ (Height $L_y = 20\text{ mm} \;\vert\; 1$)', fontsize=9.5, labelpad=6)

    ax.set_title("TikZ Rendered: FullMBBBeam3d Boundary Conditions (Huang2023 Fig 5)", fontsize=13, pad=14, color='#0F172A')

    # 移除三维背景灰色网格面，呈现超清 TikZ 矢量美感
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.set_box_aspect([3.5, 1.2, 1.2])
    ax.view_init(elev=22, azim=-55)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已成功生成 3D TikZ 风格渲染图: {output_path}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "images")
    tikz_dir = os.path.join(base_dir, "tikz")
    os.makedirs(img_dir, exist_ok=True)

    img_2d_path = os.path.join(img_dir, "mbb-beam-half-domain.png")
    img_3d_path = os.path.join(img_dir, "mbb-beam-3d-half-domain.png")

    tex_2d = os.path.join(tikz_dir, "mbb_2d_half_beam.tex")
    tex_3d = os.path.join(tikz_dir, "mbb_3d_full_beam.tex")

    print("🚀 正在从 TikZ 源码生成高分辨率 PNG 物理示意图...")
    if not compile_tikz_to_png(tex_2d, img_2d_path):
        generate_2d_mbb_diagram(img_2d_path)

    if not compile_tikz_to_png(tex_3d, img_3d_path):
        generate_3d_mbb_diagram(img_3d_path)


if __name__ == "__main__":
    main()
