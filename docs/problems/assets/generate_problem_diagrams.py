"""
SOPTX 工程基准算例物理示意图生成脚本
========================================
将 tikz/ 下的 TikZ 源码批量编译为高分辨率 (300 DPI) PNG：
按 engineering-benchmarks.md 的 A/B/C/D 分组排列, 每个工程基准算例一张:

  A 组 集中力      HalfMBBBeamPoint2d, FullMBBBeamPoint2d, FullMBBBeamPoint3d,
                   CantileverCornerPoint2d
  B 组 线载荷      HalfMBBBeamLine3d, FullMBBBeamLine3d, CantileverBottomEdgeLine3d
  C 组 满布牵引    BearingTraction2d, BridgeTraction2d
  D 组 局部牵引    CantileverMiddlePatch2d, HalfFixedFixedBeamPatch2d, PortalFramePatch2d

依赖: pdflatex + pdftoppm (缺任一则直接报错退出, 无回退渲染)。
tikz/ 下的 .tex 源码可直接 \\input{...} 进论文复用。

作者: Liang He & Antigravity Assistant
日期: 2026-08-07
"""

import os
import shutil
import subprocess
import sys

# (tex 源码, 输出 png)
DIAGRAMS = [
    # A 组: 集中力
    ("half_mbb_beam_point_2d.tex", "half-mbb-beam-point-2d.png"),
    ("full_mbb_beam_point_2d.tex", "full-mbb-beam-point-2d.png"),
    ("full_mbb_beam_point_3d.tex", "full-mbb-beam-point-3d.png"),
    ("cantilever_corner_point_2d.tex", "cantilever-corner-point-2d.png"),
    # B 组: 线载荷
    ("half_mbb_beam_line_3d.tex", "half-mbb-beam-line-3d.png"),
    ("full_mbb_beam_line_3d.tex", "full-mbb-beam-line-3d.png"),
    ("cantilever_bottom_edge_line_3d.tex", "cantilever-bottom-edge-line-3d.png"),
    # C 组: 满布边界牵引
    ("bearing_traction_2d.tex", "bearing-traction-2d.png"),
    ("bridge_traction_2d.tex", "bridge-traction-2d.png"),
    # D 组: 局部边界牵引
    ("cantilever_middle_point_2d.tex", "cantilever-middle-point-2d.png"),
    ("cantilever_middle_patch_2d.tex", "cantilever-middle-patch-2d.png"),
    ("half_fixed_fixed_beam_patch_2d.tex", "half-fixed-fixed-beam-patch-2d.png"),
    ("portal_frame_patch_2d.tex", "portal-frame-patch-2d.png"),
    # 制造解
    ("mixed_boundary_sinusoidal_elasticity_2d.tex", "mixed-boundary-sinusoidal-elasticity-2d.png"),
]


def compile_tikz_to_png(tex_file: str, output_png: str) -> bool:
    """使用 pdflatex + pdftoppm 将 TikZ .tex 源码编译为高分辨率 PNG"""
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_basename = os.path.basename(tex_file)
    name_without_ext = os.path.splitext(tex_basename)[0]

    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_basename],
        cwd=tex_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pdf_path = os.path.join(tex_dir, f"{name_without_ext}.pdf")
    if res.returncode != 0 or not os.path.exists(pdf_path):
        print(f"❌ pdflatex 编译 {tex_basename} 失败, 详见 assets/tikz/{name_without_ext}.log")
        return False

    output_prefix = os.path.splitext(output_png)[0]
    res_ppm = subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_path, output_prefix],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res_ppm.returncode != 0 or not os.path.exists(output_png):
        print(f"❌ pdftoppm 转换 {name_without_ext}.pdf 失败。")
        return False

    print(f"✅ {tex_basename} -> {output_png}")
    return True


def main() -> int:
    if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
        print("❌ 环境缺少 pdflatex 或 pdftoppm, 无法编译 TikZ 示意图。")
        return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "images")
    tikz_dir = os.path.join(base_dir, "tikz")
    os.makedirs(img_dir, exist_ok=True)

    print("🚀 正在从 TikZ 源码生成高分辨率 PNG 物理示意图...")
    n_failed = 0
    for tex_name, png_name in DIAGRAMS:
        tex_path = os.path.join(tikz_dir, tex_name)
        png_path = os.path.join(img_dir, png_name)
        if not compile_tikz_to_png(tex_path, png_path):
            n_failed += 1

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
