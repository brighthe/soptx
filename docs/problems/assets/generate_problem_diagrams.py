"""
SOPTX 工程基准算例物理示意图生成脚本
========================================
将 tikz/ 下的 TikZ 源码批量编译为高分辨率 (300 DPI) PNG：
按 engineering-benchmarks.md 的 A/B/C/D 分组排列, 每个工程基准算例一张:

  A 组 集中力      HalfMBBBeamRight2d, HalfMBBBeamRight3d, FullMBBBeam2d,
                   FullMBBBeam3d, CantileverCorner2d
  B 组 线载荷      CantileverRightBottomEdge3d
  C 组 满布牵引    BearingDevice2d, SimplySupportedBridge2d
  D 组 局部牵引    CantileverMiddle2d, FixedFixedBeamHalfDomain2d

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
    ("mbb_2d_half_beam.tex", "mbb-beam-half-domain.png"),
    ("mbb_3d_half_beam_right.tex", "mbb-beam-3d-right-half.png"),
    ("mbb_2d_full_beam.tex", "mbb-beam-2d-full.png"),
    ("mbb_3d_full_beam.tex", "mbb-beam-3d-half-domain.png"),
    ("cantilever_2d_corner.tex", "cantilever-corner-2d.png"),
    # B 组: 线载荷
    ("cantilever_3d_right_bottom_edge.tex", "cantilever-right-bottom-edge-3d.png"),
    # C 组: 满布边界牵引
    ("bearing_2d_device.tex", "bearing-device-2d.png"),
    ("bridge_2d_simply_supported.tex", "bridge-simply-supported-2d.png"),
    # D 组: 局部边界牵引
    ("cantilever_2d_middle.tex", "cantilever-middle-2d.png"),
    ("fixed_fixed_2d_half_beam.tex", "fixed-fixed-beam-half-domain-2d.png"),
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
