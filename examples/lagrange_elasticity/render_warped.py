"""纯 VTK 离屏渲染 — Warp By Vector 可视化位移场.

直接在 WSL 中运行 (不需要 ParaView GUI):
    python examples/lagrange_elasticity/render_warped.py

渲染管线 (加载 VTU、变形着色、离屏截图) 集中在
:mod:`soptx.visualization.vtk_render`, 本文件只保留 MBB 梁问题的配置与入口.

依赖: vtk (已随 fealpy 安装)
"""

from pathlib import Path
import sys

from soptx.visualization.vtk_render import (
    create_warped_actor,
    load_vtu,
    print_grid_summary,
    render_and_save,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
VTU_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "vtu"
    / "mbb-half_p1_quad_240x80.vtu"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
OUTPUT_NAME = "mbb_beam_warped.png"
WARP_SCALE = 1.0

# MBB 梁问题域 (0,60)x(0,20), 相机按中心与视野半高摆放
DOMAIN_CENTER = (30.0, 10.0)
PARALLEL_SCALE = 30.0


def main() -> int:
    if not VTU_PATH.exists():
        print(f"VTU 文件不存在: {VTU_PATH}", file=sys.stderr)
        print("请先运行: python examples/lagrange_elasticity/concentrated_load_demo.py --save-vtu", file=sys.stderr)
        return 1

    print(f"加载: {VTU_PATH}")
    grid = load_vtu(VTU_PATH)
    print_grid_summary(grid)

    actor = create_warped_actor(grid, WARP_SCALE)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    render_and_save(
        actor,
        output_path,
        camera_center=DOMAIN_CENTER,
        parallel_scale=PARALLEL_SCALE,
        resolution=(3200, 1800),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
