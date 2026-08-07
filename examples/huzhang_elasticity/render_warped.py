"""纯 VTK 离屏渲染 — Warp By Vector 可视化胡张元位移场.

直接在 WSL 中运行 (不需要 ParaView GUI):
    python examples/huzhang_elasticity/render_warped.py

前置: 先用 concentrated_load_demo 导出最密层的 VTU 文件:

    python examples/huzhang_elasticity/concentrated_load_demo.py --levels 5 --save-vtu

默认加载 ``mixed-sinusoidal_p3_tri_32x32.vtu`` (即上面的命令产物). 胡张元
的位移空间是不连续 Lagrange, ``concentrated_load_demo`` 导出前已把自由度做
跨单元平均插值到网格节点, 因此这里可以直接变形网格.

渲染管线 (加载 VTU、变形着色、离屏截图) 集中在
:mod:`soptx.visualization.vtk_render`, 本文件只保留制造解问题的配置与入口.

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
    / "mixed-sinusoidal_p3_tri_32x32.vtu"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
OUTPUT_NAME = "huzhang_warped.png"
WARP_SCALE = 1.0

# 制造解域是单位正方形 (0,1)x(0,1), 相机参数按它摆放
DOMAIN_CENTER = (0.5, 0.5)
PARALLEL_SCALE = 0.6


def main() -> int:
    if not VTU_PATH.exists():
        print(f"VTU 文件不存在: {VTU_PATH}", file=sys.stderr)
        print(
            "请先运行: python examples/huzhang_elasticity/"
            "concentrated_load_demo.py --levels 5 --save-vtu",
            file=sys.stderr,
        )
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
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
