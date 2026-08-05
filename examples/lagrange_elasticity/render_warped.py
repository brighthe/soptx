"""纯 VTK 离屏渲染 — Warp By Vector 可视化位移场.

直接在 WSL 中运行 (不需要 ParaView GUI):
    python examples/lagrange_elasticity/render_warped.py

依赖: vtk (已随 fealpy 安装)
"""

import sys
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkWarpVector
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkIOImage import vtkPNGWriter

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
WARP_SCALE = 1.0


def load_vtu(filepath: Path):
    """加载 VTU 文件, 返回 unstructured grid."""
    path_str = str(filepath)
    if path_str.endswith(".vtu"):
        reader = vtkXMLUnstructuredGridReader()
    else:
        reader = vtkUnstructuredGridReader()
    reader.SetFileName(path_str)
    reader.Update()
    return reader.GetOutput()


def create_warped_actor(grid, warp_scale: float = 1.0):
    """对网格施加位移场变形, 返回 actor."""
    # Warp
    warp = vtkWarpVector()
    warp.SetInputData(grid)
    warp.SetScaleFactor(warp_scale)
    warp.Update()

    # Mapper: 按位移幅值着色
    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(warp.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("u_mag")
    mapper.SetScalarVisibility(True)
    mapper.SetScalarRange(
        grid.GetPointData().GetArray("u_mag").GetRange()
    )

    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def render_and_save(actor, output_path: Path, resolution=(3200, 1800)):
    """离屏渲染并保存 PNG."""
    colors = vtkNamedColors()

    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("White"))
    renderer.AddActor(actor)

    # 色条
    scalar_bar = vtkScalarBarActor()
    scalar_bar.SetLookupTable(actor.GetMapper().GetLookupTable())
    scalar_bar.SetTitle("|u|")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.6)
    scalar_bar.SetPosition(0.88, 0.2)
    renderer.AddActor2D(scalar_bar)

    # 2D 平行投影 (XY 平面)
    renderer.GetActiveCamera().ParallelProjectionOn()
    renderer.ResetCamera()

    cam = renderer.GetActiveCamera()
    cam.SetPosition(30, 10, 100)
    cam.SetFocalPoint(30, 10, 0)
    cam.SetViewUp(0, 1, 0)
    cam.SetParallelScale(30)  # 调整以适应 60×20 区域

    render_win = vtkRenderWindow()
    render_win.SetOffScreenRendering(1)
    render_win.SetSize(*resolution)
    render_win.AddRenderer(renderer)
    render_win.Render()

    # 截图
    w2i = vtkWindowToImageFilter()
    w2i.SetInput(render_win)
    w2i.Update()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    print(f"截图已保存: {output_path}  ({resolution[0]}×{resolution[1]})")


def main() -> int:
    if not VTU_PATH.exists():
        print(f"VTU 文件不存在: {VTU_PATH}", file=sys.stderr)
        print("请先运行: python examples/lagrange_elasticity/concentrated_load_demo.py --save-vtu", file=sys.stderr)
        return 1

    print(f"加载: {VTU_PATH}")
    grid = load_vtu(VTU_PATH)
    print(f"  节点: {grid.GetNumberOfPoints()}, 单元: {grid.GetNumberOfCells()}")

    arrays = [grid.GetPointData().GetArrayName(i) for i in range(grid.GetPointData().GetNumberOfArrays())]
    print(f"  节点场: {arrays}")
    if "u_mag" in arrays:
        r = grid.GetPointData().GetArray("u_mag").GetRange()
        print(f"  u_mag 范围: [{r[0]:.6e}, {r[1]:.6e}]")

    actor = create_warped_actor(grid, WARP_SCALE)
    output_path = OUTPUT_DIR / "mbb_beam_warped.png"
    render_and_save(actor, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
