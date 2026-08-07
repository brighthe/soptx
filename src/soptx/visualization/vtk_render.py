"""基于 VTK 的离屏渲染辅助 — Warp By Vector 可视化位移场.

examples 里 lagrange/huzhang 的 ``render_warped.py`` 原为两份重复的 VTK 脚本,
公共渲染管线集中到这里, 各 example 只保留配置常量与入口:

* ``load_vtu``: 读取 VTU (或 VTK) 文件为 unstructured grid;
* ``create_warped_actor``: 按位移场变形网格, 以位移幅值着色;
* ``print_grid_summary``: 打印网格规模与场信息;
* ``render_and_save``: 平行投影离屏渲染并保存 PNG, 相机由调用方配置.

依赖 ``vtk`` (随 fealpy 安装). 本模块只在被显式导入时加载, 不会拖慢
``import soptx``.
"""

from __future__ import annotations

from pathlib import Path

import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkWarpVector
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkIOImage import vtkPNGWriter


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
    """对网格施加位移场变形, 按位移幅值 ``u_mag`` 着色, 返回 actor."""

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


def print_grid_summary(grid) -> None:
    """打印网格规模与可用节点场信息."""

    print(f"  节点: {grid.GetNumberOfPoints()}, 单元: {grid.GetNumberOfCells()}")

    arrays = [
        grid.GetPointData().GetArrayName(i)
        for i in range(grid.GetPointData().GetNumberOfArrays())
    ]
    print(f"  节点场: {arrays}")
    if "u_mag" in arrays:
        r = grid.GetPointData().GetArray("u_mag").GetRange()
        print(f"  u_mag 范围: [{r[0]:.6e}, {r[1]:.6e}]")


def render_and_save(
    actor,
    output_path: Path,
    *,
    camera_center: tuple[float, float] = (0.0, 0.0),
    parallel_scale: float = 1.0,
    view_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    resolution: tuple[int, int] = (1600, 1600),
    background: str = "White",
) -> None:
    """离屏渲染并保存 PNG (XY 平面 2D 平行投影).

    相机对准 ``camera_center``, 视野半高 ``parallel_scale`` —— 调用方按各自
    问题域设置 (如单位正方形域取 center=(0.5, 0.5), scale=0.6).
    """

    colors = vtkNamedColors()

    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d(background))
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
    cam.SetPosition(camera_center[0], camera_center[1], 100)
    cam.SetFocalPoint(camera_center[0], camera_center[1], 0)
    cam.SetViewUp(*view_up)
    cam.SetParallelScale(parallel_scale)

    render_win = vtkRenderWindow()
    render_win.SetOffScreenRendering(1)
    render_win.SetSize(*resolution)
    render_win.AddRenderer(renderer)
    render_win.Render()

    # 截图
    w2i = vtkWindowToImageFilter()
    w2i.SetInput(render_win)
    w2i.Update()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    print(f"截图已保存: {output_path}  ({resolution[0]}×{resolution[1]})")
