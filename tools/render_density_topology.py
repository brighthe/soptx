"""把三维拓扑优化的最终物理密度渲染成实验文档用的拓扑图.

输入为运行目录固化下来的 ``density_final.npy`` (形状 ``(nx, ny, nz)`` 的单元密度, C 序,
与 ``HexahedronMesh.from_box`` 的单元编号一致), 按阈值取出实体单元, 用 VTK 离屏渲染其外表面,
并画出设计域轮廓线. 视角、阈值、尺寸都由参数决定, 不同实验的拓扑图用同一条命令生成,
便于并列比较; 不依赖 ParaView 与手工截图.

Examples
--------
::

    python tools/render_density_topology.py \\
        --density experiments/topopt_simp_fem/results/pytorch_cuda_ea_hex_390x65x65_cg_jacobi/density_final.npy \\
        --output experiments/topopt_simp_fem/assets/ea_hex_390x65x65_final_topology.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数.

    Returns
    -------
    argparse.Namespace
        含 ``density``、``output``、``threshold``、``spacing``、``size``、``view``、``zoom``、``color``.
    """
    parser = argparse.ArgumentParser(description='把最终物理密度渲染成拓扑图 (VTK 离屏)')
    parser.add_argument('--density', type=Path, required=True, help='density_final.npy 路径, 形状 (nx, ny, nz)')
    parser.add_argument('--output', type=Path, required=True, help='输出 PNG 路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='实体单元阈值, 密度不小于该值的单元参与渲染 (默认 0.5)')
    parser.add_argument('--spacing', type=float, default=1.0, help='单元边长 (默认 1.0, 只影响轮廓与密度网格的比例)')
    parser.add_argument('--size', type=int, nargs=2, default=(2000, 1000), metavar=('W', 'H'), help='图像像素尺寸 (默认 2000 1000)')
    parser.add_argument('--view', type=float, nargs=3, default=(-0.55, 0.45, 1.0), metavar=('X', 'Y', 'Z'),
                        help='相机相对设计域中心的方向向量, y 向上 (默认 -0.55 0.45 1.0, 自左前上方俯视)')
    parser.add_argument('--zoom', type=float, default=1.9, help='相机缩放倍数 (默认 1.9)')
    parser.add_argument('--color', type=float, nargs=3, default=(0.35, 0.50, 0.72), metavar=('R', 'G', 'B'),
                        help='实体表面颜色, 0 到 1 的 RGB (默认钢蓝)')
    return parser.parse_args(argv)


def build_solid_surface(density: np.ndarray, threshold: float, spacing: float) -> tuple[vtk.vtkPolyData, vtk.vtkImageData]:
    """由单元密度构造实体外表面与设计域体数据.

    Parameters
    ----------
    density : numpy.ndarray
        形状 ``(nx, ny, nz)`` 的单元密度.
    threshold : float
        实体单元阈值.
    spacing : float
        单元边长.

    Returns
    -------
    surface : vtk.vtkPolyData
        阈值以上单元集合的外表面.
    image : vtk.vtkImageData
        设计域体数据, 供轮廓线使用.
    """
    nx, ny, nz = density.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx + 1, ny + 1, nz + 1)
    image.SetSpacing(spacing, spacing, spacing)
    image.SetOrigin(0.0, 0.0, 0.0)
    # VTK 单元按 x 最快排列, numpy 数组是 C 序 (z 最快), 需转置后展平
    values = np.ascontiguousarray(np.transpose(density, (2, 1, 0))).ravel()
    array = numpy_support.numpy_to_vtk(values.astype(np.float64), deep=True)
    array.SetName('density')
    image.GetCellData().SetScalars(array)

    selector = vtk.vtkThreshold()
    selector.SetInputData(image)
    selector.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'density')
    selector.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
    selector.SetUpperThreshold(threshold)
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(selector.GetOutputPort())
    surface_filter.Update()
    return surface_filter.GetOutput(), image


def render(surface: vtk.vtkPolyData, image: vtk.vtkImageData, args: argparse.Namespace) -> None:
    """离屏渲染并写出 PNG.

    Parameters
    ----------
    surface : vtk.vtkPolyData
        实体外表面.
    image : vtk.vtkImageData
        设计域体数据.
    args : argparse.Namespace
        命令行参数.
    """
    solid_mapper = vtk.vtkPolyDataMapper()
    solid_mapper.SetInputData(surface)
    solid_mapper.ScalarVisibilityOff()
    solid = vtk.vtkActor()
    solid.SetMapper(solid_mapper)
    solid.GetProperty().SetColor(*args.color)
    solid.GetProperty().SetAmbient(0.15)
    solid.GetProperty().SetDiffuse(0.85)
    solid.GetProperty().SetSpecular(0.15)
    solid.GetProperty().SetSpecularPower(20.0)

    outline_filter = vtk.vtkOutlineFilter()
    outline_filter.SetInputData(image)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outline_mapper)
    outline.GetProperty().SetColor(0.45, 0.45, 0.45)
    outline.GetProperty().SetLineWidth(1.5)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(solid)
    renderer.AddActor(outline)
    light_kit = vtk.vtkLightKit()
    light_kit.AddLightsToRenderer(renderer)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(8)
    window.SetSize(*args.size)
    window.AddRenderer(renderer)

    bounds = image.GetBounds()
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    direction = np.asarray(args.view, dtype=float)
    direction /= np.linalg.norm(direction)
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(np.asarray(center) + direction * 3.0 * extent))
    camera.SetViewUp(0.0, 1.0, 0.0)
    renderer.ResetCamera()
    camera.Zoom(args.zoom)

    window.Render()
    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(window)
    grabber.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(args.output))
    writer.SetInputConnection(grabber.GetOutputPort())
    writer.Write()


def main(argv=None) -> int:
    """命令行入口."""
    args = parse_args(argv)
    density = np.load(args.density)
    if density.ndim != 3:
        raise ValueError(f'密度数组须为三维 (nx, ny, nz), 实际形状 {density.shape}')
    surface, image = build_solid_surface(density, args.threshold, args.spacing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(surface, image, args)
    n_solid = int(np.count_nonzero(density >= args.threshold))
    print(f'[完成] 阈值 {args.threshold:g}: 实体单元 {n_solid} / {density.size} '
          f'({n_solid / density.size:.4f}), 表面多边形 {surface.GetNumberOfCells()}, 输出 {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
