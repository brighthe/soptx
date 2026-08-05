"""ParaView 批处理脚本: Warp By Vector 可视化 VTU 位移场并保存截图.

用法 (在 WSL 中运行):
    "/mnt/c/Program Files/ParaView 6.2.0/bin/pvpython.exe" examples/lagrange_elasticity/warp_screenshot.py

如需调整 VTU 文件路径, 修改下方 VTU_PATH。
"""

from paraview.simple import *

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
VTU_PATH = "examples/lagrange_elasticity/outputs/vtu/mbb-half_p1_quad_240x80.vtu"
OUTPUT_PATH = "examples/lagrange_elasticity/outputs/figures/mbb_beam_warped.png"
WARP_SCALE = 1.0                # 位移缩放因子 (1 = 真实变形)
CAMERA_POSITION = "XY"          # XY 平面视图

# ---------------------------------------------------------------------------
# 加载 VTU
# ---------------------------------------------------------------------------
reader = OpenDataFile(VTU_PATH)
print(f"已加载: {VTU_PATH}")
print(f"  可用场: {[p.Name for p in reader.PointData]}")

# ---------------------------------------------------------------------------
# Warp By Vector — 将位移场作用到网格几何上
# ---------------------------------------------------------------------------
# 根据实际 VTU 中的位移分量名设置向量 (本例为 u_x, u_y)
warp = WarpByVector(Input=reader)
warp.Vectors = ["POINTS", "u_x", "u_y", ""]  # 2D 向量, z 分量留空
warp.ScaleFactor = WARP_SCALE

print(f"Warp 缩放因子: {WARP_SCALE}")

# ---------------------------------------------------------------------------
# 显示设置
# ---------------------------------------------------------------------------
display = GetDisplayProperties(warp, view=None)
display.Representation = "Surface"
display.ColorArrayName = ["POINTS", "u_mag"]
display.SetScalarBarVisibility(reader, True)

# 色条标题
color_bar = GetScalarBar(display, RenderView())
if color_bar:
    color_bar.Title = "|u|"
    color_bar.ComponentTitle = ""

# ---------------------------------------------------------------------------
# 视图与相机
# ---------------------------------------------------------------------------
view = GetActiveView()
view.ViewSize = [1600, 900]

if CAMERA_POSITION == "XY":
    view.CameraPosition = [30, 10, 100]  # 从正前方看 XY 平面
    view.CameraFocalPoint = [30, 10, 0]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelProjection = True

view.ResetCamera()
Render()

# ---------------------------------------------------------------------------
# 截图与注释
# ---------------------------------------------------------------------------
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

SaveScreenshot(OUTPUT_PATH, view, ImageResolution=[3200, 1800])
print(f"截图已保存: {OUTPUT_PATH}")
