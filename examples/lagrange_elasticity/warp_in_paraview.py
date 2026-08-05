"""ParaView 内 Python Shell 脚本 — Warp By Vector + 截图.

用法:
    1. 打开 ParaView, 先手动 File → Open 加载 .vtu 文件
    2. Tools → Python Shell → Run Script, 选择本文件
    或直接在 Python Shell 中粘贴下方代码.

注意: 运行前已加载的 VTU reader 会被自动检测; 如果加载了多个,
脚本使用第一个含有 "u_x" 向量数据的 reader。
"""

from paraview.simple import *

# ---------------------------------------------------------------------------
# 自动检测已加载的 VTU reader
# ---------------------------------------------------------------------------
source = None
for proxy in GetSources().values():
    point_data = proxy.PointData if proxy.PointData else []
    fields = [p.Name for p in point_data]
    if "u_x" in fields:
        source = proxy
        print(f"检测到 VTU 数据源: {proxy}")
        print(f"  可用场: {fields}")
        break

if source is None:
    raise RuntimeError(
        "未找到包含 'u_x' 位移场的 VTU reader.\n"
        "请先用 File → Open 加载 .vtu 文件后再运行本脚本."
    )

# ---------------------------------------------------------------------------
# Warp By Vector
# ---------------------------------------------------------------------------
WARP_SCALE = 1.0

warp = WarpByVector(Input=source)
warp.Vectors = ["POINTS", "u_x", "u_y", ""]
warp.ScaleFactor = WARP_SCALE

# 显示: 位移幅值着色
display = GetDisplayProperties(warp, view=None)
display.Representation = "Surface"
display.ColorArrayName = ["POINTS", "u_mag"]
display.SetScalarBarVisibility(warp, True)

color_bar = GetScalarBar(display, RenderView())
if color_bar:
    color_bar.Title = "|u|"
    color_bar.ComponentTitle = ""

# ---------------------------------------------------------------------------
# 视图
# ---------------------------------------------------------------------------
view = GetActiveView()
view.ViewSize = [1600, 900]
view.CameraParallelProjection = True
view.ResetCamera()
Render()

# ---------------------------------------------------------------------------
# 截图
# ---------------------------------------------------------------------------
from pathlib import Path
vtu_path = None
for fname, proxy in GetSources().items():
    if proxy == source:
        vtu_path = Path(fname)
        break

out_dir = Path(vtu_path).parents[2] / "figures" if vtu_path else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = str(out_dir / "mbb_beam_warped.png")

SaveScreenshot(out_path, view, ImageResolution=[3200, 1800])
print(f"截图已保存: {out_path}")
