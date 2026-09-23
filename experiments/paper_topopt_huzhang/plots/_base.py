"""``plots/`` 下各成图模块的共用底座.

中英文字体口径、vtu 读取、产物目录定位与统一落盘口径 (dpi、输出格式、论文插图同步
目录) 只有一处定义, 十一个成图模块都从这里取; 改一次插图口径不必逐个模块翻找.

下划线开头有两重作用: 标明它不是一件可整理的产物, 且 ``compare.py:discover_cases()``
按 ``_`` 前缀跳过本模块, 扫描逻辑无须为它开特例.

原为 ``report.py`` 的前半段, 2026-09-03 下沉进 plots 包内, 使 ``report.py`` 回到与
``experiments/matrix_free_capability/report.py`` 一致的「论文表报」语义.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import vtk
from matplotlib import font_manager

import config


# 中文字体候选: WSL 下优先借用 Windows 微软雅黑, 其次发行版自带的 Droid Fallback
_CHINESE_FONT_CANDIDATES = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def chinese_font() -> font_manager.FontProperties:
    """返回可渲染中文的字体属性; 候选字体都不存在时回退到默认无衬线族."""
    for candidate in _CHINESE_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties(family="sans-serif")


def academic_rcparams() -> None:
    """学术英文排版口径: sans-serif 字族 + stix 数学字体 + 正常负号.

    四个成图模块原先各自逐字重复这四行, 收在这里只留一处; 2026-09-17 起
    ``stress_convergence`` 也调用它 —— 该图改用 ``compliance_convergence`` 的排版与
    配色口径, 中文仍靠 ``fontproperties`` 逐处指定, 与本函数的字族设置不冲突.
    余下的 ``stress_topologies`` 不调用它, 外观维持原样; ``bearing_highorder_topologies`` 只设
    ``axes.unicode_minus``, 同样不并进来 —— 并了会改动它已定稿的插图外观.

    pyplot 放在函数体内 import: 各模块都先 ``matplotlib.use("Agg")`` 再导入 pyplot,
    _base 在模块级导入 pyplot 会抢在那之前把后端定死.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(
    figure,
    stem: str,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """把插图写入本地 ``outputs/figures``, 并在论文插图目录存在时同步一份."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    directories = [config.FIGURE_DIR]
    if config.PAPER_FIGURE_DIR.is_dir():
        directories.append(config.PAPER_FIGURE_DIR)

    written: list[Path] = []
    for directory in directories:
        for suffix in formats:
            path = directory / f"{stem}.{suffix}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"[OK] {path}")
            written.append(path)
    return written


def resolve_run_dir(case_dir: Path, folder: str) -> Path | None:
    """定位一次优化运行的产物目录.

    ``folder`` 是产物目录第二层的参数标签, 按 driver 的命名
    ``analyzer-<链>__order-<k>[__<字段>-<取值>...]`` 书写 (见 driver.py 的
    _run_label)。标签自描述且与参数一一对应, 故不做名字回落: 目录不在就返回
    None, 由调用方决定报错还是画占位面板。
    """
    candidate = case_dir / folder
    return candidate if candidate.is_dir() else None


def require_run_dir(case_dir: Path, folder: str) -> Path:
    """同 :func:`resolve_run_dir`, 但产物缺失时报错而不是出一张缺数据的图."""
    run_dir = resolve_run_dir(case_dir, folder)
    if run_dir is None:
        raise FileNotFoundError(
            f"缺少产物目录 {case_dir.name}/{folder}; "
            f"请先运行 run.py --case {case_dir.name}"
        )
    return run_dir


def load_density(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 vtu 的节点坐标、三角形连接与单元密度."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    points = np.array([grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())])
    connectivity = np.array(
        [
            [grid.GetCell(cid).GetPointId(j) for j in range(grid.GetCell(cid).GetNumberOfPoints())]
            for cid in range(grid.GetNumberOfCells())
        ],
        dtype=np.int32,
    )
    density_array = grid.GetCellData().GetArray("density")
    density = np.array([density_array.GetValue(i) for i in range(grid.GetNumberOfCells())])
    return points, connectivity, density


def load_compliance(run_dir: Path) -> float | None:
    """从运行摘要读取柔顺度; 摘要缺失或字段缺失时返回 None, 不猜测数值."""
    summary_file = run_dir / "summary.json"
    if not summary_file.is_file():
        return None
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    value = data.get("compliance", data.get("objective"))
    return None if value is None else float(value)


def compliance_label(run_dir: Path) -> str:
    """构造插图标题里的柔顺度片段; 无数据时返回空串."""
    value = load_compliance(run_dir)
    return "" if value is None else f", $C = {value:.2f}$"


def mirror_half_beam(
    points: np.ndarray,
    connectivity: np.ndarray,
    density: np.ndarray,
    axis: float = 160.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把左半域沿 ``x = axis / 2`` 镜像成完整梁 (节点重复, 仅供可视化)."""
    mirrored = np.copy(points)
    mirrored[:, 0] = axis - points[:, 0]
    return (
        np.vstack([points, mirrored]),
        np.vstack([connectivity, connectivity + len(points)]),
        np.concatenate([density, density]),
    )
