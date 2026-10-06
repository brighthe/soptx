"""``plots/`` 下各成图模块的共用底座.

论文字体口径、vtu 读取、产物目录定位与统一落盘口径 (dpi、输出格式、论文插图同步
目录) 只有一处定义, 八个成图模块都从这里取; 改一次插图口径不必逐个模块翻找.

下划线开头有两重作用: 标明它不是一件可整理的产物, 且 ``plot.py:discover_cases()``
按 ``_`` 前缀跳过本模块, 扫描逻辑无须为它开特例.

原为 ``report.py`` 的前半段, 2026-09-03 下沉进 plots 包内; ``report.py`` 余下的
论文表 5.1 / 5.2 已并入自包含脚本 ``manufactured_convergence.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from matplotlib import font_manager

import config


# 论文正文字体候选: CICP 类以 mathpazo 选项排版, 正文为 Palatino. WSL 下优先借用
# Windows 的 Palatino Linotype (TrueType, PDF 后端可按 Type 42 嵌入), 其次 TeX Live
# 自带的 TeX Gyre Pagella (OpenType CFF, PNG 可用, PDF 嵌入不保证).
_PAPER_FONT_CANDIDATES = (
    (
        "Palatino Linotype",
        tuple(f"/mnt/c/Windows/Fonts/{f}.ttf" for f in ("pala", "palab", "palai", "palabi")),
    ),
    (
        "TeX Gyre Pagella",
        tuple(
            f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-{s}.otf"
            for s in ("regular", "bold", "italic", "bolditalic")
        ),
    ),
)


def paper_rcparams(base_size: float = 9.0) -> str:
    """论文插图排版口径: 与 CICP 正文同族的 Palatino 衬线字体 + 同字体的 mathtext.

    2026-09-28 起全部八个成图模块改按版心尺寸出图, 都只调用本函数; 原 DejaVu Sans
    口径的 ``academic_rcparams`` 与中文字体 ``chinese_font`` 随旧图删除. 插图按
    ``\\textwidth`` (150 mm, 5.9 in) 原尺寸嵌入, 字号不再经缩放, 故 ``base_size``
    就是纸面字号 (正文 10 pt, 题注 9 pt).
    返回实际选中的字族名; 候选字体都不存在时回退到 DejaVu Serif.
    """
    import matplotlib.pyplot as plt

    family = "DejaVu Serif"
    for name, files in _PAPER_FONT_CANDIDATES:
        present = [f for f in files if Path(f).is_file()]
        if present:
            for f in present:
                font_manager.fontManager.addfont(f)
            family = name
            break

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [family, "DejaVu Serif"]
    plt.rcParams["font.size"] = base_size
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = f"{family}:italic"
    plt.rcParams["mathtext.bf"] = f"{family}:bold"
    plt.rcParams["mathtext.cal"] = f"{family}:italic"  # 不用花体; 缺省 cursive 找不到会告警
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    return family


def save_figure(
    figure,
    stem: str,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """把插图写入本地 ``results/figures``, 并在论文插图目录存在时同步一份."""
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
    _run_label). 标签自描述且与参数一一对应, 故不做名字回落: 目录不在就返回
    None, 由调用方决定报错还是画占位面板.
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
