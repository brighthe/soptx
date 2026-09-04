"""子结构验证入口共用的默认值、参数校验和终端排版.

本模块只依赖标准库, 公开脚本解析参数时不加载有限元或求解器.
"""

from __future__ import annotations

import unicodedata
from typing import Sequence, Tuple

DEFAULT_LEVELS = {2: 4, 3: 3}
DEFAULT_WARMUP = 1
DEFAULT_REPEAT = 5
DEFAULT_DENSITY = "cell"
CONVERGENCE_SOLVERS = ("scipy", "mumps")
ORDER_MARGIN = 0.20
RELATIVE_ERROR_TOLERANCE = 1.0e-11
CONSISTENCY_TOLERANCE = 1.0e-9


def resolve_grid(
    dim: int,
    n_sub: Sequence[int] | None,
    n_fine: Sequence[int] | None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """补齐并校验子结构划分, 在创建网格之前拒绝非法规模.

    Parameters
    ----------
    dim : int
        空间维数, 取 2 或 3.
    n_sub, n_fine : sequence of int or None
        各方向子结构数与每块单元数; None 表示采用该维度的默认值.

    Returns
    -------
    tuple of tuple of int
        校验后的子结构划分与局部单元划分.

    Raises
    ------
    ValueError
        维数不支持, 或划分不是 dim 个正整数.
    """
    if dim not in (2, 3):
        raise ValueError("dim 必须为 2 或 3.")
    default_sub, default_fine = (
        ((6, 2), (5, 5)) if dim == 2 else ((6, 2, 2), (4, 4, 4))
    )
    sub = tuple(default_sub if n_sub is None else n_sub)
    fine = tuple(default_fine if n_fine is None else n_fine)
    for name, values in (("--n-sub", sub), ("--n-fine", fine)):
        if len(values) != dim:
            raise ValueError(f"{name} 在 dim={dim} 时必须提供 {dim} 个整数.")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in values):
            raise ValueError(f"{name} 的各项必须为正整数.")
    return sub, fine


def display_width(text: str) -> int:
    """计算字符串的终端显示宽度, 东亚全角字符按两列计."""
    return sum(2 if unicodedata.east_asian_width(char) in ("F", "W") else 1 for char in text)


def print_table(rows: Sequence[Sequence[str]]) -> None:
    """按终端显示宽度输出紧凑表格."""
    widths = [max(display_width(row[col]) for row in rows) for col in range(len(rows[0]))]
    for row in rows:
        print("  ".join(
            value + " " * (widths[col] - display_width(value))
            for col, value in enumerate(row)
        ).rstrip())
