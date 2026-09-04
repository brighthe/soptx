# -*- coding: utf-8 -*-
"""EA 变密度拓扑优化实验的执行入口.

模块平铺在实验根目录 (与 ``experiments/`` 下其余实验同构), 可直接执行的只有本
文件、``collect.py`` 与 ``compare.py``: 本文件挑运行、派发给驱动, collect.py 把已有
产物整理成验收结论, compare.py 与 FA 侧同名运行做 Tier 1 对照:

- 配置          ``config.py`` (路径/TOML 加载/字段声明与校验) ``provenance.py`` (溯源戳记);
- 组装          ``pipeline.py``: soptx 公共组件组装 (operator_level 参数化);
- 驱动          ``driver.py``: 跑一次运行、写这一次的全部产物;
- 产出层        ``collect.py``: 结果验收与汇总; ``compare.py``: FA/EA 统一条件对照.

运行一律由 ``--case`` / ``--all`` 驱动, 参数默认取自 ``cases.toml``::

    python run.py --list                                  # 列出全部注册工况
    python run.py --case half_mbb_2d_concentrated         # 照注册表跑基准运行
    python run.py --case <id> <id> ...                    # 跑若干个基准运行
    python run.py --all                                   # 跑全部基准运行
    python run.py --all --dry-run                         # 只打印派发计划

``--case`` 只认注册工况 id (一条 [[cases]] = 一次基准运行)。参数变化不进注册表,
在 ``--case`` 之后追加驱动认识的参数, 由 ``driver.py`` 自身的 argparse 校验;
``--override`` 在基准上改字段, 产物落在 outputs/<id>/<字段>-<值>[__...]/::

    python run.py --all --quiet
    python run.py --case <id> --timing
    python run.py --case <id> --override simp_penalty=4.0

作用在已有产物上的验收、汇总与对照归 ``collect.py`` / ``compare.py``::

    python collect.py
    python compare.py --case <id> --override simp_penalty=4.0
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys
import unicodedata

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from config import ConfigError, TopOptCase, load
from pipeline import ANALYZER_KIND

# 本实验只有一条执行路径, 因此没有 huzhang_topopt_paper 那样的 role -> 驱动
# 映射表; 派发目标写成常量, 是为了与那边同一个读法。
RUNNER = "driver"

# 只能作用于唯一一次运行的透传参数: 一组 override 只在一个基准工况上改字段
# (口径同 topopt_simp_fa/run.py, --override 必须与单个 --case 配对)。--timing /
# --quiet 是各次运行自己的日志开关, 批量长跑照样能用, 不在此列。
SINGLE_RUN_FLAGS = frozenset({"--override"})


def display_width(text: str) -> int:
    """终端里占的列数: 东亚宽字符 (W) 与全角字符 (F) 占两格, 其余按一格算."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """按显示宽度右补空格; str.ljust 数的是字符数, 含中文的列会对不齐."""
    return text + " " * max(width - display_width(text), 0)


def list_cases(cases: tuple[TopOptCase, ...]) -> int:
    """一行一个注册工况 (= 一次基准运行), 列取 id 未编码的自由度.

    首列是 case.id, 也是 --case 认的写法与产物目录的第一层; 带 override 的运行
    不在注册表里, 不在这里列 (它们的参数标签见 outputs/ 或 collect 快照)。
    不打印 cases.toml 的 summary: 那是手写文本, 与字段无同步保证, 改了 grid
    忘改 summary 就会在这里撒谎; 表格直接由 TopOptCase 字段渲染。
    一格只装一个量: analyzer / order / solver 三者与 filter / optimizer 两者都
    逐工况可变, 挤在一格里就没法按列扫读, 也对不上 huzhang_topopt_paper 的表。
    integration_order 不进表; 算子层级 (fa/ea) 由模块目录名声明, 也不进表。
    """
    header = (
        "case-id", "mesh", "analyzer", "order", "filter", "solver", "optimizer", "role"
    )
    rows = [
        (
            case.id,
            "%s %s" % (case.cell_type, "x".join(str(n) for n in case.grid)),
            ANALYZER_KIND,
            "p=%d" % case.space_degree,
            "%s r=%g" % (case.filter_type, case.filter_radius),
            case.solve_method,
            case.optimizer,
            case.role or "-",
        )
        for case in cases
    ]
    widths = [
        max(display_width(row[i]) for row in (header, *rows))
        for i in range(len(header))
    ]
    for row in (header, *rows):
        print("  ".join(pad(value, widths[i]) for i, value in enumerate(row)).rstrip())
    return 0


def select_runs(
    cases: tuple[TopOptCase, ...], identifiers: list[str], run_all: bool
) -> list[TopOptCase]:
    """把命令行给的工况 id 解析成若干次基准运行; --all 取全部, 顺序照 cases.toml.

    同一个 id 给了两次只跑一次, 顺序仍按 cases.toml。
    """
    if run_all:
        return list(cases)
    known = {case.id for case in cases}
    unknown = [name for name in identifiers if name not in known]
    if unknown:
        raise ConfigError(
            f"未知的工况 id: {', '.join(unknown)}; "
            f"可用: {', '.join(sorted(known))}."
        )
    wanted = set(identifiers)
    return [case for case in cases if case.id in wanted]


def run_runs(selected: list[TopOptCase], extra: list[str], dry_run: bool) -> int:
    """逐次把运行交给驱动; extra 是原样转交的覆盖参数, 由驱动自己校验."""
    if dry_run:
        for case in selected:
            print(" ".join([RUNNER, "<-", case.id, *extra]))
        return 0

    for case in selected:
        print()
        suffix = " " + " ".join(extra) if extra else ""
        print(f"[case] {case.id} -> {RUNNER}{suffix}")
        status = import_module(RUNNER).main(["--case", case.id, *extra]) or 0
        if status != 0:
            print(
                f"[fail] {case.id} 以状态 {status} 退出, 中止后续运行",
                file=sys.stderr,
            )
            return status
    return 0


def build_case_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: 否则 argparse 的前缀匹配会把驱动的参数误吞成本层的选项
    parser = argparse.ArgumentParser(prog="run.py", add_help=False, allow_abbrev=False)
    parser.add_argument("--list", action="store_true", help="列出全部注册工况")
    # 一个旗标两种写法: --case a b c 与 --case a --case b 等价, 免得记两个近义词.
    # nargs 与 append 叠加会得到列表的列表, 收集后统一摊平.
    parser.add_argument(
        "--case",
        nargs="+",
        action="append",
        default=[],
        help="注册工况 id, 可一次给多个, 也可重复给出",
    )
    parser.add_argument("--all", action="store_true", help="跑全部基准运行")
    parser.add_argument("--dry-run", action="store_true", help="只打印派发计划, 不执行")
    return parser


def run_case_mode(argv: list[str]) -> int:
    parser = build_case_parser()
    arguments, extra = parser.parse_known_args(argv)

    _, cases = load()
    if arguments.list:
        return list_cases(cases)

    identifiers = [name for group in arguments.case for name in group]
    if not identifiers and not arguments.all:
        print("需要给出 --case / --all / --list", file=sys.stderr)
        return 1
    if identifiers and arguments.all:
        print("--all 与 --case 互斥", file=sys.stderr)
        return 1

    selected = select_runs(cases, identifiers, arguments.all)
    if not selected:
        print("没有可运行的工况", file=sys.stderr)
        return 1

    # override 要落到唯一的基准工况上 (见 SINGLE_RUN_FLAGS); 其余透传参数逐次
    # 运行各自成立, 不受这条限制。
    single_only = [
        token for token in extra if token.split("=", 1)[0] in SINGLE_RUN_FLAGS
    ]
    if single_only and len(selected) != 1:
        print(
            f"{' '.join(single_only)} 需要唯一目标, 本次选中 {len(selected)} 个工况; "
            "请只给一个 --case.",
            file=sys.stderr,
        )
        return 1

    return run_runs(selected, extra, arguments.dry_run)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1
    try:
        return run_case_mode(argv)
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
