"""Hu--Zhang 拓扑优化投稿论文实验的执行入口.

模块平铺在实验根目录 (与 ``experiments/`` 下其余八个实验同构), 可直接执行的是本
文件与 ``plot.py`` / ``table.py``: 本文件跑算例、写运行产物, 后两者在已有产物上
整理论文里的图与表:

- 配置          ``config.py`` (路径/TOML 加载/参数拍平) ``provenance.py`` (溯源戳记);
- 组装          ``pipeline.py``: 共享组装原语 + 三族算例的装配器 + 模型名注册表;
- 驱动          ``driver.py`` (优化/能量诊断);
- 产出层        ``metrics.py`` (插图数据导出);
- ``plots/``    唯一子目录: 每张图一个模块, 文件名是 ``<算例族>_<产物>`` 语义名
                (论文图号只写在各模块 docstring 首行的括注里); ``_base.py`` 是
                八个成图模块的共用底座 (字体/vtu 读取/产物定位/落盘口径)
                (成图落在 ``results/figures/``, 与之同名会混淆, 故不叫 figures).

优化算例一律由 ``--case`` 驱动, 参数默认取自 ``cases.toml``::

    python run.py --list                             # 列出全部算例
    python run.py --case compliance-fixed-fixed-half # 照注册表跑
    python run.py --case <id> <id> ...               # 跑若干 case
    python run.py --all                              # 跑全部 ready 算例
    python run.py --all --dry-run                    # 只打印派发计划

``--case`` 之后可直接追加 ``driver.py`` 认识的覆盖参数, 由其 argparse 校验::

    python run.py --case compliance-fixed-fixed-half --analyzer all --order 2

具名开关之外的配置字段走通用覆盖通道 ``--override KEY=VALUE`` (可重复给出)::

    python run.py --case compliance-fixed-fixed-half --override optimizer=oc

作用在已有产物上的后处理, 插图归 ``plot.py``, 表格归 ``table.py``::

    python plot.py --list
    python table.py --list

论文 5.1 节的制造解收敛阶 (表 5.1 / 5.2) 不是优化算例, 由自包含脚本
``manufactured_convergence.py`` 直接运行, 不进本入口.
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

from config import (
    CASES_FILE,
    ConfigurationError,
    bootstrap_source_path,
    load_cases,
)

bootstrap_source_path()


# 已迁往 plot.py / table.py 的动词: 老命令会落进本层的 parse_known_args, 报「需要给出
# --case」这种看不懂的错, 因此显式接住并指路.
MOVED_COMMANDS = ("figure", "export", "gradients", "metrics")

def display_width(text: str) -> int:
    """终端里占的列数: 东亚宽字符 (W) 与全角字符 (F) 占两格, 其余按一格算."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """按显示宽度右补空格; str.ljust 数的是字符数, 含中文的列会对不齐."""
    return text + " " * max(width - display_width(text), 0)


# 网格配置对应的完整类名; 未登记的取值原样显示.
# --list 的 mesh 列: soptx.mesh 网格类 + 剖分方式 (两种都是 TriangleMesh, 只差对角线规则)
MESH_CLASS_NAMES = {
    "triangle-checkerboard": "TriangleMesh/checkerboard",
    "triangle-single-diagonal-symmetric": "TriangleMesh/single-diagonal-sym",
}

# 阶次记号随方法走: Hu--Zhang 的 k 是应力空间次数 (位移阶为 k-1), LFEM 的 p 是位移
# 阶. 两者数值同源但含义不同, 统一写成 p 会把应力阶读成位移阶; 未登记的方法退回 p.
ORDER_SYMBOLS = {"huzhang": "k", "lfem": "p"}


def with_radius(kind: str, radius) -> str:
    """拼成 ``density r=2.4``; 半径缺项时只显示类型, 类型也没登记就显示 "-".

    半径按 %g 打而不是照抄 TOML 原文: 2.0 与 2 在这里是同一个半径, 写法跟着
    topopt_simp_* 走, 三个实验的同一列才好横着对.
    """
    if not kind:
        return "-"
    if radius is None:
        return kind
    text = f"{radius:g}" if isinstance(radius, (int, float)) else str(radius)
    return f"{kind} r={text}"


def registered_defaults(case: dict) -> tuple[str, str, str, str, str, str]:
    """把 case 缺省跑的组合压成 (网格类名, 剖分数, 分析链, 阶次, 过滤器, 优化器) 六个显示串.

    阶次单列, 记号按各方法的惯用写法 (``huzhang`` 的 k 是应力阶, ``lfem`` 的 p 是
    位移阶, 见 ORDER_SYMBOLS), 因此列里带记号而不是裸数字: 同一个 2 在两条链上不是
    同一个量. 与 resolve_runs 的缺省口径一致: 裸跑一条 case 即 methods x
    comparison_orders 全集, 故两列列出全部取值.
    字段一律 get: 骨架状态的 planned case 允许缺项, --list 不该因此崩掉.
    """
    discretization = case.get("discretization", {})
    orders = discretization.get("comparison_orders")
    methods = tuple(case.get("methods", ()))
    analyzer = ",".join(methods) or "-"
    symbol = "/".join(ORDER_SYMBOLS.get(method, "p") for method in methods) or "p"
    order_text = (
        f"{symbol}={','.join(str(int(order)) for order in orders)}" if orders else "-"
    )
    mesh_type = str(discretization.get("mesh_type", ""))
    kind = MESH_CLASS_NAMES.get(mesh_type, mesh_type) or "-"
    nx = discretization.get("nx")
    ny = discretization.get("ny")
    size = f"{nx}x{ny}" if nx and ny else "-"
    optimization = case.get("optimization", {})
    filter_text = with_radius(
        str(optimization.get("filter_type", "")), optimization.get("filter_radius")
    )
    optimizer = str(optimization.get("optimizer", "-"))
    return (
        kind, size, analyzer, order_text, filter_text, optimizer
    )


def list_cases() -> int:
    """打印每条算例缺省会跑的那个组合, 让调用方拿到 id 就知道裸跑会跑出什么."""
    cases = load_cases(CASES_FILE)
    header = (
        "case-id", "mesh", "grid", "analyzer", "order", "load-discretization", "filter", "optimizer"
    )
    rows = []
    for case in cases:
        mesh, grid, analyzer, order, filter_text, optimizer = registered_defaults(case)
        load_discretization = str(
            case.get("model", {}).get("parameters", {}).get("load_discretization", "-")
        )
        rows.append((case["id"], mesh, grid, analyzer, order,
                     load_discretization, filter_text, optimizer))
    widths = [
        max(display_width(row[i]) for row in (header, *rows)) for i in range(len(header))
    ]
    for row in (header, *rows):
        print("  ".join(pad(value, widths[i]) for i, value in enumerate(row)).rstrip())
    return 0


def select_case_ids(identifiers: list[str], run_all: bool) -> list[dict]:
    """把命令行给的 id 解析成 case 对象; --all 取全部 ready 算例, 顺序照 cases.toml."""
    cases = load_cases(CASES_FILE)
    if run_all:
        return [case for case in cases if case.get("status") == "ready"]
    index = {case["id"]: case for case in cases}
    unknown = [name for name in identifiers if name not in index]
    if unknown:
        raise ConfigurationError(
            f"未知的 case id: {', '.join(unknown)}; 可用: {', '.join(index)}."
        )
    return [index[name] for name in identifiers]


def run_cases(selected: list[dict], extra: list[str], dry_run: bool) -> int:
    """逐条交给 driver.py; extra 是原样转交的覆盖参数, 由 driver 自己校验."""
    case_ids = [case["id"] for case in selected]
    if dry_run:
        for case_id in case_ids:
            print(" ".join(["driver", "<-", case_id, *extra]))
        return 0

    for case_id in case_ids:
        print()
        suffix = " " + " ".join(extra) if extra else ""
        print(f"[case] {case_id} -> driver{suffix}")
        status = import_module("driver").main(["--case", case_id, *extra]) or 0
        if status != 0:
            print(f"[fail] {case_id} 以状态 {status} 退出, 中止后续算例", file=sys.stderr)
            return status
    return 0


def build_case_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: 否则 argparse 的前缀匹配会把驱动的参数误吞成本层的选项
    parser = argparse.ArgumentParser(prog="run.py", add_help=False, allow_abbrev=False)
    parser.add_argument("--list", action="store_true", help="列出全部算例及其缺省组合")
    # 一个旗标两种写法: --case a b c 与 --case a --case b 等价, 免得记两个近义词.
    # nargs 与 append 叠加会得到列表的列表, 收集后统一摊平.
    parser.add_argument(
        "--case",
        nargs="+",
        action="append",
        default=[],
        help="case id, 可一次给多个, 也可重复给出",
    )
    parser.add_argument("--all", action="store_true", help="跑全部 ready 算例")
    parser.add_argument("--dry-run", action="store_true", help="只打印派发计划, 不执行")
    return parser


def run_case_mode(argv: list[str]) -> int:
    parser = build_case_parser()
    arguments, extra = parser.parse_known_args(argv)

    if arguments.list:
        return list_cases()

    identifiers = [name for group in arguments.case for name in group]
    if not identifiers and not arguments.all:
        print("需要给出 --case / --all / --list", file=sys.stderr)
        return 1
    if identifiers and arguments.all:
        print("--all 与 --case 互斥", file=sys.stderr)
        return 1

    selected = select_case_ids(identifiers, arguments.all)
    if not selected:
        print("没有可运行的算例", file=sys.stderr)
        return 1

    # 覆盖参数针对单条 case 的注册值; 选中多条时同一组覆盖会落到不同算例上,
    # 语义不明确, 直接拒绝而不是猜.
    if extra and len(selected) != 1:
        print(
            f"覆盖参数 {' '.join(extra)} 只能配合单个 --case 使用 "
            f"(本次选中 {len(selected)} 条算例).",
            file=sys.stderr,
        )
        return 1

    return run_cases(selected, extra, arguments.dry_run)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1
    if argv[0] in MOVED_COMMANDS:
        moved = " ".join(argv)
        print(f"{argv[0]} 已迁往后处理入口, 见 python plot.py --help 与 python table.py --help", file=sys.stderr)
        return 1
    try:
        return run_case_mode(argv)
    except ConfigurationError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
