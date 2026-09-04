"""Hu--Zhang 拓扑优化投稿论文实验的执行入口.

模块平铺在实验根目录 (与 ``experiments/`` 下其余八个实验同构), 可直接执行的只有本
文件与 ``compare.py``: 本文件跑算例、写运行产物, compare.py 把已有产物整理成论文
里的图与数字:

- 配置          ``config.py`` (路径/TOML 加载/参数拍平) ``provenance.py`` (溯源戳记);
- 组装          ``pipeline.py``: 共享组装原语 + 三族算例的装配器 + 模型名注册表;
- 驱动          ``driver.py`` (优化/状态对比/能量诊断) 与 ``convergence.py`` (制造解收敛阶);
- 产出层        ``report.py`` (论文表 5.1 / 5.2) ``metrics.py`` (梯度校验/冻结指标/
                插图数据导出);
- ``plots/``    唯一子目录: 每张图一个模块, 文件名是 ``<算例族>_<产物>`` 语义名
                (论文图号只写在各模块 docstring 首行的括注里); ``_base.py`` 是
                八个成图模块的共用底座 (字体/vtu 读取/产物定位/落盘口径)
                (成图落在 ``outputs/figures/``, 与之同名会混淆, 故不叫 figures).

算例一律由 ``--case`` 驱动, 参数默认取自 ``cases.toml``; case 归属哪个驱动由其 ``role``
决定, 调用方拿到 id 即可运行, 不必先知道用哪个动词::

    python run.py --list                             # 列出全部算例
    python run.py --case manufactured-native         # 照注册表跑
    python run.py --case <id> <id> ...               # 跑若干 case
    python run.py --all                              # 跑全部 ready 算例
    python run.py --all --dry-run                    # 只打印派发计划

``--case`` 之后可直接追加该驱动认识的覆盖参数, 由驱动自身的 argparse 校验::

    python run.py --case manufactured-stabilized --stabilization none
    python run.py --case compliance-fixed-fixed-half --analyzer all --order 2

具名开关之外的配置字段走通用覆盖通道 ``--override KEY=VALUE`` (可重复给出)::

    python run.py --case compliance-fixed-fixed-half --override optimizer=mma

作用在已有产物上的后处理 (插图 / 论文表 / 冻结指标) 一律归 ``compare.py``::

    python compare.py --list
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
    default_method,
    load_cases,
)

bootstrap_source_path()


# 已迁往 compare.py 的动词: 老命令会落进本层的 parse_known_args, 报「需要给出
# --case」这种看不懂的错, 因此显式接住并指路.
MOVED_COMMANDS = ("figure", "table", "export", "gradients", "metrics")

# 算例派发规则: role 决定 case 由哪个驱动执行, 覆盖参数也随之透传给该驱动
CONVERGENCE_ROLE = "convergence-verification"
RUNNERS: dict[str, str] = {
    "convergence": "convergence",
    "optimize": "driver",
}


def runner_for(case: dict) -> str:
    """按 role 判定该 case 归哪个驱动; 收敛验证走制造解套件, 其余走优化驱动."""
    return "convergence" if case.get("role") == CONVERGENCE_ROLE else "optimize"


def display_width(text: str) -> int:
    """终端里占的列数: 东亚宽字符 (W) 与全角字符 (F) 占两格, 其余按一格算."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """按显示宽度右补空格; str.ljust 数的是字符数, 含中文的列会对不齐."""
    return text + " " * max(width - display_width(text), 0)


# 网格类型的显示缩写; 未登记的取值原样显示, 免得新网格被静默显示成三角形.
MESH_ABBREVIATIONS = {"triangle-checkerboard": "tri"}

# 阶次记号随方法走: Hu--Zhang 的 k 是应力空间次数 (位移阶为 k-1), LFEM 的 p 是位移
# 阶。两者数值同源但含义不同, 统一写成 p 会把应力阶读成位移阶; 未登记的方法退回 p.
ORDER_SYMBOLS = {"huzhang": "k", "lfem": "p"}


def with_kind(kind: str, size: str) -> str:
    """拼成 ``tri 80x20``; 注册表没声明网格类型时只显示剖分."""
    return f"{kind} {size}" if kind else size


def with_radius(kind: str, radius) -> str:
    """拼成 ``density r=2.4``; 半径缺项时只显示类型, 类型也没登记就显示 "-".

    半径按 %g 打而不是照抄 TOML 原文: 2.0 与 2 在这里是同一个半径, 写法跟着
    topopt_simp_* 走, 三个实验的同一列才好横着对。
    """
    if not kind:
        return "-"
    if radius is None:
        return kind
    text = f"{radius:g}" if isinstance(radius, (int, float)) else str(radius)
    return f"{kind} r={text}"


def registered_defaults(case: dict) -> tuple[str, str, str, str, str, str]:
    """把 case 缺省跑的组合压成 (网格, 分析链, 阶次, 过滤器, 解法器, 优化器) 六个显示串.

    阶次单列, 记号按各方法的惯用写法 (``huzhang`` 的 k 是应力阶, ``lfem`` 的 p 是
    位移阶, 见 ORDER_SYMBOLS), 因此列里带记号而不是裸数字: 同一个 2 在两条链上不是
    同一个量。分析链与阶次都是单值, 与 resolve_runs 的缺省口径一致: 裸跑一条 case
    就是一次运行。注册表里的完整对比组 (methods x comparison_orders) 要 --full 才展开.

    收敛验证由 base_nx/base_ny + levels 逐级加密, nx/ny 只是兼容字段, 因此按加密
    区间显示; 这两条 case 不注册过滤器, optimizer = "none" 也只是满足 schema 的占位,
    两列一并显示成 "-"; 解法器它们照样要用, 所以 solver 列照常填。
    字段一律 get: 骨架状态的 planned case 允许缺项, --list 不该因此崩掉.
    """
    discretization = case.get("discretization", {})
    orders = discretization.get("comparison_orders")
    methods = tuple(case.get("methods", ()))
    analyzer = default_method(methods) if methods else "-"
    symbol = ORDER_SYMBOLS.get(analyzer, "p")
    order_text = (
        f"{symbol}={min(int(order) for order in orders)}" if orders else "-"
    )
    mesh_type = str(discretization.get("mesh_type", ""))
    kind = MESH_ABBREVIATIONS.get(mesh_type, mesh_type)
    solver = str(discretization.get("solve_method", "-"))
    if runner_for(case) == "convergence":
        base_nx = discretization.get("base_nx")
        base_ny = discretization.get("base_ny")
        levels = discretization.get("levels")
        if base_nx and base_ny and levels:
            factor = 2 ** (int(levels) - 1)
            size = f"{base_nx}x{base_ny}->{int(base_nx) * factor}x{int(base_ny) * factor}"
        else:
            size = "-"
        return with_kind(kind, size), analyzer, order_text, "-", solver, "-"
    nx = discretization.get("nx")
    ny = discretization.get("ny")
    size = f"{nx}x{ny}" if nx and ny else "-"
    optimization = case.get("optimization", {})
    filter_text = with_radius(
        str(optimization.get("filter_type", "")), optimization.get("filter_radius")
    )
    optimizer = str(optimization.get("optimizer", "-"))
    return (
        with_kind(kind, size), analyzer, order_text, filter_text, solver, optimizer
    )


def list_cases() -> int:
    """打印每条算例缺省会跑的那个组合, 让调用方拿到 id 就知道裸跑会跑出什么."""
    cases = load_cases(CASES_FILE)
    header = (
        "case-id", "mesh", "analyzer", "order", "filter", "solver", "optimizer", "role"
    )
    rows = [
        (case["id"], *registered_defaults(case), str(case.get("role", "-")))
        for case in cases
    ]
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
    """按 role 派发算例; extra 是原样转交给该驱动的覆盖参数, 由驱动自己校验."""
    plan = [(runner_for(case), case["id"]) for case in selected]
    if dry_run:
        for runner, case_id in plan:
            print(" ".join([RUNNERS[runner], "<-", case_id, *extra]))
        return 0

    for runner, case_id in plan:
        print()
        suffix = " " + " ".join(extra) if extra else ""
        print(f"[case] {case_id} -> {runner}{suffix}")
        status = import_module(RUNNERS[runner]).main(["--case", case_id, *extra]) or 0
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

    # 覆盖参数的合法性由目标驱动的 argparse 判定, 而两个驱动认的参数并不重叠;
    # 选中多条 case 时无法确定这些参数该按哪个驱动解释, 直接拒绝而不是猜.
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
        print(f"{argv[0]} 已迁往后处理入口, 改用: python compare.py {moved}", file=sys.stderr)
        return 1
    try:
        return run_case_mode(argv)
    except ConfigurationError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
