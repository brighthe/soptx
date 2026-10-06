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

优化算例一律由 ``--case`` 驱动, 参数一律取 ``cases.toml`` 的注册值; 裸跑一条 case
即论文该算例的完整对比组 (methods x comparison_orders)::

    python run.py --list                             # 列出全部算例
    python run.py --case compliance-fixed-fixed-half # 完整对比组
    python run.py --case <id> <id> ...               # 跑若干 case
    python run.py --all                              # 跑全部 ready 算例

单条 case 可追加 ``--analyzer`` / ``--order`` 只跑其中几组, 由 ``driver.py`` 校验::

    python run.py --case compliance-fixed-fixed-half --analyzer huzhang --order 2

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


def list_cases() -> int:
    """列出每条算例的 id、状态、缺省展开的运行组合与标题."""
    for case in load_cases(CASES_FILE):
        methods = ",".join(case.get("methods", ())) or "-"
        orders = ",".join(str(o) for o in case.get("discretization", {}).get("comparison_orders", ())) or "-"
        print(f"{case['id']:30s}{case.get('status', '-'):9s}{methods} x {orders:10s}{case.get('title', '')}")
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


def run_cases(selected: list[dict], extra: list[str]) -> int:
    """逐条交给 driver.py; extra 是原样转交的运行组合参数, 由 driver 自己校验."""
    for case in selected:
        suffix = " " + " ".join(extra) if extra else ""
        print(f"\n[case] {case['id']} -> driver{suffix}")
        status = import_module("driver").main(["--case", case["id"], *extra]) or 0
        if status != 0:
            print(f"[fail] {case['id']} 以状态 {status} 退出, 中止后续算例", file=sys.stderr)
            return status
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1
    # allow_abbrev=False: 否则 argparse 的前缀匹配会把驱动的参数误吞成本层的选项
    parser = argparse.ArgumentParser(prog="run.py", add_help=False, allow_abbrev=False)
    parser.add_argument("--list", action="store_true")
    # --case a b c 与 --case a --case b 等价; nargs 与 append 叠加得到列表的列表, 收集后摊平
    parser.add_argument("--case", nargs="+", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    arguments, extra = parser.parse_known_args(argv)

    if arguments.list:
        return list_cases()
    identifiers = [name for group in arguments.case for name in group]
    if bool(identifiers) == arguments.all:
        print("需要给出 --case 或 --all 之一 (二者互斥), 或 --list", file=sys.stderr)
        return 1
    try:
        selected = select_case_ids(identifiers, arguments.all)
    except ConfigurationError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1
    if not selected:
        print("没有可运行的算例", file=sys.stderr)
        return 1
    # --analyzer / --order 针对单条 case 的注册组合; 选中多条时语义不明确, 直接拒绝
    if extra and len(selected) != 1:
        print(f"{' '.join(extra)} 只能配合单个 --case 使用 (本次选中 {len(selected)} 条算例).",
              file=sys.stderr)
        return 1
    return run_cases(selected, extra)


if __name__ == "__main__":
    raise SystemExit(main())
