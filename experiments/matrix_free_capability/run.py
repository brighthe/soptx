# -*- coding: utf-8 -*-
"""图 2「Matrix-Free 算子 / Krylov 求解」的数据调度入口.

本脚本不实现任何物理或数值算法, 只按 ``cases.toml`` 以**子进程**调用
``examples/`` 与 ``tools/`` 下已有的脚本, 再由 ``collect.py`` 把产物收成
一份带溯源的入库快照。

用子进程而非导入函数, 有一个硬性理由: (b) 的峰值内存取 ``ru_maxrss``,
它是进程级高水位、无法按对象归因, 所以每个数据点必须独占一个进程 ——
同一进程里先建 FA 再建 EA, 测出来的 EA 峰值是被 FA 抬高过的。

命令:
    # 列出全部数据点
    python experiments/matrix_free_capability/run.py --list

    # 只打印将要执行的命令, 不运行 (先看一遍再跑)
    python experiments/matrix_free_capability/run.py --all --check-only

    # 跑某一格 / 某一个数据点 / 全部
    python experiments/matrix_free_capability/run.py --panel a
    python experiments/matrix_free_capability/run.py --case b-fa-n64
    python experiments/matrix_free_capability/run.py --all

    # 从已有产物收成快照 (不重跑)
    python experiments/matrix_free_capability/run.py --collect
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import config  # noqa: E402
import report  # noqa: E402


def command_list(cases: tuple[config.Case, ...], figure: dict) -> int:
    """打印已注册的数据点及其产物状态."""
    print(f"图 {figure.get('id')}: {figure.get('title')} —— {figure.get('question')}")
    print(f"消费方: {figure.get('consumer')}\n")
    header = f"{'case id':14} {'格':3} {'role':14} {'产物':6} 说明"
    print(header)
    print("-" * max(len(header), 88))
    for case in cases:
        present = "有" if case.artifact_path.is_file() else "缺"
        print(f"{case.id:14} {case.panel:3} {case.role:14} {present:6} {case.summary}")
    # 哪一格是占位由注册情况决定, 不写死在这里 —— 写死过一次, 结果数据组 (c) 的
    # case 注册上以后这行还在说"尚无数据点"。
    empty = [panel for panel in config.PANELS
             if not any(case.panel == panel for case in cases)]
    if empty:
        print(f"\n尚无数据点的图面分格: {', '.join(empty)} (为占位)。")
        if "c" in empty:
            print(f"    {collect.PANEL_C_REASON}")
    return 0


def command_run(cases: tuple[config.Case, ...], *, check_only: bool,
                skip_existing: bool) -> int:
    """逐个跑选中的 case.

    参数:
        cases: 选中的 case.
        check_only: 只打印将要执行的命令, 不实际运行.
        skip_existing: 产物已存在时跳过.

    返回:
        status: 全部成功返回 ``0``, 否则返回失败的 case 个数.
    """
    failed = 0
    for index, case in enumerate(cases, start=1):
        argv = case.command()
        prefix = f"[{index}/{len(cases)}] {case.id}"

        if case.output_mode == "fixed":
            print(f"{prefix}: 产物落盘位置由上游脚本写死, 不重定向 -> "
                  f"{case.artifact_path}")
        if skip_existing and case.artifact_path.is_file():
            print(f"{prefix}: 产物已存在, 跳过 (去掉 --skip-existing 可强制重跑)")
            continue

        printable = " ".join(argv)
        if check_only:
            print(f"{prefix}: {printable}")
            continue

        print(f"\n{prefix}: {case.summary}")
        print(f"  $ {printable}", flush=True)
        started = time.perf_counter()
        # 不捕获输出: 这些 case 单档可达数分钟, 需要让进度直接可见。
        # artifact 允许带子目录 (如 mpi/...), 子进程不负责建目录。
        if case.output_mode == "file":
            case.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            argv, cwd=config.REPOSITORY_ROOT, check=False,
            env=case.subprocess_env(),
        )
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            failed += 1
            print(f"  失败: 退出码 {completed.returncode}, 用时 {elapsed:.1f} s")
        elif not case.artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {case.artifact_path}")
        else:
            print(f"  完成: {elapsed:.1f} s -> {case.artifact_path.name}")

    if failed:
        print(f"\n{failed} 个 case 失败, 未执行 collect。")
    return failed


def command_collect(cases: tuple[config.Case, ...], figure: dict) -> int:
    """收产物、判门禁、写快照.

    返回:
        status: 门禁全过返回 ``0``, 有失败项返回 ``1``.
    """
    try:
        snapshot = collect.build(cases, figure)
    except collect.CollectError as error:
        # 产物缺失是常态入口错误 (还没跑), 不是异常, 不该抛 traceback。
        print(f"采集失败: {error}", file=sys.stderr)
        return 1
    path = collect.write(snapshot)
    record = snapshot["provenance"]

    print(f"快照已写入: {path}")
    print(f"  revision : {record['git_revision']}  (dirty={record['git_dirty']})")
    print(f"  采集时间 : {record['generated_at_utc']}")
    print(f"  本机内存 : {record['memory_total_bytes'] / 2 ** 30:.1f} GiB")

    for cross in snapshot["panels"]["a"]["chain_cross_check"]:
        cells = ", ".join(
            "缺档" if value is None else f"{value:.2e}"
            for value in cross["error_relative_differences"]
        )
        worst = cross["maximum"]
        worst_text = "缺档" if worst is None else f"{worst:.2e}"
        print(f"  {cross['dimension']}D 两条误差链逐档相对差: "
              f"[{cells}] (最差 {worst_text})")

    for note in snapshot["notes"]:
        print(f"  提示: {note}")

    failures = snapshot["gate_failures"]
    if failures:
        print(f"\n门禁未通过 ({len(failures)} 项):")
        for item in failures:
            print(f"  - {item}")
        return 1

    if not snapshot["reproducible"]:
        print("\n注意: 工作区 dirty 或无 git 溯源, 这批数字属开发证据, "
              "不可复现。正式投递前须在 clean revision 上重跑。")

    # 文档里的证据表与快照同源: collect 一次就刷一次, 不留手工同步的缝。
    try:
        document = report.update(snapshot_path=path)
        print(f"证据表已刷新: {document}")
    except report.ReportError as error:
        print(f"证据表未刷新: {error}", file=sys.stderr)

    print("\n门禁全部通过。")
    return 0


def command_report() -> int:
    """从已有快照重新生成 results_analysis.md 里的三格证据表.

    返回:
        status: 成功返回 ``0``, 快照缺失或标记不完整返回 ``1``.
    """
    try:
        path = report.update()
    except report.ReportError as error:
        print(f"生成失败: {error}", file=sys.stderr)
        return 1
    print(f"证据表已刷新: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="图 2 数据点的调度与快照采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true",
                           help="列出已注册的数据点及产物状态")
    selection.add_argument("--all", action="store_true", help="跑全部数据点")
    selection.add_argument("--case", help="只跑指定 case id")
    selection.add_argument("--panel", choices=config.PANELS,
                           help="只跑指定图面分格的数据点")
    selection.add_argument("--collect", action="store_true",
                           help="从已有产物收成快照, 不重跑")
    selection.add_argument("--report", action="store_true",
                           help="从已有快照重新生成 results_analysis.md 的证据表")
    parser.add_argument("--check-only", action="store_true",
                        help="只打印将要执行的命令, 不实际运行")
    parser.add_argument("--skip-existing", action="store_true",
                        help="产物已存在时跳过该 case")
    parser.add_argument("--no-collect", action="store_true",
                        help="跑完后不自动执行 collect")
    arguments = parser.parse_args(argv)

    try:
        figure, cases = config.load_cases()
    except config.ConfigError as error:
        print(f"cases.toml 有误: {error}", file=sys.stderr)
        return 2

    if arguments.list:
        return command_list(cases, figure)
    if arguments.collect:
        return command_collect(cases, figure)
    if arguments.report:
        return command_report()

    try:
        selected = config.select(
            cases,
            case_id=arguments.case,
            panel=arguments.panel,
        )
    except config.ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2

    failed = command_run(selected, check_only=arguments.check_only,
                         skip_existing=arguments.skip_existing)
    if failed or arguments.check_only or arguments.no_collect:
        return 1 if failed else 0
    print()
    return command_collect(cases, figure)


if __name__ == "__main__":
    raise SystemExit(main())
