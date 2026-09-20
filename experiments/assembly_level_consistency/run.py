# -*- coding: utf-8 -*-
"""矩阵组装层级 (FA / EA / PA) 求解链正确性的统一驱动.

本模块只做一件事: 按 cases.toml 把每个 (层级, 网格) 数据点以独立子进程跑出来。数值
代码一行都不在这里 —— 全部工况都调 ``examples/lagrange_elasticity/manufactured_convergence_demo.py``,
以制造解求解并统计 L2 误差随 h 的收敛阶。

一条链同时证两件事: 该层级自己收敛到连续解 (末档 L2 阶 >= 门禁), 以及它与 fa 那条链
逐档给出同一个离散解 (跨层级一致性)。后者原先由一个独立的 correctness 面板承担, 在
convergence 只有 FA 一条链时是必要的; 三个层级各有四格之后它已被完全覆盖, 故删除,
理由详见 cases.toml 头部注意 0。

性能与容量的测量在 ``experiments/assembly_level_capability/``。

本模块实现:
1. 调度与编排层: 读取 cases.toml, 以独立子进程 (单线程环境) 跑指定数据点
2. 终端列表与执行进度输出
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config  # noqa: E402

# -----------------------------------------------------------------------------
# 1. 调度层与子进程控制
# -----------------------------------------------------------------------------

def _width(text: str) -> int:
    """终端显示宽度: 东亚全角字符 (含 CJK 与全角标点) 占两列, 其余占一列."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐补空格, 避免 len() 把 CJK 当一列而错位."""
    return text + " " * max(0, width - _width(text))


def command_list(cases: Tuple[config.Case, ...], figure: dict) -> int:
    """列出已注册的数据点.

    一行是一个 (case id, 网格) 组合, 即一个进程一个产物; 同一 case id 的多个网格并列
    在它名下, id 只在首行印一次。不单列 panel 与 scheme: case id 就是
    ``<panel>_<scheme>``, 再各占一列是同一信息排三遍。要筛选用 ``--panel`` 与 ``--mesh``。

    列出的是"这条工况在算什么": 网格类、剖分档次与连续问题。验收口径、产物名与落盘
    状态不在此列 —— 那是 ``compare.py`` 的事, 它读产物本身, 报的是实测值与判定,
    比这里干印一个"有/缺"准确。
    """
    headers = ["case-id", "Mesh", "grid", "problem"]
    rows = []
    previous_id = None
    for c in cases:
        rows.append(["" if c.id == previous_id else c.id, c.mesh_type, c.grid, c.problem])
        previous_id = c.id
    col_widths = [_width(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], _width(str(val)))
    print(f"\nfigure: {figure.get('id')} — {figure.get('title')}")
    print(f"产物目录: {_OUTPUT_DIR}")
    print("  ".join(_pad(h, col_widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * col_widths[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(_pad(str(v), col_widths[i]) for i, v in enumerate(r)))
    print()
    return 0


def command_run(
    selected: Tuple[config.Case, ...],
    check_only: bool = False,
    skip_existing: bool = False,
) -> int:
    """按独立子进程 (单线程环境) 调度执行已选工况."""
    repo_root = Path(__file__).resolve().parents[2]
    total = len(selected)
    failed = 0

    print(f"\n============ assembly_level_consistency 调度执行 ({total} 个任务) ============")
    for idx, case in enumerate(selected, 1):
        artifact_path = case.artifact_path

        if skip_existing and artifact_path.is_file():
            print(f"[{idx}/{total}] 跳过已存在产物: {artifact_path.name}")
            continue

        # 外部脚本自己拼文件名, 本目录只能指定落盘目录; 产物是否真的生成
        # 在子进程退出后按 artifact_path 核对。
        cmd = case.to_command(repo_root) + ["--output-dir", str(_OUTPUT_DIR)]

        if check_only:
            env_str = " ".join(f"{k}={v}" for k, v in config.THREAD_ENV.items())
            print(f"[{idx}/{total}] [dry-run] {case.ref}: {env_str} {' '.join(cmd)}")
            continue

        print(f"[{idx}/{total}] 调度子进程: {case.ref} (输出 -> {artifact_path.name})")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        completed = subprocess.run(cmd, env=case.subprocess_env(), cwd=repo_root)
        elapsed = time.perf_counter() - t_start

        if completed.returncode != 0:
            failed += 1
            print(f"  失败: 退出码 {completed.returncode}, 用时 {elapsed:.1f} s")
        elif not artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {artifact_path}")
        else:
            print(f"  完成: 用时 {elapsed:.1f} s")

    if failed:
        print(f"\n{failed} 个任务执行失败。")
    return failed


# -----------------------------------------------------------------------------
# 2. 主入口与参数路由
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="assembly_level_consistency 实验驱动: 各矩阵组装层级求解链的制造解收敛阶",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点")
    parser.add_argument("--all", action="store_true", help="跑全部工况")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id")
    parser.add_argument("--case", help="指定单个 case id")
    parser.add_argument("--panel", choices=config.PANELS, help="只跑指定面板的数据点")
    parser.add_argument("--mesh", nargs="+", default=None,
                        help="只跑指定网格 (如 quad / hex); 不给则跑所选工况注册的全部网格")
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")

    args = parser.parse_args(argv)

    # ------------------------------------------------ 调度与执行分支
    try:
        figure, cases = config.load_cases()
    except config.ConfigError as error:
        print(f"cases.toml 有误: {error}", file=sys.stderr)
        return 2

    if args.list:
        return command_list(cases, figure)

    target_case_ids: List[str] = []
    panel_filter: Optional[str] = args.panel
    is_all = args.all

    raw_cases: List[str] = []
    if args.cases:
        raw_cases.extend(args.cases)
    if args.case:
        raw_cases.append(args.case)
    for item in raw_cases:
        if item.lower() == "all":
            is_all = True
        elif item.lower() in config.PANELS:
            panel_filter = item.lower()
        else:
            target_case_ids.append(item)

    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case / --cases / --panel / --all 指定要运行的工况。\n"
            "  常用示例:\n"
            "    python run.py --all\n"
            "    python run.py --case convergence_pa              # 该层级注册的全部网格\n"
            "    python run.py --case convergence_fa --mesh hex   # 落到单个数据点\n"
            "    python run.py --panel convergence --check-only\n"
            "  查看全部工况列表请使用: python run.py --list",
            file=sys.stderr,
        )
        return 2

    try:
        selected = config.select(
            cases, case_ids=target_case_ids or None, panel=panel_filter, meshes=args.mesh
        )
    except config.ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2

    failed = command_run(
        selected,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
