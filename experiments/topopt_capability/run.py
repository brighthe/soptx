# -*- coding: utf-8 -*-
"""CLI 调度入口 (TopOpt 平台能力验证)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import collect
import config


def run_case(case: config.Case) -> int:
    """运行单个用例, 并按 ``collect_as`` 把产物收编进快照基目录.

    产物目录由 ``case.outdir`` 决定; ``--outdir`` 必须指向该子目录而非 ``outputs``
    根, 否则不同规模的运行会互相覆盖.
    """
    case.outdir_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(case.script_path)]
    cmd.extend(case.args)
    cmd.extend(["--outdir", str(case.outdir_path)])

    print(f"[run] {case.id}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=config.REPOSITORY_ROOT)
    if result.returncode != 0:
        return result.returncode

    if case.collect_as is not None:
        target = case.collected_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.artifact_path, target)
        print(f"[run] {case.id}: 产物收编 -> {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TopOpt 平台能力验证调度入口")
    parser.add_argument("--list", action="store_true", help="列出全部用例")
    parser.add_argument("--collect", action="store_true", help="汇编数据快照")
    parser.add_argument("--all", action="store_true", help="运行全部用例并汇编快照")
    args = parser.parse_args()

    fig, cases = config.load()

    if args.list:
        print(f"=== {fig.get('title', 'TopOpt 平台验证')} 用例列表 ===")
        for c in cases:
            status = "已生成" if c.collected_path.is_file() else "未生成"
            print(f"  [{c.id}] ({c.panel}) {c.summary} -> {status}")
        return 0

    if args.all:
        for c in cases:
            ret = run_case(c)
            if ret != 0:
                print(f"[error] 用例 {c.id} 运行失败, 退出码 {ret}")
                return ret
        out = collect.write(collect.collect())
        print(f"[success] 快照已汇编入库: {out}")
        return 0

    if args.collect:
        out = collect.write(collect.collect())
        print(f"[success] 快照已汇编入库: {out}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
