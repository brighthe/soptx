"""把 outputs/ 下选定的运行固化到入库的论文证据目录 results/.

``outputs/<case>/<run>/`` 是 run 脚本的运行产物, 不入库; ``results/`` 是 table.py / plot.py
与论文表图的数据来源, 入库. 本脚本把一次运行整目录拷到 ``results/<case>/<run>/``, 逐步帧
``iterations/`` 与 ``evolution.pvd`` (正常已迁移到 Windows 查看目录) 不拷; 溯源信息见各运行
自带的 provenance 戳记.

加 ``--paper-tables`` 时另把运行目录里的论文表 ``table*.md`` (如制造解算例的 ``table5_1.md`` /
``table5_2.md``) 拷到 ``results/tables/``, 由命令行显式指定哪次运行进论文.

目标已存在时拒绝覆盖, 确认要替换请加 ``--force``. 只依赖标准库, 不导入 soptx.

用法::

    python sync_results.py manufactured-convergence relaxation-off --paper-tables
    python sync_results.py manufactured-convergence relaxation-on
    python sync_results.py cantilever-middle-2d-stress analyzer-huzhang__order-2 --force
    python sync_results.py bearing/nu-0.4999 analyzer-lfem__order-2   # 带材料分组的算例
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PAPER_TABLES_DIR = RESULTS_DIR / "tables"


def _copy(source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        raise SystemExit(f"目标已存在, 拒绝覆盖: {target} (确认要替换请加 --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        # 逐步帧与迁移清单不入库: 正常情况下已由 run 脚本迁移到 Windows 查看目录, 迁移被跳过时仍留在
        # 运行目录, 这里一律不拷
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            "iterations", "evolution.pvd", ".visualization_sync.json", ".visualization_sync.lock"))
    else:
        shutil.copy2(source, target)


def sync_run(case: str, run_dir: Path, force: bool) -> None:
    """整个运行目录拷到 results/<case>/<run>/."""
    _copy(run_dir, RESULTS_DIR / case / run_dir.name, force)


def sync_paper_tables(run_dir: Path, force: bool) -> None:
    """运行目录里的论文表 table*.md 拷到 results/tables/."""
    tables = sorted(run_dir.glob("table*.md"))
    if not tables:
        raise SystemExit(f"运行目录没有论文表 table*.md: {run_dir}")
    for table in tables:
        _copy(table, PAPER_TABLES_DIR / table.name, force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 outputs/<case>/<run> 固化到 results/")
    parser.add_argument("case", help="算例目录名, 如 manufactured-convergence")
    parser.add_argument("run", help="运行目录名, 如 relaxation-off")
    parser.add_argument("--paper-tables", action="store_true",
                        help="另把运行目录里的论文表 table*.md 拷到 results/tables/")
    parser.add_argument("--force", action="store_true", help="允许覆盖 results/ 中已有的同名产物")
    args = parser.parse_args(argv)

    run_dir = OUTPUTS_DIR / args.case / args.run
    if not run_dir.is_dir():
        parser.error(f"运行目录不存在: {run_dir}")
    sync_run(args.case, run_dir, args.force)
    if args.paper_tables:
        sync_paper_tables(run_dir, args.force)
    print(f"[OK] {run_dir.relative_to(EXPERIMENT_DIR)} -> results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
