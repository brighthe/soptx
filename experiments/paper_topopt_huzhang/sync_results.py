"""把 outputs/ 下选定的运行固化到入库的论文证据目录 results/.

``outputs/<case>/<run>/`` 是 run 脚本的运行产物, 不入库; ``results/`` 是 table.py / plot.py
与论文表图的数据来源, 入库. 本脚本按算例把一次运行原样拷入 results/, 并在目标处写
``synced.json`` 记录来源路径、时间与各文件的 sha256:

- ``manufactured-convergence``: 保持入库布局, ``manufactured_convergence.json`` 拷到
  ``results/manufactured-convergence/``, ``table5_*.md`` 拷到 ``results/tables/``;
- 其余算例: ``outputs/<case>/<run>/`` 整目录拷到 ``results/<case>/<run>/``, 逐步帧 ``iterations/``
  与 ``evolution.pvd`` (正常已迁移到 Windows 查看目录) 不拷.

目标已存在时拒绝覆盖, 确认要替换请加 ``--force``. 只依赖标准库, 不导入 soptx.

用法::

    python sync_results.py manufactured-convergence relaxation-on
    python sync_results.py cantilever-middle-2d-stress analyzer-huzhang__order-2 --force
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
MANUFACTURED_CASE = "manufactured-convergence"
MANUFACTURED_DATA = "manufactured_convergence.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _stamp(source: Path, copied: list[Path], target_dir: Path) -> None:
    record = {
        "synced_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source.relative_to(EXPERIMENT_DIR)),
        "files": {
            str(path.relative_to(RESULTS_DIR)): _sha256(path)
            for path in sorted(copied) if path.is_file()
        },
    }
    (target_dir / "synced.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sync_manufactured(run_dir: Path, force: bool) -> None:
    """制造解收敛算例: 数据与表格分别落到 results/manufactured-convergence/ 与 results/tables/."""
    data = run_dir / MANUFACTURED_DATA
    if not data.exists():
        raise SystemExit(f"运行目录缺少 {MANUFACTURED_DATA}: {run_dir}")
    copied = []
    target = RESULTS_DIR / MANUFACTURED_CASE / MANUFACTURED_DATA
    _copy(data, target, force); copied.append(target)
    for table in sorted(run_dir.glob("table5_*.md")):
        target = RESULTS_DIR / "tables" / table.name
        _copy(table, target, force); copied.append(target)
    _stamp(run_dir, copied, RESULTS_DIR / MANUFACTURED_CASE)


def sync_run(case: str, run_dir: Path, force: bool) -> None:
    """优化算例: 整个运行目录拷到 results/<case>/<run>/."""
    target = RESULTS_DIR / case / run_dir.name
    _copy(run_dir, target, force)
    _stamp(run_dir, list(target.rglob("*")), target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 outputs/<case>/<run> 固化到 results/")
    parser.add_argument("case", help="算例目录名, 如 manufactured-convergence")
    parser.add_argument("run", help="运行目录名, 如 relaxation-on")
    parser.add_argument("--force", action="store_true", help="允许覆盖 results/ 中已有的同名产物")
    args = parser.parse_args(argv)

    run_dir = OUTPUTS_DIR / args.case / args.run
    if not run_dir.is_dir():
        parser.error(f"运行目录不存在: {run_dir}")
    if args.case == MANUFACTURED_CASE:
        sync_manufactured(run_dir, args.force)
    else:
        sync_run(args.case, run_dir, args.force)
    print(f"[OK] {run_dir.relative_to(EXPERIMENT_DIR)} -> results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
