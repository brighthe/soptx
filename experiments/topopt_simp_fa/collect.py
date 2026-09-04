# -*- coding: utf-8 -*-
"""FA 变密度拓扑优化结果验收与汇总."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import (
    EXPERIMENT_DIR,
    FIGURE_DATA_DIR,
    NONTRIVIAL_STD,
    OUTPUT_DIR,
    VOLUME_TOLERANCE,
    ConfigError,
    TopOptCase,
    build_overridden_case,
    config_values,
    load,
)


def validate_result(
    case: TopOptCase,
    density: np.ndarray,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查数值结果、收敛性和完整 VTU 迭代历史."""
    rho = np.asarray(density, dtype=np.float64).reshape(-1)
    complete = bool(history) and all(
        record.get("iter") == index
        for index, record in enumerate(history, start=1)
    )
    scalar_names = ("compliance", "volfrac", "change", "iteration_time")
    history_finite = complete and all(
        all(np.isfinite(float(record[name])) for name in scalar_names)
        for record in history
    )
    density_finite = bool(rho.size and np.all(np.isfinite(rho)))
    bounds = density_finite and bool(
        rho.min() >= case.density_min - 1.0e-10
        and rho.max() <= 1.0 + 1.0e-10
    )
    final_volfrac = float(history[-1]["volfrac"]) if history else float("nan")
    volume_error = abs(final_volfrac - case.volfrac)
    density_std = float(rho.std()) if density_finite else float("nan")
    vtu_dir = case.output_dir / "vtu"
    expected_vtu = [
        vtu_dir / f"density_iter_{int(record['iter']):04d}.vtu"
        for record in history
    ]
    vtu_history_complete = bool(
        complete
        and (vtu_dir / "density_history.pvd").is_file()
        and all(path.is_file() for path in expected_vtu)
    )
    checks = {
        "history_complete": complete,
        "history_finite": history_finite,
        "density_finite": density_finite,
        "density_bounds": bounds,
        "volume_constraint": bool(volume_error <= VOLUME_TOLERANCE),
        "compliance_reduced": bool(
            history
            and history[-1]["compliance"]
            <= history[0]["compliance"] * (1.0 + 1.0e-10)
        ),
        "converged": bool(history and history[-1]["change"] <= case.tol_change),
        "nontrivial_topology": bool(density_std >= NONTRIVIAL_STD),
        "vtu_history_complete": vtu_history_complete,
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "volume_error": volume_error,
        "density_min": float(rho.min()) if density_finite else None,
        "density_max": float(rho.max()) if density_finite else None,
        "density_std": density_std if np.isfinite(density_std) else None,
    }


def load_case_result(case: TopOptCase) -> dict[str, Any]:
    """读取一次运行的三个标准产物并重新执行验收."""
    summary_path = case.output_dir / "summary.json"
    history_path = case.output_dir / "history.json"
    density_path = case.output_dir / "density_final.vtu"
    missing = [
        str(path)
        for path in (summary_path, history_path, density_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("运行产物缺失: " + ", ".join(missing))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    density = read_vtu_cell_data(density_path)
    if summary.get("method") != "FA-SIMP" or summary.get("assembly_level") != "full":
        raise ValueError(f"运行 {case.run_id} 不是 FA-SIMP 完整组装口径.")
    if len(history) != int(summary["iterations"]):
        raise ValueError(f"运行 {case.run_id} 的历史长度与迭代数不一致.")
    return {
        "summary": {**summary, "validation": validate_result(case, density, history)},
        "history_file": str(history_path.relative_to(EXPERIMENT_DIR)),
        "density_file": str(density_path.relative_to(EXPERIMENT_DIR)),
    }


def _replay_case(
    registered: dict[str, TopOptCase], run_dir: Path
) -> TopOptCase | str:
    """由产物目录里的 summary 在当前注册表上重放出这一次运行的 TopOptCase.

    重放成功、run_id 与目录路径一致、且 summary 里的 config 快照与重放结果逐字段
    相同才算认领; 否则返回原因文本 (注册表已删掉该工况、override 在当前注册表上
    非法、目录路径与参数推导不符、或注册表基准参数改过而产物还是旧参数跑的),
    由 collect 记为 unclaimed, 不当成有效结果, 也不静默跳过。
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    base = registered.get(summary.get("case_id"))
    if base is None:
        return f"case_id {summary.get('case_id')!r} 不在当前注册表"
    overrides = summary.get("overrides") or {}
    try:
        case = build_overridden_case(base, overrides) if overrides else base
    except ConfigError as error:
        return f"override 在当前注册表上无效: {error}"
    if case.run_id != run_dir.relative_to(OUTPUT_DIR).as_posix():
        return f"目录路径与参数推导的 run_id {case.run_id!r} 不符"
    expected = json.loads(json.dumps(config_values(case)))
    recorded = summary.get("config") or {}
    stale = sorted(
        name for name in expected if recorded.get(name) != expected[name]
    )
    if stale:
        return f"config 快照与当前注册表不符: {', '.join(stale)}"
    return case


def collect() -> dict[str, Any]:
    """汇总 outputs/ 下全部运行 (基准 + override), 并列出未运行的注册工况.

    运行清单从 outputs/ 枚举而不是从注册表推导: override 运行不在注册表里,
    只有落盘的 summary.json 知道它们存在。每个目录都在当前注册表上重放一遍
    再验收, 保证汇总里的参数快照与产物一致。
    """
    meta, cases = load()
    registered = {case.id: case for case in cases}
    runs: dict[str, dict[str, Any]] = {}
    unclaimed: dict[str, str] = {}
    # 产物目录两层 outputs/<工况 id>/<参数标签>/, run_id 就是这两段的相对路径。
    for summary_path in sorted(OUTPUT_DIR.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        # .partial / .previous 是 driver.py 原子发布的中间态, 不是产物目录。
        if run_dir.suffix in {".partial", ".previous"}:
            continue
        case = _replay_case(registered, run_dir)
        if isinstance(case, str):
            unclaimed[run_dir.relative_to(OUTPUT_DIR).as_posix()] = case
            continue
        runs[case.run_id] = {
            **load_case_result(case),
            "case_id": case.id,
            "overrides": dict(case.overrides) or None,
        }
    return {
        "meta": meta,
        "runs": runs,
        "completed_run_ids": sorted(runs),
        # 注册工况的基准运行尚未落盘的; override 运行没有 "待运行" 概念。
        "pending_case_ids": [case.id for case in cases if case.id not in runs],
        "unclaimed_run_dirs": unclaimed,
    }


def write(snapshot: dict[str, Any], path: Path | None = None) -> Path:
    target = path or FIGURE_DATA_DIR / "fa_topopt_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    snapshot = collect()
    output = write(snapshot)
    print(f"[collect] 汇总快照已写入: {output}")
    print(
        f"[collect] 运行 {len(snapshot['runs'])} 次, "
        f"待运行工况 {len(snapshot['pending_case_ids'])} 个, "
        f"未认领目录 {len(snapshot['unclaimed_run_dirs'])} 个"
    )
    for name, reason in snapshot["unclaimed_run_dirs"].items():
        print(f"[collect] 未认领 outputs/{name}: {reason}")
