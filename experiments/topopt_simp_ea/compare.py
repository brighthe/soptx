# -*- coding: utf-8 -*-
"""Tier 1 统一条件对照: EA-cg vs FA-cg, 两侧只差 operator_level.

对每个填写 ``fa_reference`` 的工况执行三段对照:

1. 参数一致性门禁: EA 运行与 FA 对照运行的参数逐项字面相等, 并核对两侧
   ``summary.json`` 的 ``config`` 块与运行口径; 失配即报错退出;
2. 轨迹对照: 两侧 ``history.json`` 的逐迭代柔顺度相对差、体积分数差、
   迭代数与 change 历史;
3. 锁步对照: 取 FA 侧密度快照 (iter 1 / 中期 / 收敛), 进程内用
   ``build_components`` 分别构建 FA 与 EA 分析链, 同一密度下比较位移、
   灵敏度与两侧真残差。

默认对照全部注册工况的基准运行; 带 override 的运行用与 run.py 相同的
``--case ID --override KEY=VALUE`` 指定, 脚本在 FA 侧重放同一组 override
找到配对运行。结论写入 ``outputs/compare/<run_id>.json``
(= ``outputs/compare/<工况 id>/<参数标签>.json``)。阈值超限先视为待调查,
不在脚本里放宽。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from fealpy.backend import backend_manager as bm
from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import (
    FA_EXPERIMENT_DIR,
    GATE_FIELDS,
    OUTPUT_DIR,
    ConfigError,
    TopOptCase,
    build_overridden_case,
    config_values,
    load,
    parse_override_args,
)
from pipeline import build_components

# ---- 判定阈值 (超限先调查, 不直接放宽) ----
LOCKSTEP_U_RTOL = 1.0e-8        # 锁步位移相对 L2 差
LOCKSTEP_DC_RTOL = 1.0e-8       # 锁步灵敏度相对 L2 差
TRAJECTORY_COMPLIANCE_RTOL = 1.0e-6  # 逐迭代柔顺度相对差
DENSITY_THRESHOLD = 0.5         # 最终拓扑阈值化水平
MAX_THRESHOLD_MISMATCH = 0      # 阈值化失配单元数

# 门禁清单 = config.GATE_FIELDS (标了 A/B/C/D 轴的全部字段), 不在这里另抄
# 一份: 抄一份就意味着 cases.toml 新增参数时门禁会静默漏掉它。两侧的清单由
# _fa_module() 加载后逐项核对, 不一致直接报错。


class CompareError(RuntimeError):
    """对照前置条件不满足 (参数失配或产物缺失)."""


def _relative_l2(reference: np.ndarray, other: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(other))
    return float(np.linalg.norm(other - reference) / denominator)


def _relative_linf(reference: np.ndarray, other: np.ndarray) -> float:
    denominator = float(np.max(np.abs(reference)))
    if denominator == 0.0:
        return float(np.max(np.abs(other)))
    return float(np.max(np.abs(other - reference)) / denominator)


def _fa_module():
    """按路径加载 FA 侧的 config 模块.

    FA 与 EA 是两个平行实验目录, 各有一个顶层 config.py, 不能同时 import;
    这里显式按文件加载, 从而复用 FA 自己的解析与校验, 不再在本文件里重写
    一遍 FA 的默认值规则 (那份复制品曾经就与 FA 的实际默认值脱节)。
    """
    path = FA_EXPERIMENT_DIR / "config.py"
    if not path.is_file():
        raise CompareError(f"FA 配置不存在: {path}")
    if "fa_config" in sys.modules:
        return sys.modules["fa_config"]
    spec = importlib.util.spec_from_file_location("fa_config", path)
    module = importlib.util.module_from_spec(spec)
    # dataclass 处理注解时要能从 sys.modules 找回自己所在模块。
    sys.modules["fa_config"] = module
    spec.loader.exec_module(module)
    if tuple(module.GATE_FIELDS) != tuple(GATE_FIELDS):
        raise CompareError(
            "两侧 GATE_FIELDS 不一致, 锁步对照无从谈起: "
            f"仅 FA 有 {sorted(set(module.GATE_FIELDS) - set(GATE_FIELDS))}, "
            f"仅 EA 有 {sorted(set(GATE_FIELDS) - set(module.GATE_FIELDS))}."
        )
    return module


def find_fa_run(case: TopOptCase) -> Any:
    """按 fa_reference 找到 FA 侧工况, 再重放同一组 override 得到配对运行.

    两侧同一组 override 推导出同一个 run_id, 不必逐次登记引用。名字只用来
    配对, 参数是否真的逐项一致由 check_parameter_gate 对全部 GATE_FIELDS
    断言 —— 基准参数在两侧注册表里写歪了, 门禁会点名是哪个字段。
    """
    module = _fa_module()
    _, fa_cases = module.load()
    matched = [fa_case for fa_case in fa_cases if fa_case.id == case.fa_reference]
    if len(matched) != 1:
        raise CompareError(
            f"FA 侧没有工况 {case.fa_reference} (EA 运行 {case.run_id} 的 "
            "fa_reference); FA 侧现有: "
            + (", ".join(fa_case.id for fa_case in fa_cases) or "无")
        )
    fa_case = matched[0]
    if case.overrides:
        try:
            fa_case = module.build_overridden_case(fa_case, dict(case.overrides))
        except module.ConfigError as error:
            raise CompareError(f"FA 侧无法重放 override: {error}") from error
    return fa_case


def _normalize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def check_parameter_gate(case: TopOptCase, fa_case: Any) -> dict[str, Any]:
    """逐项断言 EA 运行与 FA 对照运行的影响结果参数字面一致.

    run_id 相同只说明 id 与 override 标签相同, 其余字段是否一致必须在这里
    逐项断言; 真出事时报 "哪个字段不一样" 也比报 "名字对不上" 有用得多。
    """
    fa_values = config_values(fa_case)
    ea_values = config_values(case)
    mismatches = {
        field: {
            "fa": _normalize(fa_values[field]),
            "ea": _normalize(ea_values[field]),
        }
        for field in GATE_FIELDS
        if _normalize(fa_values[field]) != _normalize(ea_values[field])
    }
    if mismatches:
        raise CompareError(
            f"参数一致性门禁失配 ({case.run_id} vs {fa_case.run_id}): "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return {field: _normalize(fa_values[field]) for field in GATE_FIELDS}


def _load_artifacts(
    output_dir: Path, expected: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """读取一侧的运行产物, 核对 summary 的运行口径与参数快照."""
    summary_path = output_dir / "summary.json"
    history_path = output_dir / "history.json"
    density_path = output_dir / "density_final.vtu"
    missing = [
        str(path)
        for path in (summary_path, history_path, density_path)
        if not path.is_file()
    ]
    if missing:
        raise CompareError("运行产物缺失: " + ", ".join(missing))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if _normalize(summary.get(key)) != _normalize(value):
            raise CompareError(
                f"{summary_path} 的 {key}={summary.get(key)!r} 与预期 {value!r} 不符."
            )
    # 产物必须是当前注册表这组参数跑出来的, 而不是改参数前留下的旧结果。
    recorded = summary.get("config", {})
    stale = {
        field: {"注册表": value, "产物": recorded.get(field, "<缺失>")}
        for field, value in config.items()
        if _normalize(recorded.get(field)) != _normalize(value)
    }
    if stale:
        raise CompareError(
            f"{summary_path} 与注册表参数不一致 (产物已过期, 需重跑): "
            + json.dumps(stale, ensure_ascii=False)
        )
    return {
        "summary": summary,
        "history": json.loads(history_path.read_text(encoding="utf-8")),
        "density_final": read_vtu_cell_data(density_path),
        # 逐迭代密度全在 vtu/ 下, 锁步取样点由 compare_lockstep 自己挑。
        "vtu_dir": output_dir / "vtu",
    }


def compare_trajectory(
    fa_history: list[dict[str, Any]],
    ea_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """逐迭代轨迹对照."""
    overlap = min(len(fa_history), len(ea_history))
    compliance_rel = [
        abs(ea_history[i]["compliance"] - fa_history[i]["compliance"])
        / abs(fa_history[i]["compliance"])
        for i in range(overlap)
    ]
    volfrac_abs = [
        abs(ea_history[i]["volfrac"] - fa_history[i]["volfrac"])
        for i in range(overlap)
    ]
    change_abs = [
        abs(ea_history[i]["change"] - fa_history[i]["change"])
        for i in range(overlap)
    ]
    return {
        "fa_iterations": len(fa_history),
        "ea_iterations": len(ea_history),
        "iterations_equal": len(fa_history) == len(ea_history),
        "overlap_iterations": overlap,
        "max_compliance_rel_diff": float(max(compliance_rel)),
        "max_volfrac_abs_diff": float(max(volfrac_abs)),
        "max_change_abs_diff": float(max(change_abs)),
        "final_compliance_rel_diff": float(
            abs(ea_history[-1]["compliance"] - fa_history[-1]["compliance"])
            / abs(fa_history[-1]["compliance"])
        ),
    }


def _solve_at_density(case: TopOptCase, operator_level: str, snapshot: np.ndarray) -> dict[str, Any]:
    """在给定密度下构建一侧分析链, 求解并返回 u、dc、真残差与求解诊断."""
    components = build_components(case, operator_level)
    density = components.density
    density[:] = bm.array(snapshot, dtype=bm.float64)
    state = components.analyzer.solve_state(rho_val=density)
    uh = state["displacement"]
    dc = components.objective.jac(density, state=state)
    compliance = components.objective.fun(density, state=state)

    analyzer = components.analyzer
    K0 = analyzer.assemble_stiff_matrix(rho_val=density)
    F0 = analyzer.assemble_body_force_vector()
    K, F = analyzer.apply_bc(K0, F0)
    operator = analyzer._as_iterative_operator(K)
    residual_vector = np.asarray(
        bm.to_numpy(operator @ uh[:]) - bm.to_numpy(F[:]), dtype=np.float64
    )
    force_norm = float(np.linalg.norm(bm.to_numpy(F[:])))
    return {
        "u": np.asarray(bm.to_numpy(uh[:]), dtype=np.float64),
        "dc": np.asarray(bm.to_numpy(dc[:]), dtype=np.float64),
        "compliance": float(compliance),
        "true_residual": float(np.linalg.norm(residual_vector) / force_norm),
        "solver": state["solver"],
    }


def _lockstep_iterations(n_iterations: int) -> list[int]:
    """锁步取样点: 首次 / 中期 / 收敛。

    逐迭代密度都在 vtu/ 下, 取三点而非全部是为了控制重解成本: 算子层级的
    系统性偏差在这三点上就会暴露, 全程重解只是把同一结论重复几百遍。
    """
    return sorted({1, (n_iterations + 1) // 2, n_iterations})


def compare_lockstep(
    case: TopOptCase,
    fa_vtu_dir: Path,
    n_iterations: int,
) -> dict[str, Any]:
    """在 FA 密度快照上做 FA/EA 锁步对照."""
    iterations = _lockstep_iterations(n_iterations)
    records = {}
    for iteration in iterations:
        snapshot_path = fa_vtu_dir / f"density_iter_{iteration:04d}.vtu"
        if not snapshot_path.is_file():
            raise CompareError(
                f"FA 侧 {case.fa_reference} 缺少密度快照 {snapshot_path}."
            )
        snapshot = read_vtu_cell_data(snapshot_path)
        fa_side = _solve_at_density(case, "fa", snapshot)
        ea_side = _solve_at_density(case, "ea", snapshot)
        records[str(iteration)] = {
            "u_rel_l2": _relative_l2(fa_side["u"], ea_side["u"]),
            "dc_rel_l2": _relative_l2(fa_side["dc"], ea_side["dc"]),
            "dc_rel_linf": _relative_linf(fa_side["dc"], ea_side["dc"]),
            "compliance_rel_diff": abs(
                ea_side["compliance"] - fa_side["compliance"]
            ) / abs(fa_side["compliance"]),
            "fa_true_residual": fa_side["true_residual"],
            "ea_true_residual": ea_side["true_residual"],
            "fa_cg_niter": int(fa_side["solver"]["niter"]),
            "ea_cg_niter": int(ea_side["solver"]["niter"]),
            "fa_cg_converged": bool(fa_side["solver"]["converged"]),
            "ea_cg_converged": bool(ea_side["solver"]["converged"]),
        }
    return {
        "snapshot_iterations": iterations,
        "records": records,
        "max_u_rel_l2": max(r["u_rel_l2"] for r in records.values()),
        "max_dc_rel_l2": max(r["dc_rel_l2"] for r in records.values()),
        "max_dc_rel_linf": max(r["dc_rel_linf"] for r in records.values()),
    }


def compare_topology(fa_density: np.ndarray, ea_density: np.ndarray) -> dict[str, Any]:
    """最终拓扑对照."""
    fa = np.asarray(fa_density, dtype=np.float64).reshape(-1)
    ea = np.asarray(ea_density, dtype=np.float64).reshape(-1)
    if fa.shape != ea.shape:
        raise CompareError(f"两侧最终密度维数不一致: {fa.shape} vs {ea.shape}.")
    mismatch = int(np.sum((fa > DENSITY_THRESHOLD) != (ea > DENSITY_THRESHOLD)))
    return {
        "density_rel_l2": _relative_l2(fa, ea),
        "density_max_abs_diff": float(np.max(np.abs(fa - ea))),
        "threshold_mismatch_cells": mismatch,
        "n_cells": int(fa.size),
    }


def compare_case(case: TopOptCase) -> dict[str, Any]:
    """执行一个 Tier 1 运行的三段对照并给出 PASS/FAIL."""
    fa_case = find_fa_run(case)
    gate = check_parameter_gate(case, fa_case)
    # 两侧产物目录都由各自的参数推导, 不需要在这里拼路径; run_id 一并核对,
    # 确保读到的确实是这一次 (基准或这组 override 的) 运行。
    fa = _load_artifacts(
        fa_case.output_dir,
        {"method": "FA-SIMP", "assembly_level": "full", "run_id": fa_case.run_id},
        gate,
    )
    ea = _load_artifacts(
        case.output_dir,
        {"method": "EA-SIMP", "assembly_level": "element", "run_id": case.run_id},
        gate,
    )
    trajectory = compare_trajectory(fa["history"], ea["history"])
    lockstep = compare_lockstep(case, fa["vtu_dir"], len(fa["history"]))
    topology = compare_topology(fa["density_final"], ea["density_final"])
    checks = {
        "lockstep_u": lockstep["max_u_rel_l2"] <= LOCKSTEP_U_RTOL,
        "lockstep_dc": lockstep["max_dc_rel_l2"] <= LOCKSTEP_DC_RTOL,
        "trajectory_compliance": (
            trajectory["max_compliance_rel_diff"] <= TRAJECTORY_COMPLIANCE_RTOL
        ),
        "iterations_equal": trajectory["iterations_equal"],
        "topology_threshold_match": (
            topology["threshold_mismatch_cells"] <= MAX_THRESHOLD_MISMATCH
        ),
    }
    return {
        "ea_run_id": case.run_id,
        "fa_run_id": fa_case.run_id,
        "overrides": dict(case.overrides) or None,
        "thresholds": {
            "lockstep_u_rtol": LOCKSTEP_U_RTOL,
            "lockstep_dc_rtol": LOCKSTEP_DC_RTOL,
            "trajectory_compliance_rtol": TRAJECTORY_COMPLIANCE_RTOL,
            "density_threshold": DENSITY_THRESHOLD,
            "max_threshold_mismatch": MAX_THRESHOLD_MISMATCH,
        },
        "parameter_gate": gate,
        "trajectory": trajectory,
        "lockstep": lockstep,
        "topology": topology,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EA vs FA 统一条件 Tier 1 对照")
    parser.add_argument(
        "--case", metavar="ID", help="只对照一个注册工况; 默认全部填写 fa_reference 的工况"
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="对照带 override 的运行 (写法与 run.py 相同, 仅配合 --case); "
        "脚本在 FA 侧重放同一组 override 定位配对运行",
    )
    args = parser.parse_args()
    try:
        overrides = parse_override_args(args.override)
    except ConfigError as error:
        parser.error(str(error))
    if overrides and not args.case:
        parser.error("--override 只能与 --case 一起使用.")
    bm.set_backend("numpy")
    _, cases = load()
    selected = tuple(
        case for case in cases
        if case.fa_reference and (args.case is None or case.id == args.case)
    )
    if not selected:
        parser.error("没有匹配的 compare 工况.")
    if overrides:
        try:
            selected = (build_overridden_case(selected[0], overrides),)
        except ConfigError as error:
            parser.error(str(error))

    compare_dir = OUTPUT_DIR / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for case in selected:
        print(f"[compare] {case.run_id} vs FA/{case.fa_reference}")
        result = compare_case(case)
        # 一次运行一个结论文件, 相对路径就是 run_id, 与产物目录一一对应。
        target = compare_dir / f"{case.run_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_passed &= result["passed"]
        print(f"[compare] {'PASS' if result['passed'] else 'FAIL'} -> {target}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
