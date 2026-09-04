# -*- coding: utf-8 -*-
"""与 experiments/topopt_simp_fa 的同工况对照 (完整接口缩聚的实现门禁).

判据来自误差阶梯的第一级: ``full_trace`` 精确缩聚在代数上与普通 Lagrange
完整组装 (FA) 求解同一离散系统, 因此同工况下逐迭代柔度、体积分数与最终拓扑
应一致到舍入精度。任何超出舍入量级的偏差都是缩聚—恢复—伴随链路的实现误差,
而不是方法误差。

三个阶段:

    1. 参数一致性门禁: 两侧 cases.toml 的问题/离散/拓扑建模/算法参数必须
       字面一致, 唯一允许的差别是求解路径 (FA 完整组装 vs 子结构缩聚);
    2. 逐迭代轨迹对照: 按迭代步对齐 history.json, 给出柔度相对差、体积分数
       绝对差与迭代步数差;
    3. 最终密度场对照: 仅在两侧单元数一致时执行, 给出相对 L2 / L∞ 偏差。

用法:
    python compare.py --case mbb_2d_full_trace
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from config import (
    FA_EXPERIMENT_DIR,
    OUTPUT_DIR,
    ConfigError,
    TopOptCase,
    load,
)

# 门禁字段: 左侧为 FA 注册表字段名, 右侧为本目录工况上的取值.
# density_min 一并纳入门禁: 本目录 OC 固定使用 design_variable_min = 1e-3,
# FA 侧同工况必须以相同取值注册, 否则两条轨迹不可比。
DENSITY_MIN = 1.0e-3


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


def _normalize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _expected_fa_values(case: TopOptCase) -> Dict[str, Any]:
    """本目录工况在 FA 注册表口径下应当取到的参数值."""
    domain_size = [
        case.domain[2 * d + 1] - case.domain[2 * d] for d in range(case.dim)
    ]
    grid = [case.n_sub[d] * case.n_fine[d] for d in range(case.dim)]
    return {
        # A 问题
        "domain": _normalize(domain_size),
        "load": float(case.p_load),
        "emax": float(case.emax),
        "nu": float(case.nu),
        "volfrac": float(case.volfrac),
        # B 离散
        "dimension": float(case.dim),
        "grid": _normalize(grid),
        # C 拓扑建模
        "simp_penalty": float(case.simp_penalty),
        "void_youngs_modulus": float(case.emin),
        "filter_type": case.filter_type,
        "filter_radius": float(case.filter_radius),
        # D 算法 (求解路径除外)
        "optimizer": "oc",
        "move": 0.2,
        "density_min": DENSITY_MIN,
        "tol_change": float(case.tol_change),
    }


def _load_fa_case(case_id: str) -> Dict[str, Any]:
    cases_file = FA_EXPERIMENT_DIR / "cases.toml"
    if not cases_file.is_file():
        raise CompareError(f"FA 注册表不存在: {cases_file}")
    raw = tomllib.loads(cases_file.read_text(encoding="utf-8"))
    for entry in raw.get("cases", []):
        if entry.get("id") == case_id:
            return entry
    raise CompareError(
        f"FA 注册表中没有工况 {case_id}; "
        f"已注册: {[entry.get('id') for entry in raw.get('cases', [])]}"
    )


def _gate(case: TopOptCase) -> Dict[str, Any]:
    """阶段 1: 参数一致性门禁."""
    if not case.fa_reference:
        raise CompareError(
            f"工况 {case.id} 的 fa_reference 为空: FA 侧尚未注册同工况 MBB 参照, "
            "无法执行对照。请先在 experiments/topopt_simp_fa/cases.toml 注册"
            "同参数工况并回填 fa_reference, 本脚本不做静默降级。"
        )
    fa_values = _load_fa_case(case.fa_reference)
    expected = _expected_fa_values(case)

    mismatches: Dict[str, Any] = {}
    for field, wanted in expected.items():
        if field not in fa_values:
            mismatches[field] = {"fa": "<缺失>", "substructure": wanted}
            continue
        actual = _normalize(fa_values[field])
        if actual != wanted:
            mismatches[field] = {"fa": actual, "substructure": wanted}
    if mismatches:
        raise CompareError(
            f"参数一致性门禁失配 ({case.id} vs {case.fa_reference}): "
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    return expected


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise CompareError(f"产物缺失: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_report(
    fa_history: list[dict[str, Any]],
    sub_history: list[dict[str, Any]],
) -> Dict[str, Any]:
    """阶段 2: 逐迭代轨迹对照."""
    fa_by_iter = {int(record["iter"]): record for record in fa_history}
    rows = []
    max_relative_compliance = 0.0
    max_volfrac_difference = 0.0
    for record in sub_history:
        iteration = int(record["iter"])
        reference = fa_by_iter.get(iteration)
        if reference is None:
            continue
        c_fa = float(reference["compliance"])
        c_sub = float(record["compliance"])
        relative = abs(c_sub - c_fa) / abs(c_fa) if c_fa != 0.0 else abs(c_sub)
        volfrac_difference = abs(
            float(record["volfrac"]) - float(reference["volfrac"])
        )
        max_relative_compliance = max(max_relative_compliance, relative)
        max_volfrac_difference = max(max_volfrac_difference, volfrac_difference)
        rows.append(
            {
                "iter": iteration,
                "compliance_fa": c_fa,
                "compliance_substructure": c_sub,
                "compliance_relative_difference": relative,
                "volfrac_difference": volfrac_difference,
            }
        )
    if not rows:
        raise CompareError("两侧 history.json 没有可对齐的迭代步.")
    return {
        "aligned_iterations": len(rows),
        "iterations_fa": len(fa_history),
        "iterations_substructure": len(sub_history),
        "max_compliance_relative_difference": max_relative_compliance,
        "final_compliance_relative_difference": rows[-1][
            "compliance_relative_difference"
        ],
        "max_volfrac_difference": max_volfrac_difference,
        "rows": rows,
    }


def _density_report(fa_dir: Path, sub_dir: Path) -> Dict[str, Any]:
    """阶段 3: 最终密度场对照 (单元序一致是前提)."""
    fa_path = fa_dir / "density_final.npy"
    sub_path = sub_dir / "density_final.npy"
    if not fa_path.is_file() or not sub_path.is_file():
        return {"status": "skipped", "reason": "缺少 density_final.npy"}
    fa_density = np.load(fa_path)
    sub_density = np.load(sub_path)
    if fa_density.shape != sub_density.shape:
        return {
            "status": "skipped",
            "reason": (
                f"单元数不一致: FA {fa_density.shape} vs "
                f"子结构 {sub_density.shape}"
            ),
        }
    return {
        "status": "compared",
        "note": "前提是两侧单元序一致 (FA 结构化网格序 vs 子结构合并后的全局网格序), 首次对照需人工核对一次",
        "relative_l2": _relative_l2(fa_density, sub_density),
        "relative_linf": _relative_linf(fa_density, sub_density),
    }


def compare_case(case: TopOptCase, *, output_root: Optional[Path] = None) -> Dict[str, Any]:
    expected = _gate(case)
    sub_dir = (output_root or OUTPUT_DIR) / case.id
    fa_dir = FA_EXPERIMENT_DIR / "outputs" / case.fa_reference

    fa_summary = _read_json(fa_dir / "summary.json")
    sub_summary = _read_json(sub_dir / "summary.json")
    fa_history = _read_json(fa_dir / "history.json")
    sub_history = _read_json(sub_dir / "history.json")

    report = {
        "case_id": case.id,
        "fa_reference": case.fa_reference,
        "trace": case.trace,
        "reduction": case.reduction,
        "gate": expected,
        "final": {
            "compliance_fa": float(fa_summary["final_compliance"]),
            "compliance_substructure": float(sub_summary["final_compliance"]),
            "volfrac_fa": float(fa_summary["final_volume_fraction"]),
            "volfrac_substructure": float(sub_summary["final_volume_fraction"]),
            "iterations_fa": int(fa_summary["iterations"]),
            "iterations_substructure": int(sub_summary["iterations"]),
        },
        "trajectory": _trajectory_report(fa_history, sub_history),
        "density": _density_report(fa_dir, sub_dir),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="子结构缩聚 vs FA 完整组装的同工况对照"
    )
    parser.add_argument("--case", type=str, default="all", help="工况 ID 或 all")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="覆盖输出根目录 (默认 outputs/)"
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="把报告写到指定 JSON 文件"
    )
    arguments = parser.parse_args()

    try:
        _meta, cases = load()
    except ConfigError as error:
        print(f"[error] {error}")
        return 2

    selected = (
        cases
        if arguments.case == "all"
        else tuple(case for case in cases if case.id == arguments.case)
    )
    if not selected:
        print(f"[error] 未找到工况: {arguments.case}")
        return 2

    reports = []
    failed = False
    for case in selected:
        try:
            report = compare_case(case, output_root=arguments.output_dir)
        except CompareError as error:
            failed = True
            print(f"[compare] {case.id}: FAIL\n  {error}")
            continue
        reports.append(report)
        trajectory = report["trajectory"]
        print(
            f"[compare] {case.id} vs {case.fa_reference}: "
            f"柔度最大相对差 {trajectory['max_compliance_relative_difference']:.3e}, "
            f"末步相对差 {trajectory['final_compliance_relative_difference']:.3e}, "
            f"迭代 {trajectory['iterations_substructure']} vs "
            f"{trajectory['iterations_fa']} 步, "
            f"密度场 {report['density']['status']}"
        )

    if arguments.report is not None and reports:
        arguments.report.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[compare] 报告已写入 {arguments.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
