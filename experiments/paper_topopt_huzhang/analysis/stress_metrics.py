# -*- coding: utf-8 -*-
"""应力算例的插图场数据导出: 运行目录解析与 npz 冻结重分析.

经 ``run_cantilever_stress.build`` (与优化运行同一份代码) 重建分析链, 对最终设计做
冻结重分析, 导出论文 5.2.3 节插图所需的场量 (原 ``export_fig_data.py``,
2026-09-01 并入; 同期并入的梯度校验与冻结指标两段已随论文结果收口删除).

入口函数由 ``plot.py`` 的 ``COMMAND_MODULES`` 派发, 本模块不直接执行.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import EXPERIMENT_DIR, OUTPUT_DIR, REPOSITORY_ROOT, bootstrap_source_path

bootstrap_source_path()

import run_cantilever_stress  # noqa: E402
from soptx.postprocess.stress_report import StressPostProcessor  # noqa: E402


# ============================================ 一、运行目录解析
# 在 results/cantilever-middle-2d-stress 下按 run_cantilever_stress 的目录名定位正式运行.

OUT = OUTPUT_DIR / run_cantilever_stress.CASE_ID


def read_vtu_cell_density(path: Path | str) -> np.ndarray:
    """解析 VTKFile appended-raw 格式的 CellData density 数组."""
    return read_vtu_cell_data(path, "density")


def resolve_run_dir(
    method: str,
    order: int,
    *,
    announce: bool = True,
) -> Path:
    """定位 run_cantilever_stress 写出的正式运行, 并核对其收敛与约束元数据.

    目录名由 ``run_cantilever_stress.run_label`` 给出; 缺文件、未收敛或元数据不是
    apparent 时直接报错, 不另找替代目录.
    """
    run_dir = OUT / run_cantilever_stress.run_label(method, order)
    required = ("density_final.vtu", "summary.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"缺少 {run_dir} 下的 {missing}; 先运行 run_cantilever_stress.py "
            f"--analyzer {method} --order {order}."
        )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("converged", False)):
        raise ValueError(
            f"{run_dir}: 该运行未收敛 ({summary.get('termination_reason')}); "
            "未收敛构型不作为插图与冻结评估的来源."
        )
    formulation = summary.get("stress_constraint_formulation")
    if formulation != "apparent":
        raise ValueError(
            f"{run_dir}: summary.stress_constraint_formulation 为 {formulation!r}, 不是 apparent."
        )
    if announce:
        print(json.dumps({
            "run_source": f"{method}-k{order}",
            "run_dir": str(run_dir),
            "stress_constraint_formulation": formulation,
        }, ensure_ascii=False), flush=True)
    return run_dir


# ============================================ 二、插图场数据导出 (npz)
# 论文 5.2.3 节的图 5.9 (b)(d) 不直接读优化历程, 而是读带统一约束协议标签的
# results/cantilever-middle-2d-stress/postprocess/lfem_constraint-apparent/
# fig_data_<run>.npz; 本段是其唯一来源.
# npz 写在 results/ 下随结果入库; 数字的溯源依据是各 run 目录下
# summary.json 自带的运行戳记 (provenance.run_stamp).

CASE_ID = run_cantilever_stress.CASE_ID
POSTPROCESS_DIR = OUTPUT_DIR / CASE_ID / "postprocess" / "lfem_constraint-apparent"
# 插图只用 k=3 主对比的两次运行 (plots/stress_cubic_convergence 的主应力面板);
# 其余阶次的场图改由 discretization_probe 的 fields.npz 供数.
RUNS: dict[str, tuple[str, int]] = {
    "lfem-k3": ("lfem", 3),
    "huzhang-k3": ("huzhang", 3),
}


def export_run(name: str) -> dict[str, Any]:
    """对单次运行做冻结重分析, 返回 npz 待写入的场量字典."""
    method, order = RUNS[name]
    run_dir = resolve_run_dir(method, order)
    density_file = run_dir / "density_final.vtu"
    parts = run_cantilever_stress.build(method, order)

    rho_final = read_vtu_cell_density(density_file)
    rho = parts["density"]
    if rho.shape[0] != rho_final.shape[0]:
        raise ValueError(f"{name}: 网格单元数 {rho.shape[0]} 与构型 {rho_final.shape[0]} 不符.")
    rho[:] = rho_final

    processor = StressPostProcessor(
        analyzer=parts["analyzer"], stress_limit=run_cantilever_stress.STRESS_LIMIT
    )
    results = processor.check_stress_constraints(rho)
    mesh = parts["mesh"]
    # 被动实体区掩码: 按 summary 记录的圆心与半径复原, 供插图从 solid_mask 中剔除
    # 该区 (rho 固定为 1 但不施加约束, 不属于判据集合). 无 pad 的运行得全 False.
    from .discretization_probe import pad_mask_from_summary  # 延迟导入: 该模块拉起求解栈
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    pad_mask = pad_mask_from_summary(mesh, summary, int(rho_final.shape[0]))
    return {
        "node": np.asarray(mesh.entity("node"), dtype=np.float64),
        "cell": np.asarray(mesh.entity("cell"), dtype=np.int32),
        "rho": rho_final,
        "vm": np.asarray(results.SM, dtype=np.float64),
        "sig1": np.asarray(results.sig_1_norm, dtype=np.float64),
        "sig2": np.asarray(results.sig_2_norm, dtype=np.float64),
        "solid_mask": np.asarray(results.solid_mask, dtype=bool),
        "pad_mask": pad_mask,
        "vol": np.float64(results.volume_fraction),
    }


def compare(fields: dict[str, Any], path: Path) -> list[str]:
    """与既有 npz 逐键比对, 返回不一致的键名列表."""
    if not path.is_file():
        return ["<文件不存在>"]
    with np.load(path) as reference:
        missing = set(fields) ^ set(reference.files)
        if missing:
            return [f"<键集合不一致: {sorted(missing)}>"]
        return [
            key
            for key, value in fields.items()
            if not np.allclose(reference[key], value, rtol=1e-10, atol=1e-12)
        ]



def export_fingerprint(name: str) -> str:
    """计算源结果与本地后处理实现的内容指纹."""
    method, order = RUNS[name]
    run_dir = resolve_run_dir(method, order, announce=False)
    paths = [run_dir / "density_final.vtu", run_dir / "summary.json"]
    paths.extend(sorted(EXPERIMENT_DIR.glob("*.py")))
    paths.extend(sorted((EXPERIMENT_DIR / "analysis").glob("*.py")))
    paths.extend(sorted((REPOSITORY_ROOT / "src" / "soptx").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_export(fields: dict[str, Any], path: Path, fingerprint: str) -> None:
    """写入数据及其内容校验记录; 中断或损坏的缓存不会被复用."""
    np.savez(path, **fields)
    record = {"source": fingerprint,
              "data": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")


def prepare_exports(names: list[str]) -> None:
    """只准备本图需要的运行; 全部源结果齐全后才开始重分析."""
    for name in names:
        method, order = RUNS[name]
        resolve_run_dir(method, order, announce=False)
    target = POSTPROCESS_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = target / f"fig_data_{name}.npz"
        fingerprint = export_fingerprint(name)
        valid = False
        try:
            record = json.loads(path.with_suffix(".json").read_text())
            valid = (record["source"] == fingerprint and
                     record["data"] == hashlib.sha256(path.read_bytes()).hexdigest())
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if valid:
            print(f"[cache] {name}: 复用有效绘图数据", flush=True)
            continue
        print(f"[prepare] {name}: 正在根据已有优化结果生成绘图数据"
              "（首次生成或输入已更新）", flush=True)
        fields = export_run(name)
        save_export(fields, path, fingerprint)


def run_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出/校验应力算例插图数据.")
    parser.add_argument("--run", choices=sorted(RUNS), action="append",
                        help="只处理指定运行, 可重复; 缺省处理全部.")
    parser.add_argument("--check", action="store_true",
                        help="只与现有 npz 比对, 不写盘.")
    arguments = parser.parse_args(argv)

    target_dir = POSTPROCESS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    mismatched = 0
    for name in arguments.run or sorted(RUNS):
        fields = export_run(name)
        path = target_dir / f"fig_data_{name}.npz"
        if arguments.check:
            differences = compare(fields, path)
            mismatched += bool(differences)
            verdict = "一致" if not differences else f"不一致: {', '.join(differences)}"
            print(f"[check] {path.name}: {verdict}")
        else:
            save_export(fields, path, export_fingerprint(name))
            print(f"[export] {path} (max SM = {fields['vm'].max():.4f}, "
                  f"V = {float(fields['vol']):.4f})")
    return 1 if mismatched else 0
