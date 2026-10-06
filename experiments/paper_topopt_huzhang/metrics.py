# -*- coding: utf-8 -*-
"""应力算例的插图场数据导出: 运行目录解析与 npz 冻结重分析.

按 ``cases.toml`` 的权威参数经 ``pipeline`` 的悬臂梁装配器重建分析管线, 对最终
设计做冻结重分析, 导出论文 5.2.3 节插图所需的场量 (原 ``export_fig_data.py``,
2026-09-01 并入; 同期并入的梯度校验与冻结指标两段已随论文结果收口删除).

入口函数由 ``plot.py`` 的 ``COMMAND_MODULES`` 派发, 本模块不直接执行.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from soptx.postprocess.vtk_export import read_vtu_cell_data

from config import (
    CASES_FILE,
    OUTPUT_DIR,
    bootstrap_source_path,
    flatten_parameters,
    load_cases,
)

bootstrap_source_path()

from pipeline import (  # noqa: E402
    build_stress_analysis_pipeline as build_analysis_pipeline,
    build_stress_config as build_config,
)
from soptx.postprocess.stress_report import StressPostProcessor  # noqa: E402


# ============================================ 一、运行目录解析
# 按注册口径 (cases.toml 当前参数) 在 results/cantilever-middle-2d-stress 下定位
# apparent 正式运行目录, 拒绝目录标签或 summary 协议与注册值不符的旧产物.

OUT = OUTPUT_DIR / "cantilever-middle-2d-stress"


def read_vtu_cell_density(path: Path | str) -> np.ndarray:
    """解析 VTKFile appended-raw 格式的 CellData density 数组."""
    return read_vtu_cell_data(path, "density")


def _summary_constraint_formulation(
    run_dir: Path,
) -> tuple[str, str]:
    """优先读取当前计算链的实际模型, 兼容旧运行的 LFEM 协议字段."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    formulation = summary.get("stress_constraint_formulation")
    source = "summary.stress_constraint_formulation"
    if formulation is None:
        formulation = summary.get("lfem_stress_constraint_formulation")
        source = "summary.lfem_stress_constraint_formulation"
    if formulation is None:
        constraint_type = summary.get("stress_constraint_type")
        type_to_formulation = {
            "ApparentStressConstraint": "apparent",
            "LagrangeApparentStressConstraint": "apparent",
            "VanishingStressConstraint": "vanishing",
        }
        if constraint_type is not None:
            if constraint_type not in type_to_formulation:
                raise ValueError(
                    f"{run_dir}: 未知 stress_constraint_type={constraint_type!r}."
                )
            formulation = type_to_formulation[constraint_type]
            source = "summary.stress_constraint_type"
        else:
            raise ValueError(
                f"{run_dir}: apparent 正式运行缺少约束列式元数据；"
                "不能仅凭目录名重解释历史结果."
            )
    if formulation not in {"apparent", "vanishing"}:
        raise ValueError(f"{run_dir}: 未知应力约束列式 {formulation!r}.")
    return formulation, source


# driver._run_label 的标签分隔符与取值清洗规则; 此处只做逆向解析, 不再生成标签.
_TAG_SEPARATOR = "__"
_FIXED_TAGS = ("analyzer", "order", "lfem_constraint")
# 与 driver._TAG_ALIASES 同表: 目录名里的别名 -> cases.toml 字段名.
_TAG_ALIASES = {"solid_thr": "acceptance_solid_threshold"}


def _registered_tags(method: str, order: int) -> dict[str, str]:
    """注册默认参数下该运行组合应有的目录标签.

    Parameters
    ----------
    method : str
        分析器名, ``lfem`` 或 ``huzhang``.
    order : int
        比较阶次.

    Returns
    -------
    dict
        字段名到标签取值串的映射, 与 ``driver._run_label`` 的写法一致.
    """
    parameters = case_parameters()
    tags = {
        "analyzer": method,
        "order": str(order),
        "lfem_constraint": "apparent",
    }
    for pad_field in ("load_pad_radius", "support_pad_radius"):
        radius = float(parameters.get(pad_field, 0.0))
        if radius > 0.0:
            tags[pad_field] = str(radius)
    return tags


def _parse_run_label(label: str, known_fields: set[str]) -> dict[str, str] | None:
    """把 run 目录名拆回 ``driver._run_label`` 的标签字典.

    Parameters
    ----------
    label : str
        run 目录名.
    known_fields : set of str
        允许出现的字段名: 三个固定标签加上 ``cases.toml`` 的全部扁平参数名.

    Returns
    -------
    dict or None
        解析成功时返回字段名到取值串的映射; 出现无法归属到已知字段的片段
        (如手工加的 ``order-1_epsilon1e-4``) 时返回 None, 由调用方判为不可用.
    """
    tags: dict[str, str] = {}
    names = known_fields | set(_TAG_ALIASES)
    for token in label.split(_TAG_SEPARATOR):
        # 字段名自身含下划线, 取值也可能含 '-', 故按"最长已知字段名"匹配前缀.
        candidates = [name for name in names if token.startswith(f"{name}-")]
        if not candidates:
            return None
        name = max(candidates, key=len)
        tags[_TAG_ALIASES.get(name, name)] = token[len(name) + 1:]
    return tags


def _tag_matches_registered(field: str, value: str, parameters: dict[str, Any]) -> bool:
    """判断目录里的附加标签是否只是把注册默认值显式写了一遍.

    ``--override stress_tolerance=0.005`` 与注册默认 ``5.0e-3`` 数值相同, driver
    仍会写进目录名; 这类运行与不带标签的注册运行同口径, 应当被冻结评估认走.
    取值不同的覆盖跑 (``epsilon=1e-4``、``mu_max=1e5`` 等) 则必须排除.
    """
    if field not in parameters:
        return False
    registered = parameters[field]
    try:
        return float(value) == float(registered)
    except (TypeError, ValueError):
        return value == str(registered)


# summary.json 里如实记下的运行口径字段到 cases.toml 注册字段的对应.
# 目录名只反映"相对注册表改了什么", 改不了注册默认值本身的变迁: 2026-09-17 把
# stress_tolerance 由 3e-3 提到 5e-3 后, 旧的 3e-3 产物仍占着基准目录名, 只有
# 逐项核对 summary 才能把它们挡在冻结评估之外.
_SUMMARY_PROTOCOL_FIELDS = {
    "relative_stress_tolerance": "stress_tolerance",
    "load_pad_radius": "load_pad_radius",
    "support_pad_radius": "support_pad_radius",
    "move_limit_base": "move_limit",
    "asymptote_min_distance": "asymptote_min_distance",
    "change_measure": "change_measure",
    # 2026-09-18: 乘子安全阈与 C2 实体验收子集; 旧产物缺这两个键, 自动被冻结评估排除.
    "lambda_max": "lambda_max",
    "acceptance_solid_threshold": "acceptance_solid_threshold",
}


def _protocol_mismatches(run_dir: Path, parameters: dict[str, Any]) -> list[str]:
    """列出该运行与注册默认口径不符的字段.

    Returns
    -------
    list of str
        形如 ``relative_stress_tolerance: 0.003 != 0.005`` 的说明; 字段在
        summary 中缺失时同样计为不符 (老一代产物写不出后加的字段).
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for key, field in _SUMMARY_PROTOCOL_FIELDS.items():
        if field not in parameters:
            continue
        registered = parameters[field]
        if key not in summary:
            mismatches.append(f"{key}: 缺失 != {registered}")
            continue
        if not _tag_matches_registered(field, str(summary[key]), parameters):
            mismatches.append(f"{key}: {summary[key]} != {registered}")
    return mismatches


def resolve_run_dir(
    method: str,
    order: int,
    *,
    announce: bool = True,
) -> tuple[Path, str]:
    """只定位带 apparent 标签的新正式产物并核对其约束元数据.

    目录名由 ``driver._run_label`` 按字段名排序生成, 因此不能靠拼接字符串去猜:
    本函数先按注册默认参数取基准组合, 基准目录不存在时再接受"附加标签取值全部
    等于注册默认值"的同口径目录 (例如显式写出 ``stress_tolerance-0.005``),
    并要求唯一命中, 以免在多个探索性产物之间静默选一个.
    """
    parameters = case_parameters()
    known_fields = set(parameters) | set(_FIXED_TAGS)
    required_tags = _registered_tags(method, order)
    required = ("density_final.vtu", "summary.json")

    baseline = _TAG_SEPARATOR.join(
        f"{name}-{required_tags[name]}" for name in sorted(required_tags)
    )
    candidates: list[Path] = []
    rejected: list[str] = []
    for entry in sorted(OUT.iterdir()):
        if not entry.is_dir() or any(
            not (entry / name).is_file() for name in required
        ):
            continue
        tags = _parse_run_label(entry.name, known_fields)
        if tags is None:
            continue
        if any(tags.get(name) != value for name, value in required_tags.items()):
            continue
        extra = set(tags) - set(required_tags)
        if any(
            not _tag_matches_registered(field, tags[field], parameters)
            for field in extra
        ):
            continue
        mismatches = _protocol_mismatches(entry, parameters)
        if mismatches:
            rejected.append(f"{entry.name} ({'; '.join(mismatches)})")
            continue
        candidates.append(entry)

    if not candidates:
        detail = ""
        if rejected:
            detail = " 按目录标签匹配但口径不符, 已排除: " + ", ".join(rejected) + "."
        raise FileNotFoundError(
            f"缺少 apparent 正式运行 {OUT / baseline} (或其同口径变体): {required}." + detail
            + " 旧无标签目录不是新正式产物, 其中 LFEM 产物属于 vanishing, "
            "不作为新流程 fallback."
        )
    if len(candidates) > 1:
        names = ", ".join(entry.name for entry in candidates)
        raise ValueError(
            f"{method}-k{order}: 同口径产物不唯一 ({names}); "
            "请删除或重命名多余目录, 冻结评估不替你选."
        )
    run_dir = candidates[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("converged", False)):
        raise ValueError(
            f"{run_dir}: 该运行未收敛 ({summary.get('termination_reason')}); "
            "未收敛构型不作为插图与冻结评估的来源."
        )
    formulation, source = _summary_constraint_formulation(run_dir)
    if formulation != "apparent":
        raise ValueError(
            f"{run_dir}: 目录标签为 apparent, 元数据却记录 {formulation}."
        )

    if announce:
        print(json.dumps({
            "run_source": f"{method}-k{order}",
            "run_dir": str(run_dir),
            "stress_constraint_formulation": formulation,
            "formulation_source": source,
        }, ensure_ascii=False), flush=True)
    return run_dir, formulation


# ============================================ 二、插图场数据导出 (npz)
# 论文 5.2.3 节的图 5.9 (b)(d) 不直接读优化历程, 而是读带统一约束协议标签的
# results/cantilever-middle-2d-stress/postprocess/lfem_constraint-apparent/
# fig_data_<run>.npz; 本段是其唯一来源.
# npz 写在 results/ 下随结果入库; 数字的溯源依据是各 run 目录下
# summary.json 自带的运行戳记 (provenance.run_stamp).

CASE_ID = "cantilever-middle-2d-stress"
POSTPROCESS_DIR = OUTPUT_DIR / CASE_ID / "postprocess" / "lfem_constraint-apparent"
# 插图只用 k=3 主对比的两次运行 (plots/stress_cubic_convergence 的主应力面板);
# 其余阶次的场图改由 discretization_probe 的 fields.npz 供数.
RUNS: dict[str, tuple[str, int]] = {
    "lfem-k3": ("lfem", 3),
    "huzhang-k3": ("huzhang", 3),
}


def case_parameters(case_id: str = CASE_ID) -> dict[str, Any]:
    """从 cases.toml 取该算例的扁平参数, 保证与优化时同口径."""
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return flatten_parameters(case)
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def export_run(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """对单次运行做冻结重分析, 返回 npz 待写入的场量字典."""
    method, order = RUNS[name]
    run_dir, formulation = resolve_run_dir(method, order)
    density_file = run_dir / "density_final.vtu"

    parameters = {
        **parameters,
        "comparison_orders": [order],
        "stress_constraint_formulation": formulation,
    }
    pipeline = build_analysis_pipeline(build_config(parameters), parameters, method, order)

    rho_final = read_vtu_cell_density(density_file)
    rho = pipeline.density_distribution
    if rho.shape[0] != rho_final.shape[0]:
        raise ValueError(f"{name}: 网格单元数 {rho.shape[0]} 与构型 {rho_final.shape[0]} 不符.")
    rho[:] = rho_final

    processor = StressPostProcessor(
        analyzer=pipeline.analyzer, stress_limit=float(parameters["stress_limit"])
    )
    results = processor.check_stress_constraints(rho)
    mesh = pipeline.mesh
    # 被动实体区掩码: 按 summary 记录的圆心与半径复原, 供插图从 solid_mask 中剔除
    # 该区 (rho 固定为 1 但不施加约束, 不属于判据集合). 无 pad 的运行得全 False.
    from discretization_probe import pad_mask_from_summary  # 延迟导入: 该模块拉起求解栈
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
    run_dir, _ = resolve_run_dir(method, order, announce=False)
    root = Path(__file__).resolve().parents[2]
    paths = [run_dir / "density_final.vtu", run_dir / "summary.json", CASES_FILE]
    paths.extend(sorted(Path(__file__).parent.glob("*.py")))
    paths.extend(sorted((root / "src" / "soptx").rglob("*.py")))
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
    parameters = case_parameters()
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
        fields = export_run(name, parameters)
        save_export(fields, path, fingerprint)


def run_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出/校验应力算例插图数据.")
    parser.add_argument("--run", choices=sorted(RUNS), action="append",
                        help="只处理指定运行, 可重复; 缺省处理全部.")
    parser.add_argument("--check", action="store_true",
                        help="只与现有 npz 比对, 不写盘.")
    arguments = parser.parse_args(argv)

    parameters = case_parameters()
    target_dir = POSTPROCESS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    mismatched = 0
    for name in arguments.run or sorted(RUNS):
        fields = export_run(name, parameters)
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
