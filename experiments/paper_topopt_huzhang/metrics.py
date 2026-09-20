# -*- coding: utf-8 -*-
"""应力算例的后验校核与指标复算: 梯度校验 / 冻结重分析 / 插图场数据导出.

三段共用同一条链路 —— 都按 ``cases.toml`` 的权威参数经 ``pipeline`` 的悬臂梁
装配器重建分析管线, 故合并为一个模块 (原 ``check_gradients.py`` / ``frozen_metrics.py``
/ ``export_fig_data.py``, 2026-09-01 并入; 命名沿用
``experiments/elasticity_paradigm_comparison/metrics.py``)。

入口函数由 ``compare.py`` 的 ``COMMAND_MODULES`` 派发, 本模块不直接执行.
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
    build_stress_pipeline as build_pipeline,
)
from soptx.postprocess.stress_report import StressPostProcessor  # noqa: E402


# ============================================ 一、伴随灵敏度的有限差分校验
# 验证迁移后的灵敏度链路 (伴随载荷符号 / Voigt 重排 / 隐式项重写) 对
# HuZhang (HuZhangStressConstraint) 与 LFEM (LagrangeStressConstraint) 两条路径.

def make_params(nx: int, ny: int) -> dict:
    """沿用注册算例参数, 仅替换诊断网格."""
    return {**case_parameters(), "nx": nx, "ny": ny}


def _relative_error(analytic: float, finite_difference: float) -> float:
    if not np.isfinite(analytic) or not np.isfinite(finite_difference):
        return float("inf")
    denominator = max(abs(analytic), abs(finite_difference), 1.0e-14)
    return abs(analytic - finite_difference) / denominator


_GRADIENT_FD_STEPS = (1.0e-4, 3.0e-5, 1.0e-5)
_GRADIENT_RELATIVE_TOLERANCE = 1.0e-4


def _has_adjacent_acceptable_errors(
    relative_errors: list[float],
    tolerance: float = _GRADIENT_RELATIVE_TOLERANCE,
) -> bool:
    """至少两个相邻差分步长同时达标时才接受该方向."""
    return any(
        np.isfinite(left)
        and np.isfinite(right)
        and left <= tolerance
        and right <= tolerance
        for left, right in zip(relative_errors, relative_errors[1:])
    )


def _check_gradient_space(
    label: str,
    variable: Any,
    al_gradient: np.ndarray,
    lagrangian_gradient: np.ndarray,
    evaluate: Any,
    indices: list[int],
    active_threshold: float,
) -> bool:
    """在给定变量空间同时校验 AL 目标与原 Lagrangian 梯度."""
    print(f"\n--- {label} ---")
    step_labels = [f"FD({step:.0e})" for step in _GRADIENT_FD_STEPS]
    error_labels = [f"err({step:.0e})" for step in _GRADIENT_FD_STEPS]
    print(
        f"{'elem':>5} {'objective':>10} {'analytic':>14} "
        + " ".join(f"{item:>14}" for item in step_labels)
        + " "
        + " ".join(f"{item:>10}" for item in error_labels)
    )
    ok = True
    tested = {"al": 0, "lagrangian": 0}
    skipped = {"al": 0, "lagrangian": 0}
    for i in indices:
        values = {"al": [], "lagrangian": []}
        crosses_kink = False
        original = float(variable[i])
        try:
            for eps in _GRADIENT_FD_STEPS:
                variable[i] = original + eps
                al_plus, lagrangian_plus, g_plus = evaluate()
                variable[i] = original - eps
                al_minus, lagrangian_minus, g_minus = evaluate()
                values["al"].append((al_plus - al_minus) / (2.0 * eps))
                values["lagrangian"].append(
                    (lagrangian_plus - lagrangian_minus) / (2.0 * eps)
                )
                crosses_kink = crosses_kink or bool(np.any(
                    (g_plus > active_threshold) != (g_minus > active_threshold)
                ))
        finally:
            variable[i] = original

        rows = (
            ("al", float(al_gradient[i])),
            ("lagrangian", float(lagrangian_gradient[i])),
        )
        for objective, analytic in rows:
            if objective == "al" and crosses_kink:
                print(f"{i:>5} {objective:>10}  -- 中心差分跨越激活集 kink, 跳过 --")
                skipped[objective] += 1
                continue
            tested[objective] += 1
            fd_values = values[objective]
            relative_errors = [
                _relative_error(analytic, finite_difference)
                for finite_difference in fd_values
            ]
            flag = ""
            if not _has_adjacent_acceptable_errors(relative_errors):
                flag = "  <-- MISMATCH"
                ok = False
            print(
                f"{i:>5} {objective:>10} {analytic:>14.6e} "
                + " ".join(f"{value:>14.6e}" for value in fd_values)
                + " "
                + " ".join(f"{value:>10.2e}" for value in relative_errors)
                + flag
            )
    for objective in ("al", "lagrangian"):
        print(
            f"[{label}] {objective}: tested={tested[objective]}, "
            f"skipped={skipped[objective]}, acceptance requires two adjacent "
            f"steps with relerr <= {_GRADIENT_RELATIVE_TOLERANCE:.0e}"
        )
        if tested[objective] == 0:
            print(f"[{label}] {objective}: 无有效方向, 不能判为通过.")
            ok = False
    return ok


def check_method(
    method: str,
    order: int = 2,
    nx: int = 8,
    ny: int = 40,
    *,
    solver: str = "scipy",
    beta: float | None = None,
    mu: float = 50.0,
) -> bool:
    """校验指定离散、求解器及 AL 参数下的完整梯度链."""
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu 必须为有限正数.")
    if beta is not None and (not np.isfinite(beta) or beta <= 0.0):
        raise ValueError("beta 必须为有限正数.")
    print(
        f"\n===== method={method}, order={order}, mesh={nx}x{ny}, "
        f"solver={solver}, beta={beta if beta is not None else 'default'}, mu={mu:g} ====="
    )
    params = make_params(nx, ny)
    params["solve_method"] = solver
    config = build_config(params)
    pipe = build_pipeline(config, params, method, order)

    design = pipe.design_variable
    rho = pipe.density_distribution
    NC = rho.shape[0]
    if pipe.optimizer is None:
        raise RuntimeError("完整设计变量梯度校验要求 pipeline 提供 optimizer/filter.")
    density_filter = pipe.optimizer._filter
    if beta is not None:
        strategy = density_filter._strategy
        if not hasattr(strategy, "beta"):
            raise ValueError("beta 只适用于含投影的过滤器.")
        beta_max = float(getattr(strategy, "beta_max", beta))
        if beta > beta_max:
            raise ValueError(f"beta={beta:g} 超过投影上限 beta_max={beta_max:g}.")
        strategy.beta = float(beta)

    # 非均匀密度场 (远离 mask/投影死区: [0.3, 0.9])
    rng = np.random.default_rng(0)
    design[:] = rng.uniform(0.3, 0.9, size=NC)

    al = pipe.al_objective
    # 设 lambda=1.0, 激活阈值随指定 mu 取 -lambda/mu.
    al.lamb[:] = 1.0
    al.mu = float(mu)

    def evaluate_physical_density() -> tuple[float, float, np.ndarray]:
        """在当前物理密度上求解并返回两个状态一致的标量目标."""
        state = dict(pipe.analyzer.solve_state(rho_val=rho))
        al_value = float(al.fun(rho, state))
        g = np.asarray(al._cache_g)
        lamb = np.asarray(al.lamb)
        if g.shape != lamb.shape:
            raise RuntimeError(
                f"原 Lagrangian 要求 lambda 与 g 同形, 实际 {lamb.shape} != {g.shape}."
            )
        volume = float(pipe.volume_objective.fun(rho, state))
        lagrangian_value = volume + float(np.sum(lamb * g)) / g.size
        return al_value, lagrangian_value, g.copy()

    def evaluate_design_variable() -> tuple[float, float, np.ndarray]:
        """重新过滤/投影设计变量、求解状态并评价两个标量目标."""
        density_filter.filter_design_variable(
            design_variable=design,
            physical_density=rho,
        )
        return evaluate_physical_density()

    # 基准点完整经过 design -> filter/projection -> physical density -> state.
    J0, L0, _ = evaluate_design_variable()
    state = dict(pipe.analyzer.solve_state(rho_val=rho))
    al.fun(rho, state)
    g_base = np.asarray(al._cache_g).reshape(NC).copy()
    grad_al_rho = np.asarray(al.jac(rho, state)).reshape(NC)
    grad_lagrangian_rho = np.asarray(al.lagrangian_jac(rho, state)).reshape(NC)

    # 投影灵敏度依赖最近一次 filter_design_variable 缓存的 rho_tilde.
    density_filter.filter_design_variable(design_variable=design, physical_density=rho)
    grad_al_design = np.asarray(
        density_filter.filter_objective_sensitivities(
            design_variable=design,
            obj_grad_rho=grad_al_rho,
        )
    ).reshape(NC)
    grad_lagrangian_design = np.asarray(
        density_filter.filter_objective_sensitivities(
            design_variable=design,
            obj_grad_rho=grad_lagrangian_rho,
        )
    ).reshape(NC)

    thresh = -1.0 / float(mu)
    n_active = int(np.sum(g_base > thresh))
    print(f"J0 = {J0:.6e}, L0 = {L0:.6e},  active constraints: {n_active}/{NC},  "
          f"g range [{g_base.min():.3e}, {g_base.max():.3e}]")

    # 抽取待检单元: 随机 6 个 + 四类梯度经各自无穷范数归一化后的联合最大 2 个.
    idx = list(rng.choice(NC, size=min(6, NC), replace=False))
    normalized_gradients = []
    for gradient in (
        grad_al_rho,
        grad_lagrangian_rho,
        grad_al_design,
        grad_lagrangian_design,
    ):
        scale = max(float(np.max(np.abs(gradient))), 1.0e-14)
        normalized_gradients.append(np.abs(gradient) / scale)
    combined_magnitude = np.max(np.stack(normalized_gradients), axis=0)
    idx += list(np.argsort(-combined_magnitude)[:2])
    idx = sorted(set(int(i) for i in idx))

    physical_ok = _check_gradient_space(
        "物理密度梯度 d/d(rho_phys)",
        rho,
        grad_al_rho,
        grad_lagrangian_rho,
        evaluate_physical_density,
        idx,
        thresh,
    )
    # 恢复基准物理密度与 projection 缓存后再校验完整设计变量链.
    evaluate_design_variable()
    design_ok = _check_gradient_space(
        "完整设计变量梯度 d/d(design_variable)",
        design,
        grad_al_design,
        grad_lagrangian_design,
        evaluate_design_variable,
        idx,
        thresh,
    )
    evaluate_design_variable()
    ok = physical_ok and design_ok
    print(f"[{method}] gradient check {'PASSED' if ok else 'FAILED'}")
    return ok


def run_gradient_check() -> int:
    results = {}
    for method in ("huzhang", "lfem"):
        try:
            results[method] = check_method(method)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[method] = False
            print(f"[{method}] gradient check ERROR: {exc}")
    print("\n===== summary =====")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


# ============================================ 二、冻结设计的论文口径指标
# 对 outputs/cantilever-middle-2d-stress 下两个 density_final.vtu:
# 1. 用与优化相同的方法/阶次 (k=2) 冻结求解, 按博士论文口径报实体单元 (rho>0.5)
#    最大/平均归一化应力、实体单元数、全域最大及其位置;
# 2. 用 HZ k=3 / k=4 冻结求解做独立高阶重分析.

# 本段固定按注册默认网格 80x40 取产物 (frozen_eval 也写死 make_params(80, 40)).
OUT = OUTPUT_DIR / "cantilever-middle-2d-stress"
SIGMA_LIM = 180.0


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


def registered_load_pad_tag() -> str:
    """注册表当前的载荷引入垫片半径对应的 run 目录标签片段.

    垫片半径同时改变验收区域 (豁免) 与设计域 (实体保留), driver 把非零半径写进
    run 目录名, 冻结评估必须跟着注册默认值走, 否则会静默读到上一代产物.
    """
    radius = float(case_parameters().get("load_pad_radius", 0.0))
    return "" if radius <= 0.0 else f"__load_pad_radius-{radius}"


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
    仍会写进目录名; 这类运行与不带标签的注册运行同口径, 应当被冻结评估认走。
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
    "mu_update_rule": "mu_update_rule",
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


def frozen_eval(
    method: str,
    order: int,
    rho_np: np.ndarray,
    label: str,
    constraint_formulation: str = "apparent",
):
    params = make_params(80, 40)
    params["comparison_orders"] = [order]
    params["stress_constraint_formulation"] = constraint_formulation
    config = build_config(params)
    pipe = build_analysis_pipeline(config, params, method, order)
    rho = pipe.density_distribution
    assert rho.shape[0] == rho_np.shape[0], (rho.shape, rho_np.shape)
    rho[:] = rho_np

    state = dict(pipe.analyzer.solve_state(rho_val=rho))
    pipe.al_objective.fun(rho, state)
    SM = np.asarray(pipe.stress_constraint.compute_stress_measure(rho=rho, state=state))
    SM = SM.reshape(rho_np.shape[0], -1).max(axis=1)  # (NC,) NQ=1

    solid = rho_np > 0.5
    imax = int(np.argmax(SM))
    bc = np.asarray(pipe.mesh.entity_barycenter("cell"))
    print(f"[{label}] 全域 max = {SM.max():.4f} @elem {imax} "
          f"(rho={rho_np[imax]:.3f}, xy=({bc[imax][0]:.1f},{bc[imax][1]:.1f}))")
    print(f"[{label}] 实体单元数 = {int(solid.sum())}, "
          f"实体 max = {SM[solid].max():.4f}, 实体 mean = {SM[solid].mean():.4f}, "
          f"volfrac = {rho_np.mean():.4f}")

    # 高阶重分析时额外报更高积分阶的逐点最大 (仅 HZ)
    if method == "huzhang" and order >= 3:
        sq = pipe.analyzer.extract_stress_at_quadrature_points(
            stress_dof=state["stress"], integration_order=4)
        vm = np.asarray(pipe.analyzer.material.calculate_von_mises_stress(sq)) / SIGMA_LIM
        vm_e = vm.reshape(rho_np.shape[0], -1).max(axis=1)
        print(f"[{label}] (积分阶4) 全域 max = {vm_e.max():.4f}, "
              f"实体 max = {vm_e[solid].max():.4f}")
    return SM


def run_frozen_metrics() -> int:
    designs = {}
    for method in ("lfem", "huzhang"):
        run_dir, formulation = resolve_run_dir(method, 2)
        designs[method] = (
            read_vtu_cell_density(run_dir / "density_final.vtu"),
            formulation,
        )

    print("========== 一、论文口径 (各自方法 k=2 冻结求解) ==========")
    print("博士论文参考: LFEM max=1.0008, 实体 2266, 实体 mean=0.6067, V*=0.3499")
    print("             HZ   max=0.9978, 实体 2549, 实体 mean=0.5509, V*=0.3877")
    frozen_eval(
        "lfem", 2, designs["lfem"][0], "LFEM设计/LFEM-k2", designs["lfem"][1]
    )
    frozen_eval(
        "huzhang", 2, designs["huzhang"][0], "HZ设计/HZ-k2", designs["huzhang"][1]
    )

    print("\n========== 二、独立高阶重分析 (HZ k=3 / k=4) ==========")
    for design_name, (rho_np, _) in designs.items():
        for k in (3, 4):
            frozen_eval(
                "huzhang", k, rho_np, f"{design_name}设计/HZ-k{k}", "apparent"
            )
    return 0


# ============================================ 三、插图场数据导出 (npz)
# 论文 5.2.3 节的三张插图不直接读优化历程, 而是读带统一约束协议标签的
# outputs/cantilever-middle-2d-stress/postprocess/lfem_constraint-apparent/
# fig_data_<run>.npz; 本段是其唯一来源.
# npz 属 outputs/ 下的中间产物, 不入版本控制; 数字的溯源依据是各 run 目录下
# summary.json 自带的运行戳记 (provenance.run_stamp).

CASE_ID = "cantilever-middle-2d-stress"
POSTPROCESS_DIR = OUTPUT_DIR / CASE_ID / "postprocess" / "lfem_constraint-apparent"
# 插图涉及的五次运行: LFEM k=2 基线与 k=3 主对比 + Hu--Zhang k=2/3/4。
# 2026-09-17 曾去掉 huzhang-k4 (全域 C2 口径下 1000 步不收敛); 2026-09-18 C2 改为
# 实体子集验收 (acceptance_solid_threshold=0.5) 后该臂 258 步收敛, 加回本表;
# 同日为图 5.13 的主应力面板加入 lfem-k3。
RUNS: dict[str, tuple[str, int]] = {
    "lfem-k2": ("lfem", 2),
    "lfem-k3": ("lfem", 3),
    "huzhang-k2": ("huzhang", 2),
    "huzhang-k3": ("huzhang", 3),
    "huzhang-k4": ("huzhang", 4),
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
    # FEALPy 为 editable 依赖, 同时覆盖其实际加载目录中的 Python 实现.
    import fealpy
    paths.extend(sorted(Path(fealpy.__file__).resolve().parent.rglob("*.py")))
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


def run_audit_final_stress() -> int:
    """只重分析现有 k=2 最终物理密度, 不更新设计或覆盖原始结果."""
    parameters = case_parameters()
    all_feasible = True
    for method in ("lfem", "huzhang"):
        run_dir, formulation = resolve_run_dir(method, 2)
        run_parameters = {
            **parameters,
            "stress_constraint_formulation": formulation,
        }
        config = build_config(run_parameters)
        pipeline = build_analysis_pipeline(config, run_parameters, method, 2)
        values = read_vtu_cell_density(run_dir / "density_final.vtu")
        rho = pipeline.density_distribution
        if rho.shape != values.shape:
            raise ValueError(f"{method}: 最终密度与当前网格不一致")
        rho[:] = values
        state = pipeline.analyzer.solve_state(rho_val=rho)
        g = pipeline.stress_constraint.fun(rho, state)
        sm = pipeline.stress_constraint.compute_stress_measure(rho, state)
        max_g, max_sm = float(g.max()), float(sm.max())
        # 相对各自模型阈值归一化, 不对不同缩放的 g 使用同一容差.
        g_array, sm_array = np.asarray(g), np.asarray(sm)
        stiffness = np.asarray(state["stiffness_ratio"])
        vm_array = np.asarray(state["von_mises"])
        if method == "lfem":
            if not np.all(np.isfinite(stiffness) & (stiffness > 0)):
                raise ValueError("lfem: 相对刚度必须有限且为正")
        if "eta_threshold" in state:
            eta = np.asarray(state["eta_threshold"])
            if not np.all(np.isfinite(eta) & (eta > 0)):
                raise ValueError(f"{method}: 松弛阈值必须有限且为正")

        relative_violation = np.asarray(
            pipeline.stress_constraint.compute_relative_violation(rho, state))
        if not all(np.all(np.isfinite(array)) for array in
                   (g_array, sm_array, relative_violation)):
            raise ValueError(f"{method}: 应力审计指标包含非有限值")
        max_relative = float(relative_violation.max())
        # C2 口径与优化器 / driver 一致: 注册了 acceptance_solid_threshold 时在
        # rho >= 阈值的单元上判可行, 全域值照样打印作诊断。
        solid_threshold = getattr(config, "acceptance_solid_threshold", None)
        relative_violation_cell = relative_violation.reshape(len(values), -1).max(axis=1)
        if solid_threshold is not None and bool(np.any(values >= solid_threshold)):
            max_relative_solid = float(
                relative_violation_cell[values >= solid_threshold].max())
        else:
            max_relative_solid = max_relative
        node = np.asarray(pipeline.mesh.entity("node"))
        cell = np.asarray(pipeline.mesh.entity("cell"))
        def point_details(index):
            element = int(index[0])
            ratio = float(stiffness[element])
            native_vm = float(vm_array[index]) / config.stress_limit
            details = {
                "cell_index": element, "evaluation_index": list(map(int, index[1:])),
                "centroid": node[cell[element]].mean(axis=0).tolist(),
                "physical_density": float(values[element]), "stiffness_ratio": ratio,
                "native_stress_ratio": native_vm,
                "apparent_stress_ratio": float(sm_array[index]),
                "constraint_value": float(g_array[index]),
                "relative_violation": float(relative_violation[index]),
            }
            if "eta_threshold" in state:
                details["eta"] = float(np.asarray(state["eta_threshold"])[element])
            if method == "lfem":
                details["solid_stress_deviation"] = native_vm - 1.0
            return details
        diagnostics = {
            "maximum_constraint_point": point_details(np.unravel_index(g_array.argmax(), g_array.shape)),
            "maximum_apparent_stress_point": point_details(np.unravel_index(sm_array.argmax(), sm_array.shape)),
            "maximum_relative_violation_point": point_details(
                np.unravel_index(relative_violation.argmax(), relative_violation.shape)),
            "relative_violating_points": int(np.count_nonzero(
                relative_violation > config.stress_tolerance)),
        }
        feasible = max_relative_solid <= config.stress_tolerance
        all_feasible = all_feasible and feasible
        print(json.dumps({"method": method, "run_dir": str(run_dir),
                          "stress_constraint_formulation": formulation,
                          "volume_fraction": float(pipeline.volume_objective.fun(rho)),
                          "max_constraint": max_g, "max_apparent_stress_ratio": max_sm,
                          "max_relative_violation": max_relative,
                          "max_relative_violation_solid_region": max_relative_solid,
                          "acceptance_solid_threshold": solid_threshold,
                          "relative_stress_tolerance": config.stress_tolerance,
                          "relative_stress_feasible": feasible, **diagnostics}, ensure_ascii=False), flush=True)
    return 0 if all_feasible else 1
