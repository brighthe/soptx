"""Hu--Zhang 拓扑优化投稿论文实验主 Runner.

本脚本在同一物理问题、同一材料插值 (MSIMP) 与统一网格下比较两条求解路径:
1. ``LagrangeFEMAnalyzer``: 经典 Lagrange 位移有限元 (LFEM) 求解链;
2. ``HuZhangMFEMAnalyzer``: 对称弱形式胡--张混合有限元 (Hu--Zhang) 求解链.

受控比较协议:
- 给定阶次 ``k``, LFEM 采用位移阶 ``p=k``; Hu--Zhang 采用应力阶 ``k`` (对应位移阶 ``k-1``);
- 统一高斯积分阶 ``q=2k+2``;
- 载荷统一通过 ``project_patch_traction_to_p1_trace`` 投影至底边连续 P1 迹空间, 消除强施加与弱积分的几何不对齐误差;
- 正文比较阶次为 ``k=2, 3, 4``; Hu--Zhang ``k=1`` 因 P0 常数位移空间缺失刚体旋转模态、在拓扑
  演化中引发人工刚度硬化而不进任何优化算例; 轴承算例经 ``supplementary_orders`` 放开的
  ``1`` 只服务 LFEM p=1 的体积闭锁对照, 见 docs/fem/huzhang-mixed-fem-implementation.md.

运行: 执行完整拓扑优化迭代, 产物写入
  ``results/<case-id>/analyzer-<链>__order-<k>[__<字段>-<取值>...]/``,
  包含最终密度场 ``density_final.vtu``、收敛历史 ``history.json`` 与运行摘要 ``summary.json``;
  应力约束算例的每帧 VTU 另带单元场 ``von_mises_normalized`` (归一化表观应力比),
  取自优化器在同一密度上求解得到的场 (2026-09-11 加).

本模块同时承载能量恒等式诊断 (供 ``compliance_reanalysis``)、真相对残差与结果落盘
(原 ``diagnostics.py``).

使用方法:
    # 1. 运行胡张元 (k=2) 拓扑优化 (冒烟测试 3 步)
    python experiments/paper_topopt_huzhang/run.py --case compliance-fixed-fixed-half --analyzer huzhang --order 2 --max-iterations 3 --solver scipy

    # 2. 运行全量论文矩阵对比 (LFEM 与 Hu--Zhang, k=2,3,4)
    python experiments/paper_topopt_huzhang/run.py --case compliance-fixed-fixed-half --analyzer all --solver scipy
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import re
import sys
from typing import Any

# sys.path 由唯一入口 run.py 通过 config.bootstrap_source_path() 统一注入.

import provenance
from config import (
    CASES_FILE,
    OUTPUT_DIR,
    ConfigurationError,
    UnsupportedModelError,
    configuration_summary,
    flatten_parameters,
    load_cases,
    resolve_runs,
    select_cases,
)
from pipeline import (
    MESH_EVEN_AXES,
    MESH_TYPES,
    assembler_for,
    resolve_stress_constraint_formulation,
)

import numpy as np

from soptx.backend import backend_manager as bm


# ------------------------------------------------------------------ 能量诊断
# 以下四个函数原为 diagnostics.py; 除本模块外无人引用, 2026-09-01 并入.
def relative_residual(pipeline: Any, state: dict[str, Any]) -> float:
    """计算状态解的真相对平衡残差.

    参数:
        pipeline: 优化/分析管线对象.
        state: 求解器返回的状态字典.

    返回:
        相对平衡残差标量.
    """
    if pipeline.method == "huzhang":
        return float(pipeline.analyzer.relative_state_residual())

    analyzer = pipeline.analyzer
    displacement = state["displacement"]
    residual = analyzer.stiffness_matrix.matmul(displacement[:]) - analyzer.force_vector
    is_boundary = analyzer.tensor_space.is_boundary_dof(
        threshold=pipeline.problem.is_dirichlet_boundary(), method="interp"
    )
    numerator = float(bm.linalg.norm(residual[~is_boundary]))
    denominator = max(float(bm.linalg.norm(analyzer.force_vector[~is_boundary])), 1.0e-30)
    return numerator / denominator


def energy_identity_diagnostics(
    pipeline: Any,
    state: dict[str, Any],
) -> dict[str, float | str]:
    """计算各离散方法在给定密度场下可直接验证的能量恒等式.

    参数:
        pipeline: 组装好的分析管线对象.
        state: 前向求解状态字典.

    返回:
        包含能量分量与相对对偶一致性缺陷的字典.
    """
    if pipeline.method == "lfem":
        displacement = state["displacement"][:]
        force = pipeline.analyzer.force_vector
        stiffness = pipeline.analyzer.stiffness_matrix
        external_work = float(bm.einsum("i, i ->", displacement, force[:]))
        strain_energy = float(
            bm.einsum("i, i ->", displacement, stiffness.matmul(displacement))
        )
        return {
            "identity": "fTu_equals_uKu",
            "external_work": external_work,
            "internal_energy": strain_energy,
            "relative_defect": abs(external_work - strain_energy)
            / max(abs(external_work), 1.0e-30),
        }

    stress = state["stress"][:]
    displacement = state["displacement"][:]
    stress_matrix = pipeline.analyzer.get_stress_matrix(
        rho_val=pipeline.density_distribution
    )
    mix_matrix = pipeline.analyzer.mix_matrix
    complementary_energy = float(
        bm.einsum("i, i ->", stress, stress_matrix.matmul(stress))
    )
    coupling_work = float(
        bm.einsum("i, i ->", stress, mix_matrix.matmul(displacement))
    )
    traction_dual_work = complementary_energy + coupling_work
    return {
        "identity": "sigmaAsigma_plus_sigmaBu_equals_traction_dual_work",
        "complementary_energy": complementary_energy,
        "coupling_work": coupling_work,
        "traction_dual_work": traction_dual_work,
        "relative_coupling_ratio": abs(coupling_work)
        / max(abs(complementary_energy), 1.0e-30),
    }


# 优化器 history.field_histories 键 -> VTU 单元场名. 场按积分点存 (NC, NQ...),
# 写盘前在积分点上取最大, 与 summary 的 max_apparent_stress_ratio 同口径.
# 柔顺度算例没有这个键, 自动跳过. 判据量 r_e 不落盘: 标量进 summary 即可.
_VTU_CELL_FIELDS: dict[str, str] = {
    "von_mises_stress": "von_mises_normalized",
}

# 每次运行落盘的文件名: write_optimization_result 与 _write_stress_optimizer_state
# 共用同一份清单, 改名只改这里.
# VTU 名不含扩展名 (write_vtu 自行追加 .vtu); 演化序列按帧编号写入 vtu/ 子目录.
OUTPUT_FILES: dict[str, str] = {
    "summary": "summary.json",
    "history": "history.json",
    "density_final": "density_final",
    "vtu_dir": "vtu",
    "vtu_frame": "density_iter_{index:03d}",
}
# 仅应力约束算例 (AL 优化器) 额外写出的终态诊断文件.
STRESS_OUTPUT_FILES: dict[str, str] = {
    "optimizer_state": "final_optimizer_state.npz",
    "outer_history": "outer_history.json",
}


def _per_cell_max(value: Any) -> np.ndarray:
    """把 (NC, NQ...) 的积分点场压成 (NC,) 的单元最大值; 已是 (NC,) 的原样返回."""
    from soptx.backend import backend_manager as bm

    array = np.asarray(bm.to_numpy(value), dtype=np.float64)
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1).max(axis=1)


def _history_cell_fields(history: Any, index: int) -> dict[str, np.ndarray]:
    """取历史第 index 帧的应力类单元场 (VTU 字段名 -> (NC,) 数组); 无记录时为空."""
    field_histories = getattr(history, "field_histories", None) or {}
    cell_fields = {}
    for key, vtu_name in _VTU_CELL_FIELDS.items():
        values = field_histories.get(key)
        if values and len(values) > index:
            cell_fields[vtu_name] = _per_cell_max(values[index])
    return cell_fields


def _write_stress_optimizer_state(output: Path, pipeline: Any, density: Any) -> dict | None:
    """保存 AL 终态与内层诊断, 供冻结复算使用而非精确断点续算.

    Parameters
    ----------
    output : Path
        当前运行的产物目录.
    pipeline : Any
        含 AL 目标和优化器的应力优化管线.
    density : Any
        与最终设计对应的物理密度.

    Returns
    -------
    dict or None
        状态文件及内层控制元数据; 非 AL 管线返回 None.
    """
    from soptx.backend import backend_manager as bm

    optimizer = getattr(pipeline, "optimizer", None)
    objective = getattr(pipeline, "al_objective", None)
    design = getattr(optimizer, "final_design_variable", None)
    if objective is None or design is None:
        return None
    arrays = {
        "design": np.asarray(bm.to_numpy(design[:]), dtype=np.float64),
        "density": np.asarray(bm.to_numpy(density[:]), dtype=np.float64),
        "lamb": np.asarray(bm.to_numpy(objective.lamb), dtype=np.float64),
        "mu": np.asarray(float(objective.mu)),
    }
    if arrays["design"].shape != arrays["density"].shape:
        raise ValueError("终态设计与物理密度形状不一致.")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("AL 终态包含非有限值, 不能作为诊断状态保存.")
    options = optimizer.options
    meta = {
        "file": STRESS_OUTPUT_FILES["optimizer_state"],
        "purpose": "frozen-analysis-and-inner-diagnostics",
        "exact_restart_supported": False,
        "beta": getattr(optimizer._filter, "beta", None),
        "inner_stop_rule": getattr(options, "inner_stop_rule", "legacy"),
        "inner_relative_tolerance": getattr(options, "inner_relative_tolerance", None),
        "inner_absolute_tolerance": getattr(options, "inner_absolute_tolerance", None),
        "max_inner_iterations": options.mma_iters_per_al,
        "last_inner_diagnostics": getattr(optimizer, "last_inner_diagnostics", None),
    }
    np.savez_compressed(output / meta["file"], **arrays)
    (output / STRESS_OUTPUT_FILES["outer_history"]).write_text(
        json.dumps(getattr(optimizer, "outer_history", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def write_optimization_result(
    output: Path,
    pipeline: Any,
    density: Any,
    history: Any,
    summary: dict[str, Any],
    final_cell_fields: dict[str, Any] | None = None,
) -> None:
    """保存最终密度场 VTU、标量收敛历史与运行摘要 JSON.

    参数:
        output: 输出目标文件夹路径.
        pipeline: 优化管线对象.
        density: 最终单元密度场.
        history: 优化迭代历史记录对象.
        summary: 汇总指标字典.
        final_cell_fields: 写进 density_final.vtu 的附加单元场 (VTU 字段名 -> 场);
            应力约束算例传终态重新求解的应力比, 缺省则回退到历史末帧.
    """
    output.mkdir(parents=True, exist_ok=True)
    from soptx.postprocess.vtk_export import write_vtu
    from soptx.backend import backend_manager as bm

    n_frames = len(getattr(history, "physical_densities", None) or [])

    # 1. 最终密度场便捷文件 (应力约束算例附带终态应力场)
    density_np = np.asarray(bm.to_numpy(density[:]), dtype=np.float64).flatten()
    if final_cell_fields is not None:
        final_fields = {name: _per_cell_max(value) for name, value in final_cell_fields.items()}
    else:
        final_fields = _history_cell_fields(history, n_frames - 1) if n_frames else {}
    write_vtu(
        mesh=pipeline.mesh,
        filepath=str(output / OUTPUT_FILES["density_final"]),
        cell_data={"density": density_np, **final_fields},
    )

    # 2. 完整迭代演化序列 (供 ParaView 作为动画时间序列直接加载); 每帧的应力场
    #    来自优化器在该帧密度上的状态求解 (_accept_mma_step 对接受态重解), 与密度同步.
    #    第 0 帧是优化前的初始构型, 只带密度: 初始密度上的应力场不参与任何优化量与
    #    验收指标, 为它单独做一次状态求解不划算, 因此该帧没有应力字段.
    initial_density = getattr(history, "initial_physical_density", None)
    if n_frames or initial_density is not None:
        vtu_dir = output / OUTPUT_FILES["vtu_dir"]
        vtu_dir.mkdir(parents=True, exist_ok=True)
        if initial_density is not None:
            rho_0_np = np.asarray(bm.to_numpy(initial_density[:]), dtype=np.float64).flatten()
            write_vtu(
                mesh=pipeline.mesh,
                filepath=str(vtu_dir / OUTPUT_FILES["vtu_frame"].format(index=0)),
                cell_data={"density": rho_0_np},
            )
        for iter_idx, rho_i in enumerate(history.physical_densities, start=1):
            rho_i_np = np.asarray(bm.to_numpy(rho_i), dtype=np.float64).flatten()
            write_vtu(
                mesh=pipeline.mesh,
                filepath=str(vtu_dir / OUTPUT_FILES["vtu_frame"].format(index=iter_idx)),
                cell_data={"density": rho_i_np, **_history_cell_fields(history, iter_idx - 1)},
            )
    optimizer_state = _write_stress_optimizer_state(output, pipeline, density)
    if optimizer_state is not None:
        summary["optimizer_state"] = optimizer_state
    history_payload = {
        "iter_indices": history.iter_indices,
        "changes": history.changes,
        "iteration_times": history.iteration_times,
        "scalar_histories": history.scalar_histories,
    }
    (output / OUTPUT_FILES["history"]).write_text(
        json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / OUTPUT_FILES["summary"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------ 驱动


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """解析算例选择、离散方法和临时覆盖参数.

    返回:
        命令行参数命名空间对象.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CASES_FILE, help="TOML 配置文件路径.")
    parser.add_argument("--list", action="store_true", help="列出已注册的论文算例.")
    parser.add_argument(
        "--case", action="append", help="算例 id, 可重复指定; 使用 all 选择所有 ready 算例."
    )
    parser.add_argument(
        # 主名与 --list 的 analyzer 列对齐 (求解链 + 空间次数才唯一确定一个分析器);
        # --method 留作旧命令的别名, dest 仍是 method, 免得 summary.json 的 method
        # 字段与 config.resolve_runs 的口径跟着一起改.
        "--analyzer",
        "--method",
        dest="method",
        choices=("lfem", "huzhang", "all"),
        help="分析链: LFEM 基线 / Hu--Zhang 混合元 / 全部; 缺省只跑 huzhang.",
    )
    parser.add_argument(
        "--order",
        type=int,
        action="append",
        help="受控比较空间有限元次数 k, 可重复传入; 缺省只跑 comparison_orders 的最小值.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="按 cases.toml 的 methods x comparison_orders 全跑, 即论文的完整对比组.",
    )
    parser.add_argument("--solver", choices=("scipy", "mumps"), help="线性求解器后端.")
    parser.add_argument(
        "--optimizer", choices=("oc", "mma"), help="临时覆盖优化器 (OC 或 MMA), 默认取 cases.toml."
    )
    parser.add_argument(
        "--interpolation",
        choices=("auto", "E", "E+nu"),
        help=(
            "临时覆盖材料插值对象: E 只插值 Young 模量, E+nu 同时插值 Poisson 比 "
            "(仅近不可压缩材料可用), auto 按材料自动决定; 默认取 cases.toml."
        ),
    )
    parser.add_argument(
        "--filter-type", choices=("density", "projection"),
        help="临时覆盖过滤器类型 (density/projection), 默认取 cases.toml.",
    )
    parser.add_argument(
        "--mesh-type",
        dest="mesh_type",
        choices=MESH_TYPES,
        help=(
            "临时覆盖三角剖分方式: triangle-checkerboard 棋盘格交替对角 (nx, ny 须为偶数); "
            "triangle-single-diagonal-symmetric 左半 / 右半 \\ 镜像对称单向对角 (nx 须为偶数, "
            "论文闭锁对照); 默认取 cases.toml."
        ),
    )
    parser.add_argument("--nx", type=int, help="临时覆盖横向网格剖分数.")
    parser.add_argument("--ny", type=int, help="临时覆盖纵向网格剖分数.")
    parser.add_argument("--max-iterations", type=int, help="临时覆盖最大优化迭代次数.")
    parser.add_argument(
        # 具名开关只覆盖最常用的那几个字段; 其余字段走这个通用通道, 免得每加一个
        # 参数就往 parser 里塞一个开关. 字段名以 build_config 造出的配置对象为准,
        # 也就是 cases.toml 的 discretization/optimization 键名; 运行组合维度
        # analyzer / order 也一并接住 (见 _apply_run_selection), 口径与 EA/FA
        # 两个实验的 --override 一致.
        "--override",
        nargs="+",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "按 KEY=VALUE 临时覆盖配置字段或运行组合 (analyzer/order), 可重复给出; "
            "如 --override optimizer=mma order=2,3."
        ),
    )
    parser.add_argument("--check-only", action="store_true", help="仅校验配置和运行组合.")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR, help="运行产物输出根目录."
    )
    arguments = parser.parse_args(argv)
    arguments.overrides = _collect_overrides(arguments.override, parser)
    _apply_run_selection(arguments, arguments.overrides, parser)
    return arguments


def _collect_overrides(
    groups: list[list[str]], parser: argparse.ArgumentParser
) -> dict[str, str]:
    """把 --override 的分组文本摊平成 {字段: 文本}; 语法错误当场由 parser 报错.

    同一字段重复给出直接判错: 静默取最后一次会让命令行与实际跑的参数对不上.
    """
    overrides: dict[str, str] = {}
    for item in (text for group in groups for text in group):
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            parser.error(f"--override 需要 KEY=VALUE 形式: {item}")
        name = name.strip()
        if name in overrides:
            parser.error(f"--override 重复指定了字段 {name}.")
        overrides[name] = value.strip()
    return overrides


# 运行组合维度: 决定跑哪几组, 而不是某一组怎么算, 因此进不了配置对象.
# EA/FA 侧的 --override 能改除登记项以外的全部字段, 这里保持同一套口径, 把这两个
# 键接住后转写成对应具名开关的取值, 同一串命令行在三个实验里写法一致.
_RUN_SELECTION_FIELDS = ("analyzer", "order")
_ANALYZER_CHOICES = ("lfem", "huzhang", "all")


def _apply_run_selection(
    arguments: argparse.Namespace,
    overrides: dict[str, str],
    parser: argparse.ArgumentParser,
) -> None:
    """把 --override 里的 analyzer / order 转写成具名开关取值, 并从覆盖字典摘除.

    多值用逗号分隔 (``order=2,3`` 等价于 ``--order 2 --order 3``), 与 EA/FA 侧
    --override 的列表写法一致; ``analyzer`` 同时给出两条链等价于 ``all``.
    与具名开关撞车时报错而不是定先后顺序, 口径同 ``_override_changes``.
    """
    for name in _RUN_SELECTION_FIELDS:
        if name not in overrides:
            continue
        values = [
            item.strip() for item in overrides.pop(name).split(",") if item.strip()
        ]
        if not values:
            parser.error(f"--override {name}= 需要非空取值.")
        if name == "analyzer":
            if arguments.method is not None:
                parser.error("--analyzer 与 --override analyzer= 重复指定了同一维度.")
            invalid = sorted(set(values) - set(_ANALYZER_CHOICES))
            if invalid:
                parser.error(
                    f"--override analyzer= 取值非法: {', '.join(invalid)}; "
                    f"可选 {', '.join(_ANALYZER_CHOICES)}."
                )
            arguments.method = "all" if len(set(values)) > 1 else values[0]
            continue
        if arguments.order:
            parser.error("--order 与 --override order= 重复指定了同一维度.")
        try:
            arguments.order = [int(item) for item in values]
        except ValueError:
            parser.error(
                f"--override order= 需要整数 (可逗号分隔): {','.join(values)}."
            )


# 产物目录第二层的标签分隔符; 与 experiments/topopt_simp_fa|ea 同一套写法.
_TAG_SEPARATOR = "__"
# 目录标签里的字段别名: 字段名过长会把 \\wsl.localhost 下的 Windows 路径推过
# MAX_PATH (260), ParaView 等未声明 long-path aware 的程序就列不出目录. 别名只用于
# 目录名, summary.json 与 cases.toml 仍用完整字段名; metrics._parse_run_label 负责映回.
_TAG_ALIASES = {"acceptance_solid_threshold": "solid_thr"}

_BOOLEAN_TEXTS = {"true": True, "false": False}
# 允许用 none/null 覆盖成 None 的 Optional 字段 (注册值非 None 时现值类型是 float,
# 光看现值猜不出它可空): lambda_max=none 复原无阈更新, acceptance_solid_threshold=none
# 复原 C2 全域口径.
_OPTIONAL_OVERRIDE_FIELDS = frozenset({"lambda_max", "acceptance_solid_threshold"})
_NONE_TEXTS = {"none", "null"}


def _coerce(name: str, text: str, current: Any) -> Any:
    """按配置对象里现有取值的类型转换覆盖文本; 类型不认识就原样当字符串.

    不读 dataclass 的类型注解: 模块开头有 from __future__ import annotations,
    注解此时是字符串, 拿现值的类型更可靠.
    """
    if isinstance(current, bool):
        if text.lower() not in _BOOLEAN_TEXTS:
            raise ConfigurationError(f"覆盖值非法: {name}={text} (需要 true/false).")
        return _BOOLEAN_TEXTS[text.lower()]
    if name in _OPTIONAL_OVERRIDE_FIELDS and text.strip().lower() in _NONE_TEXTS:
        return None
    for caster in (int, float):
        if isinstance(current, caster):
            try:
                return caster(text)
            except ValueError as error:
                raise ConfigurationError(
                    f"覆盖值非法: {name}={text} (需要 {caster.__name__})."
                ) from error
    if current is None:
        # Optional 字段 (lambda_max、acceptance_solid_threshold) 的现值可能是 None,
        # 按文本猜类型: 整数、浮点、none/null, 都不是才原样当字符串.
        for caster in (int, float):
            try:
                return caster(text)
            except ValueError:
                continue
        if text.strip().lower() in _NONE_TEXTS:
            return None
    return text


def _override_changes(
    config: Any, overrides: dict[str, str], named: dict[str, Any]
) -> dict[str, Any]:
    """校验并转换 --override, 返回可直接喂给 dataclasses.replace 的改动字典."""
    if not overrides:
        return {}
    field_names = {field.name for field in fields(config)}
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        # analyzer / order 已在 _apply_run_selection 摘除, 但要进「可覆盖」清单,
        # 否则报错信息会让人以为这两个维度不能用 --override 给.
        available = sorted(field_names | set(_RUN_SELECTION_FIELDS))
        raise ConfigurationError(
            f"未知的覆盖字段: {', '.join(unknown)}; 可覆盖: {', '.join(available)}."
        )
    # 与具名开关撞车时报错而不是定一个先后顺序: 两个写法给同一个字段不同取值,
    # 无论哪边赢都有一半命令行是假的.
    conflicts = sorted(set(overrides) & set(named))
    if conflicts:
        raise ConfigurationError(
            f"--override 与具名开关重复指定了同一字段: {', '.join(conflicts)}."
        )
    return {
        name: _coerce(name, text, getattr(config, name))
        for name, text in overrides.items()
    }


def _run_label(method: str, order: int, config: Any, changes: dict[str, Any]) -> str:
    """产物目录第二层: 这一次运行相对注册表基准的参数标签.

    analyzer 与 order 是运行组合维度 —— 同一个 case 目录下并排躺着好几组, 故恒进
    标签; 其余字段只在被覆盖时进标签, 探索性运行因此不会盖掉注册运行的产物. 标签
    按字段名排序、用 ``__`` 连接, 与 experiments/topopt_simp_fa|ea 的第二层同一套
    写法, 三个实验的 outputs/ 用同一种读法. 过长的字段名按 ``_TAG_ALIASES`` 缩写.
    """
    tags: dict[str, Any] = {"analyzer": method, "order": order}
    tags.update({name: getattr(config, name) for name in changes})
    formulation = getattr(config, "stress_constraint_formulation", None)
    if formulation is not None:
        # 新旧 LFEM 约束协议必须恒进目录名, 防止注册默认值切换后覆盖历史产物.
        tags.pop("stress_constraint_formulation", None)
        tags["lfem_constraint"] = formulation
    for pad_field in ("load_pad_radius", "support_pad_radius"):
        # 垫片半径改变的是验收区域与可设计区域本身, 同上必须进目录名: 取 0 时
        # 标签复原成旧名, 与 2026-09-16 之前那批无垫片产物就地可比.
        # 两侧半径各占一个标签, 因此"只处置载荷侧"与"两侧都处置"不会同名覆盖.
        pad_radius = getattr(config, pad_field, None)
        if pad_radius is None:
            continue
        tags.pop(pad_field, None)
        if float(pad_radius) > 0.0:
            tags[pad_field] = pad_radius
    # C2 的验收子集改变的是停止准则本身 (2026-09-18 起注册 0.5), 同上恒进目录名;
    # 取 None (全域口径) 时标签复原成旧名, 与 09-17 之前的全域口径产物同名可比.
    # lambda_max 与 mu_max 同类, 只在被覆盖时进标签.
    solid_threshold = getattr(config, "acceptance_solid_threshold", None)
    tags.pop("acceptance_solid_threshold", None)
    if solid_threshold is not None:
        tags["acceptance_solid_threshold"] = solid_threshold
    # 别名替换后再排序, 目录名按别名排, 与 metrics 的反解一致.
    tags = {_TAG_ALIASES.get(name, name): value for name, value in tags.items()}
    return _TAG_SEPARATOR.join(
        re.sub(r"[^0-9A-Za-z_.\-]+", "-", f"{name}-{tags[name]}")
        for name in sorted(tags)
    )


# 命令行临时覆盖: 配置字段名 -> 命名空间属性名 (迭代上限字段按算例另行确定)
_OVERRIDE_FIELDS = (
    ("mesh_type", "mesh_type"),
    ("nx", "nx"),
    ("ny", "ny"),
    ("solve_method", "solver"),
    ("optimizer", "optimizer"),
    ("filter_type", "filter_type"),
    ("interpolation_variables", "interpolation"),
)


def build_model_pipeline(
    case: dict[str, Any],
    method: str,
    order: int,
    overrides: argparse.Namespace,
    *,
    analysis_only: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """按模型名从 pipeline.ASSEMBLERS 取装配器, 组装分析链或优化链, 并施加命令行覆盖.

    第三个返回值是本次实际生效的覆盖改动, 供 run 目录名判断要不要另起标签.
    """
    model_name = case["model"]["name"]
    try:
        assembler = assembler_for(model_name)
    except KeyError as error:
        raise UnsupportedModelError(
            f"{case['id']}: 模型 {model_name} 尚未注册 Runner."
        ) from error

    params = flatten_parameters(case)
    config = assembler.build_config(params)

    # 应力约束算例的迭代上限字段是 max_al_iterations, 其 max_iterations 为只读属性
    field_names = {field.name for field in fields(config)}
    iteration_field = (
        "max_al_iterations" if "max_al_iterations" in field_names else "max_iterations"
    )
    changes = {
        field: value
        for field, value in (
            *((field, getattr(overrides, attribute, None)) for field, attribute in _OVERRIDE_FIELDS),
            (iteration_field, overrides.max_iterations),
        )
        if value is not None
    }
    unsupported = set(changes) - field_names
    if unsupported:
        raise ConfigurationError(f"本算例不支持参数: {sorted(unsupported)}.")
    changes.update(
        _override_changes(config, getattr(overrides, "overrides", {}), changes)
    )
    config = replace(config, **changes)
    if config.nx <= 0 or config.ny <= 0:
        raise ConfigurationError("覆盖后的 nx 和 ny 必须为正数.")
    # 奇偶性按剖分方式: 棋盘格要 nx, ny 偶数, 镜像对称单向对角只要 nx 偶数, 单向对角无要求
    mesh_type = getattr(config, "mesh_type", "triangle-checkerboard")
    sizes = {"nx": config.nx, "ny": config.ny}
    odd = [axis for axis in MESH_EVEN_AXES.get(mesh_type, ("nx", "ny")) if sizes[axis] % 2]
    if odd:
        raise ConfigurationError(f"{mesh_type} 剖分要求覆盖后的 {' 和 '.join(odd)} 为偶数.")
    if config.max_iterations <= 0:
        raise ConfigurationError("覆盖后的最大迭代次数必须为正数.")

    factory = (
        assembler.build_analysis_pipeline if analysis_only else assembler.build_pipeline
    )
    return factory(config, params, method, order, model_name), config, changes


# 对称半域模型: 左半域柔顺度为完整域的一半, 报告完整结构柔顺度时乘以 2
_HALF_DOMAIN_MODELS = {"FixedFixedBeamHalfDomain2d"}

# 实体区口径: 报告未加权实体应力比时用的默认密度阈值 (config 未注册
# acceptance_solid_threshold 时的回退值). 2026-09-18 起, 注册了该阈值的算例在同一
# 子集上判定 C2 终态复核, 见 _solid_region_threshold.
_SOLID_REGION_THRESHOLD = 0.5


def _solid_region_threshold(config: Any) -> float:
    """实体区密度阈值: config 注册了 acceptance_solid_threshold 则用之, 否则回退 0.5."""
    threshold = getattr(config, "acceptance_solid_threshold", None)
    return _SOLID_REGION_THRESHOLD if threshold is None else float(threshold)


def _effective_interpolation(pipeline: Any) -> dict[str, Any] | None:
    """实际生效的材料插值对象, 而非配置里写的值.

    ``MaterialInterpolationScheme`` 只在材料近不可压缩 (``is_incompressible``)
    时插值 Poisson 比; 这里按分析器手里的插值格式与材料重算一遍, summary.json
    用它, 避免读日志的人误以为可压缩组也插值了 nu.
    """
    scheme = getattr(pipeline.analyzer, "interpolation_scheme", None)
    material = getattr(pipeline.analyzer, "material", None)
    if scheme is None or material is None:
        return None
    options = scheme.options
    targets = list(options.get("target_variables", ["E"]))
    nu_active = "nu" in targets and bool(getattr(material, "is_incompressible", False))
    if not nu_active:
        return {"variables": "E"}
    return {
        "variables": "E+nu",
        "nu_penalty_factor": float(options.get("nu_penalty_factor", 1.0)),
        "void_poisson_ratio": float(options.get("void_poisson_ratio", 0.3)),
    }


def _display_path(path: Path) -> str:
    """路径能相对当前工作目录就打相对形式, 否则打绝对路径; 回执里的路径可直接复制."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def print_run_banner(
    case: dict[str, Any],
    method: str,
    order: int,
    position: tuple[int, int],
    output: Path,
) -> None:
    """在迭代日志之前打印进度 (第几组 / 共几组) 与产物目录.

    完整配置随 summary.json 落盘, 不在回显里重复.
    """
    index, total = position
    print(f"[run] {index}/{total} | {case.get('title', '-')} | analyzer={method}, order={order}")
    print(f"[output] {_display_path(output)}/")


def run_one(
    case: dict[str, Any],
    method: str,
    order: int,
    arguments: argparse.Namespace,
    position: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    """执行单一阶次与离散方法的完整拓扑优化管线."""
    model_name = case["model"]["name"]
    pipeline, config, changes = build_model_pipeline(case, method, order, arguments)
    if pipeline.optimizer is None:
        raise RuntimeError("优化模式要求已创建优化器.")
    label = _run_label(method, order, config, changes)
    output = arguments.output / case["id"] / label
    print_run_banner(case, method, order, position, output)

    density, history = pipeline.optimizer.optimize(
        design_variable=pipeline.design_variable,
        density_distribution=pipeline.density_distribution,
    )
    state = pipeline.analyzer.solve_state(rho_val=density)

    # 两类列式的目标函数不同: 应力算例 min 体积, 柔顺度算例 min 柔顺度.
    # 不属于本列式的那一半没有意义, 故按列式取舍, 不拿 0.0 占位冒充数值结果.
    volume_minimizing = hasattr(pipeline, "volume_objective")
    if volume_minimizing:
        compliance = None
        volume_fraction = float(pipeline.volume_objective.fun(density))
    else:
        compliance = pipeline.objective.fun(density=density, state=state)
        volume_fraction = config.volume_fraction + float(pipeline.constraint.fun(density))

    # 终止判据的唯一出处是优化器: ALMMMAOptimizer 按 C0 连续化终止 / C1 设计稳定 /
    # C2 松弛可行 / C3 持续 hold_steps 步给出结论. 不具备该属性的优化器 (柔顺度算例
    # 的 OC/MMA) 退回原有的设计变化单条件, 不在消费端另拼一套判据.
    optimizer_verdict = getattr(pipeline.optimizer, "converged", None)
    termination_reason = getattr(pipeline.optimizer, "termination_reason", None)
    if optimizer_verdict is None:
        converged = bool(history.changes and history.changes[-1] <= config.change_tolerance)
    else:
        converged = bool(optimizer_verdict)

    summary = {
        "case_id": case["id"],
        "model": model_name,
        "method": method,
        "order": order,
        "mesh_type": getattr(config, "mesh_type", None),
        "nx": config.nx,
        "ny": config.ny,
        "optimizer": config.optimizer,
        "load_discretization": getattr(config, "load_discretization", None),
        "integration_order": 2 * order + 2,
        "stabilization_coefficient": getattr(pipeline.analyzer, "stabilization_coefficient", None),
        "volume_fraction": volume_fraction,
        "relative_equilibrium_residual": relative_residual(pipeline, state),
        "optimization_iterations": len(history.iter_indices),
        "converged": converged,
        "termination_reason": termination_reason,
        "solver": config.solve_method,
        "interpolation": _effective_interpolation(pipeline),
        "overrides": {name: str(value) for name, value in sorted(changes.items())} or None,
        "provenance": provenance.run_stamp(),
    }
    if volume_minimizing:
        summary["stress_constraint_type"] = type(pipeline.stress_constraint).__name__
        summary["lfem_stress_constraint_formulation"] = (
            config.stress_constraint_formulation
        )
        summary["stress_constraint_formulation"] = resolve_stress_constraint_formulation(
            config, method
        )
        # 表观应力比用于展示, 约束对象定义的超限量 g 用于验收.
        # 最终密度重新评价, 不从历史末项复制可能过期的应力.
        constraint_values = pipeline.stress_constraint.fun(density, state)
        stress_measure = pipeline.stress_constraint.compute_stress_measure(density, state)
        summary["max_constraint"] = float(constraint_values.max())
        summary["max_von_mises"] = float(stress_measure.max())
        relative_violation = pipeline.stress_constraint.compute_relative_violation(density, state)
        summary["max_apparent_stress_ratio"] = float(stress_measure.max())
        summary["max_relative_violation"] = float(relative_violation.max())
        # density_final.vtu 带同一次终态求解的应力比, 与 max_apparent_stress_ratio 严格一致.
        final_cell_fields = {"von_mises_normalized": stress_measure}
        summary["relative_stress_tolerance"] = config.stress_tolerance
        # C2 终态复核与优化器同一口径: 注册了 acceptance_solid_threshold 时只看
        # rho_phys >= 阈值的单元 (2026-09-18 起), 否则全域; 全域值始终另存作诊断.
        density_array = np.asarray(bm.to_numpy(density[:])).reshape(-1)
        solid_threshold = _solid_region_threshold(config)
        solid_region = density_array >= solid_threshold
        relative_violation_cell = np.asarray(bm.to_numpy(relative_violation)).reshape(
            density_array.shape[0], -1).max(axis=1)
        summary["acceptance_solid_threshold"] = (
            None if config.acceptance_solid_threshold is None
            else float(config.acceptance_solid_threshold))
        summary["max_relative_violation_solid_region"] = (
            float(relative_violation_cell[solid_region].max())
            if bool(solid_region.any()) else None)
        c2_quantity = (
            summary["max_relative_violation"]
            if config.acceptance_solid_threshold is None
            or summary["max_relative_violation_solid_region"] is None
            else summary["max_relative_violation_solid_region"])
        summary["relative_stress_feasible"] = bool(c2_quantity <= config.stress_tolerance)
        # 保留旧字段供现有消费者读取, 含义与相对超限判据一致.
        summary["stress_feasible"] = summary["relative_stress_feasible"]
        # 未加权实体应力比: 强制诊断量, 只报告不判据. 验收量按刚度加权, 空洞
        # 单元的 m_E ~ 1e-9 会把它压到阈值之下, 未加权量才反映实体材料真实的
        # 应力水平; 其在实体区的取值直接反映真实承载安全性.
        solid_stress_ratio = np.asarray(bm.to_numpy(
            pipeline.stress_constraint.compute_solid_stress_ratio(density, state)))
        summary["max_solid_stress_ratio"] = float(solid_stress_ratio.max())
        summary["solid_region_threshold"] = solid_threshold
        summary["max_solid_stress_ratio_solid_region"] = (
            float(solid_stress_ratio[solid_region].max())
            if bool(solid_region.any()) else None)
        # 几何奇点垫片: 必须随产物落盘, 否则 summary 里的 max_constraint /
        # max_relative_violation 无从知道是在哪个集合上取的最大值. 垫片内的量
        # 一律单列, 使"垫片掩盖了多大的应力"可被直接读出, 不靠重算 —— 读的时候
        # 要看 max_constraint_pad 而不是 max_solid_stress_ratio_pad: 后者会被
        # rho~0 的空洞单元主导 (那里 g 恒等于 -epsilon, 无害).
        # 两侧垫片分列: 载荷侧的牵引间断端点与支撑侧的固支角点是两个独立的几何
        # 奇点, 半径各自标定, 因此诊断量也必须能分开读, 否则无从判断某一侧的半径
        # 是否取够 (取小了热点会搬到掩码边界, 表现为约束区最大值不降反升).
        summary["load_pad_radius"] = config.load_pad_radius
        summary["support_pad_radius"] = getattr(config, "support_pad_radius", 0.0)
        problem = getattr(pipeline, "problem", None)
        endpoints = getattr(problem, "traction_patch_endpoints", ())
        summary["load_pad_centers"] = [
            [float(value) for value in center] for center in endpoints
        ] or None
        corners = getattr(problem, "clamped_corner_points", ())
        summary["support_pad_centers"] = [
            [float(value) for value in center] for center in corners
        ] or None
        pad_mask = getattr(pipeline.stress_constraint, "exemption_mask", None)
        if pad_mask is None:
            summary["pad_cells"] = 0
            summary["load_pad_cells"] = 0
            summary["support_pad_cells"] = 0
            summary["max_solid_stress_ratio_constrained"] = (
                summary["max_solid_stress_ratio"])
            summary["max_solid_stress_ratio_pad"] = None
            summary["max_constraint_pad"] = None
            summary["max_solid_stress_ratio_support_pad"] = None
            summary["max_constraint_support_pad"] = None
        else:
            pad = np.asarray(bm.to_numpy(pad_mask)).astype(bool)

            def _component_mask(name: str) -> np.ndarray:
                """取单侧垫片掩码; 旧 pipeline 未分列时退化为全 False."""
                component = getattr(pipeline, name, None)
                if component is None:
                    return np.zeros_like(pad)
                return np.asarray(bm.to_numpy(component)).astype(bool)

            load_pad = _component_mask("load_pad_mask")
            support_pad = _component_mask("support_pad_mask")
            # 垫片内约束值要用未豁免的口径重算: pipeline 的 constraint_values
            # 在这些单元上已被换成哨兵 -1.0, 直接取最大值只会读回哨兵.
            pad_constraint = np.asarray(bm.to_numpy(
                pipeline.stress_constraint.compute_unexempted_constraint(
                    density, state)))
            summary["pad_cells"] = int(pad.sum())
            summary["load_pad_cells"] = int(load_pad.sum())
            summary["support_pad_cells"] = int(support_pad.sum())
            summary["max_solid_stress_ratio_constrained"] = (
                float(solid_stress_ratio[~pad].max())
                if bool((~pad).any()) else None)
            summary["max_solid_stress_ratio_pad"] = (
                float(solid_stress_ratio[load_pad].max())
                if bool(load_pad.any()) else None)
            summary["max_constraint_pad"] = (
                float(pad_constraint[load_pad].max())
                if bool(load_pad.any()) else None)
            summary["max_solid_stress_ratio_support_pad"] = (
                float(solid_stress_ratio[support_pad].max())
                if bool(support_pad.any()) else None)
            summary["max_constraint_support_pad"] = (
                float(pad_constraint[support_pad].max())
                if bool(support_pad.any()) else None)
        # ALM 内部状态的必报诊断量, 同样不进判据. mu 在本实现里是全局标量,
        # 逐外层步统一放大, 不是逐单元罚参数. 乘子相对变化只有优化器能算
        # (需要上一外层步的 lambda), 故读优化器属性而非在此重建.
        al_objective = getattr(pipeline, "al_objective", None)
        if al_objective is not None:
            multiplier = np.asarray(bm.to_numpy(al_objective.lamb))
            constraint_np = np.asarray(bm.to_numpy(constraint_values))
            summary["penalty_parameter"] = float(al_objective.mu)
            summary["max_multiplier"] = float(multiplier.max())
            summary["complementarity_residual"] = float(
                np.abs(multiplier * constraint_np).max())
        multiplier_change = getattr(pipeline.optimizer, "last_multiplier_change", None)
        # 外层一步未走完时该量为 nan, 落盘写 None 而不是非法 JSON 的 NaN.
        summary["multiplier_relative_change"] = (
            None if multiplier_change is None or not np.isfinite(multiplier_change)
            else float(multiplier_change))
        # 停止准则 C1 的度量对象与移动限制的生效值必须原样落盘: 悄悄换掉度量
        # 比原问题更糟, 读者需要能自己核 change 与 change_physical 的比值.
        optimizer_options = getattr(pipeline.optimizer, "options", None)
        summary["change_measure"] = str(
            getattr(optimizer_options, "change_measure", "design"))
        for key in ("last_change_outer_mean", "last_change_outer_max"):
            value = getattr(pipeline.optimizer, key, None)
            summary[key.replace("last_", "")] = (
                None if value is None or not np.isfinite(value) else float(value))
        lambda_max = getattr(optimizer_options, "lambda_max", None)
        summary["lambda_max"] = None if lambda_max is None else float(lambda_max)
        summary["multiplier_capped_count"] = int(
            getattr(pipeline.optimizer._al_objective, "last_capped_count", 0)
            if getattr(pipeline.optimizer, "_al_objective", None) is not None else 0)
        move_limit_base = getattr(optimizer_options, "move_limit", None)
        summary["move_limit_base"] = (
            None if move_limit_base is None else float(move_limit_base))
        floor = getattr(optimizer_options, "asymptote_min_distance", None)
        summary["asymptote_min_distance"] = (
            None if floor is None else float(floor))
        # 优化器的判据基于迭代末态; 这里在终态密度重新求解后再核一次 C2,
        # 两者都成立才记为收敛.
        summary["converged"] = bool(converged and summary["stress_feasible"])
    else:
        final_cell_fields = None
        # 原始产物统一保存计算域柔顺度, 完整结构换算只在展示层进行.
        summary["compliance"] = float(compliance)
        summary["compliance_domain"] = "half" if model_name in _HALF_DOMAIN_MODELS else "full"
        summary["full_structure_factor"] = 2.0 if model_name in _HALF_DOMAIN_MODELS else 1.0
    write_optimization_result(
        output, pipeline, density, history, summary, final_cell_fields=final_cell_fields,
    )
    if volume_minimizing:
        max_von_mises = summary["max_von_mises"]
        stress_note = "" if max_von_mises is None else f", max_vm={max_von_mises:.6f}"
        result_text = f"volfrac={volume_fraction:.6f}{stress_note}"
    else:
        result_text = f"compliance={compliance:.8e}, volfrac={volume_fraction:.6f}"
    print(f"[done] {_display_path(output)}/: {result_text}")
    return summary


def main(argv: list[str] | None = None) -> int:
    """列出、校验或执行论文拓扑优化算例."""
    arguments = parse_arguments(argv)
    try:
        cases = load_cases(arguments.config)
        if arguments.list:
            for case in cases:
                print(f"{case['id']}\t{case['model']['name']}\t{case['status']}")
            return 0

        selected = select_cases(cases, arguments.case)
        prepared = []
        for case in selected:
            runs = resolve_runs(case, arguments)
            if arguments.check_only:
                pipeline, config, _ = build_model_pipeline(
                    case, "lfem", int(case["discretization"]["comparison_orders"][0]),
                    arguments, analysis_only=True,
                )
                prepared.append(
                    configuration_summary(case, config, runs, pipeline.problem.domain)
                )
            else:
                prepared.extend(
                    run_one(case, method, order, arguments, (index, len(runs)))
                    for index, (method, order) in enumerate(runs, 1)
                )

    except (ConfigurationError, UnsupportedModelError, KeyError, TypeError, ValueError) as error:
        print(f"配置错误: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if arguments.check_only:
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return 0
