"""Hu--Zhang 拓扑优化投稿论文实验主 Runner.

本脚本在同一物理问题、同一材料插值 (MSIMP) 与统一网格下比较两条求解路径:
1. ``LagrangeFEMAnalyzer``: 经典 Lagrange 位移有限元 (LFEM) 求解链;
2. ``HuZhangMFEMAnalyzer``: 对称弱形式胡--张混合有限元 (Hu--Zhang) 求解链.

受控比较协议:
- 给定阶次 ``k``, LFEM 采用位移阶 ``p=k``; Hu--Zhang 采用应力阶 ``k`` (对应位移阶 ``k-1``);
- 统一高斯积分阶 ``q=2k+2``;
- 载荷统一通过 ``project_patch_traction_to_p1_trace`` 投影至底边连续 P1 迹空间, 消除强施加与弱积分的几何不对齐误差;
- 正文比较阶次为 ``k=2, 3, 4``; ``k=1`` 因 P0 常数位移空间缺失刚体旋转模态、在拓扑演化中
  引发人工刚度硬化而不进正文, 仅由 compliance_k1_comparison 作为反例呈现 (故仍在部分算例的
  ``comparison_orders`` 中声明), 见 docs/fem/huzhang-mixed-fem-implementation.md.

运行模式:
- ``--mode optimization`` (默认): 执行完整 OC 拓扑优化迭代, 产物写入
  ``outputs/<case-id>/analyzer-<链>__order-<k>[__<字段>-<取值>...]/``,
  包含最终密度场 ``density_final.vtu``、收敛历史 ``history.json`` 与运行摘要 ``summary.json``;
- ``--mode state-compare``: 在固定初始密度 (rho=0.4) 下执行单次状态前向分析, 输出相对柔顺度差异与能量恒等式诊断.

本模块同时承载能量恒等式诊断、真相对残差与结果落盘 (原 ``diagnostics.py``).

使用方法:
    # 1. 运行固定梁单次状态对比 (k=2,3)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed-half --analyzer all --order 2 --order 3 --mode state-compare --solver scipy

    # 2. 运行胡张元 (k=2) 拓扑优化 (冒烟测试 3 步)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed-half --analyzer huzhang --order 2 --max-iterations 3 --solver scipy

    # 3. 运行全量论文矩阵对比 (LFEM 与 Hu--Zhang, k=2,3,4)
    python experiments/huzhang_topopt_paper/run.py --case compliance-fixed-fixed-half --analyzer all --solver scipy
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from importlib import import_module
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
from pipeline import assembler_for

import numpy as np

from fealpy.backend import backend_manager as bm


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


def write_optimization_result(
    output: Path,
    pipeline: Any,
    density: Any,
    history: Any,
    summary: dict[str, Any],
) -> None:
    """保存最终密度场 VTU、标量收敛历史与运行摘要 JSON.

    参数:
        output: 输出目标文件夹路径.
        pipeline: 优化管线对象.
        density: 最终单元密度场.
        history: 优化迭代历史记录对象.
        summary: 汇总指标字典.
    """
    output.mkdir(parents=True, exist_ok=True)
    from soptx.postprocess.vtk_export import write_vtu
    from fealpy.backend import backend_manager as bm

    # 1. 最终密度场便捷文件
    density_np = np.asarray(bm.to_numpy(density[:]), dtype=np.float64).flatten()
    write_vtu(
        mesh=pipeline.mesh,
        filepath=str(output / "density_final"),
        cell_data={"density": density_np},
    )

    # 2. 完整迭代演化序列 (供 ParaView 作为动画时间序列直接加载)
    if hasattr(history, "physical_densities") and history.physical_densities:
        vtu_dir = output / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        for iter_idx, rho_i in enumerate(history.physical_densities, start=1):
            rho_i_np = np.asarray(bm.to_numpy(rho_i), dtype=np.float64).flatten()
            write_vtu(
                mesh=pipeline.mesh,
                filepath=str(vtu_dir / f"density_iter_{iter_idx:03d}"),
                cell_data={"density": rho_i_np},
            )
    history_payload = {
        "iter_indices": history.iter_indices,
        "changes": history.changes,
        "iteration_times": history.iteration_times,
        "scalar_histories": history.scalar_histories,
    }
    (output / "history.json").write_text(
        json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_state_comparison(payload: dict[str, Any]) -> None:
    """在终端格式化打印单次状态分析的关键比较指标与能量诊断.

    参数:
        payload: 状态对比数据字典.
    """
    print()
    print("固定初始密度状态对比")
    extra_info = []
    if "plane_type" in payload:
        extra_info.append(f"plane_type={payload['plane_type']}")
    if "poisson_ratio" in payload and payload["poisson_ratio"] is not None:
        extra_info.append(f"nu={payload['poisson_ratio']}")
    extra_str = f", {', '.join(extra_info)}" if extra_info else ""
    print(
        f"  case={payload['case_id']}, model={payload['model']}{extra_str}, "
        f"rho0={payload['initial_density']:.3f}"
    )
    print("  协议: 给定阶次 k, LFEM 采用位移阶 p=k; Hu--Zhang 采用应力阶 k (对应位移阶 k-1); 统一高斯积分阶 q=2k+2.")
    print()
    print("  method   k   q       compliance          volfrac       residual")
    print("  ------- --- --- ------------------ ------------- ----------------")
    for row in payload["rows"]:
        print(
            f"  {row['method']:<7} {row['order']:>3} {row['integration_order']:>3} "
            f"{row['compliance']:>18.8e} {row['volume_fraction']:>13.6f} "
            f"{row['relative_equilibrium_residual']:>16.3e}"
        )
        energy = row["energy_diagnostics"]
        if row["method"] == "lfem":
            print(
                f"           能量: fTu={energy['external_work']:.8e}, "
                f"uKu={energy['internal_energy']:.8e}, "
                f"相对缺陷={energy['relative_defect']:.3e}"
            )
        else:
            print(
                f"           能量: sigmaAsigma={energy['complementary_energy']:.8e}, "
                f"sigmaBu={energy['coupling_work']:.8e}, "
                f"牵引对偶功={energy['traction_dual_work']:.8e}"
            )

    rows_by_order: dict[int, dict[str, dict[str, Any]]] = {}
    for row in payload["rows"]:
        rows_by_order.setdefault(row["order"], {})[row["method"]] = row
    print()
    for order, rows in sorted(rows_by_order.items()):
        lfem = rows.get("lfem")
        huzhang = rows.get("huzhang")
        if lfem is None or huzhang is None:
            continue
        diff = abs(huzhang["compliance"] - lfem["compliance"]) / max(
            abs(lfem["compliance"]), 1.0e-30
        )
        print(f"  k={order}: |C_HZ - C_LFEM| / |C_LFEM| = {diff:.2%}")
    print(
        "  注: 残差小说明各自线性系统已解收敛. 结构合力守恒由 "
        "examples/huzhang_elasticity/concentrated_load_demo.py 承担核查."
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
        "--filter-type", choices=("density", "sensitivity", "projection"),
        help="临时覆盖过滤器类型 (density/sensitivity/projection), 默认取 cases.toml.",
    )
    parser.add_argument("--nx", type=int, help="临时覆盖横向网格剖分数.")
    parser.add_argument("--ny", type=int, help="临时覆盖纵向网格剖分数.")
    parser.add_argument("--max-iterations", type=int, help="临时覆盖最大优化迭代次数.")
    parser.add_argument(
        # 具名开关只覆盖最常用的那几个字段; 其余字段走这个通用通道, 免得每加一个
        # 参数就往 parser 里塞一个开关。字段名以 build_config 造出的配置对象为准,
        # 也就是 cases.toml 的 discretization/optimization 键名; 运行组合维度
        # analyzer / order 也一并接住 (见 _apply_run_selection), 口径与 EA/FA
        # 两个实验的 --override 一致。
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
        "--mode",
        choices=("optimization", "state-compare"),
        default="optimization",
        help="运行 OC 拓扑优化, 或仅执行相同初始密度下的单次状态对比.",
    )
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

    同一字段重复给出直接判错: 静默取最后一次会让命令行与实际跑的参数对不上。
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
    --override 的列表写法一致; ``analyzer`` 同时给出两条链等价于 ``all``。
    与具名开关撞车时报错而不是定先后顺序, 口径同 ``_override_changes``。
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

_BOOLEAN_TEXTS = {"true": True, "false": False}


def _coerce(name: str, text: str, current: Any) -> Any:
    """按配置对象里现有取值的类型转换覆盖文本; 类型不认识就原样当字符串.

    不读 dataclass 的类型注解: 模块开头有 from __future__ import annotations,
    注解此时是字符串, 拿现值的类型更可靠。
    """
    if isinstance(current, bool):
        if text.lower() not in _BOOLEAN_TEXTS:
            raise ConfigurationError(f"覆盖值非法: {name}={text} (需要 true/false).")
        return _BOOLEAN_TEXTS[text.lower()]
    for caster in (int, float):
        if isinstance(current, caster):
            try:
                return caster(text)
            except ValueError as error:
                raise ConfigurationError(
                    f"覆盖值非法: {name}={text} (需要 {caster.__name__})."
                ) from error
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
        # 否则报错信息会让人以为这两个维度不能用 --override 给。
        available = sorted(field_names | set(_RUN_SELECTION_FIELDS))
        raise ConfigurationError(
            f"未知的覆盖字段: {', '.join(unknown)}; 可覆盖: {', '.join(available)}."
        )
    # 与具名开关撞车时报错而不是定一个先后顺序: 两个写法给同一个字段不同取值,
    # 无论哪边赢都有一半命令行是假的。
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
    标签; 其余字段只在被覆盖时进标签, 探索性运行因此不会盖掉注册运行的产物。标签
    按字段名排序、用 ``__`` 连接, 与 experiments/topopt_simp_fa|ea 的第二层同一套
    写法, 三个实验的 outputs/ 用同一种读法。
    """
    tags: dict[str, Any] = {"analyzer": method, "order": order}
    tags.update({name: getattr(config, name) for name in changes})
    return _TAG_SEPARATOR.join(
        re.sub(r"[^0-9A-Za-z_.\-]+", "-", f"{name}-{tags[name]}")
        for name in sorted(tags)
    )


# 命令行临时覆盖: 配置字段名 -> 命名空间属性名 (迭代上限字段按算例另行确定)
_OVERRIDE_FIELDS = (
    ("nx", "nx"),
    ("ny", "ny"),
    ("solve_method", "solver"),
    ("optimizer", "optimizer"),
    ("filter_type", "filter_type"),
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

    第三个返回值是本次实际生效的覆盖改动, 供 run 目录名判断要不要另起标签。
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
            *((field, getattr(overrides, attribute)) for field, attribute in _OVERRIDE_FIELDS),
            (iteration_field, overrides.max_iterations),
        )
        if value is not None
    }
    changes.update(
        _override_changes(config, getattr(overrides, "overrides", {}), changes)
    )
    config = replace(config, **changes)
    if config.nx <= 0 or config.ny <= 0 or config.nx % 2 or config.ny % 2:
        raise ConfigurationError("覆盖后的 nx 和 ny 必须为正偶数.")
    if config.max_iterations <= 0:
        raise ConfigurationError("覆盖后的最大迭代次数必须为正数.")

    factory = (
        assembler.build_analysis_pipeline if analysis_only else assembler.build_pipeline
    )
    return factory(config, params, method, order, model_name), config, changes


# 对称半域模型: 左半域柔顺度为完整域的一半, 报告完整结构柔顺度时乘以 2
_HALF_DOMAIN_MODELS = {"FixedFixedBeamHalfDomain2d"}


def _discretization_note(pipeline: Any, order: int) -> str:
    """有限元格式; 含低阶稳定化的实际取值."""
    if pipeline.method == "lfem":
        assembly = getattr(pipeline.analyzer, "assembly_method", "standard")
        return f"analyzer=lfem 位移 P{order} (assembly_method={assembly})"

    # Hu--Zhang: 应力阶 k, 位移为间断 P(k-1); k >= GD + 1 时原生稳定, 否则加跳量稳定化
    geo_dimension = int(pipeline.mesh.geo_dimension())
    if order >= geo_dimension + 1:
        stability = f"原生稳定 (k >= GD+1 = {geo_dimension + 1})"
    else:
        scheme = getattr(pipeline.analyzer, "stabilization", "matrix_jump")
        stability = (
            f"stabilization=none (消融, k <= GD = {geo_dimension})"
            if scheme == "none"
            else f"stabilization={scheme} (k <= GD = {geo_dimension})"
        )
    return f"analyzer=huzhang 应力 k={order} + 间断位移 P{order - 1}, {stability}"


def _domain_note(problem: Any) -> str:
    """几何区域; 取自物理问题本身, 保证与实际剖分的 box 一致 (半域算例即半域)."""
    domain = getattr(problem, "domain", None)
    if domain is None or len(domain) != 4:
        return ""
    x0, x1, y0, y1 = (float(v) for v in domain)
    return f"domain=[{x0:g}, {x1:g}] x [{y0:g}, {y1:g}], "


def _load_note(config: Any) -> str:
    """载荷大小与分布宽度; 轴承算例用 traction 字段, 其余算例用 load/load_width."""
    load = getattr(config, "load", None)
    if load is None:
        traction = getattr(config, "traction", None)
        return "" if traction is None else f", traction={traction}"
    # 只写连续问题本身: 宽度不是修饰, 少了它 load=-400 会被读成集中力.
    width = getattr(config, "load_width", None)
    suffix = "" if width is None else f" (load_width={width:g})"
    return f", load={load:g}{suffix}"


def _load_scheme_note(config: Any) -> str:
    """载荷离散方式; 取值是方案名 (p1_trace_l2_projection / patch), 不是分段数."""
    scheme = getattr(config, "load_discretization", None)
    return "" if scheme is None else f", load_discretization={scheme}"


def _formulation_note(pipeline: Any, config: Any) -> str:
    """优化列式; 应力算例与柔顺度算例的目标与约束正好对调, 必须分别表述."""
    if hasattr(pipeline, "volume_objective"):
        return f"min 体积 s.t. stress_limit <= {config.stress_limit:g}"
    return f"min 柔顺度 s.t. volume_fraction <= {config.volume_fraction:g}"


def _algorithm_note(config: Any) -> str:
    """优化算法与终止准则; AL-MMA 的迭代预算是两层的, 与 OC 的单层写法不同.

    数值一律按配置字段名打印 (不写 max_iter 这类简称): 回执上看到的名字就是
    ``--override`` 能用的名字。应力算例的 max_iterations 是只读属性, 真正能改的
    是 max_al_iterations, 打字段名同时也把这件事说清楚了。
    """
    outer = getattr(config, "max_al_iterations", None)
    budget = (
        f"max_iterations={config.max_iterations}"
        if outer is None
        else f"max_al_iterations={outer} x mma_iters_per_al={config.mma_iters_per_al}"
    )
    parts = [
        config.optimizer.upper(),
        f"终止准则 {budget} 或 change_tolerance<={config.change_tolerance:g}",
    ]
    stress_tolerance = getattr(config, "stress_tolerance", None)
    if stress_tolerance is not None:
        parts.append(f"应力容差 stress_tolerance={stress_tolerance:g}")
    return ", ".join(parts)


def print_run_banner(
    case: dict[str, Any],
    method: str,
    order: int,
    pipeline: Any,
    config: Any,
    position: tuple[int, int] = (1, 1),
) -> None:
    """在迭代日志之前打印本次运行的物理问题、离散、规模与优化列式.

    分行与 cases.toml 同一分类轴: A 问题 / B 离散 / C 拓扑建模 / D 算法, 标签词汇
    与 topopt_simp_fa 对齐, 两个实验的同名行是同一件事、改起来也是同一处。
    B 层再拆成网格 / 空间 / 求解三行: 受控比较里网格固定、方法 x 阶次才是变量,
    分开写便于批量运行时扫读; 状态求解单列 ``[solve]``, 因为鞍点系统对称不定、
    只能用直接法, 这是离散格式的后果而非规模的后果。
    载荷大小与宽度是连续问题, 归 ``[problem]``; 载荷怎么离散到网格上是 B 层决策,
    归 ``[space]``。设计变量数与初始密度归 ``[topopt]``: 它们是拓扑建模层的产物,
    不是算法参数, 且单元数 ``[mesh]`` 已经给过。
    方法与阶次只在 ``[space]`` 出现一次 (``analyzer=lfem 位移 P2`` /
    ``analyzer=huzhang 应力 k=2``
    已完整编码二者); case id 由 run.py 的 ``[case]`` 行给出, ``[run]`` 不重复打,
    只作进度行 (第几组 / 共几组), 便于 --full 连跑时定位当前是哪一组。
    自由度按各空间实取而非按节点数推算。
    B/C/D 三层的可调数值一律按配置字段名打印 (penalty_factor 而非 p, filter_radius
    而非 rmin), 回执上看到的名字就是 ``--override`` 能用的名字; 只有 E / nu 例外,
    保留连续问题的论文符号。
    """
    mesh = pipeline.mesh
    n_cells = int(mesh.number_of_cells())

    index, total = position
    print(f"[run] {index}/{total} | {case.get('title', '-')}")
    print(
        f"[problem] model={case['model']['name']}, "
        f"{_domain_note(pipeline.problem)}"
        f"E={config.youngs_modulus:g}, nu={config.poisson_ratio:g}, "
        f"{config.plane_type}{_load_note(config)}"
    )
    print(
        f"[mesh] {type(mesh).__name__} nx={config.nx}, ny={config.ny} = {n_cells} 单元, "
        f"{int(mesh.number_of_nodes())} 节点"
    )

    n_disp = int(pipeline.analyzer.tensor_space.number_of_global_dofs())
    stress_space = getattr(pipeline.analyzer, "huzhang_space", None)
    if stress_space is None:
        dof_text = f"位移 {n_disp}"
    else:
        n_stress = int(stress_space.number_of_global_dofs())
        dof_text = f"应力 {n_stress} + 位移 {n_disp} = {n_stress + n_disp}"
    relaxation = getattr(config, "use_relaxation", None)
    relaxation_text = (
        "" if relaxation is None or method != "huzhang"
        else f", use_relaxation={'true' if relaxation else 'false'}"
    )
    print(
        f"[space] {_discretization_note(pipeline, order)}{relaxation_text}, "
        f"自由度 {dof_text}, 积分阶 {2 * order + 2}{_load_scheme_note(config)}"
    )

    initial_density = getattr(config, "initial_density", None)
    initial_text = (
        ""
        if initial_density is None
        else f", 初值均匀 initial_density={initial_density:g}"
    )
    print(
        f"[topopt] {_formulation_note(pipeline, config)}, "
        f"interpolation_method={config.interpolation_method} "
        f"(penalty_factor={config.penalty_factor:g}, "
        f"void_youngs_modulus={config.void_youngs_modulus:g}), "
        f"filter_type={config.filter_type} (filter_radius={config.filter_radius:g}), "
        f"设计变量 {n_cells} (单元密度){initial_text}"
    )
    print(f"[solve] solve_method={config.solve_method} (直接法)")
    print(f"[optim] {_algorithm_note(config)}")


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
    print_run_banner(case, method, order, pipeline, config, position)

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

    label = _run_label(method, order, config, changes)
    summary = {
        "case_id": case["id"],
        "model": model_name,
        "method": method,
        "order": order,
        "nx": config.nx,
        "ny": config.ny,
        "optimizer": config.optimizer,
        "integration_order": 2 * order + 2,
        "volume_fraction": volume_fraction,
        "relative_equilibrium_residual": relative_residual(pipeline, state),
        "optimization_iterations": len(history.iter_indices),
        "converged": bool(history.changes and history.changes[-1] <= config.change_tolerance),
        "solver": config.solve_method,
        "overrides": {name: str(value) for name, value in sorted(changes.items())} or None,
        "provenance": provenance.run_stamp(),
    }
    if volume_minimizing:
        # 归一化最大 von Mises: <= 1 即应力约束满足, 是这类算例的验收量
        stress_history = history.scalar_histories.get("max_von_mises")
        summary["max_von_mises"] = float(stress_history[-1]) if stress_history else None
    else:
        # 原始产物统一保存计算域柔顺度, 完整结构换算只在展示层进行.
        summary["compliance"] = float(compliance)
        summary["compliance_domain"] = "half" if model_name in _HALF_DOMAIN_MODELS else "full"
        summary["full_structure_factor"] = 2.0 if model_name in _HALF_DOMAIN_MODELS else 1.0
    write_optimization_result(
        arguments.output / case["id"] / label, pipeline, density, history, summary
    )
    if volume_minimizing:
        max_von_mises = summary["max_von_mises"]
        stress_note = "" if max_von_mises is None else f", max_vm={max_von_mises:.6f}"
        result_text = f"volfrac={volume_fraction:.6f}{stress_note}"
    else:
        result_text = f"compliance={compliance:.8e}, volfrac={volume_fraction:.6f}"
    print(f"{case['id']}/{label}: {result_text}")
    return summary


def run_state_comparison(
    case: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """在固定初始密度场下求解单次状态方程并收集对比指标."""
    model_name = case["model"]["name"]
    rows: list[dict[str, Any]] = []
    config: Any = None
    for method, order in resolve_runs(case, arguments):
        pipeline, config, _ = build_model_pipeline(
            case, method, order, arguments, analysis_only=True
        )
        state = pipeline.analyzer.solve_state(rho_val=pipeline.density_distribution)
        compliance = pipeline.objective.fun(
            density=pipeline.density_distribution,
            state=state,
        )
        volume_fraction = config.volume_fraction + float(
            pipeline.constraint.fun(pipeline.density_distribution)
        )
        rows.append(
            {
                "method": method,
                "order": order,
                "integration_order": 2 * order + 2,
                "compliance": float(compliance),
                "volume_fraction": volume_fraction,
                "relative_equilibrium_residual": relative_residual(pipeline, state),
                "energy_diagnostics": energy_identity_diagnostics(pipeline, state),
                "cells": int(pipeline.mesh.number_of_cells()),
            }
        )

    if config is None:
        raise ConfigurationError(f"{case['id']}: state-compare 模式没有可执行的方法与阶次组合.")

    payload = {
        "case_id": case["id"],
        "model": model_name,
        "mode": "state-compare",
        "compliance_domain": "half" if model_name in _HALF_DOMAIN_MODELS else "full",
        "full_structure_factor": 2.0 if model_name in _HALF_DOMAIN_MODELS else 1.0,
        "plane_type": getattr(config, "plane_type", "unknown"),
        "poisson_ratio": getattr(config, "poisson_ratio", None),
        "initial_density": config.volume_fraction,
        "comparison_protocol": "LFEM p=k versus Hu--Zhang stress order k, q=2k+2",
        "provenance": provenance.run_stamp(),
        "rows": rows,
    }
    output = arguments.output / case["id"] / f"state-comparison-{config.nx}x{config.ny}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "state_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_state_comparison(payload)
    print(f"\n{case['id']}/state-compare: 已完成 {len(rows)} 组单次状态分析.")
    return payload


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
            elif arguments.mode == "state-compare":
                prepared.append(run_state_comparison(case, arguments))
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
