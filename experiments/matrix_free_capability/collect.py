# -*- coding: utf-8 -*-
"""把各 case 的原始产物收成一份入库快照 ``figure_data/fig2_data.json``.

分工: ``cases.toml`` 描述"跑什么", 本模块描述"读什么、怎么判".
两者分开的原因是提取逻辑带门禁判断和跨源一致性核对, 写不进 TOML.

本模块只读产物、不触发计算; 缺产物时报缺失, 不会隐式重跑。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import config
import provenance

SCHEMA_VERSION = 1

# 门禁阈值的唯一来源是 tools/matrix_free_evidence/contract.py, 此处复述用于
# 离线校验; 两边不一致时以 contract.py 为准 (由 _check_contract_drift 检出).
MINIMUM_FINAL_L2_ORDER = 1.5
EA_FA_SOLUTION_RELATIVE_TOL = 1.0e-9
EXPECTED_REFINEMENT_LEVELS = 5
EXPECTED_BASE_SUBDIVISIONS = 4

# 两条误差链在重合档上的一致性门禁。只在 collect 阶段生效 —— 它核对的是
# examples/lagrange_elasticity 的 FA 链与本管线 EA 三档这两个不同来源, 不属于
# contract.py 覆盖的 stage-1 单跑/跨跑门禁, 故不进 _check_contract_drift。
# 数值上与旧判据 "最少吻合 8 位有效数字" 完全等价 (8 位 <=> 相对差 < 1e-8),
# 换成相对差只是把单位统一到全文其余门禁上, 不改变松紧。
EA_FA_ERROR_CHAIN_RELATIVE_TOL = 1.0e-8

# 峰值 RSS 的分阶段采样点顺序, 必须与 benchmark_cpu_ea.py 里 ``stages`` 的追加
# 顺序逐字一致。``load`` 是后加的采样点: 早于该次插桩的产物只有一个合并的 ``bc``
# 阶段, 无法把体力组装与 Dirichlet 处理分开, 因此按缺阶段直接判失败而不是静默补零。
STAGE_ORDER = ["baseline", "mesh", "operator", "load", "bc", "solve"]

# (c) 的验收门禁: CPU 与单卡 GPU 两侧解的相对差上限。取值与 (a) 的 EA/FA 判据
# 相同 —— 两处问的是同一件事 "换了实现路径后还是不是同一个解", 没有理由用两把尺子。
DEVICE_SOLUTION_RELATIVE_TOL = 1.0e-9

# (c) 在尚未注册 case 时的占位原因, 写进快照供 run.py 打印、消费方按需引用。
#
# 已于 2026-08-22 注册 c-dev-n{8,16,32,64} 四个 case, 正常路径下 panel c 走
# _collect_device_speedup 出实测值; 本常量只在 cases.toml 里 panel c 被清空时
# 才会再次生效, 保留它是为了让"退回占位"仍有可读的解释, 而不是抛 KeyError。
PANEL_C_REASON = (
    "(c) 问的是同一个 matrix-free 算子换到单块 GPU 上能快多少, 横轴与 (b) 同为自由度; "
    "验收门禁是 GPU 与 CPU 解的相对差不大于 1e-9。占位不是因为缺算子路径 —— 现有 EA "
    "实现不含 numpy/scipy 直接调用, 换 backend 即可上 CUDA —— 而是因为还没有注册 "
    "CPU/GPU 对照的 case: examples/gpu_elasticity 下已有的 GPU 结果走 2D 三角形全组装 "
    "路径, 与 (b) 的 3D 四面体 matrix-free 不同轴, 不能直接充数。"
)


# (d) 的验收门禁。进程级强扩展换的是分区数, 不是算法, 因此代数结果必须不随
# 进程数变化 —— 迭代数逐档相同是硬门禁 (不同即 Krylov 轨迹已分叉), 残差只要求
# 同量级: 归约次序随分区变, 末位浮点差异是正常的, 不是错误。
MPI_RESIDUAL_RELATIVE_TOL = 1.0e-3

# (d) 在尚未注册 case 时的占位原因, 与 (c) 同理。
PANEL_D_REASON = (
    "(d) 问的是同一个 matrix-free 算子在进程级 (MPI) 上的强扩展: 规模固定在 "
    "n=64, 只加进程数; 验收门禁是迭代数逐档不变。占位说明 cases.toml 里 "
    "panel d 当前没有 case。"
)


class CollectError(RuntimeError):
    """产物缺失或未通过门禁."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CollectError(
            f"产物缺失 -> {path}\n  先用 run.py 跑出对应 case, 再执行 collect。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _check_contract_drift(failures: list[str]) -> None:
    """核对本模块复述的阈值是否仍与 ``contract.py`` 一致.

    参数:
        failures: 门禁失败信息的累加列表, 就地追加.
    """
    try:
        import sys
        sys.path.insert(0, str(config.REPOSITORY_ROOT))
        from tools.matrix_free_evidence import contract
    except Exception as error:  # noqa: BLE001 - 契约不可导入时降级为提示
        failures.append(f"contract.py 无法导入, 阈值未核对: {error!r}")
        return
    pairs = (
        ("MINIMUM_FINAL_L2_ORDER", MINIMUM_FINAL_L2_ORDER),
        ("EA_FA_SOLUTION_RELATIVE_TOL", EA_FA_SOLUTION_RELATIVE_TOL),
    )
    for name, local in pairs:
        upstream = getattr(contract, name, None)
        if upstream != local:
            failures.append(
                f"阈值漂移: contract.{name} = {upstream!r}, collect.py 复述为 {local!r}"
            )


def _relative_difference(a: float, b: float) -> float | None:
    """两个正数的相对差, 以较大者为分母.

    这是核对 FA 与 EA 两条独立误差链吻合程度的主指标。全文其余门禁 (解相对差、
    残差、黄金参考) 都以相对差计, 此处沿用同一单位, 读者不必在 "位数" 与
    "相对差" 之间换算。

    参数:
        a: 第一个数, 需为有限正数.
        b: 第二个数, 需为有限正数.

    返回:
        relative: 相对差; 完全相等时为 ``0.0``; 输入不合法时为 ``None``.
    """
    if a == b:
        return 0.0
    if a <= 0 or b <= 0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(a - b) / max(abs(a), abs(b))


def _matching_significant_digits(relative: float | None) -> int:
    """把相对差换算成吻合的有效数字位数.

    只为快照留档 —— 报告表格已改用相对差 (见 ``_relative_difference``), 但外部
    引用 (china-postdoc 申请书正文) 仍以 "吻合到 N 位有效数字" 陈述该结论, 故
    该派生量继续写进 ``fig2_data.json``。

    注意口径: ``floor(-log10(relative))`` 不等价于 "打印出来的前 N 位数字逐位
    相同" —— 跨十进制档口时两者会分叉, 例如 ``0.9999999999995`` 与
    ``1.0000000000005`` 相对差 ``1e-12`` 记 12 位, 首位却不同。它衡量的是相对
    差的量级, 引用时按此陈述。

    参数:
        relative: ``_relative_difference`` 的返回值.

    返回:
        digits: 吻合的有效数字位数; 完全相等时返回 ``17``; 输入为 ``None`` 时
            返回 ``0``.
    """
    if relative is None:
        return 0
    if relative == 0.0:
        return 17
    return max(0, int(math.floor(-math.log10(relative))))


def _collect_fa_chain(case: config.Case, failures: list[str]) -> dict[str, Any]:
    """提取一条显式组装的误差链.

    参数:
        case: ``role`` 为 ``fa-chain-2d`` 或 ``fa-chain-3d`` 的 case.
        failures: 门禁失败信息的累加列表, 就地追加.

    返回:
        chain: 含网格尺寸、相对 L2 误差、逐段观测阶与求解器信息的字典.
    """
    payload = _load(case.artifact_path)
    levels = payload["levels"]

    if payload.get("passed") is not True:
        failures.append(f"{case.id}: 产物 passed != true")
    if len(levels) != EXPECTED_REFINEMENT_LEVELS:
        failures.append(
            f"{case.id}: 档数为 {len(levels)}, 期望 {EXPECTED_REFINEMENT_LEVELS} "
            "(--levels 5 才够看出观测阶朝理论阶上升是趋势而非巧合)"
        )
    if payload.get("base_subdivisions") != EXPECTED_BASE_SUBDIVISIONS:
        failures.append(
            f"{case.id}: base_subdivisions = {payload.get('base_subdivisions')!r}, "
            f"期望 {EXPECTED_BASE_SUBDIVISIONS} (--base 4 必须显式传, 缺省 2D 从 n=8 起步)"
        )
    final_order = payload.get("final_l2_order")
    if final_order is None or final_order < MINIMUM_FINAL_L2_ORDER:
        failures.append(
            f"{case.id}: 末段观测阶 {final_order!r} 未过门禁 {MINIMUM_FINAL_L2_ORDER}"
        )

    return {
        "label": f"{payload['dimension']}D {payload['mesh_type']} "
                 f"{payload['model']} p{payload['space_degree']}",
        "dimension": payload["dimension"],
        "mesh_type": payload["mesh_type"],
        "mesh_label": payload.get("mesh_label"),
        "model": payload["model"],
        # 制造解类名。表 a-1 的表头要指名道姓写出"误差是相对哪个精确解量的",
        # 只写 model 别名 (sinusoidal / divfree-poly) 读者查不到定义。
        "problem": payload.get("problem"),
        "material_hypothesis": payload.get("material_hypothesis"),
        "degree": payload["space_degree"],
        "solver": payload["solver"],
        "solver_options": payload.get("solver_options", {}),
        "assembly_method": payload.get("assembly_method"),
        # 档位参数入快照后, 表头就不必再把 "五档" 与 "--base 4 --levels 5" 写死。
        "base_subdivisions": payload.get("base_subdivisions"),
        "refinement_levels": payload.get("refinement_levels"),
        "theoretical_order": payload.get("theoretical_order"),
        # 代数误差的量级。表 a-1 要说明"直接解使代数误差可忽略, 测的是纯离散误差",
        # 这个断言需要一个可核验的数, 而不是一句形容。
        "max_residual": payload.get("max_residual"),
        "residual_tolerance": payload.get("residual_tolerance"),
        "subdivisions": [lv["subdivisions"] for lv in levels],
        "mesh_size": [lv["mesh_size"] for lv in levels],
        "dofs": [lv["dofs"] for lv in levels],
        "l2_relative": [lv["l2_relative"] for lv in levels],
        "l2_order": [lv["l2_order"] for lv in levels],
        "final_l2_order": final_order,
        "seconds": [lv["seconds"] for lv in levels],
    }


def _collect_ea_rings(case: config.Case, failures: list[str]) -> dict[str, Any]:
    """提取 stage-1 冻结链里 matrix-free 的三档误差与 EA/FA 解相对差.

    参数:
        case: ``role`` 为 ``ea-rings`` 的 case.
        failures: 门禁失败信息的累加列表, 就地追加.

    返回:
        rings: 以 ``"2"``/``"3"`` 为键的字典, 每项含三档相对 L2 误差、
            观测阶、``coarse`` 档的 EA/FA 解相对差、档位与显式参考核对.
    """
    payload = _load(case.artifact_path)
    if payload.get("passed") is not True:
        failures.append(f"{case.id}: stage-1 汇总 passed != true")

    # 显式参考核对的验收阈值来自 contract, 产物只存核对结果、不存阈值。
    try:
        import sys
        sys.path.insert(0, str(config.REPOSITORY_ROOT))
        from tools.matrix_free_evidence import contract
        explicit_tol = contract.EXPLICIT_SOLUTION_RELATIVE_TOL
    except Exception:  # noqa: BLE001 - 契约不可导入时降级为 None, 渲染侧跳过引用
        explicit_tol = None

    rings: dict[str, Any] = {}
    for dimension, block in payload["dimensions"].items():
        comparison = block["comparison"]
        if block.get("passed") is not True:
            failures.append(f"{case.id}: {dimension}D 分块 passed != true")

        # EA/FA 逐档解相对差: coarse 单值字段向后兼容 (旧消费方), 逐档列表
        # 按 EA 档位顺序供表 a-2' 渲染; 门禁逐档判, 任一档超阈值即失败。
        differences = comparison.get("ea_fa_solution_relative_differences") or {}
        gap = comparison["coarse_solution_ea_fa_relative_difference"]
        gaps = [
            float(differences[role]) for role in ("coarse", "medium", "fine")
            if role in differences
        ]
        for role, value in differences.items():
            if value >= EA_FA_SOLUTION_RELATIVE_TOL:
                failures.append(
                    f"{case.id}: {dimension}D 的 {role} 档 EA/FA 解相对差 "
                    f"{value:.6g} 未低于 {EA_FA_SOLUTION_RELATIVE_TOL}"
                )

        errors = comparison["relative_l2_errors"]
        # 键形如 ea-coarse-1rank, 顺序即 coarse/medium/fine, 不重排。
        # CG 的相对残差容差。表 a-1 要说明"收敛阶的举证为什么不交给 matrix-free",
        # 依据是这个容差会给误差垫地板 —— 那句话必须引 EA 侧的 rtol, 不能拿 FA
        # 产物里同为 1e-10 的 residual_tolerance(那是直接解残差的门禁阈值)顶替。
        first_case = next(iter(block["cases"].values()), {})
        # 档位 (每轴剖分数) 按 EA 三档的顺序取; 显式参考核对取全部单 rank 算例
        # (EA 三档 + FA coarse) 的最大相对差, 证明 CG 解已站在 spsolve 直接解上。
        # 并行算例 (mpi_size != 1) 不构造显式参考, 不参与核对。
        refinements: list[int] = []
        explicit_errors: list[float] = []
        for name in errors:
            resolution = block["cases"][name].get("parameters", {}).get("resolution")
            if resolution:
                refinements.append(int(resolution[0]))
        for name, run in block["cases"].items():
            if run.get("mpi_size", 1) != 1:
                continue
            reference = run.get("explicit_solution_reference") or {}
            if reference.get("relative_error") is not None:
                explicit_errors.append(float(reference["relative_error"]))
        explicit_max = max(explicit_errors) if explicit_errors else None
        # 门禁: CG 解与 spsolve 直接解的偏差必须落在 EXPLICIT_SOLUTION_RELATIVE_TOL
        # 内 —— 表 a-2 "CG 解已站在直接解上"的论断靠它兜底, 阈值来自 contract。
        if not explicit_errors:
            failures.append(
                f"{case.id}: {dimension}D 未找到单 rank 算例的 "
                "explicit_solution_reference, 无法核对 CG 解与 spsolve 直接解"
            )
        elif explicit_tol is not None and explicit_max >= explicit_tol:
            failures.append(
                f"{case.id}: {dimension}D 的 CG/spsolve 直接解相对差最大值 "
                f"{explicit_max:.6g} 未低于 "
                f"EXPLICIT_SOLUTION_RELATIVE_TOL {explicit_tol:g}"
            )
        rings[dimension] = {
            "cg_rtol": first_case.get("parameters", {}).get("rtol"),
            "roles": list(errors),
            "refinements": refinements,
            "l2_relative": list(errors.values()),
            "observed_l2_orders": comparison["observed_relative_l2_orders"],
            "gated_l2_order": comparison["gated_relative_l2_order"],
            "solution_relative_difference_vs_fa": gap,
            "solution_relative_differences_vs_fa": gaps,
            "max_explicit_solution_relative_error": explicit_max,
            "explicit_solution_relative_tol": explicit_tol,
        }
    return rings


def _cross_check_chains(fa: dict[str, Any], rings: dict[str, Any],
                        notes: list[str]) -> list[dict[str, Any]]:
    """核对 FA 与 EA 两条独立误差链在重合档上的相对差.

    两处出自不同脚本、不同求解器 (直接解 vs CG rtol=1e-10), 画的却是同一个
    离散; 若某一档相对差骤增, 说明两条链已经不是同一个离散, 必须查。

    比的是误差标量 (两条链在同一档的相对 L2 误差值), 与 ``_collect_ea_rings``
    记录的解向量相对差是两个对象, 不可互换引用: 前者说 "落在同一条收敛曲线
    上", 后者说 "解本身一样"。

    参数:
        fa: ``{"2d": chain, "3d": chain}`` 形式的显式组装误差链.
        rings: ``_collect_ea_rings`` 的返回值.
        notes: 提示信息的累加列表, 就地追加.

    返回:
        report: 每个维度一条记录, 含逐档相对差、最差档的相对差与派生的吻合
            位数; 缺档处相对差为 ``None``.
    """
    report: list[dict[str, Any]] = []
    for dimension, key in (("2", "2d"), ("3", "3d")):
        chain = fa.get(key)
        ring = rings.get(dimension)
        if chain is None or ring is None:
            continue
        by_subdivision = dict(zip(chain["subdivisions"], chain["l2_relative"]))
        relatives: list[float | None] = []
        # stage-1 的三档固定为 n = 8/16/32 (contract.REFINEMENTS)。
        for subdivision, ea_error in zip((8, 16, 32), ring["l2_relative"]):
            fa_error = by_subdivision.get(subdivision)
            if fa_error is None:
                notes.append(f"{dimension}D: FA 链缺 n={subdivision} 档, 无法核对")
                relatives.append(None)
                continue
            relatives.append(_relative_difference(fa_error, ea_error))
        measured = [value for value in relatives if value is not None]
        worst = max(measured) if measured else None
        record = {
            "dimension": int(dimension),
            "subdivisions": [8, 16, 32],
            "error_relative_differences": relatives,
            "maximum": worst,
            # 派生量, 供外部按 "吻合 N 位有效数字" 引用; 报告表格不再使用。
            "matching_significant_digits": [
                _matching_significant_digits(value) for value in relatives
            ],
            "minimum_matching_significant_digits": (
                _matching_significant_digits(worst) if worst is not None else 0
            ),
        }
        if worst is not None and worst > EA_FA_ERROR_CHAIN_RELATIVE_TOL:
            notes.append(
                f"{dimension}D: 两条误差链最差档相对差 {worst:.3e}, 超出阈值 "
                f"{EA_FA_ERROR_CHAIN_RELATIVE_TOL:g} (常态在 1e-11 量级), "
                "需确认是否仍是同一个离散"
            )
        report.append(record)
    return report


def _stage_series(payload: dict[str, Any], label: str) -> list[int]:
    """按 ``STAGE_ORDER`` 取出一次运行的分阶段高水位.

    参数:
        payload: 单次 peak-RSS 运行的产物字典.
        label: 出错时用于定位的算例标识, 形如 ``ea-n64``.

    返回:
        list[int]: 与 ``STAGE_ORDER`` 同序的累积高水位, 单位字节.

    异常:
        CollectError: 产物缺少 ``STAGE_ORDER`` 中的任一阶段, 说明它由旧版插桩
            产生, 需要重跑该算例.
    """
    stages = payload.get("peak_rss_stage_bytes") or {}
    missing = [stage for stage in STAGE_ORDER if stage not in stages]
    if missing:
        raise CollectError(
            f"{label}: 产物缺少阶段 {missing}, 现有阶段 {sorted(stages)}; "
            "这是旧版插桩留下的产物 (体力组装与 Dirichlet 处理曾合并在 bc 一个阶段), "
            "需用当前版 benchmark_cpu_ea.py 重跑八个 peak-RSS 算例"
        )
    return [stages[stage] for stage in STAGE_ORDER]


def _decompose_peaks(level_panel: dict[str, Any],
                     stage_order: list[str]) -> dict[str, Any]:
    """把逐档峰值 RSS 拆成"必需存储"与"瞬态"两部分.

    峰值本身回答不了"这些内存是非占不可, 还是组装路上的临时开销" —— 而 (b)
    格的因果论断 (matrix-free 省的不是存得少, 是从不物化全局 COO/CSR) 恰恰要
    靠这个拆分才立得住, 故在采集侧算出来, 消费方不必各自复算。

    下界取 ``baseline + mesh + stored_operator``: 解释器与导入的常量开销、网格
    与函数空间、以及该层级必须长期持有的算子数据 (FA 是全局 CSR, EA 是逐单元
    ``K_e`` 缓存)。峰值减去下界即瞬态。

    ⚠️ 下界是**该实现路径**的下界, 不是问题本身的下界: 换算子层级 (如只存积分
    点因子的 PA) 会改变 ``stored_operator`` 这一项。

    参数:
        level_panel: 单个算子层级的数据, 需含 ``peak_rss_bytes``、
            ``stored_operator_bytes`` 与 ``stage_bytes``.
        stage_order: 阶段名顺序, 用于定位 ``baseline`` 与 ``mesh``.

    返回:
        decomposition: 逐档的 ``mesh_bytes``、``floor_bytes``、
            ``transient_bytes``、``transient_fraction`` 与 ``peak_over_floor``.
    """
    baseline_index = stage_order.index("baseline")
    mesh_index = stage_order.index("mesh")

    mesh_bytes, floor_bytes, transient_bytes = [], [], []
    transient_fraction, peak_over_floor = [], []
    for peak, stored, stages in zip(level_panel["peak_rss_bytes"],
                                    level_panel["stored_operator_bytes"],
                                    level_panel["stage_bytes"]):
        # 逐档取该次运行自己的 baseline 高水位, 而非八次运行的最小值, 这样四段
        # 之和严格等于该档峰值, 拆分不会因基线抖动而对不平。
        baseline = stages[baseline_index]
        mesh = stages[mesh_index] - baseline
        floor = baseline + mesh + stored
        mesh_bytes.append(mesh)
        floor_bytes.append(floor)
        transient_bytes.append(peak - floor)
        transient_fraction.append((peak - floor) / peak if peak else None)
        peak_over_floor.append(peak / floor if floor else None)
    return {
        "mesh_bytes": mesh_bytes,
        "floor_bytes": floor_bytes,
        "transient_bytes": transient_bytes,
        "transient_fraction": transient_fraction,
        "peak_over_floor": peak_over_floor,
    }


def _collect_peak_rss(cases: tuple[config.Case, ...],
                      failures: list[str]) -> dict[str, Any]:
    """提取八个单进程峰值内存数据点并按算子层级归拢.

    参数:
        cases: ``role`` 为 ``peak-rss`` 的全部 case.
        failures: 门禁失败信息的累加列表, 就地追加.

    返回:
        panel: 含四档自由度、两个层级的峰值/稳态/分阶段字节数与基线的字典.

    异常:
        CollectError: 两个层级的档位集合不一致, 无法逐档配对.
    """
    by_level: dict[str, dict[int, dict[str, Any]]] = {"fa": {}, "ea": {}}
    for case in cases:
        payload = _load(case.artifact_path)
        level = case.extra["level"]
        resolution = case.extra["resolution"]

        # 逐项核对产物确实是这个 case 声明的那次运行, 防止旧产物被当成新的读进来。
        expectations = (
            ("operator_level", level),
            ("resolution", resolution),
            ("assembly_method", "fast"),
            ("mesh_type", "tet"),
            ("dimension", 3),
            ("degree", 1),
            ("mode", "serial-peak-rss"),
        )
        for field, expected in expectations:
            if payload.get(field) != expected:
                failures.append(
                    f"{case.id}: 产物 {field} = {payload.get(field)!r}, 期望 {expected!r}"
                )
        if payload.get("solved") is not True:
            failures.append(f"{case.id}: solved != true")
        if payload.get("cg_converged") is not True:
            failures.append(f"{case.id}: cg_converged != true")

        by_level[level][resolution] = payload

    resolutions = sorted(by_level["fa"])
    if resolutions != sorted(by_level["ea"]):
        raise CollectError(
            f"FA 与 EA 的档位集合不一致: fa={resolutions}, ea={sorted(by_level['ea'])}"
        )

    def series(level: str, field: str) -> list[Any]:
        return [by_level[level][n][field] for n in resolutions]

    # 基线是解释器与 FEALPy 导入的常量开销, 八次运行应当几乎相同;
    # 抖动大意味着测量环境不干净, 峰值比也就不可信。
    baselines = (series("fa", "peak_rss_baseline_bytes")
                 + series("ea", "peak_rss_baseline_bytes"))
    baseline_spread = max(baselines) - min(baselines)

    panel: dict[str, Any] = {
        "resolution": resolutions,
        "dofs": series("fa", "dofs"),
        "cells": series("fa", "cells"),
        "baseline_bytes": min(baselines),
        "baseline_spread_bytes": baseline_spread,
        "stage_order": STAGE_ORDER,
    }
    for level in ("fa", "ea"):
        panel[level] = {
            "peak_rss_bytes": series(level, "peak_rss_bytes"),
            "peak_rss_above_baseline_bytes":
                series(level, "peak_rss_above_baseline_bytes"),
            "stored_operator_bytes": series(level, "stored_operator_bytes"),
            "stage_bytes": [
                _stage_series(by_level[level][n], f"{level}-n{n}")
                for n in resolutions
            ],
            "construction_seconds": series(level, "construction_seconds"),
            "solve_seconds": series(level, "solve_seconds"),
            "cg_iterations": series(level, "cg_iterations"),
            "true_relative_residual": series(level, "true_relative_residual"),
        }
        panel[level].update(_decompose_peaks(panel[level], panel["stage_order"]))

    # 两个层级各自的"必需存储"下界之比。它与峰值比是相反方向的量: 峰值比 EA 占
    # 优 (躲开组装瞬态), 下界比却是 FA 占优 (CSR 比逐单元 K_e 缓存小), 引用时
    # 不可只取一侧。
    panel["floor_ratio_ea_over_fa"] = [
        ea / fa if fa else None
        for ea, fa in zip(panel["ea"]["floor_bytes"], panel["fa"]["floor_bytes"])
    ]

    # 逐档迭代数必须一致: 两个层级是同一个离散, 迭代数不同就说明算子不同。
    if panel["fa"]["cg_iterations"] != panel["ea"]["cg_iterations"]:
        failures.append(
            f"FA 与 EA 的 CG 迭代数逐档不一致: "
            f"{panel['fa']['cg_iterations']} vs {panel['ea']['cg_iterations']}"
        )
    return panel


def _collect_device_speedup(cases: tuple[config.Case, ...],
                            failures: list[str]) -> dict[str, Any]:
    """提取四档 CPU / 单卡 GPU 耗时对照, 并核对两侧算的是同一个问题.

    加速比只有在"两边解同一个离散问题"成立时才有意义, 所以本函数的门禁不看耗时,
    只看等价性: 解的相对差、CG 迭代数是否逐档相同、两侧是否都真的收敛。耗时快慢
    不设门禁 —— 慢也是结论 (n=8 那档 GPU 就更慢), 不该被判失败。

    参数:
        cases: ``role`` 为 ``device-speedup`` 的全部 case.
        failures: 门禁失败信息的累加列表, 就地追加.

    返回:
        panel: 含四档自由度、两侧耗时、加速比与等价性证据的字典.
    """
    by_resolution: dict[int, dict[str, Any]] = {}
    for case in cases:
        payload = _load(case.artifact_path)
        resolution = case.extra["resolution"]

        # 与 (b) 同样逐项核对产物身份, 防止旧产物或改了参数的产物被当成本 case 的。
        expectations = (
            ("mode", "device-speedup-ea"),
            ("resolution", resolution),
            ("operator_level", case.extra["level"]),
            ("assembly_method", "fast"),
            ("mesh_type", "tet"),
            ("dimension", 3),
            ("degree", 1),
            ("backend", "pytorch"),
        )
        for field, expected in expectations:
            if payload.get(field) != expected:
                failures.append(
                    f"{case.id}: 产物 {field} = {payload.get(field)!r}, 期望 {expected!r}"
                )

        gap = payload.get("solution_relative_gap")
        if not isinstance(gap, (int, float)) or gap > DEVICE_SOLUTION_RELATIVE_TOL:
            failures.append(
                f"{case.id}: CPU/GPU 解相对差 {gap!r} 未通过 "
                f"<= {DEVICE_SOLUTION_RELATIVE_TOL:.0e}"
            )
        if payload.get("cg_iterations_match") is not True:
            failures.append(f"{case.id}: 两侧 CG 迭代数不一致, Krylov 轨迹已分叉")
        for device in ("cpu", "cuda"):
            record = payload.get("devices", {}).get(device, {})
            if record.get("cg_converged") is not True:
                failures.append(f"{case.id}: {device} 侧 CG 未收敛")
        # 产物自带一次门禁判定。它与本模块复述的阈值应当同向; 不同向说明脚本的
        # --gap-gate 被调松过, 这时以本模块为准并显式报出来。
        if payload.get("gate_passed") is not True:
            failures.append(f"{case.id}: 产物自判 gate_passed != true")

        by_resolution[resolution] = payload

    resolutions = sorted(by_resolution)

    def series(field: str) -> list[Any]:
        return [by_resolution[n][field] for n in resolutions]

    def device_series(device: str, field: str) -> list[Any]:
        return [by_resolution[n]["devices"][device].get(field) for n in resolutions]

    # 计时口径必须四档一致, 否则中位数之间不可比。
    repeats = {by_resolution[n].get("repeats") for n in resolutions}
    warmups = {by_resolution[n].get("warmup") for n in resolutions}
    if len(repeats) != 1 or len(warmups) != 1:
        failures.append(
            f"(c) 四档的计时口径不一致: warmup={sorted(warmups)}, repeats={sorted(repeats)}"
        )
    gpu_names = {name for name in device_series("cuda", "gpu_name") if name}
    if len(gpu_names) > 1:
        failures.append(f"(c) 四档不是同一块卡: {sorted(gpu_names)}")

    panel: dict[str, Any] = {
        "resolution": resolutions,
        "dofs": series("dofs"),
        "cells": series("cells"),
        "warmup": by_resolution[resolutions[0]].get("warmup"),
        "repeats": by_resolution[resolutions[0]].get("repeats"),
        "gpu_name": sorted(gpu_names)[0] if gpu_names else None,
        "solution_relative_gap": series("solution_relative_gap"),
        "cg_iterations_match": series("cg_iterations_match"),
        "speedup_solve": series("speedup_solve"),
        "speedup_build": series("speedup_build"),
    }
    for device in ("cpu", "cuda"):
        panel[device] = {
            "solve_seconds": device_series(device, "solve_seconds"),
            "build_seconds": device_series(device, "build_seconds"),
            "solve_samples": device_series(device, "solve_samples"),
            "build_samples": device_series(device, "build_samples"),
            "cg_iterations": device_series(device, "cg_iterations"),
            "true_relative_residual": device_series(device, "true_relative_residual"),
        }
    # 显存高水位与卡容量只有 CUDA 侧有, 但它是 (c) 的能力边界证据 —— 不带进快照,
    # 文档就只能手抄, 手抄的数迟早与产物脱节。CPU 线程数决定"加速比是对几个核说
    # 的", 不记下来这个对照就没法复核; 早于该字段的产物取到 None, 报告端跳过。
    panel["cuda"]["gpu_peak_allocated_bytes"] = device_series(
        "cuda", "gpu_peak_allocated_bytes"
    )
    panel["cuda"]["gpu_memory_total_bytes"] = device_series(
        "cuda", "gpu_memory_total_bytes"
    )
    panel["cpu"]["torch_threads"] = device_series("cpu", "torch_threads")
    return panel


def _collect_mpi_strong(cases: tuple[config.Case, ...],
                        failures: list[str]) -> dict[str, Any]:
    """把各 rank 档的 MPI 强扩展产物收成 (d) 一格.

    与 (c) 的差别不只是横轴换成进程数: (c) 比的是两条实现路径 (CPU vs GPU) 解同一
    个问题, (d) 比的是同一条路径切成不同份数, 因此正确性判据也不同 —— (c) 判两侧
    解的相对差, (d) 判迭代数是否逐档不变。

    参数:
        cases: role 为 ``mpi-strong`` 的全部 case.
        failures: 门禁失败信息的收集列表, 原地追加.

    返回:
        panel: (d) 一格的快照内容.
    """
    by_ranks: dict[int, dict[str, Any]] = {}
    for case in cases:
        payload = _load(case.artifact_path)
        ranks = int(case.extra["ranks"])
        if payload.get("ranks") != ranks:
            failures.append(
                f"{case.id}: 产物 ranks = {payload.get('ranks')!r}, 期望 {ranks}"
            )
        for field, expected in (("mode", "mpi-ea-strong"),
                                ("model", "polynomial"),
                                ("mesh_type", "tet"),
                                ("assembly_method", "fast")):
            if payload.get(field) != expected:
                failures.append(
                    f"{case.id}: 产物 {field} = {payload.get(field)!r}, "
                    f"期望 {expected!r}"
                )
        if payload.get("solver", {}).get("converged") is not True:
            failures.append(f"{case.id}: CG 未收敛")
        by_ranks[ranks] = payload

    if not by_ranks:
        return {"ranks": [], "dofs": None}

    ordered = sorted(by_ranks)
    if 1 not in by_ranks:
        failures.append("(d) 缺 1 进程档, 强扩展没有分母")

    def series(field: str) -> list[Any]:
        return [by_ranks[p][field] for p in ordered]

    def solver_series(field: str) -> list[Any]:
        return [by_ranks[p].get("solver", {}).get(field) for p in ordered]

    def profile_series(field: str) -> list[Any]:
        return [by_ranks[p].get("ea_overlap_matvec_profile_seconds_max_rank", {})
                .get(field) for p in ordered]

    # 强扩展固定总规模, 自由度必须逐档相同 —— 不同就不是强扩展而是弱扩展。
    dof_set = set(series("global_vector_dofs"))
    if len(dof_set) != 1:
        failures.append(f"(d) 各档自由度不同, 不构成强扩展: {sorted(dof_set)}")

    # 迭代数逐档不变是本格的正确性门禁: 分区只改数据分布, 不改代数。
    iterations = solver_series("iterations")
    if len(set(iterations)) != 1:
        failures.append(
            f"(d) 迭代数随进程数变化 {iterations}, 分区改变了代数结果"
        )

    residuals = [r for r in solver_series("true_relative_residual")
                 if isinstance(r, (int, float))]
    if len(residuals) == len(ordered) and residuals:
        floor = max(abs(min(residuals)), 1e-300)
        spread = (max(residuals) - min(residuals)) / floor
        if spread > MPI_RESIDUAL_RELATIVE_TOL:
            failures.append(
                f"(d) 各档残差相对离散 {spread:.3e} 超出 "
                f"{MPI_RESIDUAL_RELATIVE_TOL:.0e}"
            )

    # 计时口径与 (c) 同理必须逐档一致, 否则各档时间不可比。早于该字段的产物取到
    # None; 全为 None 时集合仍是单元素, 不判失败, 由报告端标注口径未知。
    warmups = {by_ranks[p].get("warmup") for p in ordered}
    repeats = {by_ranks[p].get("repeats") for p in ordered}
    if len(warmups) != 1 or len(repeats) != 1:
        failures.append(
            f"(d) 各档计时口径不一致: warmup={sorted(map(str, warmups))}, "
            f"repeats={sorted(map(str, repeats))}"
        )

    cg = series("ea_cg_seconds_max_rank")
    baseline = by_ranks[1]["ea_cg_seconds_max_rank"] if 1 in by_ranks else None
    speedup = [baseline / t for t in cg] if baseline else None
    efficiency = [s / p for s, p in zip(speedup, ordered)] if speedup else None

    return {
        "ranks": ordered,
        "dofs": sorted(dof_set)[0],
        "resolution": by_ranks[ordered[0]]["resolution"][0],
        "warmup": by_ranks[ordered[0]].get("warmup"),
        "repeats": by_ranks[ordered[0]].get("repeats"),
        "cg_seconds": cg,
        "speedup": speedup,
        "efficiency": efficiency,
        "construction_seconds": series("ea_construction_seconds_max_rank"),
        "pipeline_seconds": series("ea_pipeline_seconds_max_rank"),
        "matvec_seconds": series("ea_system_matvec_seconds_max_rank"),
        "local_kernel_seconds": profile_series("local_kernel_seconds"),
        "input_sync_seconds": profile_series("input_sync_seconds"),
        "output_sync_seconds": profile_series("output_sync_seconds"),
        "cg_iterations": iterations,
        "true_relative_residual": solver_series("true_relative_residual"),
    }


def build(cases: tuple[config.Case, ...], figure: dict[str, Any]) -> dict[str, Any]:
    """构造快照内容.

    参数:
        cases: 全部已注册 case.
        figure: ``cases.toml`` 的 ``[figure]`` 段.

    返回:
        snapshot: 待写入 ``figure_data/fig2_data.json`` 的字典.
    """
    failures: list[str] = []
    notes: list[str] = []
    _check_contract_drift(failures)

    # by_role 只容得下每个 role 一个 case, 因此多档的 role 必须先摘出去,
    # 否则同 role 的后一个会静默覆盖前一个。
    multi_case_roles = {"peak-rss", "device-speedup", "mpi-strong"}
    by_role = {c.role: c for c in cases if c.role not in multi_case_roles}
    peak_cases = tuple(c for c in cases if c.role == "peak-rss")
    device_cases = tuple(c for c in cases if c.role == "device-speedup")
    mpi_cases = tuple(c for c in cases if c.role == "mpi-strong")

    fa = {
        "2d": _collect_fa_chain(by_role["fa-chain-2d"], failures),
        "3d": _collect_fa_chain(by_role["fa-chain-3d"], failures),
    }
    rings = _collect_ea_rings(by_role["ea-rings"], failures)
    cross = _cross_check_chains(fa, rings, notes)
    panel_b = _collect_peak_rss(peak_cases, failures)

    # 没有注册 case 时退回占位: 这一格的存在性由 cases.toml 决定, 不靠人工改常量。
    if device_cases:
        panel_c_body = _collect_device_speedup(device_cases, failures)
        # (b) 与设备对照(渲染进图面的 (d))共用一条自由度横轴, 两格才能连起来读成
        # "能算多大 / 算得多快"。⚠️ 这里的 panel_c 是**数据组**键名, 它进的是图面
        # 的 (d); 图面的 (c) 是 panels.d 的 MPI 强扩展, 横轴是进程数, 不在这条轴上。
        # 但 (c) 允许比 (b) 短: 单卡显存未必跟得上 (b) 在主机内存里能到的最细档。
        # 判据因此是"(c) 的档位构成 (b) 的前缀"—— 同起点、同档位、只是早一步停,
        # 两格仍在同一条轴上; 若两者错位或交错, 那才是真的不同轴, 必须报出来。
        # 放宽到"随便短一截都行"会漏掉真正的错位, 所以用前缀而不是子集。
        prefix = panel_b["dofs"][: len(panel_c_body["dofs"])]
        if panel_c_body["dofs"] != prefix:
            failures.append(
                f"(b) 与 (d) 的自由度序列不同轴: {panel_b['dofs']} vs {panel_c_body['dofs']}"
            )
        elif len(panel_c_body["dofs"]) < len(panel_b["dofs"]):
            notes.append(
                f"(d) 比 (b) 少 {len(panel_b['dofs']) - len(panel_c_body['dofs'])} 档: "
                f"(b) 最细到 {panel_b['dofs'][-1]:,} 自由度, (d) 止于 "
                f"{panel_c_body['dofs'][-1]:,} —— 更细的档在单卡显存里装不下, "
                f"见 cases.toml 中撤销 c-dev-n80 的说明。"
            )
        panel_c = {
            "status": "measured",
            "question": "能不能更快",
            "gate": {"solution_relative_tol": DEVICE_SOLUTION_RELATIVE_TOL},
            **panel_c_body,
        }
    else:
        panel_c = {
            "status": "placeholder",
            "question": "能不能更快",
            "reason": PANEL_C_REASON,
        }

    # (d) 与 (c) 同样由 cases.toml 决定存在性, 没注册就退回占位。
    if mpi_cases:
        panel_d = {
            "status": "measured",
            "question": "多进程能不能更快",
            "parallel_level": "process",
            "gate": {"iterations_identical": True,
                     "residual_relative_tol": MPI_RESIDUAL_RELATIVE_TOL},
            **_collect_mpi_strong(mpi_cases, failures),
        }
    else:
        panel_d = {
            "status": "placeholder",
            "question": "多进程能不能更快",
            "reason": PANEL_D_REASON,
        }

    record = provenance.collect()
    return {
        "schema_version": SCHEMA_VERSION,
        "figure": figure,
        "provenance": record,
        "reproducible": provenance.reproducible(record),
        "panels": {
            "a": {
                "status": "measured",
                "question": "对不对",
                "gate": {"minimum_final_l2_order": MINIMUM_FINAL_L2_ORDER,
                         "ea_fa_solution_relative_tol": EA_FA_SOLUTION_RELATIVE_TOL,
                         "ea_fa_error_chain_relative_tol":
                             EA_FA_ERROR_CHAIN_RELATIVE_TOL},
                "fa": fa,
                "ea": rings,
                "chain_cross_check": cross,
            },
            "b": {
                "status": "measured",
                "question": "能算多大",
                "memory_total_bytes": record.get("memory_total_bytes"),
                **panel_b,
            },
            "c": panel_c,
            "d": panel_d,
        },
        # script 是产物的事实源 (run.py 以子进程调用它生成产物): 与产物路径、
        # 指纹一并入快照, 使表 a-1 的来源说明由 report.py 按此渲染, 不硬编码路径。
        "sources": [
            {"case_id": c.id, "role": c.role, "script": c.script,
             **provenance.file_digest(c.artifact_path)}
            for c in cases
        ],
        "gate_failures": failures,
        "notes": notes,
    }


def write(snapshot: dict[str, Any], path: Path | None = None) -> Path:
    """把快照写入磁盘.

    参数:
        snapshot: ``build()`` 的返回值.
        path: 目标路径, ``None`` 时用 ``figure_data/fig2_data.json``.

    返回:
        path: 实际写入的路径.
    """
    path = path or config.FIGURE_DATA_DIR / "fig2_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
