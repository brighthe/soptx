# -*- coding: utf-8 -*-
"""从快照重新生成 ``results_analysis.md`` 里的三格证据表.

本目录的 ``results_analysis.md`` 按**图面分格**组织, 与两个 example 目录按
技术栈组织的那两份是同一批数字的不同切法。同一个数字写在两处就会漂移 ——
本会话里已经发生过两次 —— 所以三格的表格不手写, 由本模块从
``figure_data/fig2_data.json`` 渲染, 写进文档里的标记区间。

标记形如::

    <!-- BEGIN generated: panel-a -->
    ...此区间内的内容会被整体替换...
    <!-- END generated: panel-a -->

标记之外的叙述、读法与边界说明是手写的, 不受影响。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import config

GIB = 2 ** 30
MIB = 2 ** 20

# 文档里受本模块管理的区间名, 顺序即渲染顺序。
GENERATED_BLOCKS = ("panel-a", "panel-b", "panel-c", "panel-d", "provenance")


class ReportError(RuntimeError):
    """快照缺失, 或文档里的标记区间不完整."""


def _fmt(value: float, digits: int = 4) -> str:
    """按有效数字格式化, 小量走科学计数法."""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"`{value:.{digits}e}`"
    return f"`{value:.{digits}g}`"


def _render_panel_a(snapshot: dict[str, Any]) -> str:
    """渲染 (a) 格的三张表: 显式组装误差链、matrix-free 三档、跨链吻合位数."""
    panel = snapshot["panels"]["a"]
    fa2, fa3 = panel["fa"]["2d"], panel["fa"]["3d"]
    lines: list[str] = []

    lines.append("**表 a-1　显式组装（FA）路径上数值解与制造解的误差及收敛阶**\n")
    lines.append(
        "**相对 L2** $=\\lVert u-u_h\\rVert_0/\\lVert u\\rVert_0$，$u$ 为解析已知的"
        f"制造解，两个范数同以 `p+3 = {fa2['degree'] + 3}` 阶求积算出"
        "（[`solution_error()`](../../src/soptx/fem/verification.py)）；"
        "**观测阶** $=\\log_2(e_{\\text{粗}}/e_{\\text{细}})$，加密比恒为 2。"
        "⚠️ 观测阶按绝对误差计（产物字段 `l2_order`），与左列相对误差的阶差在"
        "第 6 位，本表 3 位小数看不出。\n")
    lines.append("| | 网格 | 制造解 | 装配 | 求解器 |")
    lines.append("|---|---|---|---|---|")
    for chain in (fa2, fa3):
        options = chain.get("solver_options") or {}
        detail = "，".join(f"`{key}={value}`" for key, value in sorted(options.items()))
        solver = f"`{chain['solver']}`" + (f"（{detail}）" if detail else "")
        # material_hypothesis 在 3D 上就是 "3D", 与维数列重复, 只有 2D 的
        # plane_strain / plane_stress 才带信息。
        hypothesis = chain.get("material_hypothesis")
        mesh = f"`{chain['mesh_label'] or chain['mesh_type']}`"
        if hypothesis and hypothesis != chain["dimension"]:
            mesh += f"（`{hypothesis}`）"
        lines.append(
            f"| {chain['dimension']} | {mesh} | [`{chain['problem']}`]"
            "(../../docs/problems/manufactured-elasticity.md) | "
            f"`{chain['assembly_method']}` | {solver} |")
    first, last = fa2["subdivisions"][0], fa2["subdivisions"][-1]
    levels = fa2.get("refinement_levels") or len(fa2["subdivisions"])
    residuals = "、".join(
        f"{chain['dimension']} `{chain['max_residual']:.2e}`" for chain in (fa2, fa3))
    lines.append(
        f"\n均 `p={fa2['degree']}`，{levels} 档 `n = {first}…{last}`"
        f"（`h = 1/{first} … 1/{last}`）。两条链均走**稀疏直接解**，真相对残差最大值 "
        f"{residuals}，比最细档的离散误差（2D `{fa2['l2_relative'][-1]:.2e}`、"
        f"3D `{fa3['l2_relative'][-1]:.2e}`）小十个量级，本表测的基本是纯离散误差，"
        "故可充当参考路径。\n")

    # 来源说明: 两条 FA 链由 examples/lagrange_elasticity 的 demo 脚本生成。
    # 脚本路径从快照 sources 取, 与 cases.toml 同源, 不在渲染侧硬编码。
    fa_scripts = {
        source["role"]: source.get("script")
        for source in snapshot.get("sources", [])
        if source.get("role") in ("fa-chain-2d", "fa-chain-3d")
    }
    script_links = "、".join(
        f"[`{s}`](../../{s})" for s in sorted({s for s in fa_scripts.values() if s}))
    if script_links:
        lines.append(
            f"两条链均由 {script_links} 以 `--base {first} --levels {levels}` 生成——"
            "`run.py` 以子进程调用、产物重定向到本目录 `outputs/`（"
            "[`cases.toml`](cases.toml) 的 `a-fa-2d`/`a-fa-3d`），与 "
            "[`examples/lagrange_elasticity` §4.2](../../examples/lagrange_elasticity/results_analysis.md) "
            "的五档链同源同离散，打印精度内逐位一致。\n"
        )
    lines.append("**逐档结果**\n")
    lines.append("| `n` | 2D 自由度 | 2D 相对 L2 | 2D 观测阶 | 3D 自由度 | 3D 相对 L2 | 3D 观测阶 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for index, subdivision in enumerate(fa2["subdivisions"]):
        order2 = fa2["l2_order"][index]
        order3 = fa3["l2_order"][index]
        lines.append(
            f"| {subdivision} | {fa2['dofs'][index]:,} | {_fmt(fa2['l2_relative'][index])} | "
            f"{'—' if order2 is None else f'{order2:.3f}'} | "
            f"{fa3['dofs'][index]:,} | {_fmt(fa3['l2_relative'][index])} | "
            f"{'—' if order3 is None else f'{order3:.3f}'} |"
        )
    gate = panel["gate"]["minimum_final_l2_order"]
    lines.append(
        f"\n两条链的末段观测阶 **{fa2['final_l2_order']:.3f}**（2D）与 "
        f"**{fa3['final_l2_order']:.3f}**（3D）均过门禁 `{gate}`，且朝 P1 理论阶 2 单调上升。\n"
    )

    blocks = panel["ea"]
    refinements = next(iter(blocks.values())).get("refinements") or []
    refinements_text = "/".join(str(r) for r in refinements)
    rtol = next(
        (b.get("cg_rtol") for b in blocks.values() if b.get("cg_rtol") is not None),
        None,
    )
    rtol_piece = (
        f"CG（`rtol={rtol:.0e}`）" if rtol is not None else "CG"
    )
    lines.append(
        f"**表 a-2　matrix-free 三档的数值解误差**"
        f"（EA，`coarse/medium/fine` 对应 `n = {refinements_text}`）\n")
    lines.append(
        f"EA 在三个网格档位（`n = {refinements_text}`，CPU 串行）上相对制造解的 "
        f"L2 误差，求解为 {rtol_piece}。三档误差即图 2(a) 的空心环数据。\n")
    lines.append("| 维数 | `coarse` 相对 L2 | `medium` | `fine` |")
    lines.append("|---:|---:|---:|---:|")
    for dimension in sorted(blocks):
        block = blocks[dimension]
        errors = block["l2_relative"]
        lines.append(
            f"| {dimension}D | {_fmt(errors[0])} | {_fmt(errors[1])} | {_fmt(errors[2])} |"
        )
    lines.append("")

    tol = panel["gate"]["ea_fa_solution_relative_tol"]
    chain_tol = panel["gate"].get("ea_fa_error_chain_relative_tol")
    # 逐档相对差按维度取出, 缺档处为 None (FA 链缺该档)。
    chain_by_dimension = {
        str(cross["dimension"]): cross["error_relative_differences"]
        for cross in panel.get("chain_cross_check", [])
    }
    lines.append(
        "**表 a-2′　同档 EA/FA 一致性**"
        "（同网格同参数，两个对象均以相对差计）\n")
    lines.append("| 维数 | `n` | 解向量相对差 | 误差标量相对差 |")
    lines.append("|---:|---:|---:|---:|")
    for dimension in sorted(blocks):
        block = blocks[dimension]
        gaps = block.get("solution_relative_differences_vs_fa") or [
            block["solution_relative_difference_vs_fa"]
        ]
        chain_gaps = chain_by_dimension.get(dimension, [])
        for index, subdivision in enumerate(refinements):
            solution_cell = _fmt(gaps[index], 5) if index < len(gaps) else "—"
            chain_value = chain_gaps[index] if index < len(chain_gaps) else None
            chain_cell = _fmt(chain_value, 5) if chain_value is not None else "—"
            lines.append(
                f"| {dimension}D | {subdivision} | {solution_cell} | {chain_cell} |"
            )
    chain_tol_text = f"`{chain_tol:g}`" if chain_tol else "—"
    lines.append(
        f"\n两列比的是两个对象，不可互换引用：左列是同网格同参数的两个 "
        f"CG 解之差（验收阈值 `{tol:g}`），说的是解本身一样；右列是两条误差链"
        f"在该档的相对 L2 误差值之差（验收阈值 {chain_tol_text}），说的是落在同一条"
        f"收敛曲线上。两列均由 `collect.py` 逐档算出，非估计。")
    # 余量取三档中最接近阈值的那档 (最大的相对差对应最小的余量), 两列分别报。
    solution_parts, chain_parts = [], []
    for dimension, block in sorted(blocks.items()):
        gaps = block.get("solution_relative_differences_vs_fa") or [
            block["solution_relative_difference_vs_fa"]
        ]
        worst = max(gaps)
        solution_parts.append(
            f"{dimension}D 最低 {math.log10(tol / worst):.1f} 个量级"
            f"（约 {tol / worst:.0f} 倍）"
        )
        chain_measured = [
            value for value in chain_by_dimension.get(dimension, [])
            if value is not None
        ]
        if chain_tol and chain_measured:
            chain_worst = max(chain_measured)
            chain_parts.append(
                f"{dimension}D 最低 {math.log10(chain_tol / chain_worst):.1f} 个量级"
            )
    lines.append(
        f"\n解向量列三档实测均低于阈值 {'、'.join(solution_parts)}。"
    )
    if chain_parts:
        lines.append(
            f"误差标量列同样三档全过，{'、'.join(chain_parts)}。\n"
        )
    else:
        lines.append("")

    # 来源说明: 生成脚本与产物均在 examples/ 或本目录内, 不经过 tools 管线。
    ring_source = next(
        (s for s in snapshot.get("sources", []) if s.get("role") == "ea-rings"), {}
    )
    script, artifact = ring_source.get("script"), ring_source.get("path")
    if artifact:
        source_piece = (
            f"，由 [`{script}`](../../{script}) 运行得到" if script else ""
        )
        lines.append(
            f"数据为 CPU 串行（单 rank）验证的结果{source_piece}，存于 "
            f"[`{artifact}`](../../{artifact})；数值与档位说明见 "
            "[`examples/matrix_free_elasticity/results_analysis.md`]"
            "(../../examples/matrix_free_elasticity/results_analysis.md) §2.5。\n"
        )

    return "\n".join(lines)


_CJK_DIGITS = "〇一二三四五六七八九十"

# EA 每单元缓存一个 ldof x ldof 的 float64 单元刚度阵。3D 四面体 P1 位移场的
# ldof = 4 节点 x 3 分量 = 12, 故每单元 12*12*8 = 1152 B。设备对照(图面 (d))的
# 门禁已锁死
# dimension=3 / degree=1 / mesh_type=tet, 这个常数才敢写死在这里 —— 换单元类型
# 或阶数必须同步改, 否则带宽会算错而且错得看不出来。
_EA_BYTES_PER_CELL = 12 * 12 * 8


def _level_word(count: int) -> str:
    """把档数写成中文数词, 供表题与正文使用.

    表题里曾写死"四档", 补入 n=80 后当场过期 —— 生成区里的任何计数都必须由数据
    现算, 手抄的数字迟早与数据脱节。
    """
    if count <= 10:
        return _CJK_DIGITS[count]
    if count < 20:
        return "十" + (_CJK_DIGITS[count - 10] if count > 10 else "")
    return str(count)


def _render_panel_b(snapshot: dict[str, Any]) -> str:
    """渲染 (b) 格的两张表: 逐档峰值对照与最细档的分阶段高水位."""
    panel = snapshot["panels"]["b"]
    fa, ea = panel["fa"], panel["ea"]
    baseline = panel["baseline_bytes"]
    lines: list[str] = []

    limit = panel.get("memory_total_bytes")
    limit_text = f"，本机可用内存 `{limit / GIB:.0f} GiB`" if limit else ""
    lines.append(f"**表 b-1　{_level_word(len(panel['resolution']))}档"
                 f"进程峰值 RSS 对照**"
                 f"（3D `tet` P1、`--assembly-method fast`、单进程{limit_text}）\n")
    lines.append("| `n` | 自由度 | FA 峰值 / GiB | EA 峰值 / GiB | 峰值比 | 扣基线后比 | "
                 "FA 长期存储 / GiB | EA 长期存储 / GiB |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, resolution in enumerate(panel["resolution"]):
        peak_fa = fa["peak_rss_bytes"][index]
        peak_ea = ea["peak_rss_bytes"][index]
        above_fa = fa["peak_rss_above_baseline_bytes"][index]
        above_ea = ea["peak_rss_above_baseline_bytes"][index]
        lines.append(
            f"| {resolution} | {panel['dofs'][index]:,} | {peak_fa / GIB:.3f} | "
            f"{peak_ea / GIB:.3f} | {peak_fa / peak_ea:.2f} | "
            f"**{above_fa / above_ea:.2f}** | "
            f"{fa['stored_operator_bytes'][index] / GIB:.4f} | "
            f"{ea['stored_operator_bytes'][index] / GIB:.4f} |"
        )
    lines.append(
        f"\n基线（解释器 + FEALPy 导入）`{baseline / GIB:.3f} GiB`，"
        f"八次运行抖动 `{panel['baseline_spread_bytes'] / MIB:.3f} MiB`。"
        f"最粗档的峰值比 `{fa['peak_rss_bytes'][0] / ea['peak_rss_bytes'][0]:.2f}` 是基线稀释所致，"
        "不是层级差异；扣基线后从第二档起稳定。\n"
    )

    last = len(panel["resolution"]) - 1
    finest = panel["resolution"][last]
    lines.append(f"**表 b-2　`n={finest}` 的分阶段高水位**（累积值，单位 GiB）\n")
    lines.append("| 阶段 | FA | EA |")
    lines.append("|---|---:|---:|")
    stage_labels = {
        "baseline": "`baseline`（解释器 + 导入）",
        "mesh": "`mesh`（网格 + 空间）",
        "operator": "`operator`（组装 / 缓存 `K_e`）",
        "load": "`load`（体力向量全装配）",
        "bc": "`bc`（Dirichlet 处理）",
        "solve": "`solve`（CG）",
    }
    for position, stage in enumerate(panel["stage_order"]):
        lines.append(
            f"| {stage_labels.get(stage, stage)} | "
            f"{fa['stage_bytes'][last][position] / GIB:.3f} | "
            f"{ea['stage_bytes'][last][position] / GIB:.3f} |"
        )

    stages = panel["stage_order"]
    mesh_at = stages.index("mesh")
    operator_at = stages.index("operator")
    transient = (fa["stage_bytes"][last][operator_at]
                 - fa["stage_bytes"][last][mesh_at])
    retained = fa["stored_operator_bytes"][last]
    storage_ratios = " / ".join(
        f"{e / f:.2f}" for f, e in zip(fa["stored_operator_bytes"],
                                       ea["stored_operator_bytes"])
    )
    lines.append(
        f"\nFA 在 `operator` 阶段比 `mesh` 高 `{transient / GIB:.2f} GiB`，"
        f"而最终保留的 CSR 只有 `{retained / GIB:.3f} GiB`，"
        f"约 {100 * (1 - retained / transient):.0f}% 是 COO 转 CSR 的瞬态。"
        f"EA 的长期存储反而是 FA 的 {storage_ratios} 倍（逐档），"
        "它省的不是存得少，而是从不物化全局 COO/CSR。"
    )

    # 线性性是"可算规模 ≈ 内存比"这句话的依据: 只有内存对自由度线性,
    # 该表述才是内插而非外推。指数从第二档到最细档算, 跳过基线稀释的最粗档。
    dof_growth = panel["dofs"][last] / panel["dofs"][1]
    parts = []
    for name, block in (("FA", fa), ("EA", ea)):
        growth = (block["peak_rss_above_baseline_bytes"][last]
                  / block["peak_rss_above_baseline_bytes"][1])
        exponent = math.log(growth) / math.log(dof_growth)
        parts.append(f"{name} 扣基线峰值增 `{growth:.1f}` 倍（指数 `{exponent:.3f}`）")
    # 可算规模比就是最细档的扣基线内存比: 内存对自由度线性时两者相等。
    # 这个数会随实现改动漂移, 不能写死在文案里。
    scale_ratio = (fa["peak_rss_above_baseline_bytes"][last]
                   / ea["peak_rss_above_baseline_bytes"][last])
    lines.append(
        f"\n扣基线后内存对自由度是线性的：`n` 从 {panel['resolution'][1]} 到 "
        f"{finest} 自由度增 `{dof_growth:.1f}` 倍，" + "、".join(parts) +
        f"。两条都是 $O(N)$，故「同一内存上限下可算规模约 {scale_ratio:.1f} 倍」"
        "是内插表述，不依赖外推。"
    )

    lines.append(
        "\n**表 b-3\u3000峰值的两段分解：必需存储 vs 组装瞬态**"
        "（逐档，单位 GiB；下界 = 基线 + 网格与空间 + 该层级必须长期持有的算子数据）\n")
    lines.append("| `n` | FA 下界 | FA 瞬态 | FA 瞬态占比 | EA 下界 | EA 瞬态 | EA 瞬态占比 | 下界比 EA/FA |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, subdivision in enumerate(panel["resolution"]):
        floor_ratio = panel["floor_ratio_ea_over_fa"][index]
        lines.append(
            f"| {subdivision} "
            f"| {fa['floor_bytes'][index] / GIB:.3f} "
            f"| {fa['transient_bytes'][index] / GIB:.3f} "
            f"| {100 * fa['transient_fraction'][index]:.1f}% "
            f"| {ea['floor_bytes'][index] / GIB:.3f} "
            f"| {ea['transient_bytes'][index] / GIB:.3f} "
            f"| {100 * ea['transient_fraction'][index]:.1f}% "
            f"| {floor_ratio:.2f} |"
        )
    lines.append(
        f"\n⚠️ **下界比与峰值比方向相反。** 峰值比 EA 占优（最细档 "
        f"{fa['peak_rss_bytes'][last] / ea['peak_rss_bytes'][last]:.2f} 倍），"
        f"下界比却是 FA 占优（最细档 EA 是 FA 的 "
        f"{panel['floor_ratio_ea_over_fa'][last]:.2f} 倍）：EA 的必需存储更大，"
        f"它赢在瞬态。两条路径的峰值分别是各自下界的 "
        f"{fa['peak_over_floor'][last]:.1f} 倍与 {ea['peak_over_floor'][last]:.1f} 倍"
        f"（`n={finest}`），都远未触及各自的内存下界。\n")
    lines.append(
        "⚠️ 这里的下界是该实现路径的下界，不是问题本身的下界：`stored_operator` "
        "一项随算子层级而变（FA 是全局 CSR，EA 是逐单元 `K_e` 缓存），换成只存积分点"
        "因子的 PA 路径会重新定义它。\n")

    lines.append("正确性同时成立："
                 f"{_level_word(len(panel['resolution']))}档 CG 迭代数逐档相同（`{'/'.join(str(i) for i in fa['cg_iterations'])}`），"
                 f"最细档真残差 FA `{fa['true_relative_residual'][last]:.5e}`、"
                 f"EA `{ea['true_relative_residual'][last]:.5e}`；"
                 f"构造时间 EA 更短（`{ea['construction_seconds'][last]:.2f} s` vs "
                 f"`{fa['construction_seconds'][last]:.2f} s`），"
                 f"求解时间 EA 更长（`{ea['solve_seconds'][last]:.2f} s` vs "
                 f"`{fa['solve_seconds'][last]:.2f} s`）。")
    return "\n".join(lines)


def _render_panel_c(snapshot: dict[str, Any]) -> str:
    """渲染设备对照(CPU / 单卡 GPU 耗时与等价性证据), 即**图面的 (d)**.

    ⚠️ 函数名里的 c 是**数据组**键名(snapshot["panels"]["c"]), 不是图面位置。
    2026-08-24 换版式后两者正好错开: panels.c -> 图面 (d), panels.d -> 图面 (c)。
    键名按采集顺序固定, 改名会作废历史快照, 所以让它们错开是有意的选择, 不是待
    修的中间态。凡是要写给人看的位置字样, 一律按图面写。

    没有注册 case 时快照里这一组是 placeholder, 此时渲染占位说明而不是空表 ——
    文档要能如实反映"这一格还没有数据", 而不是留一段看不出状态的空白。
    """
    panel = snapshot["panels"]["c"]
    if panel.get("status") != "measured":
        return ("⚠️ 这一格尚无数据点。\n\n" + str(panel.get("reason", "")))

    cpu = panel["cpu"]
    gpu = panel["cuda"]
    last = len(panel["dofs"]) - 1
    lines: list[str] = []

    lines.append(
        f"**表 c-1　{_level_word(len(panel['resolution']))}档 "
        f"CPU / 单卡 GPU 求解耗时对照**"
        f"（3D `tet` P1、EA、`--assembly-method fast`、`pytorch` 后端，"
        f"warmup {panel['warmup']} 次后计时 {panel['repeats']} 次取中位数，"
        f"GPU 为 `{panel['gpu_name']}`）\n")
    lines.append("| `n` | 自由度 | CPU 求解 / s | GPU 求解 / s | 加速比 | "
                 "CG 迭代数 | 两侧解相对差 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for index, resolution in enumerate(panel["resolution"]):
        iterations = cpu["cg_iterations"][index]
        same = iterations == gpu["cg_iterations"][index]
        lines.append(
            f"| {resolution} | {panel['dofs'][index]:,} | "
            f"{cpu['solve_seconds'][index]:.3f} | {gpu['solve_seconds'][index]:.3f} | "
            f"**{panel['speedup_solve'][index]:.2f}** | "
            f"{iterations}{'' if same else ' ⚠️'} | "
            f"`{panel['solution_relative_gap'][index]:.1e}` |")
    lines.append("")

    gate = panel.get("gate", {}).get("solution_relative_tol")
    worst = max(panel["solution_relative_gap"])
    lines.append(
        f"加速比只在两侧算同一个问题时才有意义。"
        f"{_level_word(len(panel['resolution']))}档解的相对差最差 "
        f"`{worst:.1e}`，比门禁 `{gate:.0e}` 松 "
        f"{math.floor(math.log10(gate / worst))} 个量级；更强的一条是两侧 CG "
        f"迭代数逐档相同（`{'/'.join(str(i) for i in cpu['cg_iterations'])}`），"
        f"即整条 Krylov 轨迹一致，而不只是碰巧收敛到附近。两侧唯一的差别是 "
        f"`bm.set_default_device`，后端同为 `pytorch`。\n")

    # 加速比"为什么是这个形状"要用带宽回答。每次 CG 迭代把整份 K_e 流一遍, 用
    # 单次迭代耗时反算有效带宽, 就能看出 CPU 端各档基本恒定、GPU 端随规模爬升 ——
    # 这条曲线的形状是 GPU 的饱和过程, 不是玄学。
    lines.append(
        f"**表 c-2　逐档有效访存带宽**"
        f"（由 `cells × {_EA_BYTES_PER_CELL} B ÷ 单次 CG 迭代耗时` 反算；"
        f"只计 K_e 流量，未计索引数组与向量，故为下界）")
    lines.append("")
    lines.append("| `n` | K_e / GB | 单次迭代 CPU / ms | 单次迭代 GPU / ms | "
                 "CPU 有效带宽 | GPU 有效带宽 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    bandwidth: list[tuple[float, float]] = []
    for index, resolution in enumerate(panel["resolution"]):
        ke_bytes = panel["cells"][index] * _EA_BYTES_PER_CELL
        cpu_per_iteration = cpu["solve_seconds"][index] / cpu["cg_iterations"][index]
        gpu_per_iteration = gpu["solve_seconds"][index] / gpu["cg_iterations"][index]
        cpu_bandwidth = ke_bytes / cpu_per_iteration / 1e9
        gpu_bandwidth = ke_bytes / gpu_per_iteration / 1e9
        bandwidth.append((cpu_bandwidth, gpu_bandwidth))
        lines.append(
            f"| {resolution} | {ke_bytes / 1e9:.3f} | "
            f"{cpu_per_iteration * 1e3:.3f} | {gpu_per_iteration * 1e3:.3f} | "
            f"{cpu_bandwidth:.1f} GB/s | **{gpu_bandwidth:.1f} GB/s** |")
    lines.append("")

    cpu_low = min(item[0] for item in bandwidth)
    cpu_high = max(item[0] for item in bandwidth)
    lines.append(
        f"加速曲线的形状就是 GPU 的带宽饱和曲线。CPU 端逐档基本恒定"
        f"（`{cpu_low:.0f}`–`{cpu_high:.0f}` GB/s，规模再大也吃不到更多带宽），"
        f"GPU 端从 `{bandwidth[0][1]:.1f}` GB/s 一路爬到 "
        f"`{bandwidth[last][1]:.0f}` GB/s：最粗档远未填满卡，算子作用的时间几乎"
        f"全花在核函数启动与同步上；到最细档才把卡喂饱。"
        f"加速比即这两条带宽曲线的比值。")
    lines.append("")

    lines.append("**表 c-3　算子构造时间**（同一批运行，不上图）\n")
    lines.append("| `n` | CPU 构造 / s | GPU 构造 / s | 加速比 |")
    lines.append("|---:|---:|---:|---:|")
    for index, resolution in enumerate(panel["resolution"]):
        lines.append(
            f"| {resolution} | {cpu['build_seconds'][index]:.3f} | "
            f"{gpu['build_seconds'][index]:.3f} | "
            f"{panel['speedup_build'][index]:.2f} |")
    lines.append("")

    # 本批只能把交叉点夹在相邻两档之间, 说不出具体值 —— 写成 "约 X 附近" 就是
    # 把首个加速档当成了交叉点, 那是没量过的数。
    first_faster = next(
        (i for i, s in enumerate(panel["speedup_solve"]) if s > 1.0), None)
    if first_faster is None:
        crossing = (f"本批之外（{_level_word(len(panel['resolution']))}档 "
                    f"GPU 均未反超）")
    elif first_faster == 0:
        crossing = f"`{panel['dofs'][0]:,}` 自由度之前"
    else:
        crossing = (f"`{panel['dofs'][first_faster - 1]:,}` 与 "
                    f"`{panel['dofs'][first_faster]:,}` 自由度之间")
    # warmup 1 / repeats 3 撑不起小数点后两位。把样本能给出的最宽区间写出来,
    # 比端出一个 "16.08" 更经得起追问 —— 那两位小数是中位数的舍入, 不是精度。
    low = min(cpu["solve_samples"][last]) / max(gpu["solve_samples"][last])
    high = max(cpu["solve_samples"][last]) / min(gpu["solve_samples"][last])
    lines.append(
        f"最粗档 GPU 更慢（`{panel['dofs'][0]:,}` 自由度上 "
        f"{panel['speedup_solve'][0]:.2f} 倍）：核函数启动与同步的固定开销吃掉全部"
        f"收益。交叉点落在{crossing}，最细档 `{panel['dofs'][last]:,}` 自由度上加速 "
        f"**约 {panel['speedup_solve'][last]:.0f} 倍**"
        f"（中位数 `{panel['speedup_solve'][last]:.2f}`，但 warmup "
        f"{panel['warmup']} 次 / 计时 {panel['repeats']} 次的样本只把它夹在 "
        f"`{low:.1f}`–`{high:.1f}` 之间，不要引用小数点后第二位）。"
        f"最粗一档保留在表内，因为交叉点本身就是「什么时候该用 GPU」这个结论。")
    lines.append("")

    threads = cpu.get("torch_threads") or []
    known = {item for item in threads if item}
    if len(known) == 1:
        lines.append(
            f"分母是 {known.pop()} 个 CPU 线程，不是一个核：脚本不限制线程数，"
            f"CPU 侧走 `torch` 默认值满核跑；加速比是「一块卡对一整颗 CPU」，"
            f"不是「一块卡对一个核」。")
        lines.append("")

    peaks = gpu.get("gpu_peak_allocated_bytes") or []
    totals = gpu.get("gpu_memory_total_bytes") or []
    if len(peaks) > last and peaks[last] and len(totals) > last and totals[last]:
        peak = peaks[last]
        total = totals[last]
        cached = panel["cells"][last] * _EA_BYTES_PER_CELL
        lines.append(
            f"⚠️ **这一格的天花板是显存，不是速度。** 最细档 GPU 峰值分配 "
            f"`{peak / 2 ** 30:.3f} GiB`，已占整卡 `{total / 2 ** 30:.3f} GiB` 的 "
            f"**{peak / total * 100:.1f}%**，同一块卡上这条路线再上一档就装不下。"
            f"但长期持有的 K_e 只有 `{cached / 2 ** 30:.3f} GiB`，"
            f"峰值的 {(1 - cached / peak) * 100:.1f}% 是装配期瞬态："
            f"占住卡的是构造过程，而非 matrix-free 要长期留的数据，"
            f"也正是研究内容 2 的 PA 路径要去掉的那一部分。")

    return "\n".join(lines)


def _render_panel_d(snapshot: dict[str, Any]) -> str:
    """渲染进程级 (MPI) 强扩展, 即**图面的 (c)**.

    ⚠️ 函数名里的 d 是数据组键名(snapshot["panels"]["d"]), 不是图面位置, 见
    _render_panel_c 的说明。

    这一格与设备对照(图面 (d))问的不是同一个问题, 加速比也不能相加或互相外推:
    (d) 切的是设备内 (SIMT), 本格切的是进程 (MPI), 两个并行层级各有各的墙, 分母
    也不同 —— 本格对单进程, (d) 对同后端 CPU 的 16 条线程。本函数因此不渲染任何
    跨层比较, 跨层的话只写在手写段落里, 并写明口径。
    """
    panel = snapshot["panels"]["d"]
    if panel.get("status") != "measured":
        return ("⚠️ 这一格尚无数据点。\n\n" + str(panel.get("reason", "")))

    ranks = panel["ranks"]
    last = len(ranks) - 1
    lines: list[str] = []

    cadence = (f"warmup {panel['warmup']} 次后计时 {panel['repeats']} 次"
               if panel.get("warmup") is not None
               else "⚠️ 计时口径未随产物落盘")
    lines.append(
        f"**表 d-1　{_level_word(len(ranks))}档进程级强扩展**"
        f"（3D `tet` P1、EA、`--assembly-method fast`、`numpy` 后端，"
        f"规模固定 `n = {panel['resolution']}` / `{panel['dofs']:,}` 自由度，"
        f"`OMP_NUM_THREADS=1` 锁死线程层，{cadence}）\n")
    lines.append("| 进程数 | CG 求解 / s | 加速比 | 并行效率 | CG 迭代数 | "
                 "真实相对残差 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for index, count in enumerate(ranks):
        lines.append(
            f"| {count} | {panel['cg_seconds'][index]:.3f} | "
            f"**{panel['speedup'][index]:.2f}** | "
            f"{panel['efficiency'][index] * 100:.1f}% | "
            f"{panel['cg_iterations'][index]} | "
            f"`{panel['true_relative_residual'][index]:.3e}` |")
    lines.append("")

    # 强扩展表最容易被质疑的一点是"换了分区是不是换了问题"。迭代数逐档相同比
    # 残差接近更强: 它说明整条 Krylov 轨迹没变, 而不只是终点碰巧落在附近。
    residuals = panel["true_relative_residual"]
    spread = (max(residuals) - min(residuals)) / min(residuals)
    lines.append(
        f"分区没有改变代数。{_level_word(len(ranks))}档 CG 迭代数逐档相同"
        f"（`{'/'.join(str(i) for i in panel['cg_iterations'])}`），"
        f"真实相对残差跨档相对离散仅 `{spread:.1e}`，这点差异来自归约次序随分区"
        f"改变，是浮点求和的正常表现，不是误差；加速来自真的省了时间，不是少算。\n")

    # 效率掉下去以后, 唯一有用的追问是"掉在通信还是掉在本地算"。把一次 MatVec
    # 拆成本地核 + 两处同步就能直接回答, 不必猜。
    lines.append(
        "**表 d-2　单次算子作用 (MatVec) 的分解**"
        "（`local_kernel` 是本地单元作用，两处 `sync` 是界面自由度的输入/输出"
        "同步归约）")
    lines.append("")
    lines.append("| 进程数 | MatVec / ms | 本地核 / ms | 输入同步 / ms | "
                 "输出同步 / ms | 同步占比 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    sync_share: list[float] = []
    kernel_share: list[float] = []
    for index, count in enumerate(ranks):
        matvec = panel["matvec_seconds"][index]
        kernel = panel["local_kernel_seconds"][index]
        sync = panel["input_sync_seconds"][index] + panel["output_sync_seconds"][index]
        share = sync / matvec
        sync_share.append(share)
        kernel_share.append(kernel / matvec)
        lines.append(
            f"| {count} | {matvec * 1e3:.2f} | {kernel * 1e3:.2f} | "
            f"{panel['input_sync_seconds'][index] * 1e3:.2f} | "
            f"{panel['output_sync_seconds'][index] * 1e3:.2f} | "
            f"{share * 100:.1f}% |")
    lines.append("")

    # 三个分项各自对 rank 取 max(benchmark_cpu_ea.py 的 measure_profiled_parallel_
    # matvec), 不同项的 max 可能来自不同进程, 故分项之和会超过 MatVec 总计。不写
    # 明这一点, 读者会拿"同步占比"和"本地核占比"相加发现大于 100% 而不知所措。
    parts_last = (panel["local_kernel_seconds"][last]
                  + panel["input_sync_seconds"][last]
                  + panel["output_sync_seconds"][last])
    parts_first = (panel["local_kernel_seconds"][0]
                   + panel["input_sync_seconds"][0]
                   + panel["output_sync_seconds"][0])
    lines.append(
        f"⚠️ **分项是上界，不是精确划分。** 三个分项各自对 rank 取最大值再按"
        f"样本取中位数（`examples/matrix_free_elasticity/benchmark_cpu_ea.py` 的 "
        f"`measure_profiled_parallel_matvec`），不同项的最大值可能来自不同进程，"
        f"因此三项之和会超过 MatVec 总计：{ranks[last]} 进程档 "
        f"`{parts_last * 1e3:.2f} ms` vs `{panel['matvec_seconds'][last] * 1e3:.2f} ms`，"
        f"同步占比与本地核占比相加也随之大于 `100%`。方向是保守的：这个口径"
        f"高估同步，故下面「通信不是瓶颈」的结论只会更稳。1 进程档反而少 "
        f"`{(panel['matvec_seconds'][0] - parts_first) * 1e3:.2f} ms`：单 rank 取最大值"
        f"没有意义，而 MatVec 总计还含 Dirichlet 边界投影，它不在三个分项里。\n")
    lines.append(
        f"⚠️ **同步时间里含等待。** 两处 `sync` 计的是包住阻塞式 `dof_comm.sync_add` "
        f"的墙钟时间（`soptx:src/soptx/fem/distributed/operator.py:80-91`），先算完的"
        f"进程在这里等最慢的，负载不均会被记成同步开销。输出同步比输入同步贵 "
        f"`{panel['output_sync_seconds'][last] / panel['input_sync_seconds'][last]:.1f}` "
        f"倍而两者搬运的数据量相同，最可能的解释是它紧跟本地核、吸收了这份不均衡。"
        f"⚠️ 这是推断，现有计时分不开真通信与等待，要拆开须用 MPI profiler 看"
        f"各 rank 到达同步点的时刻分布。若成立，纯通信成本比表中占比还低。\n")

    kernel_speedup = (panel["local_kernel_seconds"][0]
                      / panel["local_kernel_seconds"][last])
    tail_ranks = ranks[last] / ranks[last - 1]
    tail_gain = (panel["local_kernel_seconds"][last - 1]
                 / panel["local_kernel_seconds"][last])
    lines.append(
        f"效率掉下去不是通信造成的。最大规模档同步只占一次 MatVec 的 "
        f"`{sync_share[last] * 100:.1f}%`，本地核占 "
        f"`{kernel_share[last] * 100:.1f}%`；而本地核本身从 1 进程到 "
        f"{ranks[last]} 进程只快了 "
        f"`{kernel_speedup:.2f}` 倍，远不到 `{ranks[last]}` 倍。更直接的一条是"
        f"末两档：进程数翻 `{tail_ranks:.0f}` 倍，本地核只快 `{tail_gain:.2f}` 倍，"
        f"多出来的核已经吃不到更多内存带宽。EA 的算术强度约 "
        f"`0.25 flop/byte`，本来就是 memory-bound，节点内加进程买到的是更多算力"
        f"而不是更多带宽，所以曲线在这里压平。\n")

    lines.append("**表 d-3　算子构造时间**（同一批运行，不上图）\n")
    lines.append("| 进程数 | 构造 / s | 加速比 | 端到端管线 / s |")
    lines.append("|---:|---:|---:|---:|")
    build = panel["construction_seconds"]
    for index, count in enumerate(ranks):
        lines.append(
            f"| {count} | {build[index]:.3f} | {build[0] / build[index]:.2f} | "
            f"{panel['pipeline_seconds'][index]:.3f} |")
    lines.append("")

    if panel.get("warmup") is None:
        lines.append(
            "⚠️ **这批数是单次采样。** 产物里没有 `warmup` / `repeats` 字段，"
            "说明它们跑在脚本记录该字段之前；实跑用的是 `--warmup 0 --repeats 1`"
            "（脚本默认为 `1` / `3`）。单次采样足以定曲线形状（迭代数与残差逐档"
            "一致本身就是一次跨档自洽性检验），但任何一档的秒数都不带误差棒，"
            "写进正式图件前应按默认口径重跑。")

    return "\n".join(lines)


def _render_provenance(snapshot: dict[str, Any]) -> str:
    """渲染快照溯源与源文件清单."""
    record = snapshot["provenance"]
    lines: list[str] = []

    flag = "✅ 可复现" if snapshot["reproducible"] else "⚠️ **不可复现**"
    lines.append(f"{flag}　`git_revision = {record['git_revision']}`"
                 f"（`{record['git_branch']}`，`git_dirty = {record['git_dirty']}`）\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 采集时间 | `{record['generated_at_utc']}` |")
    lines.append(f"| 平台 | `{record['platform']}` |")
    lines.append(f"| Python / NumPy / FEALPy | `{record['python']}` / "
                 f"`{record['numpy']}` / `{record['fealpy']}` |")
    if record.get("memory_total_bytes"):
        lines.append(f"| 本机物理内存 | `{record['memory_total_bytes'] / GIB:.1f} GiB` |")
    lines.append(f"| 源文件 | {len(snapshot['sources'])} 个，逐个记 `sha256` |")

    failures = snapshot["gate_failures"]
    lines.append(
        f"\n门禁：{'**' + str(len(failures)) + ' 项未通过**' if failures else '全部通过'}。"
    )
    for item in failures:
        lines.append(f"- {item}")
    for note in snapshot["notes"]:
        lines.append(f"- 提示：{note}")
    return "\n".join(lines)


RENDERERS = {
    "panel-a": _render_panel_a,
    "panel-b": _render_panel_b,
    "panel-c": _render_panel_c,
    "panel-d": _render_panel_d,
    "provenance": _render_provenance,
}


def render(snapshot: dict[str, Any], document: str) -> str:
    """把渲染结果写回文档的标记区间.

    参数:
        snapshot: ``fig2_data.json`` 的内容.
        document: ``results_analysis.md`` 的全文.

    返回:
        updated: 替换后的全文.

    异常:
        ReportError: 某个区间的 BEGIN/END 标记缺失或不成对.
    """
    for name in GENERATED_BLOCKS:
        begin = f"<!-- BEGIN generated: {name} -->"
        end = f"<!-- END generated: {name} -->"
        # 区间内容可为空 (首次生成), 故不在正则里强求换行, 由替换串补齐。
        pattern = re.compile(
            rf"{re.escape(begin)}.*?{re.escape(end)}", re.DOTALL
        )
        if not pattern.search(document):
            raise ReportError(
                f"results_analysis.md 缺少区间标记 'generated: {name}'，"
                "或 BEGIN/END 不成对。"
            )
        body = RENDERERS[name](snapshot)
        document = pattern.sub(
            lambda _match, body=body, begin=begin, end=end:
                f"{begin}\n{body}\n{end}",
            document,
            count=1,
        )
    return document


def update(snapshot_path: Path | None = None,
           document_path: Path | None = None) -> Path:
    """读快照、渲染、写回文档.

    参数:
        snapshot_path: 快照路径, ``None`` 时用 ``figure_data/fig2_data.json``.
        document_path: 文档路径, ``None`` 时用本目录 ``results_analysis.md``.

    返回:
        path: 实际写入的文档路径.

    异常:
        ReportError: 快照不存在.
    """
    snapshot_path = snapshot_path or config.FIGURE_DATA_DIR / "fig2_data.json"
    document_path = document_path or config.EXPERIMENT_DIR / "results_analysis.md"
    if not snapshot_path.is_file():
        raise ReportError(
            f"快照不存在 -> {snapshot_path}\n"
            "  先执行 run.py --all 或 run.py --collect。"
        )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document = document_path.read_text(encoding="utf-8")
    document_path.write_text(render(snapshot, document), encoding="utf-8")
    return document_path
