# -*- coding: utf-8 -*-
"""悬臂梁应力算例的应力奇异性网格加密探针: 冻结设计, 只做状态求解.

固定网格上的一个读数不能判定奇异性 —— 奇异性是 ``h -> 0`` 的行为. 本模块把设计
变量整体拿掉 (要么钉为均质 ``rho = 1``, 要么冻结一份已收敛构型), 在一致加密的
网格序列上各做一次前向弹性求解, 报告两处几何奇点邻域的归一化 von Mises 读数随
``h`` 的走向.

两组探针分工不同, 缺一不可::

    A. 均质参考 (``--plan A``)   rho = 1 的整块矩形. 没有杆件与孔洞, 除两处奇点
                                 外应力场光滑, 读数增长只能来自奇异项. 回答
                                 "连续问题里有没有奇异性".
    B. 冻结设计 (``--plan B``)   已收敛构型的 rho 场按面积保持映射到细网格. 回答
                                 "实际构型里奇异性是否仍然主导".
    C. 设计无关性 (``--plan C``) 换一份构型重测, 核验结论不依赖冻结了谁.

两处奇点的理论预言不同, 判据也不同::

    固支角点 (0,0),(0,40)    Dirichlet--Neumann 直角楔, Williams 幂律
                             sigma ~ r^(lambda-1), 平面应力 nu=0.25 时
                             lambda = 0.78107 -> 指数 -0.2189. 乘性信号:
                             网格每加倍, 最内环读数 x 2^0.2189 = 1.164.
    载荷端点 (80,17),(80,23) 直边上纯切向牵引的阶跃间断, Muskhelishvili 对数型
                             sigma ~ log(1/r). 加性信号: 网格每加倍, 读数增加一
                             个常数, 幅度远小于幂律, 需要均质参考才能从结构背景
                             中分离.

评价一律在不豁免 (``load_pad_radius = support_pad_radius = 0``) 的约束对象上做:
要报的正是被垫片掩盖掉的那些数, 垫片在这里只会把待测量抹掉.

用法 (本模块独立执行, 不经 ``compare.py`` 派发)::

    python singularity_h_probe.py --plan A
    python singularity_h_probe.py --plan B
    python singularity_h_probe.py --plan C
    python singularity_h_probe.py --plan A --dry-run          # 只打印矩阵与自由度
    python singularity_h_probe.py --levels 80x40,160x80 \
        --discretizations lfem-2 --probe homogeneous          # 手工指定

产出写到 ``outputs/<case>/postprocess/singularity_h_probe_<plan>.json``, 带
provenance 戳记; 终端同时打印分环表与逐层增长比表.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from fealpy.backend import backend_manager as bm

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
    build_stress_analysis_pipeline,
    build_stress_config,
)
import provenance  # noqa: E402


CASE_ID = "cantilever-middle-2d-stress"

# Williams 特征值: 直角 Dirichlet--Neumann 楔, 平面应力 nu = 0.25.
# 出处见 soptx.problems.elasticity.cantilever.CantileverMiddle2d.clamped_corner_points.
WILLIAMS_LAMBDA = 0.78107

# 固定物理半径的环边界 (mm): 跨网格可比, 是判发散的主量.
# 最内环 [0, 0.5) 在 80x40 上只含最靠角点的 2 个单元, 加密后单元向奇点逼近,
# 幂律情形下该环的最大值应按 h^(lambda-1) 增长.
DEFAULT_RING_EDGES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

DEFAULT_LEVELS: tuple[tuple[int, int], ...] = ((80, 40), (160, 80), (320, 160))

# 各方案的 (网格层, 离散) 矩阵. 分层裁剪的依据:
#   判发散只需一族离散跑满三层 -> LFEM p=2 (最便宜, 且定环分析中与理论最吻合);
#   排除单族数值伪像只需两族在两个共同层上一致 -> Hu--Zhang k=2 跑前两层;
#   Hu--Zhang k=3 在 320x160 约 290 万自由度的不定鞍点系统, 既无额外 h 趋势也无
#   额外跨族证据, 且有 MUMPS 内存风险 (见 docs/known-issues), 故只留最粗层.
PLANS: dict[str, dict[str, Any]] = {
    "A": {
        "probe": "homogeneous",
        "design_dir": None,
        "matrix": {
            (80, 40): (("lfem", 2),),
            (160, 80): (("lfem", 2),),
            (320, 160): (("lfem", 2),),
        },
        "note": "均质参考 rho = 1: 判连续问题里两处奇点的存在性.",
    },
    "B": {
        "probe": "frozen",
        "design_dir": (
            "analyzer-lfem__lfem_constraint-apparent__load_pad_radius-1.5__order-2"
        ),
        "matrix": {
            (80, 40): (("lfem", 2), ("huzhang", 2), ("huzhang", 3)),
            (160, 80): (("lfem", 2), ("huzhang", 2)),
            (320, 160): (("lfem", 2),),
        },
        "note": "冻结 LF k=2 有垫片构型: 判实际构型里奇异性是否仍主导.",
    },
    "C": {
        "probe": "frozen",
        "design_dir": (
            "analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-3"
        ),
        # 必须两层: 指数是层间量, 单层只能给读数, 给不出增长比.
        "matrix": {
            (80, 40): (("lfem", 2),),
            (160, 80): (("lfem", 2),),
        },
        "note": "换 HZ k=3 构型抽检: 核验角点增长率不依赖于冻结了谁的设计.",
    },
}


# ==================================================================== 一、参数


def case_parameters(case_id: str = CASE_ID) -> dict[str, Any]:
    """从 cases.toml 取该算例的扁平参数, 保证与优化时同口径."""
    for case in load_cases(CASES_FILE):
        if case["id"] == case_id:
            return flatten_parameters(case)
    raise SystemExit(f"cases.toml 中没有算例 {case_id}.")


def parse_level(text: str) -> tuple[int, int]:
    """把 ``80x40`` 解析为 ``(80, 40)``."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise SystemExit(f"网格层写法应为 <nx>x<ny>, 收到 {text!r}.")
    return int(parts[0]), int(parts[1])


def parse_discretization(text: str) -> tuple[str, int]:
    """把 ``lfem-2`` 解析为 ``('lfem', 2)``."""
    method, _, order = text.rpartition("-")
    if method not in ("lfem", "huzhang"):
        raise SystemExit(f"未知的离散族 {method!r}, 应为 lfem 或 huzhang.")
    return method, int(order)


def estimate_dofs(nx: int, ny: int, method: str, order: int) -> int:
    """粗估自由度规模, 仅用于 ``--dry-run`` 的体量提示.

    Notes
    -----
    LFEM 的计数是精确的 (Euler 公式给棱数); Hu--Zhang 的计数是量级估算, 未逐项
    核对应力空间的连续性约束, 不可用作性能结论的依据.
    """
    n_vertex = (nx + 1) * (ny + 1)
    n_cell = 2 * nx * ny
    n_edge = n_vertex + n_cell - 1
    if method == "lfem":
        if order == 1:
            return 2 * n_vertex
        if order == 2:
            return 2 * (n_vertex + n_edge)
        interior = (order - 1) * (order - 2) // 2
        return 2 * (n_vertex + (order - 1) * n_edge + interior * n_cell)
    local = 3 * (order + 1) * (order + 2) // 2
    sigma = 3 * n_vertex + 2 * (order - 1) * n_edge
    sigma += n_cell * max(local - 9 - 6 * (order - 1), 0)
    return sigma + n_cell * order * (order + 1)


# ============================================================== 二、密度场映射


def _quad_index(
    points: np.ndarray,
    origin: tuple[float, float],
    spacing: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """把坐标映射到结构化四边形的 ``(ix, iy)`` 整数索引, 越界钳制到边界格."""
    ix = np.floor((points[:, 0] - origin[0]) / spacing[0]).astype(np.int64)
    iy = np.floor((points[:, 1] - origin[1]) / spacing[1]).astype(np.int64)
    return np.stack(
        [np.clip(ix, 0, shape[0] - 1), np.clip(iy, 0, shape[1] - 1)], axis=1
    )


def map_density_to_mesh(
    design: np.ndarray,
    coarse_mesh: Any,
    coarse_shape: tuple[int, int],
    fine_mesh: Any,
    domain: Sequence[float],
) -> np.ndarray:
    """把粗网格上的逐单元密度按"落点归属"映射到细网格.

    映射是面积保持的分片常数延拓: 细单元取其重心所在粗单元的密度. 两套网格都是
    结构化矩形的三角剖分, 故先用重心的整数格索引把候选粗单元缩到同一四边形内的
    2 个, 再用重心坐标定位, 不必做全局点定位.

    Parameters
    ----------
    design : ndarray, shape (NC_coarse,)
        粗网格上的逐单元密度.
    coarse_mesh, fine_mesh : object
        fealpy 三角网格对象, 需支持 ``entity('node')`` 与 ``entity('cell')``.
    coarse_shape : tuple of int
        粗网格的 ``(nx, ny)``.
    domain : sequence of float
        计算域 ``[xmin, xmax, ymin, ymax]``.

    Returns
    -------
    ndarray, shape (NC_fine,)
        细网格上的逐单元密度.

    Notes
    -----
    细单元重心恰落在粗单元对角线上时无严格包含关系, 此时取重心坐标最小分量最大
    的那个候选 (即"最不越界"的一个), 避免因浮点抖动漏映射.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in domain)
    nx, ny = coarse_shape
    spacing = ((xmax - xmin) / nx, (ymax - ymin) / ny)

    node_c = np.asarray(bm.to_numpy(coarse_mesh.entity("node")), dtype=np.float64)
    cell_c = np.asarray(bm.to_numpy(coarse_mesh.entity("cell")), dtype=np.int64)
    bary_c = node_c[cell_c].mean(axis=1)
    bary_f = np.asarray(
        bm.to_numpy(fine_mesh.entity_barycenter("cell")), dtype=np.float64
    )

    idx_c = _quad_index(bary_c, (xmin, ymin), spacing, (nx, ny))
    idx_f = _quad_index(bary_f, (xmin, ymin), spacing, (nx, ny))
    key_c = idx_c[:, 0] * ny + idx_c[:, 1]
    key_f = idx_f[:, 0] * ny + idx_f[:, 1]

    # 每个四边形恰含 2 个粗三角形, 按 key 排序后成对取出.
    order = np.argsort(key_c, kind="stable")
    sorted_key = key_c[order]
    starts = np.searchsorted(sorted_key, key_f, side="left")
    stops = np.searchsorted(sorted_key, key_f, side="right")

    tri = node_c[cell_c]                      # (NC_coarse, 3, 2)
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]

    mapped = np.empty(bary_f.shape[0], dtype=np.float64)
    for i in range(bary_f.shape[0]):
        candidates = order[starts[i]:stops[i]]
        if candidates.size == 0:
            # 理论上不会发生; 退化到全局最近重心, 并保留可追溯的行为.
            mapped[i] = design[int(np.argmin(np.sum((bary_c - bary_f[i]) ** 2, axis=1)))]
            continue
        rel = bary_f[i] - tri[candidates, 0]
        a = (rel[:, 0] * v1[candidates, 1] - rel[:, 1] * v1[candidates, 0]) / det[candidates]
        b = (v0[candidates, 0] * rel[:, 1] - v0[candidates, 1] * rel[:, 0]) / det[candidates]
        score = np.minimum(np.minimum(a, b), 1.0 - a - b)
        mapped[i] = design[int(candidates[int(np.argmax(score))])]
    return mapped


# ================================================================== 三、求解


def _per_cell_max(values: Any) -> np.ndarray:
    """把 (NC, ...) 的评价点量约化为逐单元最大值."""
    array = np.asarray(bm.to_numpy(values), dtype=np.float64)
    return array.reshape(array.shape[0], -1).max(axis=1)


def solve_level(
    parameters: dict[str, Any],
    nx: int,
    ny: int,
    method: str,
    order: int,
    density: np.ndarray | None,
) -> dict[str, Any]:
    """在给定网格与离散上做一次前向状态求解.

    Parameters
    ----------
    density : ndarray or None
        细网格上的逐单元密度; ``None`` 表示均质参考 (整体钉为 1).

    Returns
    -------
    dict
        含 ``solid_stress_ratio`` (逐单元 sigma^sol_vM / sigma_bar)、``density``、
        ``barycenter`` 与 ``seconds``.
    """
    run_parameters = {
        **parameters,
        "nx": nx,
        "ny": ny,
        "comparison_orders": [order],
        # 探针要测的正是垫片盖住的那两处, 故一律关掉豁免与实体保留.
        "load_pad_radius": 0.0,
        "support_pad_radius": 0.0,
    }
    config = build_stress_config(run_parameters)
    pipeline = build_stress_analysis_pipeline(config, run_parameters, method, order)
    rho = pipeline.density_distribution
    if density is None:
        rho[:] = 1.0
    else:
        if rho.shape[0] != density.shape[0]:
            raise SystemExit(
                f"{method}-{order} @ {nx}x{ny}: 网格单元数 {rho.shape[0]} 与映射后的"
                f"密度 {density.shape[0]} 不符."
            )
        rho[:] = bm.asarray(density, dtype=rho.dtype)
    started = time.perf_counter()
    state = pipeline.analyzer.solve_state(rho_val=rho)
    constraint = pipeline.stress_constraint
    # 约束求值会就地把 stress_solid / von_mises / stiffness_ratio 填进 state,
    # compute_solid_stress_ratio 依赖这些键, 故必须先走一遍. 本探针关掉了豁免,
    # 用 compute_unexempted_constraint 以免将来改了默认掩码后读到哨兵值.
    constraint_value = constraint.compute_unexempted_constraint(rho, state)
    ratio = constraint.compute_solid_stress_ratio(rho, state)
    return {
        "pipeline": pipeline,
        "constraint_value": _per_cell_max(constraint_value),
        "solid_stress_ratio": _per_cell_max(ratio),
        "density": np.asarray(bm.to_numpy(rho), dtype=np.float64).reshape(-1),
        "barycenter": np.asarray(
            bm.to_numpy(pipeline.mesh.entity_barycenter("cell")), dtype=np.float64
        ),
        "seconds": time.perf_counter() - started,
    }


# ================================================================== 四、统计


def ring_statistics(
    barycenter: np.ndarray,
    values: np.ndarray,
    density: np.ndarray,
    center: tuple[float, float],
    ring_edges: Sequence[float],
    solid_threshold: float,
) -> dict[str, Any]:
    """以 ``center`` 为心做定半径分环统计, 并对环内最大值做 log-log 斜率拟合.

    Notes
    -----
    环边界取固定物理半径而非 ``h`` 的倍数: 只有这样, 同一个环在不同网格上才是
    同一块区域, 其最大值随 ``h`` 的走向才可解释为发散或收敛. 另外单列
    ``innermost`` 一栏, 报离奇点最近的那圈单元 (其半径本身随 ``h`` 收缩), 它是
    最原始的发散信号.
    """
    radius = np.hypot(barycenter[:, 0] - center[0], barycenter[:, 1] - center[1])
    solid = density > solid_threshold

    rings: list[dict[str, Any]] = []
    for lower, upper in zip(ring_edges[:-1], ring_edges[1:]):
        band = (radius >= lower) & (radius < upper)
        both = band & solid
        rings.append({
            "r_min": float(lower),
            "r_max": float(upper),
            "n_cells": int(band.sum()),
            "n_solid_cells": int(both.sum()),
            "max_all": float(values[band].max()) if band.any() else None,
            "max_solid": float(values[both].max()) if both.any() else None,
            "mean_radius_solid": float(radius[both].mean()) if both.any() else None,
        })

    # 最靠近奇点的一圈单元: 半径按浮点分组后取最小的一组.
    window = radius < float(ring_edges[-1])
    inner = window & solid
    innermost: dict[str, Any] = {"radius": None, "max": None, "n_cells": 0}
    if inner.any():
        r_inner = radius[inner]
        r_min = r_inner.min()
        same = inner & (radius < r_min * (1.0 + 1.0e-9) + 1.0e-12)
        innermost = {
            "radius": float(r_min),
            "max": float(values[same].max()),
            "n_cells": int(same.sum()),
        }

    # 窗口逐级收窄的 log-log 斜率: 奇异项主导时, 窗口越窄斜率越接近理论指数.
    slopes: dict[str, Any] = {}
    for cut in (2.0, 2.5, 3.0, 3.5, float(ring_edges[-1])):
        sel = inner & (radius > 0.0) & (radius <= cut)
        if int(sel.sum()) < 3:
            slopes[f"r<{cut:g}"] = None
            continue
        # 每个不同半径只取该半径上的最大值, 避免同环多点把拟合拉平.
        r_sel, v_sel = radius[sel], values[sel]
        uniq = np.unique(np.round(r_sel, 9))
        if uniq.size < 3:
            slopes[f"r<{cut:g}"] = None
            continue
        peak = np.array([v_sel[np.isclose(r_sel, u)].max() for u in uniq])
        good = peak > 0.0
        if int(good.sum()) < 3:
            slopes[f"r<{cut:g}"] = None
            continue
        slope = float(np.polyfit(np.log(uniq[good]), np.log(peak[good]), 1)[0])
        slopes[f"r<{cut:g}"] = slope
    return {"rings": rings, "innermost": innermost, "slopes": slopes}


def hot_spot_report(
    barycenter: np.ndarray,
    values: np.ndarray,
    density: np.ndarray,
    points: dict[str, tuple[float, float]],
    h: float,
    solid_threshold: float,
    top_n: int = 5,
    neighbor_factor: float = 1.6,
) -> list[dict[str, Any]]:
    """定位全域最热的若干实体单元, 并测其邻域的密度落差.

    冻结设计探针把粗网格的分片常数密度延拓到细网格, 孔洞与实体的界面因此变成
    阶梯, 每级台阶都是 90 度凹角 —— 那是映射制造出来的人造奇点, 不是原设计的
    性质. 本函数给出区分依据: 单元邻域内的密度落差 ``rho_max - rho_min``.
    落差近 0 说明该单元位于均匀区 (读数可信), 落差近 1 说明它贴着密度界面
    (读数含阶梯成分, 不可用作物理结论).

    Parameters
    ----------
    h : float
        当前网格的 ``hx``; 邻域半径取 ``neighbor_factor * h``, 约为一圈相邻单元.
    top_n : int
        报告前几个最热单元; 取 0 关闭.

    Returns
    -------
    list of dict
        按应力比降序; 每项含坐标、密度、邻域密度落差、到最近奇点的距离与判读.
    """
    if top_n <= 0:
        return []
    from scipy.spatial import cKDTree

    solid = np.flatnonzero(density > solid_threshold)
    if solid.size == 0:
        return []
    ranked = solid[np.argsort(values[solid])[::-1][:top_n]]
    tree = cKDTree(barycenter)
    radius = neighbor_factor * h

    report: list[dict[str, Any]] = []
    for index in ranked:
        neighbors = tree.query_ball_point(barycenter[index], radius)
        local = density[neighbors]
        jump = float(local.max() - local.min())
        distances = {
            name: float(np.hypot(barycenter[index, 0] - c[0],
                                 barycenter[index, 1] - c[1]))
            for name, c in points.items()
        }
        nearest = min(distances, key=distances.get)
        if jump < 0.05:
            verdict = "均匀区"
        elif jump > 0.5:
            verdict = "密度界面"
        else:
            verdict = "过渡带"
        report.append({
            "index": int(index),
            "barycenter": [float(v) for v in barycenter[index]],
            "solid_stress_ratio": float(values[index]),
            "density": float(density[index]),
            "neighbor_radius": float(radius),
            "n_neighbors": int(len(neighbors)),
            "density_min": float(local.min()),
            "density_max": float(local.max()),
            "density_jump": jump,
            "verdict": verdict,
            "nearest_singularity": nearest,
            "distance_to_nearest_singularity": distances[nearest],
        })
    return report


def probe_points(pipeline: Any) -> dict[str, tuple[float, float]]:
    """两类奇点的坐标, 键名带类别前缀便于分区汇总."""
    points: dict[str, tuple[float, float]] = {}
    for name, point in zip(("lower", "upper"), pipeline.problem.clamped_corner_points):
        points[f"corner_{name}"] = (float(point[0]), float(point[1]))
    for name, point in zip(("lower", "upper"), pipeline.problem.traction_patch_endpoints):
        points[f"load_{name}"] = (float(point[0]), float(point[1]))
    return points


# ================================================================== 五、编排


def build_probe_pipeline(
    parameters: dict[str, Any],
    nx: int,
    ny: int,
    method: str = "lfem",
    order: int = 2,
) -> Any:
    """只做装配不求解, 用于取计算域、网格与奇点坐标."""
    run_parameters = {
        **parameters,
        "nx": nx,
        "ny": ny,
        "comparison_orders": [order],
        "load_pad_radius": 0.0,
        "support_pad_radius": 0.0,
    }
    config = build_stress_config(run_parameters)
    return build_stress_analysis_pipeline(config, run_parameters, method, order)


def load_frozen_design(design_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取冻结构型及其运行摘要, 并做可行性与收敛性的前置校验."""
    density_file = design_dir / "density_final.vtu"
    summary_file = design_dir / "summary.json"
    for path in (density_file, summary_file):
        if not path.is_file():
            raise SystemExit(f"缺少 {path}.")
    design = np.asarray(
        read_vtu_cell_data(density_file, "density"), dtype=np.float64
    ).reshape(-1)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not summary.get("converged", False):
        print(
            f"[probe] 警告: {design_dir.name} 的 summary.json 记为未收敛, "
            "在其上做加密探针得到的趋势不可解释为奇异性."
        )
    meta = {
        "run_dir": design_dir.name,
        "density_file": provenance.file_digest(density_file),
        "converged": summary.get("converged"),
        "iterations": summary.get("optimization_iterations"),
        "volume_fraction": summary.get("volume_fraction"),
        "max_constraint": summary.get("max_constraint"),
        "change_outer_max": summary.get("change_outer_max"),
    }
    return design, meta


def run_matrix(
    probe: str,
    matrix: dict[tuple[int, int], tuple[tuple[str, int], ...]],
    design_dir: Path | None,
    ring_edges: Sequence[float],
    solid_threshold: float,
    top_n: int = 5,
) -> dict[str, Any]:
    """按 (网格层, 离散) 矩阵逐格求解并汇总.

    Notes
    -----
    先在基准网格上装配一次 (不求解) 取计算域与粗网格, 冻结探针的密度映射复用
    该粗网格; 均质探针不需要映射, 该次装配只用于取计算域.
    """
    parameters = case_parameters()
    base_nx, base_ny = int(parameters["nx"]), int(parameters["ny"])
    base_pipeline = build_probe_pipeline(parameters, base_nx, base_ny)
    domain = tuple(float(v) for v in base_pipeline.problem.domain)
    coarse_mesh = base_pipeline.mesh

    design: np.ndarray | None = None
    design_meta: dict[str, Any] | None = None
    if probe == "frozen":
        if design_dir is None:
            raise SystemExit("--probe frozen 需要 --design-dir 指定构型来源目录.")
        design, design_meta = load_frozen_design(design_dir)
        n_coarse = int(coarse_mesh.number_of_cells())
        if design.shape[0] != n_coarse:
            raise SystemExit(
                f"构型单元数 {design.shape[0]} 与基准网格 {base_nx}x{base_ny} 的 "
                f"{n_coarse} 不符; 冻结探针要求构型产自 cases.toml 的基准网格."
            )

    results: dict[str, Any] = {}
    for (nx, ny) in sorted(matrix, key=lambda lv: lv[0] * lv[1]):
        mapped: np.ndarray | None = None
        if probe == "frozen":
            if (nx, ny) == (base_nx, base_ny):
                mapped = design
            else:
                print(f"[probe] 映射密度场 {base_nx}x{base_ny} -> {nx}x{ny} ...",
                      flush=True)
                fine_pipeline = build_probe_pipeline(parameters, nx, ny)
                mapped = map_density_to_mesh(
                    design, coarse_mesh, (base_nx, base_ny),
                    fine_pipeline.mesh, domain,
                )
                del fine_pipeline
        for method, order in matrix[(nx, ny)]:
            tag = f"{method}-{order}@{nx}x{ny}"
            print(f"[probe] 求解 {tag} ...", flush=True)
            result = solve_level(parameters, nx, ny, method, order, mapped)
            points = probe_points(result["pipeline"])
            solid = result["density"] > solid_threshold
            entry: dict[str, Any] = {
                "nx": nx,
                "ny": ny,
                "h": (domain[1] - domain[0]) / nx,
                "method": method,
                "order": order,
                "n_cells": int(result["solid_stress_ratio"].shape[0]),
                "estimated_dofs": estimate_dofs(nx, ny, method, order),
                "seconds": result["seconds"],
                "max_solid_stress_ratio_global": float(
                    result["solid_stress_ratio"][solid].max()
                ) if solid.any() else None,
                "points": {},
            }
            entry["hot_spots"] = hot_spot_report(
                result["barycenter"], result["solid_stress_ratio"],
                result["density"], points, entry["h"], solid_threshold, top_n,
            )
            for name, center in points.items():
                entry["points"][name] = ring_statistics(
                    result["barycenter"], result["solid_stress_ratio"],
                    result["density"], center, ring_edges, solid_threshold,
                )
            results[tag] = entry
            print(f"        {result['seconds']:.1f} s, "
                  f"全域最大 {entry['max_solid_stress_ratio_global']:.4f}")
            del result
    return {
        "case_id": CASE_ID,
        "probe": probe,
        "design": design_meta,
        "domain": list(domain),
        "base_level": [base_nx, base_ny],
        "ring_edges": [float(v) for v in ring_edges],
        "solid_threshold": solid_threshold,
        "williams_lambda": WILLIAMS_LAMBDA,
        "predicted_corner_growth_per_doubling": float(2.0 ** (1.0 - WILLIAMS_LAMBDA)),
        "runs": results,
    }


# ================================================================== 六、报表


def _fmt(value: Any, width: int = 9, digits: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.{digits}f}".rjust(width)


def print_report(payload: dict[str, Any]) -> None:
    """打印分环表与逐层增长比表."""
    runs = payload["runs"]
    edges = payload["ring_edges"]
    predicted = payload["predicted_corner_growth_per_doubling"]

    print("\n" + "=" * 78)
    print(f"探针: {payload['probe']}   Williams lambda = {payload['williams_lambda']}")
    print(f"角点理论增长率 (每次网格加倍): x {predicted:.4f}")
    if payload["design"]:
        print(f"冻结构型: {payload['design']['run_dir']}")
    print("=" * 78)

    for name in ("corner_lower", "corner_upper", "load_lower", "load_upper"):
        print(f"\n--- {name} : 各环内实体单元的最大 sigma_vM / sigma_bar ---")
        header = f"{'离散@网格':<22}"
        for lower, upper in zip(edges[:-1], edges[1:]):
            header += f"{f'[{lower:g},{upper:g})':>10}"
        header += f"{'最内圈 r':>10}{'最内圈值':>10}"
        print(header)
        print("-" * len(header))
        for key, entry in runs.items():
            stats = entry["points"][name]
            row = f"{key:<22}"
            for ring in stats["rings"]:
                row += _fmt(ring["max_solid"], 10)
            row += _fmt(stats["innermost"]["radius"], 10, 3)
            row += _fmt(stats["innermost"]["max"], 10)
            print(row)

        print(f"\n    log-log 斜率 (窗口逐级收窄; 角点理论值 {WILLIAMS_LAMBDA - 1.0:+.4f})")
        slope_keys = sorted(
            {k for e in runs.values() for k in e["points"][name]["slopes"]},
            key=lambda s: float(s.split("<")[1]),
        )
        header = f"    {'离散@网格':<22}" + "".join(f"{k:>10}" for k in slope_keys)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for key, entry in runs.items():
            slopes = entry["points"][name]["slopes"]
            row = f"    {key:<22}"
            for sk in slope_keys:
                row += _fmt(slopes.get(sk), 10)
            print(row)

    if any(entry.get("hot_spots") for entry in runs.values()):
        print("\n" + "=" * 78)
        print("全域最热单元定位 (仅实体单元)")
        print("  density_jump = 邻域 (半径 1.6h) 内的 rho_max - rho_min")
        print("  均匀区 = 落差 < 0.05 (读数可信); 密度界面 = 落差 > 0.5 (含阶梯成分)")
        print("=" * 78)
        for key, entry in runs.items():
            spots = entry.get("hot_spots") or []
            if not spots:
                continue
            print(f"\n  {key}   h = {entry['h']:.4f}")
            print(f"    {'排名':<5}{'坐标':>18}{'应力比':>9}{'rho':>7}"
                  f"{'落差':>7}  {'判读':<10}{'最近奇点':<16}{'距离':>7}")
            for rank, spot in enumerate(spots, start=1):
                x, y = spot["barycenter"]
                print(
                    f"    {rank:<5}{f'({x:.3f}, {y:.3f})':>18}"
                    f"{spot['solid_stress_ratio']:>9.4f}{spot['density']:>7.3f}"
                    f"{spot['density_jump']:>7.3f}  {spot['verdict']:<10}"
                    f"{spot['nearest_singularity']:<16}"
                    f"{spot['distance_to_nearest_singularity']:>7.3f}"
                )

    # 同一离散跨网格层的增长比: 判发散的主量.
    #
    # 主量取 innermost (离奇点最近的那圈单元): 结构化网格加密时该圈的半径严格
    # 减半, 故它在各层上是几何相似的探针, 幂律情形下比值应恰为 2^(1-lambda),
    # 与环边界的选取无关. 定半径环 [0, edges[1]) 作为旁证一并报出: 它是固定区域,
    # 最粗层上可能为空 (80x40 时最近单元重心 r=0.745 落在 [0,0.5) 之外).
    print("\n" + "=" * 78)
    print("逐层增长比 (同一离散, 网格加倍后的读数之比)")
    print(f"  主量 innermost = 最近一圈单元的最大值, 其半径随网格严格减半")
    print(f"  旁证 ring[0,{edges[1]:g}) = 固定区域内的最大值, 最粗层可能为空")
    print("=" * 78)
    families: dict[str, list[tuple[int, str]]] = {}
    for key, entry in runs.items():
        families.setdefault(f"{entry['method']}-{entry['order']}", []).append(
            (entry["nx"], key)
        )
    for family, items in families.items():
        items.sort()
        if len(items) < 2:
            print(f"\n  {family}: 只有一个网格层, 无增长比可算.")
            continue
        steps = list(zip(items[:-1], items[1:]))
        print(f"\n  {family}")
        header = f"    {'奇点':<16}{'量':<12}"
        header += "".join(f"{f'{a}->{b}':>14}" for (a, _), (b, _) in steps)
        header += f"{'理论':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for name in ("corner_lower", "corner_upper", "load_lower", "load_upper"):
            theory = predicted if name.startswith("corner") else None
            for quantity in ("innermost", f"ring[0,{edges[1]:g})"):
                row = f"    {name if quantity == 'innermost' else '':<16}{quantity:<12}"
                for (_, k0), (_, k1) in steps:
                    s0 = runs[k0]["points"][name]
                    s1 = runs[k1]["points"][name]
                    if quantity == "innermost":
                        v0, v1 = s0["innermost"]["max"], s1["innermost"]["max"]
                    else:
                        v0 = s0["rings"][0]["max_solid"]
                        v1 = s1["rings"][0]["max_solid"]
                    row += _fmt(v1 / v0 if v0 and v1 else None, 14)
                row += _fmt(theory, 10)
                print(row)
        # 对数型奇点判的是差值不是比值: 每次加倍应增加一个常数.
        print(f"\n    {'(载荷端点为对数型, 下表报差值)':<28}"
              + "".join(f"{f'{a}->{b}':>14}" for (a, _), (b, _) in steps))
        for name in ("load_lower", "load_upper"):
            row = f"    {name:<28}"
            for (_, k0), (_, k1) in steps:
                v0 = runs[k0]["points"][name]["innermost"]["max"]
                v1 = runs[k1]["points"][name]["innermost"]["max"]
                row += _fmt(v1 - v0 if v0 is not None and v1 is not None else None, 14)
            print(row)
    print()


def write_json(payload: dict[str, Any], plan: str) -> Path:
    target = OUTPUT_DIR / CASE_ID / "postprocess" / f"singularity_h_probe_{plan}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


# ================================================================== 七、入口


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="singularity_h_probe.py",
        description="悬臂梁应力算例的应力奇异性网格加密探针 (冻结设计, 零次优化).",
    )
    parser.add_argument(
        "--plan", default=None, choices=sorted(PLANS),
        help="预置方案: A 均质参考 / B 冻结设计 / C 设计无关性抽检.",
    )
    parser.add_argument(
        "--probe", default=None, choices=("homogeneous", "frozen"),
        help="手工模式下的探针类型; 给了 --plan 时由方案决定.",
    )
    parser.add_argument(
        "--design-dir", default=None, metavar="<目录名>",
        help="outputs/<case>/ 下提供构型的运行目录名 (frozen 探针必需).",
    )
    parser.add_argument(
        "--levels", default=None, metavar="80x40,160x80",
        help="逗号分隔的网格层; 手工模式下与 --discretizations 做笛卡尔积.",
    )
    parser.add_argument(
        "--discretizations", default=None, metavar="lfem-2,huzhang-2",
        help="逗号分隔的离散组合.",
    )
    parser.add_argument(
        "--ring-edges", default=None, metavar="0,0.5,1,1.5,2,3,4",
        help=f"定半径环边界 (mm), 默认 {','.join(f'{v:g}' for v in DEFAULT_RING_EDGES)}.",
    )
    parser.add_argument(
        "--solid-threshold", type=float, default=0.5, metavar="<rho>",
        help="判定为实体的密度阈值, 默认 0.5; 均质探针下无影响.",
    )
    parser.add_argument(
        "--hot-spots", type=int, default=5, metavar="<N>",
        help="报告全域最热的前 N 个实体单元及其邻域密度落差, 0 关闭; 默认 5.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印求解矩阵与自由度估算, 不求解.",
    )
    parser.add_argument(
        "--tag", default=None, metavar="<后缀>",
        help="输出文件名后缀, 默认取 --plan 或 custom.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if arguments.plan is not None:
        plan = PLANS[arguments.plan]
        probe = plan["probe"]
        matrix = dict(plan["matrix"])
        design_name = arguments.design_dir or plan["design_dir"]
        tag = arguments.tag or arguments.plan
        print(f"[probe] 方案 {arguments.plan}: {plan['note']}")
    else:
        if arguments.probe is None or arguments.levels is None \
                or arguments.discretizations is None:
            raise SystemExit(
                "手工模式需要同时给出 --probe / --levels / --discretizations, "
                "或改用 --plan A|B|C."
            )
        probe = arguments.probe
        levels = [parse_level(t) for t in arguments.levels.split(",") if t.strip()]
        combos = tuple(
            parse_discretization(t)
            for t in arguments.discretizations.split(",") if t.strip()
        )
        matrix = {level: combos for level in levels}
        design_name = arguments.design_dir
        tag = arguments.tag or "custom"

    ring_edges = (
        tuple(float(v) for v in arguments.ring_edges.split(","))
        if arguments.ring_edges else DEFAULT_RING_EDGES
    )
    design_dir = (
        OUTPUT_DIR / CASE_ID / design_name if design_name else None
    )

    total = sum(len(v) for v in matrix.values())
    print(f"[probe] 探针={probe}  求解格数={total}  构型={design_name or '(均质 rho=1)'}")
    for level in sorted(matrix, key=lambda lv: lv[0] * lv[1]):
        for method, order in matrix[level]:
            dofs = estimate_dofs(level[0], level[1], method, order)
            print(f"        {level[0]:>4}x{level[1]:<4} {method}-{order:<2} "
                  f"约 {dofs:>9,} 自由度")
    if arguments.dry_run:
        print("[probe] --dry-run: 未求解.")
        return 0

    payload = run_matrix(
        probe, matrix, design_dir, ring_edges, float(arguments.solid_threshold),
        int(arguments.hot_spots),
    )
    payload["provenance"] = provenance.run_stamp()
    print_report(payload)
    target = write_json(payload, tag)
    print(f"[probe] {target.relative_to(OUTPUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
