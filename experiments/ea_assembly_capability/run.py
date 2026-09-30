# -*- coding: utf-8 -*-
"""EA 单元级无矩阵算子三个数据点的自包含调度与测量入口.

本脚本同时承担「调度器」与「独立子进程测量器」两个角色, 结构与
``experiments/fa_assembly_capability/run.py`` 一致, 共用代码在 ``experiments/_common/``.

1. 调度模式 (默认入口, 每个数据点独占一个子进程):
   - python run.py --list                                   # 列出已注册数据点
   - python run.py --all --check-only                       # 只打印将执行的子进程命令
   - python run.py --case element-cache --grid 32 --monitor
   - python run.py --case ea-matvec --grid 32 --repeats 20 --monitor
   - python run.py --case ea-continuous --grid 32 --monitor
   - python run.py --case ea-cg-solve --grid 32 --monitor
   - python run.py --case ea-update --grid 32 --monitor               # 标准 EA 的 reassemble (op.update 重新积分)
   - python run.py --case element-cache --mesh tet --p 2 --grid 16    # 换网格与次数, 默认 --mesh hex --p 1
   - python run.py --case cpu-baseline --monitor                # 单核硬件基线 (memcpy 带宽 + dgemm 算力)
   - python run.py --case element-cache-per-element --grid 32 --monitor   # 逐单元参考 EA, 另有 ea-continuous-per-element / ea-update-per-element
   - python run.py --case element-cache-shared --grid 32 --monitor   # 共享参考 EA, 另有 ea-continuous-shared / ea-update-shared
   - python run.py --verify-fa --mesh tet --n 8             # 逐位核对 K_e / K_e^0 / cell2dof 与 fa 构建路径一致 (仅 tet, p = 1)
   - python run.py --verify-shared --mesh hex --p 1 --n 4   # 核对逐单元参考 EA、共享参考 EA 与标准 EA 一致到 1e-11

2. Worker 模式 (由调度器在独立进程中调用, 保证内存高水位严格隔离):
   - python run.py --worker --cache  --method fast --mesh hex --p 1 --n 32 --output outputs/cache_fast_hex_p1_n32.json
   - python run.py --worker --matvec --method fast --mesh hex --p 1 --n 32 --repeats 20 \
         --output outputs/matvec_fast_hex_p1_n32.json
   - python run.py --worker --continuous --method fast --mesh hex --p 1 --n 32 --repeats 20 \
         --output outputs/cache_matvec_continuous_fast_hex_p1_n32.json
   - python run.py --worker --solve  --method fast --mesh hex --p 1 --n 32 --maxiter 5000 --tol 1e-6 \
         --output outputs/solve_fast_hex_p1_n32.json
   - python run.py --worker --baseline --output outputs/baseline_cpu.json
   - python run.py --worker --update --update-mode reassemble --method fast --mesh hex --p 1 --n 32 --rounds 5 \
         --output outputs/update_reassemble_fast_hex_p1_n32.json
   - python run.py --worker --cache --variant shared --method fast --mesh hex --p 1 --n 32 \
         --output outputs/cache_shared_fast_hex_p1_n32.json

被测对象是仓库核心代码 ``soptx.fem.matrix_free.ElasticityEAOperator`` (门面) 及其底层:
``LagrangeFEMAnalyzer.assemble_stiff_matrix('ea')`` 用 ``LinearElasticIntegrator.const`` 缓存 K_e 与
cell2dof 并装进未 assembly 的 ``soptx.fem.BilinearForm``; ``@`` 走 ``BilinearForm.__matmul__``
(gather -> einsum -> index_add) 外包 ``DirichletBCOperator`` (Pi_I K Pi_I + Pi_D); Jacobi-PCG 用
``soptx.solvers.cg`` 与 ``DiagonalPreconditioner``, 对角由 ``assemble_operator_diagonal`` 给出.
本脚本不含任何算子或求解器的自有实现.

EA 变体 (``--variant``) 三种, 常驻量不同, 是同一个离散算子:

- standard: 标准 EA ``ElementAssembly``, 常驻 K_e. 门面的分析器不做拓扑优化 (系数为 None), 其 'ea' 路径
  即标准 EA; ``update`` 按新系数重新积分.
- per_element: 逐单元参考 EA, 即 N_k = NC 的 ``SharedReferenceElementAssembly``, 常驻 K_e^0 与 s_e. 这是
  分析器在单元密度拓扑优化下 'ea' 路径的形式, ``update`` 只换 s_e, 对网格没有要求.
- shared: 共享参考 EA, 即 N_k < NC 的同一个类, 每个平移类常驻一份 K_k^0, 依赖 ``from_box`` 编号约定.

后两者绕开门面由 ``_setup_reference`` 显式构造 (K_e^0 或代表单元的 K_k^0 + ``ElementRestriction`` + s_e = 1),
只支持 cache / continuous / update 三个面板; matvec / solve 需要门面, 只跑标准 EA. 三者是否为同一离散算子
由 ``--verify-shared`` 在同一进程内核对.

问题、网格、空间、材料由 ``_common.fe_problem.build_problem_space`` 构建: ``--mesh`` 取 tri / quad /
tet / hex (默认 hex), ``--p`` 为空间次数 (默认 1), 均为 ``from_box`` 结构化网格. 积分参数
q 由 ``_common.fe_problem.integration_order`` 给出 (单纯形 p + 3, 张量积单元 p + 1, 后者为每方向点数),
经 ``ElasticityEAOperator(integration_order=q)`` 传入分析器. tet 且 p = 1 时构建路径与 fa 完全相同,
q = 4 也与 fa 直接调用 ``LinearElasticIntegrator`` 的默认阶 ``p + 3`` 相同, 阶段 1 的 K_e 与 fa 阶段 1
是否逐位一致由 ``--verify-fa`` 在同一进程内用 ``np.array_equal`` 核对, 不靠代码同源推断.

内存口径 (CPU): 每个阶段先记 before = 当前 VmRSS, 再向 /proc/self/clear_refs 写 5 重置 VmHWM,
阶段结束读 VmHWM 作为该阶段的绝对峰值 peak, net = peak - before. 全程峰值 (process_max_rss)
= 各阶段峰值的最大值.

阶段划分:
  cache  面板: mesh (网格 + 空间 + 材料 + 分析器) -> cache (assemble_stiff_matrix: K_e + cell2dof)
  matvec 面板: mesh -> assemble (facade.assemble(): K_e + 体力右端 + Dirichlet 投影) -> warmup
               -> matvec (刚度算子乘 K x = operator.form @ x, 重复 repeats 次)
  solve  面板: mesh -> assemble -> setup_solve (对角 + 预条件子) -> solve (cg, 每步调用 facade @ x = (P_I K P_I + P_D) x)
  update 面板: mesh -> setup (s_e = 1) -> update_first -> update_rest, 每轮 op.update(rho) 换单元密度;
               标准 EA 为 reassemble (重新积分), 逐单元参考 EA 与共享参考 EA 为 rescale (只换 s_e), 见 ``measure_update``
  baseline 面板: 与网格无关, 单线程 (cases.toml 的 env 限制) memcpy 带宽与 dgemm 算力, 供阶段 2 换算占比
matvec / solve 面板不单列 cache 阶段: ``ElasticityEAOperator.assemble()`` 内部会再次调用
``assemble_stiff_matrix``, 单列会把 K_e 算两遍, 阶段 1 的数字以 cache 面板为准.

产物命名: <kind>_<method>_<mesh>_p<P>_n<N>.json (update 为 update_<mode>_<method>_<mesh>_p<P>_n<N>.json);
逐单元参考 EA 与共享参考 EA 在 method 段前多一个变体段, 如 cache_per_element_fast_hex_p1_n32.json、
update_rescale_shared_fast_hex_p1_n32.json. 不含 mesh / p 段的旧产物是 tet、p = 1; 旧的 update_scale / update_inplace
产物来自已删除的"缩放 K_e"写法, 不再生成. 后处理与对比表见 compare.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
for _p in (_THIS_DIR, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from _common import scheduler  # noqa: E402
from _common.fe_problem import (  # noqa: E402
    DEFAULT_MESH,
    MESH_SPECS,
    MESH_TYPE,
    METHOD_NAMES,
    PROBLEM_NAME,
    build_problem_space,
    integration_order,
    import_fe_stack_cpu,
    mesh_facts,
)
from _common.baseline import measure_baseline, print_baseline  # noqa: E402
from _common.metrology import StageMeter, cur_rss_kib, peak_rss_kib, reset_peak_rss  # noqa: E402

# -----------------------------------------------------------------------------
# 1. 核心算子的构建与测量 (Worker 核心)
# -----------------------------------------------------------------------------

DEVICE = "cpu"

# 本进程测量的网格类型、空间次数与 EA 变体, 由 main 按 --mesh / --p / --variant 设置 (见 _configure)
MESH = DEFAULT_MESH
P = 1
VARIANT = "standard"

# standard: 门面分析器 'ea' 路径的 ElementAssembly, 常驻 K_e;
# per_element: 显式构造的 SharedReferenceElementAssembly (N_k = NC), 常驻 K_e^0 与 s_e;
# shared: 显式构造的 SharedReferenceElementAssembly (N_k < NC), 常驻 K_k^0 与 s_e
VARIANTS = ("standard", "per_element", "shared")


def _configure(mesh: str, p: int, variant: str = "standard") -> None:
    """设置本进程测量的网格类型、空间次数与 EA 变体."""
    global MESH, P, VARIANT
    if mesh not in MESH_SPECS:
        raise ValueError(f"未知的网格类型 {mesh!r}, 可选 {tuple(MESH_SPECS)}")
    if p < 1:
        raise ValueError(f"p 须为正整数, 得到 {p}")
    if variant not in VARIANTS:
        raise ValueError(f"未知的 EA 变体 {variant!r}, 可选 {VARIANTS}")
    MESH, P, VARIANT = mesh, p, variant


def _mesh_fields() -> Dict[str, Any]:
    """产物中标识问题、离散与 EA 变体的公共字段."""
    spec = MESH_SPECS[MESH]
    return {
        "problem": spec.problem_name,
        "mesh_type": spec.mesh_class,
        "mesh": MESH,
        "GD": spec.GD,
        "p": P,
        "q": integration_order(MESH, P),
        "variant": VARIANT,
    }


def _flops_per_cell(Ke: np.ndarray) -> int:
    """y_e = K_e x_e 的乘加次数 2 LDOF^2, LDOF 取自 K_e 的形状 (NC, LDOF, LDOF)."""
    return 2 * int(Ke.shape[-1]) ** 2


def _element_data(facade: Any) -> tuple[np.ndarray, np.ndarray]:
    """从分析器持有的 EA 算子取出常驻的 K_e (NC, LDOF, LDOF) 与 cell2dof (NC, LDOF).

    ``assemble_stiff_matrix('ea')`` 构造的 ``ElementAssembly`` 即 ``analyzer.assembly_level``,
    K_e 在其 ``element_matrices``, cell2dof 在其单元限制 ``restriction`` 中; 本脚本只读不写.
    """
    ea = facade.analyzer.assembly_level
    if ea is None:
        raise RuntimeError("K_e 尚未缓存: 需先调用 assemble_stiff_matrix() 或 assemble()")
    return np.asarray(ea.element_matrices), np.asarray(ea.restriction.cell2dof)


def _reference_kx(Ke: np.ndarray, cell2dof: np.ndarray, x: np.ndarray) -> np.ndarray:
    """核对用的纯 numpy 参考 y = sum_e G_e^T K_e G_e x (assembly-levels.md §2.3), 不计时."""
    y = np.zeros(x.shape[0], dtype=x.dtype)
    np.add.at(y, cell2dof.ravel(), np.einsum("cij,cj->ci", Ke, x[cell2dof]).ravel())
    return y


def _reference_shared_kx(K0: np.ndarray, scale: np.ndarray, cell2dof: np.ndarray, x: np.ndarray) -> np.ndarray:
    """核对逐单元参考 EA 与共享参考 EA 用的纯 numpy 参考 y = sum_k sum_{e in C_k} s_e G_e^T K_k^0 G_e x, 不计时.

    N_k = NC 时逐单元直接作用 s_e (K_e^0 x_e). N_k < NC 时按类循环, 第 k 类取单元 e = k, k + N_k, ...
    (from_box 编号约定 k(e) = e mod N_k), 不复用算子按 (格子, 类) 重排的写法; 两种都不展开 s_e K_e^0.
    """
    num_classes = K0.shape[0]
    y = np.zeros(x.shape[0], dtype=x.dtype)
    if num_classes == cell2dof.shape[0]:
        y_e = scale[:, None] * np.einsum("cij,cj->ci", K0, x[cell2dof])
        np.add.at(y, cell2dof.ravel(), y_e.ravel())
        return y
    for k in range(num_classes):
        c2d = cell2dof[k::num_classes]
        y_e = scale[k::num_classes, None] * (x[c2d] @ K0[k].T)
        np.add.at(y, c2d.ravel(), y_e.ravel())
    return y


def _relerr(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_ref)) / max(float(np.max(np.abs(y_ref))), 1e-300))


def _build_facade(method: str, n: int) -> tuple[Dict[str, Any], StageMeter]:
    """构建网格 / 空间 / 材料与 ``ElasticityEAOperator`` 门面 (mesh 阶段), 不触发装配.

    Parameters
    ----------
    method : str
        单刚组装方式 (standard/voigt/fast), 透传给门面的 ``assembly_method``.
    n : int
        网格每方向段数.

    Returns
    -------
    ctx : dict
        含 mesh / vs / problem / material / facade / facts.
    meter : StageMeter
        已记录 mesh 阶段.
    """
    import_fe_stack_cpu()
    from soptx.fem.matrix_free import ElasticityEAOperator

    meter = StageMeter()
    with meter.stage("mesh"):
        problem, mesh, vs, material = build_problem_space(n, mesh_type=MESH, p=P)
        facade = ElasticityEAOperator(vs, problem, material, degree=P, assembly_method=method,
                                      integration_order=integration_order(MESH, P))

    ctx: Dict[str, Any] = {
        "mesh": mesh,
        "vs": vs,
        "problem": problem,
        "material": material,
        "facade": facade,
        "facts": mesh_facts(mesh, vs),
    }
    return ctx, meter


def _num_classes() -> int:
    """当前网格类型的平移类数 N_k (每个格子剖出的单元数), 由单格子的 ``create_box_mesh`` 读出."""
    from soptx.mesh import create_box_mesh

    GD = MESH_SPECS[MESH].GD
    return int(create_box_mesh(MESH, [0.0, 1.0] * GD, *(1, ) * GD).classes.num_classes)


def _build_reference(method: str, n: int) -> tuple[Dict[str, Any], StageMeter]:
    """构建网格 / 空间 / 材料 (mesh 阶段), 供逐单元参考 EA 与共享参考 EA 显式构造; 不建分析器与门面.

    网格与标准 EA 同由 ``build_problem_space`` 的 FEALPy ``from_box`` 生成, 未重编号, 满足共享参考
    EA 的类归属约定 k(e) = e mod N_k; 逐单元参考 EA 取 N_k = NC, 不依赖这一约定.

    Parameters
    ----------
    method : str
        参考单元矩阵的积分方式 (standard/voigt/fast), 透传给 ``LinearElasticIntegrator``.
    n : int
        网格每方向段数.

    Returns
    -------
    ctx : dict
        含 mesh / vs / problem / material / method / num_classes / facts.
    meter : StageMeter
        已记录 mesh 阶段.
    """
    import_fe_stack_cpu()

    meter = StageMeter()
    with meter.stage("mesh"):
        problem, mesh, vs, material = build_problem_space(n, mesh_type=MESH, p=P)

    facts = mesh_facts(mesh, vs)
    ctx: Dict[str, Any] = {
        "mesh": mesh,
        "vs": vs,
        "problem": problem,
        "material": material,
        "method": method,
        "num_classes": int(facts["NC"]) if VARIANT == "per_element" else _num_classes(),
        "facts": facts,
    }
    return ctx, meter


def _setup_reference(ctx: Dict[str, Any]) -> Any:
    """逐单元参考 EA 与共享参考 EA 的 setup: 积分参考单元矩阵, 再构造 G 与算子, s_e 取 1.

    ``ctx["num_classes"]`` 等于 NC 时不带 index 对全部单元积分, 参考即 K_e^0 (与标准 EA 在 s_e = 1 时的
    K_e 同一次调用); 小于 NC 时只对第一个格子里的代表单元 0, ..., N_k - 1 积分出 K_k^0. 积分阶与标准 EA
    相同, 均为 ``integration_order``. G 与标准 EA 同由 ``ElementRestriction.from_integrator(..., layout='flat')``
    构造.
    """
    from soptx.fem.integrators import LinearElasticIntegrator
    from soptx.fem.kernels import ElementRestriction
    from soptx.fem.levels import SharedReferenceElementAssembly

    vs, material = ctx["vs"], ctx["material"]
    q = integration_order(MESH, P)
    num_classes = int(ctx["num_classes"])
    kwargs = {} if num_classes == int(vs.mesh.number_of_cells()) else {"index": np.arange(num_classes)}
    K0 = LinearElasticIntegrator(material, coef=None, q=q, method=ctx["method"], **kwargs).assembly(vs)
    restriction = ElementRestriction.from_integrator(LinearElasticIntegrator(material, q=q), vs, layout="flat")
    return SharedReferenceElementAssembly(vs, restriction, K0)


def _reference_data(op: Any) -> Dict[str, Any]:
    """逐单元参考 EA 与共享参考 EA 常驻的 K_k^0 (N_k, LDOF, LDOF)、s_e (NC, ) 与 cell2dof (NC, LDOF); 只读不写."""
    return {
        "op": op,
        "K0": np.asarray(op.reference_matrices),
        "scale": np.asarray(op.scale),
        "cell2dof": np.asarray(op.restriction.cell2dof),
    }


def _build(method: str, n: int) -> tuple[Dict[str, Any], StageMeter]:
    """按 VARIANT 构建 mesh 阶段: standard 建门面, per_element / shared 只建网格与空间."""
    return _build_facade(method, n) if VARIANT == "standard" else _build_reference(method, n)


def _setup_level(ctx: Dict[str, Any]) -> Any:
    """按 VARIANT 做 setup 并返回 EA 算子: standard 为分析器的 ``ElementAssembly``, per_element / shared 为
    显式构造的 ``SharedReferenceElementAssembly``."""
    if VARIANT == "standard":
        return ctx["facade"].analyzer.assemble_stiff_matrix()
    return _setup_reference(ctx)


def _reference_fields(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """逐单元参考 EA 与共享参考 EA 产物的专有字段: 类数 N_k 与 K_k^0、s_e 的形状和字节数."""
    K0, scale = ctx["K0"], ctx["scale"]
    return {
        "num_classes": int(ctx["num_classes"]),
        "K0_shape": [int(s) for s in K0.shape],
        "K0_MiB": round(int(K0.nbytes) / 2**20, 3),
        "scale_MiB": round(int(scale.nbytes) / 2**20, 1),
    }


def _require_standard(panel: str) -> None:
    if VARIANT != "standard":
        raise ValueError(f"{panel} 面板需要分析器与门面, 只支持 --variant standard, 得到 {VARIANT!r}")


def _assemble_system(ctx: Dict[str, Any], meter: StageMeter) -> None:
    """assemble 阶段: ``facade.assemble()`` 一次给出 K_e 缓存、体力右端与 Dirichlet 投影算子."""
    facade = ctx["facade"]
    with meter.stage("assemble"):
        operator, load = facade.assemble()
    Ke, cell2dof = _element_data(facade)
    ctx.update(
        {
            "operator": operator,  # DirichletBCOperator, 即 facade.system_operator
            "load": np.asarray(load),
            "Ke": Ke,
            "cell2dof": cell2dof,
            "is_bd": np.asarray(facade.boundary_dofs, dtype=bool),
        }
    )


def _finish(panel: str, ctx: Dict[str, Any], method: str, n: int, meter: StageMeter) -> Dict[str, Any]:
    """在全部阶段结束后组装公共字段 (网格事实、K_e 理论量、算子常驻、各阶段峰值 / 净增与单价).

    K_e 理论量各变体同口径, 都是 (NC, LDOF, LDOF) 的 float64 字节数, 即标准 EA 常驻的 K_e; per_element /
    shared 并不常驻 K_e, 其 ``Ke_shape`` 只用于给出这一理论量, 实际常驻见 ``K0_shape`` / ``operator_persistent_MiB``.
    """
    facts = ctx["facts"]
    Ndof = facts["Ndof"]
    c2d = ctx["cell2dof"]
    if VARIANT != "standard":
        K0 = ctx["K0"]
        ldof = int(K0.shape[-1])
        ke_shape = [int(facts["NC"]), ldof, ldof]
        persistent = int(ctx["op"].persistent_bytes())
        operator_impl = "soptx.fem.levels.SharedReferenceElementAssembly"
    else:
        ke_shape = [int(s) for s in ctx["Ke"].shape]
        persistent = int(ctx["Ke"].nbytes + c2d.nbytes)
        operator_impl = "soptx.fem.matrix_free.ElasticityEAOperator"
    ke_theory = int(np.prod(ke_shape)) * 8

    def kb_per_dof(nbytes: float) -> float:
        return round(nbytes / Ndof / 1000, 2)

    out: Dict[str, Any] = {
        "panel": panel,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": meter.memory_kind,
        **_mesh_fields(),
        "operator_impl": operator_impl,
        "method": method,
        "n": n,
        **facts,
        "Ke_shape": ke_shape,
        "Ke_theory_MiB": round(ke_theory / 2**20, 1),
        "cell2dof_MiB": round(int(c2d.nbytes) / 2**20, 1),
        **(_reference_fields(ctx) if VARIANT != "standard" else {}),
        "operator_persistent_MiB": round(persistent / 2**20, 1),
        "operator_persistent_KB_per_dof": kb_per_dof(persistent),
        **meter.fields(),
        "process_max_rss_KB_per_dof": kb_per_dof(meter.max_peak_kib() * 1024),
    }
    if "cache" in meter.records:
        out.update(
            {
                "cache_KB_per_dof": kb_per_dof(meter.net_bytes("cache")),
                "cache_peak_KB_per_dof": kb_per_dof(meter.peak_bytes("cache")),
                "cache_net_over_Ke_theory": round(meter.net_bytes("cache") / ke_theory, 2),
            }
        )
    return out


def _malloc_trim() -> bool:
    """把 glibc 持有但未归还内核的空闲页归还; 非 glibc 平台返回 False."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
        return True
    except (OSError, AttributeError):
        return False


def measure_cache(method: str, n: int) -> dict:
    """阶段 1 (panel cache): 只测 ``assemble_stiff_matrix('ea')`` 缓存 K_e 与 cell2dof (与 fa 阶段 1 同口径).

    常驻量与 fa ``measure_stage1`` 同一套动作: 阶段开始前先 ``gc`` 并归还建网格留下的 glibc 空闲页
    (否则它们被本阶段大块临时量占用后随 munmap 一起还给内核, 使"结束 RSS - 起点 RSS"不闭合),
    阶段结束后再取一次 RSS 增量, ``malloc_trim`` 前后各记一个值. EA 比 fa 多常驻一个 cell2dof.

    ``--variant per_element`` / ``shared`` 时 cache 阶段换成 ``_setup_reference`` (积分 K_e^0 或代表单元的
    K_k^0 + G + s_e), 其余动作相同; 常驻为参考单元矩阵 + s_e + cell2dof.
    """
    import gc

    from _common.metrology import cur_rss_kib

    ctx, meter = _build(method, n)
    rss_before_trim_kib = cur_rss_kib()
    gc.collect()
    trimmed = _malloc_trim()
    with meter.stage("cache"):
        op = _setup_level(ctx)
    if VARIANT != "standard":
        ctx.update(_reference_data(op))
    else:
        del op  # 分析器持有同一个 ElementAssembly
        Ke, cell2dof = _element_data(ctx["facade"])
        ctx.update({"Ke": Ke, "cell2dof": cell2dof})

    gc.collect()
    base_kib = meter.records["cache"].before_kib
    retained_kib = max(0, cur_rss_kib() - base_kib)
    if trimmed:
        _malloc_trim()
    after_kib = max(0, cur_rss_kib() - base_kib) if trimmed else retained_kib

    out = _finish("cache", ctx, method, n, meter)
    Ndof = out["Ndof"]
    out.update(
        {
            "cache_before_no_trim_MiB": round(rss_before_trim_kib / 1024, 1),
            "malloc_trim_supported": trimmed,
            "cache_retained_MiB": round(retained_kib / 1024, 1),
            "cache_retained_KB_per_dof": round(retained_kib * 1024 / Ndof / 1000, 2),
            "cache_retained_after_trim_MiB": round(after_kib / 1024, 1),
        }
    )
    return out


def measure_matvec_allocations(method: str, n: int) -> dict:
    """独立跟踪一次预热后算子乘的分配, 不测性能耗时.

    Parameters
    ----------
    method : str
        单刚组装方式.
    n : int
        网格每方向段数.

    Returns
    -------
    dict
        NumPy 跟踪校验, 可跟踪分配峰值及保留量, 探针 RSS.
        未接入 tracemalloc 的底层分配不在跟踪范围内.
    """
    import gc
    import tracemalloc

    from _common.metrology import cur_rss_kib, peak_rss_kib, reset_peak_rss

    if tracemalloc.is_tracing():
        raise RuntimeError("请在未启用 tracemalloc 的独立进程中运行探针")
    _require_standard("matvec")
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    bform = ctx["operator"].form
    x = np.random.default_rng(0).standard_normal(ctx["facts"]["Ndof"])
    warmup = bform @ x
    del warmup
    gc.collect()

    # 已知大小的 NumPy 分配校验不计入算子乘.
    tracemalloc.start()
    try:
        check_before, _ = tracemalloc.get_traced_memory()
        check = np.empty(2**20, dtype=np.float64)
        check_current, _ = tracemalloc.get_traced_memory()
        expected = int(check.nbytes)
        observed = check_current - check_before
        check_ok = expected <= observed <= expected + 64 * 1024
        del check
    finally:
        tracemalloc.stop()
    if not check_ok:
        raise RuntimeError(f"NumPy 跟踪校验失败: 期望 {expected} B, 捕获 {observed} B")

    gc.collect()
    rss_before = cur_rss_kib()
    rss_reset = reset_peak_rss()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        y = bform @ x
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = cur_rss_kib()
    rss_peak = peak_rss_kib()
    return {
        "panel": "matvec_allocations",
        "method": method,
        "n": n,
        **ctx["facts"],
        "numpy_tracking_check_passed": check_ok,
        "numpy_tracking_expected_bytes": expected,
        "numpy_tracking_observed_bytes": observed,
        "allocation_peak_increment_bytes": peak - before,
        "allocation_retained_increment_bytes": current - before,
        "allocation_peak_minus_retained_bytes": peak - current,
        "output_bytes": int(y.nbytes),
        "probe_rss_before_MiB": rss_before / 1024,
        "probe_rss_after_MiB": rss_after / 1024,
        "probe_rss_peak_MiB": rss_peak / 1024 if rss_reset else None,
        "probe_rss_peak_increment_MiB": max(0, rss_peak - rss_before) / 1024 if rss_reset else None,
        "probe_rss_reset_supported": rss_reset,
        "allocation_scope": "一次预热后的 form @ x; 包含输出; 非累计分配量; 不含未接入跟踪的底层分配",
        "rss_scope": "探针 RSS 含跟踪器开销, 不用于无跟踪器性能结论",
    }


def _timed(fn: Any, repeats: int) -> tuple[list[float], Any]:
    times: list[float] = []
    y = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = fn()
        times.append(time.perf_counter() - t0)
    return times, y


def measure_matvec(method: str, n: int, repeats: int = 20, seed: int = 0) -> dict:
    """阶段 2 (panel matvec): 预热后重复 ``repeats`` 次刚度算子乘 K x, 记录中位 / 最小耗时与有效带宽下界.

    计时对象是 ``operator.form @ x``, 即 ``soptx.fem.BilinearForm.__matmul__``: gather ``x[cell2dof]`` ->
    ``einsum("cij, cj -> ci", K_e, x_e)`` -> ``index_add`` scatter-add, 对应 assembly-levels.md §2.3 的
    EA MatVec, 与 fa 的 CSR ``K @ x`` 同口径. 含 Dirichlet 投影的系统算子乘 ``facade @ x`` 只在阶段 3
    由 cg 调用, 不在本阶段单独计时. 结果与纯 numpy 参考实现核对 (相对误差).

    带宽下界按每次算子乘至少搬运 K_e + cell2dof + 读 x 写 y (16 B/dof) 计, 不含 gather 与
    scatter 的随机访问放大, 因此是真实带宽的下界.
    """
    _require_standard("matvec")
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    bform = ctx["operator"].form
    Ke, cell2dof = ctx["Ke"], ctx["cell2dof"]
    Ndof = ctx["facts"]["Ndof"]
    NC = ctx["facts"]["NC"]

    x = np.random.default_rng(seed).standard_normal(Ndof)

    with meter.stage("warmup"):
        y = bform @ x

    with meter.stage("matvec"):
        times, y = _timed(lambda: bform @ x, repeats)

    y = np.asarray(y)
    y_ref = _reference_kx(Ke, cell2dof, x)

    t_med = statistics.median(times)
    bytes_moved_min = Ke.nbytes + cell2dof.nbytes + 16 * Ndof

    out = _finish("matvec", ctx, method, n, meter)
    out.update(
        {
            "repeats": repeats,
            "seed": seed,
            "matvec_impl": "soptx.fem.BilinearForm.__matmul__ (inherited from fealpy): gather -> einsum -> index_add",
            "matvec_seconds_median": round(t_med, 6),
            "matvec_seconds_min": round(min(times), 6),
            "matvec_seconds_all": [round(t, 6) for t in times],
            "bytes_moved_min_per_matvec": int(bytes_moved_min),
            "effective_gbps_lower_bound": round(bytes_moved_min / t_med / 1e9, 2),
            "gflops": round(_flops_per_cell(Ke) * NC / t_med / 1e9, 2),
            "y_norm": float(np.linalg.norm(y)),
            "matvec_vs_reference_relerr": _relerr(y, y_ref),
        }
    )
    return out


def measure_solve(method: str, n: int, maxiter: int = 5000, tol: float = 1e-6) -> dict:
    """阶段 3 (panel solve): 核心 Jacobi-PCG (``soptx.solvers.cg`` + ``DiagonalPreconditioner``) 求解制造解问题.

    系统 A = Pi_I K Pi_I + Pi_D 与右端来自 ``facade.assemble()`` (体力 + Dirichlet 消去), 初值取
    ``prescribed_solution`` (边界为给定位移、内部为零), 对角由 ``assemble_operator_diagonal`` 给出
    (Dirichlet 处为 1). 停机判据为 cg 的 'natural' 口径 ||r_k||_{M^-1} <= tol ||r_0||_{M^-1};
    另报告求解后的真残差 ||b - A x|| / ||b||.
    """
    from soptx.solvers import DiagonalPreconditioner, cg

    _require_standard("solve")
    ctx, meter = _build_facade(method, n)
    _assemble_system(ctx, meter)
    facade = ctx["facade"]
    operator = ctx["operator"]
    load = ctx["load"]

    with meter.stage("setup_solve"):
        diag = facade.analyzer.assemble_operator_diagonal(operator)
        precond = DiagonalPreconditioner(diag)
        x0 = facade.prescribed_solution

    with meter.stage("solve"):
        x, info = cg(
            operator, load, x0, precond,
            atol=0.0, rtol=tol, maxit=maxiter, returninfo=True, print_level=0,
        )

    it_count = int(info["niter"])
    residual = float(info["residual"])
    reference = float(info["reference_norm"])
    true_res = float(np.linalg.norm(np.asarray(operator @ x) - load))
    load_norm = float(np.linalg.norm(load))
    solve_s = meter.seconds("solve")
    reason = info.get("reason")

    out = _finish("solve", ctx, method, n, meter)
    out.update(
        {
            "solver_impl": "soptx.solvers.cg + DiagonalPreconditioner",
            "preconditioner": "jacobi",
            "boundary": "pde-dirichlet",
            "rhs": "pde-body-force",
            "tolerance": tol,
            "maxiter": maxiter,
            "iterations": it_count,
            "converged": bool(info["converged"]),
            "reason": str(getattr(reason, "name", reason)),
            "final_relres": residual / reference if reference > 0 else float("nan"),
            "true_relres": true_res / load_norm if load_norm > 0 else float("nan"),
            "iterations_per_n": round(it_count / n, 3),
            "solve_seconds": round(solve_s, 3),
            "seconds_per_iteration": round(solve_s / it_count, 6) if it_count else 0.0,
            "n_boundary_dofs": int(ctx["is_bd"].sum()),
        }
    )
    return out


# 各变体的 update 写法: 标准 EA 按新系数重新积分 K_e, 逐单元参考 EA 与共享参考 EA 只换 s_e
UPDATE_MODES = {"standard": ("reassemble", ), "per_element": ("rescale", ), "shared": ("rescale", )}
ALL_UPDATE_MODES = ("reassemble", "rescale")


def update_modes(variant: str) -> tuple[str, ...]:
    """变体可用的 update 写法."""
    return UPDATE_MODES[variant]


def measure_update(method: str, n: int, mode: str, rounds: int = 5) -> dict:
    """update 面板: 同一进程内测量 setup -> update_first -> update_rest.

    单元密度 rho_e (NC, ) 下每轮调用 ``op.update(rho)``, 每个变体一个独立进程:

    - reassemble (standard): ``ElementAssembly.update`` 把 rho 设为积分子系数后重新积分, 替换 K_e; 新旧 K_e
      在替换完成前同时常驻. 门面的分析器不做拓扑优化, ``assemble_stiff_matrix(rho_val)`` 会忽略密度, 故直接
      调用层级的 ``update``.
    - rescale (per_element / shared): ``SharedReferenceElementAssembly.update`` 只把 s_e 换成 rho 的副本,
      参考单元矩阵与 G 不动.

    Parameters
    ----------
    method : 单刚组装方式, 透传给门面 (standard) 或参考单元矩阵的积分 (per_element / shared).
    n : 网格每方向段数.
    mode : update 写法, 取 ``update_modes(VARIANT)`` 之一.
    rounds : update 轮数, 第 1 轮单列为 update_first, 其余计入 update_rest.

    Returns
    -------
    result : 各阶段 before / peak / after / net 与逐轮耗时.

    Notes
    -----
    setup 时系数为 None (s_e = 1). 正确性核对只取前 ``n_check`` 个单元的矩阵: setup 后另存这些单元的 K_e
    (standard) 或参考单元矩阵 (per_element / shared) 的小副本, 约 1 MiB, 不影响水位. standard 每轮核对
    ``element_matrices[:n_check]`` 与 ``rho * K_e^0`` 一致; per_element / shared 每轮核对全部 s_e 与 rho 一致,
    且参考单元矩阵仍是 setup 时的同一数组、前 ``n_check`` 份数值未变. 每轮的 rho 在计时区间外生成.
    """
    import gc

    valid = update_modes(VARIANT)
    if mode not in valid:
        raise ValueError(f"--variant {VARIANT} 的 mode 必须是 {valid} 之一, 得到 {mode!r}")
    if rounds < 2:
        raise ValueError("rounds 至少为 2, 以区分首轮与稳态")

    standard = VARIANT == "standard"
    ctx, mesh_meter = _build(method, n)
    stages: Dict[str, Dict[str, Any]] = {}
    rng = np.random.default_rng(0)
    times = [0.0] * rounds
    max_error = 0.0

    @contextlib.contextmanager
    def stage(name: str):
        before = cur_rss_kib()
        reset = reset_peak_rss()
        start = time.perf_counter()
        yield
        seconds = time.perf_counter() - start
        after = cur_rss_kib()
        peak = max(before, after, peak_rss_kib())
        stages[name] = {
            "before_kib": before,
            "peak_kib": peak,
            "after_kib": after,
            "net_kib": peak - before,
            "t_s": seconds,
            "reset_supported": reset,
        }

    gc.collect()
    trim_supported = _malloc_trim()
    with stage("setup"):
        ea = _setup_level(ctx)

    NC = int(ctx["facts"]["NC"])
    n_check = min(NC, 1000)
    if standard:
        K0_check = np.array(ea.element_matrices[:n_check], copy=True)
    else:
        K0 = ea.reference_matrices
        K0_check = np.array(K0[:n_check], copy=True)

    def one_round(i: int) -> None:
        nonlocal max_error
        rho = rng.uniform(1e-3, 1.0, NC)
        start = time.perf_counter()
        ea.update(rho)
        times[i] = time.perf_counter() - start
        if standard:
            ref = rho[:n_check, None, None] * K0_check
            err = float(np.max(np.abs(np.asarray(ea.element_matrices[:n_check]) - ref)) / np.max(np.abs(ref)))
        else:
            if ea.reference_matrices is not K0 or not np.array_equal(np.asarray(K0[:n_check]), K0_check):
                raise RuntimeError("rescale 改动了参考单元矩阵")
            err = float(np.max(np.abs(np.asarray(ea.scale) - rho)) / np.max(np.abs(rho)))
        max_error = max(max_error, err)

    with stage("update_first"):
        one_round(0)
    with stage("update_rest"):
        for i in range(1, rounds):
            one_round(i)

    facts = ctx["facts"]
    c2d_bytes = int(ea.restriction.persistent_bytes())
    if standard:
        variant_fields = {"Ke_MiB": round(int(ea.element_matrices.nbytes) / 2**20, 1)}
        scope = "同一进程: setup -> update_first -> update_rest; 每轮 op.update(rho) 重新积分 K_e; 单元密度 U(1e-3, 1); 阶段间不额外 gc/trim"
    else:
        ctx.update(_reference_data(ea))
        variant_fields = {**_reference_fields(ctx), "operator_persistent_MiB": round(ea.persistent_bytes() / 2**20, 1)}
        scope = "同一进程: setup -> update_first -> update_rest; 每轮 op.update(rho) 只换 s_e; 单元密度 U(1e-3, 1); 阶段间不额外 gc/trim"
    peak = max(mesh_meter.max_peak_kib(), *(r["peak_kib"] for r in stages.values()))
    result = {
        "panel": "update",
        "mode": mode,
        **_mesh_fields(),
        "n": n,
        **facts,
        "method": method,
        "operator_impl": type(ea).__module__ + "." + type(ea).__name__,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "rounds": rounds,
        "trim_before_setup_supported": trim_supported,
        "mesh_fields": mesh_meter.fields(),
        "stages": stages,
        "process_peak_kib": peak,
        **variant_fields,
        "cell2dof_MiB": round(c2d_bytes / 2**20, 1),
        "update_seconds_first": times[0],
        "update_seconds_rest_median": statistics.median(times[1:]),
        "update_seconds_all": times,
        "check_cells": n_check,
        "check_relerr_max": max_error,
    }
    if max_error > 1e-12 or not all(r["reset_supported"] for r in stages.values()):
        raise RuntimeError("update 测量核对失败, 请检查原始记录")
    return result


def measure_continuous(method: str, n: int, repeats: int = 20) -> dict:
    """连续测量面板: 同一进程内测量 cache -> input -> first_matvec -> repeat_matvec.

    用于观测工作区缓冲的初次物化净增 (首次算子乘) 以及稳态重复调用的零内存增长,
    同时提供跨阶段连续水位演进数据. ``--variant per_element`` / ``shared`` 时 cache 阶段为 ``_setup_reference``,
    算子乘为 ``SharedReferenceElementAssembly.__matmul__``, 与纯 numpy 参考 ``_reference_shared_kx`` 核对.
    """
    import gc

    ctx, mesh_meter = _build(method, n)
    stages = {}
    rng = np.random.default_rng(0)
    times = [0.0] * repeats

    @contextlib.contextmanager
    def stage(name: str):
        before = cur_rss_kib()
        reset = reset_peak_rss()
        start = time.perf_counter()
        yield
        seconds = time.perf_counter() - start
        after = cur_rss_kib()
        peak = max(before, after, peak_rss_kib())
        stages[name] = {
            "before_kib": before,
            "peak_kib": peak,
            "after_kib": after,
            "net_kib": peak - before,
            "t_s": seconds,
            "reset_supported": reset,
        }

    gc.collect()
    trim_supported = _malloc_trim()
    with stage("cache"):
        operator = _setup_level(ctx)
    with stage("input"):
        x = rng.standard_normal(ctx["facts"]["Ndof"])
    with stage("first_matvec"):
        y = operator @ x
    with stage("repeat_matvec"):
        for i in range(repeats):
            start = time.perf_counter()
            y = operator @ x
            times[i] = time.perf_counter() - start

    # 正确性核对
    c2d = np.asarray(operator.restriction.cell2dof)
    if VARIANT != "standard":
        ctx.update(_reference_data(operator))
        Ke = ctx["K0"]  # 只用于取 LDOF 算乘加次数
        ref = _reference_shared_kx(ctx["K0"], ctx["scale"], c2d, x)
        persistent_bytes = int(operator.persistent_bytes())
        variant_fields = _reference_fields(ctx)
    else:
        Ke = np.asarray(operator.element_matrices)
        ref = _reference_kx(Ke, c2d, x)
        persistent_bytes = int(Ke.nbytes + c2d.nbytes)
        variant_fields = {}
    error = _relerr(y, ref)
    med = statistics.median(times)
    peak = max(mesh_meter.max_peak_kib(), *(r["peak_kib"] for r in stages.values()))
    facts = ctx["facts"]
    result = {
        "panel": "continuous",
        **_mesh_fields(),
        "n": n,
        **facts,
        "method": method,
        "operator_impl": type(operator).__module__ + "." + type(operator).__name__,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "同一进程: cache -> input -> first_matvec -> repeat_matvec; 无右端与边界处理; 阶段间不额外 gc/trim; 保留输入与输出",
        "trim_before_cache_supported": trim_supported,
        "stages": stages,
        "mesh_fields": mesh_meter.fields(),
        "process_peak_kib": peak,
        "operator_persistent_bytes": persistent_bytes,
        **variant_fields,
        "matvec_seconds_median": med,
        "matvec_seconds_all": times,
        "effective_gbps_lower_bound": (persistent_bytes + 16 * facts["Ndof"]) / med / 1e9,
        "gflops": _flops_per_cell(Ke) * facts["NC"] / med / 1e9,
        "reference_relerr": error,
    }
    if error > 1e-12 or not all(r["reset_supported"] for r in stages.values()):
        raise RuntimeError("连续测量核对失败, 请检查原始记录")
    return result



# -----------------------------------------------------------------------------
# 1b. 与 fa 的逐位一致性核对
# -----------------------------------------------------------------------------

def verify_against_fa(n: int, methods: Sequence[str]) -> int:
    """逐位核对核心路径缓存的 K_e / K_e^0 / cell2dof 与 fa 构建路径生成的是否完全相同.

    同一进程内用 ``fa_assembly_capability/run.py`` 的 ``_build_problem_space`` 建问题并直接调用
    ``LinearElasticIntegrator(material, method).assembly(vs)`` (fa 阶段 1 的做法), 再用本目录的
    ``ElasticityEAOperator(...).analyzer.assemble_stiff_matrix()`` 取标准 EA 常驻的 K_e 与 cell2dof,
    用 ``_setup_reference`` (N_k = NC) 取逐单元参考 EA 常驻的 K_e^0, 用 ``np.array_equal`` 逐位比较
    (形状、dtype、数值). 只在 CPU 上核对, 不落盘. fa 只有 tet、p = 1 的构建路径, 故本核对固定取
    ``mesh_type="tet", p=1``, 与 --mesh / --p 无关.

    Parameters
    ----------
    n : int
        网格每方向段数, 默认 8 即可 (秒级).
    methods : sequence of str
        要核对的单刚组装方式.

    Returns
    -------
    int
        全部一致返回 0, 任一不一致返回 1.
    """
    import importlib.util

    fa_path = config.REPOSITORY_ROOT / "experiments" / "fa_assembly_capability" / "run.py"
    spec = importlib.util.spec_from_file_location("_fa_run", fa_path)
    fa_run = importlib.util.module_from_spec(spec)
    sys.modules["_fa_run"] = fa_run  # dataclass 装饰器要求模块已注册
    spec.loader.exec_module(fa_run)

    import_fe_stack_cpu()
    from soptx.fem.integrators import LinearElasticIntegrator
    from soptx.fem.matrix_free import ElasticityEAOperator

    _configure("tet", 1)  # _setup_reference 按 MESH / P 取积分阶
    _, _, vs_fa, mat_fa = fa_run._build_problem_space(n)
    problem, mesh_ea, vs_ea, mat_ea = build_problem_space(n, mesh_type="tet", p=1)
    NC = int(mesh_ea.number_of_cells())

    def same_array(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)

    ok = True
    c_fa = np.asarray(vs_fa.cell_to_dof())
    for m in methods:
        k_fa = np.asarray(LinearElasticIntegrator(mat_fa, method=m).assembly(vs_fa))
        facade = ElasticityEAOperator(vs_ea, problem, mat_ea, degree=1, assembly_method=m,
                                      integration_order=integration_order("tet", 1))
        facade.analyzer.assemble_stiff_matrix()
        k_ea, c_ea = _element_data(facade)
        k_ref = np.asarray(_setup_reference({"vs": vs_ea, "material": mat_ea, "method": m,
                                             "num_classes": NC}).reference_matrices)

        same_c = same_array(c_fa, c_ea)
        same_k = same_array(k_fa, k_ea)
        same_ref = same_array(k_fa, k_ref)
        ok &= same_c and same_k and same_ref
        if same_k:
            verdict = "identical (bitwise)"
        elif k_fa.shape == k_ea.shape:
            verdict = f"DIFFER, max|diff| = {float(np.max(np.abs(k_fa - k_ea))):.3e}"
        else:
            verdict = f"DIFFER, shape {k_fa.shape} vs {k_ea.shape}"
        print(
            f"n={n} method={m:<8} K_e {k_fa.shape} {k_fa.dtype}: {verdict} | "
            f"K_e^0 (per_element): {'identical' if same_ref else 'DIFFER'} | "
            f"cell2dof {c_fa.shape} {c_fa.dtype}: {'identical' if same_c else 'DIFFER'}"
        )

    print("RESULT:", "ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


def verify_shared(n: int, methods: Sequence[str], rtol: float = 1e-11) -> int:
    """核对逐单元参考 EA、共享参考 EA 与标准 EA 在当前 --mesh / --p 上是同一个离散算子 (到舍入).

    同一进程内用 ``ElasticityEAOperator(...).analyzer.assemble_stiff_matrix()`` 取标准 EA, 用
    ``_setup_reference`` 分别以 N_k = NC 与平移类数 N_k 显式构造逐单元参考 EA 与共享参考 EA, 逐项比较相对误差
    (以标准 EA 结果的最大模为尺度):

    1. 参考前提: 每个单元的 K_e 与所属类的 K_k(e)^0 之差 (以 max|K_k^0| 为尺度); 逐单元参考 EA 下即 K_e 与
       K_e^0 之差;
    2. 单列 ``@ x``, 多列 ``@ X`` 与 ``diagonal()``;
    3. ``update(rho)`` 后的 ``@ x``, 与 rho 缩放的 K_e 的纯 numpy 参考比较.

    同类单元的节点坐标只在舍入意义下相等, 故核对到 ``rtol`` 而非逐位. 只在 CPU 上核对, 不落盘.

    Parameters
    ----------
    n : int
        网格每方向段数, 默认 4 即可 (秒级).
    methods : sequence of str
        要核对的单刚组装方式, 三种 EA 取同一种.
    rtol : float, optional
        各项相对误差的上限, 默认 1e-11.

    Returns
    -------
    int
        全部通过返回 0, 任一超限返回 1.
    """
    import_fe_stack_cpu()
    from soptx.fem.matrix_free import ElasticityEAOperator

    problem, mesh, vs, material = build_problem_space(n, mesh_type=MESH, p=P)
    facts = mesh_facts(mesh, vs)
    NC = int(facts["NC"])
    variants = {"per_element": NC, "shared": _num_classes()}
    rng = np.random.default_rng(0)
    x = rng.standard_normal(facts["Ndof"])
    X = rng.standard_normal((facts["Ndof"], 3))
    rho = rng.uniform(1e-3, 1.0, NC)

    print(f"{MESH_SPECS[MESH].mesh_class} p = {P}, q = {integration_order(MESH, P)}, n = {n}: "
          f"NC = {NC:,}, Ndof = {facts['Ndof']:,}, N_k = {variants['shared']}")
    ok = True
    for m in methods:
        standard = ElasticityEAOperator(vs, problem, material, degree=P, assembly_method=m,
                                        integration_order=integration_order(MESH, P)).analyzer.assemble_stiff_matrix()
        Ke = np.asarray(standard.element_matrices)
        c2d = np.asarray(standard.restriction.cell2dof)
        rho_kx = _reference_kx(rho[:, None, None] * Ke, c2d, x)

        for variant, num_classes in variants.items():
            op = _setup_reference({"vs": vs, "material": material, "method": m, "num_classes": num_classes})
            K0 = np.asarray(op.reference_matrices)
            # 单元 e = 格子 * N_k + k, 按 (格子, 类) 重排后第 1 维即 k(e), 与 K_k^0 广播相减
            Ke_G = Ke.reshape((NC // num_classes, ) + K0.shape)

            errs = {
                "K_e vs K_k^0": float(np.max(np.abs(Ke_G - K0))) / float(np.max(np.abs(K0))),
                "K x": _relerr(np.asarray(op @ x), np.asarray(standard @ x)),
                "K X": _relerr(np.asarray(op @ X), np.asarray(standard @ X)),
                "diag": _relerr(np.asarray(op.diagonal()), np.asarray(standard.diagonal())),
            }
            op.update(rho)
            errs["rho K x"] = _relerr(np.asarray(op @ x), rho_kx)

            passed = all(v < rtol for v in errs.values())
            ok &= passed
            detail = " | ".join(f"{k} {v:.1e}" for k, v in errs.items())
            print(f"  method={m:<8} {variant:<11} {'PASS' if passed else 'FAIL'}: {detail}")

    print("RESULT:", f"ALL < {rtol:.0e}" if ok else "MISMATCH")
    return 0 if ok else 1


# -----------------------------------------------------------------------------
# 2. 控制台树状卡片看板
# -----------------------------------------------------------------------------

def _fmt_mib(mib: float | None) -> str:
    if mib is None:
        return "--"
    if mib >= 1024:
        return f"{mib / 1024:.2f} GiB ({mib:,.1f} MiB)"
    return f"{mib:,.1f} MiB"


def _fmt_s(t: float | None) -> str:
    if t is None:
        return "--"
    return f"{t * 1000:.1f} ms" if t < 1.0 else f"{t:.2f} s"


def _stage_line(out: dict, name: str, label: str, unit_key: str | None = None, unit: str = "KB/dof") -> str:
    peak = out.get(f"{name}_peak_MiB")
    net = out.get(f"{name}_net_MiB")
    text = f"{label:<16}: peak {_fmt_mib(peak)} | net {_fmt_mib(net)}"
    if unit_key and out.get(unit_key) is not None:
        text += f" | {out[unit_key]:.2f} {unit}"
    t = out.get(f"t_{name}_s")
    if t is not None:
        text += f" | {_fmt_s(t)}"
    return text


def print_dashboard(out: dict[str, Any]) -> None:
    """打印树状卡片式实测摘要 (每阶段绝对峰值 / 净增)."""
    n = out.get("n", 0)
    nc = out.get("NC", 0)
    ndof = out.get("Ndof", 0)
    mesh_line = (f"{out.get('mesh_type')} p = {out.get('p')}, q = {out.get('q')} "
                 f"(grid = {n}^{out.get('GD')}) | {nc:,} cells | {ndof:,} DOFs")
    panel = out.get("panel")
    titles = {"cache": "element-cache", "matvec": "ea-matvec", "solve": "ea-cg-solve"}
    rep = out.get("repeats", 0)

    print(f"\n● [{titles.get(panel, panel)}] {out.get('problem')}")
    print(f"  ├── Mesh & DOFs   : {mesh_line}")
    variant = out.get("variant", "standard")
    reference = variant != "standard"
    print(
        f"  ├── Operator      : {out.get('operator_impl')} | variant = {variant} | "
        f"method = {out.get('method')} | K_e theory = {out.get('Ke_theory_MiB', 0):,.1f} MiB | "
        f"persistent = {_fmt_mib(out.get('operator_persistent_MiB'))} "
        f"({out.get('operator_persistent_KB_per_dof', 0):.2f} KB/dof)"
    )
    if reference:
        print(
            f"  ├── Reference     : N_k = {out.get('num_classes')} | K_k^0 {out.get('K0_shape')} = "
            f"{out.get('K0_MiB', 0):.3f} MiB | s_e = {_fmt_mib(out.get('scale_MiB'))} | "
            f"cell2dof = {_fmt_mib(out.get('cell2dof_MiB'))}"
        )
    print(f"  ├── {_stage_line(out, 'mesh', 'Mesh & Space')}")
    if panel == "cache":
        label = {"standard": "Cache (K_e)", "per_element": "Cache (K_e^0)"}.get(variant, "Cache (K_k^0)")
        print(f"  ├── {_stage_line(out, 'cache', label, 'cache_KB_per_dof')}")
    else:
        print(f"  ├── {_stage_line(out, 'assemble', 'Assemble (bc)')}")
    if panel == "matvec":
        print(f"  ├── {_stage_line(out, 'warmup', 'Warmup')}")
        print(f"  ├── {_stage_line(out, 'matvec', f'K x (form@x) x{rep}')}")
        print(
            f"  ├── Per K x       : median {_fmt_s(out.get('matvec_seconds_median'))} | "
            f"min {_fmt_s(out.get('matvec_seconds_min'))} | "
            f"eff >= {out.get('effective_gbps_lower_bound', 0):.2f} GB/s | "
            f"{out.get('gflops', 0):.2f} GFLOP/s | relerr {out.get('matvec_vs_reference_relerr', 0):.1e}"
        )
    elif panel == "solve":
        print(f"  ├── {_stage_line(out, 'setup_solve', 'Solve setup')}")
        print(f"  ├── {_stage_line(out, 'solve', 'Jacobi-PCG')}")
        status = "converged" if out.get("converged") else f"NOT converged ({out.get('reason')})"
        print(
            f"  ├── Iterations    : {out.get('iterations', 0):,} ({status}, relres {out.get('final_relres', 0):.2e}, "
            f"true {out.get('true_relres', 0):.2e}) | "
            f"{out.get('iterations_per_n', 0):.2f} it/n | {_fmt_s(out.get('seconds_per_iteration'))} per it"
        )
    print(
        f"  └── Absolute Peak : {_fmt_mib(out.get('process_max_rss_MiB'))} "
        f"({out.get('process_max_rss_KB_per_dof', 0):.2f} KB/dof)\n"
    )


# -----------------------------------------------------------------------------
# 3. 调度: Case -> 子进程执行计划
# -----------------------------------------------------------------------------

LIST_COLUMNS = (
    ("panel", "panel", "-"),
    ("mesh", "mesh_type", MESH_TYPE),
    ("p", "p", "1"),
    ("grid", "grid", "-"),
    ("problem", "problem", PROBLEM_NAME),
    ("method", "method", "fast"),
    ("variant", "variant", "standard"),
)


def resolve_runs(
    cases: tuple[config.Case, ...],
    is_all: bool,
    overrides: dict[str, Any],
) -> list[scheduler.Run]:
    """将选中的 Case 与参数覆盖解析为具体的单次子进程执行计划.

    三个面板默认只跑 case 的 method (fast); 仅显式 --method all 时展开 METHOD_NAMES, --all 不展开 (is_all 仅保留签名).
    网格类型与次数: 覆盖值优先, 否则取 case 的 mesh / p, 再缺省为 DEFAULT_MESH 与 1; 二者进入产物名.
    EA 变体只取 case 的 variant (缺省 standard), 不设覆盖: per_element / shared 在产物名的 method 段前多一个
    变体段, standard 的产物名不变. update 写法按变体的可用集合展开, 'all' 取整个集合; 与变体不符的写法不生成子进程.
    """
    runs: list[scheduler.Run] = []
    for case in cases:
        if case.panel == "baseline":
            out_p = config.OUTPUT_DIR / case.artifact
            argv = [sys.executable, str(case.script_path), "--worker", "--baseline", "--output", str(out_p)]
            runs.append((case.id, argv, out_p, case.summary, case.subprocess_env()))
            continue
        n = overrides.get("n", case.extra.get("n", 32))
        mesh = overrides.get("mesh", case.extra.get("mesh", DEFAULT_MESH))
        p = overrides.get("p", case.extra.get("p", 1))
        variant = case.extra.get("variant", "standard")
        if variant not in VARIANTS:
            raise ValueError(f"case {case.id!r} 的 variant {variant!r} 未知, 可选 {VARIANTS}")
        if variant != "standard" and case.panel in ("matvec", "solve"):
            raise ValueError(f"case {case.id!r}: {case.panel} 面板只支持 variant = 'standard'")
        disc = [mesh, f"p{p}"]  # 产物名中的网格与次数段
        head = [] if variant == "standard" else [variant]  # 产物名中 method 段前的变体段
        env = case.subprocess_env()
        common = [sys.executable, str(case.script_path), "--worker"]
        tail = ["--mesh", mesh, "--p", str(p), "--n", str(n)]
        if variant != "standard":
            tail += ["--variant", variant]
        methods = scheduler.expand(
            overrides.get("method"),
            False,
            list(METHOD_NAMES),
            case.extra.get("method", "fast"),
        )

        for m in methods:
            if case.panel == "cache":
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("cache", [*head, m, *disc], n, DEVICE)
                argv = [*common, "--cache", "--method", m, *tail, "--output", str(out_p)]
                detail = f"variant={variant}, method={m}, mesh={mesh}, p={p}, n={n}"
            elif case.panel == "matvec":
                repeats = overrides.get("repeats", case.extra.get("repeats", 20))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("matvec", [m, *disc], n, DEVICE)
                argv = [*common, "--matvec", "--method", m, *tail, "--repeats", str(repeats), "--output", str(out_p)]
                detail = f"method={m}, mesh={mesh}, p={p}, n={n}, repeats={repeats}"
                if overrides.get("probe_allocations"):
                    out_p = config.OUTPUT_DIR / scheduler.artifact_name("matvec_allocations", [m, *disc], n, DEVICE)
                    argv = [*common, "--matvec", "--probe-allocations", "--method", m, *tail, "--output", str(out_p)]
                    detail = f"method={m}, mesh={mesh}, p={p}, n={n}, allocation probe"
            elif case.panel == "update":
                rounds = overrides.get("rounds", case.extra.get("rounds", 5))
                # 覆盖值优先, 否则取 case 的 update_mode; 二者为 'all' 时展开该变体的全部写法
                valid = update_modes(variant)
                requested = overrides.get("update_mode", case.extra.get("update_mode", "all"))
                modes = list(valid) if requested == "all" else [requested] if requested in valid else []
                for mode in modes:
                    out_p = config.OUTPUT_DIR / scheduler.artifact_name("update", [mode, *head, m, *disc], n, DEVICE)
                    argv = [*common, "--update", "--update-mode", mode, "--method", m, *tail,
                            "--rounds", str(rounds), "--output", str(out_p)]
                    detail = f"mode={mode}, variant={variant}, method={m}, mesh={mesh}, p={p}, n={n}, rounds={rounds}"
                    runs.append((f"{case.id} [{mode}, {m}]", argv, out_p, f"{case.summary} ({detail})", env))
                continue
            elif case.panel == "continuous":
                repeats = overrides.get("repeats", case.extra.get("repeats", 20))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("cache_matvec_continuous", [*head, m, *disc], n, DEVICE)
                argv = [*common, "--continuous", "--method", m, *tail, "--repeats", str(repeats), "--output", str(out_p)]
                detail = f"variant={variant}, method={m}, mesh={mesh}, p={p}, n={n}, repeats={repeats}"
            else:  # solve
                maxiter = overrides.get("maxiter", case.extra.get("maxiter", 5000))
                tol = overrides.get("tol", case.extra.get("tol", 1e-6))
                out_p = config.OUTPUT_DIR / scheduler.artifact_name("solve", [m, *disc], n, DEVICE)
                argv = [
                    *common, "--solve", "--method", m, *tail,
                    "--maxiter", str(maxiter), "--tol", str(tol), "--output", str(out_p),
                ]
                detail = f"method={m}, mesh={mesh}, p={p}, n={n}, maxiter={maxiter}, tol={tol}"
            label = f"{case.id} [{m}]"
            runs.append((label, argv, out_p, f"{case.summary} ({detail})", env))
    return runs


# -----------------------------------------------------------------------------
# 4. 主入口与参数路由
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """调度与测量的主入口函数.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数列表, 缺省使用 sys.argv[1:].

    Returns
    -------
    int
        程序退出状态码.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="ea_assembly_capability 实验图面驱动: 测量核心 EA 算子的内存开销、算子乘耗时与 Jacobi-PCG 迭代",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点")
    parser.add_argument("--all", action="store_true", help="跑全部工况 (每个 case 一条, method 取 cases.toml 的 fast)")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id (如 --cases element-cache)")
    parser.add_argument("--case", help="指定单个 case id (等价于 --cases <id>)")
    parser.add_argument("--panel", choices=config.PANELS, help="只跑指定面板 (cache/matvec/solve/baseline) 数据点")
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")
    parser.add_argument("--monitor", action="store_true", help="运行时实时显示独立 Worker 的 CPU 与内存占用")
    parser.add_argument(
        "--monitor-interval", type=float, default=0.5, metavar="SECONDS",
        help="实时监控刷新间隔, 单位为秒 (默认 0.5)",
    )

    # 2. 工况动态覆盖参数 (Overrides)
    parser.add_argument("-n", "--n", "--grid", dest="n", type=int, default=None, help="动态覆盖网格剖分段数 (如 -n 32 或 --grid 32)")
    parser.add_argument("--method", choices=METHOD_NAMES + ("all",), default=None, help="指定或覆盖单刚算法")
    parser.add_argument("--mesh", choices=tuple(MESH_SPECS), default=None,
                        help=f"网格类型 (from_box 结构化网格), 覆盖 case 的 mesh; 缺省 {DEFAULT_MESH}")
    parser.add_argument("--p", type=int, default=None, help="Lagrange 空间次数, 覆盖 case 的 p; 缺省 1")
    parser.add_argument("--repeats", type=int, default=None, help="matvec: 计时重复次数 (默认 20)")
    parser.add_argument("--maxiter", type=int, default=None, help="solve: PCG 最大迭代数 (默认 5000)")
    parser.add_argument("--tol", type=float, default=None, help="solve: 相对残差收敛阈值 (默认 1e-6)")
    parser.add_argument("--update-mode", choices=ALL_UPDATE_MODES + ("all",), default=None,
                        help="update: 写法, standard 取 reassemble, per_element / shared 取 rescale (默认 all, 由调度层按变体展开)")
    parser.add_argument("--rounds", type=int, default=None, help="update: 轮数 (默认 5, 至少 2)")

    # 3. Worker 测量层底层参数 (供子进程调用)
    parser.add_argument("--worker", action="store_true", help="进入子进程 worker 测量模式")
    parser.add_argument("--cache", action="store_true", help="cache 面板: K_e 与 cell2dof 缓存")
    parser.add_argument("--matvec", action="store_true", help="matvec 面板: 核心 EA 刚度算子乘 K x 计时")
    parser.add_argument("--solve", action="store_true", help="solve 面板: 核心 Jacobi-PCG 求解")
    parser.add_argument("--continuous", action="store_true", help="continuous 面板: 同一进程连续测量 cache -> input -> first_matvec -> repeat_matvec")
    parser.add_argument("--baseline", action="store_true", help="baseline 面板: 单核 memcpy 带宽与 dgemm 算力")
    parser.add_argument("--update", action="store_true", help="update 面板: 单元密度下 op.update 的峰值、常驻与耗时")
    parser.add_argument("--variant", choices=VARIANTS, default="standard",
                        help="Worker 测量的 EA 变体: standard (分析器的 ElementAssembly, 默认), per_element "
                             "(N_k = NC) 或 shared (N_k < NC) 的显式构造 SharedReferenceElementAssembly (后两者只支持 "
                             "cache / continuous / update); 调度模式下由 case 的 variant 决定")
    parser.add_argument("--output", type=Path, default=None, help="产物落盘路径")

    # 4. 一致性核对 (进程内, 不落盘)
    parser.add_argument(
        "--verify-fa", action="store_true",
        help="逐位核对核心路径缓存的 K_e / K_e^0 / cell2dof 与 fa_assembly_capability 的构建路径一致 (仅 tet, p = 1; 默认 --n 8, --method all)",
    )
    parser.add_argument(
        "--verify-shared", action="store_true",
        help="核对逐单元参考 EA、共享参考 EA 与标准 EA 的 K x / K X / 对角 / update 后 K x 一致到 1e-11 (当前 --mesh / --p; 默认 --n 4, --method all)",
    )

    parser.add_argument("--probe-allocations", action="store_true", help="matvec: 独立测一次预热后的分配峰值, 不计时, 单独落盘")
    args = parser.parse_args(argv)
    if args.probe_allocations and (args.cache or args.solve or args.baseline or args.verify_fa or args.verify_shared
                                   or args.continuous):
        parser.error("--probe-allocations 仅适用于 matvec")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval 必须大于 0")
    if args.p is not None and args.p < 1:
        parser.error("--p 必须为正整数")

    # ------------------------------------------------ 与 fa 的逐位核对
    if args.verify_fa:
        if (args.mesh not in (None, "tet")) or (args.p not in (None, 1)):
            parser.error("--verify-fa 只支持 tet、p = 1 (fa 只有这一条构建路径), 请用 --mesh tet 或省略 --mesh / --p")
        methods = list(METHOD_NAMES) if args.method in (None, "all") else [args.method]
        return verify_against_fa(args.n if args.n is not None else 8, methods)

    # ------------------------------------------------ 逐单元参考 EA、共享参考 EA 与标准 EA 的一致性核对
    if args.verify_shared:
        _configure(args.mesh or DEFAULT_MESH, args.p or 1)
        methods = list(METHOD_NAMES) if args.method in (None, "all") else [args.method]
        return verify_shared(args.n if args.n is not None else 4, methods)

    # ------------------------------------------------ Worker 测量分支
    if args.baseline:
        out = measure_baseline()
        print_baseline(out)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.worker or args.cache or args.matvec or args.solve or args.continuous or args.update:
        if args.n is None:
            parser.error("Worker 模式必须指定 --n")
        if args.method == "all":
            parser.error("Worker 模式不接受 --method all, 由调度层展开")
        method = args.method or "fast"
        _configure(args.mesh or DEFAULT_MESH, args.p or 1, args.variant)
        if args.variant != "standard" and (args.matvec or args.solve):
            parser.error(f"--variant {args.variant} 只支持 --cache, --continuous 与 --update (matvec / solve 需要分析器与门面)")
        if args.update:
            if args.update_mode in (None, "all"):
                parser.error("Worker 模式的 --update 需指定单个 --update-mode, all 由调度层展开")
            if args.update_mode not in update_modes(args.variant):
                parser.error(f"--variant {args.variant} 的 --update-mode 须为 {update_modes(args.variant)} 之一")
            out = measure_update(method, args.n, args.update_mode, rounds=args.rounds or 5)
        elif args.continuous:
            out = measure_continuous(method, args.n, repeats=args.repeats or 20)
        elif args.matvec:
            out = (measure_matvec_allocations(method, args.n) if args.probe_allocations
                   else measure_matvec(method, args.n, repeats=args.repeats or 20))
        elif args.solve:
            out = measure_solve(
                method, args.n,
                maxiter=args.maxiter or 5000,
                tol=args.tol if args.tol is not None else 1e-6,
            )
        elif args.cache:
            out = measure_cache(method, args.n)
        else:
            parser.error("Worker 模式需指定 --cache, --matvec, --solve, --continuous 或 --update")

        if args.update:
            if out["variant"] != "standard":
                resident = f"K_k^0 (N_k = {out['num_classes']}) = {out['K0_MiB']} MiB, s_e = {out['scale_MiB']} MiB"
            else:
                resident = f"K_e = {out['Ke_MiB']} MiB"
            print(f"EA update 面板 (variant = {out['variant']}, mode = {out['mode']}, rounds = {out['rounds']}, {resident})")
            for stage_name, sinfo in out["stages"].items():
                print(f"  [{stage_name}] before: {sinfo['before_kib']/1024:.1f} MiB | peak: {sinfo['peak_kib']/1024:.1f} MiB | after: {sinfo['after_kib']/1024:.1f} MiB | net: {sinfo['net_kib']/1024:.1f} MiB | time: {sinfo['t_s']:.3f} s")
            print(f"  每轮耗时: 首轮 {_fmt_s(out['update_seconds_first'])} | 稳态中位数 {_fmt_s(out['update_seconds_rest_median'])} | 核对 relerr {out['check_relerr_max']:.1e}")
        elif args.continuous:
            print(f"EA 连续测量面板 (variant = {out['variant']}, cache -> input -> first_matvec -> repeat_matvec x {args.repeats or 20})")
            for stage_name, sinfo in out["stages"].items():
                print(f"  [{stage_name}] before: {sinfo['before_kib']/1024:.1f} MiB | peak: {sinfo['peak_kib']/1024:.1f} MiB | net: {sinfo['net_kib']/1024:.1f} MiB | time: {sinfo['t_s']:.3f} s")
            print(f"  稳态耗时中位数: {out['matvec_seconds_median']*1000:.2f} ms | 有效带宽下界: {out['effective_gbps_lower_bound']:.2f} GB/s")
        elif args.probe_allocations:
            print("EA 分配探针 (不计时, NumPy 跟踪校验通过)")
            for key in ("allocation_peak_increment_bytes", "allocation_retained_increment_bytes",
                        "allocation_peak_minus_retained_bytes", "output_bytes"):
                print(f"  {key}: {out[key] / 2**20:.3f} MiB")
            print(f"  RSS peak: {out['probe_rss_peak_MiB']} MiB | increment: {out['probe_rss_peak_increment_MiB']} MiB")
        else:
            print_dashboard(out)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ------------------------------------------------ 调度与执行分支
    try:
        figure, cases = config.load_cases()
    except config.ConfigError as error:
        print(f"cases.toml 有误: {error}", file=sys.stderr)
        return 2

    if args.list:
        return scheduler.print_case_table(cases, LIST_COLUMNS)

    target_case_ids: list[str] = []
    panel_filter: str | None = args.panel
    known_ids = {c.id for c in cases}
    is_all = args.all

    alias_map = {
        "elem-cache": "element-cache",
        "cg": "ea-cg-solve",
        "cg-solve": "ea-cg-solve",
        "continuous": "ea-continuous",
    }

    def resolve_case_id(name: str) -> str:
        if name in known_ids:
            return name
        return alias_map.get(name.lower(), name)

    raw_cases: list[str] = []
    if args.cases:
        raw_cases.extend(args.cases)
    if args.case:
        raw_cases.append(args.case)

    for item in raw_cases:
        item_lower = item.lower()
        if item_lower == "all":
            is_all = True
        elif item_lower in config.PANELS:
            panel_filter = item_lower
        else:
            target_case_ids.append(resolve_case_id(item))

    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case/--cases/--panel/--all 指定要运行的工况或面板。\n"
            "  常用示例:\n"
            "    python run.py --case element-cache --grid 32 --monitor\n"
            "    python run.py --case ea-matvec --grid 32 --repeats 20\n"
            "    python run.py --case cpu-baseline --monitor\n"
            "    python run.py --all --check-only\n"
            "  查看全部工况列表请使用: python run.py --list",
            file=sys.stderr,
        )
        return 2

    try:
        selected = config.select(
            cases,
            case_ids=target_case_ids if target_case_ids else None,
            panel=panel_filter,
        )
    except config.ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2

    if args.probe_allocations and any(case.panel != "matvec" for case in selected):
        parser.error("--probe-allocations 只能选择 matvec 工况")
    overrides: dict[str, Any] = {"probe_allocations": args.probe_allocations}
    if args.n is not None:
        overrides["n"] = args.n
    if args.method is not None:
        overrides["method"] = args.method
    if args.mesh is not None:
        overrides["mesh"] = args.mesh
    if args.p is not None:
        overrides["p"] = args.p
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.maxiter is not None:
        overrides["maxiter"] = args.maxiter
    if args.tol is not None:
        overrides["tol"] = args.tol
    if args.update_mode is not None:
        overrides["update_mode"] = args.update_mode
    if args.rounds is not None:
        overrides["rounds"] = args.rounds

    runs = resolve_runs(selected, is_all=is_all, overrides=overrides)
    failed = scheduler.command_run(
        runs,
        cwd=config.REPOSITORY_ROOT,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
