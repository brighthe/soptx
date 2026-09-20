# -*- coding: utf-8 -*-
"""FA 全装配两阶段数据点的自包含调度与测量入口.

本脚本同时承担「调度器」与「独立子进程测量器」两个角色.

1. 调度模式 (默认入口, 每个数据点独占一个子进程):
   - python run.py --list                                   # 列出已注册数据点
   - python run.py --all --check-only                       # 只打印将执行的子进程命令
   - python run.py --case element-stiffness --method all --grid 32 --monitor
   - python run.py --case global-merge --route all --grid 32 --monitor
   - python run.py --case full-assembly --route all --grid 80 --monitor
   - python run.py --case full-assembly --route pattern --grid 24 --via-bilinearform

2. Worker 模式 (由调度器在独立进程中调用, 保证内存高水位严格隔离):
   - python run.py --worker --stage1 --method fast --n 32 --output outputs/stage1_fast_n32.json
   - python run.py --worker --stage2 --route pattern --n 32 --output outputs/stage2_pattern_n32.json
   - python run.py --worker --full --method fast --route pattern --n 32 \\
         --output outputs/full_fast_pattern_n32.json

内存口径 (CPU): 每个阶段先记 before = 当前 VmRSS, 再向 /proc/self/clear_refs 写 5 重置 VmHWM,
阶段结束读 VmHWM 作为该阶段的绝对峰值 peak, net = peak - before. 全程峰值 (final_peak /
process_max_rss) = 各阶段峰值的最大值 (clear_refs 会连带重置 ru_maxrss, 故不用它).
决定是否 OOM 的是绝对峰值而不是净增.

产物命名: <kind>_<...>_n<N>[_cuda][_bform].json; 后处理与对比表见 compare.py.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import os
import resource
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config  # noqa: E402


# -----------------------------------------------------------------------------
# 1. 测量与物理构件 (Worker 核心)
# -----------------------------------------------------------------------------

T_TET4 = 288  # tet4 线弹性: 每自由度对应的三元组数 (144 * NC / Ndof 的渐近值)
DEFAULT_MEMORY_TOTAL = 47.04 * 2**30  # WSL 来宾 MemTotal (字节), 仅作事实记录
MEMORY_BUDGET = 45 * 2**30  # 峰值内存预算 (字节): 容量结论以进程绝对峰值 RSS 不超过此值为界
ROUTE_NAMES = ("coalesce", "scipy", "pattern")
FULL_ROUTE_NAMES = ("pattern", "coalesce", "scipy")
METHOD_NAMES = ("standard", "voigt", "fast")
PROBLEM_NAME = "DivergenceFreePolynomialElasticity3D"


def rss_kib() -> int:
    """进程 RSS 高水位 (ru_maxrss, KiB). 注意: 向 clear_refs 写 5 会连带重置它, 测量阶段内不要依赖."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _read_self_status(*names: str) -> Dict[str, int]:
    """读取 /proc/self/status 中指定字段 (KiB)."""
    values: Dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as file:
        for line in file:
            name = line.partition(":")[0]
            if name in names:
                values[name] = int(line.split()[1])
    return values


def cur_rss_kib() -> int:
    """进程当前常驻内存 VmRSS (KiB)."""
    return _read_self_status("VmRSS").get("VmRSS", -1)


def peak_rss_kib() -> int:
    """进程 VmHWM (KiB): 自上次重置以来的 RSS 高水位."""
    return _read_self_status("VmHWM").get("VmHWM", -1)


def reset_peak_rss() -> bool:
    """向 /proc/self/clear_refs 写 5, 把 VmHWM 重置为当前 RSS; 内核不支持时返回 False."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as file:
            file.write("5")
    except OSError:
        return False
    return True


@dataclass
class StageRecord:
    """单个阶段的内存与耗时记录 (KiB / s)."""

    before_kib: int
    peak_kib: int
    t_s: float

    @property
    def net_kib(self) -> int:
        return max(0, self.peak_kib - self.before_kib)


class StageMeter:
    """按阶段记录 before / peak / net 的 CPU 内存测量器 (基于 VmHWM 重置).

    用法::

        meter = StageMeter()
        with meter.stage("stage1"):
            K_e = integrator.assembly(vs)
        fields = meter.fields()   # stage1_before_MiB / stage1_peak_MiB / stage1_net_MiB / t_stage1_s

    每个阶段的 peak 是该阶段内的绝对 RSS 高水位, 与 OOM 直接可比; net 是相对阶段开始时的净增.
    """

    def __init__(self) -> None:
        self.records: Dict[str, StageRecord] = {}
        self.reset_supported = True

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        gc.collect()
        before = cur_rss_kib()
        if not reset_peak_rss():
            self.reset_supported = False
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        peak = max(peak_rss_kib(), before)
        self.records[name] = StageRecord(before, peak, t1 - t0)

    def combine(self, name: str, parts: Sequence[str]) -> None:
        """把连续的若干子阶段合成一个阶段: before 取首个, peak 取最大, 耗时求和."""
        recs = [self.records[p] for p in parts]
        self.records[name] = StageRecord(
            recs[0].before_kib, max(r.peak_kib for r in recs), sum(r.t_s for r in recs)
        )

    def net_bytes(self, name: str) -> int:
        return self.records[name].net_kib * 1024

    def peak_bytes(self, name: str) -> int:
        return self.records[name].peak_kib * 1024

    def before_bytes(self, name: str) -> int:
        return self.records[name].before_kib * 1024

    def fields(self) -> Dict[str, Any]:
        """导出全部阶段字段 (MiB / s) 与进程级绝对高水位."""
        out: Dict[str, Any] = {}
        for name, rec in self.records.items():
            out[f"{name}_before_MiB"] = round(rec.before_kib / 1024, 1)
            out[f"{name}_peak_MiB"] = round(rec.peak_kib / 1024, 1)
            out[f"{name}_net_MiB"] = round(rec.net_kib / 1024, 1)
            out[f"t_{name}_s"] = round(rec.t_s, 3)
        out["process_max_rss_MiB"] = round(self.max_peak_kib() / 1024, 1)
        out["peak_reset_supported"] = self.reset_supported
        return out

    def max_peak_kib(self) -> int:
        """全部已测阶段峰值的最大值 = 测量区间内的进程绝对高水位.

        注意 clear_refs 重置 VmHWM 时会连带重置 ru_maxrss, 所以全程峰值不能再读 ru_maxrss.
        """
        return max((rec.peak_kib for rec in self.records.values()), default=0)


def cell_path(n: int) -> Path:
    """网格单元拓扑缓存文件路径 (阶段 2 合成输入复用)."""
    return _OUTPUT_DIR / f"cell_n{n}.npz"


def build_mesh(n: int) -> Tuple[np.ndarray, int]:
    """构建 3D 结构化四面体网格并返回单元拓扑 (NC, 4) 与节点总数."""
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    from fealpy.mesh import TetrahedronMesh

    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    cell = np.ascontiguousarray(np.asarray(mesh.entity("cell")), dtype=np.int64)
    return cell, int(mesh.number_of_nodes())


def load_cell(n: int) -> Tuple[np.ndarray, int]:
    """读取 (必要时生成并缓存) 网格单元拓扑."""
    path = cell_path(n)
    if not path.exists():
        cell, NN = build_mesh(n)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, cell=cell, NN=NN)
        return cell, NN
    with np.load(path) as z:
        return np.ascontiguousarray(z["cell"]), int(z["NN"])


def cell_to_dof_gd(cell: np.ndarray) -> np.ndarray:
    """gd_priority 自由度编号 3*node+comp 下的 cell_to_dof, 形状 (NC, 12); 与 TensorFunctionSpace(shape=(-1, 3)) 一致."""
    NC = cell.shape[0]
    return (3 * cell[:, :, None] + np.arange(3, dtype=np.int64)).reshape(NC, 12)


class _ScalarCellSpace:
    """标量层鸭子类型空间: ``cell_to_dof`` 即节点级 ``cell``."""

    def __init__(self, cell: np.ndarray, NN: int) -> None:
        self._c2d = cell
        self._gdof = int(NN)

    def cell_to_dof(self) -> np.ndarray:
        return self._c2d

    def number_of_global_dofs(self) -> int:
        return self._gdof


class _CellSpace:
    """鸭子类型的张量函数空间, 供生产 ``build_csr_pattern`` 直接消费.

    暴露 ``scalar_space`` / ``dof_numel`` / ``dof_priority``, 使其走与
    ``TensorFunctionSpace(shape=(-1, 3))`` 完全相同的标量级骨架路径; 不构建
    mesh / 函数空间对象, 避免其高水位污染阶段 2 测量.
    """

    dof_numel = 3
    dof_priority = False  # 3*node+comp, 与 shape=(-1, 3) 一致

    def __init__(self, cell: np.ndarray, NN: int) -> None:
        self._cell = cell
        self._gdof = 3 * int(NN)
        self.scalar_space = _ScalarCellSpace(cell, NN)

    def cell_to_dof(self) -> np.ndarray:
        return cell_to_dof_gd(self._cell)

    def number_of_global_dofs(self) -> int:
        return self._gdof


def det_val(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """由全局自由度索引 (I, J) 生成确定性数值, 保证各路线的数值输入逐位一致."""
    h = (i * 2654435761 + j * 40503) & 0xFFFFFFFF
    return (h % 1000003).astype(np.float64) * 1e-6


def full_triplets(cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """物化全长三元组 (I, J, V), 作为 coalesce / scipy 路线的输入."""
    NC = cell.shape[0]
    e2 = cell_to_dof_gd(cell)
    I = np.broadcast_to(e2[:, :, None], (NC, 12, 12)).ravel()
    J = np.broadcast_to(e2[:, None, :], (NC, 12, 12)).ravel()
    V = det_val(I, J)
    return I, J, V


def synthetic_element_matrices(cell: np.ndarray) -> np.ndarray:
    """合成单刚 (NC, 12, 12) float64, 数值与 ``full_triplets`` 的 V 一致, 作为 pattern 路线的输入."""
    e2 = cell_to_dof_gd(cell)
    return det_val(e2[:, :, None], e2[:, None, :])


def _is_cuda(device_str: str) -> bool:
    return device_str.startswith("cuda") or device_str == "gpu"


def _setup_cuda(device_str: str) -> Tuple[Any, str]:
    """切换 FEALPy 后端到 PyTorch 并绑定 CUDA 设备; 返回 (torch.device, 规范化设备名)."""
    import torch
    from fealpy.backend import backend_manager as bm

    device_str = "cuda:0" if device_str in ("gpu", "cuda") else device_str
    bm.set_backend("pytorch")
    bm.set_default_device(device_str)
    device = torch.device(device_str)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    return device, device_str


def _build_problem_space(n: int, device: Any = None) -> Tuple[Any, Any, Any, Any]:
    """构建制造解问题、tet4 网格、向量 P1 空间与材料 (需先设置后端)."""
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TetrahedronMesh
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems.elasticity import DivergenceFreePolynomialElasticity3D

    problem = DivergenceFreePolynomialElasticity3D()
    mesh = TetrahedronMesh.from_box(list(problem.domain), nx=n, ny=n, nz=n)
    scalar = LagrangeFESpace(mesh, p=1, ctype="C")
    vs = TensorFunctionSpace(scalar, shape=(-1, 3))
    if device is None:
        from fealpy.backend import backend_manager as bm

        device = bm.get_device(mesh)
    material = IsotropicLinearElasticMaterial(
        hypothesis="3D",
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=device,
    )
    return problem, mesh, vs, material


def _import_fe_stack_cpu() -> None:
    """在测量开始前把 FEALPy / SOPTX 相关模块全部导入, 避免 import 开销混入阶段测量."""
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    import fealpy.functionspace  # noqa: F401
    import fealpy.mesh  # noqa: F401
    import fealpy.sparse  # noqa: F401
    import soptx.fem.integrators  # noqa: F401
    import soptx.fem.matrix.csr_pattern  # noqa: F401
    import soptx.materials  # noqa: F401
    import soptx.problems.elasticity  # noqa: F401


def _mesh_facts(mesh: Any, vs: Any) -> Dict[str, int]:
    return {
        "NC": int(mesh.number_of_cells()),
        "NN": int(mesh.number_of_nodes()),
        "Ndof": int(vs.number_of_global_dofs()),
    }


def measure_stage1(method: str, n: int, device_str: str = "cpu") -> dict:
    """阶段 1 单刚: 只测 ``LinearElasticIntegrator.assembly`` 生成 (NC, 12, 12) 单刚张量的开销.

    Parameters
    ----------
    method : str
        单刚组装方式 (standard/voigt/fast).
    n : int
        网格每方向段数.
    device_str : str, default="cpu"
        计算设备 ('cpu' 或 'cuda:0').
    """
    from soptx.fem.integrators import LinearElasticIntegrator

    if _is_cuda(device_str):
        import torch

        device, device_str = _setup_cuda(device_str)
        _, mesh, vs, material = _build_problem_space(n)
        integrator = LinearElasticIntegrator(material, method=method)

        _ = integrator.assembly(vs)  # 预热 1 次
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        Ke = integrator.assembly(vs)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        net_bytes = max(0, peak_bytes - baseline_bytes)
        facts = _mesh_facts(mesh, vs)
        ke_theory = int(Ke.shape[0]) * int(Ke.shape[1]) * int(Ke.shape[2]) * 8
        del Ke
        torch.cuda.empty_cache()
        Ndof = facts["Ndof"]
        return {
            "stage": 1,
            "device": torch.cuda.get_device_name(device),
            "device_type": "cuda",
            "memory_kind": "vram",
            "method": method,
            "n": n,
            **facts,
            "Ke_theory_MiB": round(ke_theory / 2**20, 1),
            "stage1_before_MiB": round(baseline_bytes / 2**20, 1),
            "stage1_peak_MiB": round(peak_bytes / 2**20, 1),
            "stage1_net_MiB": round(net_bytes / 2**20, 1),
            "stage1_B_per_dof": round(net_bytes / Ndof, 1),
            "stage1_KB_per_dof": round(net_bytes / Ndof / 1000, 2),
            "t_stage1_s": round(t1 - t0, 4),
        }

    _import_fe_stack_cpu()
    meter = StageMeter()
    with meter.stage("mesh"):
        _, mesh, vs, material = _build_problem_space(n)
    integrator = LinearElasticIntegrator(material, method=method)

    with meter.stage("stage1"):
        Ke = integrator.assembly(vs)

    Ke = np.asarray(Ke)
    gc.collect()
    base_kib = meter.records["stage1"].before_kib
    retained_kib = max(0, cur_rss_kib() - base_kib)
    trimmed = _malloc_trim()
    after_kib = max(0, cur_rss_kib() - base_kib) if trimmed else retained_kib
    facts = _mesh_facts(mesh, vs)
    Ndof = facts["Ndof"]
    net_bytes = meter.net_bytes("stage1")
    ke_theory = int(np.prod(Ke.shape)) * Ke.dtype.itemsize
    return {
        "stage": 1,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": "rss",
        "method": method,
        "n": n,
        **facts,
        "Ke_shape": [int(s) for s in Ke.shape],
        "Ke_theory_MiB": round(ke_theory / 2**20, 1),
        **meter.fields(),
        "stage1_B_per_dof": round(net_bytes / Ndof, 1),
        "stage1_KB_per_dof": round(net_bytes / Ndof / 1000, 2),
        "stage1_peak_KB_per_dof": round(meter.peak_bytes("stage1") / Ndof / 1000, 2),
        "stage1_retained_MiB": round(retained_kib / 1024, 1),
        "stage1_retained_KB_per_dof": round(retained_kib * 1024 / Ndof / 1000, 2),
        "malloc_trim_supported": trimmed,
        "stage1_retained_after_trim_MiB": round(after_kib / 1024, 1),
    }


def _probe_obj_attr(obj: Any, name: str, make_wrapper) -> Any:
    """把 ``obj.name`` 替换为探针包装, 返回可调用的还原句柄; 属性不存在时返回 None."""
    original = getattr(obj, name, None)
    if original is None or not callable(original):
        return None
    had_own = name in getattr(obj, "__dict__", {})
    setattr(obj, name, make_wrapper(name, original))

    def restore() -> None:
        if had_own:
            setattr(obj, name, original)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                setattr(obj, name, original)

    return restore


def _result_bytes(value: Any) -> Tuple[int, Any]:
    """返回 (字节数, 形状); 元组按元素求和, 非数组返回 (0, None)."""
    if isinstance(value, tuple):
        total = 0
        shapes = []
        for item in value:
            size, shape = _result_bytes(item)
            total += size
            shapes.append(shape)
        return total, shapes
    nbytes = getattr(value, "nbytes", None)
    if nbytes is None:
        return 0, None
    return int(nbytes), [int(s) for s in getattr(value, "shape", ())]


def _run_stage1_probe(integrator: Any, vs: Any, material: Any, snapshot_at: Tuple[str, str] | None):
    """在 tracemalloc 下跑一次 ``integrator.assembly(vs)``, 逐调用记录调用前存活量与调用内峰值.

    Parameters
    ----------
    snapshot_at : tuple of (str, str), optional
        若给定 ``(调用名, 调用点)``, 在最后一次匹配该调用发生前拍一张快照, 用于分解此刻的存活构成.
    """
    import tracemalloc

    calls: list[dict] = []
    state = {"index": 0, "depth": 0, "snapshot": None}

    def make_wrapper(name, original):
        def wrapper(*args, **kwargs):
            if state["depth"] > 0:  # 只记最外层, 避免嵌套重复计数
                return original(*args, **kwargs)
            index = state["index"]
            state["index"] = index + 1
            frame = sys._getframe(1)
            site = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
            if snapshot_at is not None and (name, site) == snapshot_at:
                state["snapshot"] = tracemalloc.take_snapshot()
            live_before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            state["depth"] += 1
            try:
                result = original(*args, **kwargs)
            finally:
                state["depth"] -= 1
            live_after, peak_in = tracemalloc.get_traced_memory()
            out_bytes, out_shape = _result_bytes(result)
            calls.append({
                "index": index,
                "call": name,
                "site": site,
                "out_shape": out_shape,
                "out_MiB": round(out_bytes / 2**20, 1),
                "live_before_MiB": round(live_before / 2**20, 1),
                "peak_in_MiB": round(peak_in / 2**20, 1),
                "live_after_MiB": round(live_after / 2**20, 1),
                "transient_MiB": round(max(0, peak_in - max(live_before, live_after)) / 2**20, 1),
            })
            return result

        return wrapper

    from fealpy.backend import backend_manager as bm

    scalar_space = getattr(vs, "scalar_space", vs)
    restores = [
        _probe_obj_attr(bm.get_current_backend(), name, make_wrapper)
        for name in ("einsum", "zeros", "set_at", "concat", "tensordot")
    ]
    restores.append(_probe_obj_attr(material, "strain_matrix", make_wrapper))
    restores.append(_probe_obj_attr(scalar_space, "grad_basis", make_wrapper))

    gc.collect()
    tracemalloc.start(20)
    try:
        t0 = time.perf_counter()
        Ke = integrator.assembly(vs)
        t1 = time.perf_counter()
        live_end, traced_peak = tracemalloc.get_traced_memory()
        snapshot = state["snapshot"]
    finally:
        tracemalloc.stop()
        for restore in restores:
            if restore is not None:
                restore()
    del Ke
    gc.collect()
    # 每次被测调用前都做过 reset_peak, 故 get_traced_memory 的峰值只覆盖最后一段;
    # 阶段峰值取全部调用内峰值与该尾段峰值的最大者。
    stage_peak = max([traced_peak] + [c["peak_in_MiB"] * 2**20 for c in calls])
    return {
        "calls": calls,
        "t_assembly_s": round(t1 - t0, 4),
        "traced_peak_MiB": round(stage_peak / 2**20, 1),
        "traced_live_end_MiB": round(live_end / 2**20, 1),
    }, snapshot


def _snapshot_lines(snapshot: Any, limit: int = 24) -> list[dict]:
    """把快照按 traceback 归并, 取最大的若干条, 每条报告最靠近被测代码的一帧."""
    self_file = Path(__file__).resolve()
    stats = snapshot.statistics("traceback")
    rows = []
    for stat in stats[:limit]:
        preferred = None
        fallback = None
        for frame in stat.traceback:
            if Path(frame.filename).resolve() == self_file:
                continue
            site = f"{Path(frame.filename).name}:{frame.lineno}"
            if "linear_elastic_integrator" in frame.filename:
                preferred = site
            elif fallback is None and ("/src/soptx/" in frame.filename or "/fealpy/" in frame.filename):
                fallback = site
        top = stat.traceback[0]
        rows.append({
            "site": preferred or fallback or f"{Path(top.filename).name}:{top.lineno}",
            "MiB": round(stat.size / 2**20, 1),
            "count": stat.count,
        })
    return rows


def measure_stage1_probe(method: str, n: int) -> dict:
    """阶段 1 峰值归因: 直接在目标规模上用 ``tracemalloc`` 实测单刚计算内部的分配构成.

    对 ``bm.einsum`` / ``bm.zeros`` / ``bm.set_at`` / ``bm.concat`` / ``bm.tensordot`` /
    ``material.strain_matrix`` / ``space.grad_basis`` 打桩, 逐调用记录:
    调用前已存活的分配量 (co-resident)、调用内峰值、调用内临时量 (peak 减两端存活量的较大者)。
    跑两遍: 第一遍定位峰值所在调用, 第二遍在该调用前拍快照以分解此刻的存活构成。

    Parameters
    ----------
    method : str
        单刚组装方式 (standard/voigt/fast).
    n : int
        网格每方向段数.
    """
    from soptx.fem.integrators import LinearElasticIntegrator

    _import_fe_stack_cpu()
    _, mesh, vs, material = _build_problem_space(n)
    integrator = LinearElasticIntegrator(material, method=method)

    first, _ = _run_stage1_probe(integrator, vs, material, snapshot_at=None)
    target = max(first["calls"], key=lambda c: c["peak_in_MiB"])

    integrator = LinearElasticIntegrator(material, method=method)
    second, snapshot = _run_stage1_probe(
        integrator, vs, material, snapshot_at=(target["call"], target["site"])
    )
    peak_call = max(second["calls"], key=lambda c: c["peak_in_MiB"])

    facts = _mesh_facts(mesh, vs)
    return {
        "stage": 1,
        "pipeline": "stage1_probe",
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": "tracemalloc",
        "method": method,
        "n": n,
        **facts,
        "traced_peak_MiB": second["traced_peak_MiB"],
        "traced_peak_MiB_pass1": first["traced_peak_MiB"],
        "traced_live_end_MiB": second["traced_live_end_MiB"],
        "t_assembly_s": second["t_assembly_s"],
        "peak_call": peak_call,
        "peak_live_composition": _snapshot_lines(snapshot) if snapshot is not None else [],
        "calls": second["calls"],
    }

def _malloc_trim() -> bool:
    """把 glibc 持有但未归还内核的空闲页归还; 非 glibc 平台返回 False."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
        return True
    except (OSError, AttributeError):
        return False


def measure_mesh(n: int, device_str: str = "cpu") -> dict:
    """建网格与空间: 只跑 ``_build_problem_space``, 不做任何装配.

    这是 FA/EA/PA/UA 全部装配层级共享的下游前提, 因此它的峰值是与装配层级无关的公共容量上界.
    ``meshbuild`` 阶段为 ``TetrahedronMesh.from_box``; ``space`` 阶段为 ``LagrangeFESpace`` +
    ``TensorFunctionSpace`` + ``IsotropicLinearElasticMaterial``; ``mesh`` 是两者的合成,
    与 stage1/full 产物的 ``mesh_*`` 字段同口径, 可直接横比.
    ``retained`` 是构建结束 (gc 后) 仍驻留的 RSS 净增, 与峰值之比即"建"与"持有"的代价差.

    Parameters
    ----------
    n : int
        网格每方向段数.
    device_str : str, default="cpu"
        计算设备; 本用例只做 CPU RSS 口径.
    """
    if _is_cuda(device_str):
        raise SystemExit("mesh 用例只测 CPU RSS 口径, 不支持 --device cuda")

    _import_fe_stack_cpu()
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TetrahedronMesh
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems.elasticity import DivergenceFreePolynomialElasticity3D

    problem = DivergenceFreePolynomialElasticity3D()
    meter = StageMeter()
    with meter.stage("meshbuild"):
        mesh = TetrahedronMesh.from_box(list(problem.domain), nx=n, ny=n, nz=n)
    with meter.stage("space"):
        scalar = LagrangeFESpace(mesh, p=1, ctype="C")
        vs = TensorFunctionSpace(scalar, shape=(-1, 3))
        material = IsotropicLinearElasticMaterial(
            hypothesis="3D",
            lame_lambda=problem.lam,
            shear_modulus=problem.mu,
            device=bm.get_device(mesh),
        )
    meter.combine("mesh", ("meshbuild", "space"))

    facts = _mesh_facts(mesh, vs)
    Ndof = facts["Ndof"]
    base_kib = meter.records["meshbuild"].before_kib

    gc.collect()
    retained_kib = max(0, cur_rss_kib() - base_kib)
    trimmed = _malloc_trim()
    after_kib = max(0, cur_rss_kib() - base_kib) if trimmed else retained_kib
    assert material is not None  # 保持引用, 使 retained 反映真实驻留

    peak_bytes = meter.peak_bytes("mesh")
    return {
        "pipeline": "mesh_only",
        "stage": 0,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": "rss",
        "n": n,
        **facts,
        **meter.fields(),
        "base_before_MiB": round(base_kib / 1024, 1),
        "mesh_net_KB_per_dof": round(meter.net_bytes("mesh") / Ndof / 1000, 2),
        "mesh_peak_KB_per_dof": round(peak_bytes / Ndof / 1000, 2),
        "retained_MiB": round(retained_kib / 1024, 1),
        "retained_KB_per_dof": round(retained_kib * 1024 / Ndof / 1000, 2),
        "malloc_trim_supported": trimmed,
        "retained_after_trim_MiB": round(after_kib / 1024, 1),
        "retained_after_trim_KB_per_dof": round(after_kib * 1024 / Ndof / 1000, 2),
        "final_peak_MiB": round(peak_bytes / 2**20, 1),
        "final_peak_GiB": round(peak_bytes / 2**30, 2),
        "final_peak_KB_per_dof": round(peak_bytes / Ndof / 1000, 2),
    }


def measure_stage2(
    route: str, n: int, free_inputs: bool = True, device_str: str = "cpu"
) -> dict:
    """阶段 2 合并: 用合成输入 (与网格拓扑一致) 单独测量总刚合并路线的内存单价.

    ``inputs`` 阶段生成路线所需输入 (I/J/V 或合成单刚 K_e), ``merge`` 阶段执行合并;
    ``merge_B_per_triplet`` 是相对 merge 开始时的净增, ``merge_incl_inputs_*`` 把输入也算进去.

    Parameters
    ----------
    route : str
        合并路线 (coalesce/scipy/pattern); pattern 为生产 ``CSRPattern``.
    n : int
        网格每方向段数.
    free_inputs : bool, default=True
        coalesce 路线在 stack 出 indices 后是否立即释放 I/J.
    device_str : str, default="cpu"
        计算设备.
    """
    if _is_cuda(device_str):
        import torch

        device, device_str = _setup_cuda(device_str)
        from soptx.fem.matrix.csr_pattern import assemble_csr, build_csr_pattern

        _, mesh, vs, _ = _build_problem_space(n)
        facts = _mesh_facts(mesh, vs)
        NC, Ndof = facts["NC"], facts["Ndof"]
        ntri = 144 * NC

        if route == "pattern":
            pattern = build_csr_pattern(vs, device=device_str)
            K_e = torch.randn((NC, 12, 12), device=device, dtype=torch.float64)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_bytes = torch.cuda.memory_allocated(device)
            t0 = time.perf_counter()
            K = assemble_csr(K_e, pattern)
            torch.cuda.synchronize(device)
            t_merge = time.perf_counter() - t0
            peak_bytes = torch.cuda.max_memory_allocated(device)
            nnz = int(K.nnz)
        elif route == "coalesce":
            from fealpy.sparse import COOTensor

            c2d = vs.cell_to_dof()
            I = torch.broadcast_to(c2d[:, :, None], (NC, 12, 12)).reshape(-1)
            J = torch.broadcast_to(c2d[:, None, :], (NC, 12, 12)).reshape(-1)
            indices = torch.stack([I, J], dim=0)
            values = torch.randn(NC * 144, device=device, dtype=torch.float64)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_bytes = torch.cuda.memory_allocated(device)
            t0 = time.perf_counter()
            K = COOTensor(indices, values, (Ndof, Ndof)).coalesce()
            torch.cuda.synchronize(device)
            t_merge = time.perf_counter() - t0
            peak_bytes = torch.cuda.max_memory_allocated(device)
            nnz = int(K.nnz)
        else:
            raise ValueError(f"route {route!r} 不支持 GPU.")

        net_bytes = max(0, peak_bytes - baseline_bytes)
        return {
            "stage": 2,
            "route": route,
            "device": torch.cuda.get_device_name(device),
            "device_type": "cuda",
            "memory_kind": "vram",
            "n": n,
            **facts,
            "triplets": ntri,
            "nnz": nnz,
            "triplets_per_nnz": round(ntri / nnz, 3),
            "merge_before_MiB": round(baseline_bytes / 2**20, 1),
            "merge_peak_MiB": round(peak_bytes / 2**20, 1),
            "merge_net_MiB": round(net_bytes / 2**20, 1),
            "merge_B_per_triplet": round(net_bytes / ntri, 2),
            "merge_KB_per_dof": round(net_bytes / Ndof / 1000, 2),
            "t_merge_s": round(t_merge, 4),
        }

    if route not in ROUTE_NAMES:
        raise ValueError(f"未知路线 {route!r}, 可选 {ROUTE_NAMES}")

    cell, NN = load_cell(n)
    NC = cell.shape[0]
    Ndof = 3 * NN
    ntri = 144 * NC
    meter = StageMeter()
    extra: Dict[str, Any] = {}

    if route == "coalesce":
        from fealpy.backend import backend_manager as bm

        bm.set_backend("numpy")
        from fealpy.sparse import COOTensor

        with meter.stage("inputs"):
            I, J, V = full_triplets(cell)
        with meter.stage("merge"):
            indices = np.stack([I, J])
            if free_inputs:
                del I, J
            K = COOTensor(indices, V, spshape=(Ndof, Ndof)).coalesce().tocsr()
        nnz = int(K.nnz)
        extra["free_inputs"] = free_inputs

    elif route == "scipy":
        with meter.stage("inputs"):
            I, J, V = full_triplets(cell)
        with meter.stage("merge"):
            K = sp.coo_matrix((V, (I, J)), shape=(Ndof, Ndof)).tocsr()
        nnz = int(K.nnz)

    else:  # pattern
        from fealpy.backend import backend_manager as bm

        bm.set_backend("numpy")
        from soptx.fem.matrix.csr_pattern import assemble_csr, build_csr_pattern

        space = _CellSpace(cell, NN)
        with meter.stage("inputs"):
            K_e = synthetic_element_matrices(cell)
        with meter.stage("symbolic"):
            pattern = build_csr_pattern(space)
        with meter.stage("numeric"):
            K = assemble_csr(K_e, pattern)
        meter.combine("merge", ("symbolic", "numeric"))
        nnz = int(pattern.nnz)
        # 跨迭代常驻的映射: 标量级槽位基址 + 标量行度数 (回退路径下即自由度级 slot_map)
        resident = pattern.slot_base.nbytes
        if pattern.row_deg is not None:
            resident += pattern.row_deg.nbytes
        extra["resident_map_MiB"] = round(resident / 2**20, 1)
        extra["scalar_nnz"] = int(pattern.scalar_nnz)

    # 保留当前输入与输出对象, 记录合并完成后的常驻 RSS.
    extra["merge_after_MiB"] = round(cur_rss_kib() / 1024, 1)

    merge_net = meter.net_bytes("merge")
    first_stage = "inputs" if "inputs" in meter.records else "merge"
    incl_inputs_net = max(0, meter.peak_bytes("merge") - meter.before_bytes(first_stage))
    return {
        "stage": 2,
        "route": route,
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": "rss",
        "n": n,
        "NC": NC,
        "NN": NN,
        "Ndof": Ndof,
        "triplets": ntri,
        "nnz": nnz,
        "triplets_per_nnz": round(ntri / nnz, 3),
        **extra,
        **meter.fields(),
        "merge_B_per_triplet": round(merge_net / ntri, 2),
        "merge_KB_per_dof": round(merge_net / Ndof / 1000, 2),
        "merge_incl_inputs_B_per_triplet": round(incl_inputs_net / ntri, 2),
        "merge_incl_inputs_KB_per_dof": round(incl_inputs_net / Ndof / 1000, 2),
        "merge_peak_KB_per_dof": round(meter.peak_bytes("merge") / Ndof / 1000, 2),
    }


def measure_full(
    method: str = "fast",
    route: str = "pattern",
    n: int = 32,
    device_str: str = "cpu",
    via_bilinearform: bool = False,
) -> dict:
    """全流程端到端真组装: 真实单刚 (阶段 1) + 真实总刚合并 (阶段 2), 报告各阶段与全程绝对峰值.

    Parameters
    ----------
    method : str
        单刚组装方式 (standard/voigt/fast).
    route : str
        总刚合并路线 (pattern/coalesce/scipy); pattern 与 coalesce 逐行复刻
        ``soptx.fem.BilinearForm.assembly`` 与 FEALPy ``BilinearForm._scalar_assembly`` 的两步.
    n : int
        网格每方向段数.
    device_str : str
        计算设备.
    via_bilinearform : bool
        为 True 时不手拆阶段, 直接调用生产入口 ``soptx.fem.BilinearForm.assembly(method=route)``,
        用于校验手拆路径与生产入口的峰值一致 (仅 pattern/coalesce).
    """
    if route not in FULL_ROUTE_NAMES:
        raise ValueError(f"full 只支持 {FULL_ROUTE_NAMES}, 实得 {route!r}")
    if via_bilinearform and route == "scipy":
        raise ValueError("--via-bilinearform 只支持 pattern / coalesce.")

    from soptx.fem.integrators import LinearElasticIntegrator

    if _is_cuda(device_str):
        import torch
        from soptx.fem import BilinearForm

        device, device_str = _setup_cuda(device_str)
        if route == "scipy":
            raise ValueError("scipy 路线不支持 GPU.")
        _, mesh, vs, material = _build_problem_space(n)
        integrator = LinearElasticIntegrator(material, method=method)
        facts = _mesh_facts(mesh, vs)
        Ndof = facts["Ndof"]

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        t0 = time.perf_counter()
        bform = BilinearForm(vs)
        bform.add_integrator(integrator)
        K = bform.assembly(format="csr", method=route)
        torch.cuda.synchronize(device)
        t_total = time.perf_counter() - t0
        peak_bytes = torch.cuda.max_memory_allocated(device)
        net_bytes = max(0, peak_bytes - baseline_bytes)
        return {
            "pipeline": "full_assembly",
            "device": torch.cuda.get_device_name(device),
            "device_type": "cuda",
            "memory_kind": "vram",
            "method": method,
            "route": route,
            "via_bilinearform": True,
            "n": n,
            **facts,
            "nnz": int(K.nnz),
            "assembly_before_MiB": round(baseline_bytes / 2**20, 1),
            "assembly_peak_MiB": round(peak_bytes / 2**20, 1),
            "assembly_net_MiB": round(net_bytes / 2**20, 1),
            "assembly_net_KB_per_dof": round(net_bytes / Ndof / 1000, 2),
            "final_peak_MiB": round(peak_bytes / 2**20, 1),
            "final_peak_GiB": round(peak_bytes / 2**30, 2),
            "final_peak_KB_per_dof": round(peak_bytes / Ndof / 1000, 2),
            "t_total_s": round(t_total, 4),
        }

    _import_fe_stack_cpu()
    from fealpy.backend import backend_manager as bm

    meter = StageMeter()
    with meter.stage("mesh"):
        _, mesh, vs, material = _build_problem_space(n)
    integrator = LinearElasticIntegrator(material, method=method)
    facts = _mesh_facts(mesh, vs)
    Ndof = facts["Ndof"]
    per_stage: Dict[str, Any] = {}

    if via_bilinearform:
        from soptx.fem import BilinearForm

        with meter.stage("assembly"):
            bform = BilinearForm(vs)
            bform.add_integrator(integrator)
            K = bform.assembly(format="csr", method=route)
        nnz = int(K.nnz)
        first_stage = "assembly"
        t_total = meter.records["assembly"].t_s
    else:
        if route == "pattern":
            # 与 soptx.fem.BilinearForm.assembly(method="pattern") 相同的顺序:
            # 先 build_csr_pattern (符号), 再算单刚, 最后 assemble_csr (数值).
            # 顺序影响峰值: 符号阶段的瞬态若压在 K_e 之上会多算一份 K_e.
            from soptx.fem.matrix.csr_pattern import assemble_csr, build_csr_pattern

            with meter.stage("symbolic"):
                pattern = build_csr_pattern(vs)
            with meter.stage("stage1"):
                K_e = integrator.assembly(vs)
            with meter.stage("numeric"):
                K = assemble_csr(K_e, pattern)
            meter.combine("stage2", ("symbolic", "numeric"))
            nnz = int(pattern.nnz)
        else:
            with meter.stage("stage1"):
                K_e = integrator.assembly(vs)

        if route == "coalesce":
            # 逐行复刻 FEALPy BilinearForm._scalar_assembly + assembly(format="csr")
            from fealpy.sparse import COOTensor

            with meter.stage("stage2"):
                sparse_shape = (Ndof, Ndof)
                M = COOTensor(
                    indices=bm.empty((2, 0), dtype=vs.itype, device=bm.get_device(vs)),
                    values=bm.empty((0,), dtype=vs.ftype, device=bm.get_device(vs)),
                    spshape=sparse_shape,
                )
                c2d = vs.cell_to_dof()
                local_shape = K_e.shape[-3:]
                I = bm.broadcast_to(c2d[:, :, None], local_shape)
                J = bm.broadcast_to(c2d[:, None, :], local_shape)
                indices = bm.stack([I.ravel(), J.ravel()], axis=0)
                values = bm.reshape(K_e, (-1,))
                M = M.add(COOTensor(indices, values, sparse_shape))
                # FEALPy 中 K_e / I / J 是 _scalar_assembly 的局部变量, 返回后即释放, coalesce 时已不在.
                del I, J, indices, values, K_e
                K = M.coalesce().tocsr()
            nnz = int(K.nnz)
        elif route == "scipy":
            with meter.stage("stage2"):
                c2d = np.asarray(vs.cell_to_dof())
                local_shape = tuple(int(s) for s in K_e.shape[-3:])
                I = np.broadcast_to(c2d[:, :, None], local_shape).ravel()
                J = np.broadcast_to(c2d[:, None, :], local_shape).ravel()
                K = sp.coo_matrix(
                    (np.asarray(K_e).ravel(), (I, J)), shape=(Ndof, Ndof)
                ).tocsr()
                del I, J
            nnz = int(K.nnz)

        # 组装区间起点: pattern 路线从 build_csr_pattern 开始 (与生产入口相同), 其余从单刚开始
        first_stage = "symbolic" if route == "pattern" else "stage1"
        t_total = meter.records["stage1"].t_s + meter.records["stage2"].t_s
        per_stage = {
            "Ke_theory_MiB": round(facts["NC"] * 144 * 8 / 2**20, 1),
            "stage1_KB_per_dof": round(meter.net_bytes("stage1") / Ndof / 1000, 2),
            "stage2_KB_per_dof": round(meter.net_bytes("stage2") / Ndof / 1000, 2),
            "stage2_B_per_triplet": round(meter.net_bytes("stage2") / (144 * facts["NC"]), 2),
        }

    # 保留当前装配对象, 记录求解前的常驻 RSS 与最高峰值所在阶段.
    assembly_after_mib = round(cur_rss_kib() / 1024, 1)
    final_peak_stage = max(meter.records, key=lambda name: meter.records[name].peak_kib)
    final_peak_bytes = meter.max_peak_kib() * 1024
    assembly_net = max(0, final_peak_bytes - meter.before_bytes(first_stage))
    return {
        "pipeline": "full_assembly",
        "device": "CPU",
        "device_type": "cpu",
        "memory_kind": "rss",
        "method": method,
        "route": route,
        "via_bilinearform": via_bilinearform,
        "n": n,
        **facts,
        "nnz": nnz,
        **per_stage,
        **meter.fields(),
        "assembly_after_MiB": assembly_after_mib,
        "final_peak_stage": final_peak_stage,
        "assembly_net_MiB": round(assembly_net / 2**20, 1),
        "assembly_net_KB_per_dof": round(assembly_net / Ndof / 1000, 2),
        "final_peak_MiB": round(final_peak_bytes / 2**20, 1),
        "final_peak_GiB": round(final_peak_bytes / 2**30, 2),
        "final_peak_KB_per_dof": round(final_peak_bytes / Ndof / 1000, 2),
        "t_total_s": round(t_total, 3),
    }


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
    dev_type = out.get("device_type", "cpu")
    dev_label = f"{out.get('device', 'CPU')} (PyTorch CUDA, VRAM)" if dev_type == "cuda" else "CPU (RSS)"
    mesh_line = f"TetrahedronMesh (grid = {n}^3) | {nc:,} cells | {ndof:,} DOFs"

    if out.get("pipeline") == "full_assembly":
        print(f"\n● [full-assembly] {PROBLEM_NAME}")
        print(f"  ├── Mesh & DOFs   : {mesh_line} | nnz = {out.get('nnz', 0):,}")
        mode = "BilinearForm.assembly" if out.get("via_bilinearform") else "手拆阶段"
        print(f"  ├── Configuration : method = {out.get('method')} | route = {out.get('route')} | {mode} | device = {dev_label}")
        if "mesh_peak_MiB" in out:
            print(f"  ├── {_stage_line(out, 'mesh', 'Mesh & Space')}")
        if "stage1_peak_MiB" in out:
            print(f"  ├── {_stage_line(out, 'stage1', 'Stage 1 (K_e)', 'stage1_KB_per_dof')}")
            print(f"  ├── {_stage_line(out, 'stage2', 'Stage 2 (merge)', 'stage2_KB_per_dof')}")
            if "symbolic_peak_MiB" in out:
                print(f"  │   ├── {_stage_line(out, 'symbolic', 'symbolic')}")
                print(f"  │   └── {_stage_line(out, 'numeric', 'numeric')}")
        if "assembly_peak_MiB" in out:
            print(f"  ├── {_stage_line(out, 'assembly', 'Assembly', 'assembly_net_KB_per_dof')}")
        print(f"  ├── Time Elapsed  : {_fmt_s(out.get('t_total_s'))} (不含网格构建)")
        print(f"  ├── Assembly Net  : {_fmt_mib(out.get('assembly_net_MiB'))} ({out.get('assembly_net_KB_per_dof', 0):.2f} KB/dof)")
        print(f"  └── Absolute Peak : {_fmt_mib(out.get('final_peak_MiB'))} ({out.get('final_peak_KB_per_dof', 0):.2f} KB/dof)\n")

    elif out.get("pipeline") == "mesh_only":
        print(f"\n● [mesh-build] {PROBLEM_NAME}")
        print(f"  ├── Mesh & DOFs   : {mesh_line} | {out.get('NN', 0):,} nodes")
        print(f"  ├── Configuration : device = {dev_label} | 只建网格与空间, 不做装配")
        print(f"  ├── {_stage_line(out, 'meshbuild', 'TetrahedronMesh')}")
        print(f"  ├── {_stage_line(out, 'space', 'Space & Material')}")
        print(f"  ├── {_stage_line(out, 'mesh', 'Mesh & Space', 'mesh_net_KB_per_dof')}")
        trim = (
            f" | malloc_trim 后 {_fmt_mib(out.get('retained_after_trim_MiB'))}"
            if out.get("malloc_trim_supported") else ""
        )
        print(f"  ├── Retained      : {_fmt_mib(out.get('retained_MiB'))} ({out.get('retained_KB_per_dof', 0):.2f} KB/dof){trim}")
        print(f"  └── Absolute Peak : {_fmt_mib(out.get('final_peak_MiB'))} ({out.get('final_peak_KB_per_dof', 0):.2f} KB/dof)\n")

    elif out.get("pipeline") == "stage1_probe":
        pc = out.get("peak_call", {})
        print(f"\n● [stage1-probe] {PROBLEM_NAME}")
        print(f"  ├── Mesh & DOFs   : {mesh_line}")
        print(f"  ├── Configuration : method = {out.get('method')} | tracemalloc 打桩 | {_fmt_s(out.get('t_assembly_s'))}")
        print(f"  ├── Traced Peak   : {_fmt_mib(out.get('traced_peak_MiB'))} | 结束存活 {_fmt_mib(out.get('traced_live_end_MiB'))}")
        print(f"  ├── Peak Call     : #{pc.get('index')} {pc.get('call')} @ {pc.get('site')} -> {pc.get('out_shape')}")
        print(f"  │   ├── live before : {_fmt_mib(pc.get('live_before_MiB'))}")
        print(f"  │   ├── peak in call: {_fmt_mib(pc.get('peak_in_MiB'))}")
        print(f"  │   └── transient   : {_fmt_mib(pc.get('transient_MiB'))} (调用内中间量)")
        print(f"  ├── Live Composition @ peak call:")
        for row in out.get("peak_live_composition", [])[:12]:
            print(f"  │   ├── {row['site']:<40} {row['MiB']:>9,.1f} MiB  x{row['count']}")
        print(f"  └── Calls         : {len(out.get('calls', []))} 次被测调用 (完整明细见 JSON)\n")

    elif out.get("stage") == 1:
        print(f"\n● [element-stiffness] {PROBLEM_NAME}")
        print(f"  ├── Mesh & DOFs   : {mesh_line}")
        print(f"  ├── Assembly      : method = {out.get('method')} | device = {dev_label} | K_e theory = {out.get('Ke_theory_MiB', 0):,.1f} MiB")
        if "mesh_peak_MiB" in out:
            print(f"  ├── {_stage_line(out, 'mesh', 'Mesh & Space')}")
        print(f"  ├── {_stage_line(out, 'stage1', 'Stage 1 (K_e)')}")
        print(f"  └── Unit Cost     : {out.get('stage1_KB_per_dof', 0):.2f} KB/dof net ({out.get('stage1_B_per_dof', 0):,.1f} B/dof)\n")

    elif out.get("stage") == 2:
        ntri = out.get("triplets", 0)
        print(f"\n● [global-merge] {PROBLEM_NAME}")
        print(f"  ├── Mesh & DOFs   : {mesh_line} | {ntri:,} triplets")
        print(f"  ├── Sparse Merge  : route = {out.get('route')} | nnz = {out.get('nnz', 0):,} | device = {dev_label}")
        if "inputs_peak_MiB" in out:
            print(f"  ├── {_stage_line(out, 'inputs', 'Inputs')}")
        print(f"  ├── {_stage_line(out, 'merge', 'Merge')}")
        if "symbolic_peak_MiB" in out:
            print(f"  │   ├── {_stage_line(out, 'symbolic', 'symbolic')}")
            print(f"  │   └── {_stage_line(out, 'numeric', 'numeric')}")
        cost = f"{out.get('merge_B_per_triplet', 0):.2f} B/triplet ({out.get('merge_KB_per_dof', 0):.2f} KB/dof)"
        if "merge_incl_inputs_B_per_triplet" in out:
            cost += f" | 含输入 {out['merge_incl_inputs_B_per_triplet']:.2f} B/triplet ({out['merge_incl_inputs_KB_per_dof']:.2f} KB/dof)"
        print(f"  └── Unit Cost     : {cost}\n")


def display_width(text: str) -> int:
    """计算包含中文字符的真实终端显示字宽 (基于 Unicode East Asian Width 规范)."""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def pad(text: str, width: int) -> str:
    """按显示字宽向右填充空格."""
    return text + " " * max(0, width - display_width(text))


def _read_proc_status_kib(pid: int) -> tuple[int | None, int | None]:
    """读取 Linux 进程的当前 RSS 与绝对峰值 RSS."""
    values: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as file:
            for line in file:
                name = line.partition(":")[0]
                if name in ("VmRSS", "VmHWM"):
                    values[name] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None, None
    return values.get("VmRSS"), values.get("VmHWM")


def _read_meminfo_kib() -> tuple[int | None, int | None]:
    """读取 Linux 整机总内存与当前可用内存."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                name = line.partition(":")[0]
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _read_process_cpu_ticks(pid: int) -> int | None:
    """读取 Linux 进程累计消耗的用户态与内核态 CPU tick."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        return int(fields[11]) + int(fields[12])
    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        ValueError,
        IndexError,
    ):
        return None


def _format_kib(value: int | None) -> str:
    """把 KiB 格式化为适合实时看板展示的容量字符串."""
    if value is None:
        return "--"
    if value >= 2**20:
        return f"{value / 2**20:.2f} GiB"
    return f"{value / 2**10:.1f} MiB"


@dataclass
class WorkerStats:
    """父进程对独立 Worker 资源占用的跨样本观测统计.

    ``last_*`` 为进程存活时最后一次观测值 (进程退出后保留, 不清空);
    ``max_*`` / ``min_*`` 为整个生命周期内的极值. 采样间隔有限, 极值只是真实峰值的下界.
    """

    samples: int = 0
    last_rss_kib: int | None = None
    last_peak_kib: int | None = None
    max_rss_kib: int | None = None
    max_peak_kib: int | None = None
    last_available_kib: int | None = None
    min_available_kib: int | None = None
    last_cpu_percent: float | None = None

    def update(
        self,
        rss_kib: int | None,
        peak_kib: int | None,
        available_kib: int | None,
        cpu_percent: float | None,
    ) -> bool:
        """吸收一次采样; 返回本次采样时 Worker 是否仍存活 (僵尸/已退出进程读不到 VmRSS)."""
        alive = rss_kib is not None
        if alive:
            self.samples += 1
            self.last_rss_kib = rss_kib
            self.max_rss_kib = rss_kib if self.max_rss_kib is None else max(self.max_rss_kib, rss_kib)
            if peak_kib is not None:
                self.last_peak_kib = peak_kib
                self.max_peak_kib = peak_kib if self.max_peak_kib is None else max(self.max_peak_kib, peak_kib)
            if cpu_percent is not None:
                self.last_cpu_percent = cpu_percent
            if available_kib is not None:
                self.last_available_kib = available_kib
                self.min_available_kib = (
                    available_kib if self.min_available_kib is None else min(self.min_available_kib, available_kib)
                )
        return alive

    def to_mib_dict(self) -> dict[str, Any]:
        """以 MiB 为单位导出, 供失败侧车 JSON 落盘."""

        def mib(v: int | None) -> float | None:
            return None if v is None else round(v / 1024, 1)

        return {
            "samples": self.samples,
            "last_rss_MiB": mib(self.last_rss_kib),
            "last_peak_rss_MiB": mib(self.last_peak_kib),
            "max_rss_MiB": mib(self.max_rss_kib),
            "max_peak_rss_MiB": mib(self.max_peak_kib),
            "last_system_available_MiB": mib(self.last_available_kib),
            "min_system_available_MiB": mib(self.min_available_kib),
            "last_cpu_percent": None if self.last_cpu_percent is None else round(self.last_cpu_percent, 1),
        }


def _runtime_monitor_lines(
    label: str,
    pid: int,
    elapsed: float,
    stats: WorkerStats,
    alive: bool,
    available_kib: int | None,
    total_kib: int | None,
) -> list[str]:
    """构造独立 Worker 的实时资源表格; 进程退出后展示死前最后一次观测值."""
    used_percent = None
    if total_kib and available_kib is not None:
        used_percent = 100.0 * (total_kib - available_kib) / total_kib
    suffix = "" if alive else " (last seen)"
    cpu = stats.last_cpu_percent

    rows = [
        ("Case", label),
        ("Worker PID", str(pid)),
        ("Elapsed", f"{elapsed:.1f} s"),
        ("CPU", ("--" if cpu is None else f"{cpu:.1f}%") + suffix),
        ("Current RSS", _format_kib(stats.last_rss_kib) + suffix),
        ("Peak RSS (VmHWM)", _format_kib(stats.last_peak_kib) + suffix),
        ("Peak RSS (observed)", _format_kib(stats.max_rss_kib)),
        ("System Available", _format_kib(available_kib)),
        ("Min System Available", _format_kib(stats.min_available_kib)),
        (
            "System Memory Used",
            "--" if used_percent is None else f"{used_percent:.1f}%",
        ),
    ]
    key_width = max(display_width(key) for key, _ in rows)
    value_width = max(display_width(value) for _, value in rows)
    inner_width = key_width + value_width + 5
    title = " Runtime Monitor " if alive else " Runtime Monitor (worker exited) "
    lines = [f"┌─{title}{'─' * (inner_width - display_width(title) - 1)}┐"]
    for key, value in rows:
        lines.append(f"│ {pad(key, key_width)} : {pad(value, value_width)} │")
    lines.append(f"└{'─' * inner_width}┘")
    return lines


def _wait_with_monitor(
    process: subprocess.Popen[str],
    label: str,
    started: float,
    interval: float,
) -> tuple[int, WorkerStats]:
    """在父进程中采样并原位刷新 Worker 的资源占用表格, 返回退出码与跨样本观测统计."""
    interactive = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    clock_ticks = os.sysconf("SC_CLK_TCK")
    previous_ticks: int | None = None
    previous_time: float | None = None
    rendered_lines = 0
    next_plain_update = 0.0
    stats = WorkerStats()

    def render(alive: bool, now: float) -> None:
        nonlocal rendered_lines, next_plain_update
        total_kib, available_kib = _read_meminfo_kib()
        lines = _runtime_monitor_lines(
            label, process.pid, now - started, stats, alive, available_kib, total_kib
        )
        if interactive:
            if rendered_lines:
                sys.stdout.write(f"\x1b[{rendered_lines}F")
            for line in lines:
                sys.stdout.write(f"\x1b[2K{line}\n")
            sys.stdout.flush()
            rendered_lines = len(lines)
        elif not alive or now >= next_plain_update:
            print(
                f"[monitor] pid={process.pid} elapsed={now - started:.1f}s "
                f"rss={_format_kib(stats.last_rss_kib)} peak={_format_kib(stats.last_peak_kib)} "
                f"observed_max={_format_kib(stats.max_rss_kib)} "
                f"min_avail={_format_kib(stats.min_available_kib)}"
                + ("" if alive else " (worker exited)"),
                flush=True,
            )
            next_plain_update = now + max(5.0, interval)

    while process.poll() is None:
        now = time.perf_counter()
        ticks = _read_process_cpu_ticks(process.pid)
        cpu_percent = None
        if (
            ticks is not None
            and previous_ticks is not None
            and previous_time is not None
            and now > previous_time
        ):
            cpu_percent = 100.0 * (ticks - previous_ticks) / clock_ticks / (
                now - previous_time
            )

        rss_kib, peak_kib = _read_proc_status_kib(process.pid)
        _, available_kib = _read_meminfo_kib()
        alive = stats.update(rss_kib, peak_kib, available_kib, cpu_percent)
        if not alive:
            # 已成僵尸 (status 里没有 VmRSS) 但尚未被 poll 回收: 交给循环外的最终一帧.
            break
        render(True, now)

        previous_ticks = ticks
        previous_time = now
        try:
            process.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            pass

    process.wait()
    # 进程已退出: 再刷一帧, 保留死前最后一次观测值而不是打 "--".
    render(False, time.perf_counter())
    return int(process.returncode or 0), stats


def _describe_exit(returncode: int) -> tuple[str, str | None, bool]:
    """把退出码翻译成可读说明; 返回 (说明, 信号名, 是否疑似 OOM)."""
    if returncode >= 0:
        return f"退出码 {returncode}", None, False
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    suspected_oom = name == "SIGKILL"
    text = f"退出码 {returncode} (被信号 {name} 终止"
    if suspected_oom:
        text += ", 疑似被内核 OOM killer 杀死"
    return text + ")", name, suspected_oom


_OOM_NEEDLES = ("invoked oom-killer", "oom-kill:", "Killed process")


def _dmesg_snapshot() -> list[str] | None:
    """读取当前 dmesg 全文 (按行); 不可用或无权限时返回 None."""
    try:
        completed = subprocess.run(
            ["dmesg"], capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.splitlines()


def _dmesg_new_oom_lines(before: list[str] | None) -> list[str] | None:
    """返回自 ``before`` 快照之后新增的 OOM 相关 dmesg 行.

    不按 PID 匹配: WSL2 各发行版运行在独立 PID 命名空间, 内核日志里的 pid 是全局编号,
    与发行版内 ``ps`` / ``Popen.pid`` 看到的不同; dmesg 时间戳在 WSL 里也不可靠.
    故以 "启动前快照 -> 退出后新增" 归因. dmesg 不可用时返回 None.
    """
    after = _dmesg_snapshot()
    if after is None or before is None:
        return None
    if after[: len(before)] == before:
        new_lines = after[len(before):]
    else:
        # 环形缓冲已翻转: 退回到快照最后一行在新输出中的位置.
        new_lines = after
        if before:
            last = before[-1]
            for idx in range(len(after) - 1, -1, -1):
                if after[idx] == last:
                    new_lines = after[idx + 1:]
                    break
    return [
        line.strip()
        for line in new_lines
        if any(needle in line for needle in _OOM_NEEDLES)
    ]


def _killed_anon_rss_kib(lines: list[str]) -> int | None:
    """从 "Out of memory: Killed process ... anon-rss:NNNkB" 行解析被杀进程的 anon-rss (KiB)."""
    import re

    for line in reversed(lines):
        match = re.search(r"Killed process .*?anon-rss:(\d+)kB", line)
        if match:
            return int(match.group(1))
    return None


def _failed_sidecar_path(artifact_path: Path) -> Path:
    """失败侧车文件路径: 与产物同名, 后缀改为 .failed.json."""
    return artifact_path.with_name(artifact_path.stem + ".failed.json")


def _report_failure(
    *,
    label: str,
    argv: list[str],
    artifact_path: Path,
    pid: int,
    returncode: int,
    elapsed: float,
    started_at: str,
    stats: WorkerStats | None,
    dmesg_before: list[str] | None,
) -> None:
    """打印失败诊断并把它落盘到 .failed.json 侧车."""
    description, signal_name, suspected_oom = _describe_exit(returncode)
    print(f"  失败: {description}, 用时 {elapsed:.1f} s")

    if stats is not None and stats.samples:
        print(
            f"  死前观测 (采样 {stats.samples} 次, 为真实峰值下界): "
            f"最后 RSS {_format_kib(stats.last_rss_kib)} | "
            f"观测最大 RSS {_format_kib(stats.max_rss_kib)} | "
            f"最小系统可用 {_format_kib(stats.min_available_kib)}"
        )

    dmesg_lines = _dmesg_new_oom_lines(dmesg_before)
    kernel_anon_rss_kib: int | None = None
    if dmesg_lines is None:
        print("  dmesg: 不可读 (无权限或不可用), 可手动执行 `sudo dmesg | grep -i \"killed process\"`")
    elif dmesg_lines:
        for line in dmesg_lines:
            if "invoked oom-killer" in line:
                continue  # 触发者行信息量低, 只留 oom-kill 与 Killed process 两行.
            print(f"  dmesg: {line}")
        kernel_anon_rss_kib = _killed_anon_rss_kib(dmesg_lines)
        if kernel_anon_rss_kib is not None:
            note = f"  内核记录被杀进程 anon-rss {_format_kib(kernel_anon_rss_kib)}"
            if stats is not None and stats.max_rss_kib:
                ratio = kernel_anon_rss_kib / stats.max_rss_kib
                verdict = "一致" if 0.95 <= ratio <= 1.05 else "不一致, 请核对是否为本 Worker"
                note += f", 与监控观测峰值 {_format_kib(stats.max_rss_kib)} {verdict}"
            note += " (dmesg 中的 pid 为内核全局编号, 与发行版内 PID 不同, 属正常现象)"
            print(note)
    elif suspected_oom:
        print("  dmesg: 运行期间无新增 OOM 记录 (SIGKILL 可能来自其他来源, 如手动 kill -9)")

    total_kib, _ = _read_meminfo_kib()
    payload: dict[str, Any] = {
        "case": label,
        "argv": argv,
        "artifact": artifact_path.name,
        "worker_pid": pid,
        "returncode": returncode,
        "signal": signal_name,
        "suspected_oom": suspected_oom,
        "started_at": started_at,
        "elapsed_s": round(elapsed, 1),
        "system_total_MiB": None if total_kib is None else round(total_kib / 1024, 1),
        "observed": None if stats is None else stats.to_mib_dict(),
        "kernel_killed_anon_rss_MiB": (
            None if kernel_anon_rss_kib is None else round(kernel_anon_rss_kib / 1024, 1)
        ),
        "dmesg": dmesg_lines,
    }
    sidecar = _failed_sidecar_path(artifact_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  失败记录已落盘 -> {sidecar}")


def command_list(cases: tuple[config.Case, ...], figure: dict) -> int:
    """打印已注册算例及其配置参数列表."""
    header = ("case-id", "mesh", "grid", "problem", "method", "route", "device")
    rows = []
    for c in cases:
        n_val = c.extra.get("n")
        mesh_type = str(c.extra.get("mesh_type", "TetrahedronMesh"))
        grid_str = str(c.extra.get("grid", f"{n_val}^3" if n_val else "-"))
        problem_str = str(c.extra.get("problem", "DivergenceFreePolynomialElasticity3D"))
        method_str = str(c.extra.get("method", "-"))
        route_str = str(c.extra.get("route", "-"))
        device_str = str(c.extra.get("device", "cpu"))
        rows.append((c.id, mesh_type, grid_str, problem_str, method_str, route_str, device_str))

    widths = [
        max(display_width(row[i]) for row in (header, *rows)) for i in range(len(header))
    ]
    print()
    print("  ".join(pad(value, widths[i]) for i, value in enumerate(header)).rstrip())
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows:
        print("  ".join(pad(value, widths[i]) for i, value in enumerate(row)).rstrip())
    print()
    return 0


def artifact_name(kind: str, parts: Sequence[str], n: int, device: str, bform: bool = False) -> str:
    """产物文件名: <kind>_<parts>_n<N>[_cuda][_bform].json; 非 CPU 设备加 _cuda 后缀以免与 CPU 结论混淆."""
    name = f"{kind}_{'_'.join(parts)}_n{n}"
    if device != "cpu":
        name += "_cuda"
    if bform:
        name += "_bform"
    return name + ".json"


def _expand(requested: str | None, is_all: bool, all_values: Sequence[str], default: str) -> list[str]:
    """把 --method/--route 覆盖值展开为具体列表: 'all' 或 --all 未指定 → 全部, 否则单值."""
    if requested == "all" or (is_all and not requested):
        return list(all_values)
    if requested:
        return [requested]
    return [default]


def resolve_runs(
    cases: tuple[config.Case, ...],
    is_all: bool,
    overrides: dict[str, Any],
) -> list[tuple[str, list[str], Path, str, dict[str, str] | None]]:
    """将选中的 Case 与参数覆盖解析为具体的单次子进程执行计划."""
    runs: list[tuple[str, list[str], Path, str, dict[str, str] | None]] = []
    for case in cases:
        n = overrides.get("n", case.extra.get("n", 32))
        device = overrides.get("device", case.extra.get("device", "cpu"))
        via_bform = bool(overrides.get("via_bilinearform", False))
        env = case.subprocess_env()
        common = [sys.executable, str(case.script_path), "--worker"]
        tail = ["--n", str(n), "--device", device]

        if case.panel == "stage1":
            methods = _expand(
                overrides.get("method"), is_all,
                case.extra.get("methods", list(METHOD_NAMES)), case.extra.get("method", "fast"),
            )
            for m in methods:
                out_p = config.OUTPUT_DIR / artifact_name("stage1", [m], n, device)
                argv = [*common, "--stage1", "--method", m, *tail, "--output", str(out_p)]
                label = f"{case.id} [{m}]"
                summary = f"{case.summary} (method={m}, n={n}, device={device})"
                runs.append((label, argv, out_p, summary, env))

        elif case.panel == "stage2":
            routes = _expand(
                overrides.get("route"), is_all,
                case.extra.get("routes", list(ROUTE_NAMES)), case.extra.get("route", "pattern"),
            )
            if device != "cpu":
                routes = [r for r in routes if r in ("coalesce", "pattern")]
            for r in routes:
                out_p = config.OUTPUT_DIR / artifact_name("stage2", [r], n, device)
                argv = [*common, "--stage2", "--route", r, *tail, "--output", str(out_p)]
                label = f"{case.id} [{r}]"
                summary = f"{case.summary} (route={r}, n={n}, device={device})"
                runs.append((label, argv, out_p, summary, env))

        elif case.panel == "mesh":
            out_p = config.OUTPUT_DIR / artifact_name("mesh", ["build"], n, device)
            argv = [*common, "--mesh", *tail, "--output", str(out_p)]
            summary = f"{case.summary} (n={n}, device={device})"
            runs.append((case.id, argv, out_p, summary, env))

        elif case.panel == "full":
            methods = _expand(
                overrides.get("method"), False,
                case.extra.get("methods", list(METHOD_NAMES)), case.extra.get("method", "fast"),
            )
            routes = _expand(
                overrides.get("route"), is_all,
                case.extra.get("routes", list(FULL_ROUTE_NAMES)), case.extra.get("route", "pattern"),
            )
            if device != "cpu" or via_bform:
                routes = [r for r in routes if r != "scipy"]
            for m in methods:
                for r in routes:
                    out_p = config.OUTPUT_DIR / artifact_name("full", [m, r], n, device, bform=via_bform)
                    argv = [*common, "--full", "--method", m, "--route", r, *tail]
                    if via_bform:
                        argv.append("--via-bilinearform")
                    argv += ["--output", str(out_p)]
                    label = f"{case.id} [{m}+{r}{'+bform' if via_bform else ''}]"
                    summary = f"{case.summary} (method={m}, route={r}, n={n}, device={device})"
                    runs.append((label, argv, out_p, summary, env))

    return runs


def command_run(
    cases: tuple[config.Case, ...],
    *,
    is_all: bool = False,
    check_only: bool,
    skip_existing: bool,
    monitor: bool,
    monitor_interval: float,
    overrides: dict[str, Any] | None = None,
) -> int:
    """逐个以独立子进程运行解析后的执行计划."""
    runs = resolve_runs(cases, is_all=is_all, overrides=overrides or {})
    failed = 0
    total = len(runs)
    for index, (label, argv, artifact_path, summary, env) in enumerate(runs, start=1):
        prefix = f"[{index}/{total}] {label}"
        if skip_existing and artifact_path.is_file():
            print(f"{prefix}: 产物已存在, 跳过")
            continue

        printable = " ".join(argv)
        if check_only:
            print(f"{prefix}: {printable}")
            continue

        if total > 1:
            print(f"\n{prefix}: {summary}", flush=True)
        started = time.perf_counter()
        started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        stats: WorkerStats | None = None
        dmesg_before = _dmesg_snapshot()
        process = subprocess.Popen(
            argv,
            cwd=config.REPOSITORY_ROOT,
            env=env,
            stdout=subprocess.PIPE if monitor else None,
            text=True,
        )
        try:
            if monitor:
                returncode, stats = _wait_with_monitor(
                    process,
                    label,
                    started,
                    monitor_interval,
                )
                worker_stdout, _ = process.communicate()
                if worker_stdout:
                    print(worker_stdout, end="")
            else:
                returncode = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        elapsed = time.perf_counter() - started

        if returncode != 0:
            failed += 1
            _report_failure(
                label=label,
                argv=argv,
                artifact_path=artifact_path,
                pid=process.pid,
                returncode=returncode,
                elapsed=elapsed,
                started_at=started_at,
                stats=stats,
                dmesg_before=dmesg_before,
            )
        elif not artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {artifact_path}")
        else:
            # 本次成功: 清掉同名的历史失败侧车, 避免与新产物并存造成误读.
            stale = _failed_sidecar_path(artifact_path)
            if stale.is_file():
                stale.unlink()

    if failed:
        print(f"\n{failed} 个任务执行失败。")
    return failed


# -----------------------------------------------------------------------------
# 3. 主入口与参数路由
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
        description="fa_assembly_capability 实验图面驱动: 测量各合并路线的内存开销与规模极限",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点及产物状态")
    parser.add_argument("--all", action="store_true", help="跑全部工况")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id (如 --cases element-stiffness)")
    parser.add_argument("--case", help="指定单个 case id (等价于 --cases <id>)")
    parser.add_argument(
        "--panel", choices=config.PANELS, help="只跑指定阶段 (stage1/stage2/full/mesh) 数据点"
    )
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="运行时实时显示独立 Worker 的 CPU 与内存占用",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="实时监控刷新间隔，单位为秒 (默认 0.5)",
    )

    # 2. 工况动态覆盖参数 (Overrides)
    parser.add_argument("-n", "--n", "--grid", dest="n", type=int, default=None, help="动态覆盖网格剖分段数 (如 -n 32 或 --grid 32)")
    parser.add_argument(
        "--device", type=str, default="cpu", help="指定计算设备 ('cpu' 或 'cuda' 等, 默认 'cpu'); 非 cpu 产物名加 _cuda 后缀"
    )
    parser.add_argument(
        "--method", choices=METHOD_NAMES + ("all",), default=None, help="指定或覆盖单刚算法"
    )
    parser.add_argument(
        "--route", choices=ROUTE_NAMES + ("all",), default=None,
        help="指定或覆盖合并路线 (full 只接受 pattern/coalesce/scipy)",
    )
    parser.add_argument(
        "--via-bilinearform", action="store_true",
        help="full: 不手拆阶段, 直接调用生产入口 soptx.fem.BilinearForm.assembly 作一致性校验 (产物名加 _bform)",
    )

    # 3. Worker 测量层底层参数 (供子进程调用)
    parser.add_argument("--worker", action="store_true", help="进入子进程 worker 测量模式")
    parser.add_argument("--stage1", action="store_true", help="阶段 1 单刚测量")
    parser.add_argument("--stage1-probe", action="store_true", help="阶段 1 峰值归因: tracemalloc 逐调用打桩")
    parser.add_argument("--stage2", action="store_true", help="阶段 2 合并路线测量 (需 --route)")
    parser.add_argument("--full", action="store_true", help="全流程端到端真组装测量 (阶段 1 + 阶段 2)")
    parser.add_argument("--mesh", action="store_true", help="只建网格与空间的公共前提测量 (不做装配)")
    parser.add_argument("--output", type=Path, default=None, help="产物落盘路径")

    args = parser.parse_args(argv)
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval 必须大于 0")

    # ------------------------------------------------ Worker 测量分支
    if args.worker or args.stage1 or args.stage1_probe or args.stage2 or args.full or args.mesh:
        if args.n is None:
            parser.error("Worker 模式必须指定 --n")
        if args.method == "all" or args.route == "all":
            parser.error("Worker 模式不接受 --method all / --route all, 由调度层展开")
        if args.mesh:
            out = measure_mesh(args.n, device_str=args.device)
        elif args.full:
            out = measure_full(
                args.method or "fast",
                args.route or "pattern",
                args.n,
                device_str=args.device,
                via_bilinearform=args.via_bilinearform,
            )
        elif args.stage1_probe:
            out = measure_stage1_probe(args.method or "fast", args.n)
        elif args.stage1:
            out = measure_stage1(args.method or "fast", args.n, device_str=args.device)
        elif args.stage2:
            if args.route is None:
                parser.error("--stage2 必须指定 --route")
            out = measure_stage2(args.route, args.n, device_str=args.device)
        else:
            parser.error("Worker 模式需指定 --full, --stage1, --stage1-probe, --stage2 或 --mesh")

        print_dashboard(out)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
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
        return command_list(cases, figure)

    # 组装与解析 case_ids
    target_case_ids: list[str] = []
    panel_filter: str | None = args.panel
    known_ids = {c.id for c in cases}
    is_all = args.all

    alias_map = {
        "elem-stiff": "element-stiffness",
        "merge": "global-merge",
        "matrix-merge": "global-merge",
        "fa-full": "full-assembly",
        "mesh-only": "mesh-build",
        "build-mesh": "mesh-build",
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

    # 显式校验必填参数
    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case/--cases/--panel/--all 指定要运行的工况或板块。\n"
            "  常用示例:\n"
            "    python run.py --case element-stiffness --method all --grid 32\n"
            "    python run.py --case full-assembly --route all --grid 80 --monitor\n"
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

    # 组装参数覆盖
    overrides: dict[str, Any] = {}
    if args.n is not None:
        overrides["n"] = args.n
    if args.device != "cpu":
        overrides["device"] = args.device
    if args.method is not None:
        overrides["method"] = args.method
    if args.route is not None:
        overrides["route"] = args.route
    if args.via_bilinearform:
        overrides["via_bilinearform"] = True

    failed = command_run(
        selected,
        is_all=is_all,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
        overrides=overrides if overrides else None,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
