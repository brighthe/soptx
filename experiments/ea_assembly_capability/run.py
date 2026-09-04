# -*- coding: utf-8 -*-
"""EA (Element Assembly) 单元装配无矩阵算子与规模能力统一驱动.

本模块自包含实现:
1. Worker 测量层:
   - 阶段 1 (cache): 单元刚度张量显式缓存与静态单价测量
   - 阶段 2 (matvec): 单次及批量 MatVec (A @ x) 瞬态显存与吞吐测量
   - 全流程 (solve): EA 算子搭载无预条件 CG 线性求解端到端峰值内存与容量天花板
2. 调度与编排层: 读取 cases.toml, 启动独立子进程跑指定数据点
3. 终端看板: Style B (Modern Tree Card) 树状卡片输出
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config  # noqa: E402

T_TET4 = 288
DEFAULT_MEMORY_TOTAL = 47.04 * 2**30  # 本机可用内存上限 (47.04 GiB)
DEFAULT_GPU_VRAM_TOTAL = 16.0 * 2**30  # 本机 RTX 5080 显存上限 (16.0 GiB)


# -----------------------------------------------------------------------------
# 1. 测量与物理构件 (Worker 核心)
# -----------------------------------------------------------------------------

def get_peak_rss_bytes() -> int:
    """获取当前进程生命周期的最高内存水位 (ru_maxrss)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss
    return usage.ru_maxrss * 1024


def detect_device_display(device_str: str) -> str:
    """根据 device_str 与硬件状态生成准确的设备展示名称."""
    dev = device_str.lower()
    if dev in ("cpu", "none"):
        return "CPU"
    try:
        import torch
        if torch.cuda.is_available() and ("cuda" in dev or dev == "gpu"):
            dev_idx = 0
            if ":" in dev:
                try:
                    dev_idx = int(dev.split(":")[1])
                except Exception:
                    dev_idx = 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            return f"{gpu_name} (PyTorch CUDA)"
    except Exception:
        pass
    return device_str.upper()


def _setup_mesh_and_spaces(n: int, device_str: str = "cpu"):
    """构建三维四面体网格与位移张量有限元空间."""
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TetrahedronMesh

    if device_str.lower() != "cpu":
        try:
            bm.set_backend("pytorch")
        except Exception:
            pass

    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 3))
    return mesh, scalar_space, tensor_space


def measure_cache(method: str, n: int, device_str: str = "cpu") -> dict:
    """阶段 1: 测量 EA 单元刚度张量显式缓存的常驻显存/内存与单价."""
    dev = device_str.lower()
    use_gpu = dev != "cpu"

    if use_gpu:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dev_obj = torch.device(device_str if ":" in device_str or "cuda" in device_str else "cuda:0")
    else:
        dev_obj = None

    t0 = time.perf_counter()
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.fem import LinearElasticIntegrator

    mesh, scalar_space, tensor_space = _setup_mesh_and_spaces(n, device_str)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3)
    integrator = LinearElasticIntegrator(material=material, q=3, method=method)

    if use_gpu:
        import torch
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(dev_obj)

    # const 构造预先算出单元矩阵 {K_e}
    const_integrator = integrator.const(tensor_space)
    K_e = const_integrator.assembly(tensor_space)

    if use_gpu:
        import torch
        if not isinstance(K_e, torch.Tensor):
            K_e = torch.as_tensor(K_e, device=dev_obj, dtype=torch.float64)
        else:
            K_e = K_e.to(dev_obj)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_bytes = int(torch.cuda.max_memory_allocated(dev_obj))
        net_cache_bytes = int(torch.cuda.memory_allocated(dev_obj) - mem_before)
    else:
        elapsed = time.perf_counter() - t0
        peak_bytes = get_peak_rss_bytes()
        net_cache_bytes = int(K_e.nbytes if hasattr(K_e, "nbytes") else sys.getsizeof(K_e))

    n_cells = int(mesh.number_of_cells())
    n_dofs = int(tensor_space.number_of_global_dofs())

    # 理论单价推导:
    # 单元刚度矩阵: NC * 12 * 12 * 8 Bytes
    # cell2dof 索引: NC * 12 * 8 Bytes
    element_tensor_bytes = n_cells * 144 * 8
    cell2dof_bytes = n_cells * 12 * 8
    theoretical_static_bytes = element_tensor_bytes + cell2dof_bytes

    unit_cost_bytes = peak_bytes / n_dofs
    unit_cost_cell = peak_bytes / n_cells
    capacity_ceiling = int(DEFAULT_MEMORY_TOTAL / unit_cost_bytes) if unit_cost_bytes > 0 else 0
    gpu_ceiling = int(DEFAULT_GPU_VRAM_TOTAL / unit_cost_bytes) if unit_cost_bytes > 0 else 0

    dev_display = detect_device_display(device_str)

    return {
        "case_id": "element-cache",
        "panel": "cache",
        "role": "element-cache-study",
        "problem": "DivergenceFreePolynomialElasticity3D",
        "mesh_type": "TetrahedronMesh",
        "grid": f"{n}^3",
        "n": n,
        "n_cells": n_cells,
        "n_dofs": n_dofs,
        "method": method,
        "device": dev_display,
        "device_raw": device_str,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mib": peak_bytes / (1024**2),
        "net_cache_bytes": net_cache_bytes,
        "net_cache_mib": net_cache_bytes / (1024**2),
        "theoretical_static_bytes": theoretical_static_bytes,
        "theoretical_static_mib": theoretical_static_bytes / (1024**2),
        "unit_cost_bytes_per_dof": unit_cost_bytes,
        "unit_cost_kb_per_dof": unit_cost_bytes / 1000,
        "unit_cost_bytes_per_cell": unit_cost_cell,
        "capacity_ceiling_47g_dofs": capacity_ceiling,
        "capacity_ceiling_gpu_16g_dofs": gpu_ceiling,
    }


class EAMatVecOperator:
    """EA 单元装配算子向量乘 (Gather-Apply-Scatter)."""

    def __init__(self, K_e: Any, cell2dof: Any, use_gpu: bool = False, device: Any = None):
        self.use_gpu = use_gpu
        self.device = device
        if use_gpu:
            import torch
            self.K_e = torch.as_tensor(K_e, device=device, dtype=torch.float64)
            self.cell2dof = torch.as_tensor(cell2dof, device=device, dtype=torch.int64)
            self.flat_cell2dof = self.cell2dof.reshape(-1)
        else:
            self.K_e = np.asarray(K_e, dtype=np.float64)
            self.cell2dof = np.asarray(cell2dof, dtype=np.int64)
            self.flat_cell2dof = self.cell2dof.reshape(-1)

    def __matmul__(self, x: Any) -> Any:
        if self.use_gpu:
            import torch
            x_e = x[self.cell2dof]
            y_e = torch.bmm(self.K_e, x_e.unsqueeze(-1)).squeeze(-1)
            y = torch.zeros_like(x)
            y.scatter_add_(0, self.flat_cell2dof, y_e.reshape(-1))
            return y
        else:
            x_e = x[self.cell2dof]
            y_e = np.einsum("cij,cj->ci", self.K_e, x_e)
            y = np.zeros_like(x)
            np.add.at(y, self.flat_cell2dof, y_e.ravel())
            return y


def measure_matvec(method: str, n: int, num_matvecs: int = 20, device_str: str = "cpu") -> dict:
    """阶段 2: 测量 EA 算子单次及批量 MatVec (A @ x) 的耗时、吞吐与瞬态显存."""
    dev = device_str.lower()
    use_gpu = dev != "cpu"

    if use_gpu:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dev_obj = torch.device(device_str if ":" in device_str or "cuda" in device_str else "cuda:0")
    else:
        dev_obj = None

    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.fem import LinearElasticIntegrator

    mesh, scalar_space, tensor_space = _setup_mesh_and_spaces(n, device_str)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3)
    integrator = LinearElasticIntegrator(material=material, q=3, method=method)

    const_integrator = integrator.const(tensor_space)
    K_e = const_integrator.assembly(tensor_space)
    cell2dof = tensor_space.cell_to_dof()

    op = EAMatVecOperator(K_e, cell2dof, use_gpu=use_gpu, device=dev_obj)

    n_cells = int(mesh.number_of_cells())
    n_dofs = int(tensor_space.number_of_global_dofs())

    # 构建随机测试向量
    if use_gpu:
        import torch
        x = torch.randn(n_dofs, dtype=torch.float64, device=dev_obj)
        # Warmup
        y = op @ x
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_matvecs):
            y = op @ x
        torch.cuda.synchronize()
        elapsed_total = time.perf_counter() - t0
        peak_bytes = int(torch.cuda.max_memory_allocated(dev_obj))
    else:
        x = np.random.randn(n_dofs)
        # Warmup
        y = op @ x
        t0 = time.perf_counter()
        for _ in range(num_matvecs):
            y = op @ x
        elapsed_total = time.perf_counter() - t0
        peak_bytes = get_peak_rss_bytes()

    avg_matvec_ms = (elapsed_total / num_matvecs) * 1000
    throughput_mdofs_per_sec = (n_dofs / (elapsed_total / num_matvecs)) / 1e6
    gflops_per_sec = (288 * n_cells / (elapsed_total / num_matvecs)) / 1e9

    unit_cost_bytes = peak_bytes / n_dofs
    dev_display = detect_device_display(device_str)

    return {
        "case_id": "ea-matvec",
        "panel": "matvec",
        "role": "matvec-benchmark",
        "problem": "DivergenceFreePolynomialElasticity3D",
        "mesh_type": "TetrahedronMesh",
        "grid": f"{n}^3",
        "n": n,
        "n_cells": n_cells,
        "n_dofs": n_dofs,
        "method": method,
        "device": dev_display,
        "device_raw": device_str,
        "num_matvecs": num_matvecs,
        "elapsed_total_seconds": elapsed_total,
        "avg_matvec_ms": avg_matvec_ms,
        "throughput_mdofs_per_sec": throughput_mdofs_per_sec,
        "gflops_per_sec": gflops_per_sec,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mib": peak_bytes / (1024**2),
        "unit_cost_bytes_per_dof": unit_cost_bytes,
        "unit_cost_kb_per_dof": unit_cost_bytes / 1000,
    }


def measure_solve(
    method: str,
    n: int,
    maxiter: int = 200,
    tol: float = 1e-6,
    device_str: str = "cpu",
) -> dict:
    """全流程: 测量 EA 算子搭载无预条件 CG 线性求解的端到端峰值内存与容量天花板."""
    dev = device_str.lower()
    use_gpu = dev != "cpu"

    if use_gpu:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dev_obj = torch.device(device_str if ":" in device_str or "cuda" in device_str else "cuda:0")
    else:
        dev_obj = None

    t0 = time.perf_counter()
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.fem import LinearElasticIntegrator

    mesh, scalar_space, tensor_space = _setup_mesh_and_spaces(n, device_str)
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3)
    integrator = LinearElasticIntegrator(material=material, q=3, method=method)

    const_integrator = integrator.const(tensor_space)
    K_e = const_integrator.assembly(tensor_space)
    cell2dof = tensor_space.cell_to_dof()

    op = EAMatVecOperator(K_e, cell2dof, use_gpu=use_gpu, device=dev_obj)

    n_cells = int(mesh.number_of_cells())
    n_dofs = int(tensor_space.number_of_global_dofs())

    # 提取边界条件与右端载荷
    is_bd_dof = tensor_space.is_boundary_dof()
    if use_gpu:
        import torch
        is_bd_mask = torch.as_tensor(is_bd_dof, device=dev_obj, dtype=torch.bool)
        is_int_mask = ~is_bd_mask

        b_vec = torch.zeros(n_dofs, dtype=torch.float64, device=dev_obj)
        b_vec[is_int_mask] = 1.0
        x = torch.zeros(n_dofs, dtype=torch.float64, device=dev_obj)

        def A_op(v):
            Av = op @ v
            return torch.where(is_int_mask, Av, v)

        r = b_vec - A_op(x)
        r[is_bd_mask] = 0.0
        p = r.clone()
        rsold = torch.dot(r, r)
        b_norm = torch.norm(b_vec)
        if b_norm == 0:
            b_norm = 1.0

        it_count = 0
        t_solve_start = time.perf_counter()
        for k in range(maxiter):
            it_count += 1
            Ap = A_op(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            r[is_bd_mask] = 0.0
            rsnew = torch.dot(r, r)
            rel_res = (torch.sqrt(rsnew) / b_norm).item()
            if rel_res < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        torch.cuda.synchronize()
        solve_elapsed = time.perf_counter() - t_solve_start
        total_elapsed = time.perf_counter() - t0
        peak_bytes = int(torch.cuda.max_memory_allocated(dev_obj))
    else:
        is_bd_mask = np.asarray(is_bd_dof, dtype=bool)
        is_int_mask = ~is_bd_mask

        b_vec = np.zeros(n_dofs, dtype=np.float64)
        b_vec[is_int_mask] = 1.0
        x = np.zeros(n_dofs, dtype=np.float64)

        def A_op(v):
            Av = op @ v
            return np.where(is_int_mask, Av, v)

        r = b_vec - A_op(x)
        r[is_bd_mask] = 0.0
        p = r.copy()
        rsold = np.dot(r, r)
        b_norm = np.linalg.norm(b_vec)
        if b_norm == 0:
            b_norm = 1.0

        it_count = 0
        t_solve_start = time.perf_counter()
        for k in range(maxiter):
            it_count += 1
            Ap = A_op(p)
            alpha = rsold / (np.dot(p, Ap) + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            r[is_bd_mask] = 0.0
            rsnew = np.dot(r, r)
            rel_res = float(np.sqrt(rsnew) / b_norm)
            if rel_res < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        solve_elapsed = time.perf_counter() - t_solve_start
        total_elapsed = time.perf_counter() - t0
        peak_bytes = get_peak_rss_bytes()

    unit_cost_bytes = peak_bytes / n_dofs
    capacity_ceiling = int(DEFAULT_MEMORY_TOTAL / unit_cost_bytes) if unit_cost_bytes > 0 else 0
    gpu_ceiling = int(DEFAULT_GPU_VRAM_TOTAL / unit_cost_bytes) if unit_cost_bytes > 0 else 0
    dev_display = detect_device_display(device_str)

    return {
        "case_id": "ea-cg-solve",
        "panel": "solve",
        "role": "ea-cg-solve-production",
        "problem": "DivergenceFreePolynomialElasticity3D",
        "mesh_type": "TetrahedronMesh",
        "grid": f"{n}^3",
        "n": n,
        "n_cells": n_cells,
        "n_dofs": n_dofs,
        "method": method,
        "device": dev_display,
        "device_raw": device_str,
        "cg_iterations": it_count,
        "relative_residual": rel_res,
        "solve_time_seconds": solve_elapsed,
        "total_time_seconds": total_elapsed,
        "time_per_iteration_ms": (solve_elapsed / it_count) * 1000 if it_count > 0 else 0,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mib": peak_bytes / (1024**2),
        "unit_cost_bytes_per_dof": unit_cost_bytes,
        "unit_cost_kb_per_dof": unit_cost_bytes / 1000,
        "capacity_ceiling_47g_dofs": capacity_ceiling,
        "capacity_ceiling_gpu_16g_dofs": gpu_ceiling,
    }


# -----------------------------------------------------------------------------
# 2. 控制台树状卡片看板 (Style B Dashboard)
# -----------------------------------------------------------------------------

def print_dashboard(out: Dict[str, Any]) -> None:
    """以统一 Style B 树状卡片格式在控制台打印 EA 算子性能测量报告."""
    case_id = out.get("case_id", "ea-metric")
    problem = out.get("problem", "Elasticity3D")
    mesh_type = out.get("mesh_type", "TetrahedronMesh")
    grid = out.get("grid", "-")
    n_cells = out.get("n_cells", 0)
    n_dofs = out.get("n_dofs", 0)
    method = out.get("method", "fast")
    device = out.get("device", "CPU")
    peak_bytes = out.get("peak_memory_bytes", 0)
    unit_cost_kb = out.get("unit_cost_kb_per_dof", 0.0)
    unit_cost_b = out.get("unit_cost_bytes_per_dof", 0.0)

    # 格式化内存
    if peak_bytes >= 1024**3:
        mem_str = f"{peak_bytes / (1024**3):.2f} GiB ({peak_bytes / (1024**2):,.1f} MiB)"
    else:
        mem_str = f"{peak_bytes / (1024**2):.1f} MiB"

    # 卡片头部
    print(f"\n● [{case_id}] {problem}")
    print(f"  ├── Mesh & DOFs   : {mesh_type} (grid = {grid}) | {n_cells:,} cells | {n_dofs:,} DOFs")

    panel = out.get("panel", "")
    if panel == "cache":
        elapsed = out.get("elapsed_seconds", 0.0)
        time_str = f"{elapsed:.2f} s" if elapsed >= 1.0 else f"{elapsed * 1000:.1f} ms"
        print(f"  ├── Cache Engine  : method = {method} | device = {device}")
        print(f"  ├── Time Elapsed  : {time_str}")
        print(f"  ├── Peak Memory   : {mem_str}")
        print(f"  └── Unit Cost     : {unit_cost_kb:.1f} KB/dof ({unit_cost_b:,.1f} B/dof)")

    elif panel == "matvec":
        avg_ms = out.get("avg_matvec_ms", 0.0)
        throughput = out.get("throughput_mdofs_per_sec", 0.0)
        gflops = out.get("gflops_per_sec", 0.0)
        print(f"  ├── MatVec Engine : method = {method} | device = {device}")
        print(f"  ├── MatVec Latency: {avg_ms:.2f} ms / call (Throughput: {throughput:.2f} MDOFs/s, {gflops:.2f} GFLOPs)")
        print(f"  ├── Peak Memory   : {mem_str}")
        print(f"  └── Unit Cost     : {unit_cost_kb:.1f} KB/dof ({unit_cost_b:,.1f} B/dof)")

    elif panel == "solve":
        it_count = out.get("cg_iterations", 0)
        solve_s = out.get("solve_time_seconds", 0.0)
        ms_per_it = out.get("time_per_iteration_ms", 0.0)
        rel_res = out.get("relative_residual", 0.0)
        ceiling_47g = out.get("capacity_ceiling_47g_dofs", 0)
        print(f"  ├── CG Solver     : {it_count} iters (rel_res = {rel_res:.2e}) | solve_time = {solve_s:.2f} s ({ms_per_it:.2f} ms/iter)")
        print(f"  ├── Peak Memory   : {mem_str}")
        print(f"  ├── Unit Cost     : {unit_cost_kb:.1f} KB/dof ({unit_cost_b:,.1f} B/dof)")
        print(f"  └── 47G Ceiling   : 约 {ceiling_47g / 1e4:,.1f} 万 DOFs")
    print()


# -----------------------------------------------------------------------------
# 3. 调度层与子进程控制
# -----------------------------------------------------------------------------

def command_list(cases: Tuple[config.Case, ...], figure: dict) -> int:
    """列出已注册的 EA 数据点."""
    headers = ["case-id", "mesh", "grid", "problem", "method", "device"]
    rows = []
    for c in cases:
        rows.append([c.id, c.mesh_type, c.grid, c.problem, c.method, c.device])

    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "  ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(f"\n{header_line}\n{sep_line}")
    for r in rows:
        print("  ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(r)))
    print()
    return 0


def command_run(
    selected: Tuple[config.Case, ...],
    is_all: bool = False,
    check_only: bool = False,
    skip_existing: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> int:
    """按独立子进程调度执行已选工况."""
    repo_root = Path(__file__).resolve().parents[2]
    total = len(selected)
    failed = 0

    print(f"\n==================== EA Capability 调度执行 ({total} 个任务) ====================")
    for idx, case in enumerate(selected, 1):
        artifact_path = case.artifact_path
        if overrides:
            n_val = overrides.get("n", case.n)
            method_val = overrides.get("method", case.method)
            dev_str = overrides.get("device", case.device)
            dev_tag = f"_{dev_str}" if dev_str != "cpu" else ""
            artifact_path = _OUTPUT_DIR / f"{case.panel}_{method_val}_n{n_val}{dev_tag}.json"

        if skip_existing and artifact_path.is_file():
            print(f"[{idx}/{total}] 跳过已存在产物: {artifact_path.name}")
            continue

        cmd = case.to_command(repo_root, overrides)
        cmd.extend(["--output", str(artifact_path)])

        if check_only:
            print(f"[{idx}/{total}] [dry-run] {' '.join(cmd)}")
            continue

        print(f"[{idx}/{total}] 调度子进程: {case.id} (输出 -> {artifact_path.name})")
        t_start = time.perf_counter()
        completed = subprocess.run(cmd, env=None)
        elapsed = time.perf_counter() - t_start

        if completed.returncode != 0:
            failed += 1
            print(f"  失败: 退出码 {completed.returncode}, 用时 {elapsed:.1f} s")
        elif not artifact_path.is_file():
            failed += 1
            print(f"  失败: 进程正常退出但产物未生成 -> {artifact_path}")

    if failed:
        print(f"\n{failed} 个任务执行失败。")
    return failed


# -----------------------------------------------------------------------------
# 4. 主入口与参数路由
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="ea_assembly_capability 实验驱动: 测量 EA 单元装配无矩阵算子的内存机制与容量极限",
    )

    # 1. 调度与工况选择参数
    parser.add_argument("--list", action="store_true", help="列出已注册数据点")
    parser.add_argument("--all", action="store_true", help="跑全部工况")
    parser.add_argument("--cases", nargs="+", help="指定要跑的一个或多个 case id")
    parser.add_argument("--case", help="指定单个 case id")
    parser.add_argument("--panel", choices=config.PANELS, help="只跑指定阶段 (cache/matvec/solve) 数据点")
    parser.add_argument("--check-only", action="store_true", help="只打印将执行的子进程命令")
    parser.add_argument("--skip-existing", action="store_true", help="产物已存在时跳过")

    # 2. 工况动态覆盖参数
    parser.add_argument("-n", "--n", "--grid", dest="n", type=int, default=None, help="动态覆盖网格剖分段数")
    parser.add_argument("--device", type=str, default="cpu", help="指定计算设备 ('cpu' 或 'cuda' 等)")
    parser.add_argument("--method", choices=config.METHODS, default=None, help="指定或覆盖单刚算法 (fast/standard/voigt)")

    # 3. Worker 测量层底层参数
    parser.add_argument("--worker", action="store_true", help="进入子进程 worker 测量模式")
    parser.add_argument("--cache", action="store_true", help="阶段 1 单元刚度张量显式缓存测量")
    parser.add_argument("--matvec", action="store_true", help="阶段 2 算子乘积 MatVec 测量")
    parser.add_argument("--solve", action="store_true", help="全流程端到端 CG 线性求解测量")
    parser.add_argument("--output", type=Path, default=None, help="产物落盘路径")

    args = parser.parse_args(argv)

    # ------------------------------------------------ Worker 测量分支
    if args.worker or args.cache or args.matvec or args.solve:
        if args.n is None:
            parser.error("Worker 模式必须指定 --n")
        method = args.method or "fast"

        if args.cache:
            out = measure_cache(method, args.n, device_str=args.device)
        elif args.matvec:
            out = measure_matvec(method, args.n, device_str=args.device)
        elif args.solve:
            out = measure_solve(method, args.n, device_str=args.device)
        else:
            parser.error("Worker 模式需指定 --cache, --matvec 或 --solve")

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
    target_case_ids: List[str] = []
    panel_filter: Optional[str] = args.panel
    known_ids = {c.id for c in cases}
    is_all = args.all

    alias_map = {
        "cache": "element-cache",
        "element-cache": "element-cache",
        "matvec": "ea-matvec",
        "ea-matvec": "ea-matvec",
        "solve": "ea-cg-solve",
        "ea-solve": "ea-cg-solve",
        "ea-cg-solve": "ea-cg-solve",
    }

    raw_cases: List[str] = []
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
        elif item_lower in alias_map:
            target_case_ids.append(alias_map[item_lower])
        elif item in known_ids:
            target_case_ids.append(item)
        else:
            target_case_ids.append(item)

    if not (is_all or target_case_ids or panel_filter):
        print(
            "错误: 必须通过 --case / --cases / --all 指定要运行的工况。\n"
            "  常用示例:\n"
            "    python run.py --case element-cache\n"
            "    python run.py --case ea-matvec\n"
            "    python run.py --case ea-cg-solve\n"
            "    python run.py --all\n"
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

    overrides: Dict[str, Any] = {}
    if args.n is not None:
        overrides["n"] = args.n
    if args.device != "cpu":
        overrides["device"] = args.device
    if args.method is not None:
        overrides["method"] = args.method

    failed = command_run(
        selected,
        is_all=is_all,
        check_only=args.check_only,
        skip_existing=args.skip_existing,
        overrides=overrides if overrides else None,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
