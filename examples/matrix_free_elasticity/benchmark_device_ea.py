"""同一 EA matrix-free 算子在 CPU 与单块 GPU 上的单次求解耗时对照.

本脚本固定制造解、材料、边界条件、P1 空间、``operator_level = "ea"`` 和 CG 停机
准则, **唯一的变量是设备**: 两侧都走 fealpy 的 ``pytorch`` 后端, 只切
``bm.set_default_device("cpu" / "cuda")``. 换后端(numpy 对 pytorch-cuda)会把后端
差异混进加速比里, 那样测出来的不是"这个算子能不能上加速器".

**不是多卡强扩展.** 一块卡就能回答"matrix-free 算子可以搬上加速器"; 多卡扩展是
另一个题目, 混进来会让读者以为多卡结果已经有了.

**验收门禁: 两侧解的相对差不大于 ``--gap-gate``(缺省 1e-9).** 加速比没有意义,
除非两边算的是同一个问题. 脚本另外核对两侧的 CG 迭代数是否逐档相同 —— 迭代数
一致比"最终解接近"更强: 它说明整条 Krylov 轨迹一致, 而不是碰巧收敛到附近.

同一进程内先后跑两个设备: 解要留在手里做相对差, 跨进程比对得把解落盘再读回,
多一层可能出错的环节. 峰值内存不在本脚本的口径内(那是 ``--mode serial-peak-rss``
的事), 因此不需要像 (b) 那样靠进程隔离来归属.

使用方法::

    python examples/matrix_free_elasticity/benchmark_device_ea.py --n 8
    python examples/matrix_free_elasticity/benchmark_device_ea.py --n 64 --output outputs/device_speedup_n64.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm

from examples.matrix_free_elasticity.benchmark_cpu_ea import (
    ATOL,
    DEGREE,
    RTOL,
    build_context,
    build_operator,
)

DEVICES = ("cpu", "cuda")
DEFAULT_GAP_GATE = 1.0e-9


def sync(device: str) -> None:
    """CUDA kernel 是异步派发的, 计时边界前后必须同步, 否则量到的是派发时间."""
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def release(device: str) -> None:
    """归还上一轮的显存. 不放的话四档连跑时缓存的 K_e 会叠着涨."""
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def solve_once(device: str, model: str, mesh_type: str, resolution: int,
               assembly_method: str, maxiter: int) -> dict[str, Any]:
    """在指定设备上完整跑一遍: 建网格与算子, 施加边界条件, CG 求解.

    每次调用都从网格重建. 复用上一轮的算子会让第二次之后的计时落在已经预热好的
    缓存上, 那不是"跑一个新问题"要花的时间.

    返回:
        dict: 含 ``t_build`` / ``t_solve`` / ``niter`` / 解向量等的一次样本.
    """
    bm.set_backend("pytorch")
    bm.set_default_device(device)

    dimension, problem, vector_space, material, mesh = build_context(
        model, mesh_type, resolution
    )

    sync(device)
    start = time.perf_counter()
    analyzer, raw_operator = build_operator(
        vector_space, problem, material, "ea", assembly_method
    )
    sync(device)
    t_build = time.perf_counter() - start

    load = analyzer.assemble_body_force_vector()
    system_operator, system_load = analyzer.apply_bc(raw_operator, load)

    ndof = int(vector_space.number_of_global_dofs())
    solution = bm.zeros((ndof,), dtype=bm.float64, device=bm.get_device(mesh))
    sync(device)
    start = time.perf_counter()
    _, solver_info = analyzer.solve_system(
        system_operator, system_load, solution,
        solver="cg", rtol=RTOL, atol=ATOL, maxiter=maxiter,
    )
    sync(device)
    t_solve = time.perf_counter() - start

    residual = system_operator @ solution - system_load
    relative_residual = float(bm.linalg.norm(residual)) / max(
        float(bm.linalg.norm(system_load)), 1.0e-30
    )
    return {
        "t_build": t_build,
        "t_solve": t_solve,
        "dofs": ndof,
        "cells": int(mesh.number_of_cells()),
        "dimension": dimension,
        "niter": int(solver_info["niter"]),
        "converged": bool(solver_info["converged"]),
        "relative_residual": relative_residual,
        "solution": bm.to_numpy(solution),
    }


def measure_device(device: str, arguments: argparse.Namespace):
    """按 warmup + repeats 测一个设备, 逐项取中位数.

    取中位数而不是最小值: 最小值是"最顺的一次", 在共享机器上不可复现; 中位数对
    偶发的系统抖动稳健, 又不像均值那样被单次长尾拖走.
    """
    for _ in range(arguments.warmup):
        solve_once(device, arguments.model, arguments.mesh_type, arguments.n,
                   arguments.assembly_method, arguments.maxiter)
        release(device)

    samples = []
    for _ in range(arguments.repeats):
        samples.append(
            solve_once(device, arguments.model, arguments.mesh_type, arguments.n,
                       arguments.assembly_method, arguments.maxiter)
        )
        release(device)

    last = samples[-1]
    record = {
        "build_seconds": median(s["t_build"] for s in samples),
        "solve_seconds": median(s["t_solve"] for s in samples),
        "build_samples": [s["t_build"] for s in samples],
        "solve_samples": [s["t_solve"] for s in samples],
        "cg_iterations": last["niter"],
        "cg_converged": all(s["converged"] for s in samples),
        "true_relative_residual": last["relative_residual"],
    }
    if device == "cpu":
        import torch

        # 加速比是"一块卡对几个 CPU 核"的比值。脚本不限制线程数, 走 torch 默认值,
        # 但那个默认值必须落到产物里 —— 否则事后无从判断分母是单核还是满核, 而
        # 这两种读法会把同一个 16 倍解释成完全不同的结论。
        record["torch_threads"] = int(torch.get_num_threads())
        record["torch_interop_threads"] = int(torch.get_num_interop_threads())
    if device == "cuda":
        import torch

        record["gpu_name"] = torch.cuda.get_device_name(0)
        record["gpu_memory_total_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
        record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return record, last


def main() -> int:
    arguments = parse_arguments()

    records: dict[str, Any] = {}
    solutions: dict[str, Any] = {}
    last: dict[str, Any] = {}
    for device in DEVICES:
        records[device], last = measure_device(device, arguments)
        solutions[device] = last["solution"]
        dofs = last["dofs"]
        cells = last["cells"]
        build = records[device]["build_seconds"]
        solve = records[device]["solve_seconds"]
        print(f"[{device}] dofs={dofs} cells={cells} build={build:.3f}s "
              f"solve={solve:.3f}s niter={last['niter']} "
              f"rel_res={last['relative_residual']:.2e}")

    import numpy as np

    reference = solutions["cpu"]
    gap = float(
        np.linalg.norm(reference - solutions["cuda"])
        / max(np.linalg.norm(reference), 1.0e-300)
    )
    iterations_match = (
        records["cpu"]["cg_iterations"] == records["cuda"]["cg_iterations"]
    )
    gate_passed = bool(
        gap <= arguments.gap_gate
        and iterations_match
        and records["cpu"]["cg_converged"]
        and records["cuda"]["cg_converged"]
    )

    speedup_solve = records["cpu"]["solve_seconds"] / records["cuda"]["solve_seconds"]
    speedup_build = records["cpu"]["build_seconds"] / records["cuda"]["build_seconds"]

    import torch

    payload = {
        "mode": "device-speedup-ea",
        "model": arguments.model,
        "mesh_type": arguments.mesh_type,
        "backend": "pytorch",
        "degree": DEGREE,
        "dimension": int(last["dimension"]),
        "resolution": arguments.n,
        "cells": int(last["cells"]),
        "dofs": int(last["dofs"]),
        "operator_level": "ea",
        "assembly_method": arguments.assembly_method,
        "cg": {"rtol": RTOL, "atol": ATOL, "maxiter": arguments.maxiter},
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "devices": records,
        "speedup_solve": speedup_solve,
        "speedup_build": speedup_build,
        "solution_relative_gap": gap,
        "solution_gap_gate": arguments.gap_gate,
        "cg_iterations_match": iterations_match,
        "gate_passed": gate_passed,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "timing_scope": (
            "solve_seconds 只含 CG 求解(含边界条件包装后的算子作用), 不含建网格、"
            "算子构造与体力向量组装; build_seconds 单列算子构造。两者均为 warmup 后 "
            "repeats 次的中位数, CUDA 侧每个计时边界前后都做 torch.cuda.synchronize"
        ),
    }

    bar = "=" * 72
    print(bar)
    print(f" CPU 对单卡 GPU [EA, {arguments.assembly_method}, "
          f"{arguments.mesh_type} n={arguments.n}, {payload['dofs']} 自由度]")
    print(bar)
    print(f" 求解加速比      : {speedup_solve:.2f}x")
    print(f" 算子构造加速比  : {speedup_build:.2f}x")
    print(f" CG 迭代数       : cpu {records['cpu']['cg_iterations']} / "
          f"cuda {records['cuda']['cg_iterations']} "
          f"({'一致' if iterations_match else '不一致'})")
    print(f" 两侧解相对差    : {gap:.3e}  (门禁 <= {arguments.gap_gate:.0e})")
    print(f" 门禁            : {'PASS' if gate_passed else 'FAIL'}")

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f" 产物            : {arguments.output}")

    return 0 if gate_passed else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="polynomial",
                        help="制造解模型; 3D 用 polynomial")
    parser.add_argument("--mesh-type", default="tet", help="网格类型")
    parser.add_argument("--n", type=int, default=8, help="每个坐标轴的剖分数")
    parser.add_argument("--assembly-method", default="fast",
                        help="单元矩阵收缩顺序; 与 (b) 保持一致用 fast")
    parser.add_argument("--warmup", type=int, default=1, help="不计时的预热次数")
    parser.add_argument("--repeats", type=int, default=3, help="取中位数的计时次数")
    parser.add_argument("--maxiter", type=int, default=5000, help="CG 最大迭代数")
    parser.add_argument("--gap-gate", type=float, default=DEFAULT_GAP_GATE,
                        help="两侧解相对差的验收上限")
    parser.add_argument("--output", type=Path, help="JSON 产物路径")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
