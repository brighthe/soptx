# -*- coding: utf-8 -*-
"""PIML 批量张量缩聚 GPU 并发硬件加速评测基准.

本脚本测试在完全相同的 PIML 代理模型与变分计算链路下,
不同子结构并发规模 N_subs in [24, 48, 96, 192, 384] 的 CPU 与 GPU 单步耗时对比:
1. PIML CPU 批量张量缩聚: PyTorch CPU 多尺度形函数推断 + CPU Batched GEMM 变分刚度重构;
2. PIML GPU 批量张量缩聚: PyTorch CUDA 多尺度形函数推断 + GPU Batched GEMM 变分刚度重构;
3. 计算纯硬件加速比, 并落盘标准证据 JSON 文件.

使用方法:
    python experiments/piml_substructure_gpu/benchmark_gpu_speedup.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import SubstructurePrototype
from soptx.ml import MLP


def create_piml_surrogate(n_in: int, n_out: int, device: torch.device) -> nn.Module:
    """构建轻量级 PIML 多尺度形函数代理网络 (复用 soptx.ml.MLP 基础模块)."""
    net = MLP(
        input_dim=n_in,
        output_dim=n_out,
        hidden_dims=(128, 128),
        activation=nn.SiLU,
        device=device,
    )
    net.eval()
    return net


def benchmark_scale(
    n_subs: int,
    proto: SubstructurePrototype,
    dim: int,
    n_fine: Tuple[int, ...],
    n_repeat: int = 25,
) -> Dict[str, float]:
    """评测特定子结构并发规模下的 PIML CPU 与 PIML GPU 单步缩聚耗时."""
    n_i = int(proto.n_i)
    n_b = int(proto.n_b)
    R_rigid = bm.to_numpy(proto.rigid_basis)
    R_perp = bm.to_numpy(proto.deformation_basis)
    Phi_i = bm.to_numpy(proto.rigid_interior_modes)
    n_reduced = int(R_perp.shape[1])
    n_in = int(np.prod(n_fine))

    # 构造基础刚度矩阵分块
    rho_sample = bm.from_numpy(np.ones((1, *n_fine)))
    K_batch = proto.assemble_local_stiffness_batch(rho_sample)
    K_local = bm.to_numpy(K_batch)[0]
    i_dofs = bm.to_numpy(proto.i_dofs)
    b_dofs = bm.to_numpy(proto.b_dofs)
    K_ii_np = K_local[np.ix_(i_dofs, i_dofs)].astype(np.float32)
    K_ib_np = K_local[np.ix_(i_dofs, b_dofs)].astype(np.float32)
    K_bb_np = K_local[np.ix_(b_dofs, b_dofs)].astype(np.float32)

    # 1. PIML CPU 批量张量缩聚
    net_cpu = create_piml_surrogate(n_in, n_i * n_reduced, torch.device("cpu"))
    dens_cpu = torch.rand((n_subs, n_in), device="cpu", dtype=torch.float32)
    Kii_cpu = torch.tensor(K_ii_np, device="cpu").unsqueeze(0).repeat(n_subs, 1, 1)
    Kib_cpu = torch.tensor(K_ib_np, device="cpu").unsqueeze(0).repeat(n_subs, 1, 1)
    Kbb_cpu = torch.tensor(K_bb_np, device="cpu").unsqueeze(0).repeat(n_subs, 1, 1)
    Rrig_cpu = torch.tensor(R_rigid, device="cpu", dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)
    Phi_cpu = torch.tensor(Phi_i, device="cpu", dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)
    Rperp_cpu = torch.tensor(R_perp, device="cpu", dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)

    def piml_cpu_step():
        M_pred = net_cpu(dens_cpu).view(n_subs, n_i, n_reduced)
        N_pred = torch.bmm(Phi_cpu, Rrig_cpu.transpose(1, 2)) + torch.bmm(M_pred, Rperp_cpu.transpose(1, 2))
        KbiN = torch.bmm(Kib_cpu.transpose(1, 2), N_pred)
        N_T_Kii_N = torch.bmm(N_pred.transpose(1, 2), torch.bmm(Kii_cpu, N_pred))
        return Kbb_cpu + KbiN + KbiN.transpose(1, 2) + N_T_Kii_N

    for _ in range(5):
        _ = piml_cpu_step()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = piml_cpu_step()
    t_cpu_ms = (time.perf_counter() - t0) / n_repeat * 1000.0

    # 2. PIML GPU 批量张量缩聚
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_gpu = create_piml_surrogate(n_in, n_i * n_reduced, device_gpu)
    dens_gpu = torch.rand((n_subs, n_in), device=device_gpu, dtype=torch.float32)
    Kii_gpu = torch.tensor(K_ii_np, device=device_gpu).unsqueeze(0).repeat(n_subs, 1, 1)
    Kib_gpu = torch.tensor(K_ib_np, device=device_gpu).unsqueeze(0).repeat(n_subs, 1, 1)
    Kbb_gpu = torch.tensor(K_bb_np, device=device_gpu).unsqueeze(0).repeat(n_subs, 1, 1)
    Rrig_gpu = torch.tensor(R_rigid, device=device_gpu, dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)
    Phi_gpu = torch.tensor(Phi_i, device=device_gpu, dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)
    Rperp_gpu = torch.tensor(R_perp, device=device_gpu, dtype=torch.float32).unsqueeze(0).repeat(n_subs, 1, 1)

    def piml_gpu_step():
        M_pred = net_gpu(dens_gpu).view(n_subs, n_i, n_reduced)
        N_pred = torch.bmm(Phi_gpu, Rrig_gpu.transpose(1, 2)) + torch.bmm(M_pred, Rperp_gpu.transpose(1, 2))
        KbiN = torch.bmm(Kib_gpu.transpose(1, 2), N_pred)
        N_T_Kii_N = torch.bmm(N_pred.transpose(1, 2), torch.bmm(Kii_gpu, N_pred))
        return Kbb_gpu + KbiN + KbiN.transpose(1, 2) + N_T_Kii_N

    for _ in range(10):
        _ = piml_gpu_step()
    if device_gpu.type == "cuda":
        torch.cuda.synchronize(device_gpu)

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = piml_gpu_step()
    if device_gpu.type == "cuda":
        torch.cuda.synchronize(device_gpu)
    t_gpu_ms = (time.perf_counter() - t0) / n_repeat * 1000.0

    return {
        "n_subs": n_subs,
        "t_cpu_ms": float(t_cpu_ms),
        "t_gpu_ms": float(t_gpu_ms),
        "speedup": float(t_cpu_ms / max(t_gpu_ms, 1e-6)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PIML 批量张量缩聚 GPU 并发硬件加速评测")
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3], help="子结构空间维度 (缺省 3D)")
    parser.add_argument("--scales", type=int, nargs="+", default=[24, 48, 96, 192, 384],
                        help="子结构并发规模列表")
    parser.add_argument("--repeats", type=int, default=30, help="测试重复轮数")
    parser.add_argument("--output-dir", type=str, default=None, help="证据保存目录")
    args = parser.parse_args()

    bm.set_backend("numpy")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"

    if args.dim == 2:
        sub_size = (1.0, 1.0)
        n_fine = (5, 5)
        elem_name = "5x5 Q1 单元"
    else:
        sub_size = (1.0, 1.0, 1.0)
        n_fine = (4, 4, 4)
        elem_name = "4x4x4 8 节点六面体单元 (Hexahedron)"

    proto = SubstructurePrototype(sub_size, n_fine, E_base=1.0, nu=0.3)

    print()
    print("【图 3(d) 算力基准】PIML 批量缩聚 GPU 硬件加速评测 (PIML CPU vs PIML GPU)")
    print("=" * 88)
    print(f"测试硬件环境 : {gpu_name} (PyTorch {torch.__version__})")
    print(f"子结构网格   : {elem_name} (内部自由度 n_i={proto.n_i}, 接口自由度 n_b={proto.n_b})")
    print(f"测试并发规模 : {args.scales}")
    print("-" * 88)
    print(f"{'子结构并发数 N_subs':<20} | {'PIML CPU 批量张量缩聚':<22} | {'PIML GPU 批量张量缩聚':<22} | {'单卡 GPU 硬件加速比':<15}")
    print("-" * 88)

    results: List[Dict[str, float]] = []
    for n in args.scales:
        r = benchmark_scale(n, proto, args.dim, n_fine, n_repeat=args.repeats)
        results.append(r)
        t_cpu_str = f"{r['t_cpu_ms']:.2f} ms"
        t_gpu_str = f"{r['t_gpu_ms']:.2f} ms"
        print(f"{r['n_subs']:<20d} | {t_cpu_str:<22} | {t_gpu_str:<22} | {r['speedup']:.1f}x ({r['speedup']:.0f} 倍)")

    print("=" * 88)

    n_subs_list = [r["n_subs"] for r in results]
    t_cpu_list = [r["t_cpu_ms"] for r in results]
    t_gpu_list = [r["t_gpu_ms"] for r in results]
    speedup_list = [r["speedup"] for r in results]

    payload = {
        "device": gpu_name,
        "dimension": f"{args.dim}D",
        "benchmark_type": "PIML CPU vs PIML GPU",
        "n_subs": n_subs_list,
        "t_cpu_ms": t_cpu_list,
        "t_gpu_ms": t_gpu_list,
        "speedup": speedup_list,
        "summary": {
            "min_speedup": float(min(speedup_list)),
            "max_speedup": float(max(speedup_list)),
            "mean_speedup": float(np.mean(speedup_list)),
        }
    }

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "piml_gpu_speedup.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[证据] 评测数据已落盘: {out_path}\n")


if __name__ == "__main__":
    main()
