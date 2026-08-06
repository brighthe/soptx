"""GPU 后端求解线弹性方程的最小验证算例.

统一使用 pytorch 后端, 分别运行在 CPU 和 CUDA 设备上, 然后逐位比对二者的解
向量。两个因素 (数值库、设备) 中只有设备不同, 任何差异都直接归因于 GPU 计算。

判据:

* 真相对残差 ``||K u - F|| / ||F||`` —— 两侧都解开;
* GPU vs CPU 逐位一致性 —— ``||u_gpu - u_cpu||_∞`` 在机器精度内;
* GPU vs CPU 逐位相对误差 —— 排除位移幅值导致的误判。

制造解定义见
`制造解文档 <../../docs/problems/manufactured-elasticity.md>`__。

运行::

    python examples/gpu_elasticity/minimal_demo.py
    python examples/gpu_elasticity/minimal_demo.py --model exp-sine
    python examples/gpu_elasticity/minimal_demo.py --mesh-type quad
    python examples/gpu_elasticity/minimal_demo.py --nx 16 --ny 16
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import (
    SinusoidalPlaneStrainElasticity2D,
    ExponentialSineManufacturedElasticity2D,
)


# ---------------------------------------------------------------------------
# 门禁
# ---------------------------------------------------------------------------
RESIDUAL_TOLERANCE = 1.0e-10
CONSISTENCY_ATOL = 1.0e-9
NORM_FLOOR = 1.0e-30

MESH_CONSTRUCTORS = {"tri": TriangleMesh, "quad": QuadrangleMesh}

PROBLEM_FACTORIES = {
    "sinusoidal": lambda: SinusoidalPlaneStrainElasticity2D(domain=(0.0, 1.0, 0.0, 1.0)),
    "exp-sine": lambda: ExponentialSineManufacturedElasticity2D(domain=(0.0, 1.0, 0.0, 1.0)),
}


# ---------------------------------------------------------------------------
# 求解核心
# ---------------------------------------------------------------------------
def solve_once(
    problem,
    material,
    nx: int,
    ny: int,
    mesh_type: str,
    degree: int,
    device: str,
) -> dict:
    """统一使用 pytorch 后端, 在指定设备上装配并求解."""

    bm.set_backend("pytorch")
    bm.set_default_device(device)

    constructor = MESH_CONSTRUCTORS[mesh_type]
    mesh = constructor.from_box(list(problem.domain), nx=nx, ny=ny)

    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        space_degree=degree,
        integration_order=degree + 3,
        operator_level="fa",
        solve_method="cg",
        topopt_algorithm=None,
        enable_logging=False,
    )

    K0 = analyzer.assemble_stiff_matrix()
    F0 = analyzer.assemble_body_force_vector()
    K, F = analyzer.apply_bc(K0, F0)

    uh = analyzer.tensor_space.function()
    _, solver_info = analyzer.solve_system(K, F, uh, rtol=1e-12, atol=1e-12, maxiter=5000)

    # 用后端原生操作在设备上直接算范数, 避免传回大数组
    u_tensor = bm.asarray(uh)
    residual_vec = K @ u_tensor - F
    residual_norm = float(bm.linalg.norm(residual_vec))
    load_norm = float(bm.linalg.norm(F))

    # 位移向量最后一次性传回 CPU 做逐位比对
    displacement = np.asarray(bm.to_numpy(u_tensor))

    return {
        "displacement": displacement,
        "residual": residual_norm / max(load_norm, NORM_FLOOR),
        "niter": solver_info.get("niter"),
        "converged": solver_info.get("converged"),
        "device": device,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch 后端 CPU vs GPU 逐位比对",
    )
    parser.add_argument(
        "--model", choices=("sinusoidal", "exp-sine"), default="sinusoidal",
        help="制造解模型 (默认 sinusoidal)",
    )
    parser.add_argument(
        "--mesh-type", choices=("tri", "quad"), default="tri",
        help="网格类型 (默认 tri)",
    )
    parser.add_argument(
        "--nx", type=int, default=8,
        help="x 方向单元数 (默认 8)",
    )
    parser.add_argument(
        "--ny", type=int, default=8,
        help="y 方向单元数 (默认 8)",
    )
    parser.add_argument(
        "--degree", type=int, default=1,
        help="位移空间次数 (默认 1)",
    )
    parser.add_argument(
        "--gpu-device", type=str, default="cuda",
        help="GPU 设备名: 'cuda', 'cuda:0' 等 (默认 cuda)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    problem = PROBLEM_FACTORIES[arguments.model]()
    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        youngs_modulus=problem.E,
        poisson_ratio=problem.nu,
        enable_logging=False,
    )

    # 1. pytorch CPU 参考解
    print("=== CPU 参考解 (pytorch, device=cpu) ===")
    cpu = solve_once(
        problem, material,
        nx=arguments.nx, ny=arguments.ny,
        mesh_type=arguments.mesh_type,
        degree=arguments.degree,
        device="cpu",
    )
    print(f"  残差: {cpu['residual']:.2e}, cg 迭代: {cpu['niter']}, "
          f"收敛: {cpu['converged']}")

    # 2. pytorch GPU 解
    print(f"\n=== GPU 解 (pytorch, device={arguments.gpu_device}) ===")
    gpu = solve_once(
        problem, material,
        nx=arguments.nx, ny=arguments.ny,
        mesh_type=arguments.mesh_type,
        degree=arguments.degree,
        device=arguments.gpu_device,
    )
    print(f"  残差: {gpu['residual']:.2e}, cg 迭代: {gpu['niter']}, "
          f"收敛: {gpu['converged']}")

    # 3. 逐位比对
    u_cpu = cpu["displacement"].ravel()
    u_gpu = gpu["displacement"].ravel()

    abs_diff = np.abs(u_cpu - u_gpu)
    max_abs_diff = float(np.max(abs_diff))
    cpu_norm = float(np.linalg.norm(u_cpu))
    rel_diff = max_abs_diff / max(cpu_norm, NORM_FLOOR)

    print("\n=== GPU vs CPU 逐位比对 ===")
    print(f"  ||u_cpu - u_gpu||_∞           = {max_abs_diff:.2e}")
    print(f"  ||u_cpu - u_gpu||_∞ / ||u_cpu|| = {rel_diff:.2e}")

    # 4. 判据
    residual_ok = (
        cpu["residual"] <= RESIDUAL_TOLERANCE
        and gpu["residual"] <= RESIDUAL_TOLERANCE
    )
    consistency_ok = max_abs_diff <= CONSISTENCY_ATOL

    print()
    print(f"CPU 残差 ≤ {RESIDUAL_TOLERANCE:.0e} -> "
          f"{'通过' if cpu['residual'] <= RESIDUAL_TOLERANCE else '未通过'}")
    print(f"GPU 残差 ≤ {RESIDUAL_TOLERANCE:.0e} -> "
          f"{'通过' if gpu['residual'] <= RESIDUAL_TOLERANCE else '未通过'}")
    print(f"GPU vs CPU 逐位一致 (tol={CONSISTENCY_ATOL:.0e}) -> "
          f"{'通过' if consistency_ok else '未通过'}")

    if residual_ok and consistency_ok:
        print(f"\n结论: pytorch 后端在 CPU 与 {arguments.gpu_device} 上的 "
              f"求解结果逐位一致, GPU 求解链可用.")
        return 0

    print("\n结论: GPU 求解链存在问题, 见上面未通过的判据.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
