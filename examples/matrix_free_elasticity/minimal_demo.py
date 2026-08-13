"""Matrix-Free 线弹性极简 Demo 驱动脚本 (minimal_demo.py)。

一键运行 2D/3D 矩阵无关 (EA) 线弹性方程 PCG 求解，无需繁琐的 CLI 参数。
制造解方程与参数契约见 `docs/problems/manufactured-elasticity.md`。

使用方法：
    python minimal_demo.py                   # 运行 2D 平面应变线弹性 Demo
    python minimal_demo.py --dim 3           # 运行 3D 多项式无散场线弹性 Demo
    mpiexec -n 2 python minimal_demo.py      # 运行 2D MPI 分布式线弹性 Demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保示例目录在 sys.path 中
example_dir = Path(__file__).resolve().parent
if str(example_dir) not in sys.path:
    sys.path.insert(0, str(example_dir))

from mpi4py import MPI
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.distributed import distribute_mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

import utils.contract as contract
from cases import create_case
from soptx.fem.solvers import PreparedLinearSystem, solve_matrix_free_system
from utils.analyzer import ElasticityEAOperator
from soptx.fem.distributed import distribute_vector_space, partition_cells
from utils.postprocess import solution_error


def run_minimal_demo(dimension: int = 2, resolution_n: int = 8) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    case = create_case(dimension)
    resolution = (resolution_n,) * dimension

    # 1. 网格与空间构造（主进程 Root Rank 建立全局网格）
    if rank == 0:
        global_mesh = case.create_mesh(resolution)
        scalar_space = LagrangeFESpace(global_mesh, p=1, ctype="C")
        vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
        cell_masks = partition_cells(
            global_mesh,
            size,
            split_coordinate=case.partition_split_coordinate(),
        )
    else:
        global_mesh = None
        scalar_space = None
        vector_space = None
        cell_masks = None

    # 2. 将网格单元与向量自由度空间分发至各 MPI Rank
    dist_mesh = distribute_mesh(global_mesh, cell_masks, comm=comm)
    dist_space = distribute_vector_space(
        scalar_space,
        vector_space,
        dist_mesh,
        cell_masks,
        components=dimension,
        root=0,
        comm=comm,
    )

    # 3. 构造 Matrix-Free EA 刚度算子与边界条件线性系统
    ea_operator = ElasticityEAOperator(
        space=dist_space.space,
        case=case,
        degree=1,
        dof_comm=dist_space.dof_comm,
    )
    operator, load = ea_operator.assemble()

    system = PreparedLinearSystem(
        operator=operator,
        load=load,
        prescribed=ea_operator.prescribed_solution,
        boundary_dofs=ea_operator.boundary_dofs,
    )

    # 4. 使用分布式重叠加权 Matrix-Free PCG 求解器
    local_sol, diagnostics = solve_matrix_free_system(
        system,
        dist_space.dof_comm,
        maxiter=1000,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    # 5. 跨 Rank 汇总并归约全局解向量至 Root 进程
    refs = dist_space.dof_comm.refs(local_sol.shape[0])
    global_sol = dist_space.dof_comm.gather_add(local_sol / refs, root=0)

    # 6. 在 Root 进程计算精确 L2 位移误差并输出求解诊断
    if rank == 0:
        sol_func = vector_space.function(dtype=bm.float64)
        sol_func[:] = global_sol
        l2_abs, l2_rel = solution_error(global_mesh, sol_func, case.problem, 1)

        print("\n" + "=" * 60)
        print(f" Matrix-Free Elasticity Demo [{dimension}D - {case.name}]")
        print("=" * 60)
        print(f" MPI Ranks          : {size}")
        print(f" Grid Resolution    : {'x'.join(str(r) for r in resolution)}")
        print(f" Global Cells       : {global_mesh.number_of_cells()}")
        print(f" Global Vector DOFs : {vector_space.number_of_global_dofs()}")
        print(f" CG Iterations      : {diagnostics['iterations']}")
        print(f" CG Converged       : {diagnostics['converged']}")
        print(f" True Rel Residual  : {diagnostics['true_relative_residual']:.5e}")
        print(f" Boundary Abs Error : {diagnostics['boundary_absolute_error']:.5e}")
        print(f" L2 Absolute Error  : {l2_abs:.5e}")
        print(f" L2 Relative Error  : {l2_rel:.5e}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix-Free 线弹性极简 Demo 驱动")
    parser.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="空间维度 (2 或 3，默认: 2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="各轴网格剖分分辨率 (默认: 8)",
    )
    args = parser.parse_args()
    run_minimal_demo(dimension=args.dim, resolution_n=args.n)
