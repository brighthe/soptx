"""EA Matrix-Free 正确性验证入口.

本脚本验证一阶 Lagrange 线弹性 EA 算子在单 Rank 与多 rank MPI 下的正确性.
单 Rank 检查 EA/FA 算子、EA-CG 与 FA/Scipy 直接解的一致性及 L2 收敛阶; 两 rank
及以上 rank 额外检查并行 EA 与串行 EA、FA/Scipy 直接解的一致性. 正式证据产物由
``tools.matrix_free_evidence`` 负责生成.

使用方法:
    # 单 Rank: 2D 三角形网格.
    python examples/matrix_free_elasticity/verify_ea_correctness.py \
        --model sinusoidal --mesh-type tri --n 8

    # 四 rank: 2D 三角形网格.
    mpiexec -n 4 python examples/matrix_free_elasticity/verify_ea_correctness.py \
        --model sinusoidal --mesh-type tri --n 8

``mpi4py`` 与 ``mpiexec`` 必须链接同一 MPI 实现; 脚本不依赖特定厂商实现.
"""

from __future__ import annotations

import argparse
import os
import sys
from math import log2
from pathlib import Path

# 门禁阈值定义于仓库根目录的 ``tools/``; 按文件路径运行时需先补入仓库根.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)
from mpi4py import MPI

from soptx.fem.distributed import (
    distribute_mesh,
    distribute_vector_space,
    partition_cells,
)
from soptx.fem.solvers import solve_ea_system
from soptx.fem.verification import (
    relative_difference,
    serial_references,
    solution_error,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)
from tools.matrix_free_evidence import contract


# 维数决定这三样, 其余一律向问题对象要, 防止制造解与算子参数漂移.
PROBLEM_FACTORIES = {
    2: {
        "sinusoidal": SinusoidalPlaneStrainElasticity2D,
        "exponential": ExponentialSineManufacturedElasticity2D,
    },
    3: {"polynomial": DivergenceFreePolynomialElasticity3D},
}
MESH_FACTORIES = {
    2: {"tri": TriangleMesh, "quad": QuadrangleMesh},
    3: {"tet": TetrahedronMesh, "hex": HexahedronMesh},
}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}
DEFAULT_MODELS = {2: "sinusoidal", 3: "polynomial"}
DEFAULT_MESH_TYPES = {2: "tri", 3: "tet"}
MODEL_DIMENSIONS = {
    model: dimension
    for dimension, factories in PROBLEM_FACTORIES.items()
    for model in factories
}
MESH_DIMENSIONS = {
    mesh_type: dimension
    for dimension, factories in MESH_FACTORIES.items()
    for mesh_type in factories
}
_MPI_SIZE_ENVIRONMENTS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")


def verdict(passed: bool) -> str:
    """将门禁结果格式化为终端输出标签."""
    return "PASS" if passed else "FAIL"


def build_problem(dimension: int, model: str):
    """按空间维度与模型名称构造受支持的制造解问题."""
    factory = PROBLEM_FACTORIES[dimension].get(model)
    if factory is None:
        supported = ", ".join(sorted(PROBLEM_FACTORIES[dimension]))
        raise ValueError(
            f"{dimension}D 不支持模型 {model!r}; 可选模型: {supported}."
        )
    return factory()


def build_mesh(
    dimension: int,
    mesh_type: str,
    domain: tuple[float, ...],
    resolution: tuple[int, ...],
):
    """按空间维度与网格类型构造均匀有限元网格."""
    factory = MESH_FACTORIES[dimension][mesh_type]
    return factory.from_box(
        list(domain),
        **dict(zip(("nx", "ny", "nz"), resolution)),
    )


def launcher_mpi_size() -> int | None:
    """读取 MPI 启动器声明的进程数, 用于发现 MPI 运行时不匹配.

    Open MPI、MPICH 与 Intel MPI 使用不同的环境变量。本函数只比较启动器
    声明的进程数和 ``mpi4py`` 实际通信域, 不绑定任一种 MPI 实现。
    """
    sizes = {
        int(value)
        for name in _MPI_SIZE_ENVIRONMENTS
        if (value := os.environ.get(name)) is not None and value.isdigit()
    }
    if len(sizes) > 1:
        raise RuntimeError(f"MPI 启动器环境变量声明了冲突的进程数: {sorted(sizes)}")
    return next(iter(sizes), None)


def run_serial_ea_correctness(
    dimension: int = 2,
    resolution_n: int = 8,
    model: str = "sinusoidal",
    mesh_type: str = "tri",
    *,
    show_details: bool = True,
) -> tuple[bool, float]:
    """验证一档网格上 EA Matrix-Free 求解的正确性.

    验证包括 EA/FA 的原始与 Dirichlet 后 MatVec 一致性、正定性探针、EA-CG
    解与 FA/Scipy 直接解一致性，以及制造解相对 L2 误差。

    参数:
        dimension: 空间维度, 取值为 ``2`` 或 ``3``.
        resolution_n: 每个坐标轴的均匀网格剖分数.
        model: 制造解模型名称.
        mesh_type: 网格类型: 2D 为 ``tri`` 或 ``quad``, 3D 为 ``tet`` 或 ``hex``.
        show_details: 是否打印该网格的完整正确性报告.

    返回:
        tuple[bool, float]: 全部 EA 正确性门禁是否通过, 以及相对 L2 位移误差.
    """
    degree = contract.DEFAULT_DEGREE
    problem = build_problem(dimension, model)
    resolution = (resolution_n,) * dimension
    mesh = build_mesh(dimension, mesh_type, problem.domain, resolution)
    scalar_space = LagrangeFESpace(mesh, p=degree, ctype="C")
    vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(mesh),
    )

    # FA/Scipy 只作为独立黄金参考; 被验证的求解路径始终是 EA Matrix-Free CG.
    matvec_reference, direct_solution = serial_references(
        vector_space,
        problem,
        material,
        degree=degree,
        seed=contract.REFERENCE_RANDOM_SEED,
    )
    ea_solution, diagnostics = solve_ea_system(
        vector_space,
        problem,
        material,
        degree=degree,
        dof_comm=None,
    )
    _, solution_relative_error = relative_difference(ea_solution, direct_solution)
    ea_function = vector_space.function(dtype=bm.float64)
    ea_function[:] = ea_solution
    _, l2_relative_error = solution_error(mesh, ea_function, problem, degree)

    matvec_gates = contract.matvec_reference_gates(matvec_reference)
    solution_gate = contract.explicit_solution_gate(solution_relative_error)
    passed = all(matvec_gates.values()) and solution_gate and diagnostics["converged"]

    if show_details:
        print("\n" + "=" * 72)
        print(f" EA Matrix-Free Correctness [{dimension}D - {type(problem).__name__}]")
        print("=" * 72)
        print(f" Grid                  : {mesh_type}, {'x'.join(str(value) for value in resolution)}")
        print(f" Global Cells           : {mesh.number_of_cells()}")
        print(f" Global Vector DOFs     : {vector_space.number_of_global_dofs()}")
        print("-" * 72)
        print(" [1] EA/FA Operator Agreement")
        print(f"   Raw MatVec           : {matvec_reference['raw_relative_error']:.5e} -> [{verdict(matvec_gates['raw_matvec'])}]")
        print(f"   Dirichlet MatVec     : {matvec_reference['dirichlet_relative_error']:.5e} -> [{verdict(matvec_gates['dirichlet_matvec'])}]")
        print(f"   Positive Energy      : {matvec_reference['random_vector_energy']:.5e} -> [{verdict(matvec_gates['positive_definite'])}]")
        print(" [2] EA-CG Solve Agreement")
        print(f"   CG Converged         : {diagnostics['converged']}")
        print(f"   True Relative Residual: {diagnostics['true_relative_residual']:.5e}")
        print(f"   Boundary Error       : {diagnostics['boundary_absolute_error']:.5e}")
        print(f"   EA-CG / FA-Direct    : {solution_relative_error:.5e} -> [{verdict(solution_gate)}]")
        print(" [3] Manufactured-Solution Accuracy")
        print(f"   Relative L2 Error    : {l2_relative_error:.5e}")
        print("-" * 72)
        print(f" Overall EA Correctness : [{verdict(passed)}]")
        print("=" * 72 + "\n")
    return passed, l2_relative_error


def run_parallel_ea_correctness(
    dimension: int = 2,
    resolution_n: int = 8,
    model: str = "sinusoidal",
    mesh_type: str = "tri",
    *,
    show_details: bool = True,
) -> tuple[bool, float] | None:
    """验证多 rank 重叠副本 EA 解与串行 EA/FA 黄金参考的一致性.

    参数:
        dimension: 空间维度, 取值为 ``2`` 或 ``3``.
        resolution_n: 每个坐标轴的均匀网格剖分数.
        model: 制造解模型名称.
        mesh_type: 网格类型: 2D 为 ``tri`` 或 ``quad``, 3D 为 ``tet`` 或 ``hex``.
        show_details: 是否由 Root 打印该网格的完整正确性报告.

    返回:
        tuple[bool, float] | None: Root 返回门禁结果与相对 L2 误差; 其他 rank 返回
        ``None``.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    problem = build_problem(dimension, model)
    resolution = (resolution_n,) * dimension

    if rank == 0:
        global_mesh = build_mesh(
            dimension,
            mesh_type,
            problem.domain,
            resolution,
        )
        scalar_space = LagrangeFESpace(global_mesh, p=contract.DEFAULT_DEGREE, ctype="C")
        vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
        split_coordinate = 0.5 * (problem.domain[0] + problem.domain[1])
        cell_masks = partition_cells(global_mesh, comm.Get_size(), split_coordinate=split_coordinate)
    else:
        global_mesh = None
        scalar_space = None
        vector_space = None
        cell_masks = None

    distributed_mesh = distribute_mesh(global_mesh, cell_masks, comm=comm)
    distributed_space = distribute_vector_space(
        scalar_space,
        vector_space,
        distributed_mesh,
        cell_masks,
        components=dimension,
        comm=comm,
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(distributed_space.space.mesh),
    )
    local_solution, diagnostics = solve_ea_system(
        distributed_space.space,
        problem,
        material,
        degree=contract.DEFAULT_DEGREE,
        dof_comm=distributed_space.dof_comm,
    )
    references = distributed_space.dof_comm.refs(local_solution.shape[0])
    parallel_solution = distributed_space.dof_comm.gather_add(
        local_solution / references,
        root=0,
    )

    if rank != 0:
        return None

    assert global_mesh is not None
    assert vector_space is not None
    assert parallel_solution is not None
    global_material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(global_mesh),
    )
    matvec_reference, direct_solution = serial_references(
        vector_space,
        problem,
        global_material,
        degree=contract.DEFAULT_DEGREE,
        seed=contract.REFERENCE_RANDOM_SEED,
    )
    serial_solution, _ = solve_ea_system(
        vector_space,
        problem,
        global_material,
        degree=contract.DEFAULT_DEGREE,
        dof_comm=None,
    )
    _, parallel_direct_error = relative_difference(parallel_solution, direct_solution)
    _, parallel_serial_error = relative_difference(parallel_solution, serial_solution)
    parallel_function = vector_space.function(dtype=bm.float64)
    parallel_function[:] = parallel_solution
    _, l2_relative_error = solution_error(
        global_mesh,
        parallel_function,
        problem,
        contract.DEFAULT_DEGREE,
    )
    matvec_gates = contract.matvec_reference_gates(matvec_reference)
    direct_gate = contract.explicit_solution_gate(parallel_direct_error)
    serial_gate = parallel_serial_error <= contract.PARALLEL_SOLUTION_RELATIVE_TOL
    passed = (
        all(matvec_gates.values())
        and direct_gate
        and serial_gate
        and diagnostics["converged"]
    )

    if show_details:
        print("\n" + "=" * 72)
        print(f" Parallel EA Correctness [{dimension}D, {comm.Get_size()} ranks]")
        print("=" * 72)
        print(f" Grid                  : {mesh_type}, {'x'.join(str(value) for value in resolution)}")
        print(f" True Relative Residual : {diagnostics['true_relative_residual']:.5e}")
        print(f" Boundary Error         : {diagnostics['boundary_absolute_error']:.5e}")
        print(f" EA/FA Raw MatVec       : {matvec_reference['raw_relative_error']:.5e} -> [{verdict(matvec_gates['raw_matvec'])}]")
        print(f" EA/FA Dirichlet MatVec : {matvec_reference['dirichlet_relative_error']:.5e} -> [{verdict(matvec_gates['dirichlet_matvec'])}]")
        print(f" EA Positive Energy     : {matvec_reference['random_vector_energy']:.5e} -> [{verdict(matvec_gates['positive_definite'])}]")
        print(f" Parallel EA / FA-Direct: {parallel_direct_error:.5e} -> [{verdict(direct_gate)}]")
        print(f" Parallel EA / Serial EA: {parallel_serial_error:.5e} -> [{verdict(serial_gate)}]")
        print(f" Relative L2 Error      : {l2_relative_error:.5e}")
        print(f" Overall EA Correctness : [{verdict(passed)}]")
        print("=" * 72 + "\n")
    return passed, l2_relative_error


def _evaluate_convergence(
    dimension: int,
    model: str,
    mesh_type: str,
    resolutions: tuple[int, ...],
    results: list[tuple[bool, float]],
) -> bool:
    """汇总多档网格的 EA 正确性门禁与相对 L2 收敛阶."""
    relative_errors = [result[1] for result in results]
    orders = [
        log2(relative_errors[index] / relative_errors[index + 1])
        for index in range(len(relative_errors) - 1)
    ]

    print("=" * 72)
    print(f" EA Relative-L2 Convergence [{dimension}D, {model}, {mesh_type}, P1]")
    print("=" * 72)
    print(" Resolution       Relative L2 Error      Observed Order")
    print(f" {resolutions[0]:<16} {relative_errors[0]:.5e}      -")
    for index, order in enumerate(orders, start=1):
        print(f" {resolutions[index]:<16} {relative_errors[index]:.5e}      {order:.5f}")
    convergence_gate = orders[-1] >= contract.MINIMUM_FINAL_L2_ORDER
    print(
        f" Final-Order Gate     : {orders[-1]:.5f} "
        f">= {contract.MINIMUM_FINAL_L2_ORDER:g} -> [{verdict(convergence_gate)}]"
    )
    print("=" * 72 + "\n")
    return all(result[0] for result in results) and convergence_gate


def run_serial_convergence_study(
    dimension: int,
    coarse_resolution_n: int,
    model: str,
    mesh_type: str,
    refinements: int,
) -> bool:
    """运行连续二倍加密网格并检查单 Rank EA 解的相对 L2 收敛阶.

    参数:
        dimension: 空间维度, 取值为 ``2`` 或 ``3``.
        coarse_resolution_n: 最粗网格每个坐标轴的剖分数.
        model: 制造解模型名称.
        mesh_type: 网格类型.
        refinements: 从最粗网格起连续二倍加密的次数, 至少为 ``2``.

    返回:
        bool: 所有网格的 EA 正确性门禁与末段收敛阶门禁是否全部通过.
    """
    resolutions = tuple(
        coarse_resolution_n * 2**level
        for level in range(refinements + 1)
    )
    results = [
        run_serial_ea_correctness(
            dimension,
            resolution_n,
            model,
            mesh_type,
            show_details=resolution_n == resolutions[-1],
        )
        for resolution_n in resolutions
    ]
    return _evaluate_convergence(dimension, model, mesh_type, resolutions, results)


def run_parallel_convergence_study(
    dimension: int,
    coarse_resolution_n: int,
    model: str,
    mesh_type: str,
    refinements: int,
) -> bool | None:
    """运行连续二倍加密网格并检查两 rank EA 解的相对 L2 收敛阶.

    参数:
        dimension: 空间维度, 取值为 ``2`` 或 ``3``.
        coarse_resolution_n: 最粗网格每个坐标轴的剖分数.
        model: 制造解模型名称.
        mesh_type: 网格类型.
        refinements: 从最粗网格起连续二倍加密的次数, 至少为 ``2``.

    返回:
        bool | None: Root 返回全部网格门禁结论; 其他 rank 返回 ``None``.
    """
    resolutions = tuple(
        coarse_resolution_n * 2**level
        for level in range(refinements + 1)
    )
    root_results: list[tuple[bool, float]] = []
    for resolution_n in resolutions:
        result = run_parallel_ea_correctness(
            dimension,
            resolution_n,
            model,
            mesh_type,
            show_details=resolution_n == resolutions[-1],
        )
        if MPI.COMM_WORLD.Get_rank() == 0:
            assert result is not None
            root_results.append(result)

    if MPI.COMM_WORLD.Get_rank() != 0:
        return None
    return _evaluate_convergence(
        dimension,
        model,
        mesh_type,
        resolutions,
        root_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EA Matrix-Free 多 rank 正确性验证")
    parser.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=None,
        help="空间维度; 未给 --model 时用于选择该维度的默认模型",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="最粗网格的每轴剖分数 (默认: 8)",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        default=2,
        help="从最粗网格起连续二倍加密的次数, 至少为 2 (默认: 2)",
    )
    parser.add_argument(
        "--model",
        choices=("sinusoidal", "exponential", "polynomial"),
        default=None,
        help="制造解模型; 指定后自动确定空间维度",
    )
    parser.add_argument(
        "--mesh-type",
        choices=("tri", "quad", "tet", "hex"),
        default=None,
        help="网格类型; 2D 可选 tri/quad, 3D 可选 tet/hex",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="显式要求单 Rank EA 验证; 多 rank MPI 启动时拒绝运行",
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n 必须为正整数")
    if args.refinements < 2:
        parser.error("--refinements 至少为 2, 以确保使用三档网格验证收敛阶")
    if args.model is None:
        dimension = args.dim or 2
        model = DEFAULT_MODELS[dimension]
    else:
        model = args.model
        dimension = MODEL_DIMENSIONS[model]
        if args.dim is not None and args.dim != dimension:
            parser.error(f"--model {model} 对应 {dimension}D, 与 --dim {args.dim} 冲突")
    mesh_type = args.mesh_type or DEFAULT_MESH_TYPES[dimension]
    if MESH_DIMENSIONS[mesh_type] != dimension:
        parser.error(f"--mesh-type {mesh_type} 不支持 {dimension}D 模型 {model}")
    comm = MPI.COMM_WORLD
    declared_size = launcher_mpi_size()
    if declared_size is not None and declared_size != comm.Get_size():
        parser.error(
            "MPI 启动器声明的进程数与 mpi4py 实际通信域不一致; "
            "请使用安装 mpi4py 的 Python 环境提供的 mpiexec "
            "(Conda 环境通常为 $CONDA_PREFIX/bin/mpiexec)"
        )
    if args.serial and comm.Get_size() != 1:
        parser.error("--serial 只允许单 Rank 运行")
    if comm.Get_size() == 1:
        success = run_serial_convergence_study(
            dimension,
            args.n,
            model,
            mesh_type,
            args.refinements,
        )
    else:
        result = run_parallel_convergence_study(
            dimension,
            args.n,
            model,
            mesh_type,
            args.refinements,
        )
        if comm.Get_rank() == 0:
            assert result is not None
            success = result
        else:
            success = None
        success = comm.bcast(success, root=0)
    sys.exit(0 if success else 1)
