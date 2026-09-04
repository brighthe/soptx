"""CPU 上 EA 与 FA 线弹性算子的效率对照.

本脚本固定 CPU NumPy 后端、制造解、材料、边界条件、P1 空间和 CG 停机准则, 只
切换 ``operator_level`` 为 ``"fa"`` 或 ``"ea"``. 它分别测量刚度算子构造、裸
MatVec 和带 Dirichlet 条件的 CG 求解时间, 并统计算子长期保存数组的字节数.

本脚本不承担正确性门禁; 运行性能测试前, 应先运行 ``verify_ea_correctness.py``.
缺省的存储统计不是进程峰值内存, 不包含网格、右端项、解向量和临时工作数组;
需要进程峰值内存时用 ``--mode serial-peak-rss``, 它一个进程只测一个算子层级.

使用方法:
    python examples/matrix_free_elasticity/benchmark_cpu_ea.py --n 16 --levels 3
    python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
        --model exponential --mesh-type quad --n 16 --levels 3
    python examples/matrix_free_elasticity/benchmark_cpu_ea.py \
        --model polynomial --mesh-type hex --n 2 --levels 3
    python -u examples/matrix_free_elasticity/benchmark_cpu_ea.py \
        --mode serial-peak-rss --model polynomial --mesh-type tet --n 64 \
        --operator-level ea --assembly-method fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any, Callable, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh
from mpi4py import MPI

from soptx.fem.distributed import (
    OverlapOperator,
    distribute_mesh,
    distribute_vector_space,
    partition_cells,
)
from soptx.fem.analyzers import build_serial_analyzer
from soptx.fem.matrix_free import (
    ElasticityEAOperator,
    PreparedLinearSystem,
    solve_matrix_free_system,
)
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    SinusoidalPlaneStrainElasticity2D,
)


PROBLEM_FACTORIES = {
    "sinusoidal": (2, SinusoidalPlaneStrainElasticity2D),
    "exponential": (2, ExponentialSineManufacturedElasticity2D),
    "polynomial": (3, DivergenceFreePolynomialElasticity3D),
}
MESH_FACTORIES = {
    "tri": (2, TriangleMesh),
    "quad": (2, QuadrangleMesh),
    "tet": (3, TetrahedronMesh),
    "hex": (3, HexahedronMesh),
}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}
DEFAULT_MESH_TYPES = {2: "tri", 3: "tet"}
DEGREE = 1
RTOL = 1.0e-10
ATOL = 1.0e-12
MAXITER = 5000


def build_context(model: str, mesh_type: str, resolution: int):
    """构造一档 CPU 网格、有限元空间和材料.

    参数:
        model: 制造解模型名称.
        mesh_type: 网格类型名称.
        resolution: 各坐标轴方向的均匀剖分数.

    返回:
        tuple: ``(dimension, problem, vector_space, material, mesh)``.
    """
    dimension, problem_factory = PROBLEM_FACTORIES[model]
    mesh_dimension, mesh_factory = MESH_FACTORIES[mesh_type]
    if mesh_dimension != dimension:
        raise ValueError(f"模型 {model!r} 与网格 {mesh_type!r} 的维数不匹配.")

    problem = problem_factory()
    mesh = mesh_factory.from_box(
        list(problem.domain),
        **dict(zip(("nx", "ny", "nz"), (resolution,) * dimension)),
    )
    scalar_space = LagrangeFESpace(mesh, p=DEGREE, ctype="C")
    vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(mesh),
    )
    return dimension, problem, vector_space, material, mesh


def tensor_nbytes(value: Any) -> int:
    """返回单个后端张量的已分配字节数.

    本 benchmark 固定 NumPy 后端, 但仍通过 ``bm.to_numpy`` 读取后端张量的存储
    元数据, 以便将来复用该统计逻辑时不依赖私有数组属性.
    """
    if value is None:
        return 0
    return int(bm.to_numpy(value).nbytes)


def nested_tensor_nbytes(value: Any) -> int:
    """递归统计张量或张量序列的已分配字节数."""
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return sum(nested_tensor_nbytes(item) for item in value)
    return tensor_nbytes(value)


def operator_storage_bytes(operator_level: str, operator: Any) -> int:
    """统计 FA 或 EA 算子为重复 MatVec 长期保存的数组字节数.

    ``fa`` 统计 COO 的数值、行索引和列索引; ``ea`` 统计每个常量积分器保存的
    单元矩阵 ``value`` 与单元到全局自由度映射 ``to_gdof``. 该口径刻意不把网格、
    右端项、解向量和 MatVec 临时数组计入, 因而不能表述为进程峰值内存.
    """
    if operator_level == "fa":
        return sum(tensor_nbytes(getattr(operator, name, None)) for name in ("data", "row", "col"))

    total = 0
    for integrator in operator.integrators.values():
        total += nested_tensor_nbytes(getattr(integrator, "value", None))
        total += nested_tensor_nbytes(getattr(integrator, "to_gdof", None))
    return total


def measure_seconds(action, warmup: int, repeats: int) -> float:
    """预热后执行动作多次并返回 wall time 中位数."""
    for _ in range(warmup):
        action()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    return float(median(samples))


def measure_parallel_seconds(action, comm: Any, warmup: int, repeats: int) -> float:
    """测量并报告以最慢 rank 为准的 wall time 中位数.

    每个样本前设置全局栅栏, 并对各 rank 的本地耗时执行 ``MPI.MAX`` 归约. 因此
    返回值描述一次同步并行阶段真正限制整体进度的最慢 rank, 而非 root 的局部时间.
    """
    for _ in range(warmup):
        comm.Barrier()
        action()
    samples: list[float] = []
    for _ in range(repeats):
        comm.Barrier()
        start = time.perf_counter()
        action()
        samples.append(comm.allreduce(time.perf_counter() - start, op=MPI.MAX))
    return float(median(samples))


def measure_profiled_parallel_matvec(
    operator: Any,
    probe: Any,
    overlap_operator: OverlapOperator,
    comm: Any,
    warmup: int,
    repeats: int,
    matvec_repeats: int,
) -> tuple[float, dict[str, float]]:
    """测量完整系统 MatVec 及其 OverlapOperator 内部阶段.

    ``operator`` 是已施加 Dirichlet 条件的系统算子. 总 MatVec 时间含边界投影和
    ``OverlapOperator``; 分解项只覆盖后者的输入同步、本地单元核和输出同步. 每个
    分解项对 rank 取最大值, 再按样本取中位数, 与并行 wall time 的统计口径一致.
    """
    def apply_repeatedly() -> None:
        for _ in range(matvec_repeats):
            _ = operator @ probe

    overlap_operator.enable_profiling()
    for _ in range(warmup):
        comm.Barrier()
        apply_repeatedly()

    total_samples: list[float] = []
    profile_samples: dict[str, list[float]] = {
        "input_sync_seconds": [],
        "local_kernel_seconds": [],
        "output_sync_seconds": [],
    }
    for _ in range(repeats):
        overlap_operator.reset_profile()
        comm.Barrier()
        start = time.perf_counter()
        apply_repeatedly()
        total_samples.append(
            comm.allreduce(time.perf_counter() - start, op=MPI.MAX) / matvec_repeats
        )
        profile = overlap_operator.profile()
        if int(profile["calls"]) != matvec_repeats:
            raise RuntimeError("MatVec 剖析计数与 --matvec-repeats 不一致.")
        for name, samples in profile_samples.items():
            maximum = comm.allreduce(float(profile[name]), op=MPI.MAX)
            samples.append(maximum / matvec_repeats)

    overlap_operator.enable_profiling(False)
    return float(median(total_samples)), {
        name: float(median(samples)) for name, samples in profile_samples.items()
    }


def build_operator(
    vector_space: TensorFunctionSpace,
    problem: Any,
    material: Any,
    operator_level: str,
    assembly_method: str = "standard",
):
    """构造指定层级的串行分析器与未施加边界条件的刚度算子.

    ``assembly_method`` 只改变单元矩阵的收缩顺序和中间张量规模, 不改变数值;
    它对进程峰值内存的影响见 ``--mode serial-peak-rss``.
    """
    analyzer = build_serial_analyzer(
        vector_space,
        problem,
        material,
        degree=DEGREE,
        operator_level=operator_level,
        assembly_method=assembly_method,
    )
    return analyzer, analyzer.assemble_stiff_matrix()


def benchmark_level(
    model: str,
    mesh_type: str,
    resolution: int,
    operator_level: str,
    warmup: int,
    repeats: int,
    matvec_repeats: int,
    assembly_method: str = "standard",
) -> dict[str, Any]:
    """测量某一网格和算子层级的构造、MatVec、CG 与保存存储.

    每个计时样本均重新从零初值运行 CG, 因而 FA 与 EA 的迭代数可直接比较. 裸
    MatVec 使用同一个确定性向量, 不将 Dirichlet 包装成本混入该指标.
    """
    dimension, problem, vector_space, material, mesh = build_context(
        model, mesh_type, resolution
    )

    def construct_once() -> None:
        build_operator(
            vector_space, problem, material, operator_level, assembly_method
        )

    construction_seconds = measure_seconds(construct_once, warmup, repeats)
    analyzer, raw_operator = build_operator(
        vector_space, problem, material, operator_level, assembly_method
    )
    stored_bytes = operator_storage_bytes(operator_level, raw_operator)
    random_rand = cast(Callable[..., Any], bm.random.rand)
    probe = 2.0 * random_rand(vector_space.number_of_global_dofs()) - 1.0

    def matvec_once() -> None:
        for _ in range(matvec_repeats):
            _ = raw_operator @ probe

    matvec_seconds = measure_seconds(matvec_once, warmup, repeats) / matvec_repeats
    load = analyzer.assemble_body_force_vector()
    system_operator, system_load = analyzer.apply_bc(raw_operator, load)
    solution = bm.zeros(
        (vector_space.number_of_global_dofs(),),
        dtype=bm.float64,
        device=bm.get_device(mesh),
    )

    def solve_once() -> None:
        solution[:] = bm.zeros_like(solution)
        analyzer.solve_system(
            system_operator,
            system_load,
            solution,
            solver="cg",
            rtol=RTOL,
            atol=ATOL,
            maxiter=MAXITER,
        )

    solve_seconds = measure_seconds(solve_once, warmup, repeats)
    solve_result = cast(
        tuple[Any, dict[str, Any]],
        analyzer.solve_system(
            system_operator,
            system_load,
            solution,
            solver="cg",
            rtol=RTOL,
            atol=ATOL,
            maxiter=MAXITER,
        ),
    )
    solver_info = solve_result[1]
    residual = system_operator @ solution - system_load
    true_relative_residual = float(bm.linalg.norm(residual)) / max(
        float(bm.linalg.norm(system_load)), 1.0e-30
    )
    return {
        "dimension": dimension,
        "resolution": resolution,
        "mesh_type": mesh_type,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(vector_space.number_of_global_dofs()),
        "operator_level": operator_level,
        "assembly_method": assembly_method,
        "construction_seconds": construction_seconds,
        "matvec_seconds": matvec_seconds,
        "solve_seconds": solve_seconds,
        "stored_operator_bytes": stored_bytes,
        "cg_iterations": int(solver_info["niter"]),
        "cg_converged": bool(solver_info["converged"]),
        "true_relative_residual": true_relative_residual,
    }


def paired_rows(
    model: str,
    mesh_type: str,
    resolutions: Iterable[int],
    warmup: int,
    repeats: int,
    matvec_repeats: int,
    assembly_method: str = "standard",
) -> list[dict[str, Any]]:
    """在每档网格上依次执行 FA 与 EA, 并合并为一行对照结果.

    两个层级在同一个进程内先后构造, 因此本函数的结果可以比较时间与算子字节数,
    但**不能**用来归属进程峰值内存 —— 那需要 ``--mode serial-peak-rss``.
    """
    rows: list[dict[str, Any]] = []
    for resolution in resolutions:
        fa = benchmark_level(
            model, mesh_type, resolution, "fa", warmup, repeats, matvec_repeats,
            assembly_method,
        )
        ea = benchmark_level(
            model, mesh_type, resolution, "ea", warmup, repeats, matvec_repeats,
            assembly_method,
        )
        rows.append(
            {
                "dimension": fa["dimension"],
                "resolution": resolution,
                "mesh_type": mesh_type,
                "cells": fa["cells"],
                "dofs": fa["dofs"],
                "fa": fa,
                "ea": ea,
                "matvec_speedup_fa_over_ea": fa["matvec_seconds"] / max(ea["matvec_seconds"], 1.0e-30),
                "solve_speedup_fa_over_ea": fa["solve_seconds"] / max(ea["solve_seconds"], 1.0e-30),
                "storage_ratio_fa_over_ea": fa["stored_operator_bytes"] / max(ea["stored_operator_bytes"], 1),
            }
        )
    return rows


def print_report(rows: list[dict[str, Any]]) -> None:
    """以紧凑表格输出 EA/FA 对照结果."""
    header = (
        "  n     cells     dofs | FA build  EA build | FA mv     EA mv     "
        "FA/EA | FA CG     EA CG     FA/EA | FA/EA store"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        fa = row["fa"]
        ea = row["ea"]
        print(
            f"{row['resolution']:>3} {row['cells']:>9} {row['dofs']:>8} | "
            f"{fa['construction_seconds']:>8.4f} {ea['construction_seconds']:>8.4f} | "
            f"{fa['matvec_seconds']:>8.2e} {ea['matvec_seconds']:>8.2e} "
            f"{row['matvec_speedup_fa_over_ea']:>5.2f} | "
            f"{fa['solve_seconds']:>8.4f} {ea['solve_seconds']:>8.4f} "
            f"{row['solve_speedup_fa_over_ea']:>5.2f} | "
            f"{row['storage_ratio_fa_over_ea']:>11.2f}"
        )
    print("\n时间单位: s. FA/EA > 1 表示 EA 的对应量更小.")
    print("存储仅含算子长期保存数组, 不是进程峰值内存.")
    for row in rows:
        for operator_level in ("fa", "ea"):
            result = row[operator_level]
            print(
                f"n={row['resolution']}, {operator_level.upper()}: "
                f"CG iters={result['cg_iterations']}, "
                f"converged={result['cg_converged']}, "
                f"true residual={result['true_relative_residual']:.3e}, "
                f"stored={result['stored_operator_bytes']} B"
            )


def benchmark_mpi_ea(arguments: argparse.Namespace) -> int:
    """测量 CPU MPI EA 的构造、MatVec 与 CG 时间.

    ``mpi-ea-strong`` 固定物理区域和全局网格. ``mpi-ea-weak`` 随 rank 数沿 x
    方向按比例扩展物理区域及网格数, 因而保持各方向单元尺寸和每个 rank 的单元
    数不变. 分区器沿 x 方向生成连续条带, 支持任意非空的正整数 rank 数.

    单次系统 MatVec 包含 ``OverlapOperator`` 的输入一致化与输出求和同步. 当前
    benchmark 报告该完整 MatVec 的总时间, 未把 MPI 同步与单元核进一步拆分; 后者
    需要在算子内部加入低扰动事件剖析, 不应由 benchmark 外部猜测.

    注意, 此处仍使用无预条件 CG. 即使局部工作量不变, 扩展全局区域也会改变椭圆
    型问题的条件数和 CG 迭代数; 因此端到端求解时间不能单独用作可扩展性结论.
    """
    comm = MPI.COMM_WORLD
    rank, ranks = comm.Get_rank(), comm.Get_size()
    dimension, problem_factory = PROBLEM_FACTORIES[arguments.model]
    reference_problem = problem_factory()
    resolution = (arguments.n,) * dimension
    if arguments.mode == "mpi-ea-weak":
        domain = list(reference_problem.domain)
        domain[1] = domain[0] + ranks * (domain[1] - domain[0])
        problem = problem_factory(domain=domain)
        resolution = (ranks * arguments.n,) + resolution[1:]
    else:
        problem = reference_problem
    if rank == 0:
        _, mesh_factory = MESH_FACTORIES[arguments.mesh_type]
        global_mesh = mesh_factory.from_box(
            list(problem.domain), **dict(zip(("nx", "ny", "nz"), resolution))
        )
        scalar_space = LagrangeFESpace(global_mesh, p=DEGREE, ctype="C")
        vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
        split_coordinate = 0.5 * (problem.domain[0] + problem.domain[1])
        masks = partition_cells(global_mesh, ranks, split_coordinate=split_coordinate)
    else:
        global_mesh = scalar_space = vector_space = masks = None
    distributed_mesh = distribute_mesh(global_mesh, masks, comm=comm)
    distributed_space = distribute_vector_space(
        scalar_space, vector_space, distributed_mesh, masks,
        components=dimension, comm=comm,
    )
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension], lame_lambda=problem.lam,
        shear_modulus=problem.mu, device=bm.get_device(distributed_space.space.mesh),
    )

    def prepare_system() -> PreparedLinearSystem:
        facade = ElasticityEAOperator(
            distributed_space.space,
            problem,
            material,
            degree=DEGREE,
            dof_comm=distributed_space.dof_comm,
            assembly_method=arguments.assembly_method,
        )
        operator, load = facade.assemble()
        return PreparedLinearSystem(
            operator=operator,
            load=load,
            prescribed=facade.prescribed_solution,
            boundary_dofs=facade.boundary_dofs,
        )

    construction_seconds = measure_parallel_seconds(
        prepare_system, comm, arguments.warmup, arguments.repeats
    )
    system = prepare_system()
    probe = bm.ones_like(system.load)
    overlap_operator = getattr(system.operator, "form", None)
    if not isinstance(overlap_operator, OverlapOperator):
        raise RuntimeError("MPI EA 系统算子未包含 OverlapOperator, 无法执行阶段剖析.")
    matvec_seconds, matvec_profile_seconds = measure_profiled_parallel_matvec(
        system.operator,
        probe,
        overlap_operator,
        comm,
        arguments.warmup,
        arguments.repeats,
        arguments.matvec_repeats,
    )

    def solve_once():
        return solve_matrix_free_system(
            system,
            distributed_space.dof_comm,
            rtol=RTOL,
            atol=ATOL,
            maxiter=MAXITER,
        )

    cg_seconds = measure_parallel_seconds(
        solve_once, comm, arguments.warmup, arguments.repeats
    )
    solution, diagnostics = solve_once()
    cg_iteration_seconds = cg_seconds / max(int(diagnostics["iterations"]), 1)
    pipeline_seconds = construction_seconds + cg_seconds
    if rank == 0:
        assert vector_space is not None
        payload = {
            "mode": arguments.mode,
            "ranks": ranks,
            "model": arguments.model,
            "mesh_type": arguments.mesh_type,
            "domain": list(problem.domain),
            "resolution": resolution,
            "global_vector_dofs": int(vector_space.number_of_global_dofs()),
            "assembly_method": arguments.assembly_method,
            # 计时口径必须随产物一起落盘: 强扩展是跨多次独立运行拼出来的一条
            # 曲线, 各档 warmup/repeats 不同则中位数之间不可比, 而口径只存在
            # 于当时的命令行里, 不记下来事后无从复核。
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "ea_construction_seconds_max_rank": construction_seconds,
            "ea_system_matvec_seconds_max_rank": matvec_seconds,
            "ea_overlap_matvec_profile_seconds_max_rank": matvec_profile_seconds,
            "ea_cg_seconds_max_rank": cg_seconds,
            "ea_cg_iteration_seconds_max_rank": cg_iteration_seconds,
            "ea_pipeline_seconds_max_rank": pipeline_seconds,
            "solver": diagnostics,
        }
        print("=" * 72)
        print(f" CPU MPI EA Benchmark [{arguments.mode}, {ranks} ranks]")
        print("=" * 72)
        print(f" Physical domain        : {tuple(problem.domain)}")
        print(f" Global grid           : {'x'.join(str(value) for value in resolution)}")
        print(f" Global vector DOFs    : {vector_space.number_of_global_dofs()}")
        print(f" EA construction / s   : {construction_seconds:.6f}")
        print(f" EA system MatVec / s  : {matvec_seconds:.6e}")
        print(
            "   input sync / ms     : "
            f"{1.0e3 * matvec_profile_seconds['input_sync_seconds']:.3f}"
        )
        print(
            "   local kernel / ms   : "
            f"{1.0e3 * matvec_profile_seconds['local_kernel_seconds']:.3f}"
        )
        print(
            "   output sync / ms    : "
            f"{1.0e3 * matvec_profile_seconds['output_sync_seconds']:.3f}"
        )
        print(f" EA CG solve / s       : {cg_seconds:.6f}")
        print(f" EA CG step / ms       : {1.0e3 * cg_iteration_seconds:.3f}")
        print(f" EA pipeline / s       : {pipeline_seconds:.6f}")
        print(f" CG iterations         : {diagnostics['iterations']}")
        print(f" True relative residual: {diagnostics['true_relative_residual']:.6e}")
        print(f" Boundary error        : {diagnostics['boundary_absolute_error']:.6e}")
        print("=" * 72)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"原始结果已写入: {arguments.output}")
    return 0


def _peak_rss_bytes() -> int:
    """返回本进程到目前为止的驻留集高水位, 单位为字节.

    Linux 上 ``ru_maxrss`` 以 KiB 计, 与 ``/usr/bin/time -v`` 的
    ``Maximum resident set size`` 同源, 因此两者可以互相校验. 该值是**进程**
    高水位, 不是某个数组的大小, 也不会随对象释放而回落.
    """
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def benchmark_peak_rss(arguments: argparse.Namespace) -> int:
    """在单个进程内只构造一个算子层级, 逐阶段记录进程峰值 RSS.

    与 ``serial-fa-ea`` 的区别是本模式**一个进程只走一条路径**: 后者在同一进程
    里先后建 FA 与 EA, 其高水位等于两者的较大值, 无法把 EA 单独隔离出来. 峰值
    内存是进程级量, 只能靠进程隔离来归属, 因此本模式一次只测一个层级.

    阶段划分为 ``mesh`` (网格与空间)、``operator`` (刚度算子构造完成)、
    ``bc`` (施加 Dirichlet 条件后) 与 ``solve`` (CG 求解结束). 报告的是各阶段
    结束时的累计高水位, 相邻两阶段之差即该阶段新增的高水位, 可直接读出"峰值由
    哪一步决定".

    返回:
        int: 进程退出码, 恒为 ``0``.
    """
    resolution = arguments.n
    # 解释器与已导入模块本身就占几百 MiB, 在小自由度档上会淹没算子的差异,
    # 因此先记一个基线阶段, 让"算子带来多少"可以从曲线里减掉.
    stages: list[tuple[str, int]] = [("baseline", _peak_rss_bytes())]

    dimension, problem, vector_space, material, mesh = build_context(
        arguments.model, arguments.mesh_type, resolution
    )
    stages.append(("mesh", _peak_rss_bytes()))

    construction_start = time.perf_counter()
    analyzer, raw_operator = build_operator(
        vector_space,
        problem,
        material,
        arguments.operator_level,
        arguments.assembly_method,
    )
    construction_seconds = time.perf_counter() - construction_start
    stages.append(("operator", _peak_rss_bytes()))

    # 只读取已有张量的 ``nbytes``, 不分配新内存, 因此不需要单独的采样点.
    stored_bytes = operator_storage_bytes(arguments.operator_level, raw_operator)
    # 体力向量组装与 Dirichlet 处理分开采样: 前者走 LinearForm 全装配, 与算子层级
    # 无关; 后者才是 matrix-free 路径特有的边界条件包装. 合并成一个 ``bc`` 阶段会
    # 把两者的瞬态混在一起, 无法判断峰值究竟由哪一步决定.
    load = analyzer.assemble_body_force_vector()
    stages.append(("load", _peak_rss_bytes()))

    system_operator, system_load = analyzer.apply_bc(raw_operator, load)
    stages.append(("bc", _peak_rss_bytes()))

    solver_info: dict[str, Any] = {}
    solve_seconds = 0.0
    true_relative_residual = float("nan")
    if not arguments.skip_solve:
        solution = bm.zeros(
            (vector_space.number_of_global_dofs(),),
            dtype=bm.float64,
            device=bm.get_device(mesh),
        )
        solve_start = time.perf_counter()
        solve_result = cast(
            tuple[Any, dict[str, Any]],
            analyzer.solve_system(
                system_operator,
                system_load,
                solution,
                solver="cg",
                rtol=RTOL,
                atol=ATOL,
                maxiter=arguments.maxiter,
            ),
        )
        solve_seconds = time.perf_counter() - solve_start
        solver_info = solve_result[1]
        residual = system_operator @ solution - system_load
        true_relative_residual = float(bm.linalg.norm(residual)) / max(
            float(bm.linalg.norm(system_load)), 1.0e-30
        )
    stages.append(("solve", _peak_rss_bytes()))

    peak_bytes = stages[-1][1]
    baseline_bytes = stages[0][1]
    payload = {
        "mode": "serial-peak-rss",
        "model": arguments.model,
        "mesh_type": arguments.mesh_type,
        "backend": "numpy",
        "degree": DEGREE,
        "dimension": dimension,
        "resolution": resolution,
        "cells": int(mesh.number_of_cells()),
        "dofs": int(vector_space.number_of_global_dofs()),
        "operator_level": arguments.operator_level,
        "assembly_method": arguments.assembly_method,
        "solved": not arguments.skip_solve,
        "cg": {"rtol": RTOL, "atol": ATOL, "maxiter": arguments.maxiter},
        "construction_seconds": construction_seconds,
        "solve_seconds": solve_seconds,
        "stored_operator_bytes": stored_bytes,
        "peak_rss_bytes": peak_bytes,
        "peak_rss_baseline_bytes": baseline_bytes,
        "peak_rss_above_baseline_bytes": peak_bytes - baseline_bytes,
        "peak_rss_stage_bytes": {name: value for name, value in stages},
        "peak_rss_scope": (
            "进程驻留集高水位 (resource.ru_maxrss), 含解释器、网格、算子与全部临时数组; "
            "与 stored_operator_bytes 的稳态口径不可比"
        ),
    }
    if solver_info:
        payload["cg_iterations"] = int(solver_info["niter"])
        payload["cg_converged"] = bool(solver_info["converged"])
        payload["true_relative_residual"] = true_relative_residual

    gib = 1024.0**3
    print("=" * 72)
    print(
        f" 进程峰值 RSS [{arguments.operator_level.upper()}, "
        f"{arguments.assembly_method}, {arguments.mesh_type} n={resolution}]"
    )
    print("=" * 72)
    print(f" 单元数 / 自由度        : {payload['cells']} / {payload['dofs']}")
    print(f" 算子构造 / s           : {construction_seconds:.4f}")
    previous = 0
    for name, value in stages:
        print(
            f" 峰值 RSS [{name:>8}] / GiB : {value / gib:8.3f}"
            f"   (较上一阶段 +{(value - previous) / gib:.3f})"
        )
        previous = value
    print(f" 扣除基线后的峰值 / GiB : {(peak_bytes - baseline_bytes) / gib:8.3f}")
    print(f" 算子长期保存数组 / GiB : {stored_bytes / gib:8.3f}")
    if solver_info:
        print(f" CG 迭代 / 收敛         : {payload['cg_iterations']} / {payload['cg_converged']}")
        print(f" 真相对残差             : {true_relative_residual:.6e}")
        print(f" CG 求解 / s            : {solve_seconds:.4f}")
    print("=" * 72)
    print("口径: 进程高水位, 不是算子字节数; 一次运行只测一个层级, 不与其他层级同进程比较.")

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"原始结果已写入: {arguments.output}")
    return 0


def parse_arguments() -> argparse.Namespace:
    """解析 CPU EA/FA 效率对照的命令行参数."""
    parser = argparse.ArgumentParser(description="CPU EA/FA 线弹性效率对照")
    parser.add_argument("--model", choices=tuple(PROBLEM_FACTORIES), default="sinusoidal")
    parser.add_argument("--mesh-type", choices=tuple(MESH_FACTORIES))
    parser.add_argument(
        "--mode", default="serial-fa-ea",
        choices=("serial-fa-ea", "serial-peak-rss", "mpi-ea-strong", "mpi-ea-weak"),
        help=(
            "serial-fa-ea 为串行 EA/FA 对照; serial-peak-rss 只测单个层级的进程"
            "峰值 RSS (需配 --operator-level); 其余为 CPU MPI EA 扩展基准"
        ),
    )
    parser.add_argument(
        "--operator-level", choices=("fa", "ea"), default="ea",
        help="serial-peak-rss 专用: 本次进程只构造该层级的算子",
    )
    parser.add_argument(
        "--assembly-method", choices=("standard", "voigt", "fast"), default="standard",
        help="单元矩阵的收缩顺序; 只影响中间张量规模与峰值内存, 不影响数值",
    )
    parser.add_argument(
        "--maxiter", type=int, default=MAXITER,
        help="serial-peak-rss 专用: CG 最大迭代次数; 其余模式固定用 MAXITER",
    )
    parser.add_argument(
        "--skip-solve", action="store_true",
        help="serial-peak-rss 专用: 只测到施加边界条件为止, 不跑 CG",
    )
    parser.add_argument("--n", type=int, default=16, help="最粗网格每个坐标轴的剖分数")
    parser.add_argument("--levels", type=int, default=3, help="包含最粗网格在内的网格档数")
    parser.add_argument("--warmup", type=int, default=1, help="每个计时指标的预热次数")
    parser.add_argument("--repeats", type=int, default=3, help="每个计时指标取中位数的重复次数")
    parser.add_argument("--matvec-repeats", type=int, default=20, help="单个 MatVec 计时样本内的连续次数")
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径; 建议写入 outputs/")
    arguments = parser.parse_args()
    if arguments.n <= 0 or arguments.levels <= 0:
        parser.error("--n 与 --levels 必须为正整数.")
    if arguments.warmup < 0 or arguments.repeats <= 0 or arguments.matvec_repeats <= 0:
        parser.error("--warmup 不得为负, --repeats 与 --matvec-repeats 必须为正整数.")
    if arguments.maxiter <= 0:
        parser.error("--maxiter 必须为正整数.")
    if arguments.mode != "serial-peak-rss" and arguments.skip_solve:
        parser.error("--skip-solve 只对 --mode serial-peak-rss 有效.")
    dimension, _ = PROBLEM_FACTORIES[arguments.model]
    if arguments.mesh_type is None:
        arguments.mesh_type = DEFAULT_MESH_TYPES[dimension]
    mesh_dimension, _ = MESH_FACTORIES[arguments.mesh_type]
    if mesh_dimension != dimension:
        parser.error(f"--mesh-type {arguments.mesh_type} 与模型 {arguments.model} 维数不匹配.")
    return arguments


def main() -> int:
    """执行 CPU EA/FA 效率对照并按需保存 JSON 原始结果."""
    arguments = parse_arguments()
    bm.set_backend("numpy")
    if arguments.mode == "serial-peak-rss":
        return benchmark_peak_rss(arguments)
    if arguments.mode != "serial-fa-ea":
        return benchmark_mpi_ea(arguments)
    resolutions = [arguments.n * 2**level for level in range(arguments.levels)]
    rows = paired_rows(
        arguments.model,
        arguments.mesh_type,
        resolutions,
        arguments.warmup,
        arguments.repeats,
        arguments.matvec_repeats,
        arguments.assembly_method,
    )
    print_report(rows)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": arguments.model,
            "mesh_type": arguments.mesh_type,
            "backend": "numpy",
            "degree": DEGREE,
            "assembly_method": arguments.assembly_method,
            "cg": {"rtol": RTOL, "atol": ATOL, "maxiter": MAXITER},
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "matvec_repeats": arguments.matvec_repeats,
            "storage_scope": "仅算子长期保存数组, 不含网格、向量与临时数组",
            "rows": rows,
        }
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"原始结果已写入: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
