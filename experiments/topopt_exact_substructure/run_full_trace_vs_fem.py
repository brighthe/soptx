"""子结构与传统有限元的完整拓扑优化流程: 三维 MBB 梁 (``FullMBBBeamPoint3d``).
"""
import os

# 运行环境: BLAS 与 OpenMP 在加载时读取线程数, 须在导入 numpy/scipy 之前写入环境变量才生效.
# 固定单线程, 与 topopt_simp_fem 的传统有限元运行口径一致
NUM_THREADS = 1
for _name in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = str(NUM_THREADS)

import argparse
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.core import (blas_environment, is_out_of_memory, limit_cpu_memory, limit_cuda_memory, measure,
                        record_out_of_memory)
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import LagrangeFEMAnalyzer, SubstructureAnalyzer
from soptx.fem.substructure import StructuredSubstructureLayout
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer


# 模型参数
ELEMENT_SIZE = 1.0
LOAD = -1.0
E0 = 1.0
EMIN = 1.0e-7
NU = 0.3

# 计算后端: --backend 的默认值; cuda 须用 pytorch
BACKEND = 'numpy'
# GPU 显存上限 (GiB), 即 --gpu-memory-limit 的默认值: 固定值使各次运行可用显存相同; 总显存 15.89 GiB, 余下约
# 1.9 GiB 留给 CUDA 上下文、cuSPARSE 等不经 PyTorch 分配器的工作区与桌面. 启动时空闲显存不足此值即拒绝运行
CUDA_MEMORY_LIMIT_GIB = 14.0

# 分析参数: 有限元离散与子结构
GRID = (210, 35, 35)
N_FINE = 5
N_SUB = tuple(n // N_FINE for n in GRID)  # 由 GRID 与 N_FINE 推出, 不单独修改
CHUNK_SIZE = 256
TRACE = 'full_trace'
ASSEMBLY_METHOD = 'fast'
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
OPERATOR_LEVEL = 'fa'

# 分析参数: 线性系统求解
SOLVER = 'cg'
CG_TOLERANCE = 1.0e-6
CG_MAXITER = 20000

# 优化参数
VOLFRAC = 0.12
FILTER_RADIUS_CELLS = 3.0
PENALTY = 3.0
OC_OPTIONS = dict(move_limit=0.2, damping_coef=0.5, initial_lambda=1.0e9,
                  bisection_tol=1.0e-3, design_variable_min=0.0)
TOLERANCE = 2.0e-4
CONVERGENCE_WINDOW = 5
VOLUME_TOLERANCE = 1.0e-6
MAX_ITER = 300

# 输出设置
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
VTU_FIELDS = ('density',)

# 各阶段的执行顺序, 仅供显存不足时判定失败阶段: measure 只在阶段正常结束时写入记录,
# 顺序中第一个尚无记录的阶段即失败阶段; 子结构布局与校验只在 substructure 下执行
SETUP_STAGES = {'fem': ('网格', '材料与插值', '分析器', '过滤矩阵', '目标与约束'),
                'substructure': ('网格', '材料与插值', '子结构布局与校验', '分析器', '过滤矩阵', '目标与约束')}
ITERATION_STAGES = ('过滤', '装配', '边界条件', '求解', '灵敏度', '输出', 'OC')

    
def parse_args(argv=None):
    """解析分析方法、预条件子、细网格、子结构划分、优化步数、计算后端、计算设备与内存上限并确定输出目录.

    substructure 下 grid、n_sub、n_fine 须满足 grid = n_sub * n_fine: 三者都给出时检查一致性,
    缺一个时由另外两个推出. n_fine 无法由 grid 与 n_sub 推出时取 N_FINE, grid 无法由 n_sub 推出时
    取 GRID. fem 只用 grid, 给出 n_sub 或 n_fine 时报错. cuda 须用 pytorch 后端, pytorch 后端只支持 none/jacobi
    预条件子. numpy/cpu 的输出目录名与之前相同, 其余组合加 ``<backend>_<device>_`` 前缀.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省读取 sys.argv.

    Returns
    -------
    argparse.Namespace
        分析方法、预条件子、细网格 grid、子结构数 n_sub 与细单元数 n_fine (fem 下均为 None)、
        最大优化步数 max_iter、计算后端 backend、计算设备 device、GPU 显存上限 gpu_memory_limit (cpu 下为
        None)、CPU 内存上限 cpu_memory_limit (未给出或 cuda 下为 None)
        和自动生成的输出目录.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-method', choices=('fem', 'substructure'), required=True,
                        help='分析方法: fem 为传统有限元, substructure 为子结构')
    parser.add_argument('--preconditioner', choices=('none', 'jacobi', 'gmg', 'scipy', 'mumps'),
                        default='gmg', help='CG 预条件子, 默认 gmg; scipy/mumps 仅支持 fem')
    parser.add_argument('--grid', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'),
                        help=f'细网格各方向单元数, 默认 {" ".join(map(str, GRID))}; '
                             'substructure 下须满足 grid = n_sub * n_fine')
    parser.add_argument('--n-sub', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'),
                        help=f'各方向子结构数, 仅 substructure; 默认 {" ".join(map(str, N_SUB))} (= GRID // N_FINE), '
                             '缺省时由 grid 与 n_fine 推出')
    parser.add_argument('--n-fine', type=int, default=None,
                        help='每个子结构各方向的细单元数 m, 仅 substructure; 缺省时由 grid 与 n_sub 推出, '
                             f'两者不全时取 {N_FINE}')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER,
                        help=f'最大优化步数, 默认 {MAX_ITER}')
    parser.add_argument('--backend', choices=('numpy', 'pytorch'), default=BACKEND,
                        help=f'计算后端, 默认 {BACKEND}; --device cuda 须用 pytorch')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='计算设备, 默认 cpu')
    parser.add_argument('--gpu-memory-limit', type=float, default=None, metavar='GIB',
                        help=f'GPU 显存上限 (GiB), 只用于 --device cuda (默认 {CUDA_MEMORY_LIMIT_GIB})')
    parser.add_argument('--cpu-memory-limit', type=float, default=None, metavar='GIB',
                        help='CPU 内存上限 (GiB), 按启动后新增的虚拟内存计, 只用于 --device cpu; 不给则不设上限')
    args = parser.parse_args(argv)
    is_substructure = args.analysis_method == 'substructure'
    if is_substructure and args.preconditioner in ('scipy', 'mumps'):
        parser.error('substructure 仅支持 none/jacobi/gmg 预条件子')
    if args.backend == 'pytorch':
        # 只在 pytorch 后端导入 torch, numpy 运行的进程内存不含 torch
        import torch
        if args.preconditioner not in ('none', 'jacobi'):
            parser.error('--backend pytorch 仅支持 none/jacobi 预条件子')
    if args.device == 'cuda':
        if args.backend != 'pytorch':
            parser.error('--device cuda 须配 --backend pytorch')
        if not torch.cuda.is_available():
            parser.error('--device cuda 需要可用的 CUDA 设备')
        if args.cpu_memory_limit is not None:
            parser.error('--cpu-memory-limit 只用于 --device cpu')
        if args.gpu_memory_limit is None:
            args.gpu_memory_limit = CUDA_MEMORY_LIMIT_GIB
        elif args.gpu_memory_limit <= 0:
            parser.error(f'--gpu-memory-limit 须为正数, 得到 {args.gpu_memory_limit}')
    else:
        if args.gpu_memory_limit is not None:
            parser.error('--gpu-memory-limit 只用于 --device cuda')
        if args.cpu_memory_limit is not None and args.cpu_memory_limit <= 0:
            parser.error(f'--cpu-memory-limit 须为正数, 得到 {args.cpu_memory_limit}')
    if args.max_iter < 1:
        parser.error('--max-iter 须为正整数')
    for option, value in (('--grid', args.grid), ('--n-sub', args.n_sub)):
        if value is not None and min(value) < 1:
            parser.error(f'{option} 各方向须为正整数')

    if not is_substructure:
        if args.n_sub is not None or args.n_fine is not None:
            parser.error('--n-sub 与 --n-fine 仅用于 substructure')
        args.grid = GRID if args.grid is None else tuple(args.grid)
    else:
        grid, n_sub, n_fine = args.grid, args.n_sub, args.n_fine
        if n_fine is None:
            if grid is not None and n_sub is not None:
                if any(g % s for g, s in zip(grid, n_sub)) or len({g // s for g, s in zip(grid, n_sub)}) != 1:
                    parser.error(f'--grid {tuple(grid)} 须为 --n-sub {tuple(n_sub)} 的同一整数倍')
                n_fine = grid[0] // n_sub[0]
            else:
                n_fine = N_FINE
        # 子结构每向至少 2 个细单元才有内部节点可消元
        if n_fine < 2:
            parser.error(f'n_fine 至少为 2, 以保留内部节点, 实际为 {n_fine}')
        if grid is None:
            grid = GRID if n_sub is None else tuple(s * n_fine for s in n_sub)
        # 子结构数由整数除法得到, 不整除时会静默截断, 须在生成目录名之前拒绝
        if n_sub is None:
            if any(g % n_fine for g in grid):
                parser.error(f'grid 各方向须能被 n_fine={n_fine} 整除, 实际为 {tuple(grid)}')
            n_sub = tuple(g // n_fine for g in grid)
        if tuple(grid) != tuple(s * n_fine for s in n_sub):
            parser.error(f'须满足 grid = n_sub * n_fine, 实际 grid={tuple(grid)}, '
                         f'n_sub={tuple(n_sub)}, n_fine={n_fine}')
        args.grid, args.n_sub, args.n_fine = tuple(grid), tuple(n_sub), n_fine
    size = 'x'.join(map(str, args.grid))
    name = (f'{TRACE}_hex_{size}_m{args.n_fine}_{SOLVER}' if is_substructure
            else f'fa_hex_{size}_{SOLVER}')
    # numpy/cpu 沿用原目录名, 已入库的结果不受影响
    prefix = '' if (args.backend, args.device) == ('numpy', 'cpu') else f'{args.backend}_{args.device}_'
    args.output = OUTPUT_DIR.resolve() / f'{prefix}{name}_{args.preconditioner}'
    return args


def main(argv=None):
    """运行选定方法的拓扑优化; 内存或显存不足时记录失败点.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    status : int
        正常结束时为 0, 内存或显存不足时为 2.
    """
    args = parse_args(argv)
    progress: dict[str, Any] = {}
    try:
        return run(args, progress)
    except (MemoryError, RuntimeError) as error:
        # torch.OutOfMemoryError 是 RuntimeError 的子类; 其余 RuntimeError 原样抛出
        if not is_out_of_memory(error):
            raise
        record_out_of_memory(args.output, args.device, progress, error,
                             SETUP_STAGES[args.analysis_method], ITERATION_STAGES)
        return 2


def run(args, progress: dict) -> int:
    """运行选定方法的完整拓扑优化流程.

    Parameters
    ----------
    args : argparse.Namespace
        ``parse_args`` 的结果.
    progress : dict
        运行中就地更新的进度, 供内存或显存不足时定位失败点, 键见 ``soptx.core.record_out_of_memory``.

    Returns
    -------
    status : int
        正常结束时为 0.
    """
    device = args.device
    cuda = device == 'cuda'
    backend = args.backend
    bm.set_backend(backend)
    # 给 PyTorch 分配器设固定显存上限, 超出即报 OOM, 不让 WDDM 驱动静默落到系统内存;
    # 须在首次分配显存之前设置
    gpu_memory = limit_cuda_memory(device, args.gpu_memory_limit) if cuda else None
    # 主机内存上限按启动后新增的虚拟内存计, 超出即抛 MemoryError, 不换出到 swap; 依赖均已导入, 基线不含后续分配
    cpu_memory = (limit_cpu_memory(args.cpu_memory_limit)
                  if not cuda and args.cpu_memory_limit is not None else None)

    is_substructure = args.analysis_method == 'substructure'
    preconditioner = args.preconditioner

    # 验网格比例, 取子结构数并计算密度过滤半径.
    grid = args.grid
    n_sub = args.n_sub
    if grid[0] != 6 * grid[1] or grid[1] != grid[2]:
        raise ValueError('MBB 梁要求网格各方向单元数满足 6:1:1')
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    domain_size = tuple(grid[d] * h for d in range(3))
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h

    # 创建输出目录.
    # 排他创建目录, 保留已有计算结果.
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    # 同一处记录公共参数, 方法专属参数另行补充.
    config: dict[str, Any] = dict(
        method=f'substructure_{TRACE}' if is_substructure else 'fem_fa',
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        operator_level=OPERATOR_LEVEL, assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        backend=backend, device=device, gpu_memory=gpu_memory, cpu_memory=cpu_memory, max_iter=args.max_iter,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=NUM_THREADS, **blas_environment()))
    if backend == 'pytorch':
        import torch
        config.update(torch=dict(version=torch.__version__, cuda=torch.version.cuda,
                                 device_name=torch.cuda.get_device_name(device) if cuda else None))
    cg_options = dict(precond=None if preconditioner == 'none' else preconditioner,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned',
                      verify_residual=True)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)
    # FA 交给求解器的是显式的约束后矩阵, 在 K 的值数组上原地对称消元, 不另存一份值数组
    if is_substructure:
        config.update(trace=TRACE, n_sub=n_sub, n_fine=(args.n_fine,) * 3,
                      chunk_size=CHUNK_SIZE)
    else:
        config.update(fa_dirichlet='eliminate_inplace')
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    method_text = f'substructure/{TRACE}, n_sub={n_sub}, m={args.n_fine}' if is_substructure else 'fem/fa'
    print(f'[配置] {method_text}, {backend}/{device}, 线程 {NUM_THREADS}, 装配 {ASSEMBLY_METHOD}, '
          f'求解 {SOLVER} ({preconditioner}, tol {CG_TOLERANCE:g}, maxiter {CG_MAXITER}), '
          f'网格 hex {grid} = {n_cells} 单元 / {n_dofs} 自由度, '
          f'问题 {type(problem).__name__}, 输出 {output}', flush=True)
    if gpu_memory is not None:
        print(f'[显存上限] 空闲 {gpu_memory["free_gib"]:.2f} / 总量 {gpu_memory["total_gib"]:.2f} GiB, '
              f'PyTorch 分配器上限 {gpu_memory["limit_gib"]:.2f} GiB (固定), 超出即报 OOM', flush=True)
    if cpu_memory is not None:
        print(f'[内存上限] 系统可用 {cpu_memory["available_gib"]:.2f} GiB, 启动时虚拟内存 '
              f'{cpu_memory["base_virtual_gib"]:.2f} GiB, 新增上限 {cpu_memory["limit_gib"]:.2f} GiB, 超出即报错', flush=True)

    setup_stages = {}
    progress.update(phase='准备', setup_stages=setup_stages, stages=setup_stages, gpu_memory=gpu_memory,
                    cpu_memory=cpu_memory)
    with measure(setup_stages, '网格', label='准备', device=device):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=device)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    # 每轮一帧 VTU 写在 output/iterations/, 由 evolution.pvd 串成 ParaView 时间序列, 均保留在运行目录
    frames = IterationFrameWriter(output, mesh)

    with measure(setup_stages, '材料与插值', label='准备', device=device):
        # 两侧共用材料与 SIMP 插值.
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=device)
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        )
    if is_substructure:
        with measure(setup_stages, '子结构布局与校验', label='准备', device=device):
            assert n_sub is not None
            layout = StructuredSubstructureLayout(
                domain_size=domain_size,
                n_sub=n_sub, n_fine=(args.n_fine,) * 3, degree=SPACE_DEGREE,
                E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh)
        with measure(setup_stages, '分析器', label='准备', device=device):
            analyzer = SubstructureAnalyzer(
                layout=layout, trace=TRACE, chunk_size=CHUNK_SIZE,
                pde=problem, material=material, space_degree=SPACE_DEGREE,
                integration_order=INTEGRATION_ORDER, solve_method=SOLVER,
                solver_options=dict(cg_options),
                topopt_algorithm='density_based', interpolation_scheme=interpolation)
    else:
        with measure(setup_stages, '分析器', label='准备', device=device):
            analyzer = LagrangeFEMAnalyzer(
                disp_mesh=mesh, assembly_method=ASSEMBLY_METHOD,
                operator_level=OPERATOR_LEVEL,
                pde=problem, material=material, space_degree=SPACE_DEGREE,
                integration_order=INTEGRATION_ORDER, solve_method=SOLVER,
                solver_options=dict(cg_options),
                topopt_algorithm='density_based', interpolation_scheme=interpolation)
    if (int(mesh.number_of_cells()), int(analyzer.tensor_space.number_of_global_dofs())) != (n_cells, n_dofs):
        raise RuntimeError('网格单元数或位移自由度数与首行打印的不符')

    with measure(setup_stages, '过滤矩阵', label='准备', device=device):
        density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=radius,
                                density_location='element')

    with measure(setup_stages, '目标与约束', label='准备', device=device):
        objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
        constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC, diff_mode='manual')
        rho = bm.full((n_cells, ), VOLFRAC, dtype=bm.float64, device=device)
        volume_gradient = density_filter.filter_constraint_sensitivities(
            design_variable=rho, con_grad_rho=constraint.jac(density=rho))

    # 第 0 帧: 初始设计经过滤与实体保留单元处理后的物理密度, 便于查看固定单元; 在计时之前写出
    initial_frame = None
    if 'density' in VTU_FIELDS:
        initial = density_filter.filter_design_variable(
            design_variable=rho,
            physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=device))[:]
        initial_frame = frames.write(0, cell_data={'density': bm.to_numpy(initial)})
        del initial

    history = []
    recent = []
    converged = False
    run_started = perf_counter()
    # 循环只经由停止判断后的 break 退出 (最后一轮必有 iteration == max_iter), 写成 while True
    # 使循环体内赋值的 physical / state / compliance 在循环后确定已绑定
    iteration = 0
    previous_displacement = None
    while True:
        iteration += 1
        started = perf_counter()
        stages = {}
        progress.update(phase='迭代', iteration=iteration, stages=stages)
        # 首轮逐阶段打印, 便于定位大规模下出问题的阶段; 之后只打印汇总行
        label = '迭代 1' if iteration == 1 else None

        # 1. 密度过滤
        with measure(stages, '过滤', label, device=device):
            physical = density_filter.filter_design_variable(
                design_variable=rho,
                physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=device))[:]
            volume_fraction = float(bm.mean(physical))

        # 2. 分析: 共用装配、边界条件与求解接口, 分阶段记录成本.
        # 子结构的装配含局部缩聚与接口散加, 求解含全场位移恢复.
        with measure(stages, '装配', label, device=device):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label, device=device):
            load = analyzer.assemble_body_force_vector()
            if is_substructure:
                system_matrix, system_load = analyzer.apply_bc(stiffness, load)
            else:
                system_matrix, system_load = analyzer.apply_bc(stiffness, load, inplace=True)
            del stiffness, load
            displacement = analyzer.tensor_space.function()
        with measure(stages, '求解', label, device=device):
            # FEM 初值为全场位移; 子结构初值为接口未知量, 首轮由分析器取零.
            if is_substructure:
                warm_start = {'x0': previous_displacement} if previous_displacement is not None else {}
            else:
                x0 = analyzer.prescribed_solution if previous_displacement is None else previous_displacement
                warm_start = {'x0': x0}
            _, solver_info = analyzer.solve_system(system_matrix, system_load, displacement, **warm_start)
            del system_matrix, system_load
        state = {'displacement': displacement, 'solver': solver_info}
        previous_displacement = (analyzer.interface_displacement
                                 if isinstance(analyzer, SubstructureAnalyzer) else displacement[:])
        if not solver_info['converged']:
            print(f'[警告] 第 {iteration} 轮 CG 未收敛: {solver_info["niter"]} 步, '
                  f'相对残差 {solver_info["relres"]:.3e}', flush=True)

        # 3. 柔顺度与灵敏度
        with measure(stages, '灵敏度', label, device=device):
            compliance = float(objective.fun(density=physical, state=state))
            if not math.isfinite(compliance) or compliance <= 0:
                raise FloatingPointError('柔顺度必须为有限正数')
            dc = density_filter.filter_objective_sensitivities(
                design_variable=rho, obj_grad_rho=objective.jac(density=physical, state=state))
            if not bool(bm.all(bm.isfinite(dc))):
                raise FloatingPointError('柔顺度灵敏度非有限')

        analysis_seconds = perf_counter() - started

        # 4. 输出: 保存本轮分析所用设计的所选场; 帧与 PVD 由 IterationFrameWriter 原子写出.
        #    位移按节点优先编号 (3 n + c), 可直接重排为 (NN, 3)
        with measure(stages, '输出', label, device=device):
            cell_data = {'density': bm.to_numpy(physical)} if 'density' in VTU_FIELDS else {}
            point_data = ({'u': bm.to_numpy(displacement[:]).reshape(-1, 3)}
                          if 'displacement' in VTU_FIELDS else {})
            frame_file = frames.write(iteration, cell_data=cell_data, point_data=point_data)
            del cell_data, point_data

        # 5. 停止判断
        relative = None if not history else abs(compliance - history[-1]['compliance']) / compliance
        if relative is not None:
            recent.append(relative)
            recent = recent[-CONVERGENCE_WINDOW:]
        converged = len(recent) == CONVERGENCE_WINDOW and all(value < TOLERANCE for value in recent)
        # 逐轮记录关于 z 中面的不对称量, 追踪对称性是否以及何时被打破
        physical_grid = bm.reshape(physical, grid)
        symmetry_z = float(bm.max(bm.abs(physical_grid - bm.flip(physical_grid, axis=2))))
        # 设计变量关于 z 中面的不对称量
        design_grid = bm.reshape(rho, grid)
        design_symmetry_z = float(bm.max(bm.abs(design_grid - bm.flip(design_grid, axis=2))))
        assembly_seconds = stages['装配']['seconds'] + stages['边界条件']['seconds']
        solve_seconds = stages['求解']['seconds']
        peak_gib = max(stage['peak_gib'] for stage in stages.values())
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=volume_fraction,
                      relative_change=relative, symmetry_error_z=symmetry_z,
                      design_symmetry_error_z=design_symmetry_z,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      cg_iterations=solver_info['niter'],
                      cg_relative_residual=solver_info['relres'],
                      cg_converged=solver_info['converged'],
                      analysis_seconds=analysis_seconds, peak_rss_gib=peak_gib,
                      vtu_file=frame_file, stages=stages)
        if cuda:
            record['peak_gpu_gib'] = max(stage['gpu_peak_gib'] for stage in stages.values())
        history.append(record)
        # 停止时不再更新, 保留刚完成分析的密度, 避免最终柔顺度与密度错位
        stop = converged or iteration == args.max_iter

        # 6. OC 更新 (计入下一轮之前, 记在本轮记录中)
        if not stop:
            with measure(stages, 'OC', label, device=device):
                rho_new = OCOptimizer.update_design_variable(
                    design_variable=rho, objective_gradient=dc, constraint_gradient=volume_gradient,
                    constraint_function=lambda candidate: bm.sum(volume_gradient * candidate) - VOLFRAC,
                    **OC_OPTIONS,
                )[:]
                if (not bool(bm.all(bm.isfinite(rho_new)))
                        or float(bm.min(rho_new)) < 0 or float(bm.max(rho_new)) > 1):
                    raise FloatingPointError('OC 更新得到无效密度')
                if float(bm.sum(volume_gradient * rho_new)) > VOLFRAC + VOLUME_TOLERANCE:
                    raise RuntimeError('OC 乘子搜索未满足物理体积约束')
            rho = rho_new

        # 每轮一行; Rel. change 为停止判断所用的柔顺度相对变化, Time 为本轮全部阶段的用时
        record['iteration_seconds'] = perf_counter() - started
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding='utf-8')
        relative_text = f'{relative:.4e}' if relative is not None else '-'
        print(f'Iteration: {iteration}, Objective: {compliance:.4f}, Volfrac: {volume_fraction:.4f}, '
              f'Rel. change: {relative_text}' + f', CG: {solver_info["niter"]}'
              + f', Time: {record["iteration_seconds"]:.1f} sec', flush=True)
        if stop:
            break

    # 最终拓扑的对称性检验
    physical_grid = bm.reshape(physical, grid)
    symmetry_x = float(bm.max(bm.abs(physical_grid - bm.flip(physical_grid, axis=0))))
    symmetry_z = float(bm.max(bm.abs(physical_grid - bm.flip(physical_grid, axis=2))))

    # 写文件须在主机内存的 numpy 数组上进行, 经 bm.to_numpy 落到 numpy
    np.save(output / 'design_density_final.npy', bm.to_numpy(bm.reshape(rho, grid)))
    np.save(output / 'density_final.npy', bm.to_numpy(physical_grid))
    np.save(output / 'displacement_final.npy', bm.to_numpy(state['displacement'][:]))
    # 初始帧与最后一帧各复制一份到运行目录顶层, 便于直接读取: 最后一帧与最终 NPY 对应同一次
    # 分析, 初始帧便于查看固定单元; 复用已导出的所选场
    copies = {'result_final.vtu': history[-1]['vtu_file']}
    if initial_frame is not None:
        copies['result_initial.vtu'] = initial_frame
    for name, frame_file in copies.items():
        temporary = output / f'.{name}.tmp'
        try:
            shutil.copyfile(output / frame_file, temporary)
            temporary.replace(output / name)
        finally:
            temporary.unlink(missing_ok=True)
    summary = dict(history[-1], converged=converged,
                   termination='converged' if converged else 'max_iter',
                   n_cells=n_cells, n_dofs=n_dofs,
                   symmetry_error_x=symmetry_x, symmetry_error_z=symmetry_z,
                   setup_stages=setup_stages,
                   process_peak_rss_gib=max(max(stage['peak_gib'] for stage in setup_stages.values()),
                                            max(record['peak_rss_gib'] for record in history)),
                   total_seconds=perf_counter() - run_started)
    if cuda:
        summary['process_peak_gpu_gib'] = max(max(stage['gpu_peak_gib'] for stage in setup_stages.values()),
                                              max(record['peak_gpu_gib'] for record in history))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
