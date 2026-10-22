"""子结构 CPU 与 GPU 拓扑优化对照: 三维 MBB 梁 (``FullMBBBeamPoint3d``).

full_trace 与 linear_corner 各在 CPU 与 GPU 上运行一次, 仅 ``--trace`` 与 ``--device`` 不同.
参数与结果对照见 substructure_cpu_vs_gpu.md.
"""

import os
import sys

# 运行环境: OpenMP 与 BLAS 在加载时读取线程数, 须在导入 numpy/scipy/torch 之前写入环境变量才生效.
# 默认每个物理核一个线程; GPU 侧的 kernel 不受此设置影响, 主机侧运算 (回拷、写盘等) 同用此线程数.
# NUM_THREADS 为 --threads 的默认值; 该参数须在导入之前生效, 故先从 sys.argv 预取, 校验仍由 parse_args 完成
NUM_THREADS = 16


def _threads_from_argv(argv: list[str], default: int) -> int:
    """在导入 numpy/torch 之前从命令行预取 ``--threads``.

    Parameters
    ----------
    argv : 命令行参数 (不含程序名).
    default : 未给出或取值不是正整数时的线程数; 非法取值由 ``parse_args`` 报错.

    Returns
    -------
    threads : 写入环境变量的线程数.
    """
    for index, arg in enumerate(argv):
        if arg == '--threads' and index + 1 < len(argv):
            value = argv[index + 1]
        elif arg.startswith('--threads='):
            value = arg.split('=', 1)[1]
        else:
            continue
        return int(value) if value.isdigit() and int(value) > 0 else default
    return default


ENV_THREADS = _threads_from_argv(sys.argv[1:], NUM_THREADS)
for _name in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = str(ENV_THREADS)

import argparse
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np
import torch

from soptx.backend import backend_manager as bm
from soptx.core import blas_environment, limit_cuda_memory, measure, record_cuda_out_of_memory
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import SubstructureAnalyzer
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

# 计算后端
BACKEND = 'pytorch'

# 分析参数: 有限元离散与子结构; GRID 与 OPERATOR_LEVEL 分别为 --grid 与 --operator-level 的默认值.
# OPERATOR_LEVEL 为接口刚度的算子层级, 'fa' 散加为全局接口 CSR, 'ea' 只保留各子结构缩聚块 (实现待补);
# ASSEMBLY_METHOD 由 SubstructureAnalyzer 固定为 'fast', 此处只作记录
GRID = (210, 35, 35)
N_FINE = 5
CHUNK_SIZE = 256
OPERATOR_LEVEL = 'fa'
ASSEMBLY_METHOD = 'fast'
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2

# 分析参数: 线性系统求解; PRECONDITIONER 为 --preconditioner 的默认值.
# GMG 只支持 numpy 后端, 两侧统一用 CG, 预条件子取 none 或 jacobi
SOLVER = 'cg'
PRECONDITIONER = 'jacobi'
CG_TOLERANCE = 1.0e-6
CG_MAXITER = 20000

# 优化参数; MAX_ITER 为 --max-iter 的默认值
VOLFRAC = 0.12
FILTER_RADIUS_CELLS = 3.0
PENALTY = 3.0
OC_OPTIONS = dict(move_limit=0.2, damping_coef=0.5, initial_lambda=1.0e9,
                  bisection_tol=1.0e-3, design_variable_min=0.0)
TOLERANCE = 2.0e-4
CONVERGENCE_WINDOW = 5
VOLUME_TOLERANCE = 1.0e-6
MAX_ITER = 300

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
VTU_FIELDS = ('density', )

# 各阶段的执行顺序, 仅供显存不足时判定失败阶段: measure 只在阶段正常结束时写入记录,
# 顺序中第一个尚无记录的阶段即失败阶段
SETUP_STAGES = ('网格', '材料与插值', '子结构布局与校验', '分析器', '过滤矩阵', '目标与约束')
ITERATION_STAGES = ('过滤', '装配', '边界条件', '接口求解', '回代', '灵敏度', '输出', 'OC')

# GPU 显存上限 (GiB), 即 --gpu-memory-limit 的默认值: 固定值使各次运行可用显存相同; 总显存 15.89 GiB, 余下约 1.9 GiB 留给 CUDA 上下文、
# cuSPARSE 等不经 PyTorch 分配器的工作区与桌面. 启动时空闲显存不足此值即拒绝运行
CUDA_MEMORY_LIMIT_GIB = 14.0


def parse_args(argv=None):
    """解析接口迹空间、计算设备、接口算子层级、预条件子、网格、优化迭代上限、线程数与 GPU 显存上限并确定输出目录.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    args : argparse.Namespace
        含 ``trace``、``device``、``operator_level``、``preconditioner``、``grid``、``max_iter``、``threads``、
        ``gpu_memory_limit`` (CPU 下为 None) 与输出目录 ``output``.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--trace', choices=('full_trace', 'linear_corner'), required=True,
                        help='接口迹空间: full_trace 保留全部接口自由度, linear_corner 为角点线性插值')
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True,
                        help='计算设备: cpu 为主机 CPU, cuda 为 GPU')
    parser.add_argument('--operator-level', choices=('fa', 'ea'), default=OPERATOR_LEVEL,
                        help=f'接口刚度的算子层级: fa 装配全局接口 CSR, ea 只保留各子结构缩聚块 '
                             f'(实现待补) (默认 {OPERATOR_LEVEL})')
    parser.add_argument('--preconditioner', choices=('none', 'jacobi'), default=PRECONDITIONER,
                        help=f'CG 预条件子 (默认 {PRECONDITIONER})')
    parser.add_argument('--grid', type=int, nargs=3, default=GRID, metavar=('NX', 'NY', 'NZ'),
                        help=f'六面体网格各方向单元数, 须能被 N_FINE={N_FINE} 整除 (默认 {GRID})')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER,
                        help=f'优化迭代上限 (默认 {MAX_ITER})')
    parser.add_argument('--threads', type=int, default=NUM_THREADS,
                        help=f'CPU 线程数, 在导入前写入 OpenMP 与 BLAS 线程变量, torch 线程池随之取值 (默认 {NUM_THREADS})')
    parser.add_argument('--gpu-memory-limit', type=float, default=None, metavar='GIB',
                        help=f'GPU 显存上限 (GiB), 只用于 --device cuda (默认 {CUDA_MEMORY_LIMIT_GIB})')

    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda 需要可用的 CUDA 设备')
    if min(args.grid) < 1:
        parser.error(f'--grid 须为正整数, 得到 {args.grid}')
    # 子结构数由整数除法得到, 不整除时会静默截断, 须在生成目录名之前拒绝
    if any(n % N_FINE for n in args.grid):
        parser.error(f'--grid 各方向须能被 N_FINE={N_FINE} 整除, 得到 {args.grid}')
    if args.max_iter < 1:
        parser.error(f'--max-iter 须为正整数, 得到 {args.max_iter}')
    if args.threads < 1:
        parser.error(f'--threads 须为正整数, 得到 {args.threads}')
    # CPU 的内存上限由外层 systemd-run 的 cgroup 控制, 不归本参数管
    if args.gpu_memory_limit is not None:
        if args.device != 'cuda':
            parser.error('--gpu-memory-limit 只用于 --device cuda')
        if args.gpu_memory_limit <= 0:
            parser.error(f'--gpu-memory-limit 须为正数, 得到 {args.gpu_memory_limit}')
    elif args.device == 'cuda':
        args.gpu_memory_limit = CUDA_MEMORY_LIMIT_GIB
    # 线程变量已在导入时按命令行预取的值写入; 以缩写等预取不到的形式给出, 或经 main(argv) 传入时两者不符
    if args.threads != ENV_THREADS:
        parser.error(f'--threads {args.threads} 与导入时写入环境变量的线程数 {ENV_THREADS} 不符; '
                     '须在命令行以 --threads N 完整给出')
    args.grid = tuple(args.grid)
    size = 'x'.join(map(str, args.grid))
    args.output = (OUTPUT_DIR / f'{args.device}_{args.operator_level}_{args.trace}_hex_{size}_m{N_FINE}_{SOLVER}'
                   f'_{args.preconditioner}_t{args.threads}')
    # 已有输出时报错, 保留历史结果; 在任何计算之前拒绝
    if args.output.exists():
        parser.error(f'输出目录已存在, 不覆盖: {args.output}')
    return args


def summarize_stages(history: list[dict]) -> dict[str, dict]:
    """按阶段汇总各轮迭代的用时, 首轮与之后各轮分开统计.

    首轮含一次性开销 (如接口 CSR 符号模式构建、GPU 的冷启动), 不能代表每轮的稳态用时.

    Parameters
    ----------
    history : 各轮记录, 每条的 ``stages`` 为 ``{阶段名: {'seconds': ...}}``.

    Returns
    -------
    stage_seconds : 以阶段名为键, 值含 ``first`` (首轮用时, 首轮无该阶段时为 None)、``steady_mean``
        (第 2 轮起的均值, 无数据时为 None)、``total`` (各轮合计) 与 ``count`` (出现的轮数).
        OC 不在停止的那一轮执行, 故其 count 比轮数少 1.
    """
    names = list(dict.fromkeys(name for record in history for name in record['stages']))
    stage_seconds = {}
    for name in names:
        first = history[0]['stages'].get(name, {}).get('seconds')
        later = [record['stages'][name]['seconds'] for record in history[1:] if name in record['stages']]
        values = ([] if first is None else [first]) + later
        stage_seconds[name] = dict(first=first, steady_mean=sum(later) / len(later) if later else None,
                                   total=sum(values), count=len(values))
    return stage_seconds


def main(argv=None):
    """运行拓扑优化; 显存不足时记录失败点.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    status : int
        正常结束时为 0, 显存不足时为 2.
    """
    args = parse_args(argv)
    progress: dict[str, Any] = {}
    try:
        return run(args, progress)
    except torch.OutOfMemoryError as error:
        record_cuda_out_of_memory(args.output, args.device, progress, error, SETUP_STAGES, ITERATION_STAGES)
        return 2


def run(args, progress: dict) -> int:
    """在选定设备上运行选定接口迹空间的完整拓扑优化流程.

    Parameters
    ----------
    args : argparse.Namespace
        ``parse_args`` 的结果.
    progress : dict
        运行中就地更新的进度, 供显存不足时定位失败点, 键见 ``soptx.core.record_cuda_out_of_memory``.

    Returns
    -------
    status : int
        正常结束时为 0.
    """
    trace = args.trace
    device = args.device
    bm.set_backend(BACKEND)
    # PyTorch 的 CPU 算子线程池导入时按 OMP_NUM_THREADS 取值, 此处只核对, 实际取值记入 config.json.
    # 不再调用 torch.set_num_threads: 在 PyTorch 2.11 + MKL 2024.2 下, 导入后以大于 1 的值调用它
    # (即使与环境变量相同) 会使批量 lu_factor 返回无效主元, linalg.solve 随之报错
    if torch.get_num_threads() != args.threads:
        raise RuntimeError(f'torch 线程池为 {torch.get_num_threads()} 个线程, 与 --threads {args.threads} 不符')
    cuda = device == 'cuda'
    # 给 PyTorch 分配器设固定显存上限, 超出即报 OOM, 不让 WDDM 驱动静默落到系统内存;
    # 须在首次分配显存之前设置
    gpu_memory = limit_cuda_memory(device, args.gpu_memory_limit) if cuda else None
    precond = args.preconditioner

    grid = args.grid
    max_iter = args.max_iter
    n_sub = tuple(n // N_FINE for n in grid)
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    domain_size = tuple(grid[d] * h for d in range(3))
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h

    # 排他创建目录, 防止与 parse_args 检查之后新建的同名目录混在一起
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    # 同一处记录公共参数, 设备相关信息另行补充.
    config: dict[str, Any] = dict(
        method=f'substructure_{args.operator_level}_{trace}_{device}', trace=trace,
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        n_sub=n_sub, n_fine=(N_FINE,) * 3, chunk_size=CHUNK_SIZE,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        operator_level=args.operator_level, assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        backend=BACKEND, device=device, max_iter=max_iter,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=args.threads, torch_num_threads=torch.get_num_threads(),
                     **blas_environment()),
        torch=dict(version=torch.__version__, cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name(device) if cuda else None),
        gpu_memory=gpu_memory)
    # 相对容差以右端项范数为参照, 与 substructure_vs_fem 同口径
    cg_options = dict(precond=None if precond == 'none' else precond,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned',
                      verify_residual=True)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    # 首行在准备阶段之前打印; 单元数与 Q1 位移自由度数由网格剖分算出, 分析器建好后核对
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    device_text = f'{device} ({config["torch"]["device_name"]})' if cuda else device
    print(f'[配置] substructure/{trace}, 接口算子层级 {args.operator_level.upper()}, n_sub={n_sub}, m={N_FINE}, '
          f'{BACKEND}/{device_text}, '
          f'CPU 线程 {torch.get_num_threads()}, 求解 {SOLVER} ({precond}, tol {CG_TOLERANCE:g}, '
          f'maxiter {CG_MAXITER}), 网格 hex {grid} = {n_cells} 单元 / {n_dofs} 自由度, '
          f'问题 {type(problem).__name__}, 输出 {output}', flush=True)
    if gpu_memory is not None:
        print(f'[显存上限] 空闲 {gpu_memory["free_gib"]:.2f} / 总量 {gpu_memory["total_gib"]:.2f} GiB, '
              f'PyTorch 分配器上限 {gpu_memory["limit_gib"]:.2f} GiB (固定), '
              '超出即报 OOM', flush=True)

    # 准备阶段与对照文档 4.x 节的耗时表一一对应, 互不嵌套
    setup_stages = {}
    progress.update(phase='准备', setup_stages=setup_stages, stages=setup_stages, gpu_memory=gpu_memory)
    with measure(setup_stages, '网格', label='准备', device=device):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=device)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    # 每轮一帧 VTU 写在 output/iterations/, 由 evolution.pvd 串成 ParaView 时间序列, 均留在运行目录
    frames = IterationFrameWriter(output, mesh)

    with measure(setup_stages, '材料与插值', label='准备', device=device):
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=device)
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        )

    with measure(setup_stages, '子结构布局与校验', label='准备', device=device):
        layout = StructuredSubstructureLayout(
            domain_size=domain_size,
            n_sub=n_sub, n_fine=(N_FINE,) * 3, degree=SPACE_DEGREE,
            E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh)

    with measure(setup_stages, '分析器', label='准备', device=device):
        analyzer = SubstructureAnalyzer(
            layout=layout, trace=trace, chunk_size=CHUNK_SIZE,
            pde=problem, material=material, space_degree=SPACE_DEGREE,
            integration_order=INTEGRATION_ORDER, operator_level=args.operator_level, solve_method=SOLVER,
            solver_options=dict(cg_options),
            topopt_algorithm='density_based', interpolation_scheme=interpolation)
    if (int(mesh.number_of_cells()), int(analyzer.tensor_space.number_of_global_dofs())) != (n_cells, n_dofs):
        raise RuntimeError('网格单元数或位移自由度数与首行打印的不符')

    # 接口未知量数 (施加支承前) 只能在分析器建好后得到, 补记到 config.json 作核对值
    n_interface_dofs = int(analyzer.interface_space.n_global)
    config.update(n_interface_dofs=n_interface_dofs)
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[接口] {trace}: {n_interface_dofs} 个接口自由度 (约束前)', flush=True)

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
    previous_interface = None
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

        # 2. 分析: 装配含局部缩聚、迹降阶与接口散加; 接口求解与回代分两个阶段计量, 不沿用旧的"求解"
        #    阶段名 (其含回代), 免得与旧结果混淆.
        with measure(stages, '装配', label, device=device):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label, device=device):
            system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
            del stiffness
            displacement = analyzer.tensor_space.function()
        with measure(stages, '接口求解', label, device=device):
            # 初值为接口未知量: full_trace 为完整接口位移, linear_corner 为角点位移; 首轮由分析器取零
            warm_start = {'x0': previous_interface} if previous_interface is not None else {}
            solver_info = analyzer.solve_interface(system_matrix, system_load, **warm_start)
            del system_matrix, system_load
        with measure(stages, '回代', label, device=device):
            analyzer.recover_displacement(displacement)
        state = {'displacement': displacement, 'solver': solver_info}
        previous_interface = analyzer.interface_displacement
        # 未收敛的解会破坏两侧统一的精度口径, 直接终止; history.json 保留此前各轮
        if not solver_info['converged']:
            raise RuntimeError(f'第 {iteration} 轮 CG 未收敛: {solver_info["niter"]} 步, '
                               f'相对残差 {solver_info["relres"]:.3e}')

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

        # 4. 输出: 保存本轮分析所用设计的所选场, 经 bm.to_numpy 拷回主机; 帧与 PVD 由 IterationFrameWriter
        #    原子写出. 位移按节点优先编号 (3 n + c), 可直接重排为 (NN, 3)
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
        solve_seconds = stages['接口求解']['seconds'] + stages['回代']['seconds']
        peak_gib = max(stage['peak_gib'] for stage in stages.values())
        # 分析器的装配与边界条件分项用时, 留作细分阶段的依据
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=volume_fraction,
                      relative_change=relative, symmetry_error_z=symmetry_z,
                      design_symmetry_error_z=design_symmetry_z,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      cg_iterations=solver_info['niter'],
                      cg_relative_residual=solver_info['relres'],
                      cg_converged=solver_info['converged'],
                      analysis_seconds=analysis_seconds, peak_rss_gib=peak_gib,
                      vtu_file=frame_file, stages=stages,
                      assembly_breakdown=dict(analyzer.assembly_timings),
                      boundary_breakdown=dict(analyzer.boundary_timings))
        if cuda:
            record['peak_gpu_gib'] = max(stage['gpu_peak_gib'] for stage in stages.values())
        history.append(record)
        # 停止时不再更新, 保留刚完成分析的密度, 避免最终柔顺度与密度错位
        stop = converged or iteration == max_iter

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

        # 每轮一行; Rel. change 为停止判断所用的柔顺度相对变化, 另附装配、接口求解与回代三阶段的用时 (其余阶段
        # 见运行结束时的耗时汇总), Time 为本轮全部阶段的用时
        record['iteration_seconds'] = perf_counter() - started
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding='utf-8')
        relative_text = f'{relative:.4e}' if relative is not None else '-'
        print(f'Iteration: {iteration}, Objective: {compliance:.4f}, Volfrac: {volume_fraction:.4f}, '
              f'Rel. change: {relative_text}, CG: {solver_info["niter"]}, '
              f'装配 {stages["装配"]["seconds"]:.2f} s, 接口求解 {stages["接口求解"]["seconds"]:.2f} s, '
              f'回代 {stages["回代"]["seconds"]:.2f} s, '
              f'Time: {record["iteration_seconds"]:.1f} sec', flush=True)
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
    # 初始帧与最后一帧各另存一份到运行目录顶层, 便于直接打开: 最后一帧与最终 NPY 对应同一次分析,
    # 初始帧便于查看固定单元; 复用已导出的所选场
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
    # 接口矩阵的非零元数与 CSR 编号类型 (按 nnz 自动选 int32 或 int64), 符号模式在首轮装配时建立
    interface_pattern = analyzer.interface_space.pattern
    interface_index_dtype = str(interface_pattern.col.dtype).replace('torch.', '')
    summary = dict(history[-1], converged=converged,
                   termination='converged' if converged else 'max_iter',
                   n_cells=n_cells, n_dofs=n_dofs, n_interface_dofs=n_interface_dofs,
                   interface_nnz=interface_pattern.nnz, interface_index_dtype=interface_index_dtype,
                   symmetry_error_x=symmetry_x, symmetry_error_z=symmetry_z,
                   setup_stages=setup_stages, stage_seconds=summarize_stages(history),
                   process_peak_rss_gib=max(max(stage['peak_gib'] for stage in setup_stages.values()),
                                            max(record['peak_rss_gib'] for record in history)),
                   total_seconds=perf_counter() - run_started)
    if cuda:
        summary['process_peak_gpu_gib'] = max(max(stage['gpu_peak_gib'] for stage in setup_stages.values()),
                                              max(record['peak_gpu_gib'] for record in history))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')
    # 逐阶段耗时汇总, 首轮与第 2 轮起的稳态分开; 准备阶段已在开始时逐项打印
    for name, item in summary['stage_seconds'].items():
        first = '-' if item['first'] is None else f'{item["first"]:.3f} s'
        steady = '-' if item['steady_mean'] is None else f'{item["steady_mean"]:.3f} s'
        print(f'[耗时/{name}] 第 1 轮 {first}, 第 2 轮起均值 {steady}, 合计 {item["total"]:.2f} s '
              f'({item["count"]} 轮)', flush=True)
    print(f'[接口矩阵] nnz {interface_pattern.nnz}, CSR 编号类型 {interface_index_dtype}', flush=True)
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
