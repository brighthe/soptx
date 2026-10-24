"""PIML 子结构的完整拓扑优化流程: 三维 MBB 梁 (``FullMBBBeamPoint3d``).

接口 (``--trace``)、预测路线 (``--route``) 与权重文件 (``--weights``) 由命令行显式给出; 当前库层只实现
linear_corner 接口下的 shape 路线, 两个选项各只有一个取值. 精确子结构 (Exact) 一侧由
``experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py`` 运行, 两个入口的
模型、离散、求解与优化常量相同. 参数与结果对照见 substructure_linear_corner_piml_shape_vs_exact.md.
"""
import os
import sys

# 运行环境: OpenMP 与 BLAS 在加载时读取线程数, 须在导入 numpy/scipy/torch 之前写入环境变量才生效; torch 同样
# 读取 OMP_NUM_THREADS. NUM_THREADS 为 --threads 的默认值, 单线程, 与 Exact 入口相同; 该参数须在导入之前
# 生效, 故先从 sys.argv 预取, 校验仍由 parse_args 完成
NUM_THREADS = 1
THREAD_VARIABLES = ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')


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


# 只在直接运行时写入, 被导入时不改动调用方的环境
ENV_THREADS = _threads_from_argv(sys.argv[1:], NUM_THREADS) if __name__ == '__main__' else None
if ENV_THREADS is not None:
    for _name in THREAD_VARIABLES:
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
from soptx.core import blas_environment, limit_cpu_memory, limit_cuda_memory, measure
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import SubstructureAnalyzer
from soptx.fem.substructure import (
    IndependentPredictionDecoder, StructuredSubstructureLayout, build_modulus_substructures,
)
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.fem.substructure.piml_corner import PIMLShapeCornerCondensation
from soptx.ml.substructure.independent_checkpoints import (
    decoder_metadata_matches, load_independent_network, load_independent_provider_metadata,
)
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

# 计算后端: --backend 与 --device 的默认值; cuda 须用 pytorch, 网络推理与缩聚同在该设备
BACKEND = 'numpy'
DEVICE = 'cpu'
# GPU 显存上限 (GiB), 即 --gpu-memory-limit 的默认值, 与 Exact 入口相同; 启动时空闲显存不足此值即拒绝运行
CUDA_MEMORY_LIMIT_GIB = 14.0

# 分析参数: 有限元离散与子结构
GRID = (210, 35, 35)
N_FINE = 5
N_SUB = tuple(n // N_FINE for n in GRID)  # 由 GRID 与 N_FINE 推出, 不单独修改
CHUNK_SIZE = 256
ASSEMBLY_METHOD = 'fast'
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
# 库层已实现的接口与预测路线, 即 --trace 与 --route 的可选值; 扩展时须同时提供对应的角点缩聚器
TRACES = ('linear_corner',)
ROUTES = ('shape',)
# 接口先行: 以下参数的 choices 列出规划中的全部取值, 尚未实现者由 parse_args 在创建输出目录前拒绝.
# --assemble-level 即 SubstructureAnalyzer 的 operator_level: fa 装配全局角点 CSR, ea 只保留各子结构缩聚块
ASSEMBLE_LEVEL = 'fa'
IMPLEMENTED = dict(assemble_level=('fa',))

# 分析参数: 线性系统求解
# 本入口只按 CG 编写 (cg_options、未收敛即终止、CG 步数统计); 预条件子默认 Jacobi, 与 Exact 入口的 linear_corner 运行相同;
# gmg 只支持 full_trace, 不列入可选值
SOLVER = 'cg'
PRECONDITIONER = 'jacobi'
CG_TOLERANCE = 1.0e-6
CG_MAXITER = 20000

# 分析参数: PIML 预测
# 权重的 cell_size 是子结构边长 N_FINE * ELEMENT_SIZE, 不是细单元边长; 权重文件无默认值, 须以 --weights
# 显式给出, 避免重新训练后静默沿用旧权重
# 产物根目录固定为脚本旁的 outputs/, 与其他实验入口相同
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
UNIFORM_THRESHOLD = None  # 子结构内 max - mean 模量差小于该值时改用实体形函数; None 不判别
if UNIFORM_THRESHOLD is not None and not UNIFORM_THRESHOLD > 0:
    raise ValueError('UNIFORM_THRESHOLD 须为正数或 None')

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

# 输出设置 (OUTPUT_DIR 见上方 PIML 参数处)
VTU_FIELDS = ('density',)

def parse_args(argv=None):
    """按子结构划分、局部缩聚、组装求解与优化、运行设置四组解析命令行参数并确定输出目录,
    其余设置均为模块顶部常量. 分组与 Exact 入口相同, 两侧只有局部缩聚一组的参数不同.

    trace、route、weights 均为必填; 权重文件记录的接口与路线由 ``main`` 载入时与之核对. assemble_level 中尚未
    实现的取值 (见 IMPLEMENTED) 在此拒绝; cuda 须用 pytorch 后端.
    grid、n_sub、n_fine 须满足 grid = n_sub * n_fine: 三者都给出时检查一致性, 缺一个时由另外两个推出.
    n_fine 无法由 grid 与 n_sub 推出时取 N_FINE, grid 无法由 n_sub 推出时取 GRID.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    args : argparse.Namespace
        子结构划分: 细网格 ``grid``、子结构数 ``n_sub``、细单元数 ``n_fine``、接口 ``trace``;
        局部缩聚: 预测路线 ``route``、权重文件 ``weights``;
        组装、求解与优化: 算子层级 ``assemble_level``、预条件子 ``preconditioner``、最大优化步数 ``max_iter``;
        运行设置: 后端 ``backend``、设备 ``device``、GPU 显存上限 ``gpu_memory_limit`` (cpu 下为 None)、CPU 内存上限
        ``cpu_memory_limit`` (未给出时为 None)、线程数 ``threads``;
        以及输出目录 ``output``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    substructure = parser.add_argument_group('子结构划分')
    substructure.add_argument('--grid', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'),
                              help=f'细网格各方向单元数, 默认 {" ".join(map(str, GRID))}; 须满足 grid = n_sub * n_fine')
    substructure.add_argument('--n-sub', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'),
                              help=f'各方向子结构数, 默认 {" ".join(map(str, N_SUB))} (= GRID // N_FINE), '
                                   '缺省时由 grid 与 n_fine 推出')
    substructure.add_argument('--n-fine', type=int, default=None,
                              help='每个子结构各方向的细单元数 m; 缺省时由 grid 与 n_sub 推出, '
                                   f'两者不全时取 {N_FINE}')
    substructure.add_argument('--trace', choices=TRACES, required=True,
                              help='子结构边界位移表示, 当前仅支持 linear_corner (角点三线性插值)')
    condensation = parser.add_argument_group('局部缩聚')
    condensation.add_argument('--route', choices=ROUTES, required=True,
                              help='预测路线, 当前仅支持 shape (网络预测内部形函数, 按变分式构造角点刚度)')
    condensation.add_argument('--weights', type=Path, required=True,
                              help='训练得到的权重文件 (如 shape_best.pt), 无默认值; 须与 --trace、--route 一致')
    solve = parser.add_argument_group('组装、求解与优化')
    solve.add_argument('--assemble-level', choices=('fa', 'ea'), default=ASSEMBLE_LEVEL,
                       help='角点刚度的算子层级: fa 装配全局 CSR, ea 只保留各子结构缩聚块 (实现待补) '
                            f'(默认 {ASSEMBLE_LEVEL})')
    solve.add_argument('--preconditioner', choices=('none', 'jacobi'), default=PRECONDITIONER,
                       help=f'CG 预条件子 (默认 {PRECONDITIONER})')
    solve.add_argument('--max-iter', type=int, default=MAX_ITER,
                       help=f'最大优化步数, 默认 {MAX_ITER}')
    runtime = parser.add_argument_group('运行设置')
    runtime.add_argument('--backend', choices=('numpy', 'pytorch'), default=BACKEND,
                         help=f'计算后端, 默认 {BACKEND}; --device cuda 须用 pytorch')
    runtime.add_argument('--device', choices=('cpu', 'cuda'), default=DEVICE,
                         help=f'计算设备, 默认 {DEVICE}; 网络推理与缩聚同在该设备')
    runtime.add_argument('--gpu-memory-limit', type=float, default=None, metavar='GIB',
                         help=f'GPU 显存上限 (GiB), 只用于 --device cuda (默认 {CUDA_MEMORY_LIMIT_GIB})')
    runtime.add_argument('--cpu-memory-limit', type=float, default=None, metavar='GIB',
                         help='CPU 内存上限 (GiB), 按启动后新增的虚拟内存计, 只用于 --device cpu; 不给则不设上限')
    runtime.add_argument('--threads', type=int, default=NUM_THREADS,
                         help=f'CPU 线程数, 在导入前写入 OpenMP 与 BLAS 线程变量, torch 推理同用 (默认 {NUM_THREADS})')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error(f'--threads 须为正整数, 得到 {args.threads}')
    # 线程数须与导入前写入环境变量的值一致; 以 --threads=N 之外的写法给出非法值, 或本模块被导入而非直接运行
    # (ENV_THREADS 为 None) 时两者不符
    if args.threads != ENV_THREADS:
        parser.error(f'--threads {args.threads} 与导入时写入环境变量的线程数 {ENV_THREADS} 不符; '
                     '须直接运行本脚本, 并在命令行以 --threads N 完整给出')
    if not args.weights.is_file():
        parser.error(f'缺少权重: {args.weights}')
    for name, implemented in IMPLEMENTED.items():
        value = getattr(args, name)
        if value not in implemented:
            parser.error(f'--{name.replace("_", "-")} {value} 尚未实现, 当前仅支持 {", ".join(implemented)}')
    if args.device == 'cuda':
        if args.backend != 'pytorch':
            parser.error('--device cuda 须配 --backend pytorch')
        if not torch.cuda.is_available():
            parser.error('--device cuda 需要可用的 CUDA 设备')
        if args.gpu_memory_limit is None:
            args.gpu_memory_limit = CUDA_MEMORY_LIMIT_GIB
        elif args.gpu_memory_limit <= 0:
            parser.error(f'--gpu-memory-limit 须为正数, 得到 {args.gpu_memory_limit}')
    elif args.gpu_memory_limit is not None:
        parser.error('--gpu-memory-limit 只用于 --device cuda')
    if args.cpu_memory_limit is not None:
        if args.device != 'cpu':
            parser.error('--cpu-memory-limit 只用于 --device cpu')
        if args.cpu_memory_limit <= 0:
            parser.error(f'--cpu-memory-limit 须为正数, 得到 {args.cpu_memory_limit}')
    if args.max_iter < 1:
        parser.error('--max-iter 须为正整数')
    for option, value in (('--grid', args.grid), ('--n-sub', args.n_sub)):
        if value is not None and min(value) < 1:
            parser.error(f'{option} 各方向须为正整数')

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
    # numpy/cpu 不带前缀, 其余组合加 <backend>_<device>_ 前缀, 与 Exact 入口相同
    prefix = '' if (args.backend, args.device) == ('numpy', 'cpu') else f'{args.backend}_{args.device}_'
    args.output = OUTPUT_DIR.resolve() / (f'{prefix}piml_{args.route}_{args.trace}_hex_{size}_m{args.n_fine}'
                                          f'_{SOLVER}_{args.preconditioner}_t{args.threads}')
    return args


def main(argv=None):
    """运行选定接口与预测路线的 PIML 子结构完整拓扑优化流程.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    status : int
        正常结束时为 0.
    """
    args = parse_args(argv)
    backend, device = args.backend, args.device
    bm.set_backend(backend)
    cuda = device == 'cuda'
    # 给 PyTorch 分配器设固定显存上限, 超出即报 OOM, 不让 WDDM 驱动静默落到系统内存; 须在首次分配显存之前设置
    gpu_memory = limit_cuda_memory(device, args.gpu_memory_limit) if cuda else None
    trace, route = args.trace, args.route
    # 主机内存上限按启动后新增的虚拟内存计, 超出即抛 MemoryError, 不换出到 swap; 依赖均已导入, 基线不含后续分配
    cpu_memory = limit_cpu_memory(args.cpu_memory_limit) if args.cpu_memory_limit is not None else None

    # 验网格比例并计算密度过滤半径; grid = n_sub * n_fine 已在 parse_args 中保证.
    grid, n_sub, n_fine = args.grid, args.n_sub, args.n_fine
    if grid[0] != 6 * grid[1] or grid[1] != grid[2]:
        raise ValueError('MBB 梁要求网格各方向单元数满足 6:1:1')
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    domain_size = tuple(grid[d] * h for d in range(3))
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h

    # 创建输出目录.
    # 排他创建目录, 已有输出时报错, 保留历史结果.
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    # 记录导入前写入的线程变量与 torch 实际线程数, 供核对
    thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    config: dict[str, Any] = dict(
        method=f'piml_{route}_{trace}', trace=trace, route=route,
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        n_sub=n_sub, n_fine=(n_fine,) * 3, chunk_size=CHUNK_SIZE,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        assemble_level=args.assemble_level, backend=backend, device=device,
        gpu_memory=gpu_memory, cpu_memory=cpu_memory, max_iter=args.max_iter,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=args.threads, environment=thread_environment,
                     torch_threads=torch.get_num_threads(), **blas_environment()),
        torch=dict(version=torch.__version__, cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name(device) if cuda else None))
    cg_options = dict(precond=args.preconditioner,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned',
                      verify_residual=True)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)

    # 从权重恢复与之一致的标签提供器, 用于契约核对与实体形函数; 再按该契约加载网络 (载入时核对文件记录的路线).
    # pivot 编号取权重记录的一组: 立方体子结构的刚体基各行范数相同, 现选结果取决于舍入, 随后端变化.
    # 刚体基在 pytorch 后端下与权重记录差在舍入量级, 故按 decoder_metadata_matches 核对 (编号逐位, 刚体基按
    # roundtrip 容差); 网络按权重自带的记录载入, 只核对权重文件自洽.
    # 网络在 CPU 上载入后移到计算设备. 载入计为准备阶段 网络, 其常驻增量 after_gib - (peak_gib - net_gib)
    # 即网络带来的内存开销
    setup_stages = {}
    with measure(setup_stages, '网络', label='准备', device=device):
        saved = load_independent_provider_metadata(args.weights, route=route)
        provider = IndependentTargetProvider(
            cell_size=tuple(saved['cell_size']), n_fine=tuple(saved['n_fine']),
            nu=saved['poisson_ratio'], trace_kind=saved['trace'], hypothesis=saved.get('material_hypothesis'),
            pivot_indices=saved['pivot_indices'])
        metadata = provider.metadata()
        if not decoder_metadata_matches(saved, metadata):
            raise ValueError('恢复的子结构配置或独立分量编号与权重不一致')
        network, source = load_independent_network(args.weights, saved, route=route)
        network.requires_grad_(False)
        network.to(device)
    if saved['trace'] != trace:
        raise ValueError(f'权重的接口 {saved["trace"]} 与 --trace {trace} 不符')
    if tuple(metadata['n_fine']) != (n_fine,) * 3 or float(metadata['poisson_ratio']) != float(NU):
        raise ValueError(f'权重的局部配置 (n_fine={metadata["n_fine"]}, nu={metadata["poisson_ratio"]}) '
                         f'与本实验 (n_fine={n_fine}, nu={NU}) 不符')
    if tuple(float(v) for v in metadata['cell_size']) != (n_fine * h,) * 3:
        raise ValueError(f'权重的子结构边长 {metadata["cell_size"]} 与本实验 m*h={n_fine * h} 不符')
    config.update(
        inference=dict(backend='pytorch', device=device, dtype='float64',
                       torch_threads=torch.get_num_threads()),
        checkpoint=source, network_class=type(network).__name__,
        n_parameters=int(sum(p.numel() for p in network.parameters())),
        uniform_threshold=UNIFORM_THRESHOLD, input_quantity='normalized_young_modulus',
    )
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[PIML] 接口 {trace}, 路线 {route}, 权重 {args.weights}, 均匀判别阈值 {UNIFORM_THRESHOLD}', flush=True)

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    print(f'[配置] piml/{route}/{trace}, n_sub={n_sub}, m={n_fine}, {backend}/{device}, 算子层级 {args.assemble_level}, '
          f'{args.threads} 线程 (torch {torch.get_num_threads()}), 装配 {ASSEMBLY_METHOD}, '
          f'求解 {SOLVER} ({args.preconditioner}, tol {CG_TOLERANCE:g}, maxiter {CG_MAXITER}), '
          f'网格 hex {grid} = {n_cells} 单元 / {n_dofs} 自由度, '
          f'问题 {type(problem).__name__}, 输出 {output}', flush=True)
    print('[线程环境] ' + ', '.join(
        f'{name}={value if value is not None else "未设置"}'
        for name, value in thread_environment.items()), flush=True)
    if gpu_memory is not None:
        print(f'[显存上限] 空闲 {gpu_memory["free_gib"]:.2f} / 总量 {gpu_memory["total_gib"]:.2f} GiB, '
              f'PyTorch 分配器上限 {gpu_memory["limit_gib"]:.2f} GiB (固定), 超出即报 OOM', flush=True)
    if cpu_memory is not None:
        print(f'[内存上限] 系统可用 {cpu_memory["available_gib"]:.2f} GiB, 启动时虚拟内存 '
              f'{cpu_memory["base_virtual_gib"]:.2f} GiB, 新增上限 {cpu_memory["limit_gib"]:.2f} GiB, 超出即报错', flush=True)

    # 准备阶段分为 网络 (见上) / 网格 / 分析器 / 过滤矩阵 / 目标与约束 五块, 互不嵌套; 后四块与 Exact 入口相同
    with measure(setup_stages, '网格', label='准备', device=device):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=device)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
        layout = StructuredSubstructureLayout(
            domain_size=domain_size,
            n_sub=n_sub, n_fine=(n_fine,) * 3, degree=SPACE_DEGREE,
            E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh)
    # 每轮一帧 VTU 写在 output/iterations/, 由 evolution.pvd 串成 ParaView 时间序列, 均留在运行目录
    frames = IterationFrameWriter(output, mesh)

    with measure(setup_stages, '分析器', label='准备', device=device):
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=device)
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        )
        # 由分析器同款参考子结构构造解码器, pivot 编号取权重记录的一组 (见上), 注入形函数预测缩聚
        # (route='shape' 是当前唯一取值)
        prototype, _, _ = build_modulus_substructures(layout, integration_order=INTEGRATION_ORDER)
        decoder = IndependentPredictionDecoder(prototype, trace_kind=trace, pivot_indices=saved['pivot_indices'])
        solid_shape = None
        if UNIFORM_THRESHOLD is not None:
            solid_shape = provider.exact_matrices(np.ones((1, int(prototype.n_cells))))['shape'][0]
        condensation = PIMLShapeCornerCondensation(
            decoder, network, provider_metadata=provider.metadata(),
            uniform_threshold=UNIFORM_THRESHOLD, solid_shape=solid_shape)
        analyzer = SubstructureAnalyzer(
            layout=layout, trace=trace, chunk_size=CHUNK_SIZE, operator_level=args.assemble_level,
            pde=problem, material=material, space_degree=SPACE_DEGREE,
            integration_order=INTEGRATION_ORDER, solve_method=SOLVER,
            solver_options=dict(cg_options),
            topopt_algorithm='density_based', interpolation_scheme=interpolation,
            corner_condensation=condensation)
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
        # 首轮逐阶段打印, 便于定位大规模下出问题的阶段; 之后只打印汇总行
        label = '迭代 1' if iteration == 1 else None

        # 1. 密度过滤
        with measure(stages, '过滤', label, device=device):
            physical = density_filter.filter_design_variable(
                design_variable=rho,
                physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=device))[:]
            volume_fraction = float(bm.mean(physical))

        # 2. 分析: 装配含局部缩聚与接口散加 (在此缓存恢复矩阵 N_i, 与 Exact 入口相同), 求解只解角点系统,
        #    恢复由缓存的 N_i 回代全场位移; 求解与恢复分开计量, 与 Exact 入口的阶段划分一致.
        with measure(stages, '装配', label, device=device):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label, device=device):
            detail_started = perf_counter()
            body_force = analyzer.assemble_body_force_vector()
            body_force_seconds = perf_counter() - detail_started
            system_matrix, system_load = analyzer.apply_bc(stiffness, body_force)
            del body_force
            detail_started = perf_counter()
            del stiffness
            displacement = analyzer.tensor_space.function()
            allocate_seconds = perf_counter() - detail_started
        assembly_breakdown = dict(analyzer.assembly_timings)
        if assembly_breakdown:
            assembly_breakdown['other'] = max(
                0.0, stages['装配']['seconds'] - sum(
                    value for key, value in assembly_breakdown.items()
                    if key != 'local_condensation_total'))
        boundary_breakdown = dict(analyzer.boundary_timings,
                                  body_force=body_force_seconds, allocate_state=allocate_seconds)
        # 分项只在第 1 轮打印 (与阶段行同一门控), 逐轮数值已记入 history.json
        if assembly_breakdown and label is not None:
            names = dict(local_stiffness='局部刚度生成', block_projection='分块与角点投影',
                         local_solve='内部线性系统求解', condensation_assembly='缩聚刚度合成',
                         global_csr='全局CSR及编排', other='其他',
                         inference='网络推理与补全', basis_assembly='基函数拼接与变分式', diagnostics='预测诊断')
            print(f'[{label}/装配分项] ' + ', '.join(
                f'{names[k]} {v:.3f}s ({v / stages["装配"]["seconds"]:.1%})'
                for k, v in assembly_breakdown.items()
                if k != 'local_condensation_total'), flush=True)
            condensation_total = assembly_breakdown['local_condensation_total']
            print(f'[{label}/局部缩聚总耗时] {condensation_total:.3f}s '
                  f'({condensation_total / stages["装配"]["seconds"]:.1%}, 上述三阶段之和)',
                  flush=True)
        if label is not None:
            print(f'[{label}/边界分项] ' + ', '.join(
                f'{k}={v}' if isinstance(v, bool) else f'{k}={v:.3f}s'
                for k, v in boundary_breakdown.items()), flush=True)
        with measure(stages, '求解', label, device=device):
            # 初值为上一轮的角点位移; 首轮由分析器取零
            warm_start = {'x0': previous_interface} if previous_interface is not None else {}
            solver_info = analyzer.solve_interface(system_matrix, system_load, **warm_start)
            del system_matrix, system_load
        previous_interface = analyzer.interface_displacement
        # 未收敛的解会破坏与 Exact 统一的精度口径, 直接终止, 不再恢复; history.json 保留此前各轮
        if not solver_info['converged']:
            raise RuntimeError(f'第 {iteration} 轮 CG 未收敛: {solver_info["niter"]} 步, '
                               f'相对残差 {solver_info["relres"]:.3e}')
        with measure(stages, '恢复', label, device=device):
            analyzer.recover_displacement(displacement)
        state = {'displacement': displacement, 'solver': solver_info}

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
        recovery_seconds = stages['恢复']['seconds']
        peak_gib = max(stage['peak_gib'] for stage in stages.values())
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=volume_fraction,
                      relative_change=relative, symmetry_error_z=symmetry_z,
                      design_symmetry_error_z=design_symmetry_z,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      recovery_seconds=recovery_seconds,
                      cg_iterations=solver_info['niter'],
                      cg_relative_residual=solver_info['relres'],
                      cg_converged=solver_info['converged'],
                      analysis_seconds=analysis_seconds, peak_rss_gib=peak_gib,
                      vtu_file=frame_file, stages=stages,
                      assembly_breakdown=assembly_breakdown, boundary_breakdown=boundary_breakdown)
        if cuda:
            record['peak_gpu_gib'] = max(stage['gpu_peak_gib'] for stage in stages.values())
        # 另记本轮预测数、均匀替换数与三项诊断比值的分布
        if analyzer.condensation_report:
            record['condensation_report'] = analyzer.condensation_report
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
    # 初始帧与最后一帧各复制一份到运行目录顶层, 不必翻 iterations/: 最后一帧与最终 NPY 对应同一次
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
                   n_cells=n_cells, n_dofs=n_dofs, n_interface_dofs=n_interface_dofs,
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
