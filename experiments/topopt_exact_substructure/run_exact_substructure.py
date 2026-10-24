"""精确子结构 (Exact) 的完整拓扑优化流程: 三维 MBB 梁 (``FullMBBBeamPoint3d``).

边界位移表示 (``--trace``)、算子层级 (``--assemble-level``)、预条件子、计算后端与设备均由命令行给出; 各对照
(full_trace 与 linear_corner、CPU 与 GPU、子结构与全尺度有限元、PIML 与 Exact) 的 Exact 一侧共用本入口.
choices 列出规划中的全部取值, 尚未实现者在创建输出目录前拒绝, 见 IMPLEMENTED. 内存或显存不足时把失败点
写入输出目录的 failure.json.
"""

import os
import sys

# 运行环境: OpenMP 与 BLAS 在加载时读取线程数, 须在导入 numpy/scipy 之前写入环境变量才生效.
# NUM_THREADS 为 --threads 的默认值, 单线程; 该参数须在导入之前生效, 故先从 sys.argv 预取,
# 校验仍由 parse_args 完成
NUM_THREADS = 1
THREAD_VARIABLES = ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')


def _threads_from_argv(argv: list[str], default: int) -> int:
    """在导入 numpy 之前从命令行预取 ``--threads``.

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


# 只在直接运行时写入: 其他脚本在导入 numpy 之后才导入本模块取顶部常量, 此时写入既不生效, 又会改动其环境
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

from soptx.backend import backend_manager as bm
from soptx.core import (blas_environment, is_out_of_memory, limit_cpu_memory, limit_cuda_memory, measure,
                        record_out_of_memory)
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

# 计算后端: --backend 的默认值, 设备默认 cpu; cuda 须用 pytorch
BACKEND = 'numpy'
# GPU 显存上限 (GiB), 即 --gpu-memory-limit 的默认值: 固定值使各次运行可用显存相同, 与全尺度有限元入口
# run_full_scale_fem.py 取值一致; 启动时空闲显存不足此值即拒绝运行
CUDA_MEMORY_LIMIT_GIB = 14.0

# 分析参数: 有限元离散与子结构
GRID = (210, 35, 35)
N_FINE = 5
N_SUB = tuple(n // N_FINE for n in GRID)  # 由 GRID 与 N_FINE 推出, 不单独修改
CHUNK_SIZE = 256
ASSEMBLY_METHOD = 'fast'
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
# FA 对照共用局部装配与 Cholesky 恢复; 隐式 EA 另保留共享参考细单元作用
ASSEMBLE_LEVEL = 'fa'
# 接口先行: 以下参数的 choices 列出规划中的全部取值, 尚未实现者由 parse_args 在创建输出目录前拒绝
IMPLEMENTED = dict(trace=('full_trace', 'linear_corner'))

# 分析参数: 线性系统求解
# 本入口只按 CG 编写 (cg_options、未收敛即终止、CG 步数统计), 统一用迭代法;
# 预条件子由命令行 --preconditioner 选择
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

# 各阶段的执行顺序, 仅供内存或显存不足时判定失败阶段: measure 只在阶段正常结束时写入记录,
# 顺序中第一个尚无记录的阶段即失败阶段
SETUP_STAGES = ('网格', '材料与插值', '子结构布局与校验', '分析器', '过滤矩阵', '目标与约束')
ITERATION_STAGES = ('过滤', '装配', '边界条件', '求解', '恢复', '灵敏度', '输出', 'OC')


def parse_args(argv=None):
    """按子结构划分、局部缩聚、组装求解与优化、运行设置四组解析命令行参数并确定输出目录,
    其余设置均为模块顶部常量. 分组与 PIML 入口相同, 两侧只有局部缩聚一组的参数不同.

    未实现的迹空间及 schur_mode / assemble_level / 后端组合在此拒绝.

    grid、n_sub、n_fine 须满足 grid = n_sub * n_fine: 三者都给出时检查一致性, 缺一个时由另外两个推出.
    n_fine 无法由 grid 与 n_sub 推出时取 N_FINE, grid 无法由 n_sub 推出时取 GRID. cuda 须用 pytorch 后端,
    pytorch 后端只支持 none/jacobi 预条件子. 输出目录名为 ``<trace>_<schur_mode>_<assemble_level>_hex_...``, numpy/cpu 不带
    前缀, 其余组合加 ``<backend>_<device>_`` 前缀.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    args : argparse.Namespace
        子结构划分: 细网格 ``grid``、子结构数 ``n_sub``、细单元数 ``n_fine``、边界位移表示 ``trace``;
        局部缩聚: Schur 实现 ``schur_mode``;
        组装、求解与优化: 装配层级 ``assemble_level``、预条件子 ``preconditioner``、最大优化步数 ``max_iter``;
        运行设置: 计算后端 ``backend``、计算设备 ``device``、GPU 显存上限 ``gpu_memory_limit`` (cpu 下为 None)、
        CPU 内存上限 ``cpu_memory_limit`` (未给出或 cuda 下为 None)、线程数 ``threads``;
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
    substructure.add_argument('--trace', choices=('full_trace', 'linear_corner', 'bezier_cubic'), required=True,
                              help='边界位移表示: full_trace 保留全部边界自由度, linear_corner 为角点三线性插值, '
                                   'bezier_cubic 为三次 Bezier 插值 (实现待补)')
    condensation = parser.add_argument_group('局部缩聚')
    condensation.add_argument('--schur-mode', choices=('explicit', 'implicit'), default='explicit',
                              help='Schur 实现: explicit 形成缩聚矩阵, implicit 用内部直接求解作用 (默认 explicit); '
                                   '须与 --assemble-level 配对')
    solve = parser.add_argument_group('组装、求解与优化')
    solve.add_argument('--assemble-level', choices=('fa', 'ea'), default=ASSEMBLE_LEVEL,
                       help=f'装配层级: explicit/fa 为接口 CSR, implicit/fa 为局部稀疏刚度作用; implicit/ea 为细单元 EA; '
                            f'其余组合暂不支持 (默认 {ASSEMBLE_LEVEL})')
    solve.add_argument('--preconditioner', choices=('jacobi', 'gmg', 'none'), default='jacobi',
                       help='CG 预条件子, 默认 jacobi; gmg 由 SubstructureAnalyzer 提供, 仅支持 full_trace 与 numpy 后端')
    solve.add_argument('--max-iter', type=int, default=MAX_ITER,
                       help=f'最大优化步数, 默认 {MAX_ITER}')
    runtime = parser.add_argument_group('运行设置')
    runtime.add_argument('--backend', choices=('numpy', 'pytorch'), default=BACKEND,
                         help=f'计算后端, 默认 {BACKEND}; --device cuda 须用 pytorch')
    runtime.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                         help='计算设备, 默认 cpu')
    runtime.add_argument('--gpu-memory-limit', type=float, default=None, metavar='GIB',
                         help=f'GPU 显存上限 (GiB), 只用于 --device cuda (默认 {CUDA_MEMORY_LIMIT_GIB})')
    runtime.add_argument('--cpu-memory-limit', type=float, default=None, metavar='GIB',
                         help='CPU 内存上限 (GiB), 按启动后新增的虚拟内存计, 只用于 --device cpu; 不给则不设上限')
    runtime.add_argument('--threads', type=int, default=NUM_THREADS,
                         help=f'CPU 线程数, 在导入前写入 OpenMP 与 BLAS 线程变量 (默认 {NUM_THREADS})')
    args = parser.parse_args(argv)
    for name, implemented in IMPLEMENTED.items():
        value = getattr(args, name)
        if value not in implemented:
            parser.error(f'--{name.replace("_", "-")} {value} 尚未实现, 当前仅支持 {", ".join(implemented)}')
    if args.schur_mode == 'explicit' and args.assemble_level != 'fa':
        parser.error('explicit 当前只支持 --assemble-level fa')
    if args.schur_mode == 'implicit':
        if args.trace != 'full_trace':
            parser.error('implicit 只支持 full_trace')
        if args.assemble_level == 'ea' and (args.backend, args.device) != ('numpy', 'cpu'):
            parser.error('implicit/ea 当前只支持 numpy + cpu; GPU 请使用 implicit/fa')
        if args.preconditioner not in ('none', 'jacobi'):
            parser.error('implicit 当前只支持 none/jacobi 预条件子')
    # 角点接口的未知量不是细网格自由度, 分析器不支持 gmg; 在创建输出目录前拒绝
    if args.trace == 'linear_corner' and args.preconditioner == 'gmg':
        parser.error('--preconditioner gmg 只支持 --trace full_trace')
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
    if args.threads < 1:
        parser.error(f'--threads 须为正整数, 得到 {args.threads}')
    # 线程变量已在导入时按命令行预取的值写入; 以缩写等预取不到的形式给出、经 main(argv) 传入,
    # 或本模块被导入而非直接运行 (ENV_THREADS 为 None) 时两者不符
    if args.threads != ENV_THREADS:
        parser.error(f'--threads {args.threads} 与导入时写入环境变量的线程数 {ENV_THREADS} 不符; '
                     '须直接运行本脚本, 并在命令行以 --threads N 完整给出')
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
    # 目录名含迹空间、算子层级、预条件子与线程数, 不同配置的运行互不覆盖;
    # numpy/cpu 不带前缀, 其余组合加 <backend>_<device>_ 前缀
    prefix = '' if (args.backend, args.device) == ('numpy', 'cpu') else f'{args.backend}_{args.device}_'
    args.output = (OUTPUT_DIR.resolve() / f'{prefix}{args.trace}_{args.schur_mode}_{args.assemble_level}_hex_{size}_m{args.n_fine}'
                   f'_{SOLVER}_{args.preconditioner}_t{args.threads}')
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
    """运行选定接口迹空间的拓扑优化; 内存或显存不足时记录失败点.

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
        record_out_of_memory(args.output, args.device, progress, error, SETUP_STAGES, ITERATION_STAGES)
        return 2


def run(args, progress: dict) -> int:
    """运行选定接口迹空间的完整拓扑优化流程.

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
    trace = args.trace
    preconditioner = args.preconditioner
    torch = None
    if backend == 'pytorch':
        import torch
        # PyTorch 的 CPU 算子线程池导入时按 OMP_NUM_THREADS 取值, 此处只核对, 实际取值记入 config.json.
        # 不调用 torch.set_num_threads: 在 PyTorch 2.11 + MKL 2024.2 下, 导入后以大于 1 的值调用它
        # (即使与环境变量相同) 会使批量 lu_factor 返回无效主元, linalg.solve 随之报错
        if torch.get_num_threads() != args.threads:
            raise RuntimeError(f'torch 线程池为 {torch.get_num_threads()} 个线程, 与 --threads {args.threads} 不符')

    # 验网格比例, 取子结构划分并计算密度过滤半径; grid = n_sub * n_fine 已在 parse_args 中保证.
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

    # 记录导入前写入的线程变量, 供核对
    thread_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    config: dict[str, Any] = dict(
        method=f'exact_substructure_{trace}_{args.schur_mode}_{args.assemble_level}', trace=trace,
        schur_mode=args.schur_mode,
        assemble_level=args.assemble_level,
        schur_implementation=(
            ('local_fa_cholesky_torch_v1' if backend == 'pytorch' else 'local_fa_cholesky_v1')
            if trace == 'full_trace' and backend in ('numpy', 'pytorch')
            and args.assemble_level == 'fa' else
            'shared_reference_ea' if args.schur_mode == 'implicit' else 'legacy_explicit'),
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        n_sub=n_sub, n_fine=(n_fine,) * 3, chunk_size=CHUNK_SIZE,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        backend=backend, device=device, gpu_memory=gpu_memory, cpu_memory=cpu_memory, max_iter=args.max_iter,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=args.threads, environment=thread_environment, **blas_environment()))
    if torch is not None:
        config.update(torch=dict(version=torch.__version__, cuda=torch.version.cuda,
                                 num_threads=torch.get_num_threads(),
                                 device_name=torch.cuda.get_device_name(device) if cuda else None))
    cg_options = dict(precond=None if preconditioner == 'none' else preconditioner,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned',
                      verify_residual=True)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    print(f'[配置] exact/{trace}/{args.schur_mode}, 算子层级 {args.assemble_level}, n_sub={n_sub}, m={n_fine}, {backend}/{device}, '
          f'{args.threads} 线程, 装配 {ASSEMBLY_METHOD}, '
          f'求解 {SOLVER} ({preconditioner}, tol {CG_TOLERANCE:g}, maxiter {CG_MAXITER}), '
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

    # 准备阶段分为 网格 / 材料与插值 / 子结构布局与校验 / 分析器 / 过滤矩阵 / 目标与约束 六块, 互不嵌套,
    # 顺序见 SETUP_STAGES
    setup_stages = {}
    progress.update(phase='准备', setup_stages=setup_stages, stages=setup_stages, gpu_memory=gpu_memory,
                    cpu_memory=cpu_memory)
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
            n_sub=n_sub, n_fine=(n_fine,) * 3, degree=SPACE_DEGREE,
            E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh)

    with measure(setup_stages, '分析器', label='准备', device=device):
        analyzer = SubstructureAnalyzer(
            layout=layout, trace=trace, chunk_size=CHUNK_SIZE,
            pde=problem, material=material, space_degree=SPACE_DEGREE,
            integration_order=INTEGRATION_ORDER, operator_level=args.assemble_level,
            schur_mode=args.schur_mode, solve_method=SOLVER,
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

        # 2. 分析: 显式装配接口 CSR; 隐式准备局部稀疏块或 EA、内部分解与 Schur 对角.
        #    NumPy/PyTorch full_trace/fa 两侧复用局部因子与耦合块, linear_corner 使用恢复缓存;
        #    隐式 full_trace 复用内部分解, 各路径均与接口求解分开计量
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
                         global_csr='全局CSR及编排', fine_operator='共享参考EA及映射',
                         internal_factorization='内部Cholesky分解', schur_diagonal='Schur对角',
                         local_setup='局部模式与映射', local_blocks='局部稀疏块缓存',
                         schur_formation='Schur形成及对角', other='其他')
            print(f'[{label}/装配分项] ' + ', '.join(
                f'{names.get(k, k)} {v:.3f}s ({v / stages["装配"]["seconds"]:.1%})'
                for k, v in assembly_breakdown.items()
                if k != 'local_condensation_total'), flush=True)
            if 'local_condensation_total' in assembly_breakdown:
                condensation_total = assembly_breakdown['local_condensation_total']
                print(f'[{label}/局部缩聚总耗时] {condensation_total:.3f}s '
                      f'({condensation_total / stages["装配"]["seconds"]:.1%}, 上述三阶段之和)',
                      flush=True)
        if label is not None:
            print(f'[{label}/边界分项] ' + ', '.join(
                f'{k}={v}' if isinstance(v, bool) else f'{k}={v:.3f}s'
                for k, v in boundary_breakdown.items()), flush=True)
        with measure(stages, '求解', label, device=device):
            # 初值为接口未知量: full_trace 为完整接口位移, linear_corner 为角点位移; 首轮由分析器取零
            warm_start = {'x0': previous_interface} if previous_interface is not None else {}
            solver_info = analyzer.solve_interface(system_matrix, system_load, **warm_start)
            del system_matrix, system_load
        previous_interface = analyzer.interface_displacement
        # 未收敛的解会破坏各对照统一的精度口径, 直接终止, 不再恢复; history.json 保留此前各轮
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
                      schur_matvec_count=solver_info.get('schur_matvec_count'),
                      analysis_seconds=analysis_seconds, peak_rss_gib=peak_gib,
                      vtu_file=frame_file, stages=stages,
                      assembly_breakdown=assembly_breakdown, boundary_breakdown=boundary_breakdown)
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

        # 每轮一行; Rel. change 为停止判断所用的柔顺度相对变化, 另附装配、求解与恢复三阶段的用时 (其余阶段
        # 见运行结束时的耗时汇总), Time 为本轮全部阶段的用时
        # OC 在初始轮摘要之后执行, 此处刷新峰值以覆盖全部已测阶段.
        record['peak_rss_gib'] = max(stage['peak_gib'] for stage in stages.values())
        if cuda:
            record['peak_gpu_gib'] = max(stage['gpu_peak_gib'] for stage in stages.values())
        record['iteration_seconds'] = perf_counter() - started
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding='utf-8')
        relative_text = f'{relative:.4e}' if relative is not None else '-'
        print(f'Iteration: {iteration}, Objective: {compliance:.4f}, Volfrac: {volume_fraction:.4f}, '
              f'Rel. change: {relative_text}, CG: {solver_info["niter"]}, '
              f'装配 {stages["装配"]["seconds"]:.2f} s, 求解 {stages["求解"]["seconds"]:.2f} s, '
              f'恢复 {stages["恢复"]["seconds"]:.2f} s, '
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
    # 接口矩阵的非零元数与 CSR 编号类型 (按 nnz 自动选 int32 或 int64), 符号模式在首轮装配时建立
    # 隐式路径不能为统计而访问 pattern: 该属性会惰性创建全局接口 CSR 模式.
    interface_nnz = interface_index_dtype = None
    if args.schur_mode == 'explicit':
        interface_pattern = analyzer.interface_space.pattern
        interface_nnz = int(interface_pattern.nnz)
        interface_index_dtype = str(interface_pattern.col.dtype).replace('torch.', '')
    summary = dict(history[-1], converged=converged,
                   termination='converged' if converged else 'max_iter',
                   n_cells=n_cells, n_dofs=n_dofs, n_interface_dofs=n_interface_dofs,
                   interface_nnz=interface_nnz, interface_index_dtype=interface_index_dtype,
                   schur_mode=args.schur_mode,
                   schur_implementation=config['schur_implementation'],
                   interface_matrix_stored=args.schur_mode == 'explicit',
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
    if args.schur_mode == 'explicit':
        print(f'[接口矩阵] nnz {interface_nnz}, CSR 编号类型 {interface_index_dtype}', flush=True)
    else:
        storage = '局部稀疏刚度 FA' if args.assemble_level == 'fa' else '共享参考 EA'
        print(f'[接口算子] implicit + {storage}, 不存储接口 CSR', flush=True)
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
