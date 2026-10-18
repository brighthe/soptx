"""传统有限元 CPU 与 GPU 拓扑优化对照: 三维 MBB 梁 (``FullMBBBeamPoint3d``).
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
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import math

import numpy as np
import torch

from soptx.backend import backend_manager as bm
from soptx.core import blas_environment, measure
from soptx.mesh import HexahedronMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
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

# 分析参数: 有限元离散; GRID 与 OPERATOR_LEVEL 分别为 --grid 与 --level 的默认值
GRID = (390, 65, 65)
# GRID = (190, 32, 32)
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
ASSEMBLY_METHOD = 'fast'
OPERATOR_LEVEL = 'ea'

# 分析参数: 线性系统求解; PRECONDITIONER 为 --preconditioner 的默认值
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


def parse_args(argv=None):
    """解析计算设备、网格、刚度算子层级、预条件子、优化迭代上限与线程数, 其余设置均为模块顶部常量.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--preconditioner', choices=('none', 'jacobi'), default=PRECONDITIONER,
                        help=f'CG 预条件子 (默认 {PRECONDITIONER})')
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True,
                        help='计算设备: cpu 为主机 CPU, cuda 为 GPU')
    parser.add_argument('--level', choices=('ea', 'fa'), default=OPERATOR_LEVEL,
                        help=f'刚度算子层级: ea 不形成全局矩阵, fa 装配全局 CSR 矩阵 (默认 {OPERATOR_LEVEL})')
    parser.add_argument('--grid', type=int, nargs=3, default=GRID, metavar=('NX', 'NY', 'NZ'),
                        help=f'六面体网格各方向单元数 (默认 {GRID})')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER,
                        help=f'优化迭代上限 (默认 {MAX_ITER})')
    parser.add_argument('--threads', type=int, default=NUM_THREADS,
                        help=f'CPU 线程数, 写入 OpenMP 与 BLAS 线程变量并设置 torch 线程池 (默认 {NUM_THREADS})')

    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda 需要可用的 CUDA 设备')
    if min(args.grid) < 1:
        parser.error(f'--grid 须为正整数, 得到 {args.grid}')
    if args.max_iter < 1:
        parser.error(f'--max-iter 须为正整数, 得到 {args.max_iter}')
    if args.threads < 1:
        parser.error(f'--threads 须为正整数, 得到 {args.threads}')
    # 线程变量已在导入时按命令行预取的值写入; 以缩写等预取不到的形式给出, 或经 main(argv) 传入时两者不符
    if args.threads != ENV_THREADS:
        parser.error(f'--threads {args.threads} 与导入时写入环境变量的线程数 {ENV_THREADS} 不符; '
                     '须在命令行以 --threads N 完整给出')
    args.grid = tuple(args.grid)
    nx, ny, nz = args.grid
    args.output = (OUTPUT_DIR / f'{args.device}_{args.level}_hex_{nx}x{ny}x{nz}_{SOLVER}_{args.preconditioner}'
                   f'_t{args.threads}')
    return args


def summarize_stages(history: list[dict]) -> dict[str, dict]:
    """按阶段汇总各轮迭代的用时, 首轮与之后各轮分开统计.

    首轮含一次性开销 (如 FA 的 CSR 骨架构建、GPU 的冷启动), 不能代表每轮的稳态用时.

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
    """在选定设备上运行完整拓扑优化流程.
    """
    args = parse_args(argv)
    device = args.device
    bm.set_backend(BACKEND)
    # PyTorch 的 CPU 算子线程池导入时按 OMP_NUM_THREADS 取值; 再显式设置一次, 实际取值以
    # torch.get_num_threads() 记入 config.json
    torch.set_num_threads(args.threads)
    cuda = device == 'cuda'
    precond = args.preconditioner

    grid = args.grid
    max_iter = args.max_iter
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h

    # outputs/ 是临时目录, 同名结果目录直接清空重建, 避免新旧运行的文件混在同一目录
    output = args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    # 同一处记录公共参数, 设备相关信息另行补充.
    config: dict[str, Any] = dict(
        method=f'fem_{args.level}_{device}',
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        operator_level=args.level, assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        backend=BACKEND, device=device, max_iter=max_iter,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=args.threads, torch_num_threads=torch.get_num_threads(),
                     **blas_environment()),
        torch=dict(version=torch.__version__, cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name(device) if cuda else None))
    # rtol 的参照量是初始残差; 固定 rtol=0, 以载荷范数设置绝对容差.
    cg_options = dict(precond=None if precond == 'none' else precond,
                      maxiter=CG_MAXITER, rtol=0.0)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    # 首行在准备阶段之前打印; 单元数与 Q1 位移自由度数由网格剖分算出, 分析器建好后核对
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    device_text = f'{device} ({config["torch"]["device_name"]})' if cuda else device
    print(f'[配置] fem/{args.level}, {BACKEND}/{device_text}, CPU 线程 {torch.get_num_threads()}, '
          f'装配 {ASSEMBLY_METHOD}, 求解 {SOLVER} ({precond}, tol {CG_TOLERANCE:g}, '
          f'maxiter {CG_MAXITER}), 网格 hex {grid} = {n_cells} 单元 / {n_dofs} 自由度, '
          f'问题 {type(problem).__name__}, 输出 {output}', flush=True)

    setup_stages = {}
    with measure(setup_stages, '网格', label='准备', device=device):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=device)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    # 每轮一帧 VTU 写在 output/iterations/, 由 evolution.pvd 串成 ParaView 时间序列; 帧留在运行目录, 需要时手动迁移
    frames = IterationFrameWriter(output, mesh)

    with measure(setup_stages, '分析器', label='准备', device=device):
        # 材料、修正 SIMP 插值 E = Emin + rho^q (E0 - Emin) 与 EA 层级的分析器
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=device)
        # 载荷落在互不相同且不受约束的节点上, ||F||_2 即各点力模长的平方和开方, 与网格无关
        load_norm = math.sqrt(sum(value ** 2 for load in problem.loads() for value in load.force()))
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        )
        analyzer = LagrangeFEMAnalyzer(
            disp_mesh=mesh, pde=problem, material=material,
            space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
            assembly_method=ASSEMBLY_METHOD, operator_level=args.level, solve_method=SOLVER,
            solver_options=dict(cg_options, atol=CG_TOLERANCE * load_norm),
            topopt_algorithm='density_based', interpolation_scheme=interpolation,
        )
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
        # 首轮逐阶段打印, 便于定位大规模下出问题的阶段; 之后只打印汇总行
        label = '迭代 1' if iteration == 1 else None

        # 1. 密度过滤
        with measure(stages, '过滤', label, device=device):
            physical = density_filter.filter_design_variable(
                design_variable=rho,
                physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=device))[:]
            volume_fraction = float(bm.mean(physical))

        # 2. 分析: 依次调用 solve_state 内部的公开步骤, 以便把装配与求解分开计量. EA 的装配只更新
        #    逐单元系数, 施加边界条件只包一层 ConstrainedOperator (Pi_I K Pi_I + Pi_D)
        with measure(stages, '装配', label, device=device):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label, device=device):
            system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
            del stiffness
            displacement = analyzer.tensor_space.function()
        with measure(stages, '求解', label, device=device):
            # 首轮以 Dirichlet 基准向量为初值 (同 solve_state 在非 FA 层级下的默认), 之后以上一轮
            # 位移热启动; 二者的 Dirichlet 分量都取给定值 (本例为零)
            x0 = analyzer.prescribed_solution if previous_displacement is None else previous_displacement
            _, solver_info = analyzer.solve_system(system_matrix, system_load, displacement, x0=x0)
            del system_matrix, system_load
        state = {'displacement': displacement, 'solver': solver_info}
        previous_displacement = displacement[:]
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

        # 每轮一行; Rel. change 为停止判断所用的柔顺度相对变化, 另附装配与求解两阶段的用时 (其余阶段
        # 见运行结束时的耗时汇总), Time 为本轮全部阶段的用时
        record['iteration_seconds'] = perf_counter() - started
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding='utf-8')
        relative_text = f'{relative:.4e}' if relative is not None else '-'
        print(f'Iteration: {iteration}, Objective: {compliance:.4f}, Volfrac: {volume_fraction:.4f}, '
              f'Rel. change: {relative_text}, CG: {solver_info["niter"]}, '
              f'装配 {stages["装配"]["seconds"]:.2f} s, 求解 {stages["求解"]["seconds"]:.2f} s, '
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
    summary = dict(history[-1], converged=converged,
                   termination='converged' if converged else 'max_iter',
                   n_cells=n_cells, n_dofs=n_dofs,
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
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
