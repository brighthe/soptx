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
from soptx.core import blas_environment, measure
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
DOMAIN = (0.0, 210.0, 0.0, 35.0, 0.0, 35.0)
LOAD = -1.0
E0 = 1.0
EMIN = 1.0e-7
NU = 0.3

# 计算后端
BACKEND = 'numpy'
DEVICE = 'cpu'

# 分析参数: 有限元离散与子结构
GRID = (210, 35, 35)
N_FINE = 5
CHUNK_SIZE = 256
TRACE = 'full_trace'
ASSEMBLY_METHOD = 'fast'
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
OPERATOR_LEVEL = 'fa'

# 分析参数: 线性系统求解
SOLVER = 'cg'
PRECONDITIONER = 'gmg'
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
# ParaView 查看根目录 (Windows 本地盘); 运行结束后帧迁移到其下同名子目录, 盘不可用时帧留在 outputs/
VIEW_ROOT = Path('/mnt/c/workspace/soptx-results/topopt_simp_substructure')

    
def parse_args(argv=None):
    """解析分析方法并确定输出目录, 其余设置均为模块顶部常量.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-method', choices=('fem', 'substructure'), required=True,
                        help='分析方法: fem 为传统有限元, substructure 为子结构')
    args = parser.parse_args(argv)
    size = 'x'.join(map(str, GRID))
    name = (f'{TRACE}_hex_{size}_m{N_FINE}_{SOLVER}' if args.analysis_method == 'substructure'
            else f'fa_hex_{size}_{SOLVER}')
    args.output = OUTPUT_DIR.resolve() / name
    return args


def main(argv=None):
    """运行选定方法的完整拓扑优化流程.
    """
    args = parse_args(argv)
    bm.set_backend(BACKEND)
    
    is_substructure = args.analysis_method == 'substructure'

    # 验网格划分与细单元边长, 计算子结构数及密度过滤半径.
    grid = GRID
    if is_substructure and any(n % N_FINE for n in grid):
        raise ValueError('GRID 各方向须能被 N_FINE 整除')
    n_sub = tuple(n // N_FINE for n in grid) if is_substructure else None
    domain = DOMAIN
    domain_size = tuple(domain[2 * d + 1] - domain[2 * d] for d in range(3))
    spacing = tuple(domain_size[d] / grid[d] for d in range(3))
    if not all(math.isclose(value, spacing[0]) for value in spacing):
        raise ValueError('当前密度过滤要求三个方向的细单元边长相同')
    h = spacing[0]
    radius = FILTER_RADIUS_CELLS * h

    # 创建输出目录.
    # outputs/ 是临时目录, 同名结果目录直接清空重建, 避免新旧运行的文件混在同一目录
    output = args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # 同一处记录公共参数, 方法专属参数另行补充.
    config: dict[str, Any] = dict(
        method=f'substructure_{TRACE}' if is_substructure else 'fem_fa',
        domain=list(domain), mesh='hex', grid=list(grid), spacing=spacing,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
        filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        operator_level=OPERATOR_LEVEL, assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
        backend=BACKEND, device=DEVICE, max_iter=MAX_ITER,
        tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
        volume_tolerance=VOLUME_TOLERANCE, vtu_fields=list(VTU_FIELDS),
        threads=dict(num_threads=NUM_THREADS, **blas_environment()))
    cg_options = dict(precond=None if PRECONDITIONER == 'none' else PRECONDITIONER,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned',
                      verify_residual=True)
    config.update(cg_options=cg_options, cg_tolerance=CG_TOLERANCE)
    if is_substructure:
        config.update(trace=TRACE, n_sub=n_sub, n_fine=(N_FINE,) * 3,
                      chunk_size=CHUNK_SIZE)
    (output / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题不依赖网格, 构造很便宜, 在准备阶段之前建好
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    method_text = f'substructure/{TRACE}, n_sub={n_sub}, m={N_FINE}' if is_substructure else 'fem/fa'
    print(f'[配置] {method_text}, {BACKEND}/{DEVICE}, 线程 {NUM_THREADS}, 装配 {ASSEMBLY_METHOD}, '
          f'求解 {SOLVER} ({PRECONDITIONER}, tol {CG_TOLERANCE:g}, maxiter {CG_MAXITER}), '
          f'网格 hex {grid} = {n_cells} 单元 / {n_dofs} 自由度, '
          f'问题 {type(problem).__name__}, 输出 {output}', flush=True)

    setup_stages = {}
    with measure(setup_stages, '网格', label='准备'):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=DEVICE)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    # 每轮一帧 VTU 写在 output/iterations/, 由 evolution.pvd 串成 ParaView 时间序列; 运行结束后迁移到 VIEW_ROOT
    frames = IterationFrameWriter(output, mesh)

    with measure(setup_stages, '材料与插值', label='准备'):
        # 两侧共用材料与 SIMP 插值.
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=DEVICE)
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        )
    if is_substructure:
        with measure(setup_stages, '子结构布局与校验', label='准备'):
            assert n_sub is not None
            layout = StructuredSubstructureLayout(
                domain_size=domain_size,
                n_sub=n_sub, n_fine=(N_FINE,) * 3, degree=SPACE_DEGREE,
                E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh)
        with measure(setup_stages, '分析器', label='准备'):
            analyzer = SubstructureAnalyzer(
                layout=layout, trace=TRACE, chunk_size=CHUNK_SIZE,
                pde=problem, material=material, space_degree=SPACE_DEGREE,
                integration_order=INTEGRATION_ORDER, solve_method=SOLVER,
                solver_options=dict(cg_options),
                topopt_algorithm='density_based', interpolation_scheme=interpolation)
    else:
        with measure(setup_stages, '分析器', label='准备'):
            analyzer = LagrangeFEMAnalyzer(
                disp_mesh=mesh, assembly_method=ASSEMBLY_METHOD,
                operator_level=OPERATOR_LEVEL,
                pde=problem, material=material, space_degree=SPACE_DEGREE,
                integration_order=INTEGRATION_ORDER, solve_method=SOLVER,
                solver_options=dict(cg_options),
                topopt_algorithm='density_based', interpolation_scheme=interpolation)
    if (int(mesh.number_of_cells()), int(analyzer.tensor_space.number_of_global_dofs())) != (n_cells, n_dofs):
        raise RuntimeError('网格单元数或位移自由度数与首行打印的不符')

    with measure(setup_stages, '过滤矩阵', label='准备'):
        density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=radius,
                                density_location='element')

    with measure(setup_stages, '目标与约束', label='准备'):
        objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
        constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC, diff_mode='manual')
        rho = bm.full((n_cells, ), VOLFRAC, dtype=bm.float64, device=DEVICE)
        volume_gradient = density_filter.filter_constraint_sensitivities(
            design_variable=rho, con_grad_rho=constraint.jac(density=rho))

    # 第 0 帧: 初始设计经过滤与实体保留单元处理后的物理密度, 便于查看固定单元; 在计时之前写出
    initial_frame = None
    if 'density' in VTU_FIELDS:
        initial = density_filter.filter_design_variable(
            design_variable=rho,
            physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=DEVICE))[:]
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
        with measure(stages, '过滤', label):
            physical = density_filter.filter_design_variable(
                design_variable=rho,
                physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=DEVICE))[:]
            volume_fraction = float(bm.mean(physical))

        # 2. 分析: 共用装配、边界条件与求解接口, 分阶段记录成本.
        # 子结构的装配含局部缩聚与接口散加, 求解含全场位移恢复.
        with measure(stages, '装配', label):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label):
            system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
            del stiffness
            displacement = analyzer.tensor_space.function()
        with measure(stages, '求解', label):
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
        with measure(stages, '灵敏度', label):
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
        with measure(stages, '输出', label):
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
        history.append(record)
        # 停止时不再更新, 保留刚完成分析的密度, 避免最终柔顺度与密度错位
        stop = converged or iteration == MAX_ITER

        # 6. OC 更新 (计入下一轮之前, 记在本轮记录中)
        if not stop:
            with measure(stages, 'OC', label):
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
    # 初始帧与最后一帧各留一份在运行目录, 迁移查看副本后仍可直接读取: 最后一帧与最终 NPY 对应同一次
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
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')
    frames.publish(VIEW_ROOT)
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
