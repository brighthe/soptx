"""传统有限元 + FA 拓扑优化: Huang2023 第 4.1 节三维 MBB 梁的完整流程.

参数取自同目录 README.md, 作为模块常量写在下方. 本脚本以 FA (全局稀疏矩阵 + 直接法)
求解, 用于小规模下给出可信参照; 大规模基线改用 EA, 其余流程不变.

流程与 ``experiments/topopt_exact_substructure/run_linear_corner.py`` 逐步对应, 只把
子结构缩聚换成细网格有限元分析, 输出的 config / history / summary 字段一致, 以便配对比较:

1. 密度过滤: ``Filter`` 的密度过滤, 锥形权重 max(0, rmin - d), 得物理密度;
2. 分析: Q1 位移元, 2 x 2 x 2 高斯积分, 修正 SIMP, FA + 直接法;
3. 柔顺度 c = F^T u 与其对物理密度的灵敏度, 经过滤的链式法则转到设计变量;
4. 停止判断: r_k = |c_k - c_{k-1}| / c_k 连续 5 个小于 2e-4;
5. OC 更新: 物理体积分数对设计变量是线性的, V(rho) = sum(g * rho), g 为体积约束经过滤
   链式法则得到的设计梯度, 只算一次, 二分乘子时判断约束只需一次点积.

停止判断在 OC 更新之前: 一旦满足停止条件即不再更新, 落盘的物理密度即最后一次分析的
设计, 与其柔顺度一一对应.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import math

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import FullMBBBeam3d
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective
from soptx.topology.optimizers import OCOptimizer

# 模型参数
DOMAIN = (0.0, 6.0, 0.0, 1.0, 0.0, 1.0)
LOAD = -1.0
E0 = 1.0
EMIN = 1.0e-7
NU = 0.3
SUPPORT = 'end_lines'

# 有限元离散与求解
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2        
OPERATOR_LEVEL = 'fa'
ASSEMBLY_METHOD = 'fast'
SOLVER = 'mumps'

# 优化参数
VOLFRAC = 0.12
FILTER_RADIUS_CELLS = 3.0    
PENALTY = 3.0
OC_OPTIONS = dict(move_limit=0.2, damping_coef=0.5, initial_lambda=1.0e9,
                  bisection_tol=1.0e-3, design_variable_min=0.0)
TOLERANCE = 2.0e-4
CONVERGENCE_WINDOW = 5
MAX_ITER = 300

OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs'


def parse_args(argv=None):
    """解析网格, 迭代与输出参数."""
    parser = argparse.ArgumentParser(description='传统有限元 + FA 拓扑优化 (MBB 梁)')
    parser.add_argument('--grid', type=int, nargs=3, default=(60, 10, 10), metavar=('NX', 'NY', 'NZ'),
                        help='网格剖分, 须满足 NX = 6 NY 且 NY = NZ (默认: 60 10 10)')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER, help=f'最大迭代数 (默认: {MAX_ITER})')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT,
                        help='输出根目录, 每次运行在其下新建带时间戳的子目录')
    args = parser.parse_args(argv)

    nx, ny, nz = args.grid
    if min(args.grid) < 1 or args.max_iter < 1:
        parser.error('grid 各方向与 max-iter 必须为正整数')
    # 三个方向单元边长相同, 过滤半径 "3 个单元" 才有唯一含义
    if nx != 6 * ny or ny != nz:
        parser.error(f'grid 须满足 NX = 6 NY 且 NY = NZ, 实际为 {tuple(args.grid)}')
    args.grid = (nx, ny, nz)
    return args


def main(argv=None):
    """过滤、分析与灵敏度计算后执行停止判断和 OC 更新, 逐轮落盘."""
    args = parse_args(argv)
    bm.set_backend('numpy')

    grid = args.grid
    mesh = HexahedronMesh.from_box(list(DOMAIN), *grid)
    h = (DOMAIN[1] - DOMAIN[0]) / grid[0]
    setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h
    n_cells = int(mesh.number_of_cells())

    output = args.output_dir / datetime.now(timezone.utc).strftime('fa_%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(method='fem_fa', domain=list(DOMAIN), grid=grid, spacing=spacing,
                  space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
                  E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
                  filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
                  support=SUPPORT, load=LOAD,
                  operator_level=OPERATOR_LEVEL, assembly_method=ASSEMBLY_METHOD, solver=SOLVER,
                  max_iter=args.max_iter, tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 问题、材料、修正 SIMP 插值 E = Emin + rho^q (E0 - Emin) 与 FA 分析器
    problem = FullMBBBeam3d(domain=DOMAIN, P=LOAD, E=E0, nu=NU,
                            support=SUPPORT, load_subdivisions=(grid[0], grid[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU,
                                              hypothesis='3D', enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        enable_logging=False,
    )
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        assembly_method=ASSEMBLY_METHOD, operator_level=OPERATOR_LEVEL, solve_method=SOLVER,
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
        enable_logging=False,
    )
    density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=radius,
                            density_location='element', enable_logging=False)
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual',
                                    enable_logging=False)
    constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC, diff_mode='manual',
                                  enable_logging=False)
    n_dofs = int(analyzer.tensor_space.number_of_global_dofs())
    print(f'[配置] FA, 网格 {grid} = {n_cells} 单元, {n_dofs} 自由度, '
          f'支承 {SUPPORT}, 载荷点 {len(problem.loads())} 个, 输出 {output}', flush=True)

    rho = bm.full((n_cells, ), VOLFRAC, dtype=bm.float64)
    volume_gradient = density_filter.filter_constraint_sensitivities(
        design_variable=rho, con_grad_rho=constraint.jac(density=rho))
    history = []
    recent = []
    converged = False
    run_started = perf_counter()
    # 循环只经由停止判断后的 break 退出 (最后一轮必有 iteration == max_iter), 写成 while True
    # 使循环体内赋值的 physical / state / compliance 在循环后确定已绑定
    iteration = 0
    while True:
        iteration += 1
        started = perf_counter()

        # 1. 密度过滤
        physical = density_filter.filter_design_variable(
            design_variable=rho, physical_density=bm.zeros((n_cells, ), dtype=bm.float64))[:]
        volume_fraction = float(bm.mean(physical))

        # 2. 分析
        stamp = perf_counter()
        state = analyzer.solve_state(rho_val=physical)
        solve_seconds = perf_counter() - stamp

        # 3. 柔顺度与灵敏度
        compliance = float(objective.fun(density=physical, state=state))
        if not math.isfinite(compliance) or compliance <= 0:
            raise FloatingPointError('柔顺度必须为有限正数')
        dc = density_filter.filter_objective_sensitivities(
            design_variable=rho, obj_grad_rho=objective.jac(density=physical, state=state))
        if not bool(bm.all(bm.isfinite(dc))):
            raise FloatingPointError('柔顺度灵敏度非有限')

        # 4. 停止判断
        relative = None if not history else abs(compliance - history[-1]['compliance']) / compliance
        if relative is not None:
            recent.append(relative)
            recent = recent[-CONVERGENCE_WINDOW:]
        converged = len(recent) == CONVERGENCE_WINDOW and all(value < TOLERANCE for value in recent)
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=volume_fraction,
                      relative_change=relative, solve_seconds=solve_seconds,
                      analysis_seconds=perf_counter() - started)
        history.append(record)
        relative_text = f'{relative:.3e}' if relative is not None else '-'
        print(f'[迭代 {iteration}] C={compliance:.10e}，体积分数 {volume_fraction:.6f}，'
              f'相对变化 {relative_text}，求解 {solve_seconds:.2f} s', flush=True)
        (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding='utf-8')
        # 停止时保留刚完成分析的密度, 避免最终柔顺度与密度错位
        if converged or iteration == args.max_iter:
            break

        # 5. OC 更新
        rho_new = OCOptimizer.update_design_variable(
            design_variable=rho, objective_gradient=dc, constraint_gradient=volume_gradient,
            constraint_function=lambda candidate: bm.sum(volume_gradient * candidate) - VOLFRAC,
            **OC_OPTIONS,
        )[:]
        if (not bool(bm.all(bm.isfinite(rho_new)))
                or float(bm.min(rho_new)) < 0 or float(bm.max(rho_new)) > 1):
            raise FloatingPointError('OC 更新得到无效密度')
        if float(bm.sum(volume_gradient * rho_new)) > VOLFRAC + 1.0e-6:
            raise RuntimeError('OC 乘子搜索未满足物理体积约束')
        rho = rho_new

    # 最终物理密度关于 x = 3 与 z = 0.5 两个中面的最大不对称量
    # torch 不支持负步长切片, 翻转用 bm.flip
    physical_grid = bm.reshape(physical, grid)
    symmetry_x = float(bm.max(bm.abs(physical_grid - bm.flip(physical_grid, axis=0))))
    symmetry_z = float(bm.max(bm.abs(physical_grid - bm.flip(physical_grid, axis=2))))

    # 写文件须在主机内存的 numpy 数组上进行, 经 bm.to_numpy 落到 numpy
    np.save(output / 'design_density_final.npy', bm.to_numpy(bm.reshape(rho, grid)))
    np.save(output / 'density_final.npy', bm.to_numpy(physical_grid))
    np.save(output / 'displacement_final.npy', bm.to_numpy(state['displacement'][:]))
    write_vtu(mesh=mesh, filepath=str(output / 'density_final'),
              cell_data={'density': bm.to_numpy(physical)})
    summary = dict(history[-1], converged=converged,
                   termination='converged' if converged else 'max_iter',
                   n_cells=n_cells, n_dofs=n_dofs,
                   symmetry_error_x=symmetry_x, symmetry_error_z=symmetry_z,
                   total_seconds=perf_counter() - run_started)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.10e}，'
          f'不对称量 x {symmetry_x:.1e} / z {symmetry_z:.1e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
