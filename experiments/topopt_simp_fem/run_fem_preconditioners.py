"""传统有限元 CG 预条件子有效性对照: 三维 MBB 梁 (``FullMBBBeamPoint3d``) 固定密度场下的网格加密.
"""

import os

# 运行环境: BLAS 与 OpenMP 在加载时读取线程数, 须在导入 numpy/scipy 之前写入环境变量才生效.
# 与 run_fem_ea_vs_fa.py 一致固定单线程
NUM_THREADS = 1
for _name in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = str(NUM_THREADS)

import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.core import blas_environment, measure
from soptx.mesh import HexahedronMesh

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.linear_solve import as_iterative_operator
from soptx.fem.preconditioners import create_preconditioner
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.interpolation import MaterialInterpolationScheme


# 模型参数 (与 run_fem_ea_vs_fa.py 相同); ELEMENT_SIZE 为基准网格的单元尺寸, 第 k 档为其 2^-k
ELEMENT_SIZE = 1.0
LOAD = -1.0
E0 = 1.0
EMIN = 1.0e-7
NU = 0.3
PENALTY = 3.0

# 计算后端
BACKEND = 'numpy'
DEVICE = 'cpu'

# 分析参数: 有限元离散; LEVELS 为 --levels 的默认加密次数, 依次运行基准网格的 2^0 .. 2^levels 倍
LEVELS = 2
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
ASSEMBLY_METHOD = 'fast'
# 刚度算子层级: fa 与 ea 为同一算子, 加入 'fa' 仅作交叉核对
OPERATOR_LEVELS = ('fa', )

# 分析参数: 线性系统求解, 停机判据与 run_fem_ea_vs_fa.py 相同
SOLVER = 'cg'
CG_TOLERANCE = 1.0e-6
CG_MAXITER = 20000
# 耗时分解: matvec 与预条件子作用各重复至少 TIMING_MIN_REPEATS 次且累计不少于 TIMING_MIN_SECONDS 秒, 取平均
TIMING_MIN_REPEATS = 3
TIMING_MIN_SECONDS = 0.5

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
# 筛选输入密度目录与可选名字, 名字 name 对应 DENSITY_DIR / f'{name}.npy'
DENSITY_DIR = Path(__file__).resolve().parent / 'fem_preconditioners'
DENSITIES = ('uniform', 'gray', 'binary')


def parse_args(argv=None):
    """解析预条件子、加密次数与密度来源, 其余设置均为模块顶部常量.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preconditioner', choices=('none', 'jacobi', 'gmg'), required=True,
                        help='CG 预条件子: none 为无预条件, jacobi 为对角预条件, gmg 为几何多重网格 V 循环')
    parser.add_argument('--levels', type=int, default=LEVELS,
                        help=f'网格加密次数 K, 依次运行基准网格的 2^0 .. 2^K 倍, 共 K + 1 档 (默认 {LEVELS})')
    parser.add_argument('--density', choices=DENSITIES, required=True,
                        help='筛选输入的单元物理密度 (读入后不过滤): uniform 为均匀密度, gray 为过滤桁架的灰度设计, '
                             'binary 为 0/1 桁架; 读 fem_preconditioners/{name}.npy, 由 make_densities.py 生成')
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error('--levels 至少为 1, 否则无法拟合步数随 h 的斜率')
    return args


def main(argv=None):
    """以所选预条件子在各档网格上依次求解, 汇总步数随 h 的斜率.
    """
    args = parse_args(argv)
    precond = args.preconditioner
    bm.set_backend(BACKEND)

    # 1. 密度与网格档次: 物理域由基准网格与 ELEMENT_SIZE 决定, 第 k 档单元尺寸 h_k = ELEMENT_SIZE / 2^k
    density_file = DENSITY_DIR / f'{args.density}.npy'
    if not density_file.exists():
        raise FileNotFoundError(f'缺少密度文件 {density_file}, 先运行 '
                                'python experiments/topopt_simp_fem/fem_preconditioners/make_densities.py')
    base_density = np.load(density_file)
    if base_density.ndim != 3:
        raise ValueError(f'密度文件须为 (NX, NY, NZ) 三维数组, 得到形状 {base_density.shape}')
    if not np.all(np.isfinite(base_density)) or base_density.min() < 0 or base_density.max() > 1:
        raise ValueError('密度文件须为 [0, 1] 内的有限值')
    base_grid = tuple(int(n) for n in base_density.shape)
    nx, ny, nz = base_grid
    if nx != 6 * ny or ny != nz:
        raise ValueError(f'基准网格须满足 NX = 6 NY, NY = NZ, 得到 {base_grid}')
    domain = tuple(value for n in base_grid for value in (0.0, n * ELEMENT_SIZE))
    grids = [tuple(n * 2 ** k for n in base_grid) for k in range(args.levels + 1)]
    sizes = [ELEMENT_SIZE / 2 ** k for k in range(args.levels + 1)]

    # outputs/ 是临时目录, 同名结果目录直接清空重建, 避免新旧运行的文件混在同一目录
    output = OUTPUT_DIR.resolve() / f'preconditioner_{precond}_{args.density}_{nx}x{ny}x{nz}_L{args.levels}'
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # 停机判据: 以施加边界条件后的实际右端项为参照, 2-范数停机并在收敛时以真残差复核
    cg_options = dict(precond=None if precond == 'none' else precond,
                      maxiter=CG_MAXITER, rtol=CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned', verify_residual=True)
    config: dict[str, Any] = dict(
        method=f'fem_preconditioner_{precond}',
        domain=list(domain), mesh='hex', base_grid=list(base_grid), levels=args.levels,
        grids=[list(grid) for grid in grids], element_sizes=sizes,
        density=dict(name=args.density, file=str(density_file),
                     mean=float(base_density.mean()), min=float(base_density.min()),
                     max=float(base_density.max()), transfer='piecewise_constant_injection'),
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY,
        problem=FullMBBBeamPoint3d.__name__, load=LOAD,
        operator_levels=list(OPERATOR_LEVELS), preconditioner=precond,
        assembly_method=ASSEMBLY_METHOD, solver=SOLVER, cg_options=cg_options,
        backend=BACKEND, device=DEVICE,
        threads=dict(num_threads=NUM_THREADS, **blas_environment()))
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[配置] {BACKEND}/{DEVICE}, 线程 {NUM_THREADS}, 装配 {ASSEMBLY_METHOD}, '
          f'求解 {SOLVER} ({precond}, tol {CG_TOLERANCE:g}, maxiter {CG_MAXITER}), 密度 {args.density} ({density_file}), '
          f'网格 {" -> ".join("x".join(map(str, grid)) for grid in grids)}, '
          f'层级 {"/".join(OPERATOR_LEVELS)}, 输出 {output}', flush=True)

    # 问题、材料与插值与网格无关, 各档共用
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                              device=DEVICE)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
    )

    # 2. 逐档求解; 单档出错只记录, 不中断其余档
    results = []
    for k, (grid, h) in enumerate(zip(grids, sizes)):
        factor = 2 ** k
        grid_text = 'x'.join(map(str, grid))
        n_cells = grid[0] * grid[1] * grid[2]
        n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
        # 分片常数注入: 单元按 x 最慢、z 最快排列 (HexahedronMesh.from_box 的公开约定), 与 C 序 reshape 一致
        density = base_density
        for axis in range(3):
            density = np.repeat(density, factor, axis=axis)
        physical = bm.from_numpy(np.ascontiguousarray(density.reshape(-1), dtype=np.float64))
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=DEVICE)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})

        reference = None
        for level in OPERATOR_LEVELS:
            record: dict[str, Any] = dict(grid=list(grid), h=h, n_cells=n_cells, n_dofs=n_dofs,
                                          operator_level=level, preconditioner=precond,
                                          status='error', preconditioner_setup_seconds=None)
            stages = {}
            started = perf_counter()
            try:
                with measure(stages, '分析器'):
                    analyzer = LagrangeFEMAnalyzer(
                        disp_mesh=mesh, pde=problem, material=material,
                        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
                        assembly_method=ASSEMBLY_METHOD, operator_level=level, solve_method=SOLVER,
                        solver_options=dict(cg_options),
                        topopt_algorithm='density_based', interpolation_scheme=interpolation,
                    )
                with measure(stages, '装配'):
                    stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
                with measure(stages, '边界条件'):
                    # apply_bc 在体力之上补入点载荷等非体力载荷
                    system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
                    del stiffness
                    displacement = analyzer.tensor_space.function()
                # 实测求解耗时含 solve_system 内部的预条件子构造 (GMG 首次构造另含层次结构的建立)
                with measure(stages, '求解'):
                    _, solver_info = analyzer.solve_system(system_matrix, system_load, displacement,
                                                           x0=analyzer.prescribed_solution)
                    del system_load
                # 耗时分解: 求解后另构造一次预条件子单独计时. GMG 的层次结构已缓存于分析器, 测得的是按当前
                # 单元系数重建各粗层并分解最粗层的成本, 即拓扑优化中每轮重建的成本. 再以 CG 内部同样的
                # A @ p 与 M @ r 对随机向量重复作用取平均; 求解耗时减去模型预测 (构造 + 步数 x (matvec + 作用))
                # 为其他开销, 含向量运算、GMG 首次建立层次结构与调用开销
                if precond == 'none':
                    preconditioner, setup_seconds = None, 0.0
                else:
                    setup_started = perf_counter()
                    preconditioner = create_preconditioner(precond, analyzer, {})
                    preconditioner.setup(system_matrix)
                    setup_seconds = perf_counter() - setup_started
                operator = as_iterative_operator(system_matrix)
                vector = bm.from_numpy(np.random.default_rng(0).standard_normal(n_dofs))
                timings = {}
                for name, apply in (('matvec', lambda: operator @ vector),
                                    ('apply', None if preconditioner is None else lambda: preconditioner @ vector)):
                    if apply is None:
                        timings[name] = 0.0
                        continue
                    repeats, timing_started = 0, perf_counter()
                    while True:
                        apply()
                        repeats += 1
                        elapsed = perf_counter() - timing_started
                        if repeats >= TIMING_MIN_REPEATS and elapsed >= TIMING_MIN_SECONDS:
                            break
                    timings[name] = elapsed / repeats
                del system_matrix, operator, preconditioner, vector
                # 柔顺度用含全部载荷 (含点载荷) 的外载向量, 不计时
                load = analyzer.assemble_external_load()
                solution = bm.to_numpy(displacement[:])
                # 位移按节点优先编号 (3 n + c) 存盘, 供与其他预条件子的运行逐分量比较
                displacement_file = f'displacement_{level}_{grid_text}.npy'
                np.save(output / displacement_file, solution)
                niter = solver_info['niter']
                # 停机阈值可能是 0 维数组, 转成 float 才能写入 JSON
                threshold = solver_info.get('stopping_threshold')
                record.update(
                    status='converged' if solver_info['converged'] else 'not_converged',
                    cg_iterations=niter,
                    relative_residual=solver_info['relres'],
                    true_residual=solver_info.get('true_residual'),
                    stopping_threshold=None if threshold is None else float(np.asarray(threshold)),
                    reason=solver_info.get('reason'),
                    compliance=float(bm.sum(load * displacement[:])),
                    setup_seconds=stages['分析器']['seconds'],
                    assembly_seconds=stages['装配']['seconds'] + stages['边界条件']['seconds'],
                    solve_seconds=stages['求解']['seconds'],
                    preconditioner_setup_seconds=setup_seconds,
                    matvec_seconds=timings['matvec'],
                    apply_seconds=timings['apply'],
                    model_solve_seconds=setup_seconds + niter * (timings['matvec'] + timings['apply']),
                    displacement_file=displacement_file)
                record['other_seconds'] = record['solve_seconds'] - record['model_solve_seconds']
                # 同一档网格上不同算子层级的收敛解逐分量比较, 核对二者解的是同一个离散问题
                if record['status'] == 'converged':
                    if reference is None:
                        reference = (level, solution)
                    else:
                        scale = float(np.max(np.abs(reference[1])))
                        record['solution_reference'] = reference[0]
                        record['solution_difference'] = float(np.max(np.abs(solution - reference[1]))) / scale
                del analyzer, load, displacement, solution
            except Exception as exc:  # noqa: BLE001 - 单档失败只记录, 继续其余档
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['total_seconds'] = perf_counter() - started
            results.append(record)
            (output / 'results.json').write_text(
                json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

            if record['status'] == 'error':
                print(f'[错误] {grid_text} {level}: {record["error"]}', flush=True)
            else:
                print(f'Grid: {grid_text}, {level}, {record["status"]}, CG: {record["cg_iterations"]}, '
                      f'Rel. res: {record["relative_residual"]:.3e}, '
                      f'Solve: {record["solve_seconds"]:.2f} sec', flush=True)

    # 3. 汇总: 每个算子层级以收敛档的 log(niter) 对 log(h) 做最小二乘拟合
    slopes = {}
    for level in OPERATOR_LEVELS:
        points = [(record['h'], record['cg_iterations']) for record in results
                  if record['operator_level'] == level and record['status'] == 'converged']
        slope = None
        if len(points) >= 2:
            h_values, iterations = zip(*points)
            slope = float(np.polyfit(np.log(h_values), np.log(np.maximum(iterations, 1)), 1)[0])
        slopes[level] = dict(slope=slope, converged_levels=len(points))
    differences = [record['solution_difference'] for record in results if 'solution_difference' in record]
    summary = dict(preconditioner=precond, slopes=slopes,
                   max_level_difference=max(differences) if differences else None,
                   n_cases=len(results),
                   n_converged=sum(record['status'] == 'converged' for record in results),
                   n_not_converged=sum(record['status'] == 'not_converged' for record in results),
                   n_error=sum(record['status'] == 'error' for record in results))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding='utf-8')

    # 每个层级一张表: 行为网格档, 列为 CG 步数与耗时分解 (秒); 未收敛记 >maxiter, 出错记 err
    columns = ('preconditioner_setup_seconds', 'matvec_seconds', 'apply_seconds', 'solve_seconds', 'other_seconds')
    for level in OPERATOR_LEVELS:
        print(f'\n[{precond}/{level}]')
        print(f'{"网格":<14}{"h":>8}{"CG 步数":>10}{"构造":>12}{"matvec":>12}{"作用":>12}{"求解":>12}{"其他":>12}')
        for record in results:
            if record['operator_level'] != level:
                continue
            status = record['status']
            iterations = {'converged': str(record.get('cg_iterations')),
                          'not_converged': f'>{CG_MAXITER}', 'error': 'err'}[status]
            cells = ''.join(f'{"-" if status == "error" else f"{record[key]:.3e}":>12}' for key in columns)
            print(f'{"x".join(map(str, record["grid"])):<14}{record["h"]:>8g}{iterations:>10}{cells}')
        slope = slopes[level]['slope']
        print(f'斜率 d log(niter) / d log(h): {"-" if slope is None else f"{slope:.3f}"}')

    max_difference = summary['max_level_difference']
    difference_text = '' if max_difference is None else f', 层级间解最大相对差 {max_difference:.2e}'
    print(f'\n[完成] {summary["n_converged"]}/{summary["n_cases"]} 收敛, 未收敛 {summary["n_not_converged"]}, '
          f'出错 {summary["n_error"]}{difference_text}, 结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
