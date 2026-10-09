"""精确子结构有限元 + linear_corner + FA 拓扑优化: 三维 MBB 梁.

每轮依次为密度过滤, 分析, 柔顺度与灵敏度, 停止判断, OC 更新; 停止判断在更新之前, 落盘的
密度即最后一次分析的设计. 分析由 SubstructureAnalyzer 完成: 逐批局部装配与精确 Schur 缩聚,
接口系统求解, 全场位移恢复; 目标、约束、过滤与 OC 与 run_fa.py 相同.
"""

import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any
import xml.etree.ElementTree as ET

import math

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.core import measure

from soptx.fem.analyzers import SubstructureAnalyzer
from soptx.fem.substructure import StructuredSubstructureLayout
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.postprocess.vtk_export import write_vtu
from soptx.problems import FullMBBBeam3d
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
SUPPORT = 'end_corners'
# 设计关于 z 中面对称
SYMMETRY = 'z'

# 有限元离散与求解
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
# 接口迹空间: full_trace 保留全部接口自由度, 与 FA 代数等价; linear_corner 为角点线性插值, 即文献的子结构方法
TRACE = 'linear_corner'
# 接口 Schur 算子的形式: fa 为各子结构局部 Schur 补散加成全局接口 CSR; 隐式作用的 ea 档尚未由库提供
OPERATOR_LEVEL = 'fa'

# 优化参数
VOLFRAC = 0.12
FILTER_RADIUS_CELLS = 3.0
PENALTY = 3.0
OC_OPTIONS = dict(move_limit=0.2, damping_coef=0.5, initial_lambda=1.0e9,
                  bisection_tol=1.0e-3, design_variable_min=0.0)
TOLERANCE = 2.0e-4
CONVERGENCE_WINDOW = 5
VOLUME_TOLERANCE = 1.0e-6


def parse_args(argv=None):
    """解析网格, 迭代与输出参数."""
    cg_defaults = dict(cg_tol=1.0e-6, cg_maxiter=20000, precond='jacobi')

    parser = argparse.ArgumentParser(description='精确子结构有限元 + linear_corner 拓扑优化 (MBB 梁)')
    parser.add_argument('--mesh', choices=('hex', ), default='hex',
                        help='网格类型 (目前只支持 hex: 结构化子结构布局与结构化密度过滤均要求六面体网格)')
    
    parser.add_argument('--n-sub', type=int, nargs=3, default=(78, 13, 13), metavar=('NX', 'NY', 'NZ'),
                        help='各方向子结构数, 须满足 NX = 6 NY 且 NY = NZ (默认: 78 13 13, 文献首档)')
    parser.add_argument('--n-fine', type=int, default=5,
                        help='每个子结构各方向的细单元数 m, 细网格为 n-sub 的 m 倍 (默认: 5, 文献取值)')
    parser.add_argument('--chunk-size', type=int, default=256,
                        help='局部刚度装配, 内部消元与位移恢复时每批最多处理的子结构数, 只影响内存 (默认: 256)')
    
    parser.add_argument('--backend', choices=('numpy', 'pytorch'), default='numpy', help='计算后端 (默认: numpy)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='计算设备 (默认: cpu); cuda 须配 pytorch 后端')
    parser.add_argument('--cg-tol', type=float, default=None,
                        help=f'CG 停机容差, 自由接口上 ||b - S q||_2 <= cg_tol ||b||_2 (默认: {cg_defaults["cg_tol"]:g}, 仅 cg)')
    parser.add_argument('--cg-maxiter', type=int, default=None,
                        help=f'CG 最大迭代数 (默认: {cg_defaults["cg_maxiter"]}, 仅 cg)')
    parser.add_argument('--precond', choices=('mg', 'jacobi', 'none'), default=None,
                        help=f'CG 预条件子: mg 为细网格几何多重网格限制到接口 (仅 full_trace), jacobi 为对角 '
                             f'(默认: {cg_defaults["precond"]}, 仅 cg)')
    parser.add_argument('--solver', choices=('cg', 'mumps', 'scipy'), default='cg',
                        help='接口系统求解器 (默认: cg, 即预条件共轭梯度法; mumps/scipy 为稀疏直接法)')
    parser.add_argument('--max-iter', type=int, default=300, help='优化最大迭代数 (默认: 300)')
    parser.add_argument('--vtu-fields', nargs='+', choices=('density', 'displacement'), default=['density'],
                        help='每轮及最终 VTU 保存的场, 可多选 (默认: density); displacement 写为节点向量 u')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs',
                        help='输出根目录, 结果写入其下 {接口}_{网格}_{nx}x{ny}x{nz}_m{n-fine}_{求解器} 子目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='结果子目录已存在时清空后重写; 不加则拒绝运行, 以免误删已有结果')
    args = parser.parse_args(argv)

    nx, ny, nz = args.n_sub
    if min(args.n_sub) < 1 or args.max_iter < 1 or args.chunk_size < 1:
        parser.error('n-sub 各方向, max-iter 与 chunk-size 必须为正整数')
    # 子结构为立方体, 三个方向细单元边长相同, 过滤半径 "3 个单元" 才有唯一含义
    if nx != 6 * ny or ny != nz:
        parser.error(f'n-sub 须满足 NX = 6 NY 且 NY = NZ, 实际为 {tuple(args.n_sub)}')
    # 子结构每向至少 2 个细单元才有内部节点可消元
    if args.n_fine < 2:
        parser.error('--n-fine 至少为 2, 以保留内部节点')
    # 直接法只支持 numpy 后端加 CPU; cuda 须用 pytorch 后端且 CUDA 可用
    if args.solver in ('mumps', 'scipy') and (args.backend != 'numpy' or args.device != 'cpu'):
        parser.error(f'直接法 {args.solver} 只支持 --backend numpy --device cpu')
    # CG 选项只对 cg 有意义, 直接法下显式给出视为误用; cg 下未给出的取默认值
    cg_given = [name for name, value in (('--cg-tol', args.cg_tol), ('--cg-maxiter', args.cg_maxiter),
                                         ('--precond', args.precond)) if value is not None]
    if args.solver != 'cg' and cg_given:
        parser.error(f'{", ".join(cg_given)} 只在 --solver cg 时有效')
    if args.solver == 'cg':
        for name, value in cg_defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if not args.cg_tol > 0 or args.cg_maxiter < 1:
            parser.error('--cg-tol 须为正数, --cg-maxiter 须为正整数')
    if args.device == 'cuda':
        if args.backend != 'pytorch':
            parser.error('--device cuda 须配合 --backend pytorch')
        import torch
        if not torch.cuda.is_available():
            parser.error('--device cuda 但当前环境 CUDA 不可用')
    args.n_sub = (nx, ny, nz)
    args.grid = (nx * args.n_fine, ny * args.n_fine, nz * args.n_fine)
    args.vtu_fields = list(dict.fromkeys(args.vtu_fields))

    # 按区分运行的关键参数命名, 细网格后缀 m 为每子结构的细单元数; 后端、设备与 CG 选项不改变离散问题, 只记入 config.json
    gx, gy, gz = args.grid
    args.output = args.output_dir / f'{TRACE}_{args.mesh}_{gx}x{gy}x{gz}_m{args.n_fine}_{args.solver}'
    if args.output.exists() and not args.overwrite:
        parser.error(f'结果目录 {args.output} 已存在; 确认要覆盖请加 --overwrite')
    return args


def main(argv=None):
    """过滤、分析与灵敏度计算后执行停止判断和 OC 更新, 逐轮落盘."""
    args = parse_args(argv)
    bm.set_backend(args.backend)

    grid = args.grid
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    spacing = (h, h, h)
    radius = FILTER_RADIUS_CELLS * h

    # 覆盖时先清空, 避免新旧运行的文件混在同一目录
    output = args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    # 每轮一帧 VTU, 由 evolution.pvd 串成 ParaView 时间序列
    iteration_dir = output / 'iterations'
    iteration_dir.mkdir()
    pvd_root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    pvd_collection = ET.SubElement(pvd_root, 'Collection')
    # 写入 config.json 的异构记录, 值类型不一
    config: dict[str, Any] = dict(method=f'substructure_{TRACE}', domain=list(domain), mesh=args.mesh, grid=grid,
                  spacing=spacing, trace=TRACE, n_sub=args.n_sub, n_fine=(args.n_fine, ) * 3, chunk_size=args.chunk_size,
                  space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
                  E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, volfrac=VOLFRAC,
                  filter_type='density', filter_radius=radius, optimizer=OC_OPTIONS,
                  support=SUPPORT, load=LOAD,
                  symmetry=SYMMETRY,
                  operator_level=OPERATOR_LEVEL, solver=args.solver,
                  backend=args.backend, device=args.device,
                  max_iter=args.max_iter, tolerance=TOLERANCE, convergence_window=CONVERGENCE_WINDOW,
                  volume_tolerance=VOLUME_TOLERANCE,
                  vtu_fields=args.vtu_fields)
    is_cg = args.solver == 'cg'
    cg_options = None
    if is_cg:
        # 停机判据 ||b - A u||_2 <= cg_tol ||F||_2: rtol 的参照量是初始残差, 热启动后会越来越严,
        # 故取 rtol = 0, 由 atol = cg_tol ||F||_2 给出 (见分析器构造处)
        cg_options = dict(precond=None if args.precond == 'none' else args.precond,
                          maxiter=args.cg_maxiter, rtol=0.0)
        config.update(cg_options=cg_options, cg_tolerance=args.cg_tol)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    # 问题不依赖网格, 构造很便宜, 先建好以便首行给出载荷点数
    problem = FullMBBBeam3d(domain=domain, P=LOAD, E=E0, nu=NU,
                            support=SUPPORT, load_subdivisions=(grid[0], grid[2]))
    # 首行在准备阶段之前打印; 单元数与 Q1 位移自由度数由网格剖分算出, 分析器建好后核对
    n_cells = grid[0] * grid[1] * grid[2]
    n_dofs = 3 * (grid[0] + 1) * (grid[1] + 1) * (grid[2] + 1)
    # CG 另给预条件子、停机容差与最大迭代数; 直接法只给名字
    solver_text = (f'{args.solver} ({args.precond}, tol {args.cg_tol:g}, maxiter {args.cg_maxiter})'
                   if is_cg else args.solver)
    print(f'[配置] {TRACE}, {args.backend}/{args.device}, 求解 {solver_text}, '
          f'细网格 {args.mesh} {grid} = {n_cells} 单元 / {n_dofs} 自由度 = 子结构 {args.n_sub} x {args.n_fine}^3, '
          f'载荷点 {len(problem.loads())} 个, '
          f'支承 {SUPPORT}, 对称约束 {SYMMETRY}, 输出 {output}',
          flush=True)

    setup_stages = {}
    with measure(setup_stages, '网格', label='准备'):
        # 布局给出细网格与位移空间, 编号与 from_box 相同; 材料参数须与下方 material 一致
        layout = StructuredSubstructureLayout(
            domain_size=(grid[0] * h, grid[1] * h, grid[2] * h), n_sub=args.n_sub, n_fine=(args.n_fine, ) * 3,
            degree=SPACE_DEGREE, E_base=E0, nu=NU, hypothesis='3D',
        )
        mesh = layout.full_mesh
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})

    with measure(setup_stages, '分析器', label='准备'):
        # 材料、修正 SIMP 插值 E = Emin + rho^q (E0 - Emin) 与子结构分析器; 插值由分析器注入参考子结构
        material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                                  device=args.device, enable_logging=False)
        # 载荷落在互不相同且不受约束的节点上, ||F||_2 即各点力模长的平方和开方, 与网格无关
        load_norm = math.sqrt(sum(value ** 2 for load in problem.loads() for value in load.force()))
        interpolation = MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
            enable_logging=False,
        )
        analyzer = SubstructureAnalyzer(
            layout=layout, pde=problem, material=material, trace=TRACE, chunk_size=args.chunk_size,
            space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER, solve_method=args.solver,
            solver_options=dict(cg_options, atol=args.cg_tol * load_norm) if cg_options is not None else None,
            topopt_algorithm='density_based', interpolation_scheme=interpolation,
            enable_logging=False,
        )
        if (int(mesh.number_of_cells()), int(analyzer.tensor_space.number_of_global_dofs())) != (n_cells, n_dofs):
            raise RuntimeError('网格单元数或位移自由度数与首行打印的不符')

    with measure(setup_stages, '过滤矩阵', label='准备'):
        density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=radius,
                                density_location='element', enable_logging=False)

    with measure(setup_stages, '目标与约束', label='准备'):
        objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual',
                                        enable_logging=False)
        constraint = VolumeConstraint(analyzer=analyzer, volume_fraction=VOLFRAC, diff_mode='manual',
                                      enable_logging=False)
        rho = bm.full((n_cells, ), VOLFRAC, dtype=bm.float64, device=args.device)
        volume_gradient = density_filter.filter_constraint_sensitivities(
            design_variable=rho, con_grad_rho=constraint.jac(density=rho))
        # 理论上已对称, 取与 z 向镜像的平均, 消去过滤求和顺序留下的舍入差异
        gradient_grid = bm.reshape(volume_gradient, grid)
        volume_gradient = bm.reshape(0.5 * (gradient_grid + bm.flip(gradient_grid, axis=2)), (-1, ))

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
                physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=args.device))[:]
            volume_fraction = float(bm.mean(physical))

        # 2. 分析: 依次调用 solve_state 内部的公开步骤, 以便把缩聚装配与接口求解分开计量, 与 solve_state
        #    逐步等价; '装配' 含逐批局部装配、精确 Schur 缩聚与接口散加, '求解' 含接口求解与全场位移恢复
        with measure(stages, '装配', label):
            stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        with measure(stages, '边界条件', label):
            system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
            del stiffness
            displacement = analyzer.tensor_space.function()
        with measure(stages, '求解', label):
            # CG 自第二轮起以上一轮接口未知量热启动 (子结构分析器的初值是 Q 而非全场位移)
            warm_start = {'x0': previous_displacement} if is_cg and previous_displacement is not None else {}
            _, solver_info = analyzer.solve_system(system_matrix, system_load, displacement, **warm_start)
            del system_matrix, system_load
        state = {'displacement': displacement, 'solver': solver_info}
        previous_displacement = analyzer.interface_displacement
        if is_cg and not solver_info['converged']:
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
            # 取与 z 向镜像的平均, 投影到对称设计的子空间: 共用变量的梯度为镜像两单元之和, 取平均
            # 只差常数因子, 不影响 OC 的乘子; 浮点加法可交换, 结果逐位对称
            dc_grid = bm.reshape(dc, grid)
            dc = bm.reshape(0.5 * (dc_grid + bm.flip(dc_grid, axis=2)), (-1, ))

        analysis_seconds = perf_counter() - started

        # 4. 输出: 保存本轮分析所用设计的所选场; VTU 写完再更新 PVD, 都先写临时文件再改名,
        #    中途中断不留下半个文件. 位移按节点优先编号 (3 n + c), 可直接重排为 (NN, 3)
        with measure(stages, '输出', label):
            cell_data = {'density': bm.to_numpy(physical)} if 'density' in args.vtu_fields else {}
            point_data = ({'u': bm.to_numpy(displacement[:]).reshape(-1, 3)}
                          if 'displacement' in args.vtu_fields else {})
            frame = iteration_dir / f'iter_{iteration:04d}.vtu'
            temporary_base = iteration_dir / f'.iter_{iteration:04d}.tmp'
            temporary_vtu = Path(str(temporary_base) + '.vtu')
            try:
                write_vtu(mesh=mesh, filepath=str(temporary_base),
                          cell_data=cell_data or None, point_data=point_data or None)
                temporary_vtu.replace(frame)
            finally:
                temporary_vtu.unlink(missing_ok=True)
            frame_file = frame.relative_to(output).as_posix()
            ET.SubElement(pvd_collection, 'DataSet', timestep=str(iteration), group='', part='0', file=frame_file)
            temporary_pvd = output / '.evolution.pvd.tmp'
            try:
                ET.ElementTree(pvd_root).write(temporary_pvd, encoding='utf-8', xml_declaration=True)
                temporary_pvd.replace(output / 'evolution.pvd')
            finally:
                temporary_pvd.unlink(missing_ok=True)
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
        # 设计变量的不对称量; 开启 z 对称约束时应恒为 0
        design_grid = bm.reshape(rho, grid)
        design_symmetry_z = float(bm.max(bm.abs(design_grid - bm.flip(design_grid, axis=2))))
        assembly_seconds = stages['装配']['seconds'] + stages['边界条件']['seconds']
        solve_seconds = stages['求解']['seconds']
        peak_gib = max(stage['peak_gib'] for stage in stages.values())
        record = dict(iteration=iteration, compliance=compliance, volume_fraction=volume_fraction,
                      relative_change=relative, symmetry_error_z=symmetry_z,
                      design_symmetry_error_z=design_symmetry_z,
                      assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
                      cg_iterations=solver_info['niter'] if is_cg else None,
                      cg_relative_residual=solver_info['relres'] if is_cg else None,
                      cg_converged=solver_info['converged'] if is_cg else None,
                      analysis_seconds=analysis_seconds, peak_rss_gib=peak_gib,
                      vtu_file=frame_file, stages=stages)
        history.append(record)
        # 停止时不再更新, 保留刚完成分析的密度, 避免最终柔顺度与密度错位
        stop = converged or iteration == args.max_iter

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
              f'Rel. change: {relative_text}' + (f', CG: {solver_info["niter"]}' if is_cg else '')
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
    # 最后一帧与最终 NPY 对应同一次分析, 复用已导出的所选场
    final_temporary = output / '.result_final.vtu.tmp'
    try:
        shutil.copyfile(output / history[-1]['vtu_file'], final_temporary)
        final_temporary.replace(output / 'result_final.vtu')
    finally:
        final_temporary.unlink(missing_ok=True)
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
    print(f'[完成] {summary["termination"]}，{iteration} 轮，C={compliance:.4f}，'
          f'不对称量 x {symmetry_x:.4e} / z {symmetry_z:.4e}，结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
