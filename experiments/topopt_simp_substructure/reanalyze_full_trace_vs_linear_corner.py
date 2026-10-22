"""固定物理密度上 full_trace 与 linear_corner 的单次重分析: 三维 MBB 梁 (``FullMBBBeamPoint3d``).

对同一物理密度场分别用两种接口迹做一次分析, 核对对照文档 §1.3 的 Ritz 关系, 结果填入 §3.1:
``C_C <= C_Gamma``; 能量范数直接算的 ``eps_E`` 与 ``sqrt(eps_C)`` 吻合; 恢复后的全场位移代回
细网格方程, 残差只在接口自由度上非零, 且与角点子空间正交 (``P^T r_Gamma = 0``). 不过滤 (``uniform`` 除外)、不做 OC 更新, 不跑优化.

模型、材料、过滤半径、CG 参数等顶部常量直接取自 ``run_substructure_full_trace_vs_linear_corner``,
两处不另抄一份; 网格可用 ``--grid`` 覆盖, 用于小算例验收与 390 档补充.
"""
import os

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.core import blas_environment, measure
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import LagrangeFEMAnalyzer, SubstructureAnalyzer
from soptx.fem.substructure import StructuredSubstructureLayout
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.filters import Filter
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_substructure_full_trace_vs_linear_corner as run  # noqa: E402  顶部常量的唯一来源

PRECONDITIONER = 'jacobi'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'


def parse_args(argv=None):
    """解析命令行: 密度来源、子结构细分数、网格与输出目录.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    args : argparse.Namespace
        含 ``density`` (``'uniform'`` 或 ``.npy`` 路径)、``n_fine``、``grid``、``output``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density', required=True,
                        help="'uniform' 为初始均匀设计经过滤后的物理密度; 否则为物理密度 .npy 路径, 不再过滤")
    parser.add_argument('--n-fine', type=int, default=run.N_FINE, help='子结构每方向细单元数 m')
    parser.add_argument('--grid', type=int, nargs=3, default=list(run.GRID), metavar=('NX', 'NY', 'NZ'),
                        help='细网格各方向单元数, 默认取优化脚本的 GRID')
    parser.add_argument('--output', type=Path, default=None,
                        help='输出目录, 默认 outputs/reanalysis_<密度标签>_hex_<网格>_m<m>_cg_jacobi; 已存在则报错')
    args = parser.parse_args(argv)
    if args.n_fine < 2:
        parser.error('--n-fine 须不小于 2')
    if any(n <= 0 for n in args.grid):
        parser.error('--grid 各方向须为正整数')
    if any(n % args.n_fine for n in args.grid):
        parser.error('--grid 各方向须能被 --n-fine 整除')
    if args.grid[0] != 6 * args.grid[1] or args.grid[1] != args.grid[2]:
        parser.error('MBB 梁要求网格各方向单元数满足 6:1:1')
    if args.density != 'uniform':
        args.density = Path(args.density)
        if not args.density.is_file():
            parser.error(f'密度文件不存在: {args.density}')
    tag = 'uniform' if args.density == 'uniform' else args.density.parent.name
    if args.output is None:
        size = 'x'.join(str(n) for n in args.grid)
        args.output = OUTPUT_DIR / f'reanalysis_{tag}_hex_{size}_m{args.n_fine}_{run.SOLVER}_{PRECONDITIONER}'
    args.output = args.output.resolve()
    args.tag = tag
    return args


def load_physical_density(args, mesh, n_cells, radius):
    """生成或读入物理密度.

    Parameters
    ----------
    args : argparse.Namespace
        ``density`` 为 ``'uniform'`` 时按优化脚本同样的过滤流程由 ``VOLFRAC`` 生成.
    mesh : HexahedronMesh
        细网格, 仅 ``uniform`` 构造过滤矩阵时用到.
    n_cells : int
        细网格单元数.
    radius : float
        过滤半径.

    Returns
    -------
    physical : (NC, ) 的物理密度.
    source : dict
        密度来源记录, 写入 ``config.json``.
    """
    if args.density == 'uniform':
        density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=radius,
                                density_location='element')
        rho = bm.full((n_cells, ), run.VOLFRAC, dtype=bm.float64, device=run.DEVICE)
        physical = density_filter.filter_design_variable(
            design_variable=rho,
            physical_density=bm.zeros((n_cells, ), dtype=bm.float64, device=run.DEVICE))[:]
        return physical, dict(kind='uniform', volfrac=run.VOLFRAC, filter_radius=radius)
    array = np.load(args.density)
    if array.ndim == 3:
        if tuple(array.shape) != tuple(args.grid):
            raise ValueError(f'密度文件形状 {array.shape} 与网格 {tuple(args.grid)} 不符')
        array = array.reshape(-1)  # 与优化脚本 bm.reshape(physical, grid) 的 C 序互逆
    if array.shape != (n_cells, ):
        raise ValueError(f'密度文件长度 {array.shape} 与单元数 {n_cells} 不符')
    if not np.all(np.isfinite(array)) or array.min() < 0 or array.max() > 1:
        raise ValueError('物理密度须为 [0, 1] 内的有限数')
    physical = bm.asarray(array, dtype=bm.float64, device=run.DEVICE)
    return physical, dict(kind='file', path=str(args.density), shape=list(np.load(args.density).shape))


def analyze_trace(trace, layout, problem, material, interpolation, cg_options, physical, stages):
    """用指定接口迹做一次分析, 返回全场位移与诊断.

    Parameters
    ----------
    trace : {'full_trace', 'linear_corner'}
    layout, problem, material, interpolation, cg_options
        与优化脚本相同的构造参数.
    physical : (NC, ) 的物理密度.
    stages : dict
        阶段计时与内存峰值写入此处, 键带 ``trace`` 前缀.

    Returns
    -------
    result : dict
        ``displacement`` (numpy 全场位移)、``interface_dofs`` (numpy, 接口未知量的全局编号)、
        ``n_interface_dofs``、``compliance_objective`` (优化脚本同一口径的柔顺度)、``solver`` 诊断;
        ``linear_corner`` 另含全局迹映射 ``trace_map`` (scipy.sparse.csr_matrix, ``P``) 与支承约束
        ``constraints`` (scipy.sparse.csr_matrix, ``C_D``).
    """
    with measure(stages, f'{trace}/分析器', label='重分析'):
        analyzer = SubstructureAnalyzer(
            layout=layout, trace=trace, chunk_size=run.CHUNK_SIZE,
            pde=problem, material=material, space_degree=run.SPACE_DEGREE,
            integration_order=run.INTEGRATION_ORDER, solve_method=run.SOLVER,
            solver_options=dict(cg_options),
            topopt_algorithm='density_based', interpolation_scheme=interpolation)
    with measure(stages, f'{trace}/装配', label='重分析'):
        stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
    with measure(stages, f'{trace}/边界条件', label='重分析'):
        body_force = analyzer.assemble_body_force_vector()
        system_matrix, system_load = analyzer.apply_bc(stiffness, body_force)
        del stiffness, body_force
        displacement = analyzer.tensor_space.function()
    with measure(stages, f'{trace}/接口求解', label='重分析'):
        info = analyzer.solve_interface(system_matrix, system_load)
        del system_matrix, system_load
    if not info['converged']:
        raise RuntimeError(f'{trace} CG 未收敛: {info["niter"]} 步, 相对残差 {info["relres"]:.3e}')
    with measure(stages, f'{trace}/回代', label='重分析'):
        analyzer.recover_displacement(displacement)
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
    compliance_objective = float(objective.fun(density=physical,
                                               state={'displacement': displacement, 'solver': info}))
    result = dict(
        displacement=bm.to_numpy(displacement[:]).astype(np.float64, copy=True),
        interface_dofs=bm.to_numpy(analyzer.interface_space.global_dofs).astype(np.int64, copy=True),
        n_interface_dofs=int(analyzer.interface_space.n_global),
        compliance_objective=compliance_objective,
        solver=dict(niter=int(info['niter']), relres=float(info['relres']), converged=bool(info['converged'])),
    )
    if trace == 'linear_corner':
        # P 的行按升序的完整接口自由度排列, 与 full_trace 的接口编号同序; 供 Galerkin 正交性核验
        _, constraints = analyzer.interface_space.constrained_conditions(problem)
        result.update(trace_map=analyzer.interface_space.global_trace_map().tocsr(),
                      constraints=constraints.tocsr())
    del analyzer
    return result


def assemble_fine_system(mesh, problem, material, interpolation, cg_options, physical, stages):
    """装配细网格刚度与载荷, 不求解.

    Returns
    -------
    K : scipy.sparse.csr_matrix
        未施加边界条件的细网格刚度.
    F : numpy.ndarray
        施加齐次 Dirichlet 条件后的载荷 (含集中力, 固定自由度为零).
    fixed : numpy.ndarray of bool
        Dirichlet 自由度掩码.
    """
    with measure(stages, 'fem/分析器', label='重分析'):
        fem = LagrangeFEMAnalyzer(
            disp_mesh=mesh, assembly_method=run.ASSEMBLY_METHOD, operator_level='fa',
            pde=problem, material=material, space_degree=run.SPACE_DEGREE,
            integration_order=run.INTEGRATION_ORDER, solve_method=run.SOLVER,
            solver_options=dict(cg_options),
            topopt_algorithm='density_based', interpolation_scheme=interpolation)
    with measure(stages, 'fem/装配', label='重分析'):
        stiffness = fem.assemble_stiff_matrix(rho_val=physical)
    with measure(stages, 'fem/边界条件', label='重分析'):
        body_force = fem.assemble_body_force_vector()
        # explicit=False 只包装不复制: 原刚度保持未消元, 残差与能量范数都要用它
        operator, load = fem.apply_bc(stiffness, body_force, explicit=False)
        fixed = bm.to_numpy(operator.is_boundary_dof).astype(bool)
        K = stiffness.to_scipy().tocsr()
        F = bm.to_numpy(load).astype(np.float64, copy=True)
    del fem, stiffness, body_force, operator, load
    return K, F, fixed


def ritz_metrics(K, F, fixed, full, corner):
    """由两侧全场位移计算 §1.3 的各项核验量.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        未施加边界条件的细网格刚度.
    F : numpy.ndarray
        约束后的载荷.
    fixed : numpy.ndarray of bool
        Dirichlet 自由度掩码.
    full, corner : dict
        ``analyze_trace`` 的返回值, 分别为 full_trace 与 linear_corner.

    Returns
    -------
    metrics : dict
        柔顺度、相对误差、能量范数、残差拆分 (linear_corner 含投影到角点子空间的接口残差
        ``interface_galerkin``) 及若干自检量.

    Raises
    ------
    ValueError
        接口编号与 ``P`` 的行不对应, 或支承约束不是落在角点上的单分量约束.
    """
    u_full = full['displacement']
    u_corner = corner['displacement']
    free = ~fixed
    interface = np.zeros(F.shape[0], dtype=bool)
    interface[full['interface_dofs']] = True  # full_trace 的接口未知量就是细网格的接口自由度
    interior_free = free & ~interface
    interface_free = free & interface
    norm_f = float(np.linalg.norm(F))

    def residual_split(u):
        r = K @ u - F
        return dict(interior=float(np.linalg.norm(r[interior_free]) / norm_f),
                    interface=float(np.linalg.norm(r[interface_free]) / norm_f),
                    fixed_displacement_max=float(np.abs(u[fixed]).max()))

    # Galerkin 正交性: 角点解的接口残差应与角点子空间正交, 即 P^T r_Gamma 在自由角点分量上为零 (到 CG 容差).
    # 只处理 C_D 每行一个单位元 (支承落在角点上) 的情形, 被约束角点分量上的反力不计入
    gamma = full['interface_dofs']
    P = corner['trace_map']
    C = corner['constraints']
    if P.shape[0] != gamma.shape[0] or np.any(np.diff(gamma) <= 0):
        raise ValueError('full_trace 的接口编号须升序且与 P 的行数一致')
    if not (np.all(np.diff(C.indptr) == 1) and np.all(C.data == 1.0)):
        raise ValueError('Galerkin 核验只处理落在角点上的单分量支承 (C_D 每行一个单位元)')
    macro_free = np.ones(P.shape[1], dtype=bool)
    macro_free[C.indices] = False
    r_corner = K @ u_corner - F
    r_corner[fixed] = 0.0
    galerkin = float(np.linalg.norm((P.T @ r_corner[gamma])[macro_free])
                     / np.linalg.norm((P.T @ F[gamma])[macro_free]))

    c_full = float(F @ u_full)
    c_corner = float(F @ u_corner)
    d = u_full - u_corner
    energy_full = float(u_full @ (K @ u_full))
    energy_diff = float(d @ (K @ d))
    eps_c = (c_full - c_corner) / c_full
    eps_e = math.sqrt(energy_diff / energy_full)
    return dict(
        compliance_full_trace=c_full, compliance_linear_corner=c_corner,
        eps_C=eps_c, sqrt_eps_C=math.copysign(math.sqrt(abs(eps_c)), eps_c), eps_E=eps_e,
        energy_full_trace=energy_full, energy_diff=energy_diff,
        # 自检: C_Gamma 应等于 U^T K U (到 CG 容差); 两侧柔顺度应与优化口径一致
        compliance_minus_energy_rel_full_trace=(c_full - energy_full) / c_full,
        compliance_objective_rel_diff=dict(
            full_trace=(full['compliance_objective'] - c_full) / c_full,
            linear_corner=(corner['compliance_objective'] - c_corner) / c_corner),
        residual=dict(full_trace=residual_split(u_full),
                      linear_corner=dict(residual_split(u_corner), interface_galerkin=galerkin)),
        n_dofs=int(F.shape[0]), n_fixed_dofs=int(fixed.sum()), n_interface_dofs=int(interface.sum()),
    )


def main(argv=None):
    """固定密度上的两种接口迹重分析.

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
    bm.set_backend(run.BACKEND)
    grid = tuple(args.grid)
    n_fine = args.n_fine
    n_sub = tuple(n // n_fine for n in grid)
    h = run.ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    domain_size = tuple(grid[d] * h for d in range(3))
    radius = run.FILTER_RADIUS_CELLS * h
    n_cells = grid[0] * grid[1] * grid[2]

    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    thread_environment = {
        name: os.environ.get(name)
        for name in ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')
    }
    cg_options = dict(precond=PRECONDITIONER, maxiter=run.CG_MAXITER, rtol=run.CG_TOLERANCE, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned', verify_residual=True)
    config: dict[str, Any] = dict(
        method='reanalysis_full_trace_vs_linear_corner',
        domain=list(domain), mesh='hex', grid=list(grid), spacing=(h, h, h),
        n_sub=n_sub, n_fine=(n_fine,) * 3, chunk_size=run.CHUNK_SIZE,
        space_degree=run.SPACE_DEGREE, integration_order=run.INTEGRATION_ORDER,
        E0=run.E0, Emin=run.EMIN, nu=run.NU, penalty=run.PENALTY,
        problem=FullMBBBeamPoint3d.__name__, load=run.LOAD,
        assembly_method=run.ASSEMBLY_METHOD, solver=run.SOLVER, cg_options=cg_options,
        backend=run.BACKEND, device=run.DEVICE,
        threads=dict(policy='external_environment_or_library_default',
                     environment=thread_environment, **blas_environment()))

    problem = FullMBBBeamPoint3d(domain=domain, P=run.LOAD, E=run.E0, nu=run.NU)
    print(f'[配置] 重分析 full_trace 对 linear_corner, 密度 {args.density}, n_sub={n_sub}, m={n_fine}, '
          f'{run.BACKEND}/{run.DEVICE}, 求解 {run.SOLVER} ({PRECONDITIONER}, tol {run.CG_TOLERANCE:g}), '
          f'网格 hex {grid} = {n_cells} 单元, 输出 {output}', flush=True)

    stages: dict = {}
    started = perf_counter()
    with measure(stages, '网格', label='重分析'):
        mesh = HexahedronMesh.from_box(list(domain), *grid, device=run.DEVICE)
        setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
        layout = StructuredSubstructureLayout(
            domain_size=domain_size, n_sub=n_sub, n_fine=(n_fine,) * 3, degree=run.SPACE_DEGREE,
            E_base=run.E0, nu=run.NU, hypothesis='3D', full_mesh=mesh)
    material = IsotropicLinearElasticMaterial(youngs_modulus=run.E0, poisson_ratio=run.NU,
                                              hypothesis='3D', device=run.DEVICE)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': run.PENALTY, 'void_youngs_modulus': run.EMIN, 'target_variables': ['E']})

    with measure(stages, '密度', label='重分析'):
        physical, source = load_physical_density(args, mesh, n_cells, radius)
    config.update(density=source, volume_fraction=float(bm.mean(physical)))
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # 先做 full_trace 并释放其接口系统, 再做 linear_corner, 最后装配细网格刚度, 避免大对象同时驻留
    full = analyze_trace('full_trace', layout, problem, material, interpolation, cg_options, physical, stages)
    corner = analyze_trace('linear_corner', layout, problem, material, interpolation, cg_options, physical, stages)
    K, F, fixed = assemble_fine_system(mesh, problem, material, interpolation, cg_options, physical, stages)
    with measure(stages, '核验量', label='重分析'):
        metrics = ritz_metrics(K, F, fixed, full, corner)
    if metrics['n_interface_dofs'] != full['n_interface_dofs']:
        raise RuntimeError('full_trace 接口未知量数与细网格接口自由度数不符')

    report = dict(
        density=source, grid=list(grid), n_sub=n_sub, n_fine=n_fine,
        volume_fraction=config['volume_fraction'],
        interface_dofs=dict(full_trace=full['n_interface_dofs'], linear_corner=corner['n_interface_dofs']),
        solver=dict(full_trace=full['solver'], linear_corner=corner['solver']),
        compliance_objective=dict(full_trace=full['compliance_objective'],
                                  linear_corner=corner['compliance_objective']),
        **metrics,
        stages=stages, total_seconds=perf_counter() - started,
        peak_rss_gib=max(stage['peak_gib'] for stage in stages.values()),
    )
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                        encoding='utf-8')
    np.save(output / 'displacement_full_trace.npy', full['displacement'])
    np.save(output / 'displacement_linear_corner.npy', corner['displacement'])
    res = metrics['residual']
    print(f'[完成] C_Gamma={metrics["compliance_full_trace"]:.6f}, C_C={metrics["compliance_linear_corner"]:.6f}, '
          f'eps_C={metrics["eps_C"]:.4e}, sqrt(eps_C)={metrics["sqrt_eps_C"]:.4e}, eps_E={metrics["eps_E"]:.4e}; '
          f'残差 linear_corner 内部 {res["linear_corner"]["interior"]:.2e} / 接口 {res["linear_corner"]["interface"]:.2e}'
          f' / 投影到角点子空间 {res["linear_corner"]["interface_galerkin"]:.2e}, '
          f'full_trace 内部 {res["full_trace"]["interior"]:.2e} / 接口 {res["full_trace"]["interface"]:.2e}; '
          f'CG {full["solver"]["niter"]} / {corner["solver"]["niter"]} 步, 结果保存到 {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
