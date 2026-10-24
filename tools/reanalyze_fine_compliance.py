"""把三维 MBB 梁拓扑优化的最终物理密度放到全尺度有限元模型上重分析, 得到细网格柔顺度 C_Fine.

各方法在优化中报告的柔顺度 C_opt 由各自的分析模型算出 (全尺度 FEM、精确子结构、PIML 子结构),
不同模型之间不能直接比较. 本工具读取各运行目录的 ``density_final.npy`` (形状 ``(nx, ny, nz)`` 的
物理密度, C 序, 与 ``HexahedronMesh.from_box`` 的单元编号一致), 在同一个全尺度 FEM 模型上各做一次
前向分析, 不再过滤, 不做优化. 模型与修正 SIMP 插值同全尺度 FEM 入口
``experiments/topopt_full_scale_fem/run_full_scale_fem.py``; 运行时与每份设计的 ``config.json``
逐项核对, 不一致即报错.

``--reference`` 指定的设计须是全尺度 FEM 自身的优化结果, 其 C_Fine 应复现自身 C_opt, 作为口径自检;
相对差超过 ``--self-check-rtol`` 时仍写出结果, 但以退出码 1 结束.

Examples
--------
::

    python tools/reanalyze_fine_compliance.py \\
        --design fem=experiments/topopt_full_scale_fem/results/pytorch_cuda_ea_hex_390x65x65_cg_jacobi \\
        --design exact=experiments/topopt_exact_substructure/results/pytorch_cuda_linear_corner_hex_390x65x65_m5_cg_jacobi_t1 \\
        --reference fem --output experiments/reports/results/fine_compliance_390.json
"""

from __future__ import annotations

import os
import sys

# BLAS 与 OpenMP 在加载时读取线程数, 须在导入 numpy/torch 之前写入; 单线程, 与各优化入口相同
THREAD_VARIABLES = ('BLIS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')
if __name__ == '__main__':
    for _name in THREAD_VARIABLES:
        os.environ[_name] = '1'

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.core import limit_cuda_memory
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective

# 模型参数, 与全尺度 FEM 入口相同; 每份设计的 config.json 须与之一致 (见 MODEL_KEYS)
ELEMENT_SIZE = 1.0
LOAD = -1.0
E0 = 1.0
EMIN = 1.0e-7
NU = 0.3
PENALTY = 3.0
SPACE_DEGREE = 1
INTEGRATION_ORDER = 2
ASSEMBLY_METHOD = 'fast'
PROBLEM = FullMBBBeamPoint3d.__name__
MODEL_KEYS = dict(E0=E0, Emin=EMIN, nu=NU, penalty=PENALTY, load=LOAD, problem=PROBLEM,
                  space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
                  spacing=[ELEMENT_SIZE] * 3)

# 求解: 只解一次, 容差比优化时的 1e-6 严, 使 C_Fine 基本不受求解误差影响; 冷启动, 步数上限放宽
CG_TOLERANCE = 1.0e-8
CG_MAXITER = 100000
SELF_CHECK_RTOL = 1.0e-5
CUDA_MEMORY_LIMIT_GIB = 14.0


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数.

    Returns
    -------
    argparse.Namespace
        含 ``design`` (名称到运行目录的有序列表)、``reference``、``output``、``assemble_level``、
        ``backend``、``device``、``gpu_memory_limit``、``cg_tol``、``cg_maxiter``、``self_check_rtol``.
    """
    parser = argparse.ArgumentParser(description='把最终物理密度放到全尺度 FEM 上重分析, 得到 C_Fine')
    parser.add_argument('--design', action='append', required=True, metavar='NAME=RUN_DIR',
                        help='设计名与运行目录, 目录内须有 density_final.npy、config.json、summary.json; 可重复')
    parser.add_argument('--reference', required=True,
                        help='口径自检用的设计名, 须为全尺度 FEM 自身的优化结果')
    parser.add_argument('--output', type=Path, required=True, help='输出 JSON 路径, 已存在则拒绝覆盖')
    parser.add_argument('--assemble-level', choices=('ea', 'fa'), default='ea',
                        help='刚度算子层级 (默认 ea, 与全尺度 FEM 的 390 档运行相同)')
    parser.add_argument('--backend', choices=('numpy', 'pytorch'), default='pytorch', help='计算后端 (默认 pytorch)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='计算设备 (默认 cuda)')
    parser.add_argument('--gpu-memory-limit', type=float, default=None, metavar='GIB',
                        help=f'GPU 显存上限 (GiB), 只用于 --device cuda (默认 {CUDA_MEMORY_LIMIT_GIB})')
    parser.add_argument('--cg-tol', type=float, default=CG_TOLERANCE,
                        help=f'CG 相对右端项容差 (默认 {CG_TOLERANCE:g})')
    parser.add_argument('--cg-maxiter', type=int, default=CG_MAXITER, help=f'CG 最大步数 (默认 {CG_MAXITER})')
    parser.add_argument('--self-check-rtol', type=float, default=SELF_CHECK_RTOL,
                        help=f'参考设计 C_Fine 与自身 C_opt 的相对差上限 (默认 {SELF_CHECK_RTOL:g})')
    args = parser.parse_args(argv)
    designs = []
    for item in args.design:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            parser.error(f'--design 须为 NAME=RUN_DIR, 得到 {item!r}')
        if name in {n for n, _ in designs}:
            parser.error(f'--design 名称重复: {name}')
        run_dir = Path(path)
        for file in ('density_final.npy', 'config.json', 'summary.json'):
            if not (run_dir / file).is_file():
                parser.error(f'{run_dir} 中缺少 {file}')
        designs.append((name, run_dir))
    args.design = designs
    if args.reference not in {n for n, _ in designs}:
        parser.error(f'--reference {args.reference} 不在 --design 的名称中')
    if args.output.exists():
        parser.error(f'拒绝覆盖已有输出: {args.output}')
    if args.device == 'cuda':
        if args.backend != 'pytorch':
            parser.error('--device cuda 须配 --backend pytorch')
        if args.gpu_memory_limit is None:
            args.gpu_memory_limit = CUDA_MEMORY_LIMIT_GIB
        elif args.gpu_memory_limit <= 0:
            parser.error(f'--gpu-memory-limit 须为正数, 得到 {args.gpu_memory_limit}')
    elif args.gpu_memory_limit is not None:
        parser.error('--gpu-memory-limit 只用于 --device cuda')
    if not args.cg_tol > 0 or args.cg_maxiter < 1 or not args.self_check_rtol > 0:
        parser.error('--cg-tol、--self-check-rtol 须为正数, --cg-maxiter 须为正整数')
    return args


def load_design(run_dir: Path) -> tuple[np.ndarray, dict, dict]:
    """读取一份设计的物理密度、运行配置与优化汇总, 并核对模型参数.

    Parameters
    ----------
    run_dir : Path
        运行目录.

    Returns
    -------
    density : ndarray
        形状 ``(nx, ny, nz)`` 的物理密度.
    config : dict
        ``config.json`` 的内容.
    summary : dict
        ``summary.json`` 的内容.

    Raises
    ------
    ValueError
        模型参数与本工具不一致, 或密度形状与网格不符.
    """
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    mismatched = {key: (config.get(key), value) for key, value in MODEL_KEYS.items() if config.get(key) != value}
    if mismatched:
        raise ValueError(f'{run_dir} 的模型参数与本工具不一致 (运行值, 本工具值): {mismatched}')
    density = np.load(run_dir / 'density_final.npy')
    if density.shape != tuple(config['grid']):
        raise ValueError(f'{run_dir} 的密度形状 {density.shape} 与网格 {config["grid"]} 不符')
    return density, config, summary


def sha256(path: Path) -> str:
    """返回文件的 sha256 十六进制摘要."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    """命令行入口.

    Returns
    -------
    status : int
        参考设计通过口径自检时为 0, 否则为 1.
    """
    args = parse_args(argv)
    backend, device = args.backend, args.device
    cuda = device == 'cuda'
    bm.set_backend(backend)
    # 须在首次分配显存之前设置
    gpu_memory = limit_cuda_memory(device, args.gpu_memory_limit) if cuda else None

    designs = [(name, run_dir, *load_design(run_dir)) for name, run_dir in args.design]
    grids = {tuple(config['grid']) for _, _, _, config, _ in designs}
    if len(grids) != 1:
        raise ValueError(f'各设计的网格须相同, 得到 {sorted(grids)}')
    grid = grids.pop()
    h = ELEMENT_SIZE
    domain = (0.0, grid[0] * h, 0.0, grid[1] * h, 0.0, grid[2] * h)
    n_cells = grid[0] * grid[1] * grid[2]

    cg_options = dict(precond='jacobi', maxiter=args.cg_maxiter, rtol=args.cg_tol, atol=0.0,
                      rtol_reference='rhs', norm_type='unpreconditioned', verify_residual=True)
    print(f'[配置] fem/{args.assemble_level}, {backend}/{device}, 网格 hex {grid} = {n_cells} 单元, '
          f'cg (jacobi, tol {args.cg_tol:g}, maxiter {args.cg_maxiter}), 设计 {[n for n, *_ in designs]}, '
          f'参考 {args.reference}', flush=True)

    started = perf_counter()
    problem = FullMBBBeamPoint3d(domain=domain, P=LOAD, E=E0, nu=NU)
    mesh = HexahedronMesh.from_box(list(domain), *grid, device=device)
    setattr(mesh, 'meshdata', {'nx': grid[0], 'ny': grid[1], 'nz': grid[2], 'hx': h, 'hy': h, 'hz': h})
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D', device=device)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
    )
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material,
        space_degree=SPACE_DEGREE, integration_order=INTEGRATION_ORDER,
        assembly_method=ASSEMBLY_METHOD, operator_level=args.assemble_level, solve_method='cg',
        solver_options=dict(cg_options),
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
    )
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
    # FA 在 K 的值数组上原地对称消元; EA 只包一层 ConstrainedOperator
    bc_options = dict(inplace=True) if args.assemble_level == 'fa' else {}
    setup_seconds = perf_counter() - started

    records = []
    for name, run_dir, density, config, summary in designs:
        started = perf_counter()
        # density_final.npy 是过滤后的物理密度, 按 C 序展平即单元编号, 直接作为物理密度输入
        physical = bm.asarray(density.reshape(-1), dtype=bm.float64, device=device)
        stiffness = analyzer.assemble_stiff_matrix(rho_val=physical)
        system_matrix, system_load = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector(), **bc_options)
        del stiffness
        displacement = analyzer.tensor_space.function()
        _, solver_info = analyzer.solve_system(system_matrix, system_load, displacement,
                                               x0=analyzer.prescribed_solution)
        del system_matrix, system_load
        if not solver_info['converged']:
            raise RuntimeError(f'{name}: CG 未收敛, {solver_info["niter"]} 步, 相对残差 {solver_info["relres"]:.3e}')
        compliance = float(objective.fun(density=physical, state={'displacement': displacement, 'solver': solver_info}))
        if not math.isfinite(compliance) or compliance <= 0:
            raise FloatingPointError(f'{name}: 柔顺度须为有限正数, 得到 {compliance}')
        c_opt = float(summary['compliance'])
        records.append(dict(
            name=name, run_dir=str(run_dir), method=config.get('method'),
            density_sha256=sha256(run_dir / 'density_final.npy'),
            volume_fraction=float(density.mean()), c_opt=c_opt, c_fine=compliance,
            model_bias=(c_opt - compliance) / compliance,
            cg_iterations=int(solver_info['niter']), cg_relative_residual=float(solver_info['relres']),
            seconds=perf_counter() - started))
        print(f'[{name}] C_Fine={compliance:.6f}, C_opt={c_opt:.6f}, 模型偏差 {(c_opt - compliance) / compliance:+.4%}, '
              f'CG {solver_info["niter"]} 步, 相对残差 {solver_info["relres"]:.2e}, '
              f'{records[-1]["seconds"]:.1f} s', flush=True)
        del physical, displacement

    reference = next(record for record in records if record['name'] == args.reference)
    for record in records:
        record['relative_to_reference'] = (record['c_fine'] - reference['c_fine']) / reference['c_fine']
    self_check = abs(reference['c_fine'] - reference['c_opt']) / reference['c_opt']
    passed = self_check <= args.self_check_rtol

    result: dict[str, Any] = dict(
        created=datetime.now().isoformat(timespec='seconds'),
        grid=list(grid), model=MODEL_KEYS, assemble_level=args.assemble_level, assembly_method=ASSEMBLY_METHOD,
        backend=backend, device=device, gpu_memory=gpu_memory, cg_options=cg_options,
        reference=args.reference, self_check=dict(relative_difference=self_check, rtol=args.self_check_rtol,
                                                  passed=passed),
        setup_seconds=setup_seconds, designs=records)
    if backend == 'pytorch':
        import torch
        result['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                               device_name=torch.cuda.get_device_name(device) if cuda else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    print(f'[自检] {args.reference}: C_Fine 与自身 C_opt 相对差 {self_check:.2e} '
          f'({"通过" if passed else "未通过"}, 上限 {args.self_check_rtol:g})', flush=True)
    for record in records:
        print(f'[汇总] {record["name"]}: C_Fine={record["c_fine"]:.6f}, '
              f'相对 {args.reference} {record["relative_to_reference"]:+.4%}', flush=True)
    print(f'[完成] 结果写入 {args.output}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
