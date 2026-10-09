"""SubstructureAnalyzer: 与细网格 FA 分析器的代数等价性及与目标函数的协作."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import LagrangeFEMAnalyzer, SubstructureAnalyzer
from soptx.fem.substructure import (
    GlobalAssembler,
    StructuredSubstructureLayout,
    build_interface_space,
    build_substructures,
    solve_constrained_system,
)
from soptx.fem.substructure.streaming import assemble_exact_interface_system
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeam3d
from soptx.sparse import COOTensor
from soptx.topology.constraints import VolumeConstraint
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective

E0, EMIN, NU, PENALTY = 1.0, 1.0e-7, 0.3, 3.0
N_SUB, N_FINE = (6, 1, 1), 2
GRID = tuple(n * N_FINE for n in N_SUB)
DOMAIN = (0.0, float(GRID[0]), 0.0, float(GRID[1]), 0.0, float(GRID[2]))


@pytest.fixture
def setup():
    """小网格 MBB 梁: 布局、问题、材料、插值方案与一组随机物理密度."""
    bm.set_backend('numpy')
    layout = StructuredSubstructureLayout(
        domain_size=tuple(float(n) for n in GRID), n_sub=N_SUB, n_fine=(N_FINE, ) * 3,
        degree=1, E_base=E0, nu=NU, hypothesis='3D',
    )
    problem = FullMBBBeam3d(domain=DOMAIN, P=-1.0, E=E0, nu=NU,
                            support='end_corners', load_subdivisions=(GRID[0], GRID[2]))
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                              enable_logging=False)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
        enable_logging=False,
    )
    rng = np.random.default_rng(7)
    density = bm.asarray(rng.uniform(0.2, 1.0, size=int(np.prod(GRID))), dtype=bm.float64)
    return layout, problem, material, interpolation, density


def _force_numpy(analyzer) -> np.ndarray:
    """细网格载荷向量; 分析器把集中力装配为稠密向量, 稀疏形式不在本测试范围内."""
    force = analyzer.force_vector
    assert not isinstance(force, COOTensor)
    return np.asarray(bm.to_numpy(force))


def _fa_analyzer(layout, problem, material, interpolation):
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID)
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material, space_degree=1, integration_order=2,
        assembly_method='fast', operator_level='fa', solve_method='scipy',
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
        enable_logging=False, reference_classes=1,
    )


def _sub_analyzer(layout, problem, material, interpolation, trace, **kwargs):
    return SubstructureAnalyzer(
        layout=layout, pde=problem, material=material, trace=trace, chunk_size=4, space_degree=1, integration_order=2,
        topopt_algorithm='density_based', interpolation_scheme=interpolation, enable_logging=False,
        **kwargs,
    )


def test_full_trace_matches_fine_grid_fa(setup) -> None:
    """full_trace 与细网格 FA 直接法: 位移、柔顺度与灵敏度一致到求解器精度."""
    layout, problem, material, interpolation, density = setup
    fa = _fa_analyzer(layout, problem, material, interpolation)
    sub = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='scipy')
    state_fa = fa.solve_state(rho_val=density)
    state_sub = sub.solve_state(rho_val=density)
    u_fa, u_sub = bm.to_numpy(state_fa['displacement'][:]), bm.to_numpy(state_sub['displacement'][:])
    assert np.max(np.abs(u_fa - u_sub)) <= 1e-9 * np.max(np.abs(u_fa))
    obj_fa, obj_sub = (ComplianceObjective(analyzer=a, state_variable='u', diff_mode='manual', enable_logging=False)
                       for a in (fa, sub))
    c_fa, c_sub = float(obj_fa.fun(density=density, state=state_fa)), float(obj_sub.fun(density=density, state=state_sub))
    assert c_sub == pytest.approx(c_fa, rel=1e-9)
    dc_fa = bm.to_numpy(obj_fa.jac(density=density, state=state_fa))
    dc_sub = bm.to_numpy(obj_sub.jac(density=density, state=state_sub))
    assert np.max(np.abs(dc_fa - dc_sub)) <= 1e-9 * np.max(np.abs(dc_fa))
    # 柔顺度等于接口载荷与接口未知量的内积, 与细网格 F^T u 同值
    assert state_sub['interface_displacement'] is not None
    assert float(np.dot(_force_numpy(sub), u_sub)) == pytest.approx(c_sub, rel=1e-12)


def test_injected_interpolation_matches_prototype_simp(setup) -> None:
    """注入插值方案后的接口系统与参考子结构自带的修正 SIMP 路径逐位相同 (linear_corner)."""
    layout, problem, material, interpolation, density = setup
    sub = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='scipy')
    system = sub.assemble_stiff_matrix(rho_val=density)
    # 旧路径: 参考子结构按 penal 与 rho_min = Emin / E0 自行插值
    assembler = GlobalAssembler(layout)
    prototype, sub_meshes, _ = build_substructures(assembler, integration_order=2, penal=PENALTY, rho_min=EMIN / E0)
    space = build_interface_space(kind='linear_corner', assembler=assembler, sub_meshes=sub_meshes, prototype=prototype)
    rho_cells = prototype.grid_to_cell_field(layout.split_global_cell_field(density))
    legacy = assemble_exact_interface_system(prototype, rho_cells, space, chunk_size=4)
    a, b = system.stiffness.to_scipy().tocsr(), legacy.stiffness.to_scipy().tocsr()
    assert a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    assert np.max(np.abs(a.data - b.data)) <= 1e-14 * np.max(np.abs(b.data))
    # 同一约束系统求解, 位移与旧路径一致
    state = sub.solve_state(rho_val=density)
    load, constraints = space.constrained_conditions(problem)
    legacy_q = solve_constrained_system(system=legacy, load=load, constraints=constraints, solver='scipy').displacement
    assert np.max(np.abs(bm.to_numpy(state['interface_displacement']) - bm.to_numpy(legacy_q))) <= 1e-12


def test_linear_corner_objective_and_constraint_work(setup) -> None:
    """linear_corner 下目标函数、体积约束与 CG 求解器都能以子类协作."""
    layout, problem, material, interpolation, density = setup
    direct = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='scipy')
    state_direct = direct.solve_state(rho_val=density)
    load_norm = float(bm.linalg.norm(direct._interface_load))
    cg = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='cg',
                       solver_options=dict(precond='jacobi', maxiter=2000, rtol=0.0, atol=1e-10 * load_norm))
    state_cg = cg.solve_state(rho_val=density)
    assert state_cg['solver']['converged'] and state_cg['solver']['niter'] > 0
    u_d, u_c = bm.to_numpy(state_direct['displacement'][:]), bm.to_numpy(state_cg['displacement'][:])
    assert np.max(np.abs(u_d - u_c)) <= 1e-8 * np.max(np.abs(u_d))
    # 热启动: 以直接法的接口位移为初值, CG 一步内即满足容差
    warm = cg.solve_state(rho_val=density, x0=state_direct['interface_displacement'])
    assert warm['solver']['niter'] <= 1
    objective = ComplianceObjective(analyzer=direct, state_variable='u', diff_mode='manual', enable_logging=False)
    constraint = VolumeConstraint(analyzer=direct, volume_fraction=0.5, diff_mode='manual', enable_logging=False)
    c = float(objective.fun(density=density, state=state_direct))
    assert np.isfinite(c) and c > 0
    dc = bm.to_numpy(objective.jac(density=density, state=state_direct))
    assert dc.shape == (int(np.prod(GRID)), ) and np.all(dc <= 0)
    assert float(constraint.fun(density=density)) == pytest.approx(float(np.mean(bm.to_numpy(density))) - 0.5)
    assert float(np.dot(_force_numpy(direct), u_d)) == pytest.approx(c, rel=1e-12)


def test_prototype_default_coefficient_is_simp(setup) -> None:
    """未注入系数函数时 stiffness_coefficient 退化为参考子结构自带的修正 SIMP."""
    layout, *_ = setup
    prototype, _, _ = build_substructures(GlobalAssembler(layout), integration_order=2, penal=PENALTY, rho_min=0.1)
    rho = bm.asarray(np.array([[0.2, 0.5], [0.9, 1.0]]))
    expected = 0.1 + 0.9 * rho ** PENALTY
    assert np.array_equal(bm.to_numpy(prototype.stiffness_coefficient(rho)), bm.to_numpy(expected))
    prototype.coefficient_function = lambda r: 2.0 * r
    assert np.array_equal(bm.to_numpy(prototype.stiffness_coefficient(rho)), 2.0 * bm.to_numpy(rho))


def test_full_trace_multigrid_preconditioner(setup) -> None:
    """full_trace 下 precond='mg': 与直接法一致, CG 步数远少于 Jacobi; linear_corner 下拒绝."""
    layout, problem, material, interpolation, density = setup
    direct = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='scipy')
    u_direct = bm.to_numpy(direct.solve_state(rho_val=density)['displacement'][:])
    load_norm = float(bm.linalg.norm(direct._interface_load))
    runs = {}
    for precond in ('jacobi', 'mg'):
        cg = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='cg',
                           solver_options=dict(precond=precond, maxiter=5000, rtol=0.0, atol=1e-9 * load_norm))
        state = cg.solve_state(rho_val=density)
        assert state['solver']['converged']
        u = bm.to_numpy(state['displacement'][:])
        assert np.max(np.abs(u - u_direct)) <= 1e-7 * np.max(np.abs(u_direct))
        runs[precond] = state['solver']['niter']
    assert runs['mg'] < runs['jacobi'] / 3
    corner = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='cg',
                           solver_options=dict(precond='mg', maxiter=100, rtol=0.0, atol=1e-6 * load_norm))
    with pytest.raises(Exception):
        corner.solve_state(rho_val=density)
