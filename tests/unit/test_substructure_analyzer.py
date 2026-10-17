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
from soptx.problems import FullMBBBeamLine3d
from soptx.problems.loads import PointForceLoad
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
        full_mesh=HexahedronMesh.from_box(list(DOMAIN), *GRID),
    )
    problem = FullMBBBeamLine3d(domain=DOMAIN, P=-1.0, E=E0, nu=NU)
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D')
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
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
    mesh = layout.full_mesh
    return LagrangeFEMAnalyzer(
        disp_mesh=mesh, pde=problem, material=material, space_degree=1, integration_order=2,
        assembly_method='fast', operator_level='fa', solve_method='scipy',
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
    )


def _sub_analyzer(layout, problem, material, interpolation, trace, **kwargs):
    return SubstructureAnalyzer(
        layout=layout, pde=problem, material=material, trace=trace, chunk_size=4, space_degree=1, integration_order=2,
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
        **kwargs,
    )


def test_cg_constraints_are_prepared_in_apply_bc(setup, monkeypatch):
    """apply_bc 已生成受约束算子, 求解与回代不再次准备约束."""
    from soptx.fem.substructure import solve as interface_solve
    from soptx.fem.substructure.solve import PreparedFixedSystem

    layout, problem, material, interpolation, density = setup
    sub = _sub_analyzer(
        layout, problem, material, interpolation, 'full_trace', solve_method='cg',
        solver_options={'precond': 'jacobi', 'rtol': 1e-10, 'atol': 0., 'maxiter': 20000},
    )
    stiffness = sub.assemble_stiff_matrix(density)
    prepared, rhs = sub.apply_bc(stiffness, sub.assemble_body_force_vector())
    assert isinstance(prepared, PreparedFixedSystem)
    np.testing.assert_array_equal(bm.to_numpy(rhs)[prepared.fixed], 0.)

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError('求解阶段重复准备约束')

    monkeypatch.setattr(interface_solve, '_prepare_fixed_system', unexpected_preparation)
    monkeypatch.setattr(interface_solve._FixedDofOperator, '__init__', unexpected_preparation)
    out = bm.zeros((layout.full_mesh.number_of_nodes() * 3,), dtype=bm.float64)
    recovered, info = sub.solve_system(prepared, rhs, out)
    fa = _fa_analyzer(layout, problem, material, interpolation)
    reference = fa.solve_state(rho_val=density)['displacement']
    np.testing.assert_allclose(bm.to_numpy(recovered), bm.to_numpy(reference[:]), rtol=1e-7, atol=1e-7)
    assert info['converged']


@pytest.mark.parametrize('general', [False, True])
def test_direct_constraints_are_prepared_in_apply_bc(setup, monkeypatch, general):
    """直接法在 apply_bc 完成消元或乘子组装, solve_system 仅求解与恢复."""
    import soptx.fem.substructure.solve as interface_solve
    from scipy.sparse import csr_matrix, vstack
    from soptx.fem.substructure.solve import PreparedDirectSystem

    layout, problem, material, interpolation, density = setup
    sub = _sub_analyzer(
        layout, problem, material, interpolation, 'full_trace', solve_method='scipy',
    )
    stiffness = sub.assemble_stiff_matrix(density)
    if general:
        constraints = sub._constraints
        constraints = constraints.to_scipy() if hasattr(constraints, 'to_scipy') else csr_matrix(constraints)
        free = np.setdiff1d(np.arange(constraints.shape[1]), constraints.indices)
        coupling = csr_matrix(([1., -1.], ([0, 0], free[:2])), shape=(1, constraints.shape[1]))
        sub._constraints = vstack((constraints, coupling), format='csr')
    prepared, rhs = sub.apply_bc(stiffness, sub.assemble_body_force_vector())
    assert isinstance(prepared, PreparedDirectSystem)
    assert (prepared.free is None) == general
    reference = solve_constrained_system(
        stiffness, sub._interface_load, sub._constraints, solver='scipy',
    )

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError('求解阶段重复准备约束')

    for name in ('qr', 'bmat', '_interface_constraint_data', 'solve_interface_system'):
        monkeypatch.setattr(interface_solve, name, unexpected_preparation)
    out = bm.zeros((layout.full_mesh.number_of_nodes() * 3,), dtype=bm.float64)
    recovered, info = sub.solve_system(prepared, rhs, out)
    np.testing.assert_allclose(sub._interface_displacement, reference.displacement, rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(bm.to_numpy(recovered)))
    assert info['constraint_relres'] < 1e-10
    assert info['relres'] < 1e-10


@pytest.mark.parametrize('configured, requested', [('cg', 'scipy'), ('scipy', 'cg')])
def test_solver_override_reaches_boundary_preparation(setup, monkeypatch, configured, requested):
    """单次状态求解切换 solver 时, apply_bc 按实际求解器准备系统."""
    import soptx.fem.analyzers.substructure_analyzer as analyzer_module

    layout, problem, material, interpolation, density = setup
    sub = _sub_analyzer(
        layout, problem, material, interpolation, 'full_trace', solve_method=configured,
        solver_options={'precond': 'jacobi', 'rtol': 1e-8, 'atol': 0., 'maxiter': 20000},
    )
    prepare = analyzer_module.prepare_constrained_system
    seen = []

    def record_preparation(*args, **kwargs):
        seen.append(kwargs['solver'])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(analyzer_module, 'prepare_constrained_system', record_preparation)
    state = sub.solve_state(rho_val=density, solver=requested)
    assert seen == [requested]
    assert state['solver']['name'] == requested
    assert state['solver']['converged']


@pytest.mark.parametrize('point', [None, (6.0, 2.0, 0.25)])
def test_full_trace_matches_fine_grid_fa(setup, point) -> None:
    """full_trace 与细网格 FA 直接法: 位移、柔顺度与灵敏度一致到求解器精度."""
    layout, problem, material, interpolation, density = setup
    if point is not None:
        problem = _PointMBB(point)
    fa = _fa_analyzer(layout, problem, material, interpolation)
    sub = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='scipy')
    state_fa = fa.solve_state(rho_val=density)
    state_sub = sub.solve_state(rho_val=density)
    u_fa, u_sub = bm.to_numpy(state_fa['displacement'][:]), bm.to_numpy(state_sub['displacement'][:])
    assert np.max(np.abs(u_fa - u_sub)) <= 1e-9 * np.max(np.abs(u_fa))
    obj_fa, obj_sub = (ComplianceObjective(analyzer=a, state_variable='u', diff_mode='manual')
                       for a in (fa, sub))
    c_fa, c_sub = float(obj_fa.fun(density=density, state=state_fa)), float(obj_sub.fun(density=density, state=state_sub))
    assert c_sub == pytest.approx(c_fa, rel=1e-9)
    dc_fa = bm.to_numpy(obj_fa.jac(density=density, state=state_fa))
    dc_sub = bm.to_numpy(obj_sub.jac(density=density, state=state_sub))
    assert np.max(np.abs(dc_fa - dc_sub)) <= 1e-9 * np.max(np.abs(dc_fa))
    # 柔顺度等于接口载荷与接口未知量的内积, 与细网格 F^T u 同值
    assert state_sub['interface_displacement'] is not None
    assert float(np.dot(_force_numpy(sub), u_sub)) == pytest.approx(c_sub, rel=1e-12)


class _PointMBB(FullMBBBeamLine3d):
    """用指定物理点力检查两条求解路径的载荷契约."""

    def __init__(self, point):
        super().__init__(domain=DOMAIN)
        self.point = point

    def loads(self):
        """返回未拆分的物理点力."""
        return (PointForceLoad(self.point, (0., -1., 0.)),)


def test_linear_corner_load_preserves_fine_grid_virtual_work(setup) -> None:
    """角点载荷由细网格形函数载荷作迹投影, 保持接口虚功."""
    layout, _, material, interpolation, _ = setup
    problem = _PointMBB((6.0, 2.0, 0.25))
    fa = _fa_analyzer(layout, problem, material, interpolation)
    sub = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='scipy')
    fine_load = np.asarray(bm.to_numpy(fa.assemble_external_load()))
    np.testing.assert_allclose(bm.to_numpy(sub.assemble_external_load()), fine_load, atol=1e-14)
    reduced = bm.to_numpy(sub._interface_load).reshape(-1, 3)
    loaded = np.flatnonzero(reduced[:, 1])
    coordinates = bm.to_numpy(layout.macro_node_coordinates())
    np.testing.assert_allclose(coordinates[loaded], [[6., 2., 0.], [6., 2., 2.]])
    np.testing.assert_allclose(reduced[loaded, 1], [-0.875, -0.125], atol=1e-14)
    # 约化载荷对刚体平移做功等于物理总力.
    q = np.zeros(sub._interface_load.shape)
    q[1::3] = 1.0
    assert float(np.dot(bm.to_numpy(sub._interface_load), q)) == pytest.approx(-1.)


@pytest.mark.parametrize('trace', ['full_trace', 'linear_corner'])
def test_interior_point_force_remains_explicitly_unsupported(setup, trace) -> None:
    """形函数分配涉及子结构内部自由度时不能静默丢弃右端项."""
    layout, _, material, interpolation, _ = setup
    with pytest.raises(ValueError, match='内部载荷'):
        _sub_analyzer(layout, _PointMBB((0.75, 0.75, 0.75)), material, interpolation,
                      trace, solve_method='scipy')


def test_injected_interpolation_matches_prototype_simp(setup) -> None:
    """注入插值方案后的接口系统与参考子结构自带的修正 SIMP 路径数值一致 (linear_corner)."""
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
    # 直接角点缩聚改变浮点运算顺序, 位移误差按解的尺度验收.
    delta = bm.to_numpy(state['interface_displacement']) - bm.to_numpy(legacy_q)
    assert np.linalg.norm(delta) / np.linalg.norm(bm.to_numpy(legacy_q)) <= 1e-10


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
    objective = ComplianceObjective(analyzer=direct, state_variable='u', diff_mode='manual')
    constraint = VolumeConstraint(analyzer=direct, volume_fraction=0.5, diff_mode='manual')
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


def test_full_trace_gmg_preconditioner(setup) -> None:
    """full_trace 下 precond='gmg': 与直接法一致, CG 步数远少于 Jacobi; linear_corner 下拒绝."""
    layout, problem, material, interpolation, density = setup
    direct = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='scipy')
    u_direct = bm.to_numpy(direct.solve_state(rho_val=density)['displacement'][:])
    load_norm = float(bm.linalg.norm(direct._interface_load))
    runs = {}
    for precond in ('jacobi', 'gmg'):
        cg = _sub_analyzer(layout, problem, material, interpolation, 'full_trace', solve_method='cg',
                           solver_options=dict(precond=precond, maxiter=5000, rtol=0.0, atol=1e-9 * load_norm))
        state = cg.solve_state(rho_val=density)
        assert state['solver']['converged']
        u = bm.to_numpy(state['displacement'][:])
        assert np.max(np.abs(u - u_direct)) <= 1e-7 * np.max(np.abs(u_direct))
        runs[precond] = state['solver']['niter']
    assert runs['gmg'] < runs['jacobi'] / 3
    corner = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner', solve_method='cg',
                           solver_options=dict(precond='gmg', maxiter=100, rtol=0.0, atol=1e-6 * load_norm))
    with pytest.raises(Exception):
        corner.solve_state(rho_val=density)


def test_corner_cached_recovery_matches_legacy_and_refreshes(setup, monkeypatch):
    """两次密度更新均与完整缩聚一致, 缓存回代不重新装配."""
    from soptx.fem.substructure.streaming import iter_exact_internal_displacement_batches
    from soptx.fem.substructure.recovery import recover_full_displacement_batches

    layout, problem, material, interpolation, density = setup
    sub = _sub_analyzer(layout, problem, material, interpolation, 'linear_corner',
                        solve_method='scipy')
    for rho in (density, density[::-1].copy()):
        current = sub.assemble_stiff_matrix(rho)
        legacy = assemble_exact_interface_system(
            sub.prototype, sub._density_cells, sub.interface_space, chunk_size=4)
        np.testing.assert_allclose(current.stiffness.to_scipy().toarray(),
                                   legacy.stiffness.to_scipy().toarray(), rtol=1e-11, atol=1e-13)
        prepared, rhs = sub.apply_bc(current, sub.assemble_body_force_vector())
        out = sub.tensor_space.function()
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('缓存回代不应重新装配局部刚度')
            patch.setattr(sub.prototype, 'iter_local_stiffness_batches', forbidden)
            sub.solve_system(prepared, rhs, out)
        q = sub.interface_displacement[sub.interface_space.local_dofs]
        old = recover_full_displacement_batches(
            layout, sub.sub_meshes, iter_exact_internal_displacement_batches(
                sub.prototype, sub._density_cells, q, sub.interface_space.trace_basis,
                chunk_size=4))
        np.testing.assert_allclose(out[:], old, rtol=1e-10, atol=1e-10)
    # 装配失败后旧缓存也不可复用.
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise RuntimeError('模拟局部装配失败')
        patch.setattr(sub.prototype, 'iter_local_stiffness_batches', fail)
        with pytest.raises(RuntimeError, match='模拟'):
            sub.assemble_stiff_matrix(density)
    assert sub._corner_recovery_batches is None
    assert sub._density_cells is None
