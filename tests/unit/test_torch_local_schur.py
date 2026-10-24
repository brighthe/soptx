"""PyTorch 局部 Schur 的设备一致性和缓存恢复回归, CUDA 不可用时跳过."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from soptx.backend import backend_manager as bm
from soptx.fem.substructure.implicit import FullInterfaceSchurOperator
from soptx.fem.substructure.local_assembly import build_local_assembly_map
from soptx.fem.substructure.local_schur import build_local_block_patterns, LocalSchurOperator
from soptx.fem.substructure.torch_local_schur import (
    TorchBlockPattern, TorchInteriorBlockSolver, TorchLocalSchurBlocks,
)


@pytest.fixture(autouse=True)
def restore_backend():
    """每个用例结束后恢复 NumPy 后端."""
    bm.set_backend('numpy')
    yield
    bm.set_backend('numpy')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('chunk', [1, 2])
def test_torch_local_schur_and_recovery(device, chunk, monkeypatch):
    """两侧对照独立消元, 并检查迭代和恢复不回到主机或重分解."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    mapping = build_local_assembly_map(np.array([[0, 1], [1, 2]]), 3)
    host_patterns = build_local_block_patterns(mapping, np.array([1]), np.array([0, 2]))
    local_np = np.array([
        [[2., -1., 0.], [-1., 3., -.5], [0., -.5, 2.]],
        [[1., -.3, 0.], [-.3, 2., -.7], [0., -.7, 1.5]],
    ])
    global_matrix = np.zeros((5, 5))
    for ids, matrix in zip(([0, 1, 2], [2, 3, 4]), local_np):
        global_matrix[np.ix_(ids, ids)] += matrix
    interface_np, interior_np = np.array([0, 2, 4]), np.array([1, 3])
    kii = global_matrix[np.ix_(interior_np, interior_np)]
    kib = global_matrix[np.ix_(interior_np, interface_np)]
    reference = global_matrix[np.ix_(interface_np, interface_np)] - kib.T @ np.linalg.solve(kii, kib)
    bm.set_backend('pytorch')
    tensor = lambda value, dtype=torch.int64: torch.as_tensor(value, dtype=dtype, device=device)
    patterns = tuple(TorchBlockPattern(pattern, device) for pattern in host_patterns)
    local = tensor(local_np, torch.float64)
    outputs = []
    for explicit in (True, False):
        blocks = TorchLocalSchurBlocks(
            TorchInteriorBlockSolver(tensor([[1], [3]]), 5),
            tensor([0, 2, 4]), tensor([[0, 1], [1, 2]]), patterns,
            keep_boundary=not explicit, chunk_size=chunk)
        timings = dict(internal_factorization=0., local_blocks=0.,
                       schur_formation=0., schur_diagonal=0.)
        assembled = local.new_zeros((3, 3))
        for first in range(0, 2, chunk):
            schur = blocks.prepare_batch(first, local[first:first + chunk], tensor([1]), tensor([0, 2]),
                                         explicit=explicit, timings=timings, column_chunk=1)
            if explicit:
                for offset, matrix in enumerate(schur):
                    ids = blocks.local_dofs[first + offset]
                    assembled[ids[:, None], ids] += matrix
        assert blocks.coupling.device.type == device
        assert blocks.interior.factors.device.type == device
        assert (blocks.boundary is None) == explicit
        np.testing.assert_allclose(blocks.diagonal.cpu(), reference.diagonal(), rtol=1e-12)
        if explicit:
            np.testing.assert_allclose(assembled.cpu(), reference, rtol=1e-12, atol=1e-14)
        q = tensor([0., .2, -.4], torch.float64)
        core = LocalSchurOperator(blocks, tensor([1, 2]))
        diagonal = blocks.diagonal.clone()
        diagonal[0] = 1.
        operator = FullInterfaceSchurOperator(core, tensor([1, 2]), diagonal)

        def forbidden(*args, **kwargs):
            raise AssertionError('迭代或恢复发生主机拷贝/重新分解')

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'cpu', forbidden)
            patch.setattr(torch.Tensor, 'numpy', forbidden)
            patch.setattr(TorchInteriorBlockSolver, 'factor_batch', forbidden)
            actual_u = blocks.recover(q)
            if not explicit:
                actual_action = operator @ q
                actual_recovery = operator.recover(q)
        assert actual_u.device.type == device
        expected = np.zeros(5)
        expected[interface_np] = q.cpu()
        expected[interior_np] = -np.linalg.solve(kii, kib @ q.cpu().numpy())
        np.testing.assert_allclose(actual_u.cpu(), expected, rtol=1e-12, atol=1e-14)
        if not explicit:
            expected_action = reference @ q.cpu().numpy()
            expected_action[0] = 0.
            np.testing.assert_allclose(actual_action.cpu(), expected_action, rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(actual_recovery, actual_u)
        outputs.append(actual_u)
    torch.testing.assert_close(outputs[0], outputs[1])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_torch_analyzer_two_material_states(device, monkeypatch):
    """正式分析器两侧一致, 热启动后换密度重建缓存, 隐式不创建接口 CSR."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    from soptx.fem.analyzers import SubstructureAnalyzer
    from soptx.fem.substructure import StructuredSubstructureLayout
    from soptx.fem.substructure import interface_space as interface_module
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.mesh import HexahedronMesh
    from soptx.problems import FullMBBBeamLine3d
    from soptx.topology.interpolation import MaterialInterpolationScheme

    bm.set_backend('pytorch')
    grid, domain = (12, 2, 2), (0., 12., 0., 2., 0., 2.)
    layout = StructuredSubstructureLayout(
        domain_size=(12., 2., 2.), n_sub=(6, 1, 1), n_fine=(2, 2, 2),
        degree=1, E_base=1., nu=.3, hypothesis='3D',
        full_mesh=HexahedronMesh.from_box(list(domain), *grid, device=device))
    problem = FullMBBBeamLine3d(domain=domain, P=-1., E=1., nu=.3)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1., poisson_ratio=.3, hypothesis='3D', device=device)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': 3., 'void_youngs_modulus': 1e-7, 'target_variables': ['E']})
    analyzers = [
        SubstructureAnalyzer(
            layout=layout, pde=problem, material=material, interpolation_scheme=interpolation,
            trace='full_trace', schur_mode=mode, operator_level='fa', chunk_size=4,
            solve_method='cg', solver_options={'precond': 'jacobi', 'rtol': 1e-10,
                                               'atol': 0., 'maxiter': 20000,
                                               'rtol_reference': 'rhs'})
        for mode in ('explicit', 'implicit')
    ]
    rho = torch.as_tensor(np.random.default_rng(7).uniform(.4, 1., np.prod(grid)),
                          dtype=torch.float64, device=device)
    previous, old = [None, None], [None, None]

    def forbidden(*args, **kwargs):
        raise AssertionError('隐式路径创建了接口 CSR')

    for density in (rho, rho.flip(0)):
        states = []
        for index, analyzer in enumerate(analyzers):
            with monkeypatch.context() as patch:
                if index == 1:
                    patch.setattr(interface_module, 'build_interface_pattern', forbidden)
                state = analyzer.solve_state(rho_val=density, x0=previous[index])
            blocks = analyzer._local_schur
            assert blocks is not old[index]
            assert blocks.interior.factors.device.type == device
            assert analyzer._fine_operator is None
            assert state['solver']['converged']
            if index == 1:
                assert analyzer.interface_space._pattern is None
            previous[index] = analyzer.interface_displacement.clone()
            old[index] = blocks
            states.append(state['displacement'][:].clone())
        torch.testing.assert_close(states[0], states[1], rtol=1e-6, atol=1e-7)