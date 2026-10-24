"""PIMLShapeCornerCondensation 在 numpy、pytorch/CPU 与 pytorch/CUDA 下的分析结果一致性.

推理、补全、变分式与诊断都在接口迹矩阵所在设备上进行; 网络取固定种子的随机线性层, 只检验各设备
算出同一结果, 不涉及预测精度. 解码器的 pivot 编号统一取 numpy 参考子结构现选的一组, 模拟由权重给出.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import SubstructureAnalyzer
from soptx.fem.substructure import (
    IndependentPredictionDecoder, StructuredSubstructureLayout, build_modulus_substructures,
)
from soptx.fem.substructure.piml_corner import DIAGNOSTIC_KEYS, PIMLShapeCornerCondensation
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective

E0, EMIN, NU, PENALTY = 1.0, 1.0e-7, 0.3, 3.0
# x 与 z 方向子结构数取偶数, 顶面中心载荷与底面四角支承都落在宏观节点上
N_SUB, N_FINE = (4, 1, 2), 2
GRID = tuple(n * N_FINE for n in N_SUB)
DOMAIN = (0.0, float(GRID[0]), 0.0, float(GRID[1]), 0.0, float(GRID[2]))
# 各设备的差异只来自浮点求和次序
CG_OPTIONS = {'precond': 'jacobi', 'rtol': 1e-12, 'atol': 0.0, 'maxiter': 20000}
UNIFORM_THRESHOLD = 1.0e-3

DEVICES = [
    pytest.param('pytorch', 'cpu', id='pytorch-cpu'),
    pytest.param('pytorch', 'cuda', id='pytorch-cuda',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA GPU 支持')),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend('numpy')
    yield
    bm.set_backend('numpy')


def _analyze(backend: str, device: str, uniform: bool, pivot_indices=None) -> dict:
    """在给定后端与设备上用随机网络做一次 PIML 形函数缩聚分析.

    Parameters
    ----------
    backend : {'numpy', 'pytorch'}
        张量后端.
    device : {'cpu', 'cuda'}
        细网格、网络及缩聚所在设备.
    uniform : bool
        True 时取全实体密度并启用均匀判别, 所有子结构改用 ``solid_shape``; False 时取随机密度且不判别.
    pivot_indices : sequence of int, optional
        解码器的 pivot 编号; 缺省时由参考子结构现选.

    Returns
    -------
    dict
        ``compliance``, ``displacement`` (numpy 数组), 缩聚器的 ``report``、恢复矩阵所在的设备类型
        ``recovery_device`` 与解码器实际使用的 ``pivot_indices``.
    """
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID, device=device)
    layout = StructuredSubstructureLayout(
        domain_size=tuple(float(n) for n in GRID), n_sub=N_SUB, n_fine=(N_FINE, ) * 3,
        degree=1, E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh,
    )
    problem = FullMBBBeamPoint3d(domain=DOMAIN, P=-1.0, E=E0, nu=NU)
    material = IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D',
                                              device=device)
    interpolation = MaterialInterpolationScheme(
        density_location='element', interpolation_method='msimp',
        options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']},
    )
    prototype, _, _ = build_modulus_substructures(layout, integration_order=2)
    decoder = IndependentPredictionDecoder(prototype, trace_kind='linear_corner', pivot_indices=pivot_indices)
    # 固定种子的随机线性层, 先在 CPU 上初始化再移到目标设备, 各设备权重逐位相同
    torch.manual_seed(2026)
    network = nn.Linear(int(prototype.n_cells), decoder.shape_codec.n_output, dtype=torch.float64)
    network.to(device)
    rng = np.random.default_rng(7)
    solid_shape = None
    if uniform:
        solid_shape = rng.uniform(-0.1, 0.1, size=(decoder.shape_codec.n_internal, decoder.shape_codec.n_trace))
    condensation = PIMLShapeCornerCondensation(
        decoder, network, provider_metadata=decoder.metadata(),
        uniform_threshold=UNIFORM_THRESHOLD if uniform else None, solid_shape=solid_shape)
    # chunk_size 小于子结构数, 覆盖跨批的装配与恢复
    analyzer = SubstructureAnalyzer(
        layout=layout, pde=problem, material=material, trace='linear_corner', chunk_size=3,
        space_degree=1, integration_order=2, solve_method='cg', solver_options=dict(CG_OPTIONS),
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
        corner_condensation=condensation,
    )
    values = np.ones(int(np.prod(GRID))) if uniform else rng.uniform(0.2, 1.0, size=int(np.prod(GRID)))
    density = bm.device_put(bm.asarray(values, dtype=bm.float64), device)
    stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
    recovery_device = bm.device_type(analyzer._corner_recovery_batches[0][2])
    prepared, rhs = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
    displacement = analyzer.tensor_space.function()
    _, info = analyzer.solve_system(prepared, rhs, displacement)
    assert info['converged']
    state = {'displacement': displacement, 'solver': info}
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
    return dict(
        compliance=float(objective.fun(density=density, state=state)),
        displacement=np.asarray(bm.to_numpy(displacement[:])),
        report=dict(analyzer.condensation_report),
        recovery_device=recovery_device,
        pivot_indices=decoder.metadata()['pivot_indices'],
    )


@pytest.mark.parametrize('uniform', [False, True], ids=['predicted', 'uniform-replaced'])
@pytest.mark.parametrize('backend,device', DEVICES)
def test_piml_corner_matches_numpy(backend, device, uniform):
    """pytorch/CPU 与 pytorch/CUDA 的柔顺度、位移、计数与诊断分布都与 numpy 一致, 恢复矩阵留在计算设备."""
    reference = _analyze('numpy', 'cpu', uniform)
    result = _analyze(backend, device, uniform, pivot_indices=reference['pivot_indices'])
    assert result['pivot_indices'] == reference['pivot_indices']
    n_substructures = int(np.prod(N_SUB))
    assert reference['report']['predicted'] == result['report']['predicted'] == n_substructures
    expected_replaced = n_substructures if uniform else 0
    assert reference['report']['uniform_replaced'] == result['report']['uniform_replaced'] == expected_replaced
    assert result['recovery_device'] == device
    assert result['compliance'] == pytest.approx(reference['compliance'], rel=1e-10)
    np.testing.assert_allclose(result['displacement'], reference['displacement'],
                               rtol=1e-9, atol=1e-10 * np.abs(reference['displacement']).max())
    for key in DIAGNOSTIC_KEYS:
        for stat in ('min', 'median', 'max'):
            assert result['report'][key][stat] == pytest.approx(reference['report'][key][stat], rel=1e-8, abs=1e-12)


def test_piml_corner_rejects_network_on_other_device():
    """网络与接口迹矩阵不在同一设备时, 在首批推理之前拒绝."""
    bm.set_backend('numpy')
    mesh = HexahedronMesh.from_box(list(DOMAIN), *GRID)
    layout = StructuredSubstructureLayout(
        domain_size=tuple(float(n) for n in GRID), n_sub=N_SUB, n_fine=(N_FINE, ) * 3,
        degree=1, E_base=E0, nu=NU, hypothesis='3D', full_mesh=mesh,
    )
    prototype, _, _ = build_modulus_substructures(layout, integration_order=2)
    decoder = IndependentPredictionDecoder(prototype, trace_kind='linear_corner')
    network = nn.Linear(int(prototype.n_cells), decoder.shape_codec.n_output, device='meta', dtype=torch.float64)
    condensation = PIMLShapeCornerCondensation(decoder, network, provider_metadata=decoder.metadata())
    analyzer = SubstructureAnalyzer(
        layout=layout, pde=FullMBBBeamPoint3d(domain=DOMAIN, P=-1.0, E=E0, nu=NU),
        material=IsotropicLinearElasticMaterial(youngs_modulus=E0, poisson_ratio=NU, hypothesis='3D'),
        trace='linear_corner', chunk_size=3, space_degree=1, integration_order=2,
        solve_method='cg', solver_options=dict(CG_OPTIONS), topopt_algorithm='density_based',
        interpolation_scheme=MaterialInterpolationScheme(
            density_location='element', interpolation_method='msimp',
            options={'penalty_factor': PENALTY, 'void_youngs_modulus': EMIN, 'target_variables': ['E']}),
        corner_condensation=condensation,
    )
    with pytest.raises(ValueError, match='同一设备'):
        analyzer.assemble_stiff_matrix(rho_val=bm.full((int(np.prod(GRID)), ), 0.5, dtype=bm.float64))
