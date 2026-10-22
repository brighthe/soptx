"""SubstructureAnalyzer 在 numpy、pytorch/CPU 与 pytorch/CUDA 下的分析结果一致性.

CG 路径的约束准备、迭代与真实残差验收都在接口刚度所在设备上进行, 接口位移留在该设备;
直接法与 GMG 仍只在主机上, 不在本文件范围内.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers import SubstructureAnalyzer
from soptx.fem.analyzers.substructure_analyzer import TRACES
from soptx.fem.substructure import StructuredSubstructureLayout
from soptx.fem.substructure.interface_space import _csr_diagonal_slots
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import HexahedronMesh
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective

E0, EMIN, NU, PENALTY = 1.0, 1.0e-7, 0.3, 3.0
# x 与 z 方向子结构数取偶数, 顶面中心载荷与底面四角支承都落在宏观节点上, 两种迹都能表达
N_SUB, N_FINE = (4, 1, 2), 2
GRID = tuple(n * N_FINE for n in N_SUB)
DOMAIN = (0.0, float(GRID[0]), 0.0, float(GRID[1]), 0.0, float(GRID[2]))
# 两侧容差都取得很紧, 差异只来自浮点求和次序
CG_OPTIONS = {'precond': 'jacobi', 'rtol': 1e-12, 'atol': 0.0, 'maxiter': 20000}

DEVICES = [
    pytest.param('cpu', id='pytorch-cpu'),
    pytest.param('cuda', id='pytorch-cuda',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA GPU 支持')),
]


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend('numpy')
    yield
    bm.set_backend('numpy')


def _build(backend: str, device: str, trace: str):
    """在给定后端与设备上构造子结构分析器与一组固定的随机物理密度.

    Parameters
    ----------
    backend : {'numpy', 'pytorch'}
        张量后端.
    device : {'cpu', 'cuda'}
        细网格及其导出张量所在设备.
    trace : {'full_trace', 'linear_corner'}
        接口迹空间.

    Returns
    -------
    analyzer : SubstructureAnalyzer
    density : TensorLike
        细单元物理密度, 与细网格同设备.
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
    # chunk_size 小于子结构数, 覆盖跨批的装配与恢复
    analyzer = SubstructureAnalyzer(
        layout=layout, pde=problem, material=material, trace=trace, chunk_size=3,
        space_degree=1, integration_order=2, solve_method='cg', solver_options=dict(CG_OPTIONS),
        topopt_algorithm='density_based', interpolation_scheme=interpolation,
    )
    rng = np.random.default_rng(7)
    density = bm.device_put(
        bm.asarray(rng.uniform(0.2, 1.0, size=int(np.prod(GRID))), dtype=bm.float64), device)
    return analyzer, density


def _analyze(backend: str, device: str, trace: str) -> dict:
    """做一次 CG 分析, 返回柔顺度、位移与右端项、接口位移所在的设备类型.

    Parameters
    ----------
    backend, device, trace : str
        同 ``_build``.

    Returns
    -------
    dict
        ``compliance``, ``displacement`` (numpy 数组), ``rhs_device`` 与 ``interface_device``.
    """
    analyzer, density = _build(backend, device, trace)
    stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
    prepared, rhs = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
    displacement = analyzer.tensor_space.function()
    _, info = analyzer.solve_system(prepared, rhs, displacement)
    assert info['converged']
    state = {'displacement': displacement, 'solver': info}
    objective = ComplianceObjective(analyzer=analyzer, state_variable='u', diff_mode='manual')
    return dict(
        compliance=float(objective.fun(density=density, state=state)),
        displacement=np.asarray(bm.to_numpy(displacement[:])),
        rhs_device=bm.device_type(rhs),
        interface_device=bm.device_type(analyzer.interface_displacement),
    )


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('trace', TRACES)
def test_pytorch_analysis_matches_numpy(trace, device):
    """pytorch 各设备上的柔顺度与全场位移与 numpy 一致, 右端项与接口位移留在所选设备."""
    reference = _analyze('numpy', 'cpu', trace)
    result = _analyze('pytorch', device, trace)
    assert result['rhs_device'] == device
    assert result['interface_device'] == device
    np.testing.assert_allclose(result['compliance'], reference['compliance'], rtol=1e-10)
    scale = float(np.max(np.abs(reference['displacement'])))
    assert float(np.max(np.abs(result['displacement'] - reference['displacement']))) <= 1e-9 * scale


@pytest.mark.parametrize('trace', TRACES)
def test_interface_diagonal_matches_csr_diagonal(trace):
    """按缓存槽位取的接口刚度对角与 CSRTensor.diagonal 相同, 分块求槽位不改变结果."""
    analyzer, density = _build('numpy', 'cpu', trace)
    stiffness = analyzer.assemble_stiff_matrix(rho_val=density).stiffness
    expected = bm.to_numpy(stiffness.diagonal())
    np.testing.assert_array_equal(bm.to_numpy(analyzer.interface_space.diagonal(stiffness)), expected)
    pattern = analyzer.interface_space.pattern
    # nnz 远小于 2^31, 行指针与列号按 scipy 规则取 int32
    assert pattern.crow.dtype == np.int32 and pattern.col.dtype == np.int32
    slots = _csr_diagonal_slots(pattern.crow, pattern.col, chunk_nnz=7)
    np.testing.assert_array_equal(bm.to_numpy(stiffness.values)[bm.to_numpy(slots)], expected)


@pytest.mark.parametrize('trace', TRACES)
def test_split_solve_matches_solve_system(trace):
    """先 solve_interface 再 recover_displacement 与 solve_system 逐位相同, 未求解时拒绝回代."""
    results = []
    for split in (False, True):
        analyzer, density = _build('numpy', 'cpu', trace)
        stiffness = analyzer.assemble_stiff_matrix(rho_val=density)
        prepared, rhs = analyzer.apply_bc(stiffness, analyzer.assemble_body_force_vector())
        displacement = analyzer.tensor_space.function()
        if split:
            with pytest.raises(RuntimeError, match='solve_interface'):
                analyzer.recover_displacement(displacement)
            info = analyzer.solve_interface(prepared, rhs)
            analyzer.recover_displacement(displacement)
        else:
            _, info = analyzer.solve_system(prepared, rhs, displacement)
        assert info['converged']
        results.append(np.asarray(bm.to_numpy(displacement[:])))
    np.testing.assert_array_equal(results[1], results[0])
