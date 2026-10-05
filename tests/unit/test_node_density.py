"""节点密度 (density_location='node') 链路与 v0.4 网格接口的回归测试.

节点密度分支曾调用 v0.4 网格已不存在的 ``cell_to_node`` / ``jacobi_matrix``, 插值格式
的 SIMP 分支还读不存在的 ``self._mesh``、求导时未在积分点处求值, 整条链路从未跑通.
这里用中心差分核对柔顺度与体积约束的灵敏度, 并检查节点过滤的测度权重. 同文件顺带
覆盖两处同类问题: 三维跳量稳定化明确报错, ``Integrator.size`` 不再调用 ``mesh.count``.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.analyzers.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from soptx.fem.bilinear_form import BilinearForm
from soptx.fem.integrators import JumpPenaltyIntegrator, LinearElasticIntegrator
from soptx.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.mesh import QuadrangleMesh, TetrahedronMesh, TriangleMesh
from soptx.problems import HalfMBBBeamRight2d
from soptx.topology.constraints.volume import VolumeConstraint
from soptx.topology.filters import Filter
from soptx.topology.interpolation.scheme import MaterialInterpolationScheme
from soptx.topology.objectives.compliance import ComplianceObjective

DOMAIN = (0.0, 6.0, 0.0, 2.0)
FD_STEP = 1e-6
FD_RTOL = 1e-5


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _node_density_case(mesh_type, method: str):
    """6x2 网格上的半 MBB 梁, 节点密度取 [0.3, 0.9] 内的固定随机值."""
    problem = HalfMBBBeamRight2d(domain=DOMAIN)
    mesh = mesh_type.from_box(list(DOMAIN), nx=6, ny=2)
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="plane_stress", enable_logging=False
    )
    scheme = MaterialInterpolationScheme(
        density_location="node",
        interpolation_method=method,
        options={"penalty_factor": 3.0, "void_youngs_modulus": 1e-9},
        enable_logging=False,
    )
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=mesh,
        pde=problem,
        material=material,
        interpolation_scheme=scheme,
        space_degree=1,
        integration_order=3,
        solve_method="scipy",
        topopt_algorithm="density_based",
        enable_logging=False,
    )
    _, rho = scheme.setup_density_distribution(
        design_variable_mesh=mesh, displacement_mesh=mesh, relative_density=0.5
    )
    rng = np.random.default_rng(0)
    rho[:] = bm.tensor(rng.uniform(0.3, 0.9, rho.shape[0]))
    return analyzer, rho, rng


def _perturbed(rho, i: int, step: float):
    """返回第 i 个节点密度加上 step 后的新密度函数."""
    values = bm.copy(rho[:])
    values[i] += step
    return rho.space.function(values)


@pytest.mark.parametrize("method", ["simp", "msimp"])
@pytest.mark.parametrize("mesh_type", [TriangleMesh, QuadrangleMesh], ids=["tri", "quad"])
def test_node_density_sensitivities_match_finite_difference(mesh_type, method):
    """柔顺度与体积约束的解析灵敏度与中心差分一致."""
    analyzer, rho, rng = _node_density_case(mesh_type, method)
    compliance = ComplianceObjective(analyzer)
    volume = VolumeConstraint(analyzer, volume_fraction=0.5)

    def compliance_at(density):
        return float(compliance.fun(density, analyzer.solve_state(rho_val=density)))

    dc = bm.to_numpy(compliance.jac(rho, analyzer.solve_state(rho_val=rho)))
    dv = bm.to_numpy(volume.jac(rho))
    assert dc.shape == dv.shape == (rho.shape[0], )

    for i in rng.choice(rho.shape[0], 3, replace=False):
        plus, minus = _perturbed(rho, i, FD_STEP), _perturbed(rho, i, -FD_STEP)
        fd_c = (compliance_at(plus) - compliance_at(minus)) / (2 * FD_STEP)
        fd_v = (float(volume.fun(plus)) - float(volume.fun(minus))) / (2 * FD_STEP)
        assert dc[i] == pytest.approx(fd_c, rel=FD_RTOL)
        assert dv[i] == pytest.approx(fd_v, rel=FD_RTOL)


@pytest.mark.parametrize("filter_type", ["sensitivity", "density", "projection"])
@pytest.mark.parametrize("mesh_type", [TriangleMesh, QuadrangleMesh], ids=["tri", "quad"])
def test_node_filter_measure_sums_to_domain_area(mesh_type, filter_type):
    """节点过滤把单元测度均分到节点, 节点测度之和等于区域面积."""
    mesh = mesh_type.from_box(list(DOMAIN), nx=6, ny=2)
    filt = Filter(
        design_mesh=mesh,
        filter_type=filter_type,
        rmin=1.5,
        density_location="node",
        enable_logging=False,
    )
    weight = bm.to_numpy(filt._strategy._measure_weight)

    assert weight.shape == (mesh.number_of_nodes(), )
    assert weight.sum() == pytest.approx(12.0, rel=1e-14)


def test_jump_penalty_rejects_3d():
    """三维跳量稳定化缺 cell_to_face_sign 且无算例验证, 应明确报 NotImplementedError."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 1, 1, 1)
    material = IsotropicLinearElasticMaterial(
        lame_lambda=1.0, shear_modulus=0.5, hypothesis="3D", enable_logging=False
    )
    space = TensorFunctionSpace(
        scalar_space=LagrangeFESpace(mesh, p=1, ctype="D"), shape=(-1, 3)
    )
    face2cell = mesh.face_to_cell()
    internal = bm.nonzero(face2cell[:, 0] != face2cell[:, 1])[0]
    form = BilinearForm(space)
    form.add_integrator(
        JumpPenaltyIntegrator(
            q=3,
            threshold=internal,
            method="matrix_jump",
            material=material,
            penalty_scaling="physical_h",
        )
    )

    with pytest.raises(NotImplementedError, match="三维跳量稳定化尚未实现"):
        form.assembly(method="coalesce")


def test_integrator_size_counts_entities():
    """未指定 region 时, Integrator.size 按 etype 返回网格实体数."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    material = IsotropicLinearElasticMaterial(
        lame_lambda=1.0, shear_modulus=0.5, hypothesis="plane_strain", enable_logging=False
    )
    integrator = LinearElasticIntegrator(material=material, method="fast")

    assert integrator.size(mesh) == mesh.number_of_cells()
