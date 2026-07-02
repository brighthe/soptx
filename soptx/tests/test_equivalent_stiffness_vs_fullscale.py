"""V1: 子结构静力缩聚 vs 全尺度 Schur 补 (机器精度), 及前向闭环正确性.

对应 soptx-piml-multiscale-integration-plan §4:
- V1  K^cond (逐子结构缩聚组装) vs 全尺度全局 Schur 补, rel_error < 1e-10;
- V2  形函数分区单位 / 刚体模态再现 (机器精度);
- T4  Exact / Mock 预测器接口互换 (均匀密度下 Mock 精确);
- T5  单步前向闭环: 接口解与细尺度恢复 vs 全尺度直解 (机器精度).
"""

import numpy as np
from fealpy.backend import backend_manager as bm

from soptx.analysis.multiscale import (
    CoarseFineMeshPair,
    ExactPredictor,
    InterfaceCondensedSystem,
    MockPredictor,
    SubstructureTemplate,
    assemble_fullscale_stiffness,
    fullscale_schur_complement,
    solve_with_dirichlet,
)
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial


def _make_material():
    return IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_stress",
    )


def _smooth_rho(mesh_pair):
    """光滑非均匀密度场 (确定性), 值域约 [0.3, 0.9]."""
    x0, x1, y0, y1 = mesh_pair.domain
    xc = x0 + (mesh_pair.cell_ix + 0.5) * mesh_pair.hx
    yc = y0 + (mesh_pair.cell_iy + 0.5) * mesh_pair.hy
    return 0.3 + 0.3 * (1.0 + np.sin(3.0 * xc + 1.0) * np.cos(2.0 * yc + 0.5))


def test_condensed_stiffness_matches_fullscale_schur():
    """V1: K^cond 与全尺度 Schur 补机器精度一致 (非均匀密度)."""
    bm.set_backend("numpy")
    material = _make_material()
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 1.0, 0.0, 1.0),
                                   ncx=2, ncy=2, L=5)
    rho = _smooth_rho(mesh_pair)

    template = SubstructureTemplate(mesh_pair, material, q=4)
    system = InterfaceCondensedSystem(mesh_pair, ExactPredictor(template), rho)

    K = assemble_fullscale_stiffness(mesh_pair, material, rho, q=4)
    S = fullscale_schur_complement(K, mesh_pair.interface_dofs)

    K_cond = system.K_cond.toarray()
    rel_error = np.linalg.norm(K_cond - S) / np.linalg.norm(S)
    assert rel_error < 1.0e-10

    # 对称性 (缩聚保持对称)
    sym_err = np.linalg.norm(K_cond - K_cond.T) / np.linalg.norm(K_cond)
    assert sym_err < 1.0e-13


def test_shape_function_rigid_body_reproduction():
    """V2: N^j 再现刚体平移/转动 (分区单位), K_s^j 对刚体模态作用为零."""
    bm.set_backend("numpy")
    material = _make_material()
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 1.0, 0.0, 1.0),
                                   ncx=2, ncy=2, L=5)
    rho = _smooth_rho(mesh_pair)
    template = SubstructureTemplate(mesh_pair, material, q=4)

    j = 3
    op = template.condense(rho[mesh_pair.sub_cells(j)])
    coords = mesh_pair.sub_node_coords(j)   # [内部; 边界] 节点序
    n_i_nodes = mesh_pair.n_local_interior_nodes

    # 刚体模态: 平移 x / 平移 y / 绕原点转动 (-y, x), 节点分量交错
    modes = [
        np.stack([np.ones(len(coords)), np.zeros(len(coords))], axis=1),
        np.stack([np.zeros(len(coords)), np.ones(len(coords))], axis=1),
        np.stack([-coords[:, 1], coords[:, 0]], axis=1),
    ]
    scale = np.linalg.norm(op.K_s)
    for mode in modes:
        full = mode.reshape(-1)
        u_b = mode[n_i_nodes:].reshape(-1)
        # N u_b 再现整个子结构上的刚体模态 (分区单位)
        err = np.linalg.norm(op.N @ u_b - full) / np.linalg.norm(full)
        assert err < 1.0e-10
        # 缩聚刚度对刚体模态作用为零
        assert np.linalg.norm(op.K_s @ u_b) / scale < 1.0e-12


def test_mock_predictor_interface_interchange():
    """T4: Mock 与 Exact 接口互换; 均匀局部密度下 Mock 精确."""
    bm.set_backend("numpy")
    material = _make_material()
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 1.0, 0.0, 1.0),
                                   ncx=2, ncy=2, L=4)
    template = SubstructureTemplate(mesh_pair, material, q=4)

    # 逐粗单元常数 (子结构内均匀) 的密度场: Mock 应与 Exact 机器精度一致
    rho = np.empty(mesh_pair.fine_mesh.number_of_cells())
    for j in range(mesh_pair.n_sub):
        rho[mesh_pair.sub_cells(j)] = 0.4 + 0.15 * j

    sys_exact = InterfaceCondensedSystem(mesh_pair, ExactPredictor(template), rho)
    sys_mock = InterfaceCondensedSystem(mesh_pair, MockPredictor(template), rho)

    diff = (sys_exact.K_cond - sys_mock.K_cond)
    rel = np.abs(diff).max() / np.abs(sys_exact.K_cond).max()
    assert rel < 1.0e-12


def test_forward_interface_solve_matches_fullscale_direct():
    """T5: 单步前向闭环 (含内部载荷缩聚) vs 全尺度直解, 机器精度."""
    bm.set_backend("numpy")
    material = _make_material()
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 2.0, 0.0, 1.0),
                                   ncx=4, ncy=2, L=4)
    rho = _smooth_rho(mesh_pair)
    template = SubstructureTemplate(mesh_pair, material, q=4)
    system = InterfaceCondensedSystem(mesh_pair, ExactPredictor(template), rho)

    fixed_dofs = mesh_pair.dofs_on_line_x(0.0)
    rng = np.random.default_rng(42)
    F = rng.normal(size=mesh_pair.n_dofs)   # 含内部自由度分量的一般载荷
    F[fixed_dofs] = 0.0

    U_b, info = system.solve_interface(F, fixed_dofs)
    assert info["interface_residual"] < 1.0e-10

    K = assemble_fullscale_stiffness(mesh_pair, material, rho, q=4)
    u_full = solve_with_dirichlet(K, F, fixed_dofs)

    u_full_b = u_full[mesh_pair.interface_dofs]
    rel_iface = np.linalg.norm(U_b - u_full_b) / np.linalg.norm(u_full_b)
    assert rel_iface < 1.0e-10

    u_rec = system.recover_fine(U_b, F)
    rel_fine = np.linalg.norm(u_rec - u_full) / np.linalg.norm(u_full)
    assert rel_fine < 1.0e-10

    # 求解规模: 接口自由度远小于全尺度自由度
    assert info["n_interface_dofs"] < info["n_fine_dofs"]
