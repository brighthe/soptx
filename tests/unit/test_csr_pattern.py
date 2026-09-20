# -*- coding: utf-8 -*-
"""模式先行 (Pattern-First) CSR 稀疏矩阵装配单元测试.

测试覆盖范围:
1. 2D (三角形、四边形) 与 3D (四面体、六面体) 有限元网格;
2. 标量拉格朗日空间与向量张量空间;
3. 动态密度系数调制 (SIMP 拓扑优化多轮迭代复用测试);
4. NumPy 后端与 PyTorch (CPU / CUDA) 双后端数值与拓扑代数严格等价性 (< 1e-14).
"""

import numpy as np
import pytest
import torch
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh
from fealpy.sparse import CSRTensor

from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.fem.matrix.csr_pattern import CSRPattern, assemble_csr, build_csr_pattern
from soptx.materials.linear_elasticity import IsotropicLinearElasticMaterial


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


@pytest.mark.parametrize(
    "mesh_factory,hypo,dim",
    [
        (lambda: TriangleMesh.from_box([0, 1, 0, 1], 3, 3), "plane_stress", 2),
        (lambda: QuadrangleMesh.from_box([0, 1, 0, 1], 3, 3), "plane_strain", 2),
        (lambda: TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 2, 2, 2), "3D", 3),
        (lambda: HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], 2, 2, 2), "3D", 3),
    ],
)
def test_csr_pattern_numpy_equivalence(mesh_factory, hypo, dim):
    """测试 NumPy 后端下模式先行装配与 FEALPy coalesce 产物的严格等价性."""
    bm.set_backend("numpy")
    mesh = mesh_factory()
    space = LagrangeFESpace(mesh, p=1)
    tspace = TensorFunctionSpace(scalar_space=space, shape=(-1, dim))

    mat = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis=hypo
    )
    integrator = LinearElasticIntegrator(material=mat, method="fast")
    K_e = integrator.assembly(tspace)

    # 1. FEALPy 传统装配基准
    bform = BilinearForm(tspace)
    bform.add_integrator(integrator)
    K_fealpy = bform.assembly()

    # 2. 模式先行装配
    pattern = build_csr_pattern(tspace)
    K_pattern = assemble_csr(K_e, pattern)

    assert isinstance(K_pattern, CSRTensor)
    assert K_pattern.sparse_shape == K_fealpy.sparse_shape

    # 3. 数值误差校验 (稠密与非零元)
    K_dense_fealpy = bm.to_numpy(K_fealpy.to_dense())
    K_dense_pattern = bm.to_numpy(K_pattern.to_dense())

    max_diff = np.max(np.abs(K_dense_fealpy - K_dense_pattern))
    assert max_diff < 1e-14, f"NumPy 后端装配误差过大: {max_diff}"


@pytest.mark.parametrize(
    "mesh_factory,hypo,dim",
    [
        (lambda: TriangleMesh.from_box([0, 1, 0, 1], 3, 3), "plane_stress", 2),
        (lambda: TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 2, 2, 2), "3D", 3),
    ],
)
def test_csr_pattern_pytorch_cpu_equivalence(mesh_factory, hypo, dim):
    """测试 PyTorch CPU 后端下模式先行装配与 FEALPy coalesce 产物的严格等价性."""
    bm.set_backend("pytorch")
    mesh = mesh_factory()
    space = LagrangeFESpace(mesh, p=1)
    tspace = TensorFunctionSpace(scalar_space=space, shape=(-1, dim))

    mat = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis=hypo
    )
    integrator = LinearElasticIntegrator(material=mat, method="fast")
    K_e = integrator.assembly(tspace)

    # 1. FEALPy 传统基准
    bform = BilinearForm(tspace)
    bform.add_integrator(integrator)
    K_fealpy = bform.assembly()

    # 2. 模式先行装配
    pattern = build_csr_pattern(tspace, device="cpu")
    K_pattern = assemble_csr(K_e, pattern)

    assert isinstance(K_pattern, CSRTensor)
    diff = torch.max(torch.abs(K_fealpy.to_dense() - K_pattern.to_dense())).item()
    assert diff < 1e-14, f"PyTorch CPU 后端装配误差过大: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 支持")
def test_csr_pattern_pytorch_cuda_equivalence():
    """测试 PyTorch CUDA GPU 显存内模式先行装配的数值正确性与显存驻留."""
    bm.set_backend("pytorch")
    device = torch.device("cuda:0")

    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 2, 2, 2)
    space = LagrangeFESpace(mesh, p=1)
    tspace = TensorFunctionSpace(scalar_space=space, shape=(-1, 3))

    mat = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="3D"
    )
    integrator = LinearElasticIntegrator(material=mat, method="fast")
    K_e = integrator.assembly(tspace)
    K_e_cuda = K_e.to(device=device)

    # 1. 符号阶段构建并送上 GPU
    pattern = build_csr_pattern(tspace, device=device)
    assert pattern.device == device
    assert pattern.crow.is_cuda
    assert pattern.col.is_cuda
    assert pattern.slot_base.is_cuda

    # 2. 数值阶段直接在 GPU 显存中装配
    K_pattern_cuda = assemble_csr(K_e_cuda, pattern)
    assert K_pattern_cuda.values.is_cuda

    # 3. 对比 CPU 端基准
    pattern_cpu = build_csr_pattern(tspace, device="cpu")
    K_pattern_cpu = assemble_csr(K_e.cpu(), pattern_cpu)

    diff = torch.max(
        torch.abs(K_pattern_cpu.to_dense() - K_pattern_cuda.to_dense().cpu())
    ).item()
    assert diff < 1e-14, f"CUDA GPU 装配与 CPU 结果不一致: {diff}"


def test_csr_pattern_topopt_density_multi_iteration():
    """模拟拓扑优化多轮迭代 (SIMP 相对密度调制), 验证 pattern 与 buffer 的重复复用能力."""
    bm.set_backend("numpy")
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 3, 3, 3)
    space = LagrangeFESpace(mesh, p=1)
    tspace = TensorFunctionSpace(scalar_space=space, shape=(-1, 3))
    NC = mesh.number_of_cells()

    mat = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="3D"
    )
    integrator = LinearElasticIntegrator(material=mat, method="fast")

    # 符号阶段只构建 1 次
    pattern = build_csr_pattern(tspace)

    np.random.seed(42)
    for it in range(5):
        # 模拟每轮迭代随机密度场 rho in [1e-3, 1.0]
        rho = np.random.uniform(0.01, 1.0, size=(NC,))
        integrator.coef = rho**3  # SIMP 惩罚

        K_e = integrator.assembly(tspace)

        # 模式先行装配
        K_pattern = assemble_csr(K_e, pattern)

        # FEALPy 传统装配
        bform = BilinearForm(tspace)
        bform.add_integrator(integrator)
        K_fealpy = bform.assembly()

        diff = np.max(np.abs(bm.to_numpy(K_fealpy.to_dense()) - bm.to_numpy(K_pattern.to_dense())))
        assert diff < 1e-14, f"迭代 {it} 误差超标: {diff}"