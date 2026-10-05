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
from soptx.fem.matrix.csr_pattern import (
    CSRChunkAccumulator,
    CSRPattern,
    assemble_csr,
    assemble_csr_chunks,
    build_csr_pattern,
    build_csr_pattern_from_dofmap,
)
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


def _ring_dofmap_case():
    """5 个自由度首尾相接的 4 个三节点局部实体, 及其随机局部矩阵."""
    local_to_global = np.array(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]], dtype=np.int64
    )
    values = np.random.default_rng(20261003).standard_normal((4, 3, 3))
    pattern = build_csr_pattern_from_dofmap(
        local_to_global, 5, allocate_buffer=False
    )
    dense = np.zeros((5, 5))
    for cell, dofs in enumerate(local_to_global):
        dense[np.ix_(dofs, dofs)] += values[cell]
    return pattern, values, dense


@pytest.mark.parametrize("splits", [[4], [1, 3], [1, 2, 1], [1, 1, 1, 1]])
def test_chunk_accumulator_matches_pull_assembly_and_dense_scatter(splits):
    """推模型累加器应与 ``assemble_csr_chunks`` 及稠密散加参考逐位一致."""
    pattern, values, dense = _ring_dofmap_case()
    bounds = np.concatenate([[0], np.cumsum(splits)])

    accumulator = CSRChunkAccumulator(pattern)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        accumulator.add(int(start), values[start:stop])
    assert accumulator.n_added == 4
    pushed = accumulator.to_csr()

    pulled = assemble_csr_chunks(
        ((int(a), values[a:b]) for a, b in zip(bounds[:-1], bounds[1:])), pattern
    )

    assert isinstance(pushed, CSRTensor)
    np.testing.assert_allclose(bm.to_numpy(pushed.to_dense()), dense, atol=1e-14)
    np.testing.assert_array_equal(
        bm.to_numpy(pushed.values), bm.to_numpy(pulled.values)
    )


def test_chunk_accumulators_on_one_pattern_do_not_share_values():
    """同一模式上先后建立的累加器在不共享缓冲区时互不污染."""
    pattern, values, dense = _ring_dofmap_case()

    first = CSRChunkAccumulator(pattern)
    first.add(0, values)
    K_first = first.to_csr()

    second = CSRChunkAccumulator(pattern)
    second.add(0, 2.0 * values)
    K_second = second.to_csr()

    np.testing.assert_allclose(bm.to_numpy(K_first.to_dense()), dense, atol=1e-14)
    np.testing.assert_allclose(
        bm.to_numpy(K_second.to_dense()), 2.0 * dense, atol=1e-14
    )


def test_chunk_accumulator_rejects_invalid_sequences():
    """起点非 0, 批次不连续, 覆盖不全, 空累加, 产出后续加和缓冲区长度不符都应报错."""
    pattern, values, _ = _ring_dofmap_case()

    with pytest.raises(ValueError, match="chunks 必须从 0 开始"):
        CSRChunkAccumulator(pattern).add(1, values[1:])

    gapped = CSRChunkAccumulator(pattern)
    gapped.add(0, values[:1])
    with pytest.raises(ValueError, match="chunks 必须连续覆盖"):
        gapped.add(2, values[2:])

    partial = CSRChunkAccumulator(pattern)
    partial.add(0, values[:3])
    with pytest.raises(ValueError, match="chunks 只覆盖 3 个局部实体"):
        partial.to_csr()

    with pytest.raises(ValueError, match="chunks 不能为空"):
        CSRChunkAccumulator(pattern).to_csr()

    finished = CSRChunkAccumulator(pattern)
    finished.add(0, values)
    finished.to_csr()
    with pytest.raises(RuntimeError, match="不能继续累加"):
        finished.add(4, values[:1])

    with pytest.raises(ValueError, match="buffer 长度必须为"):
        CSRChunkAccumulator(pattern, buffer=np.zeros(pattern.nnz + 1)).add(0, values)

    with pytest.raises(ValueError, match="局部矩阵尾部形状"):
        CSRChunkAccumulator(pattern).add(0, values[:, :2, :2])
