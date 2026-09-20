"""统一迹空间装配入口测试."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import (
    FullTraceBasis,
    GlobalAssembler,
    LinearCornerTraceBasis,
    LocalReductionBatchResult,
    ReductionDiagnostics,
    TraceBasis,
    build_substructures,
)


@pytest.fixture(autouse=True)
def numpy_backend():
    """各测试使用 NumPy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _setup(dim: int):
    """构造小规模规则子结构及确定性局部刚度."""
    assembler = GlobalAssembler(
        (1.0,) * dim,
        (2,) * dim,
        (2,) * dim,
        E_base=1.0,
        nu=0.3,
    )
    prototype, meshes, _ = build_substructures(assembler)
    n_sub = len(meshes)
    n_b = int(prototype.n_b)
    rng = np.random.default_rng(20260914 + dim)
    raw = rng.normal(size=(n_sub, n_b, n_b))
    stiffness = raw + np.swapaxes(raw, -1, -2)
    return assembler, prototype, meshes, bm.asarray(stiffness)


def _source(kind: str, stiffness, n_i: int):
    """按旧批量、逐块、无状态结果或流式契约包装同一刚度."""
    n_sub, n_b, _ = stiffness.shape
    if kind == "batch":
        return SimpleNamespace(K_s=stiffness, recover=lambda value: value)
    if kind == "list":
        return [
            SimpleNamespace(K_s=stiffness[index], recover=lambda value: value)
            for index in range(n_sub)
        ]
    if kind == "result":
        diagnostics = tuple(
            ReductionDiagnostics(
                requested_method="exact_schur",
                stiffness_source="test",
                recovery_source="test",
            )
            for _ in range(n_sub)
        )
        recovery = bm.zeros((n_sub, n_i, n_b), dtype=bm.float64)
        return LocalReductionBatchResult(stiffness, recovery, diagnostics)
    if kind == "stream":
        return SimpleNamespace(
            K_s=None,
            get_chunk_stiffness=lambda start, stop: stiffness[start:stop],
            recover=lambda value: value,
        )
    raise AssertionError(f"未知测试来源: {kind}.")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["batch", "list", "result", "stream"])
def test_full_trace_entry_matches_legacy_interface_assembly(
    dim: int,
    kind: str,
) -> None:
    """full_trace 新入口应保持旧接口矩阵及全局映射."""
    assembler, prototype, meshes, stiffness = _setup(dim)
    expected = assembler.assemble_interface_system(
        meshes, SimpleNamespace(K_s=stiffness, recover=lambda value: value)
    )
    basis = FullTraceBasis.from_prototype(prototype)

    actual = assembler.assemble_trace_system(
        meshes,
        _source(kind, stiffness, int(prototype.n_i)),
        trace_basis=basis,
        chunk_size=3 if kind == "stream" else None,
    )

    np.testing.assert_array_equal(
        bm.to_numpy(actual.global_dofs),
        bm.to_numpy(expected.global_dofs),
    )
    np.testing.assert_allclose(
        actual.stiffness.to_scipy().toarray(),
        expected.stiffness.to_scipy().toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["batch", "list", "result", "stream"])
def test_linear_corner_entry_matches_legacy_macro_assembly(
    dim: int,
    kind: str,
) -> None:
    """linear_corner 新入口应保持旧宏观矩阵及全局映射."""
    assembler, prototype, meshes, stiffness = _setup(dim)
    basis = LinearCornerTraceBasis.from_prototype(prototype)
    expected = assembler.assemble_macro_system(
        meshes, basis.project_stiffness(stiffness)
    )

    actual = assembler.assemble_trace_system(
        meshes,
        _source(kind, stiffness, int(prototype.n_i)),
        trace_basis=basis,
        chunk_size=3 if kind == "stream" else None,
    )

    np.testing.assert_array_equal(
        bm.to_numpy(actual.global_dofs),
        bm.to_numpy(expected.global_dofs),
    )
    np.testing.assert_allclose(
        actual.stiffness.to_scipy().toarray(),
        expected.stiffness.to_scipy().toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_trace_entry_rejects_basis_without_global_mapping() -> None:
    """尚无全局映射规则的自定义迹基应明确失败."""
    assembler, prototype, meshes, stiffness = _setup(2)
    custom = TraceBasis(np.eye(int(prototype.n_b)))

    with pytest.raises(TypeError, match="当前仅支持"):
        assembler.assemble_trace_system(
            meshes,
            SimpleNamespace(K_s=stiffness, recover=lambda value: value),
            trace_basis=custom,
        )