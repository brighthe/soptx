"""子结构局部刚度流式装配测试."""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import (
    ExactSchurReduction,
    GlobalAssembler,
    LinearCornerTraceBasis,
    SubstructurePrototype,
    build_substructures,
    iter_exact_element_energy_batches,
    iter_exact_trace_stiffness_batches,
)


@pytest.fixture
def prototype() -> SubstructurePrototype:
    """构造带少量单元的 2D 参考子结构."""
    bm.set_backend("numpy")
    result = SubstructurePrototype(
        cell_size=(1.0, 1.0),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    result.penal = 3.0
    result.rho_min = 1.0e-3
    return result


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
def test_streamed_batches_match_full_assembly(
    prototype: SubstructurePrototype,
    chunk_size: int,
) -> None:
    """不同批大小的流式结果应与完整批量装配一致."""
    density = np.linspace(0.2, 0.95, 6 * 4).reshape(2, 3, 2, 2)
    expected = prototype.assemble_local_stiffness_batch(density)

    chunks = list(
        prototype.iter_local_stiffness_batches(
            density,
            chunk_size=chunk_size,
        )
    )
    actual = bm.concat([chunk for _, _, chunk in chunks], axis=0)

    expected_ranges = [
        (start, min(start + chunk_size, 6))
        for start in range(0, 6, chunk_size)
    ]
    assert [(start, end) for start, end, _ in chunks] == expected_ranges
    assert all(tuple(chunk.shape) == (
        end - start,
        prototype.n_total_dofs,
        prototype.n_total_dofs,
    ) for start, end, chunk in chunks)
    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected).reshape(
            6,
            prototype.n_total_dofs,
            prototype.n_total_dofs,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_streamed_batches_keep_short_final_batch(
    prototype: SubstructurePrototype,
) -> None:
    """子结构数不能整除批大小时应保留最后一个短批次."""
    density = np.full((5, 2, 2), 0.6)

    chunks = list(
        prototype.iter_local_stiffness_batches(density, chunk_size=2)
    )

    assert [(start, end) for start, end, _ in chunks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [chunk.shape[0] for _, _, chunk in chunks] == [2, 2, 1]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_streamed_batches_reject_nonpositive_chunk_size(
    prototype: SubstructurePrototype,
    chunk_size: int,
) -> None:
    """非正批大小应在开始装配前明确报错."""
    density = np.full((2, 2, 2), 0.5)

    with pytest.raises(ValueError, match="chunk_size 必须为正整数"):
        list(
            prototype.iter_local_stiffness_batches(
                density,
                chunk_size=chunk_size,
            )
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_exact_trace_batches_match_full_reduction(
    chunk_size: int,
) -> None:
    """流式 Exact Schur 和角点投影应与完整批量路径一致."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(3.0, 1.0),
        n_sub=(3, 1),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    prototype, _, _ = build_substructures(assembler)
    prototype.penal = 3.0
    prototype.rho_min = 1.0e-3
    density = np.linspace(0.35, 0.9, 12).reshape(3, 2, 2)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)

    local_full = prototype.assemble_local_stiffness_batch(density)
    reduced_full = ExactSchurReduction(
        prototype.i_dofs,
        prototype.b_dofs,
    ).reduce_many(local_full)
    expected = trace_basis.project_stiffness(reduced_full.stiffness)

    batches = list(
        iter_exact_trace_stiffness_batches(
            prototype,
            density,
            trace_basis,
            chunk_size=chunk_size,
        )
    )
    actual = bm.concat([batch.stiffness for batch in batches], axis=0)

    assert [(batch.start, batch.end) for batch in batches] == [
        (start, min(start + chunk_size, 3))
        for start in range(0, 3, chunk_size)
    ]
    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_streamed_macro_assembly_matches_full_batch(
    chunk_size: int,
) -> None:
    """流式宏观散加应与完整 ``K_trace_batch`` 装配一致."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(3.0, 1.0),
        n_sub=(3, 1),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    density = np.linspace(0.3, 0.95, 12).reshape(3, 2, 2)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)

    local_full = prototype.assemble_local_stiffness_batch(density)
    reduced_full = ExactSchurReduction(
        prototype.i_dofs,
        prototype.b_dofs,
    ).reduce_many(local_full)
    projected_full = trace_basis.project_stiffness(reduced_full.stiffness)
    expected = assembler.assemble_macro_system(sub_meshes, projected_full)

    batches = iter_exact_trace_stiffness_batches(
        prototype,
        density,
        trace_basis,
        chunk_size=chunk_size,
    )
    actual = assembler.assemble_macro_system_batches(sub_meshes, batches)

    np.testing.assert_array_equal(
        bm.to_numpy(actual.global_dofs),
        bm.to_numpy(expected.global_dofs),
    )
    np.testing.assert_allclose(
        bm.to_numpy(actual.stiffness.to_dense()),
        bm.to_numpy(expected.stiffness.to_dense()),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_streamed_element_energy_matches_full_recovery(
    chunk_size: int,
) -> None:
    """流式恢复和单元能量应与完整批量恢复路径一致."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler(
        domain_size=(3.0, 1.0),
        n_sub=(3, 1),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    prototype, _, _ = build_substructures(assembler)
    prototype.penal = 3.0
    prototype.rho_min = 1.0e-3
    density = np.linspace(0.32, 0.91, 12).reshape(3, 2, 2)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
    trace_displacement = np.linspace(
        -0.2,
        0.3,
        3 * trace_basis.n_trace_dofs,
    ).reshape(3, trace_basis.n_trace_dofs)

    local_full = prototype.assemble_local_stiffness_batch(density)
    reduced_full = ExactSchurReduction(
        prototype.i_dofs,
        prototype.b_dofs,
    ).reduce_many(local_full)
    u_boundary = trace_basis.expand_displacement(trace_displacement)
    u_internal = reduced_full.recover(u_boundary)
    u_local = bm.zeros(
        (3, prototype.n_total_dofs),
        dtype=bm.float64,
    )
    u_local = bm.set_at(
        u_local,
        (slice(None), prototype.b_dofs),
        u_boundary,
    )
    u_local = bm.set_at(
        u_local,
        (slice(None), prototype.i_dofs),
        u_internal,
    )
    u_element = u_local[:, prototype.cell2dof]
    expected = bm.sum(
        (u_element @ prototype.KE_unit[0]) * u_element,
        axis=-1,
    )

    batches = list(
        iter_exact_element_energy_batches(
            prototype,
            density,
            trace_displacement,
            trace_basis,
            chunk_size=chunk_size,
        )
    )
    actual = bm.concat([batch.energy for batch in batches], axis=0)

    assert [(batch.start, batch.end) for batch in batches] == [
        (start, min(start + chunk_size, 3))
        for start in range(0, 3, chunk_size)
    ]
    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_streamed_element_energy_rejects_displacement_shape() -> None:
    """迹位移批量长度或迹自由度数不匹配时应明确失败."""
    bm.set_backend("numpy")
    prototype = SubstructurePrototype(
        cell_size=(1.0, 1.0),
        n_fine=(2, 2),
        E_base=1.0,
        nu=0.3,
    )
    density = np.full((2, 2, 2), 0.6)
    trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
    invalid = np.zeros((1, trace_basis.n_trace_dofs))

    with pytest.raises(ValueError, match="trace_displacement 形状必须为"):
        list(
            iter_exact_element_energy_batches(
                prototype,
                density,
                invalid,
                trace_basis,
                chunk_size=1,
            )
        )
