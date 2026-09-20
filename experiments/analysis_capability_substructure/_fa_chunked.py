"""二维规则 Q1 网格的 FA 分块 pattern 装配。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from fealpy.backend import backend_manager as bm

from soptx.fem.matrix.csr_pattern import (
    assemble_csr_chunks,
    build_csr_pattern,
)
from soptx.fem.substructure import (
    GlobalAssembler,
    build_substructures,
    solve_interface_system,
)
from examples.substructure_elasticity.verify_linear_corner_consistency import (
    make_fa_analyzer,
)
from experiments.analysis_capability_substructure._density_update import (
    _density,
    _free_residual,
    _problem,
)


DEFAULT_CHUNK_SIZE = 65_536


def _as_numpy(value: Any) -> np.ndarray:
    """把后端数组转为 NumPy 数组。"""
    return np.asarray(bm.to_numpy(value))


def _validate_cell_and_dof_ordering(
    analyzer: Any,
    unit_prototype: Any,
    total_fine: Sequence[int],
    domain_size: Sequence[float],
) -> None:
    """核对全网格单元顺序与单位单元局部自由度顺序。"""
    nx, ny = (int(value) for value in total_fine)
    mesh = analyzer.disp_mesh
    cells = _as_numpy(mesh.entity("cell"))
    nodes = _as_numpy(mesh.entity("node"))
    if len(cells) != nx * ny:
        raise AssertionError("全网格单元数与 total_fine 不一致。")

    sample = np.unique(np.asarray(
        [0, ny - 1, ny, (nx // 2) * ny + ny // 2, nx * ny - 1],
        dtype=np.int64,
    ))
    centers = nodes[cells[sample]].mean(axis=1)
    expected = np.column_stack((
        (sample // ny + 0.5) * (float(domain_size[0]) / nx),
        (sample % ny + 0.5) * (float(domain_size[1]) / ny),
    ))
    if not np.allclose(centers, expected, rtol=0.0, atol=1.0e-13):
        raise AssertionError("全网格 cell 编号与密度 C 序展平不一致。")

    full_c2d = _as_numpy(analyzer.tensor_space.cell_to_dof())[0]
    unit_c2d = _as_numpy(unit_prototype.space.cell_to_dof())[0]
    if len(full_c2d) != len(unit_c2d):
        raise AssertionError("全网格与单位单元的局部自由度数不一致。")

    def signature(coordinates: np.ndarray, dofs: np.ndarray) -> np.ndarray:
        node_ids = dofs // 2
        components = dofs % 2
        points = coordinates[node_ids]
        low = points.min(axis=0)
        span = points.max(axis=0) - low
        normalized = (points - low) / span
        return np.column_stack((normalized, components))

    unit_nodes = _as_numpy(unit_prototype.mesh.entity("node"))
    if not np.array_equal(
        signature(nodes, full_c2d),
        signature(unit_nodes, unit_c2d),
    ):
        raise AssertionError("全网格与单位单元的局部自由度顺序不一致。")


def assemble_chunked_fa(
    analyzer: Any,
    density: Any,
    *,
    domain_size: Sequence[float],
    total_fine: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[Any, dict[str, Any]]:
    """用单位单元刚度和 CSR pattern 分块装配二维 FA 刚度矩阵。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数。")
    if len(total_fine) != 2 or len(domain_size) != 2:
        raise ValueError("assemble_chunked_fa 当前仅支持二维。")

    density_np = np.asarray(
        bm.to_numpy(density), dtype=np.float64, order="C"
    )
    if tuple(density_np.shape) != tuple(int(v) for v in total_fine):
        raise ValueError(
            f"density 形状应为 {tuple(total_fine)}，当前为 {density_np.shape}。"
        )

    cell_size = tuple(
        float(length) / int(count)
        for length, count in zip(domain_size, total_fine)
    )
    unit_assembler = GlobalAssembler(
        cell_size,
        (1, 1),
        (1, 1),
        E_base=float(analyzer.pde.E),
        nu=float(analyzer.pde.nu),
    )
    unit_prototype, _, _ = build_substructures(unit_assembler)
    _validate_cell_and_dof_ordering(
        analyzer, unit_prototype, total_fine, domain_size
    )
    ke0 = np.asarray(
        bm.to_numpy(unit_prototype.KE_unit[0]), dtype=np.float64
    )

    pattern = build_csr_pattern(analyzer.tensor_space)
    rho = density_np.reshape(-1)

    def chunks() -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, len(rho), chunk_size):
            stop = min(start + chunk_size, len(rho))
            values = (
                rho[start:stop, None, None] ** 3
            ) * ke0[None, :, :]
            yield start, np.asarray(values, dtype=np.float64, order="C")

    matrix = assemble_csr_chunks(chunks(), pattern)
    return matrix, {
        "method": "pattern_chunks",
        "chunk_size": int(chunk_size),
        "cell_count": int(len(rho)),
        "unit_element_dofs": int(ke0.shape[0]),
        "pattern_nnz": int(pattern.nnz),
        "ordering_checks": "PASS",
    }


def verify_small(
    *,
    n_sub: Sequence[int] = (8, 4),
    n_fine: Sequence[int] = (5, 5),
    chunk_size: int = 127,
) -> dict[str, Any]:
    """在小网格上与生产 FA 全批量装配、位移和能量逐项比较。"""
    bm.set_backend("numpy")
    pde = _problem(2)
    domain_size = tuple(
        pde.domain[2 * index + 1] - pde.domain[2 * index]
        for index in range(2)
    )
    reference = GlobalAssembler(
        domain_size, n_sub, n_fine, E_base=pde.E, nu=pde.nu
    )
    total_fine = tuple(
        int(a * b) for a, b in zip(n_sub, n_fine)
    )
    density = _density(total_fine, n_fine, "pattern_a")

    standard_analyzer = make_fa_analyzer(
        reference.full_mesh,
        pde,
        reference.material,
        solve_method="mumps",
    )
    standard = standard_analyzer.assemble_stiff_matrix(
        rho_val=bm.reshape(density, (-1,))
    ).to_scipy().tocsr()
    standard.sum_duplicates()
    standard.sort_indices()

    chunked_analyzer = make_fa_analyzer(
        reference.full_mesh,
        pde,
        reference.material,
        solve_method="mumps",
    )
    chunked, metadata = assemble_chunked_fa(
        chunked_analyzer,
        density,
        domain_size=domain_size,
        total_fine=total_fine,
        chunk_size=chunk_size,
    )
    chunked_scipy = chunked.to_scipy().tocsr()
    chunked_scipy.sum_duplicates()
    chunked_scipy.sort_indices()

    structure_match = bool(
        np.array_equal(standard.indptr, chunked_scipy.indptr)
        and np.array_equal(standard.indices, chunked_scipy.indices)
    )
    matrix_error = float(
        np.linalg.norm(standard.data - chunked_scipy.data)
        / max(np.linalg.norm(standard.data), np.finfo(float).tiny)
    ) if structure_match else 1.0

    force = standard_analyzer.assemble_external_load()
    prescribed, fixed_mask = standard_analyzer.tensor_space.boundary_interpolate(
        gd=pde.dirichlet_bc,
        threshold=pde.is_dirichlet_boundary(),
        method="interp",
    )
    if np.any(_as_numpy(prescribed)):
        raise AssertionError("校验仅支持齐次 Dirichlet 约束。")
    fixed = bm.nonzero(fixed_mask)[0]
    system_dofs = reference.total_full_dofs
    standard_system = type("System", (), {
        "stiffness": standard,
        "global_dofs": bm.arange(system_dofs, dtype=bm.int64),
    })()
    chunked_system = type("System", (), {
        "stiffness": chunked,
        "global_dofs": bm.arange(system_dofs, dtype=bm.int64),
    })()
    standard_u = solve_interface_system(
        standard_system, force, fixed, solver="mumps"
    )
    chunked_u = solve_interface_system(
        chunked_system, force, fixed, solver="mumps"
    )
    standard_np = _as_numpy(standard_u).reshape(-1)
    chunked_np = _as_numpy(chunked_u).reshape(-1)
    displacement_error = float(
        np.linalg.norm(chunked_np - standard_np)
        / max(np.linalg.norm(standard_np), np.finfo(float).tiny)
    )
    standard_energy = 0.5 * float(bm.dot(force, standard_u))
    chunked_energy = 0.5 * float(bm.dot(force, chunked_u))
    energy_error = abs(chunked_energy - standard_energy) / abs(standard_energy)
    residual = _free_residual(
        chunked_system, chunked_u, force, fixed
    )
    passed = bool(
        structure_match
        and matrix_error <= 1.0e-13
        and displacement_error <= 1.0e-11
        and energy_error <= 1.0e-11
        and residual <= 1.0e-9
    )
    result = {
        "status": "PASS" if passed else "FAILED",
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "global_fine_grid": list(total_fine),
        "chunk": metadata,
        "matrix_structure_match": structure_match,
        "matrix_relative_error": matrix_error,
        "displacement_relative_error": displacement_error,
        "strain_energy_relative_error": energy_error,
        "equilibrium_relative_residual": residual,
        "standard_strain_energy": standard_energy,
        "chunked_strain_energy": chunked_energy,
    }
    if not passed:
        raise AssertionError(json.dumps(result, ensure_ascii=False))
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="校验二维 FA 分块 pattern 装配。"
    )
    parser.add_argument("--verify-small", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.verify_small:
        parser.error("请指定 --verify-small。")
    result = verify_small()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
