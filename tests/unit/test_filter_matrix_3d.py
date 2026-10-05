"""结构化六面体网格过滤矩阵 H 的向量化构建回归测试."""

from __future__ import annotations

from math import ceil

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh

from soptx.topology.filters import Filter, FilterMatrixBuilder, apply_structured_density_filter


CASES = (
    # (nx, ny, nz, hx, hy, hz, rmin)
    (4, 3, 5, 1.0, 1.0, 1.0, 1.5),
    (6, 2, 2, 1.0 / 2, 1.0 / 2, 1.0 / 2, 3.0 / 2),     # MBB 比例, rmin = 3h
    (5, 4, 3, 0.5, 0.7, 0.3, 1.1),                     # 各向异性, 各方向搜索半径不同
)


def _structured_hex_mesh(nx, ny, nz, hx, hy, hz):
    """生成带结构化 meshdata 的六面体网格, 使 FilterMatrixBuilder 走结构化路径."""
    mesh = HexahedronMesh.from_box([0.0, nx * hx, 0.0, ny * hy, 0.0, nz * hz], nx, ny, nz)
    mesh.meshdata = {'nx': nx, 'ny': ny, 'nz': nz, 'hx': hx, 'hy': hy, 'hz': hz}
    return mesh


def _reference_entries(nx, ny, nz, hx, hy, hz, rmin):
    """逐单元循环的参考实现, 与向量化之前的写法相同."""
    sx, sy, sz = ceil(rmin / hx), ceil(rmin / hy), ceil(rmin / hz)
    rows, cols, vals = [], [], []
    for row in range(nx * ny * nz):
        i, j, k = row // (ny * nz), (row % (ny * nz)) // nz, row % nz
        for ii in range(max(0, i - (sx - 1)), min(nx, i + sx)):
            for jj in range(max(0, j - (sy - 1)), min(ny, j + sy)):
                for kk in range(max(0, k - (sz - 1)), min(nz, k + sz)):
                    diff = np.array([ii * hx, jj * hy, kk * hz]) - np.array([i * hx, j * hy, k * hz])
                    factor = rmin - np.sqrt(np.sum(diff * diff))
                    if factor > 0:
                        rows.append(row)
                        cols.append((ii * ny + jj) * nz + kk)
                        vals.append(factor)
    return np.array(rows), np.array(cols), np.array(vals)


@pytest.mark.parametrize('case', CASES)
def test_vectorized_matrix_matches_cell_loop_bitwise(case) -> None:
    bm.set_backend('numpy')
    H = FilterMatrixBuilder(mesh=_structured_hex_mesh(*case[:6]), rmin=case[6],
                            density_location='element').build()

    rows, cols, vals = _reference_entries(*case)
    indices = bm.to_numpy(H.indices)
    np.testing.assert_array_equal(indices[0], rows)
    np.testing.assert_array_equal(indices[1], cols)
    np.testing.assert_array_equal(bm.to_numpy(H.values), vals)
    assert indices.dtype == np.int32


@pytest.mark.parametrize('case', CASES)
def test_density_filter_matches_structured_convolution(case) -> None:
    bm.set_backend('numpy')
    nx, ny, nz, hx, hy, hz, rmin = case
    density_filter = Filter(design_mesh=_structured_hex_mesh(nx, ny, nz, hx, hy, hz), filter_type='density',
                            rmin=rmin, density_location='element', enable_logging=False)
    design = np.random.default_rng(0).uniform(0.0, 1.0, nx * ny * nz)

    physical = density_filter.filter_design_variable(design_variable=bm.tensor(design),
                                                     physical_density=bm.zeros(nx * ny * nz))
    expected = apply_structured_density_filter(design.reshape(nx, ny, nz), rmin=rmin, spacing=(hx, hy, hz))
    np.testing.assert_allclose(bm.to_numpy(physical), expected.reshape(-1), rtol=1e-13, atol=0.0)


def test_density_filter_torch_backend_matches_numpy() -> None:
    pytest.importorskip('torch')
    nx, ny, nz, hx, hy, hz, rmin = CASES[1]
    design = np.random.default_rng(1).uniform(0.0, 1.0, nx * ny * nz)

    results = {}
    try:
        for backend in ('numpy', 'pytorch'):
            bm.set_backend(backend)
            density_filter = Filter(design_mesh=_structured_hex_mesh(nx, ny, nz, hx, hy, hz),
                                    filter_type='density', rmin=rmin, density_location='element',
                                    enable_logging=False)
            physical = density_filter.filter_design_variable(
                design_variable=bm.tensor(design, dtype=bm.float64),
                physical_density=bm.zeros(nx * ny * nz, dtype=bm.float64))
            results[backend] = np.asarray(bm.to_numpy(physical))
    finally:
        bm.set_backend('numpy')

    np.testing.assert_allclose(results['pytorch'], results['numpy'], rtol=1e-13, atol=0.0)
