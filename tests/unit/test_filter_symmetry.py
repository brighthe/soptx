"""过滤矩阵设计对称化 (坐标折叠, PolyFilter 的 Sym 选项) 的回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh import TriangleMesh
from soptx.topology.filters import Filter, FilterMatrixBuilder

BOX = (0.0, 2.0, 0.0, 1.0)
NX, NY = 8, 4            # 偶数剖分的棋盘格网格关于 x = 1 与 y = 0.5 都镜像对称
RMIN = 0.4


def _mirror_cell_map(mesh, axis: int, value: float) -> np.ndarray:
    """按单元形心匹配镜像单元: 单元 i 的镜像是形心关于对称面反射后最近的单元."""
    centers = bm.to_numpy(mesh.entity_barycenter('cell'))
    reflected = centers.copy()
    reflected[:, axis] = 2.0 * value - reflected[:, axis]
    dist = np.linalg.norm(centers[None, :, :] - reflected[:, None, :], axis=2)
    mapping = np.argmin(dist, axis=1)
    assert np.all(dist[np.arange(mapping.size), mapping] < 1e-12), "网格不关于该平面对称"
    return mapping


def _filtered(mesh, design_variable: np.ndarray, **kwargs) -> np.ndarray:
    density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=RMIN,
                            density_location='element', filter_q=3, **kwargs)
    rho = density_filter.filter_design_variable(
        bm.array(design_variable), bm.zeros(design_variable.shape[0], dtype=bm.float64))
    return bm.to_numpy(rho)


@pytest.mark.parametrize('axis, value', [(0, 1.0), (1, 0.5)])
def test_folded_filter_output_is_mirror_symmetric(axis: int, value: float) -> None:
    bm.set_backend('numpy')
    mesh = TriangleMesh.from_checkerboard_box(BOX, NX, NY)
    mapping = _mirror_cell_map(mesh, axis, value)
    rng = np.random.default_rng(0)
    design_variable = rng.random(mesh.number_of_cells())

    plain = _filtered(mesh, design_variable)
    folded = _filtered(mesh, design_variable, symmetry_planes={axis: value})

    # 随机设计不对称, 普通过滤不会把它变对称; 折叠过滤的输出严格镜像对称
    assert np.abs(plain - plain[mapping]).max() > 1e-3
    np.testing.assert_allclose(folded, folded[mapping], rtol=0.0, atol=1e-13)
    # 折叠过滤仍是行归一化的平均: 输出落在设计变量的取值范围内
    assert folded.min() >= design_variable.min() - 1e-13
    assert folded.max() <= design_variable.max() + 1e-13


def test_folded_rows_of_mirror_cells_are_identical() -> None:
    bm.set_backend('numpy')
    mesh = TriangleMesh.from_checkerboard_box(BOX, NX, NY)
    mapping = _mirror_cell_map(mesh, 0, 1.0)
    H = FilterMatrixBuilder(mesh=mesh, rmin=RMIN, density_location='element', q=3,
                            symmetry_planes={0: 1.0}).build()
    dense = np.zeros(H.sparse_shape)
    indices, values = bm.to_numpy(H.indices), bm.to_numpy(H.values)
    np.add.at(dense, (indices[0], indices[1]), values)
    # 镜像单元的权重行相同 (折叠后坐标相同), 且每行关于列的镜像也对称
    np.testing.assert_allclose(dense, dense[mapping], atol=1e-13)
    np.testing.assert_allclose(dense, dense[:, mapping], atol=1e-13)


def test_symmetry_planes_forces_general_path_and_validates() -> None:
    bm.set_backend('numpy')
    mesh = TriangleMesh.from_checkerboard_box(BOX, NX, NY)
    with pytest.raises(ValueError):
        FilterMatrixBuilder(mesh=mesh, rmin=RMIN, density_location='element', symmetry_planes={2: 0.0})
    with pytest.raises(ValueError):
        FilterMatrixBuilder(mesh=mesh, rmin=RMIN, density_location='element', symmetry_planes={0: float('nan')})
    # 带结构化 meshdata 的网格在给定对称面时不得走结构化快路径 (单元数不匹配会在快路径上出错)
    mesh.meshdata = {'nx': NX, 'ny': NY, 'hx': 0.25, 'hy': 0.25}
    H = FilterMatrixBuilder(mesh=mesh, rmin=RMIN, density_location='element', q=3,
                            symmetry_planes={0: 1.0}).build()
    assert tuple(H.sparse_shape) == (mesh.number_of_cells(), mesh.number_of_cells())
