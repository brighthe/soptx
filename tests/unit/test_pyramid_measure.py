"""四棱锥体积的回归测试.

``LagrangePyramidSchema.measure`` 曾按四面体 (0, 1, 3, 4) 与 (0, 3, 2, 4) 剖分求和,
而底面顶点为循环顺序, 只在底面为平行四边形或梯形时恰好正确.
"""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh.schema import LagrangePyramidSchema
from soptx.mesh.storage import EntityContext, EntitySector, MeshBlock


@pytest.fixture(autouse=True)
def reset_backend():
    """每个测试前后重置为 numpy 后端."""
    bm.set_backend("numpy")
    yield
    bm.set_backend("numpy")


def _context(base, apex):
    """由循环顺序的底面四点与顶点构造单个四棱锥的计算上下文."""
    node = np.vstack([np.asarray(base, dtype=np.float64), np.asarray([apex], dtype=np.float64)])
    block = MeshBlock(positions=bm.tensor(node))
    sector = EntitySector(id="pyr", schema=LagrangePyramidSchema(), indices=bm.tensor([[0, 1, 2, 3, 4]]))
    block.add_sector(sector, root=True)
    return EntityContext(block=block, sector=sector)


def _shoelace(base):
    """平面多边形 (z=0) 的面积."""
    x, y = np.asarray(base)[:, 0], np.asarray(base)[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


PLANAR_BASES = {
    "square": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    "trapezoid": [[0, 0, 0], [2, 0, 0], [1.5, 1, 0], [0.5, 1, 0]],
    "general": [[0, 0, 0], [2, 0, 0], [1.7, 1.3, 0], [0.2, 0.8, 0]],
}


@pytest.mark.parametrize("name", PLANAR_BASES, ids=list(PLANAR_BASES))
def test_planar_base_volume_is_exact(name):
    """平面底面的四棱锥体积等于底面积乘高除以 3."""
    base = PLANAR_BASES[name]
    height = 1.7
    ctx = _context(base, [0.9, 0.4, height])
    volume = float(LagrangePyramidSchema.measure(ctx, None)[0])
    assert volume == pytest.approx(_shoelace(base) * height / 3, rel=1e-13)


def test_nonplanar_base_matches_high_order_quadrature():
    """非平面底面时, 体积与高阶积分 Jacobi 行列式的结果一致 (2 点公式已精确)."""
    base = [[0, 0, 0.1], [2, 0, -0.2], [1.7, 1.3, 0.3], [0.2, 0.8, 0.0]]
    ctx = _context(base, [1.0, 0.5, 1.5])
    qf = LagrangePyramidSchema.quadrature_formula(6)
    bcs, ws = qf.get_quadrature_points_and_weights()
    jacobian = bm.to_numpy(LagrangePyramidSchema.jacobi_matrix(ctx, bcs, None))
    reference = np.einsum("q,cq->c", bm.to_numpy(ws), np.abs(np.linalg.det(jacobian)))[0]
    assert float(LagrangePyramidSchema.measure(ctx, None)[0]) == pytest.approx(reference, rel=1e-13)
