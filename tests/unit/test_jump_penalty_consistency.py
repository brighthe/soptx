"""跳量稳定化项的相容性测试.

稳定化项 ``c_h(u, v) = sum_F alpha * h_F * int_F [[u]] : [[v]] ds`` 只有在跳量对
全局连续场恒为零时才是相容的. 该性质一旦被破坏 (例如面两侧单元的积分点未对齐,
导致 ``w^+`` 与 ``w^-`` 取在互为镜像的物理点上), 位移与应力的 L2 收敛阶仍然正常,
只有 ``div sigma_h`` 掉一阶, 极难通过常规收敛表察觉, 故在此单独设卡.
"""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TriangleMesh

from soptx.fem.integrators import JumpPenaltyIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


def _continuous_field(displacement_degree: int):
    """构造次数不超过位移空间的全局连续多项式场.

    场必须落在位移空间内, 否则逐单元插值的结果本身就是间断的, 跳量理应非零.
    位移空间为 ``P_0`` 时只有常值场满足该要求.
    """
    if displacement_degree == 0:
        def field(points):
            ones = bm.ones_like(points[..., 0])
            return bm.stack((ones, 2.0 * ones), axis=-1)
    elif displacement_degree == 1:
        def field(points):
            x, y = points[..., 0], points[..., 1]
            return bm.stack((x + 2.0 * y, 3.0 * x - y), axis=-1)
    else:
        def field(points):
            x, y = points[..., 0], points[..., 1]
            return bm.stack(
                (x**2 + 2.0 * x * y - y**2, 3.0 * x**2 - x * y + y),
                axis=-1,
            )

    return field


def _internal_face_penalty(mesh, degree: int, method: str):
    """只在内部面装配跳量惩罚块, 并返回连续场对应的自由度向量."""
    scalar_space = LagrangeFESpace(mesh, p=degree - 1, ctype="D")
    space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 2))

    face2cell = mesh.face_to_cell()
    internal = bm.nonzero(face2cell[:, 0] != face2cell[:, 1])[0]

    material = IsotropicLinearElasticMaterial(
        hypothesis="plane_strain",
        lame_lambda=1.0,
        shear_modulus=0.5,
        enable_logging=False,
    )
    form = BilinearForm(space)
    form.add_integrator(
        JumpPenaltyIntegrator(
            q=2 * degree + 2,
            threshold=internal,
            method=method,
            material=material,
            penalty_scaling="physical_h",
        )
    )
    matrix = form.assembly(format="csr").to_scipy()
    vector = bm.to_numpy(space.interpolate(_continuous_field(degree - 1))[:])

    return matrix, vector


@pytest.mark.parametrize("method", ("matrix_jump", "vector_jump"))
@pytest.mark.parametrize("degree", (1, 2, 3))
@pytest.mark.parametrize("nx", (4, 7))
def test_penalty_vanishes_on_globally_continuous_fields(
    method: str, degree: int, nx: int
) -> None:
    """连续场落在惩罚块的零空间上, 即 ``v^T J v`` 为机器零.

    ``nx`` 取偶数与奇数各一, 以覆盖两种局部面定向的分布.
    """
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=nx)

    matrix, vector = _internal_face_penalty(mesh, degree, method)

    quadratic = float(vector @ (matrix @ vector))
    reference = float(vector @ vector)

    assert abs(quadratic) <= 1e-12 * reference, (
        f"{method} (k={degree}, nx={nx}) 的跳量惩罚对连续场不为零: "
        f"v^T J v = {quadratic:.6e}, |v|^2 = {reference:.6e}; "
        "稳定化项不相容, 会污染 div sigma_h 的收敛阶"
    )


@pytest.mark.parametrize("method", ("matrix_jump", "vector_jump"))
def test_penalty_is_not_identically_zero(method: str) -> None:
    """反向对照: 惩罚块本身非退化, 上一测试不是被零矩阵蒙混过关."""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=4, ny=4)

    matrix, _ = _internal_face_penalty(mesh, 2, method)

    assert matrix.nnz > 0
    assert np.abs(matrix.data).max() > 1e-8


@pytest.mark.parametrize("nx", (4, 7))
def test_both_sides_of_a_face_see_the_same_trace(nx: int) -> None:
    """内部面两侧单元对同一连续场取到相同的迹.

    这是相容性的几何根因: 若两侧的面积分点未对齐, 取到的是互为镜像的物理点,
    跳量装配随即退化为非跳量形式. 该测试直接调用生产代码的定向映射, 以便缺陷
    复现时能定位到 ``_oriented_cell_basis`` 而不止于惩罚块的数值.
    """
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=nx)

    scalar_space = LagrangeFESpace(mesh, p=1, ctype="D")
    space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 2))
    uh = bm.to_numpy(space.interpolate(_continuous_field(1))[:])
    cell2dof = bm.to_numpy(space.cell_to_dof())

    integrator = JumpPenaltyIntegrator(q=6, method="matrix_jump")
    bcs, weights = mesh.quadrature_formula(6, "face").get_quadrature_points_and_weights()

    cell2face = bm.to_numpy(mesh.cell_to_face())
    face2cell = bm.to_numpy(mesh.face_to_cell())
    number_of_cells = mesh.number_of_cells()
    number_of_faces = mesh.number_of_faces()

    traces = np.full((number_of_faces, 2, len(weights), 2), np.nan)
    seen = np.zeros((number_of_faces, 2), dtype=bool)

    for local_face in range(3):
        phi = bm.to_numpy(integrator._oriented_cell_basis(space, bcs, local_face))
        value = np.einsum("cj, cqjd -> cqd", uh[cell2dof], phi)
        for c in range(number_of_cells):
            f = cell2face[c, local_face]
            side = 0 if not seen[f, 0] else 1
            traces[f, side] = value[c]
            seen[f, side] = True

    internal = np.nonzero(face2cell[:, 0] != face2cell[:, 1])[0]
    deviation = np.abs(traces[internal, 0] - traces[internal, 1]).max()

    assert deviation < 1e-12, (
        f"内部面两侧取到的迹不一致 (最大偏差 {deviation:.3e}), "
        "面积分点未对齐, 跳量装配会退化为非跳量形式"
    )
