"""FEALPy 张量积网格缺陷的最小复现.

只依赖 FEALPy 自身接口。四项判据均取自有限元的基础性质, 与具体 PDE 无关:

  A. 线性函数的节点插值必须被 basis 精确重现
  B. 线性函数的梯度必须被 grad_basis 精确重现
  C. sum_q ws * detJ 必须等于 entity_measure
  D. 扭曲(非矩形)单元上 jacobi_matrix 必须与 bc_to_point 的数值微分一致
  E. bc_to_point 返回 (NC, NQ, GD), 积分点维与 ws 一致
  F. mesh.error 对单纯形网格可用(缺陷 1, 影响所有网格类型)

判据 D 不可省略: 均匀矩形网格会掩盖节点错位。
判据 F 是其余判据的前置条件: 它失败时 mesh.error 在三角形上就会抛异常。

运行::

    python reproduce_tensor_product_issue.py
"""

from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

bm.set_backend("numpy")

TOL_EXACT = 1e-12
TOL_GRAD = 1e-10
failures: list[str] = []


def record(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"  [{tag}] {name}: {detail}")


def jacobi(mesh, bcs):
    """兼容新旧两种入口."""
    try:
        return mesh.Entity("cell").jacobi_matrix(bcs)
    except AttributeError:
        return mesh.jacobi_matrix(bcs)


def flat_points(mesh, bcs):
    pts = np.asarray(mesh.bc_to_point(bcs))
    if pts.ndim > 3:  # (NC, nq_x, nq_y, nq_z, GD) -> (NC, NQ, GD)
        pts = pts.reshape(pts.shape[0], -1, pts.shape[-1])
    return pts


def linear(coefficients):
    def f(p):
        return sum(c * p[..., i] for i, c in enumerate(coefficients))

    def grad(p):
        return np.stack(
            [np.full(p.shape[:-1], c) for c in coefficients], axis=-1
        )

    return f, grad


def check_mesh(name, mesh, coefficients):
    print(f"\n{name}")
    f, gradf = linear(coefficients)
    qf = mesh.quadrature_formula(4)
    bcs, ws = qf.get_quadrature_points_and_weights()

    # --- E ---  积分点维必须已展平, 与 ws 对应
    raw = np.asarray(mesh.bc_to_point(bcs))
    expected = (mesh.number_of_cells(), len(ws), mesh.geo_dimension())
    record(
        f"{name} E bc_to_point 形状",
        raw.shape == expected,
        f"{raw.shape}, 应为 {expected}",
    )

    # --- A / B ---
    space = LagrangeFESpace(mesh, p=1, ctype="C")
    c2d = np.asarray(space.cell_to_dof())
    pts = flat_points(mesh, bcs)
    uh = space.function()
    uh[:] = bm.asarray(f(np.asarray(mesh.entity("node"))))
    dofs = np.asarray(uh)[c2d]

    phi = np.asarray(space.basis(bcs))
    if phi.shape[0] == 1 and dofs.shape[0] != 1:
        phi = np.broadcast_to(phi, (dofs.shape[0],) + phi.shape[1:])
    err = np.abs(np.einsum("cqi,ci->cq", phi, dofs) - f(pts)).max()
    record(f"{name} A 插值一致性", err < TOL_EXACT, f"max err = {err:.3e}")

    gphi = np.asarray(space.grad_basis(bcs, variable="x"))
    gerr = np.abs(np.einsum("cqid,ci->cqd", gphi, dofs) - gradf(pts)).max()
    record(f"{name} B 梯度一致性", gerr < TOL_GRAD, f"max err = {gerr:.3e}")

    # --- C ---  单纯形的 detJ 是 d! * 测度, 且装配不走 detJ, 故跳过
    if name in ("quadrangle", "hexahedron"):
        detJ = bm.abs(bm.linalg.det(jacobi(mesh, bcs)))
        total = float(bm.einsum("q,cq->c", ws, detJ)[0])
        cm = float(mesh.entity_measure("cell")[0])
        record(
            f"{name} C 面积一致性",
            abs(total - cm) < TOL_EXACT,
            f"sum ws*detJ = {total:.10f}, entity_measure = {cm:.10f}",
        )


def check_distorted_quad():
    """判据 D: 以 bc_to_point 的数值微分为真值."""
    print("\ndistorted quadrangle")
    node = np.array([[0.0, 0.0], [1.0, 0.2], [1.3, 1.1], [0.1, 0.9]])
    mesh = QuadrangleMesh(bm.asarray(node), bm.asarray(np.array([[0, 1, 2, 3]])))

    def bcs_at(xi, eta):
        return (
            bm.asarray([[1.0 - xi, xi]], dtype=bm.float64),
            bm.asarray([[1.0 - eta, eta]], dtype=bm.float64),
        )

    def point(xi, eta):
        return np.asarray(mesh.bc_to_point(bcs_at(xi, eta))).reshape(-1, 2)[0]

    h = 1e-6
    for xi, eta in ((0.25, 0.4), (0.7, 0.6)):
        J = np.asarray(jacobi(mesh, bcs_at(xi, eta)))
        J = J.reshape(-1, J.shape[-2], J.shape[-1])[0]
        numeric = np.stack(
            [
                (point(xi + h, eta) - point(xi - h, eta)) / (2 * h),
                (point(xi, eta + h) - point(xi, eta - h)) / (2 * h),
            ],
            axis=-1,
        )
        diff = np.abs(J - numeric).max()
        record(
            f"D jacobi_matrix @ ({xi}, {eta})",
            diff < 1e-6,
            f"|det| = {abs(np.linalg.det(J)):.10f}, "
            f"真值 = {abs(np.linalg.det(numeric)):.10f}, "
            f"max|J - J_num| = {diff:.2e}",
        )


def check_mesh_error_on_simplex():
    """判据 F: value/grad_value 的 TD 计算.

    mesh.error 会把单纯形的 bcs 也包装成长度为 1 的 tuple, 若 TD 取 len(bc)
    就会得到 1 而非 2, 导致 einsum 的局部自由度维不匹配。
    """
    print("\nmesh.error on simplex (缺陷 1)")
    from fealpy.functionspace import TensorFunctionSpace

    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    scalar = LagrangeFESpace(mesh, p=1, ctype="C")
    space = TensorFunctionSpace(scalar, shape=(-1, 2))

    def exact(p):
        return bm.stack(
            [2.0 * p[..., 0] + 3.0 * p[..., 1], bm.zeros_like(p[..., 0])],
            axis=-1,
        )

    uh = space.function()
    uh[:] = bm.asarray(np.asarray(exact(mesh.entity("node"))).reshape(-1))
    try:
        err = float(mesh.error(exact, uh, q=4))
        record(
            "F mesh.error 可用 (triangle)",
            err < TOL_GRAD,
            f"插值精确解的误差 = {err:.3e}",
        )
    except Exception as error:
        record(
            "F mesh.error 可用 (triangle)",
            False,
            f"{type(error).__name__}: {error}",
        )


def main() -> int:
    box2 = [0.0, 1.0, 0.0, 1.0]
    box3 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    check_mesh_error_on_simplex()

    check_mesh("triangle", TriangleMesh.from_box(box2, nx=2, ny=2), (2.0, 3.0))
    check_mesh("quadrangle", QuadrangleMesh.from_box(box2, nx=2, ny=2), (2.0, 3.0))
    check_mesh(
        "tetrahedron",
        TetrahedronMesh.from_box(box3, nx=2, ny=2, nz=2),
        (2.0, 3.0, 4.0),
    )
    try:
        check_mesh(
            "hexahedron",
            HexahedronMesh.from_box(box3, nx=2, ny=2, nz=2),
            (2.0, 3.0, 4.0),
        )
    except Exception as error:
        failures.append("hexahedron")
        print(f"  [FAIL] hexahedron: {type(error).__name__}: {error}")

    check_distorted_quad()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
