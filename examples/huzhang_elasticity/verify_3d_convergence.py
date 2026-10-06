"""三维胡张混合元的制造解收敛阶验证.

``HuZhangMFEMAnalyzer`` 的边界装配只实现了二维, 本脚本绕开分析器, 直接用
``BilinearForm`` 与两个胡张积分子装配三维鞍点系统

.. math::

    \\begin{bmatrix} A & B \\\\ B^{\\mathsf T} & 0 \\end{bmatrix}
    \\begin{bmatrix} \\sigma \\\\ u \\end{bmatrix}
    = \\begin{bmatrix} 0 \\\\ -f \\end{bmatrix},

其中 :math:`A` 为柔度项 :math:`(\\mathcal A\\sigma, \\tau)`, :math:`B` 为 :math:`(\\operatorname{div}\\tau, v)`.
制造解 :math:`u_i = c_i \\sin\\pi x \\sin\\pi y \\sin\\pi z` 在边界上为零, 位移边界项消失.
三维跳量稳定化未实现, 只验证 :math:`p \\ge 4` (原生格式稳定的最低次数).

理论阶 (Hu & Zhang): :math:`\\|\\sigma - \\sigma_h\\|_0 = O(h^{p+1})`,
:math:`\\|u - u_h\\|_0 = O(h^{p})`, :math:`\\|\\operatorname{div}(\\sigma - \\sigma_h)\\|_0 = O(h^{p})`.
门禁取最后一对网格的观测阶, 允许比理论阶低 ``ORDER_MARGIN``.

用法::

    PYTHONPATH=src python examples/huzhang_elasticity/verify_3d_convergence.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from soptx.backend import backend_manager as bm
from soptx.decorator import cartesian
from soptx.fem.bilinear_form import BilinearForm
from soptx.fem.integrators import HuZhangMixIntegrator, HuZhangStressIntegrator, SourceIntegrator
from soptx.fem.linear_form import LinearForm
from soptx.functionspace import HuZhangFESpace, LagrangeFESpace, TensorFunctionSpace
from soptx.mesh import TetrahedronMesh
from soptx.sparse.ops import bmat

E, NU = 1.0, 0.3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
C = np.array([1.0, -0.5, 0.25])  # 位移各分量的幅值, 互不相同以免对称性掩盖错误
ORDER_MARGIN = 0.3
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _sin_terms(x):
    """sin(pi x_k), cos(pi x_k) 及 s = prod_k sin(pi x_k)."""
    s_k, c_k = np.sin(np.pi * x), np.cos(np.pi * x)
    return s_k, c_k, np.prod(s_k, axis=-1)


def _grad_s(x):
    """grad s, 形状 (..., 3)."""
    s_k, c_k, _ = _sin_terms(x)
    g = np.empty(x.shape)
    for j in range(3):
        others = [k for k in range(3) if k != j]
        g[..., j] = np.pi * c_k[..., j] * s_k[..., others[0]] * s_k[..., others[1]]
    return g


def _hess_s(x):
    """Hessian of s, 形状 (..., 3, 3)."""
    s_k, c_k, s = _sin_terms(x)
    H = np.empty(x.shape + (3,))
    for i in range(3):
        for j in range(3):
            if i == j:
                H[..., i, j] = -np.pi ** 2 * s
            else:
                k = 3 - i - j
                H[..., i, j] = np.pi ** 2 * c_k[..., i] * c_k[..., j] * s_k[..., k]
    return H


def displacement(x):
    """u = c s, 形状 (..., 3)."""
    return _sin_terms(x)[2][..., None] * C


def stress(x):
    """sigma = mu (c grad s^T + grad s c^T) + lam (c . grad s) I, Voigt [xx, xy, xz, yy, yz, zz]."""
    g = _grad_s(x)
    S = MU * (C[:, None] * g[..., None, :] + g[..., :, None] * C[None, :])
    S = S + LAM * np.einsum("k,...k->...", C, g)[..., None, None] * np.eye(3)
    return np.stack([S[..., 0, 0], S[..., 0, 1], S[..., 0, 2], S[..., 1, 1], S[..., 1, 2], S[..., 2, 2]], axis=-1)


def div_stress(x):
    """div sigma = mu c Δs + (mu + lam) H c."""
    H = _hess_s(x)
    lap = np.trace(H, axis1=-2, axis2=-1)
    return MU * lap[..., None] * C + (MU + LAM) * np.einsum("...ij,j->...i", H, C)


def solve_level(n: int, p: int, q: int) -> dict:
    """在 n x n x n 剖分上求解并返回误差与规模."""
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    space_sigma = HuZhangFESpace(mesh, p=p)
    space_u = TensorFunctionSpace(scalar_space=LagrangeFESpace(mesh, p=p - 1, ctype="D"), shape=(-1, 3))

    t0 = time.perf_counter()
    lambda0, lambda1 = (1 + NU) / E, NU / E
    aform = BilinearForm(space_sigma)
    aform.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1, q=q))
    A = aform.assembly(format="csr", method="coalesce")
    bform = BilinearForm((space_u, space_sigma))
    bform.add_integrator(HuZhangMixIntegrator(q=q))
    B = bform.assembly(format="csr", method="coalesce")
    K = bmat([[A, B], [B.T, None]], format="csr").to_scipy().tocsc()

    @cartesian
    def body_force(points):
        return bm.tensor(-div_stress(bm.to_numpy(points)))

    lform = LinearForm(space_u)
    lform.add_integrator(SourceIntegrator(source=body_force, q=q))
    f = bm.to_numpy(lform.assembly(format="dense"))
    gs = space_sigma.number_of_global_dofs()
    rhs = np.concatenate([np.zeros(gs), -f])
    t1 = time.perf_counter()
    x = spla.spsolve(K, rhs)
    t2 = time.perf_counter()

    sigma_h, u_h = bm.tensor(x[:gs]), bm.tensor(x[gs:])
    bcs, ws = mesh.quadrature_formula(q + 2).get_quadrature_points_and_weights()
    cm = bm.to_numpy(mesh.entity_measure("cell"))
    ws = bm.to_numpy(ws)
    points = bm.to_numpy(mesh.bc_to_point(bcs))
    voigt_weight = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0])

    def l2(values):
        return math.sqrt(float(np.einsum("q,c,cq->", ws, cm, values)))

    e_sigma = bm.to_numpy(space_sigma.value(sigma_h, bcs)) - stress(points)
    e_div = bm.to_numpy(space_sigma.div_value(sigma_h, bcs)) - div_stress(points)
    e_u = bm.to_numpy(space_u.function(u_h)(bcs)) - displacement(points)
    return {
        "n": n,
        "h": 1.0 / n,
        "sigma_dofs": int(gs),
        "u_dofs": int(space_u.number_of_global_dofs()),
        "sigma_L2": l2(np.einsum("cqk,k->cq", e_sigma ** 2, voigt_weight)),
        "u_L2": l2(np.sum(e_u ** 2, axis=-1)),
        "div_sigma_L2": l2(np.sum(e_div ** 2, axis=-1)),
        "assembly_seconds": round(t1 - t0, 3),
        "solve_seconds": round(t2 - t1, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degree", type=int, default=4, help="应力空间次数, 须 >= 4")
    parser.add_argument("--levels", default="2,3,4", help="逗号分隔的每方向剖分数; n=1 在渐近区之外")
    parser.add_argument("--json", type=Path, default=None, help="结果 JSON 路径, 缺省写入 outputs/")
    args = parser.parse_args()
    if args.degree < 4:
        parser.error("三维跳量稳定化未实现, 原生格式要求 degree >= 4.")

    bm.set_backend("numpy")
    p = args.degree
    q = 2 * p
    levels = [int(s) for s in args.levels.split(",")]
    rows = [solve_level(n, p, q) for n in levels]

    keys = ("sigma_L2", "u_L2", "div_sigma_L2")
    expected = {"sigma_L2": p + 1, "u_L2": p, "div_sigma_L2": p}
    for prev, cur in zip(rows, rows[1:]):
        ratio = prev["h"] / cur["h"]
        for k in keys:
            cur[f"{k}_order"] = math.log(prev[k] / cur[k]) / math.log(ratio)

    print(f"三维胡张元 p={p}, 制造解 u_i = c_i sin(pi x) sin(pi y) sin(pi z), c = {C.tolist()}")
    print(f"{'n':>3} {'sigma dofs':>11} {'||s-sh||':>11} {'阶':>6} {'||u-uh||':>11} {'阶':>6} {'||div(s-sh)||':>14} {'阶':>6} {'求解(s)':>8}")
    for r in rows:
        o = [f"{r.get(k + '_order', float('nan')):6.2f}" for k in keys]
        print(f"{r['n']:>3} {r['sigma_dofs']:>11} {r['sigma_L2']:11.3e} {o[0]} {r['u_L2']:11.3e} {o[1]} "
              f"{r['div_sigma_L2']:14.3e} {o[2]} {r['solve_seconds']:8.2f}")

    passed = True
    if len(rows) >= 2:
        last = rows[-1]
        for k in keys:
            ok = last[f"{k}_order"] >= expected[k] - ORDER_MARGIN
            passed &= ok
            print(f"[{'OK' if ok else 'FAIL'}] {k} 末阶 {last[f'{k}_order']:.2f} >= {expected[k]} - {ORDER_MARGIN}")
    else:
        print("[SKIP] 只有一层网格, 不判定收敛阶")

    out = args.json or OUTPUT_DIR / f"huzhang_3d_convergence_p{p}_levels{'-'.join(map(str, levels))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"degree": p, "quadrature_order": q, "E": E, "nu": NU, "c": C.tolist(),
                               "order_margin": ORDER_MARGIN, "passed": bool(passed), "levels": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果写入 {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
