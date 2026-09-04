"""Tests for :class:`soptx.solvers.CGSolver`.

Covers the branch complementary to ``DirectSolver``: an empty ``requires``,
so a matrix-free operator must be accepted. Also pins the info contract, the
``M=`` slot, preconditioner setup cascading, the vector-layer type
normalisation that lets a fealpy ``Function`` in, and the pass-through of the
PETSc/MFEM-flavoured knobs (``norm_type``, ``divtol``, ``monitor``,
``print_level``) down to :func:`~soptx.solvers.cg.cg`.
"""

from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from soptx.solvers import (
    CGSolver,
    ConvergedReason,
    DiagonalPreconditioner,
    DirectSolver,
    LinearSolver,
)
from soptx.solvers.cg import NORM_TYPES

bm.set_backend("numpy")


def poisson_1d(n: int = 50) -> np.ndarray:
    """SPD tridiagonal 1D Poisson matrix."""
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def graded_spd(n: int = 60) -> np.ndarray:
    """对角尺度差异大的 SPD 矩阵, Jacobi 预条件在其上有明显效果."""
    A = poisson_1d(n)
    scale = np.logspace(0, 3, n)
    return A * scale[:, None] * scale[None, :]


class MatvecOnlyOperator:
    """站位的 'ea' 层级算子: 只支持 ``@``, 给不出矩阵, 也给不出对角."""

    def __init__(self, A) -> None:
        self._A = A

    def __matmul__(self, other):
        return self._A @ other


# --------------------------------------------------------------------------
# 能力协商: requires 为空, matrix-free 算子必须被接受
# --------------------------------------------------------------------------

def test_matrix_free_operator_accepted() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(A.shape[0])

    x, info = CGSolver(rtol=1e-12).setup(MatvecOnlyOperator(A)).solve(b)

    assert info["converged"] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)


def test_solve_before_setup_raises() -> None:
    _, b = poisson_1d(10), np.ones(10)
    with pytest.raises(RuntimeError, match="尚未 setup"):
        CGSolver().solve(b)


def test_negative_residual_refresh_rejected() -> None:
    with pytest.raises(ValueError, match="residual_refresh"):
        CGSolver(residual_refresh=-1)


def test_setup_returns_self_for_chaining() -> None:
    A = poisson_1d(10)
    solver = CGSolver()
    assert solver.setup(A) is solver
    assert solver.is_setup is True


# --------------------------------------------------------------------------
# info 契约
# --------------------------------------------------------------------------

def test_info_carries_required_and_extra_keys() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(1)
    b = rng.standard_normal(A.shape[0])

    _, info = CGSolver(rtol=1e-12).setup(A).solve(b)

    assert info["niter"] > 0
    assert info["converged"] is True
    assert info["relres"] < 1e-10
    # cg 自己的诊断键原样保留
    assert info["breakdown"] is None
    assert "recursive_residual" in info


def test_relres_matches_direct_solver_definition() -> None:
    """两个后端的 relres 必须是同一个量, 否则无法互相对照."""
    A = poisson_1d()
    rng = np.random.default_rng(2)
    b = rng.standard_normal(A.shape[0])

    x_cg, info_cg = CGSolver(rtol=1e-12).setup(A).solve(b)
    manual = np.linalg.norm(b - A @ x_cg) / np.linalg.norm(b)
    assert info_cg["relres"] == pytest.approx(manual, rel=1e-10)


def test_maxit_exhaustion_reports_not_converged() -> None:
    A = poisson_1d(200)
    b = np.ones(A.shape[0])

    _, info = CGSolver(atol=1e-16, rtol=1e-16, maxit=3).setup(A).solve(b)

    assert info["converged"] is False
    assert info["niter"] == 3
    assert info["relres"] > 1e-6


def test_zero_rhs_reports_zero_residual() -> None:
    A = poisson_1d(10)
    x, info = CGSolver().setup(A).solve(np.zeros(10))

    np.testing.assert_allclose(x, np.zeros(10), atol=1e-14)
    assert info["relres"] == 0.0
    assert info["converged"] is True


def test_residual_refresh_reports_true_residual() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(3)
    b = rng.standard_normal(A.shape[0])

    _, info = CGSolver(rtol=1e-12, residual_refresh=5).setup(A).solve(b)

    assert info["true_residual"] is not None
    assert info["converged"] is True


# --------------------------------------------------------------------------
# M= 位
# --------------------------------------------------------------------------

def test_preconditioner_reduces_iterations() -> None:
    A = graded_spd()
    rng = np.random.default_rng(4)
    b = rng.standard_normal(A.shape[0])

    _, plain = CGSolver(rtol=1e-10, maxit=2000).setup(A).solve(b)
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))
    _, preconditioned = CGSolver(M=M, rtol=1e-10, maxit=2000).setup(A).solve(b)

    assert plain["converged"] and preconditioned["converged"]
    assert preconditioned["niter"] < plain["niter"]


def test_setup_cascades_to_unset_preconditioner() -> None:
    A = poisson_1d(20)
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))
    assert M.is_setup is False

    solver = CGSolver(M=M).setup(A)

    assert M.is_setup is True
    assert M.op is A
    assert solver.M is M


def test_setup_leaves_already_setup_preconditioner_alone() -> None:
    """多重网格这类预条件子绑的算子未必是外层 Krylov 的这一个."""
    A = poisson_1d(20)
    other = poisson_1d(20) * 2.0
    M = DiagonalPreconditioner(bm.tensor(np.diag(A))).setup(other)

    CGSolver(M=M).setup(A)

    assert M.op is other


def test_direct_solver_usable_as_preconditioner() -> None:
    """统一类型的意义: 直接法能填进 M= 位, 一步就把残差解干净."""
    from scipy.sparse import csr_matrix

    A = poisson_1d(30)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(A.shape[0])

    M = DirectSolver("scipy").setup(csr_matrix(A))
    assert isinstance(M, LinearSolver)

    x, info = CGSolver(M=M, rtol=1e-12).setup(A).solve(b)

    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)
    # 预条件恰是 A^{-1} 时, CG 一步收敛
    assert info["niter"] == 1


def test_matmul_gives_preconditioner_mode() -> None:
    A = poisson_1d(30)
    rng = np.random.default_rng(6)
    b = rng.standard_normal(A.shape[0])

    solver = CGSolver(rtol=1e-12).setup(A)

    np.testing.assert_allclose(solver @ b, np.linalg.solve(A, b), atol=1e-8)
    np.testing.assert_allclose(solver.apply(b), np.linalg.solve(A, b),
                               atol=1e-8)


# --------------------------------------------------------------------------
# 批量右端项
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch_first", [False, True])
def test_batched_rhs(batch_first: bool) -> None:
    A = poisson_1d(30)
    rng = np.random.default_rng(7)
    b_cols = rng.standard_normal((A.shape[0], 3))
    reference = np.linalg.solve(A, b_cols)

    b = b_cols.T if batch_first else b_cols
    x, info = CGSolver(
        rtol=1e-12, batch_first=batch_first
    ).setup(A).solve(b)

    expected = reference.T if batch_first else reference
    np.testing.assert_allclose(x, expected, atol=1e-8)
    assert info["converged"] is True
    assert info["relres"] < 1e-10


def test_dot_product_rejects_batched_rhs() -> None:
    A = poisson_1d(20)
    b = np.ones((20, 2))
    solver = CGSolver(dot_product=lambda u, v: float(np.sum(u * v))).setup(A)

    with pytest.raises(NotImplementedError, match="dot_product"):
        solver.solve(b)
# --------------------------------------------------------------------------
# 入口类型归一化: 算法里的断言保持严格, 放宽发生在向量层
# --------------------------------------------------------------------------

def test_fealpy_function_accepted_as_rhs() -> None:
    """'ea' 层级的 ``ElasticityEAOperator.assemble()`` 返回的正是 Function.

    它的 MRO 是 ``(Function, Generic, object)``, 不是 ``TensorLike`` 的注册
    子类; 不在 solve 入口归一化, 整条 matrix-free 路径就进不来.
    """
    from fealpy.backend import TensorLike
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    space = LagrangeFESpace(TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4),
                            p=1, ctype="C")
    ndof = space.number_of_global_dofs()
    A = poisson_1d(ndof)
    values = np.random.default_rng(10).standard_normal(ndof)
    b = space.function()
    b[:] = bm.tensor(values)
    assert not isinstance(b, TensorLike), "Function 若已是 TensorLike, 本用例失去意义"

    x, info = CGSolver(rtol=1e-12).setup(A).solve(b)

    assert info["converged"] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, values), atol=1e-8)


def test_plain_list_accepted_as_rhs() -> None:
    A = poisson_1d(10)
    x, info = CGSolver(rtol=1e-12).setup(A).solve([1.0] * 10)

    assert info["converged"] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, np.ones(10)), atol=1e-8)


def test_function_accepted_as_initial_guess() -> None:
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    space = LagrangeFESpace(TriangleMesh.from_box([0, 1, 0, 1], nx=3, ny=3),
                            p=1, ctype="C")
    ndof = space.number_of_global_dofs()
    A = poisson_1d(ndof)
    rng = np.random.default_rng(11)
    b = rng.standard_normal(ndof)
    x0 = space.function()

    x, info = CGSolver(rtol=1e-12).setup(A).solve(b, x0=x0)

    assert info["converged"] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)


# --------------------------------------------------------------------------
# 退出原因与 info 契约
# --------------------------------------------------------------------------

def test_reason_is_carried_through_and_agrees_with_converged() -> None:
    A = graded_spd()
    b = np.random.default_rng(12).standard_normal(A.shape[0])

    _, ok = CGSolver(rtol=1e-10, maxit=5000).setup(A).solve(b)
    _, cut = CGSolver(rtol=1e-16, atol=1e-16, maxit=3).setup(A).solve(b)

    assert ok["reason"] == ConvergedReason.CONVERGED_RTOL
    assert cut["reason"] == ConvergedReason.DIVERGED_ITS
    for info in (ok, cut):
        assert (int(info["reason"]) > 0) is bool(info["converged"])


def test_inconsistent_reason_is_rejected_by_the_base_class() -> None:
    """契约在基类强制, 不靠各实现自觉 -- 这里直接构造一个自相矛盾的 info."""
    class _Liar(LinearSolver):
        def _solve(self, b, x0=None, **kwargs):
            return b, {"niter": 1, "relres": 0.0, "converged": True,
                       "reason": ConvergedReason.DIVERGED_ITS}

    with pytest.raises(ValueError, match="自相矛盾"):
        _Liar().setup(poisson_1d(5)).solve(np.ones(5))


def test_reference_norm_tracks_the_initial_residual() -> None:
    """rtol 的参照量是 ||r0||, 热启动下它必须随之下降, 不是 ||b||."""
    A = poisson_1d()
    b = np.random.default_rng(13).standard_normal(A.shape[0])
    exact = np.linalg.solve(A, b)

    solver = CGSolver(rtol=1e-12).setup(A)
    _, cold = solver.solve(b)
    _, warm = solver.solve(b, x0=exact * (1.0 + 1e-4))

    assert cold["reference_norm"] == pytest.approx(np.linalg.norm(b),
                                                   rel=1e-10)
    assert warm["reference_norm"] < 1e-2 * cold["reference_norm"]


# --------------------------------------------------------------------------
# norm_type / divtol / monitor 的透传与校验
# --------------------------------------------------------------------------

@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_type_passthrough(norm_type: str) -> None:
    A = graded_spd()
    b = np.random.default_rng(14).standard_normal(A.shape[0])
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))

    x, info = CGSolver(M=M, rtol=1e-10, maxit=5000,
                       norm_type=norm_type).setup(A).solve(b)

    assert info["converged"] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-6)


def test_unpreconditioned_norm_makes_relres_and_criterion_one_measure() -> None:
    """判据与 relres 同为 2-范数时, 停机点的 relres 必须真的达到 rtol.

    natural 范数下 Jacobi 的 diag^-1 会把判据左边放大若干量级, 两个口径
    对不上 -- 这正是 analyzer 的 jacobi 分支显式选 unpreconditioned 的理由.
    """
    A = graded_spd()
    b = np.random.default_rng(15).standard_normal(A.shape[0])
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))

    _, info = CGSolver(M=M, rtol=1e-8, maxit=5000,
                       norm_type="unpreconditioned").setup(A).solve(b)

    assert info["converged"] is True
    assert info["relres"] <= 1e-8


def test_divtol_passthrough_reports_diverged_dtol() -> None:
    A = graded_spd()
    b = np.random.default_rng(16).standard_normal(A.shape[0])

    _, info = CGSolver(maxit=500, divtol=1e-3).setup(A).solve(b)

    assert info["reason"] == ConvergedReason.DIVERGED_DTOL
    assert info["converged"] is False


def test_monitor_and_print_level_passthrough() -> None:
    A = poisson_1d()
    b = np.random.default_rng(17).standard_normal(A.shape[0])
    calls = []

    solver = CGSolver(rtol=1e-12, monitor=lambda *args: calls.append(args),
                      print_level=0).setup(A)
    x, info = solver.solve(b)

    # it=0 一次, 每步一次, 停机后 final 一次
    assert len(calls) == info["niter"] + 2
    assert calls[-1][3] is True
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)


@pytest.mark.parametrize("keywords,pattern", [
    (dict(norm_type="bogus"), "norm_type"),
    (dict(divtol=-1.0), "divtol"),
    (dict(print_level=-1), "print_level"),
])
def test_constructor_rejects_bad_arguments(keywords: dict,
                                           pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        CGSolver(**keywords)


# --------------------------------------------------------------------------
# 批量右端项按 N 次独立求解处理
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch_first", [False, True])
def test_column_reasons_exposed_for_batched_rhs(batch_first: bool) -> None:
    A = poisson_1d(30)
    rng = np.random.default_rng(18)
    columns = rng.standard_normal((A.shape[0], 3))
    b = columns.T if batch_first else columns

    _, info = CGSolver(rtol=1e-12,
                       batch_first=batch_first).setup(A).solve(b)

    assert len(info["column_reasons"]) == 3
    assert all(code > 0 for code in info["column_reasons"])
    assert info["converged"] is True


def test_batch_is_not_converged_unless_every_column_is() -> None:
    """一列 NaN 不该被其余列的成功掩盖, 也不该拖垮其余列."""
    A = poisson_1d(30)
    rng = np.random.default_rng(19)
    good = rng.standard_normal(A.shape[0])
    b = np.stack([good, np.full(A.shape[0], np.nan)], axis=1)

    x, info = CGSolver(rtol=1e-12, maxit=500).setup(A).solve(b)

    assert info["converged"] is False
    assert info["column_reasons"][0] > 0
    assert info["column_reasons"][1] == ConvergedReason.DIVERGED_NANORINF
    np.testing.assert_allclose(x[:, 0], np.linalg.solve(A, good), atol=1e-8)
