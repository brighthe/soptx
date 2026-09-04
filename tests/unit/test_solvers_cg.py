"""Regression tests for :mod:`soptx.solvers` — the CG ported from fealpy.

Covers the serial contract SOPTX now owns: correctness against a dense
reference solve, batched right-hand sides, the Jacobi preconditioner,
true-residual refresh, and maxit exhaustion.  Also pins the semantics
borrowed from PETSc / MFEM: ``norm_type``, ``ConvergedReason``, per-column
freezing of batched right-hand sides, ``divtol``, ``monitor`` and
``print_level``.  The distributed dot_product path is covered separately by
``test_matrix_free_cg.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fealpy import logger
from fealpy.backend import backend_manager as bm

from soptx.solvers import ConvergedReason, DiagonalPreconditioner, cg
from soptx.solvers.cg import NORM_TYPES

# TensorLike 的 isinstance 判定依赖已注册的 backend
bm.set_backend("numpy")


def poisson_1d(n: int = 50) -> np.ndarray:
    """SPD tridiagonal 1D Poisson matrix."""
    A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return A


def test_cg_matches_dense_reference() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(A.shape[0])

    x, info = cg(A, b, atol=1e-12, rtol=1e-12, maxit=1000, returninfo=True)

    assert info["converged"]
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)


@pytest.mark.parametrize("batch_first", [False, True])
def test_cg_batched_rhs(batch_first: bool) -> None:
    A = poisson_1d(30)
    rng = np.random.default_rng(1)
    b_cols = rng.standard_normal((A.shape[0], 3))
    reference = np.linalg.solve(A, b_cols)

    b = b_cols.T.copy() if batch_first else b_cols
    x = cg(A, b, batch_first=batch_first, atol=1e-12, rtol=1e-12, maxit=1000)
    solution = x.T if batch_first else x

    np.testing.assert_allclose(solution, reference, atol=1e-8)


def test_diagonal_preconditioner_reduces_iterations() -> None:
    # 病态但对角占优: Jacobi 应把条件数压回 O(1), 迭代数大幅下降
    n = 60
    scales = np.logspace(0, 6, n)
    A = poisson_1d(n) * np.sqrt(scales[:, None] * scales[None, :])
    rng = np.random.default_rng(2)
    b = rng.standard_normal(n)

    _, plain = cg(A, b, atol=0.0, rtol=1e-10, maxit=10000, returninfo=True)
    M = DiagonalPreconditioner(np.diag(A).copy())
    x, pcg = cg(A, b, M=M, atol=0.0, rtol=1e-10, maxit=10000, returninfo=True)

    assert pcg["converged"]
    assert pcg["niter"] < plain["niter"]
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert residual < 1e-8


def test_diagonal_preconditioner_rejects_nonpositive_diag() -> None:
    with pytest.raises(ValueError, match="非正"):
        DiagonalPreconditioner(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="一维"):
        DiagonalPreconditioner(np.eye(3))


def test_residual_refresh_reports_true_residual() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(3)
    b = rng.standard_normal(A.shape[0])

    x, info = cg(
        A, b, atol=1e-12, rtol=1e-12, maxit=1000,
        residual_refresh=5, returninfo=True,
    )

    assert info["converged"]
    assert info["true_residual"] is not None
    assert info["true_residual"] < max(1e-12, 1e-12 * np.linalg.norm(b))
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-8)


def test_maxit_exhaustion_reports_not_converged() -> None:
    A = poisson_1d()
    rng = np.random.default_rng(4)
    b = rng.standard_normal(A.shape[0])

    _, info = cg(A, b, atol=0.0, rtol=1e-14, maxit=3, returninfo=True)

    assert not info["converged"]
    assert info["niter"] == 3
# --------------------------------------------------------------------------
# norm_type (对齐 PETSc KSPSetNormType)
# --------------------------------------------------------------------------

def spd(n: int = 120, cond_exp: float = 6.0, seed: int = 7) -> np.ndarray:
    """给定条件数量级的稠密 SPD 矩阵."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(np.logspace(0.0, cond_exp, n)) @ Q.T
    return 0.5 * (A + A.T)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_type_is_inert_without_preconditioner(norm_type: str) -> None:
    """M=None 时三个判据范数是同一个量, 结果必须逐位相同."""
    A = spd(60, 4.0, seed=11)
    b = np.random.default_rng(11).standard_normal(A.shape[0])

    reference, ref_info = cg(A, b, rtol=1e-10, maxit=2000, returninfo=True)
    x, info = cg(A, b, rtol=1e-10, maxit=2000, norm_type=norm_type,
                 returninfo=True)

    assert info["niter"] == ref_info["niter"]
    assert np.array_equal(x, reference)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_type_converges_with_preconditioner(norm_type: str) -> None:
    """有预条件时三个范数是不同的量, 但都得收敛到同一个解."""
    A = spd(60, 4.0, seed=12)
    b = np.random.default_rng(12).standard_normal(A.shape[0])
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))

    x, info = cg(A, b, M=M, rtol=1e-10, maxit=2000, norm_type=norm_type,
                 returninfo=True)

    assert info["converged"] and info["reason"] > 0
    relerr = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    assert relerr < 1e-8


def test_unpreconditioned_reference_norm_is_the_two_norm() -> None:
    """x0=0 时 unpreconditioned 的参照量应恰为 ||b||_2."""
    A = spd(60, 4.0, seed=13)
    b = np.random.default_rng(13).standard_normal(A.shape[0])
    M = DiagonalPreconditioner(bm.tensor(np.diag(A)))

    _, info = cg(A, b, M=M, rtol=1e-10, maxit=2000,
                 norm_type="unpreconditioned", returninfo=True)

    assert info["reference_norm"] == pytest.approx(np.linalg.norm(b),
                                                   rel=1e-12)


def test_invalid_norm_type_rejected() -> None:
    A = poisson_1d(10)
    with pytest.raises(ValueError, match="norm_type"):
        cg(A, np.ones(10), norm_type="bogus")


def test_nonpositive_divtol_rejected() -> None:
    A = poisson_1d(10)
    with pytest.raises(ValueError, match="divtol"):
        cg(A, np.ones(10), divtol=0.0)


def test_negative_print_level_rejected() -> None:
    A = poisson_1d(10)
    with pytest.raises(ValueError, match="print_level"):
        cg(A, np.ones(10), print_level=-1)


# --------------------------------------------------------------------------
# 退出原因 (对齐 PETSc KSPConvergedReason)
# --------------------------------------------------------------------------

def test_rtol_exit_reports_converged_rtol() -> None:
    A = poisson_1d()
    b = np.random.default_rng(2).standard_normal(A.shape[0])

    _, info = cg(A, b, atol=1e-30, rtol=1e-10, maxit=1000, returninfo=True)

    assert info["reason"] == ConvergedReason.CONVERGED_RTOL
    assert info["converged"] is True


def test_atol_exit_reports_converged_atol() -> None:
    """atol 松到一开始就成立时, 退出原因必须区分于 rtol."""
    A = poisson_1d()
    b = np.random.default_rng(3).standard_normal(A.shape[0])

    _, info = cg(A, b, atol=1e3, rtol=1e-30, maxit=200, returninfo=True)

    assert info["reason"] == ConvergedReason.CONVERGED_ATOL


def test_maxit_exhaustion_reports_diverged_its() -> None:
    A = poisson_1d(200)

    _, info = cg(A, np.ones(A.shape[0]), atol=1e-16, rtol=1e-16, maxit=3,
                 returninfo=True)

    assert info["reason"] == ConvergedReason.DIVERGED_ITS
    assert info["converged"] is False


def test_nan_rhs_reports_nanorinf() -> None:
    """密度场退化出 NaN 时必须立刻停机, 不是跑满 maxit."""
    A = poisson_1d()
    b = np.random.default_rng(4).standard_normal(A.shape[0])
    b[3] = np.nan

    _, info = cg(A, b, maxit=500, returninfo=True)

    assert info["reason"] == ConvergedReason.DIVERGED_NANORINF
    assert info["converged"] is False
    assert info["niter"] < 500


def test_divtol_reports_diverged_dtol() -> None:
    """divtol 收到 1e-3 时第一步就该触发, 是机制自检不是数值巧合."""
    A = spd(40, 3.0, seed=5)
    b = np.random.default_rng(5).standard_normal(A.shape[0])

    _, info = cg(A, b, maxit=500, divtol=1e-3, returninfo=True)

    assert info["reason"] == ConvergedReason.DIVERGED_DTOL
    assert info["converged"] is False


def test_default_divtol_does_not_fire_on_healthy_solve() -> None:
    A = spd(40, 3.0, seed=5)
    b = np.random.default_rng(5).standard_normal(A.shape[0])

    _, info = cg(A, b, maxit=500, returninfo=True)

    assert info["converged"] and info["reason"] > 0


def test_indefinite_operator_reports_indefinite_mat() -> None:
    rng = np.random.default_rng(6)
    S = np.linalg.qr(rng.standard_normal((30, 30)))[0]
    A = S @ np.diag(np.concatenate([np.ones(29), [-5.0]])) @ S.T
    A = 0.5 * (A + A.T)

    _, info = cg(A, rng.standard_normal(30), maxit=200, returninfo=True)

    assert info["reason"] == ConvergedReason.DIVERGED_INDEFINITE_MAT
    assert info["breakdown"] is not None


def test_reason_and_converged_never_disagree() -> None:
    """约定 reason > 0 等价于 converged; 各条退出路径都得守住."""
    A = spd(40, 3.0, seed=8)
    b = np.random.default_rng(8).standard_normal(A.shape[0])
    cases = [
        dict(rtol=1e-10, maxit=2000),
        dict(rtol=1e-16, atol=1e-16, maxit=3),
        dict(atol=1e3, rtol=1e-30, maxit=200),
        dict(maxit=500, divtol=1e-3),
    ]
    for keywords in cases:
        _, info = cg(A, b, returninfo=True, **keywords)
        assert (int(info["reason"]) > 0) is bool(info["converged"]), keywords


# --------------------------------------------------------------------------
# 逐列冻结: 批量右端项是 N 次独立求解, 不是并排做算术
# --------------------------------------------------------------------------

def _mixed_difficulty_batch():
    """同一算子, 三列难度悬殊: 列 0/1 病态, 列 2 沿特征向量一两步即收敛."""
    A = spd(120, 6.0, seed=7)
    rng = np.random.default_rng(2026)
    _w, V = np.linalg.eigh(A)
    columns = [rng.standard_normal(120), rng.standard_normal(120),
               V[:, 0] * 3.0]
    return A, columns


def test_easy_column_does_not_break_down_the_whole_batch() -> None:
    """回归 D2: 已收敛列的 curvature 下溢曾把整批判成 breakdown."""
    A, columns = _mixed_difficulty_batch()
    B = np.stack(columns, axis=1)

    X, info = cg(A, B, maxit=5000, rtol=1e-8, returninfo=True)

    assert info["converged"] and info["breakdown"] is None
    assert all(code > 0 for code in info["column_reasons"])
    for j in range(3):
        relerr = (np.linalg.norm(A @ X[:, j] - B[:, j])
                  / np.linalg.norm(B[:, j]))
        assert relerr < 1e-6, f"col {j} 未收敛"


def test_batched_columns_match_solving_them_one_by_one() -> None:
    """逐列独立判定的定义: 批量解等于 N 次单列解."""
    A, columns = _mixed_difficulty_batch()
    B = np.stack(columns, axis=1)

    X, _ = cg(A, B, maxit=5000, rtol=1e-8, returninfo=True)

    for j, column in enumerate(columns):
        reference, _ = cg(A, column, maxit=5000, rtol=1e-8, returninfo=True)
        diff = np.linalg.norm(X[:, j] - reference) / np.linalg.norm(reference)
        assert diff < 1e-6, f"col {j} 与单列解不一致"


def test_nan_column_does_not_poison_its_siblings() -> None:
    A, columns = _mixed_difficulty_batch()
    B = np.stack([columns[0], np.full(120, np.nan), columns[2]], axis=1)

    X, info = cg(A, B, maxit=5000, rtol=1e-8, returninfo=True)

    reasons = info["column_reasons"]
    assert reasons[1] == ConvergedReason.DIVERGED_NANORINF
    assert reasons[0] > 0 and reasons[2] > 0
    assert info["converged"] is False
    for j in (0, 2):
        relerr = (np.linalg.norm(A @ X[:, j] - B[:, j])
                  / np.linalg.norm(B[:, j]))
        assert relerr < 1e-6


def test_maxit_freezes_only_the_hard_columns() -> None:
    A, columns = _mixed_difficulty_batch()
    B = np.stack(columns, axis=1)

    _, info = cg(A, B, maxit=40, rtol=1e-8, returninfo=True)

    reasons = info["column_reasons"]
    assert reasons[2] > 0, "易解列应已收敛并冻结"
    assert reasons[0] == ConvergedReason.DIVERGED_ITS
    assert info["converged"] is False


def test_single_column_reports_no_column_reasons() -> None:
    """1-D 右端项走同一套逐列代码, 但报告边界必须折回标量."""
    A = poisson_1d()
    b = np.random.default_rng(9).standard_normal(A.shape[0])

    _, info = cg(A, b, maxit=1000, rtol=1e-8, returninfo=True)

    assert info["column_reasons"] is None
    assert isinstance(info["reason"], ConvergedReason)


# --------------------------------------------------------------------------
# monitor 与 print_level (对齐 MFEM IterativeSolverMonitor / PrintLevel)
# --------------------------------------------------------------------------

def test_monitor_sees_every_iteration_without_changing_results() -> None:
    A = spd(80, 3.0, seed=14)
    b = np.random.default_rng(14).standard_normal(A.shape[0])
    calls = []

    reference, ref_info = cg(A, b, rtol=1e-10, maxit=2000, returninfo=True)
    x, info = cg(A, b, rtol=1e-10, maxit=2000, returninfo=True,
                 monitor=lambda *args: calls.append(args))

    assert np.array_equal(x, reference)
    assert info["niter"] == ref_info["niter"]
    iterations = [call[0] for call in calls]
    # it=0 报初始残差, 随后每步一次, 最后补一次 final
    assert iterations[:-1] == list(range(0, info["niter"] + 1))
    assert iterations[-1] == info["niter"]
    assert [call[3] for call in calls] == [False] * (len(calls) - 1) + [True]
    assert calls[0][1] == pytest.approx(info["reference_norm"], abs=1e-14)
    assert calls[-1][1] == pytest.approx(info["recursive_residual"],
                                         abs=1e-14)


def test_monitor_fires_final_callback_on_failure() -> None:
    A = spd(80, 3.0, seed=15)
    b = np.random.default_rng(15).standard_normal(A.shape[0])
    calls = []

    _, info = cg(A, b, rtol=1e-14, maxit=5, returninfo=True, print_level=0,
                 monitor=lambda *args: calls.append(args))

    assert info["reason"] == ConvergedReason.DIVERGED_ITS
    assert calls[-1][3] is True and calls[-1][0] == info["niter"]


def test_monitor_fires_once_on_the_zero_rhs_early_exit() -> None:
    A = poisson_1d(20)
    calls = []

    _, info = cg(A, np.zeros(20), returninfo=True,
                 monitor=lambda *args: calls.append(args))

    assert info["converged"] is True
    assert len(calls) == 1
    assert calls[0][0] == 0 and calls[0][3] is True


def test_monitor_receives_the_whole_batch_residual() -> None:
    A = spd(80, 3.0, seed=16)
    rng = np.random.default_rng(16)
    B = np.stack([rng.standard_normal(80), rng.standard_normal(80)], axis=1)
    calls = []

    _, info = cg(A, B, rtol=1e-10, maxit=2000, returninfo=True, print_level=0,
                 monitor=lambda *args: calls.append(args))

    assert calls[-1][2].shape == B.shape
    assert calls[0][1] == pytest.approx(info["reference_norm"], abs=1e-14)


@pytest.mark.parametrize("print_level,per_iteration,summary",
                         [(0, False, 0), (1, False, 1), (2, True, 1)])
def test_print_level_controls_log_volume(print_level: int,
                                         per_iteration: bool,
                                         summary: int) -> None:
    A = spd(80, 3.0, seed=17)
    b = np.random.default_rng(17).standard_normal(A.shape[0])
    captured: list = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture()
    previous = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        _, info = cg(A, b, rtol=1e-10, maxit=2000, returninfo=True,
                     print_level=print_level)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    messages = [m for m in captured if m.startswith("CG")]
    per_step = [m for m in messages if m.startswith("CG iteration")]
    # 每步日志含 it=0 那条; final 那次不打, 摘要归 _finalize 负责
    assert len(per_step) == (info["niter"] + 1 if per_iteration else 0)
    assert len(messages) - len(per_step) == summary
