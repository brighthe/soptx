"""``soptx.fem.preconditioners`` 预条件子注册表的测试."""

from __future__ import annotations

import pytest

from soptx.fem.preconditioners import available_preconditioners, create_preconditioner, register_preconditioner
from soptx.solvers import DiagonalPreconditioner, DirectSolver


def test_registered_keys() -> None:
    assert available_preconditioners() == ('gmg', 'jacobi', 'mumps', 'scipy')


def test_jacobi_builds_unset_diagonal_preconditioner() -> None:
    M = create_preconditioner('jacobi', None, {})
    assert isinstance(M, DiagonalPreconditioner) and not M.is_setup


def test_direct_keys_build_unset_direct_solver() -> None:
    for key in ('scipy', 'mumps'):
        M = create_preconditioner(key, None, {})
        assert isinstance(M, DirectSolver) and M.backend == key and not M.is_setup


def test_gmg_delegates_to_analyzer_with_options() -> None:
    calls = []

    class _Analyzer:
        def _gmg_preconditioner(self, **options):
            calls.append(options)
            return 'hierarchy'

    assert create_preconditioner('gmg', _Analyzer(), {'gmg_sweeps': 2}) == 'hierarchy'
    assert calls == [{'gmg_sweeps': 2}]


def test_unknown_key_lists_available() -> None:
    # 旧键 'mg' 已更名为 'gmg', 'diagonal' 别名已删除, 都不保留
    with pytest.raises(KeyError, match="可用: gmg, jacobi, mumps, scipy"):
        create_preconditioner('mg', None, {})
    with pytest.raises(KeyError):
        create_preconditioner('diagonal', None, {})


def test_duplicate_key_rejected() -> None:
    with pytest.raises(KeyError, match="已被占用"):
        register_preconditioner('jacobi')(lambda analyzer, options: None)
