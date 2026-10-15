"""求解器注册表 :mod:`soptx.solvers.registry` 的测试."""

from __future__ import annotations

import pytest

from soptx.backend import backend_manager as bm
from soptx.solvers import CGSolver, DirectSolver, LinearSolver, available, create, register
from soptx.solvers import registry

bm.set_backend("numpy")


def test_builtin_keys_are_registered() -> None:
    assert {"cg", "scipy", "mumps"} <= set(available())
    assert list(available()) == sorted(available())


def test_create_applies_preset_and_forwards_kwargs() -> None:
    assert create("scipy").backend == "scipy"
    # 构造 MUMPS 求解器不需要 PyMUMPS, 只有 setup 才需要
    mumps = create("mumps", sym=1)
    assert isinstance(mumps, DirectSolver) and mumps.backend == "mumps"
    assert isinstance(create("cg", rtol=1e-6), CGSolver)


def test_create_rejects_overriding_preset() -> None:
    with pytest.raises(ValueError, match="预置参数"):
        create("scipy", backend="mumps")


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(KeyError, match="未注册的求解器键"):
        create("no-such-solver")


def test_duplicate_registration_is_rejected(monkeypatch) -> None:
    # 换一张空表, 不污染全局注册表
    monkeypatch.setattr(registry, "_REGISTRY", {})

    class _Dummy(LinearSolver):
        def _solve(self, b, x0=None):
            return b, {"niter": 0, "relres": None, "converged": True}

    register("dummy")(_Dummy)
    assert available() == ("dummy",)
    with pytest.raises(ValueError, match="已被 _Dummy 占用"):
        register("dummy")(_Dummy)
