"""求解器派发与构造契约, 不需要 MPI runtime。"""

from __future__ import annotations

import pytest

# matrix_free_analyzer 经由 fealpy.distributed 引入 mpi4py, 属可选 extra
pytest.importorskip("mpi4py")

from soptx.fem.solvers import elasticity_operator
from soptx.fem.solvers.matrix_free_analyzer import (
    DISTRIBUTED_SOLVERS,
    DistributedElasticityAnalyzer,
)


class RecordingRoutine:
    """记录收到的关键字, 代替真实求解器。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, K, F, **kwargs):
        self.calls.append(kwargs)
        return F, {"name": "recording"}


class Bare:
    """跳过 __init__ 的分析器实例, 只测 solve_system 的派发。"""

    def __new__(cls):
        instance = object.__new__(DistributedElasticityAnalyzer)
        instance._dof_comm = object()
        return instance


def test_cg_is_the_registered_solver() -> None:
    assert "cg" in DISTRIBUTED_SOLVERS


def test_an_unweighted_solver_is_refused_with_an_explanation() -> None:
    """转发到 fealpy 的 gmres 会重复计数共享自由度, 必须拒绝而不是照跑。"""

    instance = Bare()

    with pytest.raises(NotImplementedError, match="gmres"):
        instance.solve_system(object(), [0.0], [0.0], solver="gmres")


def test_registered_solvers_receive_the_forwarded_options() -> None:
    """自有选项要能透传, 否则新增求解器无法配置。"""

    routine = RecordingRoutine()
    instance = Bare()

    try:
        DISTRIBUTED_SOLVERS["recording"] = routine
        out = [0.0, 0.0]
        instance.solve_system(
            object(),
            [1.0, 2.0],
            out,
            solver="recording",
            maxiter=7,
            restart=30,
        )
    finally:
        DISTRIBUTED_SOLVERS.pop("recording", None)

    assert len(routine.calls) == 1
    call = routine.calls[0]
    assert call["maxiter"] == 7
    assert call["restart"] == 30
    assert call["dof_comm"] is instance._dof_comm
    assert out == [1.0, 2.0]


def test_a_distributed_analyzer_refuses_a_missing_communicator() -> None:
    with pytest.raises(ValueError, match="dof_comm"):
        DistributedElasticityAnalyzer(dof_comm=None)


def test_omitting_the_communicator_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        DistributedElasticityAnalyzer()


def test_the_serial_builder_is_the_documented_alternative() -> None:
    assert callable(elasticity_operator.build_serial_analyzer)
    assert callable(elasticity_operator.build_distributed_analyzer)
