"""与 FEM 解耦的代理模型训练循环测试。"""

import numpy as np
import torch

from soptx.ml import MLP
from soptx.ml.substructure import TrainingConfig, train_surrogate


def test_train_surrogate_restores_best_validation_state_and_preserves_rng() -> None:
    x = np.linspace(-1.0, 1.0, 40, dtype=np.float32).reshape(20, 2)
    y = (2.0 * x[:, :1] - 0.5 * x[:, 1:2]).astype(np.float32)
    model = MLP(2, 1, hidden_dims=(8,))
    config = TrainingConfig(
        epochs=20,
        batch_size=5,
        learning_rate=0.02,
        seed=9,
        patience=5,
    )
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()

    result = train_surrogate(model, x[:16], y[:16], x[16:], y[16:], config)

    assert 1 <= result.best_epoch <= result.epochs_run <= config.epochs
    assert np.isfinite(result.best_validation_loss)
    assert np.isfinite(result.final_training_loss)
    assert len(result.training_losses) == result.epochs_run
    assert torch.equal(torch.random.get_rng_state(), state_before)
    with torch.no_grad():
        restored_loss = torch.nn.functional.mse_loss(
            model(torch.as_tensor(x[16:])), torch.as_tensor(y[16:])
        )
    assert np.isclose(float(restored_loss), result.best_validation_loss)


def test_train_surrogate_selects_physics_snapshot_at_fixed_intervals() -> None:
    x = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(10, 2)
    y = (x[:, :1] + x[:, 1:2]).astype(np.float32)
    model = MLP(2, 1, hidden_dims=(4,))
    expected_scores = iter((3.0, 1.0, 2.0))
    snapshots: list[dict[str, torch.Tensor]] = []

    def evaluator(current: torch.nn.Module) -> dict[str, float]:
        snapshots.append(
            {name: value.detach().clone() for name, value in current.state_dict().items()}
        )
        return {"selection_score": next(expected_scores)}

    result = train_surrogate(
        model,
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        TrainingConfig(
            epochs=4,
            batch_size=4,
            learning_rate=0.01,
            seed=5,
            physics_eval_interval=2,
        ),
        evaluator=evaluator,
    )

    assert [item["epoch"] for item in result.physics_history] == [1, 2, 4]
    assert result.best_epoch == 2
    assert result.best_selection_score == 1.0
    assert result.evaluation["selection_score"] == 1.0
    restored = model.state_dict()
    for name, value in snapshots[1].items():
        assert torch.equal(restored[name], value)


def test_fixed_epoch_refit_restores_final_epoch_without_early_stopping() -> None:
    x = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(10, 2)
    y = (x[:, :1] - x[:, 1:2]).astype(np.float32)
    model = MLP(2, 1, hidden_dims=(4,))
    result = train_surrogate(
        model,
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        TrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=0.01,
            seed=7,
            select_final_state=True,
        ),
    )

    assert result.epochs_run == 3
    assert result.best_epoch == 3
    assert result.best_selection_score is None
