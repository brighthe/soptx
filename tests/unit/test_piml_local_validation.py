"""局部预测验证的精确参考、采样和权重加载回归测试."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from fealpy.backend import backend_manager as bm
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.validation import LocalPredictionEvaluator, load_local_model
from soptx.ml.substructure.independent_contract import (
    ACTIVATIONS, HIDDEN_DIMS, SCHEMA, build_network,
)

# 允许仅将 src 加入搜索路径的 pytest 配置导入实验模块.
_root = str(Path(__file__).resolve().parents[2])
_original_path = sys.path[:]
try:
    sys.path.insert(0, _root)
    from experiments.analysis_capability_piml_substructure.local_validation import (
        evaluate_local_predictions,
    )
finally:
    sys.path[:] = _original_path


class ExactPredictor(torch.nn.Module):
    """以精确标签充当预测值, 验证误差度量和自由度排列."""

    def __init__(self, provider, route, factor=1.0):
        super().__init__()
        self.provider = provider
        self.route = route
        self.factor = factor

    def forward(self, inputs):
        values = self.provider(inputs.detach().cpu().numpy())[self.route]
        return torch.as_tensor(values, device=inputs.device, dtype=inputs.dtype) * self.factor


def make_provider(dim=2, trace="linear_corner"):
    """创建小规模参考原型, 不生成历史实验数据."""
    bm.set_backend("numpy")
    return IndependentTargetProvider(cell_size=(1.0,) * dim, n_fine=(2,) * dim,
                                     trace_kind=trace)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("trace", ["linear_corner", "full_trace"])
@pytest.mark.parametrize("route", ["shape", "stiffness"])
def test_exact_predictions_have_roundoff_errors(tmp_path, dim, trace, route):
    """精确预测的误差应为舍入量级, 且刚度变形子空间为正定."""
    provider = make_provider(dim, trace)
    result = evaluate_local_predictions(
        ExactPredictor(provider, route), provider, route, n_test=2,
        batch_size=1, seed=2026, output_dir=tmp_path / "result",
    )
    assert result["status"] == "COMPLETED"
    assert result["accuracy_acceptance"] == "not_evaluated"
    assert result["n_completed"] == 2
    assert result["metrics"]["stiffness_relative_error"]["max"] < 1e-10
    assert result["metrics"]["rigid_nullspace_residual"]["max"] < 1e-10
    if route == "shape":
        assert result["metrics"]["shape_relative_error"]["max"] < 1e-10
        assert result["metrics"]["rigid_reproduction_residual"]["max"] < 1e-10
    assert result["deformation_spectrum_counts"]["positive"] == 2
    rows = (tmp_path / "result" / "per_sample.jsonl").read_text().splitlines()
    assert [json.loads(row)["sample_index"] for row in rows] == [0, 1]


def test_sampling_is_batch_invariant_and_separate(tmp_path):
    """更换批量不改变材料场, 测试流不同于训练和验证子流."""
    provider = make_provider()
    for batch in (1, 3):
        evaluate_local_predictions(ExactPredictor(provider, "shape"), provider, "shape",
                                   n_test=3, batch_size=batch, seed=2026,
                                   output_dir=tmp_path / str(batch))
    inputs = np.load(tmp_path / "1" / "inputs.npy")
    np.testing.assert_array_equal(inputs, np.load(tmp_path / "3" / "inputs.npy"))
    for stream in np.random.SeedSequence(2026).spawn(2):
        other = np.random.default_rng(stream).uniform(1e-6, 1.0, size=inputs.shape)
        assert not np.array_equal(inputs, other)
    with pytest.raises(FileExistsError):
        evaluate_local_predictions(ExactPredictor(provider, "shape"), provider, "shape",
                                   n_test=1, output_dir=tmp_path / "1")


def test_negative_stiffness_is_reported_without_fallback(tmp_path):
    """负刚度只登记诊断, 不替换为精确刚度或宣称精度通过."""
    provider = make_provider()
    result = evaluate_local_predictions(ExactPredictor(provider, "stiffness", -1),
                                       provider, "stiffness", n_test=2,
                                       output_dir=tmp_path / "negative")
    assert result["deformation_spectrum_counts"]["negative"] == 2
    assert result["metrics"]["stiffness_relative_error"]["mean"] == pytest.approx(2.0)
    assert result["accuracy_acceptance"] == "not_evaluated"


def test_nonfinite_prediction_records_failure(tmp_path):
    """坏预测保留失败摘要, 不输出伪造的有限误差."""
    provider = make_provider()
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="非有限"):
        evaluate_local_predictions(ExactPredictor(provider, "shape", float("nan")),
                                   provider, "shape", n_test=1, output_dir=output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["n_completed"] == 0


def test_load_stiffness_without_shape_checkpoint(tmp_path):
    """刚度局部加载只需自身权重, 并校验路线与配置."""
    provider = make_provider()
    meta = provider.metadata()
    model = build_network(meta, route="stiffness", num_networks=1)
    width = meta["n_stiffness_targets"]
    path = tmp_path / "stiffness_best.pt"
    torch.save({
        "schema": SCHEMA, "route": "stiffness", "model_state": model.state_dict(),
        "dataset": {"complete": True, "provider": meta, "seed": 2026,
                    "input_quantity": "normalized_young_modulus"},
        "architecture": {"input_dim": meta["n_cells"], "output_dim": width,
                         "hidden_dims": HIDDEN_DIMS,
                         "activations": [cls.__name__ for cls in ACTIVATIONS],
                         "num_networks": 1, "output_groups": (tuple(range(width)),),
                         "model_class": type(model).__name__, "dtype": "float64"},
    }, path)
    loaded = load_local_model(path, provider, "stiffness")
    assert not loaded.training
    assert len(loaded.local_validation_source["sha256"]) == 64
    x = torch.ones(1, meta["n_cells"], dtype=torch.float64)
    torch.testing.assert_close(loaded(x), model(x))
    with pytest.raises(ValueError, match="不匹配"):
        load_local_model(path, provider, "shape")
    with pytest.raises(ValueError, match="不匹配"):
        load_local_model(path, make_provider(trace="full_trace"), "stiffness")


def test_core_validation_without_experiment_output(tmp_path):
    """核心接口只消费材料数组, 不创建实验文件."""
    provider = make_provider()
    inputs = np.full((2, provider.metadata()["n_cells"]), 0.5)
    evaluator = LocalPredictionEvaluator(ExactPredictor(provider, "shape"), provider, "shape")
    records = evaluator.evaluate(inputs)
    assert len(records) == 2
    assert max(row["stiffness_relative_error"] for row in records) < 1e-10
    assert list(tmp_path.iterdir()) == []
