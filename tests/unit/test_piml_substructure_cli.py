"""PIML 子结构入口的几何参数解析测试."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def entrypoint():
    """加载入口定义并恢复其添加的导入搜索路径."""
    path = (Path(__file__).resolve().parents[2] / "experiments"
            / "analysis_capability_piml_substructure" / "run.py")
    spec = importlib.util.spec_from_file_location("piml_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


@pytest.mark.parametrize("dim, sizes, expected", [
    (2, [], [1.0, 1.0]),
    (3, [], [1.0, 1.0, 1.0]),
    (2, ["2.0", "1.0"], [2.0, 1.0]),
    (3, ["2.0", "1.0", "0.5"], [2.0, 1.0, 0.5]),
])
@pytest.mark.parametrize("stage", [
    ["--generate-samples"], ["--all"],
])
def test_cell_size_defaults_and_explicit_values(entrypoint, monkeypatch, dim, sizes, expected, stage):
    """各入口使用相同的尺寸默认值和显式参数, 不启动计算."""
    argv = ["run.py", *stage, "--dim", str(dim)]
    if sizes:
        argv += ["--cell-size", *sizes]
    monkeypatch.setattr(sys, "argv", argv)
    assert entrypoint.parse_args().cell_size == expected


@pytest.mark.parametrize("sizes", [
    ["1"], ["1", "1", "1"], ["0", "1"], ["-1", "1"],
    ["nan", "1"], ["inf", "1"], ["text", "1"],
])
def test_invalid_cell_size(entrypoint, monkeypatch, sizes):
    """错误尺寸数量、非有限值和非正数应在参数解析时拒绝."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--generate-samples", "--dim", "2", "--cell-size", *sizes,
    ])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2

@pytest.mark.parametrize("dim, option, expected", [
    (2, [], "plane_stress"), (3, [], None),
    (2, ["--hypothesis", "plane_stress"], "plane_stress"),
    (2, ["--hypothesis", "plane_strain"], "plane_strain"),
])
def test_material_hypothesis(entrypoint, monkeypatch, dim, option, expected):
    """二维默认值及显式选择应正确解析, 三维保持默认本构."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--generate-samples", "--dim", str(dim), *option,
    ])
    assert entrypoint.parse_args().hypothesis == expected


@pytest.mark.parametrize("hypothesis", ["plane_stress", "plane_strain"])
def test_3d_rejects_plane_hypothesis(entrypoint, monkeypatch, hypothesis):
    """三维入口不得接受二维本构假设."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--generate-samples", "--dim", "3", "--hypothesis", hypothesis,
    ])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("name, momentum", [("adam", 0.0), ("adamw", 0.0), ("sgd", 0.9)])
def test_optimizer_cli(entrypoint, monkeypatch, name, momentum):
    """优化器及专用参数可从命令行显式配置."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--train", "--dataset", "unused", "--optimizer", name,
        "--momentum", str(momentum), "--weight-decay", "0.01",
    ])
    args = entrypoint.parse_args()
    assert args.optimizer == name
    assert args.momentum == momentum
    assert args.weight_decay == 0.01


@pytest.mark.parametrize("options", [
    ["--optimizer", "unknown"], ["--momentum", "0.9"],
    ["--optimizer", "adamw", "--momentum", "0.9"],
    ["--optimizer", "sgd", "--momentum", "-1"],
    ["--momentum", "nan"], ["--momentum", "inf"],
    ["--weight-decay", "-1"], ["--weight-decay", "nan"],
    ["--weight-decay", "inf"], ["--lr", "inf"],
])
def test_invalid_optimizer_cli(entrypoint, monkeypatch, options):
    """不合法的优化器参数在解析阶段拒绝, 不启动训练."""
    monkeypatch.setattr(sys, "argv", ["run.py", "--all", *options])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2

@pytest.mark.parametrize("options", [
    ["--dim", "3"], ["--cell-size", "1", "1", "1"], ["--n-fine", "5"],
    ["--trace-kind", "linear_corner"], ["--trace-kind", "full_trace"],
    ["--hypothesis", "plane_strain"],
])
def test_training_rejects_geometry_overrides(entrypoint, monkeypatch, options):
    """训练配置以数据集为准, 拒绝重复指定几何和接口参数."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--train", "--dataset", "unused", *options,
    ])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2

@pytest.mark.parametrize("options", [
    ["--dim", "3"], ["--cell-size", "1", "1", "1"], ["--n-fine", "5"],
    ["--trace-kind", "linear_corner"], ["--trace-kind", "full_trace"],
    ["--hypothesis", "plane_strain"],
])
def test_analysis_rejects_geometry_overrides(entrypoint, monkeypatch, options):
    """整体分析以形函数权重为准, 拒绝重复指定子结构配置."""
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--analyze", "--checkpoint-dir", "unused", *options,
    ])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("dim, expected", [
    (2, (2, 1)), (3, (2, 1, 1)),
])
def test_analysis_substructure_defaults_follow_restored_dimension(entrypoint, dim, expected):
    """整体结构默认排列由权重恢复后的空间维数决定."""
    assert entrypoint.resolve_analysis_n_sub(None, dim) == expected


@pytest.mark.parametrize("dim, values", [
    (2, (2, 1, 1)), (3, (2, 1)), (2, (0, 1)), (3, (2, -1, 1)),
])
def test_analysis_substructure_layout_must_match_restored_dimension(entrypoint, dim, values):
    """显式整体结构排列须与恢复的空间维数一致且全部为正."""
    with pytest.raises(ValueError, match="正整数"):
        entrypoint.resolve_analysis_n_sub(values, dim)


@pytest.mark.parametrize("route", [None, "shape", "stiffness"])
def test_analysis_accepts_single_route(entrypoint, monkeypatch, route):
    """整体分析默认形函数路线, 也允许显式选择任一单路线."""
    argv = ["run.py", "--analyze", "--checkpoint-dir", "unused"]
    if route is not None:
        argv += ["--route", route]
    monkeypatch.setattr(sys, "argv", argv)
    assert entrypoint.parse_args().route == (route or "shape")


@pytest.mark.parametrize("stage", [
    ["--analyze", "--checkpoint-dir", "unused"],
    ["--train", "--dataset", "unused"],
    ["--all"],
])
def test_rejects_both_routes(entrypoint, monkeypatch, capsys, stage):
    """训练与整体分析均拒绝 both, 在读取数据或创建结果目录前报错."""
    monkeypatch.setattr(sys, "argv", ["run.py", *stage, "--route", "both"])
    with pytest.raises(SystemExit) as error:
        entrypoint.parse_args()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("stage", [["--train", "--dataset", "unused"], ["--all"]])
def test_training_defaults_to_shape_route(entrypoint, monkeypatch, stage):
    """训练阶段默认只训练形函数路线."""
    monkeypatch.setattr(sys, "argv", ["run.py", *stage])
    assert entrypoint.parse_args().route == "shape"


def test_entrypoint_import_does_not_load_compute_dependencies(monkeypatch):
    """导入入口不加载计算依赖, 不解析命令行或启动求解."""
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "fealpy", "soptx", "analysis"}:
            raise AssertionError(f"入口导入不应加载 {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = (Path(__file__).resolve().parents[2] / "experiments"
            / "analysis_capability_piml_substructure" / "run.py")
    spec = importlib.util.spec_from_file_location("piml_import_only", path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    assert callable(module.run_analysis)


@pytest.mark.parametrize("route", ["shape", "stiffness"])
def test_analysis_dispatches_to_same_module(entrypoint, monkeypatch, tmp_path, route):
    """使用替身验证整体分析分派与参数传递, 不执行有限元计算."""
    from types import ModuleType, SimpleNamespace

    metadata = {"trace": "linear_corner", "cell_size": [1.0] * 3,
                "n_fine": [5] * 3, "spatial_dimension": 3}
    provider = SimpleNamespace(metadata=lambda: metadata)
    networks, sources = {"shape": object()}, {"shape": "checkpoint"}
    if route == "stiffness":
        networks["stiffness"] = object()
    backend_calls, received = [], {}

    def forbidden(*args, **kwargs):
        raise AssertionError("整体分析不应生成样本或训练网络")

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    module("soptx.fem.substructure.independent_targets", IndependentTargetProvider=forbidden)
    module("soptx.ml.substructure.independent_contract", build_network=forbidden)
    module("soptx.ml.substructure.independent_data", prepare_training_data=forbidden)
    module("soptx.ml.substructure.independent_training", train_network=forbidden)
    module("soptx.ml.substructure.training", TrainingConfig=forbidden)
    module("fealpy.backend", backend_manager=SimpleNamespace(set_backend=backend_calls.append))
    module("soptx.ml.substructure.independent_checkpoints", load_analysis_provider=lambda path: provider,
           load_analysis_networks=lambda path, data, **kwargs: (networks, sources))

    def fake_analysis(*args, **kwargs):
        received.update(args=args, kwargs=kwargs)
        return {"status": "COMPLETED"}

    monkeypatch.setattr(entrypoint, "run_analysis", fake_analysis)
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--analyze", "--checkpoint-dir", str(tmp_path / "weights"),
        "--output-dir", str(tmp_path / "outputs"), "--route", route,
        "--n-sub", "78", "13", "13",
    ])
    entrypoint.main()
    assert backend_calls == ["numpy"]
    assert received["args"][:2] == (provider, networks)
    assert received["kwargs"]["routes"] == (route,)
    assert received["kwargs"]["n_sub"] == (78, 13, 13)
    assert received["kwargs"]["checkpoint_sources"] is sources
    assert not (tmp_path / "outputs").exists()
