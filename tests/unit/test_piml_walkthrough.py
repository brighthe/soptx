"""PIML 离线训练与在线分析走查入口的隔离测试."""

import builtins
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "analysis_capability_piml_substructure"
)
TRAINING_SCRIPT = ROOT / "walkthrough_training.py"
ANALYSIS_SCRIPT = ROOT / "walkthrough_analysis.py"


def load_script(path):
    """在非主模块命名空间加载脚本."""
    namespace = {"__name__": f"{path.stem}_under_test", "__file__": str(path)}
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def install(monkeypatch, name, **attrs):
    """安装轻量模块替身."""
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


def block_compute_imports(monkeypatch):
    """禁止导入计算模块, 以捕获导入时的意外计算."""
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "fealpy", "soptx", "experiments"}:
            raise AssertionError(f"不应加载计算依赖: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("script", [TRAINING_SCRIPT, ANALYSIS_SCRIPT])
def test_import_does_not_compute(monkeypatch, script):
    """两个入口导入时均不加载计算依赖或启动流程."""
    block_compute_imports(monkeypatch)
    namespace = load_script(script)
    assert callable(namespace["parse_args"])
    assert callable(namespace["main"])


@pytest.mark.parametrize("route,missing", [
    ("shape", "shape_best.pt"),
    ("stiffness", "stiffness_best.pt"),
])
def test_analysis_missing_checkpoint_fails_before_compute(
        monkeypatch, tmp_path, route, missing):
    """缺少必要权重时在加载计算依赖及创建整体网格前报错."""
    namespace = load_script(ANALYSIS_SCRIPT)
    block_compute_imports(monkeypatch)
    monkeypatch.setattr(Path, "is_file", lambda self: self.name != missing)
    options = ["--shape-dir", str(tmp_path), "--route", route]
    if route == "stiffness":
        options += ["--stiffness-dir", str(tmp_path)]
    with pytest.raises(FileNotFoundError, match=missing):
        namespace["main"](options)


@pytest.mark.parametrize("options", [
    [],
    ["--samples-only", "--samples-dir", "samples"],
    ["--from-scratch", "--samples-dir", "samples"],
    ["--samples-only", "--from-scratch"],
])
def test_training_requires_exactly_one_mode(options):
    """三种模式必须且只能选择一种."""
    parse_args = load_script(TRAINING_SCRIPT)["parse_args"]
    with pytest.raises(SystemExit):
        parse_args(options)


def test_training_samples_dir_leaves_generation_config_unset():
    """复用样本时不填充样本生成配置, 相对路径以脚本目录为基准."""
    args = load_script(TRAINING_SCRIPT)["parse_args"](["--samples-dir", "samples"])
    assert args.samples_dir == ROOT / "samples"
    assert not args.samples_only and not args.from_scratch
    assert args.dim is None
    assert args.n_train is None


@pytest.mark.parametrize("mode", ["--samples-only", "--from-scratch"])
def test_training_generation_defaults(mode):
    """两种生成模式保留正式规模、三维局部问题和训练默认值."""
    args = load_script(TRAINING_SCRIPT)["parse_args"]([mode])
    assert args.samples_dir is None
    assert args.dim == 3
    assert args.n_fine == 5
    assert args.cell_size == [1.0, 1.0, 1.0]
    assert args.trace_kind == "linear_corner"
    assert args.n_train == 400_000
    assert args.n_validation == 40_000
    assert args.sample_seed == 2026
    assert args.generation_batch_size == 32
    assert args.epochs == 500
    assert args.batch_size == 256
    assert args.train_seed == 2026
    assert args.route == "shape"
    assert args.num_networks == 4


@pytest.mark.parametrize("option,value", [
    ("--dim", "2"),
    ("--n-fine", "5"),
    ("--cell-size", "1"),
    ("--nu", "0.3"),
    ("--hypothesis", "plane_stress"),
    ("--trace-kind", "linear_corner"),
    ("--n-train", "8"),
    ("--n-validation", "4"),
    ("--min-modulus", "1e-6"),
    ("--sample-seed", "2026"),
    ("--generation-batch-size", "2"),
])
def test_samples_dir_rejects_generation_options(option, value):
    """复用样本时拒绝重复指定 manifest 已记录的配置."""
    parse_args = load_script(TRAINING_SCRIPT)["parse_args"]
    with pytest.raises(SystemExit):
        parse_args(["--samples-dir", "samples", option, value])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_training_rejects_non_positive_num_networks(value):
    """网络数量须为正整数."""
    parse_args = load_script(TRAINING_SCRIPT)["parse_args"]
    with pytest.raises(SystemExit):
        parse_args(["--samples-dir", "samples", "--num-networks", value])


@pytest.mark.parametrize("lr", ["1e-7", "nan", "inf"])
def test_training_rejects_lr_below_scheduler_floor(lr):
    """初始学习率须为有限数且不低于调度下限 1e-6."""
    parse_args = load_script(TRAINING_SCRIPT)["parse_args"]
    with pytest.raises(SystemExit):
        parse_args(["--samples-dir", "samples", "--lr", lr])
    assert parse_args(["--samples-dir", "samples", "--lr", "1e-6"]).lr == 1e-6


def provider_metadata(dim=3):
    """构造隔离测试使用的 provider 元数据."""
    hypothesis = "plane_stress" if dim == 2 else None
    metadata = {
        "schema": "soptx.piml-substructure.independent.v1",
        "spatial_dimension": dim,
        "cell_size": [1.0] * dim,
        "n_fine": [5] * dim,
        "poisson_ratio": 0.3,
        "trace": "linear_corner",
        "material_hypothesis": hypothesis,
        "material_scaling": {
            "reference_young_modulus": 1.0, "penal": 1.0, "rho_min": 0.0,
        },
        "n_cells": 1,
        "n_i": 1,
        "n_trace": 1,
        "n_rigid": dim,
        "shape_independent_indices": [[0, 0]],
        "stiffness_independent_indices": [[0, 0]],
    }
    return metadata


def make_provider(metadata, *, assemble=None):
    """构造训练与在线入口共用的 provider 替身."""
    prototype = SimpleNamespace(
        n_total_dofs=1,
        n_cells=1,
        assemble_local_stiffness_batch=assemble or (lambda *args, **kwargs: None),
    )
    return SimpleNamespace(
        metadata=lambda: metadata,
        prototype=prototype,
        chunk_size=None,
        codecs={"shape": object(), "stiffness": object()},
        trace_kind=metadata["trace"],
    )


def install_training_modules(
        monkeypatch, provider_factory, calls, *, existing=None, skipped=()):
    """安装离线入口依赖并记录参数传递.

    Parameters
    ----------
    existing : Path or None
        find_training_data 替身返回的同配置样本目录.
    skipped : sequence of Path
        find_training_data 替身报告的跳过目录.
    """
    backend = SimpleNamespace(set_backend=lambda name: calls.append(("backend", name)))
    install(monkeypatch, "soptx.backend", backend_manager=backend)
    install(
        monkeypatch, "soptx.fem.substructure.independent_targets",
        IndependentTargetProvider=provider_factory,
    )

    def build_network(metadata, *, route, seed, num_networks):
        calls.append(("build", route, seed, num_networks))
        return object()

    def find_training_data(samples_root, provider, **kwargs):
        calls.append(("find", samples_root, kwargs))
        return existing, list(skipped)

    def prepare_training_data(provider, output, **kwargs):
        calls.append(("generate", output, kwargs))
        return output

    def train_network(dataset, output, **kwargs):
        calls.append(("train", dataset, output, kwargs))
        return {
            "best_epoch": 3, "best_validation_loss": 1.25e-4,
            "checkpoint": str(output / f"{kwargs['route']}_best.pt"),
            "epochs_run": 5,
        }

    install(
        monkeypatch, "soptx.ml.substructure.independent_contract",
        SCHEMA="soptx.piml-substructure.independent.v1",
        build_network=build_network,
        provider_metadata_matches=lambda saved, current: saved == current,
    )
    install(
        monkeypatch, "soptx.ml.substructure.independent_data",
        find_training_data=find_training_data,
        prepare_training_data=prepare_training_data,
    )
    install(
        monkeypatch, "soptx.ml.substructure.independent_training",
        train_network=train_network,
    )

    class TrainingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls.append(("config", kwargs))

    install(
        monkeypatch, "soptx.ml.substructure.training",
        TrainingConfig=TrainingConfig,
    )


def test_training_generation_passes_sample_and_training_config(monkeypatch, tmp_path):
    """--from-scratch 查重后生成样本, 只训练所选路线并写入分路线目录."""
    namespace = load_script(TRAINING_SCRIPT)
    calls = []
    metadata = provider_metadata()
    provider = make_provider(metadata)

    def provider_factory(**kwargs):
        calls.append(("provider", kwargs))
        return provider

    install_training_modules(monkeypatch, provider_factory, calls)
    namespace["main"]([
        "--from-scratch",
        "--outputs-root", str(tmp_path),
        "--route", "stiffness",
        "--num-networks", "1",
        "--n-train", "8",
        "--n-validation", "4",
        "--generation-batch-size", "2",
        "--sample-seed", "7",
        "--epochs", "3",
        "--batch-size", "2",
        "--train-seed", "11",
        "--optimizer", "adamw",
        "--lr", "0.002",
        "--weight-decay", "0.01",
        "--patience", "5",
    ])

    samples_root = tmp_path / "independent_15_layer" / "samples"
    found = next(call for call in calls if call[0] == "find")
    assert found[1] == samples_root
    assert found[2] == {
        "n_train": 8, "n_validation": 4, "min_modulus": 1e-6, "seed": 7,
    }
    generated = next(call for call in calls if call[0] == "generate")
    assert generated[1].parent == samples_root
    assert generated[2] == {
        "n_train": 8,
        "n_validation": 4,
        "batch_size": 2,
        "min_modulus": 1e-6,
        "seed": 7,
    }
    config = next(call[1] for call in calls if call[0] == "config")
    assert config == {
        "epochs": 3,
        "batch_size": 2,
        "optimizer": "adamw",
        "optimizer_params": {"lr": 0.002, "weight_decay": 0.01},
        "seed": 11,
        "patience": 5,
    }
    assert [call[1:] for call in calls if call[0] == "build"] == [("stiffness", 11, 1)]
    trained = next(call for call in calls if call[0] == "train")
    assert trained[1] == generated[1]
    assert trained[2].parent == tmp_path / "independent_15_layer" / "training" / "stiffness"
    assert trained[2].name == generated[1].name
    assert trained[3]["route"] == "stiffness"
    assert trained[3]["provider"] is provider


def test_training_samples_only_skips_training(monkeypatch, tmp_path):
    """--samples-only 生成样本后返回, 不构建网络也不训练."""
    namespace = load_script(TRAINING_SCRIPT)
    calls = []
    metadata = provider_metadata()
    skipped = [tmp_path / "broken"]
    install_training_modules(
        monkeypatch, lambda **kwargs: make_provider(metadata), calls,
        skipped=skipped,
    )
    namespace["main"](["--samples-only", "--outputs-root", str(tmp_path)])

    assert [call[0] for call in calls] == ["backend", "find", "generate"]
    assert not (tmp_path / "independent_15_layer" / "training").exists()


@pytest.mark.parametrize("mode", ["--samples-only", "--from-scratch"])
def test_training_generation_refuses_existing_samples(monkeypatch, tmp_path, mode):
    """已有同配置样本时报错并提示 --samples-dir, 不生成也不训练."""
    namespace = load_script(TRAINING_SCRIPT)
    calls = []
    metadata = provider_metadata()
    existing = tmp_path / "independent_15_layer" / "samples" / "old"
    install_training_modules(
        monkeypatch, lambda **kwargs: make_provider(metadata), calls,
        existing=existing,
    )
    with pytest.raises(FileExistsError, match="--samples-dir"):
        namespace["main"]([mode, "--outputs-root", str(tmp_path)])
    assert any(call[0] == "find" for call in calls)
    assert not any(call[0] in {"generate", "build", "train"} for call in calls)


def test_training_reuses_manifest_and_skips_generation(monkeypatch, tmp_path):
    """复用样本时从 manifest 恢复 provider 并校验独立分量编号."""
    namespace = load_script(TRAINING_SCRIPT)
    calls = []
    metadata = provider_metadata(dim=2)
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "manifest.json").write_text(
        json.dumps({
            "schema": "soptx.piml-substructure.independent.v1",
            "complete": True,
            "input_quantity": "normalized_young_modulus",
            "provider": metadata,
        }),
        encoding="utf-8",
    )

    def provider_factory(**kwargs):
        calls.append(("provider", kwargs))
        return make_provider(metadata)

    install_training_modules(monkeypatch, provider_factory, calls)
    namespace["main"]([
        "--samples-dir", str(samples),
        "--outputs-root", str(tmp_path),
        "--epochs", "1",
    ])

    assert not any(call[0] in {"find", "generate"} for call in calls)
    restored = next(call[1] for call in calls if call[0] == "provider")
    assert restored == {
        "cell_size": (1.0, 1.0),
        "n_fine": (5, 5),
        "nu": 0.3,
        "trace_kind": "linear_corner",
        "hypothesis": "plane_stress",
    }
    trained = next(call for call in calls if call[0] == "train")
    assert trained[1] == samples
    assert trained[2].parent == tmp_path / "independent_15_layer" / "training" / "shape"
    assert trained[3]["route"] == "shape"


@pytest.mark.parametrize("failure", ["incomplete", "metadata_mismatch"])
def test_training_rejects_invalid_reused_manifest(
        monkeypatch, tmp_path, failure):
    """复用样本时拒绝未完成记录或不一致的独立分量元数据."""
    namespace = load_script(TRAINING_SCRIPT)
    calls = []
    saved = provider_metadata(dim=2)
    samples = tmp_path / "samples"
    samples.mkdir()
    manifest = {
        "schema": "soptx.piml-substructure.independent.v1",
        "complete": failure != "incomplete",
        "input_quantity": "normalized_young_modulus",
        "provider": saved,
    }
    (samples / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    current = dict(saved)
    if failure == "metadata_mismatch":
        current["shape_independent_indices"] = [[1, 0]]

    def provider_factory(**kwargs):
        calls.append(("provider", kwargs))
        return make_provider(current)

    install_training_modules(monkeypatch, provider_factory, calls)
    with pytest.raises(ValueError, match="未完成|独立分量编号"):
        namespace["main"]([
            "--samples-dir", str(samples),
            "--outputs-root", str(tmp_path),
            "--epochs", "1",
        ])
    assert not any(call[0] in {"generate", "train"} for call in calls)


FORMAL_TRAINING_DIR = Path(
    "/home/brighthe/workspace/data/soptx/piml_substructure/"
    "independent_15_layer/training/20260922T065924289373Z"
)


def test_analysis_default_weight_dirs():
    """两个权重目录默认指向正式训练结果, 刚度目录仅在 stiffness 路线填充."""
    parse_args = load_script(ANALYSIS_SCRIPT)["parse_args"]
    args = parse_args([])
    assert args.shape_dir == FORMAL_TRAINING_DIR
    assert args.stiffness_dir is None
    args = parse_args(["--route", "stiffness"])
    assert args.shape_dir == FORMAL_TRAINING_DIR
    assert args.stiffness_dir == FORMAL_TRAINING_DIR


@pytest.mark.parametrize("options", [
    ["--shape-dir", "weights", "--route", "both"],
    ["--shape-dir", "weights", "--n-sub", "0", "1"],
    ["--shape-dir", "weights", "--seed", "-1"],
    ["--shape-dir", "weights", "--mem-limit-gb", "0"],
    ["--stiffness-dir", "weights"],
    ["--training-dir", "weights"],
    ["--E-simp-penalty", "0"],
    ["--E-simp-penalty", "-1"],
    ["--E-simp-penalty", "nan"],
    ["--E-simp-penalty", "inf"],
    ["--local-batch-size", "0"],
    ["--local-batch-size", "-1"],
])
def test_analysis_invalid_cli_rejected(options):
    """在线入口在解析阶段拒绝非法配置."""
    parse_args = load_script(ANALYSIS_SCRIPT)["parse_args"]
    with pytest.raises(SystemExit):
        parse_args(options)


def test_analysis_defaults():
    """在线入口保留既有整体规模、路线、求解器和随机种子默认值."""
    args = load_script(ANALYSIS_SCRIPT)["parse_args"]([
        "--shape-dir", "weights",
    ])
    assert args.n_sub == [78, 13, 13]
    assert args.route == "shape"
    assert args.solver == "mumps"
    assert args.seed == 0
    assert args.mem_limit_gb == 35.0
    assert args.E_simp_penalty == 3.0
    assert args.local_batch_size == 32
    assert args.shape_dir == ROOT / "weights"
    assert args.stiffness_dir is None


def install_analysis_modules(
        monkeypatch, metadata, provider, calls, *, stiffness_metadata=None,
        n_substructures=1):
    """安装在线入口依赖并记录预测、装配及投影的批次边界.

    Parameters
    ----------
    stiffness_metadata : dict or None
        刚度权重记录的元数据, None 表示与形函数权重一致.
    n_substructures : int
        隔离测试中布局提供的子结构数量.
    """
    import numpy as np

    backend = SimpleNamespace(
        set_backend=lambda name: calls.append(("backend", name)),
        reshape=np.reshape,
        asarray=np.asarray,
        to_numpy=np.asarray,
        float64=np.float64,
    )
    install(monkeypatch, "soptx.backend", backend_manager=backend)

    def load_network(path, current, *, route):
        calls.append(("load_network", route, path))
        assert path.name == f"{route}_best.pt"
        assert current == metadata
        return object(), {}

    def load_metadata(path, route):
        if route == "stiffness" and stiffness_metadata is not None:
            return stiffness_metadata
        return metadata

    install(
        monkeypatch, "soptx.ml.substructure.independent_checkpoints",
        load_independent_provider_metadata=load_metadata,
        load_independent_network=load_network,
        decoder_metadata_matches=lambda saved, current: saved == current,
    )
    install(
        monkeypatch, "soptx.ml.substructure.independent_contract",
        provider_metadata_matches=lambda saved, current: saved == current,
    )
    install(
        monkeypatch, "soptx.ml.substructure.inference",
        predict_independent_outputs=lambda *args: pytest.fail("不应提前执行网络预测"),
    )
    dim = metadata["spatial_dimension"]
    n_cells = metadata["n_cells"]
    local_shape = (n_substructures, n_cells) + (1,) * (dim - 1)
    layout = SimpleNamespace(
        total_fine=(n_substructures * n_cells,) + (1,) * (dim - 1),
        split_global_cell_field=lambda value: value.reshape(local_shape),
    )
    assembler = SimpleNamespace(layout=layout)

    def make_assembler(current):
        assert current is layout
        calls.append(("assembler", current))
        return assembler

    def build(current, **kwargs):
        assert current is layout
        calls.append(("build_modulus_substructures", current))
        return (
            provider.prototype, [object()] * n_substructures,
            [object()] * n_substructures,
        )

    def to_cell_density(values):
        grid = np.asarray(values).reshape(n_substructures, n_cells)
        ordered = grid[:, ::-1].copy()
        calls.append(("cell_order", grid.copy(), ordered.copy()))
        return ordered

    provider.prototype.n_cells = n_cells
    provider.prototype.to_cell_density = to_cell_density
    provider.prototype.i_dofs = np.array([], dtype=np.int64)
    provider.prototype.b_dofs = np.array([0], dtype=np.int64)
    provider.prototype.trace_interface_bases = lambda trace: (
        np.zeros((1, dim)), np.ones((1, 1)), np.zeros((1, dim)),
    )
    provider.trace = SimpleNamespace(matrix=np.ones((1, 1)), n_trace_dofs=1)

    def codec(route):
        def decode(values):
            calls.append(("decode", route, len(values)))
            return np.asarray(values).reshape(len(values), 1, 1)
        return SimpleNamespace(n_output=1, decode=decode)

    provider.shape_codec = codec("shape")
    provider.stiffness_codec = codec("stiffness")
    provider.codecs = {
        "shape": provider.shape_codec, "stiffness": provider.stiffness_codec,
    }

    class ShapeBuilder:
        """记录每批内部延拓与细网格刚度的投影调用."""

        def __init__(self, *args, **kwargs):
            pass

        def assemble_reduced_stiffness(self, local_stiffness, recovery):
            assert local_stiffness.shape[0] == recovery.shape[0]
            calls.append(("project", len(local_stiffness)))
            return np.ones((len(local_stiffness), 1, 1))

    install(
        monkeypatch, "soptx.fem.substructure",
        StructuredSubstructureLayout=lambda *args, **kwargs: layout,
        GlobalAssembler=make_assembler,
        IndependentPredictionDecoder=lambda prototype, **kwargs: provider,
        ShapeFunctionCondensation=ShapeBuilder,
        build_modulus_substructures=build,
    )
    install(
        monkeypatch, "soptx.materials",
        IsotropicLinearElasticMaterial=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    install(
        monkeypatch, "soptx.problems.elasticity",
        CantileverCorner2d=lambda **kwargs: SimpleNamespace(**kwargs),
        FullMBBBeam3d=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def interpolation(**kwargs):
        options = kwargs["options"]
        calls.append(("interpolation", dict(options)))
        return SimpleNamespace(
            interpolate_material=lambda material, rho_val: (
                material.youngs_modulus * rho_val ** options["penalty_factor"]
            ),
        )

    install(
        monkeypatch, "soptx.topology.interpolation",
        MaterialInterpolationScheme=interpolation,
    )
    install(monkeypatch, "resource", RLIMIT_AS=9, setrlimit=lambda *args: calls.append("limit"))


def test_analysis_two_dimensional_weights_require_two_n_sub(
        monkeypatch, tmp_path):
    """二维权重不能沿用三维默认 n_sub, 且在创建整体网格前失败."""
    namespace = load_script(ANALYSIS_SCRIPT)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    calls = []
    metadata = provider_metadata(dim=2)
    provider = make_provider(metadata)
    install_analysis_modules(monkeypatch, metadata, provider, calls)

    with pytest.raises(ValueError, match="二维权重需要显式"):
        namespace["main"](["--shape-dir", str(tmp_path)])
    assert not any(call == "limit" for call in calls)
    assert not any(
        isinstance(call, tuple) and call[0] == "load_network" for call in calls
    )


@pytest.mark.parametrize("route", ["shape", "stiffness"])
@pytest.mark.parametrize("dim,n_sub", [
    (2, ["3", "1"]),
    (3, ["3", "1", "1"]),
])
def test_analysis_loads_weights_and_stops_at_condensed_stiffness(
        monkeypatch, tmp_path, route, dim, n_sub):
    """在线入口按批构造预测缩聚刚度, 刚度路线不装配细网格刚度."""
    import numpy as np

    namespace = load_script(ANALYSIS_SCRIPT)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    calls = []
    metadata = dict(provider_metadata(dim=dim), n_cells=2)

    def assemble(values, **kwargs):
        calls.append(("local_stiffness", len(values)))
        assert route == "shape", "直接刚度路线不应装配细网格刚度"
        assert 0 < len(values) <= 2
        return np.ones((len(values), 1, 1))

    provider = make_provider(metadata, assemble=assemble)
    install_analysis_modules(
        monkeypatch, metadata, provider, calls, n_substructures=3,
    )
    trap = lambda *args, **kwargs: pytest.fail("在线入口不应生成样本或训练")
    install(
        monkeypatch, "soptx.ml.substructure.independent_contract",
        build_network=trap,
        provider_metadata_matches=lambda saved, current: saved == current,
    )
    install(monkeypatch, "soptx.ml.substructure.independent_data",
            prepare_training_data=trap)
    install(monkeypatch, "soptx.ml.substructure.independent_training",
            train_network=trap)

    def predict(model, inputs, current_route):
        calls.append(("predict", current_route, np.asarray(inputs).copy()))
        assert current_route == route
        assert 0 < len(inputs) <= 2
        return np.ones((len(inputs), 1))

    install(
        monkeypatch, "soptx.ml.substructure.inference",
        predict_independent_outputs=predict,
    )
    dirs = {"shape": tmp_path / "shape", "stiffness": tmp_path / "stiffness"}
    options = [
        "--shape-dir", str(dirs["shape"]),
        "--domain", *[value for count in n_sub for value in ("0", count)],
        "--n-sub", *n_sub,
        "--route", route,
        "--local-batch-size", "2",
        "--E-simp-penalty", "2.0",
    ]
    if route == "stiffness":
        options += ["--stiffness-dir", str(dirs["stiffness"])]
    namespace["main"](options)

    expected_routes = ["shape"] if route == "shape" else ["shape", "stiffness"]
    loaded = [
        call for call in calls
        if isinstance(call, tuple) and call[0] == "load_network"
    ]
    assert [call[1] for call in loaded] == expected_routes
    assert all(call[2].parent == dirs[call[1]] for call in loaded)
    flow = [
        call[0] for call in calls if isinstance(call, tuple)
        and call[0] in {"predict", "decode", "local_stiffness", "project"}
    ]
    expected_batch = (
        ["predict", "decode", "local_stiffness", "project"]
        if route == "shape" else ["predict", "decode"]
    )
    assert flow == expected_batch * 2
    predicted_inputs = np.concatenate([
        call[2] for call in calls
        if isinstance(call, tuple) and call[0] == "predict"
    ])
    reordered = next(
        call[2] for call in calls
        if isinstance(call, tuple) and call[0] == "cell_order"
    )
    np.testing.assert_array_equal(predicted_inputs, reordered)
    interpolation = next(
        call[1] for call in calls
        if isinstance(call, tuple) and call[0] == "interpolation"
    )
    assert interpolation["penalty_factor"] == 2.0
    assert interpolation["void_youngs_modulus"] == 0.0
    assert "limit" in calls


def test_analysis_rejects_mismatched_route_weights(monkeypatch, tmp_path):
    """两路权重的子结构配置不一致时, 在加载网络和限制内存前报错."""
    namespace = load_script(ANALYSIS_SCRIPT)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    calls = []
    metadata = provider_metadata()
    other = dict(metadata, trace="full_trace")
    install_analysis_modules(
        monkeypatch, metadata, make_provider(metadata), calls,
        stiffness_metadata=other,
    )
    with pytest.raises(ValueError, match="不一致"):
        namespace["main"]([
            "--shape-dir", str(tmp_path / "shape"),
            "--stiffness-dir", str(tmp_path / "stiffness"),
            "--route", "stiffness",
        ])
    assert "limit" not in calls
    assert not any(
        isinstance(call, tuple) and call[0] == "load_network" for call in calls
    )


def test_exact_walkthrough_import_does_not_compute(monkeypatch):
    """精确走查导入时也不解析参数或运行有限元计算."""
    block_compute_imports(monkeypatch)
    path = ROOT.parent / "analysis_capability_substructure" / "walkthrough.py"
    namespace = load_script(path)
    assert callable(namespace["parse_args"])
    assert callable(namespace["main"])
