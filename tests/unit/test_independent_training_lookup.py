"""同配置样本查找 find_training_data 的契约测试."""

import json
from types import SimpleNamespace

import pytest

from fealpy.backend import backend_manager as bm
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.independent_data import (
    find_training_data, prepare_training_data,
)


SAMPLING = {"n_train": 4, "n_validation": 2, "min_modulus": 1e-3, "seed": 7}


@pytest.fixture
def provider():
    """最小二维提供器, 精确求解代价可忽略."""
    bm.set_backend("numpy")
    return IndependentTargetProvider(cell_size=(1.0, 1.0), n_fine=(2, 2))


@pytest.fixture
def generated(provider, tmp_path):
    """用真实生成流程写入一份已完成样本, 返回样本根目录与其 manifest."""
    root = tmp_path / "samples"
    path = prepare_training_data(
        provider, root / "20260101T000000000000Z", batch_size=3, **SAMPLING,
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return root, manifest


def write_manifest(root, name, manifest):
    """在样本根目录下写入只含 manifest.json 的子目录."""
    directory = root / name
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_missing_root_returns_nothing(provider, tmp_path):
    """样本根目录不存在时不报错, 也不创建目录."""
    root = tmp_path / "absent"
    assert find_training_data(root, provider, **SAMPLING) == (None, [])
    assert not root.exists()


def test_generated_samples_are_found(provider, generated):
    """生成流程写入的记录可被同配置查找命中, 生成批大小不参与匹配."""
    root, _ = generated
    path, skipped = find_training_data(root, provider, **SAMPLING)
    assert path == root / "20260101T000000000000Z"
    assert skipped == []


def test_newest_match_wins_and_newer_broken_dirs_are_skipped(provider, generated):
    """多份匹配时返回目录名最新者; 更新的损坏或未完成目录被跳过并报告."""
    root, manifest = generated
    newest = write_manifest(root, "20260301T000000000000Z", manifest)
    incomplete = write_manifest(
        root, "20260401T000000000000Z", dict(manifest, complete=False),
    )
    broken = root / "20260501T000000000000Z"
    broken.mkdir()
    (broken / "manifest.json").write_text("{", encoding="utf-8")
    empty = root / "20260601T000000000000Z"
    empty.mkdir()
    not_dict = write_manifest(root, "20260701T000000000000Z", [manifest])
    (root / "20261231T000000000000Z.txt").write_text("", encoding="utf-8")

    path, skipped = find_training_data(root, provider, **SAMPLING)
    assert path == newest
    assert skipped == [not_dict, empty, broken, incomplete]


@pytest.mark.parametrize("override", [
    {"n_train": 5},
    {"n_validation": 3},
    {"min_modulus": 1e-6},
    {"seed": 8},
])
def test_sampling_mismatch_is_not_found(provider, generated, override):
    """样本数、采样下界或种子任一不同都不命中."""
    root, _ = generated
    path, skipped = find_training_data(
        root, provider, **{**SAMPLING, **override},
    )
    assert path is None
    assert skipped == []


def test_provider_mismatch_is_not_found(generated):
    """局部问题配置不同则不命中, 即使采样参数相同."""
    root, _ = generated
    other = IndependentTargetProvider(
        cell_size=(1.0, 1.0), n_fine=(2, 2), trace_kind="full_trace",
    )
    assert find_training_data(root, other, **SAMPLING) == (None, [])


@pytest.mark.parametrize("field, value", [
    ("schema", "independent_entries_v0"),
    ("sampling", "latin_hypercube"),
    ("dtype", "float32"),
])
def test_record_field_mismatch_is_not_found(provider, tmp_path, generated,
                                            field, value):
    """记录格式字段与当前生成流程不一致的目录不命中."""
    _, manifest = generated
    root = tmp_path / "other_root"
    root.mkdir()
    write_manifest(root, "20260101T000000000000Z", dict(manifest, **{field: value}))
    assert find_training_data(root, provider, **SAMPLING) == (None, [])


def test_legacy_provider_record_without_material_scaling_matches(provider, tmp_path,
                                                                 generated):
    """缺少 material_scaling 的旧记录按旧版固定标度补齐后仍可命中."""
    _, manifest = generated
    legacy = dict(manifest, provider=dict(manifest["provider"]))
    del legacy["provider"]["material_scaling"]
    root = tmp_path / "legacy_root"
    root.mkdir()
    directory = write_manifest(root, "20260101T000000000000Z", legacy)
    assert find_training_data(root, provider, **SAMPLING) == (directory, [])


def test_lookup_uses_metadata_only(generated):
    """查找只调用 provider.metadata(), 不做精确求解."""
    root, manifest = generated
    stub = SimpleNamespace(metadata=lambda: manifest["provider"])
    path, _ = find_training_data(root, stub, **SAMPLING)
    assert path == root / "20260101T000000000000Z"
