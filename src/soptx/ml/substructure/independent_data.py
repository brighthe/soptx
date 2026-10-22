"""独立分量路线的磁盘数据集生成与查找.

Notes
-----
数据按批次生成并直接写入内存映射数组, 全部写完后才标记完成.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .independent_contract import (
    SCHEMA, metadata_widths, provider_metadata_matches, write_json,
)

TARGET_NAMES = ("shape", "stiffness")


def normalize_targets(targets):
    """校验并规范标签路线集合, 返回按 TARGET_NAMES 顺序排列的元组.

    Parameters
    ----------
    targets : iterable of str
        要生成或要求的标签路线, 取自 "shape" 与 "stiffness", 至少一个.

    Returns
    -------
    tuple of str
        去重并按 TARGET_NAMES 顺序排列的路线.

    Raises
    ------
    ValueError
        含未知路线或为空.
    """
    chosen = tuple(name for name in TARGET_NAMES if name in set(targets))
    unknown = set(targets) - set(TARGET_NAMES)
    if unknown or not chosen:
        raise ValueError(f"targets 须为 {TARGET_NAMES} 的非空子集, 得到 {tuple(targets)!r}")
    return chosen


def dataset_targets(manifest):
    """读取数据集记录中实际包含的标签路线; 旧记录无 targets 字段时视为两类齐全.

    Parameters
    ----------
    manifest : dict
        数据集的 manifest.json 内容.

    Returns
    -------
    tuple of str
        数据集含有的标签路线.
    """
    return normalize_targets(manifest.get("targets", TARGET_NAMES))


def generate_samples(
    provider, output_dir, *, file_prefix, n_samples, rng,
    batch_size=64, min_modulus=1e-6, input_sampler=None, targets=TARGET_NAMES,
):
    """分批生成一组独立随机材料及精确标签并写入磁盘.

    Parameters
    ----------
    provider : callable
        接收 (batch, n_cells) 数组, 返回 shape 与 stiffness 独立分量.
        metadata() 必须描述独立分量编号及有限元配置.
    output_dir : str or Path
        已存在的输出目录.
    file_prefix : str
        输出文件名前缀, 不限定样本用途.
    n_samples : int
        本组样本总数.
    rng : numpy.random.Generator
        本组样本使用的随机数生成器.
    batch_size : int
        每次局部精确计算的样本数.
    min_modulus : float
        归一化杨氏模量的严格正下界, 避免零刚度样本.

    input_sampler : callable or None
        可选采样回调 (rng, count, n_cells), 返回有限且位于 [min_modulus, 1] 的输入.
        None 保持原有独立均匀采样. 回调状态及分组由调用方记录.
    targets : iterable of str
        要写盘的标签路线, "shape" 与 "stiffness" 的非空子集, 默认两类都写.
        provider 仍同时算出两类标签, 未选的路线只是不写盘.

    Returns
    -------
    None
        将 {file_prefix}_inputs.npy 与所选路线的 {file_prefix}_<route>_targets.npy
        写入 output_dir. 所有数组均为 float64, 第一维为 n_samples.

    Notes
    -----
    batch_size 仅控制每批生成数量, 最后一批可能不足 batch_size.
    """
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples 与 batch_size 必须为正")
    if not np.isfinite(min_modulus) or not 0 < min_modulus < 1:
        raise ValueError("min_modulus 必须位于 (0, 1)")
    if not isinstance(file_prefix, str) or not file_prefix:
        raise ValueError("file_prefix 必须为非空字符串")
    targets = normalize_targets(targets)

    widths = metadata_widths(provider.metadata())
    widths = {name: widths[name] for name in ("inputs", *(f"{route}_targets" for route in targets))}
    output_dir = Path(output_dir)
    paths = {
        name: output_dir / f"{file_prefix}_{name}.npy"
        for name in widths
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"样本文件已存在, 不覆盖: {existing[0]}")

    arrays = {
        name: np.lib.format.open_memmap(
            paths[name], mode="w+", dtype=np.float64,
            shape=(n_samples, width),
        )
        for name, width in widths.items()
    }
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        inputs = (rng.uniform(min_modulus, 1.0, size=(stop - start, widths["inputs"]))
                  if input_sampler is None else
                  np.asarray(input_sampler(rng, stop - start, widths["inputs"]), dtype=np.float64))
        if (inputs.shape != (stop - start, widths["inputs"])
                or not np.isfinite(inputs).all()
                or np.any(inputs < min_modulus) or np.any(inputs > 1.0)):
            raise ValueError(f"{file_prefix} 样本 {start}:{stop} 的输入形状或取值非法")
        labels = provider(inputs)
        arrays["inputs"][start:stop] = inputs
        for name in (f"{route}_targets" for route in targets):
            width = widths[name]
            values = np.asarray(labels[name.removesuffix("_targets")])
            if values.shape != (stop - start, width) or not np.isfinite(values).all():
                raise ValueError(f"{file_prefix} 样本 {start}:{stop} 的 {name} 非法")
            arrays[name][start:stop] = values
        if start == 0 or stop == n_samples or stop % (batch_size * 100) == 0:
            print(f"{file_prefix}: {stop}/{n_samples}", flush=True)
    for array in arrays.values():
        array.flush()
    del arrays


def _dataset_manifest(meta, *, n_train, n_validation, min_modulus, seed, targets=TARGET_NAMES):
    """构造除完成标记外的数据集记录, 供生成与查找共用."""
    return {
        "schema": SCHEMA, "provider": meta,
        "seed": seed, "input_quantity": "normalized_young_modulus",
        "sampling": "independent_uniform", "min_modulus": min_modulus,
        "max_modulus": 1.0, "dtype": "float64",
        "counts": {"train": n_train, "validation": n_validation},
        "targets": list(normalize_targets(targets)),
    }


def prepare_training_data(
    provider, output_dir, *, n_train=400_000, n_validation=40_000,
    batch_size=64, min_modulus=1e-6, seed=2026, targets=TARGET_NAMES,
):
    """准备相互独立的训练集与验证集及其精确标签.

    Parameters
    ----------
    provider : callable
        接收 (batch, n_cells) 数组, 返回 shape 与 stiffness 独立分量.
        metadata() 必须描述独立分量编号及有限元配置.
    output_dir : str or Path
        新训练数据目录, 已有目录不覆盖.
    n_train, n_validation : int
        互相独立的训练与验证样本数.
    batch_size : int
        每次局部精确计算的样本数.
    min_modulus : float
        归一化杨氏模量的严格正下界, 避免零刚度样本.
    seed : int
        派生训练与验证独立随机流的主种子.
    targets : iterable of str
        要写盘的标签路线, "shape" 与 "stiffness" 的非空子集, 默认两类都写;
        未选的路线不生成对应 NPY 文件, 记入 manifest 的 targets.

    Returns
    -------
    Path
        新生成的训练数据目录路径, 数据与元信息保存在以下文件中:

        - manifest.json: 数据格式版本、参考子结构及独立分量配置、
          输入物理量、采样方式、随机种子、材料取值范围、数据类型、
          训练与验证样本数、标签路线 targets 及完成标记 complete.
        - train_inputs.npy: 训练集归一化杨氏模量,
          形状为 (n_train, n_cells).
        - validation_inputs.npy: 验证集归一化杨氏模量,
          形状为 (n_validation, n_cells).
        - train_shape_targets.npy: 训练集内部延拓矩阵的独立条目标签,
          形状为 (n_train, n_shape_targets).
        - validation_shape_targets.npy: 验证集内部延拓矩阵的独立条目标签,
          形状为 (n_validation, n_shape_targets).
        - train_stiffness_targets.npy: 训练集缩聚刚度矩阵的独立条目标签,
          形状为 (n_train, n_stiffness_targets).
        - validation_stiffness_targets.npy: 验证集缩聚刚度矩阵的独立条目标签,
          形状为 (n_validation, n_stiffness_targets).

        所有数组均为 float64. 第一维为对应数据集的总样本数,
        batch_size 仅控制每批生成数量, 最后一批可能不足 batch_size.
        仅在全部数据写入完成后, complete 才会设为 True.
    """
    if min(n_train, n_validation, batch_size) <= 0:
        raise ValueError("样本数与 batch_size 必须为正")
    if not np.isfinite(min_modulus) or not 0 < min_modulus < 1:
        raise ValueError("min_modulus 必须位于 (0, 1)")
    targets = normalize_targets(targets)
    meta = provider.metadata()
    metadata_widths(meta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "complete": False,
        **_dataset_manifest(
            meta, n_train=n_train, n_validation=n_validation,
            min_modulus=min_modulus, seed=seed, targets=targets,
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    streams = np.random.SeedSequence(seed).spawn(2)
    for (file_prefix, count), stream in zip(manifest["counts"].items(), streams):
        generate_samples(
            provider, output_dir, file_prefix=file_prefix, n_samples=count,
            rng=np.random.default_rng(stream), batch_size=batch_size,
            min_modulus=min_modulus, targets=targets,
        )
    manifest["complete"] = True
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def find_training_data(
    samples_root, provider, *, n_train=400_000, n_validation=40_000,
    min_modulus=1e-6, seed=2026, targets=TARGET_NAMES,
):
    """在样本根目录下按目录名降序查找与给定配置一致的已完成数据集.

    Parameters
    ----------
    samples_root : str or Path
        prepare_training_data 输出目录的父目录, 支持配置名称或旧时间戳目录.
    provider : callable
        局部精确求解提供器, 仅使用 metadata() 与数据集记录比较.
    n_train, n_validation : int
        期望的训练与验证样本数.
    min_modulus : float
        期望的归一化杨氏模量采样下界.
    seed : int
        期望的材料采样主种子.
    targets : iterable of str
        期望的标签路线集合; 旧记录无 targets 字段时视为两类齐全.

    Returns
    -------
    path : Path or None
        按目录名降序命中的首个匹配数据集目录, 无匹配时为 None.
    skipped : list of Path
        遍历至命中目录前遇到的缺少 manifest.json、无法解析或 complete
        不为 True 的子目录, 按目录名降序排列. 这些目录不参与匹配,
        也不被修改.

    Notes
    -----
    除 provider 字段用 provider_metadata_matches 比较外, 其余记录字段须
    与 prepare_training_data 写入的内容逐项相等. 生成批大小不影响输入
    随机序列, 也未写入记录, 因此不参与匹配.
    """
    samples_root = Path(samples_root)
    if not samples_root.is_dir():
        return None, []
    expected = _dataset_manifest(
        provider.metadata(), n_train=n_train, n_validation=n_validation,
        min_modulus=min_modulus, seed=seed, targets=targets,
    )
    expected_provider = expected.pop("provider")
    expected_targets = tuple(expected.pop("targets"))
    skipped = []
    for directory in sorted(
            (path for path in samples_root.iterdir() if path.is_dir()),
            key=lambda path: path.name, reverse=True):
        try:
            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8"),
            )
        except (OSError, ValueError):
            skipped.append(directory)
            continue
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            skipped.append(directory)
            continue
        try:
            actual_targets = dataset_targets(manifest)
        except ValueError:
            skipped.append(directory)
            continue
        if (provider_metadata_matches(manifest.get("provider"), expected_provider)
                and actual_targets == expected_targets
                and all(manifest.get(key) == value
                        for key, value in expected.items())):
            return directory, skipped
    return None, skipped
