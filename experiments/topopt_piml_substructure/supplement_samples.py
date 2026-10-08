"""补充稀疏空间相关训练家族, 构造更新次数相同的回放对照数据集.

Notes
-----
这是基于训练难例分布的实验扩展. 验证与测试仅复制并核验摘要,
输入仅用于跨划分重复指纹检查, 不参与补样范围选择.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from mixed_sampling import MixedMaterialSampler
from soptx.ml.substructure.independent_contract import write_json, provider_metadata_matches
from soptx.ml.substructure.independent_data import generate_samples


def digest(path):
    """流式计算数据文件摘要, 避免将完整标签载入内存."""
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class SparseMaterialSampler:
    """复用带邻域过滤及 SIMP 采样, 按训练难例密度范围保留镜像家族.

    Parameters
    ----------
    base : MixedMaterialSampler
        提供材料候选, 单元编号与镜像指纹的既有采样器.
    count : int
        含镜像的训练补样数, 必须为正偶数.
    """

    def __init__(self, base, count):
        if count <= 0 or count % 2:
            raise ValueError("补样数须为正偶数")
        self.base, self.count = base, count
        self.pending, self.cursor, self.attempts = None, 0, 0
        self.groups = np.empty(count, dtype=np.int64)
        self.family = 0

    def __call__(self, rng, count, n_cells):
        """按批量返回材料场, 保持跨批次镜像配对."""
        if rng is not self.base.rng or n_cells != self.base.prototype.n_cells:
            raise ValueError("随机流或单元编号不一致")
        values = []
        for _ in range(count):
            if self.cursor >= self.count:
                raise ValueError("请求超过补样数量")
            if self.pending is not None:
                value, self.pending = self.pending, None
            else:
                for _ in range(100000):
                    value = self.base._correlated()
                    self.attempts += 1
                    rho = ((value - self.base.emin) / (1 - self.base.emin)) ** (1 / self.base.penal)
                    if not (.004 <= rho.mean() <= .04 and .025 <= rho.max() <= .15
                            and rho.min() <= 1e-12):
                        continue
                    mirrored, fingerprint = self.base._mirror_and_fingerprint(value)
                    if fingerprint in self.base.seen:
                        continue
                    self.base.seen.add(fingerprint)
                    self.pending = mirrored
                    self.family += 1
                    break
                else:
                    raise RuntimeError("稀疏材料采样连续拒绝, 请检查范围")
            self.groups[self.cursor] = self.family
            self.cursor += 1
            values.append(value)
        return np.stack(values)


def prepare_pair(source_dir, output_dir, *, count=2000, seed=2036, batch_size=32, resume_generated=False):
    """生成补样和训练回放对照, 将原验证及测试逐文件复制并核验.

    Parameters
    ----------
    source_dir, output_dir : Path
        原混合数据与新的配对数据根目录, 禁止覆盖.
    count : int
        每组增加的空间相关训练样本数, 含镜像.
    seed : int
        新训练候选与回放索引的独立随机子流主种子.
    batch_size : int
        补充材料的局部精确求解批量.

    resume_generated : bool
        仅恢复本脚本合并失败的生成结果, 重放采样并逐批核验输入.

    Returns
    -------
    dict
        新数据路径, 补样统计和隔离核验记录.
    """
    from soptx.backend import backend_manager as bm
    from soptx.fem.substructure.independent_targets import IndependentTargetProvider

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    raw = (source_dir / "manifest.json").read_bytes()
    source = json.loads(raw)
    if not source.get("complete") or source.get("sampling") != "mixed_material_v1":
        raise ValueError("来源须为完整混合样本")
    if count <= 0 or count % 2 or seed < 0 or seed == source["seed"] or batch_size <= 0:
        raise ValueError("补样数须为正偶数, batch_size 须为正, seed 须非负且不同于来源")
    bm.set_backend("numpy")
    meta = source["provider"]
    provider = IndependentTargetProvider(cell_size=tuple(meta["cell_size"]),
        n_fine=tuple(meta["n_fine"]), nu=meta["poisson_ratio"], trace_kind=meta["trace"],
        hypothesis=meta.get("material_hypothesis"))
    if not provider_metadata_matches(provider.metadata(), meta):
        raise ValueError("补样局部契约不匹配")
    streams = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(streams[0])
    params = source["sampling_parameters"]
    base = MixedMaterialSampler(provider.prototype, 10, rng, source["min_modulus"],
                                tuple(params["scale_range"]), params["filter_radius"], params["penal"])
    # 原划分输入只用于 canonical 指纹去重, 不查看验证或测试预测误差.
    for split in source["counts"]:
        inputs = np.load(source_dir / f"{split}_inputs.npy", mmap_mode="r")
        for value in inputs:
            base.seen.add(base._mirror_and_fingerprint(value)[1])
    sampler = SparseMaterialSampler(base, count)
    output_dir.mkdir(parents=True, exist_ok=resume_generated)
    extra = output_dir / "supplement"
    extra.mkdir(exist_ok=resume_generated)
    if resume_generated:
        if (output_dir / "summary.json").exists():
            raise FileExistsError("已完成数据不能恢复或覆盖")
        marker = json.loads((extra / "manifest.json").read_text(encoding="utf-8"))
        if (marker.get("source_manifest_sha256") != sha256(raw).hexdigest()
                or marker.get("source_dir") != str(source_dir)
                or marker.get("seed") != seed or marker.get("samples") != count
                or not provider_metadata_matches(marker.get("provider"), meta)):
            raise ValueError("恢复来源, 配置或局部契约不一致")
        saved = np.load(extra / "train_inputs.npy", mmap_mode="r")
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            expected = sampler(rng, stop - start, provider.prototype.n_cells)
            if not np.array_equal(saved[start:stop], expected):
                raise ValueError("恢复输入与同种子采样不一致")
        for name, width in (("inputs", meta["n_inputs"] if "n_inputs" in meta else provider.prototype.n_cells),
                            ("shape_targets", meta["n_shape_targets"]),
                            ("stiffness_targets", meta["n_stiffness_targets"])):
            array = np.load(extra / f"train_{name}.npy", mmap_mode="r")
            if (array.shape != (count, width) or array.dtype != np.float64
                    or not np.isfinite(array).all() or np.any(~np.any(array != 0, axis=1))):
                raise ValueError("恢复数组未完成")
    else:
        write_json(extra / "manifest.json", {"complete": False, "split": "train_only",
                   "source_dir": str(source_dir), "source_manifest_sha256": sha256(raw).hexdigest(),
                   "seed": seed, "samples": count, "provider": meta})
        generate_samples(provider, extra, file_prefix="train", n_samples=count, rng=rng,
                         batch_size=batch_size, min_modulus=source["min_modulus"], input_sampler=sampler)
    if sampler.pending is not None or sampler.cursor != count:
        raise RuntimeError("补样镜像家族不完整")
    old_groups = np.load(source_dir / "train_sample_groups.npy")
    old_types = np.load(source_dir / "train_sample_types.npy")
    family_ids = np.unique(old_groups[old_types == 2])
    replay_rng = np.random.default_rng(streams[1])
    chosen = replay_rng.choice(family_ids, count // 2, replace=False)
    replay = np.concatenate([np.flatnonzero(old_groups == family) for family in chosen])
    if len(replay) != count:
        raise ValueError("原空间相关家族须均为镜像双样本")
    np.save(extra / "train_sample_groups.npy", sampler.groups + max(
        np.max(np.load(source_dir / f"{split}_sample_groups.npy")) for split in source["counts"]))
    np.save(extra / "train_sample_types.npy", np.full(count, 2, dtype=np.uint8))
    np.save(output_dir / "control_replay_indices.npy", replay)
    audit = {"source_dir": str(source_dir), "source_manifest_sha256": sha256(raw).hexdigest(),
             "seed": seed, "training_only": True, "heldout_error_selection": False,
             "heldout_input_use": "canonical duplicate exclusion only",
             "density_selection": {"mean": [.004, .04], "max": [.025, .15], "min": [0., 1e-12]},
             "candidate_count": sampler.attempts, "families": sampler.family, "samples": count,
             "sampling": params, "stream_keys": [list(stream.spawn_key) for stream in streams],
             "preserved_files": {}, "datasets": {}}
    for mode in ("control", "supplemented"):
        target = output_dir / mode
        target.mkdir(exist_ok=resume_generated)
        manifest = deepcopy(source)
        manifest.update(complete=False, sampling="mixed_training_supplement_v1", seed=seed,
                        augmentation={"mode": mode, "source_manifest_sha256": sha256(raw).hexdigest(),
                                      "added_samples": count, "validation_and_test": "byte_identical_copy"})
        manifest["counts"]["train"] += count
        manifest["source_sampling_parameters"] = manifest.pop("sampling_parameters")
        manifest["source_split_streams"] = manifest.pop("split_streams")
        manifest["family_isolation"] = "original splits retained; new families train only; replay train only"
        write_json(target / "manifest.json", manifest)
        for name in ("inputs", "shape_targets", "stiffness_targets", "sample_types", "sample_groups"):
            original = np.load(source_dir / f"train_{name}.npy", mmap_mode="r")
            additional = (original[replay] if mode == "control" else
                          np.load(extra / f"train_{name}.npy", mmap_mode="r"))
            shape = (len(original) + count,) + original.shape[1:]
            merged = np.lib.format.open_memmap(target / f"train_{name}.npy", mode="w+",
                                               dtype=original.dtype, shape=shape)
            for start in range(0, len(original), 256):
                stop = min(start + 256, len(original))
                merged[start:stop] = original[start:stop]
            merged[len(original):] = additional
            merged.flush()
            del merged
        for split in ("validation", "test"):
            for path in source_dir.glob(f"{split}_*.npy"):
                destination = target / path.name
                before = digest(path)
                shutil.copyfile(path, destination)
                if digest(destination) != before or digest(path) != before:
                    raise RuntimeError("原验证或测试文件被改变")
                audit["preserved_files"].setdefault(mode, {})[path.name] = before
        manifest["complete"] = True
        write_json(target / "manifest.json", manifest)
        audit["datasets"][mode] = str(target)
    audit["complete"] = True
    write_json(extra / "manifest.json", {**audit, "provider": meta, "split": "train_only"})
    write_json(output_dir / "summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume-generated", action="store_true")
    args = parser.parse_args()
    prepare_pair(args.source_dir, args.output_dir, count=args.count, seed=args.seed,
                 batch_size=args.batch_size, resume_generated=args.resume_generated)
