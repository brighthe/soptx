"""生成带独立数据划分及厚度镜像配对的混合材料样本.

Notes
-----
这是针对分布覆盖不足的实验扩展, 不属于 Huang2023 明示的采样配置.
精确标签复用 IndependentTargetProvider, 不加载优化快照.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from soptx.ml.substructure.independent_contract import (
    SCHEMA, provider_metadata_matches, write_json,
)
from soptx.ml.substructure.independent_data import generate_samples
from soptx.topology.filters.structured import apply_structured_density_filter


class MixedMaterialSampler:
    """按 20/20/60 配比分批采样, 保持镜像家族在同一数据划分内.

    Parameters
    ----------
    prototype : SubstructurePrototype
        提供网格与真实单元顺序的参考原型.
    count : int
        本组样本数, 必须是 10 的正整数倍.
    rng : numpy.random.Generator
        本组独立随机子流.
    min_modulus : float
        模量下界.
    scale_range : tuple of float
        整体模量缩放的 log-uniform 区间.
    radius, penal : float
        密度过滤半径与 SIMP 指数.
    replay : numpy.ndarray or None
        仅训练划分可使用的原始独立均匀训练输入.
    group_offset : int
        使不同划分的家族编号不相交的偏移量.
    seen_families : set or None
        可选的跨划分模式指纹集合. 原始模式及其镜像视为同一家族.
    """

    def __init__(self, prototype, count, rng, min_modulus, scale_range,
                 radius, penal, replay=None, group_offset=0, seen_families=None):
        if count <= 0 or count % 10:
            raise ValueError("混合样本数量必须为 10 的正整数倍")
        self.prototype, self.rng = prototype, rng
        self.emin, self.scales = min_modulus, scale_range
        self.radius, self.penal = radius, penal
        self.uniform_count = count // 5
        self.scaled_count = count // 5
        self.types = np.repeat(np.array([0, 1, 2], dtype=np.uint8),
                               [self.uniform_count, self.scaled_count, count*3//5])
        self.groups = np.empty(count, dtype=np.int64)
        self.replay = replay
        self.replay_indices = None if replay is None else rng.choice(
            len(replay), self.uniform_count, replace=False,
        )
        self.cursor, self.group, self.pending = 0, group_offset, None
        self.seen = set() if seen_families is None else seen_families
        spacing = np.asarray(prototype.cell_size) / np.asarray(prototype.n_fine)
        self.spacing = tuple(float(v) for v in spacing)
        self.halo = tuple(int(np.ceil(radius/h)) for h in spacing)
        axes = [(np.arange(n+2*halo)+.5-halo-n/2)*h
                for n, halo, h in zip(prototype.n_fine, self.halo, spacing)]
        self.points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        self.crop = tuple(slice(halo, halo+n) for halo, n in zip(self.halo, prototype.n_fine))

    def _correlated_candidate(self):
        """构造随机平滑梯度, 材料界面或杆状密度场并执行带邻域过滤.

        Returns
        -------
        numpy.ndarray
            按参考单元编号排列的 SIMP 模量.
        """
        rng = self.rng
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        center = rng.uniform(-.4, .4, 3)*np.asarray(self.prototype.cell_size)
        relative = self.points-center
        axial = relative @ direction
        kind = int(rng.integers(3))
        if kind == 0:
            width = rng.uniform(.25, 2.)*min(self.spacing)
            pattern = .5+.5*np.tanh(axial/width)
        elif kind == 1:
            pattern = (axial > 0).astype(np.float64)
        else:
            radial = np.linalg.norm(relative-axial[..., None]*direction, axis=-1)
            rod_radius = rng.uniform(.25, 1.5)*min(self.spacing)
            pattern = (radial < rod_radius).astype(np.float64)
        amplitude = 1. if rng.random() < .5 else 10**rng.uniform(np.log10(.03), 0.)
        density = amplitude*pattern
        filtered = apply_structured_density_filter(density, self.radius, self.spacing)[self.crop]
        modulus = self.emin+(1-self.emin)*filtered**self.penal
        return np.asarray(self.prototype.grid_to_cell_field(modulus), dtype=np.float64)

    def _correlated(self):
        """排除退化常量场, 避免不同家族重复生成全空或全实体样本.

        Returns
        -------
        numpy.ndarray
            有非零材料变化的局部模量.
        """
        for _ in range(100):
            values = self._correlated_candidate()
            if values.max()-values.min() > 1e-10*values.mean():
                return values
        raise RuntimeError("空间相关采样连续退化, 请检查过滤半径与局部网格")

    def _mirror_and_fingerprint(self, values):
        """返回厚度镜像及与镜像方向无关的家族指纹.

        Parameters
        ----------
        values : numpy.ndarray
            单个局部材料场.

        Returns
        -------
        tuple
            镜像场与 canonical SHA256 指纹.
        """
        mirrored = self.prototype.cell_to_grid_field(values)[:, :, ::-1]
        mirrored = np.asarray(self.prototype.grid_to_cell_field(mirrored)).copy()
        canonical = min(values.tobytes(), mirrored.tobytes())
        return mirrored, sha256(canonical).hexdigest()

    def __call__(self, rng, count, n_cells):
        """返回下一批输入, 保持批量边界两侧的镜像家族连续.

        Parameters
        ----------
        rng : numpy.random.Generator
            分批写入函数传入的同一个随机流.
        count, n_cells : int
            本批数量与每个样本的单元数.

        Returns
        -------
        numpy.ndarray
            形状为 (count, n_cells) 的输入.
        """
        if rng is not self.rng or n_cells != self.prototype.n_cells:
            raise ValueError("采样流或单元数量不一致")
        values = []
        for _ in range(count):
            i = self.cursor
            if i >= len(self.types):
                raise ValueError("请求数量超过混合样本配置")
            kind = self.types[i]
            if kind == 0:
                x = (self.rng.uniform(self.emin, 1., n_cells) if self.replay is None
                     else np.asarray(self.replay[self.replay_indices[i]], dtype=np.float64))
                _, fingerprint = self._mirror_and_fingerprint(x)
                if fingerprint in self.seen:
                    raise ValueError("独立均匀或回放输入存在重复家族")
                self.seen.add(fingerprint)
                self.group += 1
            elif self.pending is not None:
                x = self.pending
                self.pending = None
            else:
                for _ in range(100):
                    if kind == 1:
                        scale = np.exp(self.rng.uniform(np.log(self.scales[0]), np.log(self.scales[1])))
                        x = np.maximum(self.emin, scale*self.rng.uniform(self.emin, 1., n_cells))
                    else:
                        x = self._correlated()
                    mirrored, fingerprint = self._mirror_and_fingerprint(x)
                    if fingerprint not in self.seen:
                        self.seen.add(fingerprint)
                        self.pending = mirrored
                        break
                else:
                    raise RuntimeError("采样连续产生重复家族, 无法满足独立划分")
                self.group += 1
            self.groups[i] = self.group
            values.append(x)
            self.cursor += 1
        return np.stack(values)


def prepare_mixed_training_data(provider, output_dir, *, n_train, n_validation, n_test,
                                batch_size, min_modulus, seed, replay_dir,
                                scale_range=(1e-5, 1.), filter_radius=3., penal=3.):
    """准备混合训练, 验证及独立测试数据, 不覆盖已有目录.

    Parameters
    ----------
    provider : IndependentTargetProvider
        与原始权重相容的精确标签提供器.
    output_dir, replay_dir : pathlib.Path
        新输出目录及原训练数据目录.
    n_train, n_validation, n_test : int
        三个划分数量, 均须为 10 的正整数倍.
    batch_size : int
        每批精确求解数量.
    min_modulus : float
        严格正的材料下界.
    seed : int
        派生三个独立随机流的主种子.
    scale_range : tuple of float
        缩放材料的 log-uniform 区间.
    filter_radius, penal : float
        密度过滤半径和 SIMP 指数.

    Returns
    -------
    pathlib.Path
        具有精确双路线标签, 来源与家族编号记录的新数据目录.
    """
    output_dir, replay_dir = Path(output_dir), Path(replay_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    counts = dict(train=n_train, validation=n_validation, test=n_test)
    if any(n <= 0 or n % 10 for n in counts.values()):
        raise ValueError("三个划分数量均须为 10 的正整数倍")
    raw = (replay_dir/"manifest.json").read_bytes()
    source = json.loads(raw)
    if (source.get("complete") is not True or source.get("schema") != SCHEMA
            or source.get("sampling") != "independent_uniform"
            or source.get("input_quantity") != "normalized_young_modulus"
            or not provider_metadata_matches(source.get("provider"), provider.metadata())):
        raise ValueError("回放数据须为相同局部契约的已完成独立均匀样本")
    if seed == source.get("seed"):
        raise ValueError("mixed seed 须不同于原数据 seed, 避免新验证输入复用原随机流")
    if source.get("min_modulus", 0.) < min_modulus:
        raise ValueError("回放数据采样下界不能低于当前 min-modulus")
    replay = np.load(replay_dir/"train_inputs.npy", mmap_mode="r")
    if (replay.dtype != np.float64 or replay.shape !=
            (source["counts"]["train"], provider.prototype.n_cells)
            or n_train//5 > len(replay)):
        raise ValueError("回放输入形状, 类型或可用数量不符合要求")
    streams = np.random.SeedSequence(seed).spawn(3)
    manifest = dict(schema=SCHEMA, complete=False, provider=provider.metadata(),
                    input_quantity="normalized_young_modulus", dtype="float64",
                    counts=counts, sampling="mixed_material_v1", min_modulus=min_modulus,
                    max_modulus=1., seed=seed,
                    sampling_parameters=dict(proportions=[.2, .2, .6],
                        type_names=["uniform_or_train_replay", "scaled_uniform", "filtered_density_simp"],
                        scale_range=scale_range, scale_distribution="log_uniform_with_Emin_floor",
                        filter_radius=filter_radius, penal=penal, mirror_axis=2,
                        mirror_types=[1, 2], normalization="none",
                        correlated_patterns=["smooth_gradient", "material_interface", "rod"],
                        density_amplitude="50% at 1; 50% log-uniform in [0.03,1]",
                        filter_boundary="halo_then_crop"),
                    replay_source=dict(directory=str(replay_dir.resolve()),
                        manifest_sha256=sha256(raw).hexdigest(), split="train_only"),
                    family_isolation="split independent streams before scale/mirror augmentation",
                    duplicate_family_guard="canonical SHA256 of material and z mirror; shared across splits",
                    optimization_snapshots="not_used")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir/"manifest.json", manifest)
    offset = 0
    seen_families = set()
    for (split, count), stream in zip(counts.items(), streams):
        rng = np.random.default_rng(stream)
        sampler = MixedMaterialSampler(provider.prototype, count, rng, min_modulus,
            scale_range, filter_radius, penal, replay if split == "train" else None, offset,
            seen_families)
        generate_samples(provider, output_dir, file_prefix=split, n_samples=count,
                         rng=rng, batch_size=batch_size, min_modulus=min_modulus,
                         input_sampler=sampler)
        if sampler.cursor != count or sampler.pending is not None:
            raise RuntimeError("混合样本或镜像配对未完成")
        np.save(output_dir/f"{split}_sample_types.npy", sampler.types)
        np.save(output_dir/f"{split}_sample_groups.npy", sampler.groups)
        if sampler.replay_indices is not None:
            np.save(output_dir/"train_replay_indices.npy", sampler.replay_indices)
        manifest.setdefault("split_streams", {})[split] = dict(spawn_key=list(stream.spawn_key),
            type_counts=[count//5, count//5, count*3//5], group_id_range=[offset+1, sampler.group])
        offset += count
        write_json(output_dir/"manifest.json", manifest)
    manifest["complete"] = True
    write_json(output_dir/"manifest.json", manifest)
    return output_dir
