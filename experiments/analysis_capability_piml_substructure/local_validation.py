"""独立测试材料场上的局部预测与矩阵诊断."""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from soptx.ml.substructure.validation import LocalPredictionEvaluator, load_local_model


def _write_json(path, value):
    """将有限数值与显式空值写入 JSON.

    Parameters
    ----------
    path : Path
        输出文件路径.
    value : dict
        可序列化记录.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_local_predictions(network, provider, route, *, n_test=1000,
                               min_modulus=1e-6, batch_size=32, seed=2027,
                               device="cpu", output_dir):
    """分批执行独立测试并保存逐样本误差与约束诊断.

    Parameters
    ----------
    network : torch.nn.Module
        已加载权重的网络.
    provider : IndependentTargetProvider
        精确矩阵与约束补全提供器.
    route : str
        shape 或 stiffness.
    n_test, batch_size : int
        测试样本总数与计算批量.
    min_modulus : float
        独立均匀采样的严格正下界, 上界为 1.
    seed : int
        SeedSequence 熵, 使用保留的 spawn_key=(2,).
    device : str
        网络预测设备; 精确参考经 provider 返回 NumPy float64.
    output_dir : str or Path
        必须不存在的新结果目录.

    Returns
    -------
    dict
        指标统计与完成状态. COMPLETED 仅表示流程完成, 不表示精度验收通过.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    for name, value in (("n_test", n_test), ("batch_size", batch_size), ("seed", seed)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} 必须为有效整数")
    if isinstance(min_modulus, (bool, np.bool_)) or not isinstance(min_modulus, Real) or not np.isfinite(min_modulus) or not 0 < min_modulus < 1:
        raise ValueError("min_modulus 必须满足 0 < min_modulus < 1")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    completed = 0
    inputs_file = None
    try:
        metadata = provider.metadata()
        source = getattr(network, "local_validation_source", None)
        if source is not None:
            if source.get("route") != route or source.get("provider_metadata") != metadata:
                raise ValueError("加载模型的路线或子结构配置与当前验证不一致")
        else:
            source = {"status": "unregistered", "route": route}
        config = {"route": route, "n_test": int(n_test), "batch_size": int(batch_size),
                  "min_modulus": float(min_modulus), "seed": int(seed), "spawn_key": [2],
                  "device": str(device), "dtype": "float64", "provider_metadata": metadata,
                  "model_source": source, "accuracy_acceptance": "not_evaluated",
                  "zero_reference_policy": "relative metric is null when reference norm is zero",
                  "eigenvalue_tolerance": "100 * float64_eps * max(abs(eigenvalues))"}
        _write_json(destination / "config.json", config)
        rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(2,)))
        inputs_file = np.lib.format.open_memmap(destination / "inputs.npy", mode="w+", dtype=np.float64,
                                              shape=(int(n_test), int(metadata["n_cells"])))
        evaluator = LocalPredictionEvaluator(network, provider, route, device)
        collected = {}
        spd_counts = {"positive": 0, "near_zero": 0, "negative": 0}
        with (destination / "per_sample.jsonl").open("x", encoding="utf-8") as stream:
            for start in range(0, int(n_test), int(batch_size)):
                stop = min(start + int(batch_size), int(n_test))
                inputs = rng.uniform(float(min_modulus), 1.0, size=(stop - start, int(metadata["n_cells"])))
                inputs_file[start:stop] = inputs
                for offset, diagnostics in enumerate(evaluator.evaluate(inputs)):
                    category = diagnostics["deformation_spectrum"]
                    spd_counts[category] += 1
                    for name, value in diagnostics.items():
                        if name not in ("deformation_spectrum", "undefined_metrics"):
                            collected.setdefault(name, []).append(value)
                    record = {"sample_index": start + offset, **diagnostics}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    completed += 1
        inputs_file.flush()
        statistics = {}
        for name, values in collected.items():
            finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
            statistics[name] = {"count": int(finite.size), "undefined_count": len(values) - int(finite.size),
                                "mean": float(finite.mean()) if finite.size else None,
                                "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
                                "max": float(finite.max()) if finite.size else None,
                                "min": float(finite.min()) if finite.size else None}
        summary = {"status": "COMPLETED", "accuracy_acceptance": "not_evaluated", "n_completed": completed,
                   "route": route, "metrics": statistics, "deformation_spectrum_counts": spd_counts,
                   "model_source": source, "output_dir": str(destination.resolve())}
        _write_json(destination / "summary.json", summary)
        return summary
    except Exception as error:
        if inputs_file is not None:
            inputs_file.flush()
        _write_json(destination / "summary.json", {"status": "FAILED", "n_completed": completed,
                                                    "error_type": type(error).__name__, "error": str(error)})
        raise
