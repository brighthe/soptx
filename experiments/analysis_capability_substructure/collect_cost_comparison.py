"""比较三条成本测量路径保存的完整位移与应变能.

输入取 fa、full_trace、linear_corner 各自首个正式样本的 record.json: 只有首个正式样本
会在同目录保存 displacement.npy, 重复样本不保存. 脚本先核对三者的物理条件、网格与位移
文件指纹, 再分块计算 full_trace 与 linear_corner 相对 FA 的全场位移和应变能误差.
--output 不覆盖已有文件; 等价性未通过时仍写出数值结果, 并以非零状态退出.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_sample(path: Path) -> tuple[dict, np.ndarray]:
    """读取已通过验收的样本，并分块核对位移指纹."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] != "PASS":
        raise ValueError(f"样本未通过: {path}")
    artifact = record["displacement_artifact"]
    values = np.load(path.parent / artifact["path"], mmap_mode="r")
    if list(values.shape) != artifact["shape"] or str(values.dtype) != artifact["dtype"]:
        raise ValueError(f"位移形状或类型不匹配: {path}")
    digest = hashlib.sha256()
    flat = values.reshape(-1)
    for start in range(0, flat.size, 262144):
        block = np.ascontiguousarray(flat[start:start + 262144])
        if not np.isfinite(block).all():
            raise ValueError(f"位移含非有限值: {path}")
        digest.update(memoryview(block).cast("B"))
    if digest.hexdigest() != artifact["sha256"]:
        raise ValueError(f"位移文件指纹不匹配: {path}")
    if digest.hexdigest() != record["validation"]["displacement_sha256"]:
        raise ValueError(f"位移与数值验收指纹不匹配: {path}")
    return record, flat


def relative_displacement(value: np.ndarray, reference: np.ndarray) -> float:
    """分块计算完整位移的相对二范数误差."""
    if value.shape != reference.shape:
        raise ValueError("全场位移长度不一致.")
    numerator = denominator = 0.0
    for start in range(0, reference.size, 262144):
        ref = reference[start:start + 262144]
        delta = value[start:start + 262144] - ref
        numerator += float(np.dot(delta, delta))
        denominator += float(np.dot(ref, ref))
    if denominator <= 0.0:
        raise ValueError("FA 位移范数为零.")
    return float(np.sqrt(numerator / denominator))


def main() -> int:
    """保存物理条件检查、相对误差与 full_trace 等价性判定."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fa", type=Path, required=True)
    parser.add_argument("--full-trace", type=Path, required=True)
    parser.add_argument("--linear-corner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有证据: {args.output}")
    paths = {"fa": args.fa, "full_trace": args.full_trace,
             "linear_corner": args.linear_corner}
    loaded = {name: load_sample(path) for name, path in paths.items()}
    fa, fa_u = loaded["fa"]
    reference_energy = fa["validation"]["strain_energy"]
    if not np.isfinite(reference_energy) or reference_energy <= 0.0:
        raise ValueError("FA 应变能无效.")
    comparisons = {}
    for name, (record, values) in loaded.items():
        if record["physical_data"] != fa["physical_data"]:
            raise ValueError(f"{name} 与 FA 的物理设置不一致.")
        for key in ("full_dofs", "global_fine_grid"):
            if record["problem_data"][key] != fa["problem_data"][key]:
                raise ValueError(f"{name} 与 FA 的 {key} 不一致.")
        energy = record["validation"]["strain_energy"]
        comparisons[name] = {
            "displacement_relative_error": relative_displacement(values, fa_u),
            "strain_energy_relative_error": abs(energy - reference_energy) / reference_energy,
            "equilibrium_relative_residual": record["validation"]["equilibrium_relative_residual"],
        }
    full = comparisons["full_trace"]
    passed = all(full[key] <= 1.0e-11 for key in (
        "displacement_relative_error", "strain_energy_relative_error"))
    result = {
        "schema_version": "substructure-cost-comparison-v1",
        "physical_conditions_match": True,
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "comparisons": comparisons,
        "full_trace_fa_tolerance": 1.0e-11,
        "full_trace_fa_equivalence": "PASS" if passed else "FAIL",
        "linear_corner_fa_error": "REPORTED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
