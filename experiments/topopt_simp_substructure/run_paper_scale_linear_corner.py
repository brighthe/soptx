"""运行 Huang2023 最粗 MBB 规模的角点接口优化, 独立保留每次输出."""
import argparse
import json
from pathlib import Path

import run_substructure_full_trace_vs_linear_corner as experiment


def main():
    """解析独立运行目录, 复用优化流程并记录参考求解资源估计.

    Returns
    -------
    int
        优化入口的退出状态.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="不存在的新运行目录, 已存在时拒绝运行")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    # 排他创建父目录, 防止原入口的清空逻辑触及已有运行.
    run_dir.mkdir(parents=True, exist_ok=False)
    experiment.GRID = (390, 65, 65)
    experiment.OUTPUT_DIR = run_dir
    source = Path(experiment.__file__).read_bytes()
    (run_dir / "optimization_entry_snapshot.py").write_bytes(source)
    (run_dir / "launcher_snapshot.py").write_bytes(Path(__file__).read_bytes())
    result = experiment.main(["--trace", "linear_corner", "--preconditioner", "jacobi"])
    # 优化结束后记录粗略资源需求. 此估计不能替代装配存储结构分析.
    memory = {}
    meminfo = Path("/proc/meminfo")
    for line in (meminfo.read_text().splitlines() if meminfo.exists() else []):
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[key + "_gib"] = int(value.split()[0]) / 1024**2
    n_sub = 78 * 13 * 13
    full_dofs = 3 * 391 * 66 * 66
    interface_dofs = full_dofs - 3 * n_sub * 4**3
    estimate = {
        "grid": list(experiment.GRID),
        "full_dofs": full_dofs,
        "full_trace_interface_dofs": interface_dofs,
        "substructure_count": n_sub,
        "historical_full_trace_peak_gib": 10.256149291992188,
        "rough_scaled_full_trace_peak_gib": 10.256149291992188 * n_sub / (42 * 7 * 7),
        "memory": memory,
        "status": "requires_reference_solver_memory_review",
        "note": "线性缩放仅为初估, 尚未执行 full_trace 参考求解."
    }
    (run_dir / "reference_resource_estimate.json").write_text(
        json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(estimate, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
