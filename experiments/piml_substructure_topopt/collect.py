# -*- coding: utf-8 -*-
"""PIML 子结构拓扑优化结果汇总与对比制图 (通用 2D/3D)."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURE_DATA_DIR, OUTPUT_DIR, load


def summarize_pair(prefix: str, label: str) -> None:
    piml_json = OUTPUT_DIR / f"{prefix}_piml_route_a_result.json"
    fea_json = OUTPUT_DIR / f"{prefix}_fea_baseline_result.json"

    if not piml_json.is_file() or not fea_json.is_file():
        return

    piml_data = json.loads(piml_json.read_text(encoding="utf-8"))
    fea_data = json.loads(fea_json.read_text(encoding="utf-8"))

    piml_hist = piml_data["history"]
    fea_hist = fea_data["history"]

    c_piml_final = piml_data["final_compliance"]
    c_fea_final = fea_data["final_compliance"]
    rel_comp_err = abs(c_piml_final - c_fea_final) / c_fea_final * 100.0

    t_solve_piml = piml_data["mean_solve_time_ms"]
    t_solve_fea = fea_data["mean_solve_time_ms"]

    print(f"\n=======================================================")
    print(f" [{label}] PIML 子结构拓扑优化 vs 精确 FEA 基线 对比汇总")
    print(f"=======================================================")
    print(f"  PIML 最终柔度: {c_piml_final:.4f}")
    print(f"  FEA  最终柔度: {c_fea_final:.4f}")
    print(f"  最终柔度相对误差: {rel_comp_err:.4f}%")
    print(f"  PIML 平均单步求解耗时: {t_solve_piml:.2f} ms")
    print(f"  FEA  平均单步求解耗时: {t_solve_fea:.2f} ms")
    print(f"  PIML 迭代步数: {len(piml_hist)}, FEA 迭代步数: {len(fea_hist)}")

    # 绘制收敛历程对比图
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    iters_piml = [h["iter"] for h in piml_hist]
    comp_piml = [h["compliance"] for h in piml_hist]
    iters_fea = [h["iter"] for h in fea_hist]
    comp_fea = [h["compliance"] for h in fea_hist]

    axes[0].plot(iters_fea, comp_fea, 'k--', label="Exact FEA Baseline", linewidth=2.0)
    axes[0].plot(iters_piml, comp_piml, 'r-', label="PIML Route A (Huang 2023 Eq 17)", linewidth=1.8)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Compliance")
    axes[0].set_title(f"[{label}] Compliance Convergence (Error: {rel_comp_err:.2f}%)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    change_piml = [h["max_change"] for h in piml_hist]
    change_fea = [h["max_change"] for h in fea_hist]
    axes[1].plot(iters_fea, change_fea, 'k--', label="Exact FEA", linewidth=1.5)
    axes[1].plot(iters_piml, change_piml, 'b-', label="PIML Route A", linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\max |\Delta \rho|$")
    axes[1].set_title(f"[{label}] Density Change History (Log Scale)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    comp_plot_path = FIGURE_DATA_DIR / f"{prefix}_convergence_comparison.png"
    plt.savefig(comp_plot_path)
    plt.close()
    print(f"[+] 收敛曲线对比图已保存至: {comp_plot_path}")

    summary_report = {
        "case_prefix": prefix,
        "final_compliance_fea": c_fea_final,
        "final_compliance_piml": c_piml_final,
        "relative_compliance_error_percent": rel_comp_err,
        "mean_solve_time_ms_fea": t_solve_fea,
        "mean_solve_time_ms_piml": t_solve_piml,
        "total_iters_fea": len(fea_hist),
        "total_iters_piml": len(piml_hist),
    }
    summary_json = FIGURE_DATA_DIR / f"{prefix}_summary_comparison.json"
    summary_json.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"[+] 汇总指标已保存至: {summary_json}")


def main() -> None:
    summarize_pair("mbb", "2D MBB Beam")
    summarize_pair("mbb_3d", "3D MBB Beam (Huang 2023 Fig 6)")


if __name__ == "__main__":
    main()
