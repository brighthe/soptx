"""悬臂梁三组离散的冻结验收、图 5.8--5.10 与表 5.5.

由 ``plot.py --case stress-cantilever-comparison`` 调用. 只重分析已入库的
最终物理密度, 不运行优化; 复用现有牵引跳量与约束验收接口.
"""

import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import config
from analysis import provenance
from analysis.discretization_probe import probe_design, write_outputs
from ._base import load_density, paper_rcparams, save_figure

SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "analyzer-lfem__order-3__lambda-max-10000.0",
    "analyzer-huzhang__order-2",
    "analyzer-huzhang__order-3",
)
LABELS = (r"LFEM $p=3$", r"HZMFEM $k=2$", r"HZMFEM $k=3$")
TAGS = ("lfem-3", "huzhang-2", "huzhang-3")
COLORS = ("#d62728", "#2ca02c", "#9467bd")


def prepare():
    """复核三组终态, 保存来源、全单元可行性和牵引跳量证据.

    Returns
    -------
    list of dict
        每组的网格、场量、历史、摘要和验收结果.
    """
    base = config.OUTPUT_DIR / SOURCE_CASE
    rows = []
    for run, tag, label in zip(REQUIRED_RUNS, TAGS, LABELS):
        directory = base / run
        summary = json.loads((directory / "summary.json").read_text())
        settings = json.loads((directory / "config.json").read_text())
        if (not summary["converged"] or summary["acceptance_solid_threshold"] is not None
                or summary["load_discretization"] != "p1_trace_interpolation"):
            raise ValueError(f"{run}: 运行未满足本次比较口径")
        history = json.loads((directory / "history.json").read_text())["scalar_histories"]
        with np.load(directory / "final_optimizer_state.npz") as state:
            peak = max(float(np.max(history["max_multiplier"])), float(state["lamb"].max()))
        if peak >= 3000:
            raise ValueError(f"{run}: 不能以未激活的乘子上限解释参数差异")
        payload, fields, mesh = probe_design(directory)
        if not all(check["passed"] for check in payload["checks"].values()):
            raise RuntimeError(f"{run}: 冻结再分析未通过验收")
        payload["provenance"] = provenance.run_stamp()
        payload["observed_multiplier_max"] = peak
        write_outputs(payload, fields, mesh)
        node, cell, rho = load_density(directory / "density_final.vtu")
        if not np.array_equal(rho, fields["density"]):
            raise RuntimeError("绘图密度与再分析密度不一致")
        rows.append(dict(tag=tag, label=label, summary=summary, history=history,
                         settings=settings, payload=payload, fields=fields,
                         tri=Triangulation(node[:, 0], node[:, 1], cell)))
        print(f"[verified] {tag}: max g={payload['max_g_accepted']:.9g}, "
              f"jump={payload['traction_jump']['cell_abs_solid_max']:.6g}", flush=True)
    reference = rows[0]["settings"]
    for row in rows[1:]:
        for key in ("model", "stress_constraint"):
            if row["settings"][key] != reference[key]:
                raise ValueError(f"{key}: 三组设置不一致")
        a = dict(reference["optimization"])
        b = dict(row["settings"]["optimization"])
        a.pop("lambda_max")
        b.pop("lambda_max")
        if a != b:
            raise ValueError("除未激活乘子上限以外的优化设置不一致")
    return rows


def draw(rows):
    """生成三组构型与应力、收敛历史及实体带牵引跳量图.

    Parameters
    ----------
    rows : list of dict
        ``prepare`` 返回的已验收数据.
    """
    paper_rcparams()
    fig, axes = plt.subplots(3, 2, figsize=(5.9, 5.0), layout="constrained")
    vmax = 1.05
    for i, row in enumerate(rows):
        fields, tag = row["fields"], row["tag"]
        for j, (values, cmap, title) in enumerate((
                (fields["density"], "gray_r", "Physical density"),
                (fields[f"vm_app__{tag}"], "jet", "Apparent stress ratio"))):
            ax = axes[i, j]
            artist = ax.tripcolor(row["tri"], facecolors=values, cmap=cmap,
                                 vmin=0, vmax=1 if j == 0 else vmax, edgecolors="none")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"({chr(97+2*i+j)}) {row['label']}", fontsize=9)
        fig.colorbar(artist, ax=axes[i, 1], fraction=.035, pad=.025,
                     ticks=[0, .25, .5, .75, 1])
    save_figure(fig, "stress_cantilever_comparison_topologies", dpi=600)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.65), layout="constrained")
    for row, color, ls in zip(rows, COLORS, ("-", "--", "-.")):
        h = row["history"]
        x = np.arange(1, len(h["volfrac"])+1)
        axes[0].plot(x, h["volfrac"], color=color, ls=ls, lw=1.1, label=row["label"])
        axes[1].semilogy(x, h["max_relative_violation_solid"], color=color,
                         ls=ls, lw=1.1, label=row["label"])
    axes[0].set_ylabel("Volume fraction")
    axes[1].set_ylabel(r"Maximum constraint $g_{\max}$")
    axes[1].axhline(.005, color="0.25", ls=":", lw=1, label=r"$\delta_g=0.005$")
    for ax, title in zip(axes, ("(a) Volume history", "(b) Constraint history")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("MMA updates")
        ax.tick_params(labelsize=8)
        ax.grid(ls=":", alpha=.4)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=6, framealpha=.95, handlelength=1.5, borderpad=.4)
    save_figure(fig, "stress_cantilever_comparison_history", formats=("png", "pdf"), dpi=600)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(5.9, 6.7), layout="constrained")
    jump_max = max(row["payload"]["traction_jump"]["cell_abs_solid_max"] for row in rows)
    for i, (ax, row) in enumerate(zip(axes, rows)):
        f, tri = row["fields"], row["tri"]
        mask = (f["density"] > .9) & ~f["pad_mask"]
        ax.tripcolor(tri, facecolors=f["density"], cmap="gray_r", vmin=0, vmax=1)
        subset = Triangulation(tri.x, tri.y, tri.triangles[mask])
        artist = ax.tripcolor(subset, facecolors=f[f"absjump__{row['tag']}"][mask],
                             cmap="viridis", vmin=0, vmax=jump_max)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"({chr(97+i)}) {row['label']}", fontsize=9)
    fig.colorbar(artist, ax=list(axes), fraction=.025, pad=.02, label=r"Traction jump $A_e$")
    save_figure(fig, "stress_cantilever_comparison_traction", dpi=600)
    plt.close(fig)
    # 统一采用本地首次渲染产物, 避免二次保存时布局微调或 PDF 时间戳不同.
    if config.PAPER_FIGURE_DIR.is_dir():
        for filename in ("stress_cantilever_comparison_topologies.png",
                         "stress_cantilever_comparison_history.png",
                         "stress_cantilever_comparison_history.pdf",
                         "stress_cantilever_comparison_traction.png"):
            shutil.copy2(config.FIGURE_DIR / filename, config.PAPER_FIGURE_DIR / filename)


def main():
    """更新已入库三组结果的图件、数值表和来源清单."""
    rows = prepare()
    draw(rows)
    lines = ["表 5.5 三组离散的优化结果与牵引跳量统计。",
             "", "| 离散 | MMA 更新次数 | 体积分数 (%) | 最大约束值 | 最大归一化牵引跳量 |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        s, p = row["summary"], row["payload"]
        j = p["traction_jump"]
        lines.append(f"| {row['label']} | {s['optimization_iterations']} | "
                     f"{100*s['volume_fraction']:.4f} | {p['max_g_accepted']:.6e} | "
                     f"{j['cell_abs_solid_max']:.6e} |")
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLE_DIR / "table5_5.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    target = config.OUTPUT_DIR / SOURCE_CASE / "postprocess" / "comparison_manifest.json"
    target.write_text(json.dumps([r["payload"] for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

