"""图 5.9: 复用已验收的历史与终态数据, 绘制上二下三组合图."""

import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from analysis import provenance
from ._base import paper_rcparams, save_figure
from .stress_cantilever_comparison import LABELS, TAGS, COLORS

SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "analyzer-lfem__order-3__lambda-max-10000.0",
    "analyzer-huzhang__order-2",
    "analyzer-huzhang__order-3",
    "postprocess/principal_stress_comparison/manifest.json",
)


def main():
    """核对数据来源并重新排版, 不执行状态求解或优化."""
    paper_rcparams(base_size=8)
    base = config.OUTPUT_DIR / SOURCE_CASE
    output = base / "postprocess/principal_stress_comparison"
    records = json.loads((output / "manifest.json").read_text())
    rows = []
    for run, label in zip(REQUIRED_RUNS[:3], LABELS):
        directory = base / run
        record = next(r for r in records if r["run"] == run)
        assert record["density_digest"] == provenance.file_digest(directory / "density_final.vtu")
        history = json.loads((directory / "history.json").read_text())["scalar_histories"]
        rows.append(dict(history=history, label=label))
    fig = plt.figure(figsize=(7.2, 5.3), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1])
    upper = outer[0].subgridspec(1, 2)
    lower = outer[1].subgridspec(1, 3)
    axes = [fig.add_subplot(upper[0, j]) for j in range(2)]
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
    axes = [fig.add_subplot(lower[0, j]) for j in range(3)]
    theta = np.linspace(0, 2*np.pi, 500)
    c, s = np.cos(theta), np.sin(theta)
    scale = 1/np.sqrt(c*c-c*s+s*s)
    for i, (tag, label, ax, record) in enumerate(zip(TAGS, LABELS, axes, records)):
        with np.load(output / f"{tag}.npz") as data:
            x, y, vm, pad = [data[k] for k in ("sig1", "sig2", "vm", "pad_mask")]
        keep = ~pad
        assert int(keep.sum()) == record["points"]
        assert np.isclose(vm[keep].max(), record["max_vm"])
        gmax = record["max_g"]
        ax.plot(c*scale, s*scale, color="#d62728", lw=1,
                label=r"$\sigma_{\mathrm{vm}}^{\mathrm{app}}=\bar\sigma$")
        ax.legend(loc="upper left", frameon=False, fontsize=6,
                  handlelength=1.4, borderpad=.2, handletextpad=.4)
        artist = ax.scatter(x[keep], y[keep], c=vm[keep], cmap="jet", vmin=0, vmax=1.01,
                            s=2, linewidths=0, rasterized=True)
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal",
               xlabel=r"$\sigma_1^{\mathrm{app}}/\bar\sigma$",
               title=f"({chr(99+i)}) {label}")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=7)
        ax.grid(ls=":", alpha=.35)
        print(f"[verified] {tag}: points={keep.sum()}, max_vm={vm[keep].max():.8g}, max_g={gmax:.8g}", flush=True)
    axes[0].set_ylabel(r"$\sigma_2^{\mathrm{app}}/\bar\sigma$")
    cax = axes[-1].inset_axes([1.06, 0.0, 0.04, 1.0])
    fig.colorbar(artist, cax=cax, format="%.2f",
                 label=r"$\sigma_{\mathrm{vm}}^{\mathrm{app}}/\bar\sigma$",
                 ticks=[0, .25, .5, .75, 1])
    stem = "stress_cantilever_history_principal"
    save_figure(fig, stem, formats=("png", "pdf"), dpi=600)
    plt.close(fig)
    if config.PAPER_FIGURE_DIR.is_dir():
        for ext in ("png", "pdf"):
            shutil.copy2(config.FIGURE_DIR / f"{stem}.{ext}", config.PAPER_FIGURE_DIR / f"{stem}.{ext}")
