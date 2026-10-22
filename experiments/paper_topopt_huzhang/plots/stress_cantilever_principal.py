"""当前三组悬臂梁设计的表观主应力散点图 (图 5.9)."""

import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from analysis.discretization_probe import evaluate, load_design, pad_mask_from_summary
from analysis import provenance
from soptx.postprocess.stress_report import StressPostProcessor
from ._base import paper_rcparams, save_figure
from .stress_cantilever_comparison import LABELS, TAGS

SOURCE_CASE = "cantilever-middle-2d-stress"
REQUIRED_RUNS = (
    "analyzer-lfem__order-3__lambda-max-10000.0",
    "analyzer-huzhang__order-2",
    "analyzer-huzhang__order-3",
)


def main():
    """冻结再分析三组终态, 核对应力与约束后导出散点图和来源记录."""
    paper_rcparams()
    base = config.OUTPUT_DIR / SOURCE_CASE
    output = base / "postprocess" / "principal_stress_comparison"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), layout="constrained")
    records = []
    theta = np.linspace(0, 2*np.pi, 500)
    c, s = np.cos(theta), np.sin(theta)
    scale = 1/np.sqrt(c*c-c*s+s*s)
    for i, (run, tag, label, ax) in enumerate(zip(REQUIRED_RUNS, TAGS, LABELS, axes)):
        directory = base / run
        rho, summary = load_design(directory)
        assert summary["converged"] and summary["acceptance_solid_threshold"] is None
        assert summary["load_discretization"] == "p1_trace_interpolation"
        method, order = tag.split("-")
        result = evaluate(method, int(order), rho)
        pipeline = result["pipeline"]
        post = StressPostProcessor(pipeline.analyzer, stress_limit=180.0)
        fields = post.check_stress_constraints(pipeline.density, state=result["state"])
        x, y, vm = map(np.asarray, (fields.sig_1_norm, fields.sig_2_norm, fields.SM))
        pad = pad_mask_from_summary(pipeline.mesh, summary, len(rho))
        keep = ~pad
        assert x.shape == y.shape == vm.shape == rho.shape
        # von Mises 平方量下限为 1e-12, 允许归一化后的近零误差.
        assert np.allclose(np.sqrt(x*x-x*y+y*y), vm, rtol=1e-8, atol=1e-6/180.0)
        assert np.allclose(vm, result["apparent_stress_ratio"], rtol=1e-8, atol=1e-10)
        gmax = float(result["constraint_value"][keep].max())
        recorded = summary.get("max_relative_violation_accepted", summary.get("max_relative_violation_solid_region"))
        assert np.isclose(gmax, recorded, rtol=1e-6, atol=1e-9)
        np.savez(output / f"{tag}.npz", sig1=x, sig2=y, vm=vm, pad_mask=pad, density=rho)
        records.append(dict(run=run, density_digest=provenance.file_digest(directory / "density_final.vtu"),
                            points=int(keep.sum()), max_vm=float(vm[keep].max()), max_g=gmax,
                            source=provenance.run_stamp()))
        ax.plot(c*scale, s*scale, color="#d62728", lw=1,
                label=r"$\sigma_{\mathrm{vm}}^{\mathrm{app}}=\bar\sigma$")
        ax.legend(loc="upper left", frameon=False, fontsize=7,
                  handlelength=1.4, borderpad=.2, handletextpad=.4)
        artist = ax.scatter(x[keep], y[keep], c=vm[keep], cmap="jet", vmin=0, vmax=1.01,
                            s=2, linewidths=0, rasterized=True)
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal",
               xlabel=r"$\sigma_1^{\mathrm{app}}/\bar\sigma$",
               title=f"({chr(97+i)}) {label}")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=8)
        ax.grid(ls=":", alpha=.35)
        print(f"[verified] {tag}: points={keep.sum()}, max_vm={vm[keep].max():.8g}, max_g={gmax:.8g}", flush=True)
    axes[0].set_ylabel(r"$\sigma_2^{\mathrm{app}}/\bar\sigma$")
    cax = axes[-1].inset_axes([1.06, 0.0, 0.04, 1.0])
    fig.colorbar(artist, cax=cax, format="%.2f",
                 label=r"$\sigma_{\mathrm{vm}}^{\mathrm{app}}/\bar\sigma$",
                 ticks=[0, .25, .5, .75, 1])
    stem = "stress_cantilever_comparison_principal"
    save_figure(fig, stem, formats=("png", "pdf"), dpi=600)
    plt.close(fig)
    if config.PAPER_FIGURE_DIR.is_dir():
        for ext in ("png", "pdf"):
            shutil.copy2(config.FIGURE_DIR / f"{stem}.{ext}", config.PAPER_FIGURE_DIR / f"{stem}.{ext}")
    (output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

