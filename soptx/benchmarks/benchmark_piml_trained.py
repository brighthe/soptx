"""T4b: TrainedPredictor 回填 benchmark (deck 帧 7 证据 ④ 真实预测误差).

在与 benchmark_piml_forward 完全相同的算例 (悬臂, 光滑非均匀密度场,
粗网格 8x8) 上, 用已训练的极小 MLP 度量真实预测误差:

1. 逐子结构 ‖K̂_s − K_s‖_F/‖K_s‖_F (真值 = ExactPredictor 静力缩聚):
   在实际 64 个子结构上取 mean/median/max —— 即 deck ④ 主口径, 与证据
   ①②③ 同场;
2. assembled v1_trained = ‖K^cond_trained − S‖_F/‖S‖_F (vs 全尺度全局
   Schur 补): 与 Mock (~3e-2) / Exact (~1e-15) 同表直接可比.

前置: 先运行 train_piml_predictor 产出 outputs/piml_trained_predictor_L{L}.pt.

运行 (仓库根目录):

    .\\.venv\\Scripts\\python.exe -m soptx.benchmarks.benchmark_piml_trained \
        --coarse 8x8 --levels 5,10 --output outputs/piml_trained_prototype.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from fealpy.backend import backend_manager as bm

from soptx.analysis.multiscale import (
    CoarseFineMeshPair,
    ExactPredictor,
    InterfaceCondensedSystem,
    MockPredictor,
    SubstructureTemplate,
    assemble_fullscale_stiffness,
    fullscale_schur_complement,
    load_predictor,
)
from soptx.benchmarks.benchmark_piml_forward import smooth_rho
from soptx.interpolation.linear_elastic_material import (
    IsotropicLinearElasticMaterial,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


def per_substructure_ks_error(mesh_pair, template, predictor,
                              rho: np.ndarray) -> np.ndarray:
    """逐子结构 ‖K̂_s − K_s‖_F/‖K_s‖_F (真值 = 精确静力缩聚)."""
    errs = np.empty(mesh_pair.n_sub)
    for j in range(mesh_pair.n_sub):
        rho_local = rho[mesh_pair.sub_cells(j)]
        K_true = template.condense(rho_local).K_s
        K_hat = predictor.predict_K_s(rho_local)
        errs[j] = (np.linalg.norm(K_hat - K_true)
                   / np.linalg.norm(K_true))
    return errs


def run_case(ncx: int, ncy: int, L: int, ckpt_dir: Path,
             device: str, q: int = 4, schur_chunk: int = 512) -> dict:
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_stress",
    )
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 2.0, 0.0, 1.0),
                                   ncx=ncx, ncy=ncy, L=L)
    rho = smooth_rho(mesh_pair)
    template = SubstructureTemplate(mesh_pair, material, q=q)

    ckpt = ckpt_dir / f"piml_trained_predictor_L{L}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"缺少训练权重 {ckpt}; 先运行 "
            f"python -m soptx.benchmarks.train_piml_predictor --levels {L}")
    trained = load_predictor(ckpt, template, device=device)

    # -- ④ 主口径: 逐子结构 K_s 预测误差 (实际 64 子结构, 与 ①②③ 同场) --
    errs = per_substructure_ks_error(mesh_pair, template, trained, rho)

    # -- assembled v1: 三种预测器同场对照 (vs 全尺度全局 Schur 补) --
    K = assemble_fullscale_stiffness(mesh_pair, material, rho, q=q)
    S = fullscale_schur_complement(K, mesh_pair.interface_dofs,
                                   chunk=schur_chunk)
    norm_S = np.linalg.norm(S)

    def v1(pred):
        sysm = InterfaceCondensedSystem(mesh_pair, pred, rho)
        return np.linalg.norm(sysm.K_cond.toarray() - S) / norm_S

    v1_exact = v1(ExactPredictor(template))
    v1_mock = v1(MockPredictor(template))
    v1_trained = v1(trained)

    return {
        "coarse": f"{ncx}x{ncy}",
        "L": L,
        "fine": f"{mesh_pair.nx}x{mesh_pair.ny}",
        "n_sub": mesh_pair.n_sub,
        "ks_relF_mean": float(errs.mean()),
        "ks_relF_median": float(np.median(errs)),
        "ks_relF_max": float(errs.max()),
        "v1_rel_error_exact": v1_exact,
        "v1_rel_error_mock": v1_mock,
        "v1_rel_error_trained": v1_trained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", default="8x8")
    parser.add_argument("--levels", default="5,10")
    parser.add_argument("--ckpt-dir", default="outputs")
    parser.add_argument("--output", default="outputs/piml_trained_prototype.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--schur-chunk", type=int, default=512)
    args = parser.parse_args()

    bm.set_backend("numpy")
    ncx, ncy = (int(v) for v in args.coarse.lower().split("x"))
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    ckpt_dir = Path(args.ckpt_dir)

    rows = []
    for L in levels:
        print(f"[run] coarse={ncx}x{ncy}, L={L} ...", flush=True)
        rows.append(run_case(ncx, ncy, L, ckpt_dir, args.device,
                             schur_chunk=args.schur_chunk))

    header = list(rows[0].keys())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== T4b TrainedPredictor 回填 (deck 帧 7 证据 ④) ===")
    for row in rows:
        print(f"\n粗网格 {row['coarse']}, L={row['L']} (细网格 {row['fine']}, "
              f"{row['n_sub']} 子结构):")
        print(f"  ④ 逐子结构 ‖K̂_s−K_s‖_F/‖K_s‖_F: "
              f"mean={row['ks_relF_mean']:.3e} "
              f"median={row['ks_relF_median']:.3e} "
              f"max={row['ks_relF_max']:.3e}")
        print(f"  assembled v1 (vs 全尺度 Schur): "
              f"Trained={row['v1_rel_error_trained']:.3e} | "
              f"Mock={row['v1_rel_error_mock']:.3e} | "
              f"Exact={row['v1_rel_error_exact']:.3e}")
    print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()
