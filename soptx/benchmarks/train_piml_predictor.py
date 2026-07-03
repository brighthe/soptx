"""T4b: 训练极小 TrainedPredictor (回填 deck 帧 7 证据 ④ 真实预测误差).

对每个粗/细比 L 训练一个极小 MLP: rho_local ~ U[0.3, 0.9]^m 采样,
SubstructureTemplate.condense 精确缩聚标注 K_s (取上三角 vech), 损失为
标准化 vech(K_s) 的 MSE; 验收指标为验证集逐样本 ‖K̂_s − K_s‖_F/‖K_s‖_F
(与 deck ④ 同口径). 权重存 outputs/piml_trained_predictor_L{L}.pt
(outputs/ 默认 gitignore, 重跑即得).

运行 (仓库根目录):

    .\\.venv\\Scripts\\python.exe -m soptx.benchmarks.train_piml_predictor \
        --coarse 8x8 --levels 5,10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 控制台若为 GBK, 组合符号 K̂ (̂) 会触发 UnicodeEncodeError; 统一 UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from fealpy.backend import backend_manager as bm

from soptx.analysis.multiscale import (
    CoarseFineMeshPair,
    SubstructureTemplate,
    save_predictor,
    train_ks_mlp,
)
from soptx.interpolation.linear_elastic_material import (
    IsotropicLinearElasticMaterial,
)


def train_case(ncx: int, ncy: int, L: int, *, out_dir: Path,
               n_train: int, n_val: int, epochs: int, hidden: int,
               depth: int, lr: float, seed: int, device: str,
               lo: float, hi: float) -> dict:
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_stress",
    )
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 2.0, 0.0, 1.0),
                                   ncx=ncx, ncy=ncy, L=L)
    template = SubstructureTemplate(mesh_pair, material, q=4)

    m = template.local_c2d.shape[0]
    print(f"[train] coarse={ncx}x{ncy}, L={L}: m={m}, n_b={template.n_b}, "
          f"out_dim={template.n_b * (template.n_b + 1) // 2}", flush=True)

    t0 = time.perf_counter()
    result = train_ks_mlp(
        template, n_train=n_train, n_val=n_val, lo=lo, hi=hi,
        hidden=hidden, depth=depth, epochs=epochs, lr=lr,
        seed=seed, device=device, verbose=True,
    )
    dt = time.perf_counter() - t0

    out_path = out_dir / f"piml_trained_predictor_L{L}.pt"
    save_predictor(out_path, result)

    print(f"[train] L={L} done in {dt:.1f}s | "
          f"val relF: mean={result['val_rel_frob_mean']:.3e} "
          f"median={result['val_rel_frob_median']:.3e} "
          f"max={result['val_rel_frob_max']:.3e} -> {out_path}", flush=True)
    return {
        "L": L, "val_rel_frob_mean": result["val_rel_frob_mean"],
        "val_rel_frob_median": result["val_rel_frob_median"],
        "val_rel_frob_max": result["val_rel_frob_max"],
        "train_s": dt, "path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", default="8x8")
    parser.add_argument("--levels", default="5,10")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--n-val", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rho-lo", type=float, default=0.3)
    parser.add_argument("--rho-hi", type=float, default=0.9)
    args = parser.parse_args()

    bm.set_backend("numpy")
    ncx, ncy = (int(v) for v in args.coarse.lower().split("x"))
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for L in levels:
        rows.append(train_case(
            ncx, ncy, L, out_dir=out_dir, n_train=args.n_train,
            n_val=args.n_val, epochs=args.epochs, hidden=args.hidden,
            depth=args.depth, lr=args.lr, seed=args.seed,
            device=args.device, lo=args.rho_lo, hi=args.rho_hi,
        ))

    print("\n=== T4b TrainedPredictor 训练汇总 (验证集 ‖K̂_s−K_s‖_F/‖K_s‖_F) ===")
    for r in rows:
        print(f"  L={r['L']:2d}: mean={r['val_rel_frob_mean']:.3e} "
              f"median={r['val_rel_frob_median']:.3e} "
              f"max={r['val_rel_frob_max']:.3e}  ({r['train_s']:.1f}s)")


if __name__ == "__main__":
    main()
