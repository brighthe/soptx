"""对三种灵敏度执行三维 MBB 的短程 OC 优化对照.

Notes
-----
精确子结构, shape 完整导数与冻结形函数导数共用所有模型与 OC 参数.
每个状态均经精确子结构重分析; 此入口仅验证短程更新, 不声明优化收敛.
"""

import argparse
import json
from time import perf_counter

import numpy as np
from soptx.backend import backend_manager as bm
from soptx.ml.substructure.independent_checkpoints import load_analysis_provider, load_independent_network
from soptx.topology.filters.structured import (
    apply_structured_density_filter, apply_structured_density_filter_adjoint,
)
from soptx.topology.optimizers import OCOptimizer
from validate_sensitivity import ROOT, NAME, CASE, build_evaluator


def main():
    """保存三条路线的密度快照, 历史与精确重分析对照.

    Returns
    -------
    None
        验证结果写入新目录, 不更新权重.
    """
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--routes", nargs="+", choices=("exact","shape_full","shape_frozen"),
                        default=("exact","shape_full","shape_frozen"))
    parser.add_argument("--reference-dir", type=Path,
                        help="只运行部分路线时, 从此处复用相同配置的其余路线结果")
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("updates 必须为正整数")
    if len(set(args.routes)) != len(args.routes):
        parser.error("routes 不能重复")
    if len(args.routes)<3 and args.reference_dir is None:
        parser.error("部分路线运行须提供 reference-dir")
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖: {args.output_dir}")
    bm.set_backend("numpy")
    previous = json.loads((ROOT / "structure_validation" / CASE / "run_config.json").read_text())
    provider = load_analysis_provider(ROOT / "training/shape" / NAME)
    network, source = load_independent_network(
        ROOT / "training/shape" / NAME / "shape_best.pt", provider.metadata(), route="shape",
    )
    network.requires_grad_(False)
    evaluator = build_evaluator(previous, provider, network, uniform_threshold=1e-4)
    grid = tuple(previous["fine_grid"])
    weights = apply_structured_density_filter_adjoint(np.ones(grid), 3., (1., 1., 1.))
    options = dict(move_limit=.2, damping_coef=.5, initial_lambda=1e9,
                   bisection_tol=1e-3, design_variable_min=0.)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"checkpoint": source, "model_config": previous,
              "updates": args.updates, "initial_density":.12, "volume_fraction":.12,
              "filter_radius":3., "spacing":[1.,1.,1.], "oc":options,
              "uniform_threshold":1e-4, "routes":["exact","shape_full","shape_frozen"],
              "backend":"numpy", "inference_backend":"pytorch", "device":"cpu",
              "termination":"fixed_number_of_updates", "reference":"same_linear_corner_space"}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2,allow_nan=False), encoding="utf-8")
    config["executed_routes"] = list(args.routes)
    config["reused_reference_dir"] = str(args.reference_dir) if args.reference_dir else None
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2,allow_nan=False), encoding="utf-8")
    finals, histories = {}, {}
    if args.reference_dir is not None:
        previous_run = json.loads((args.reference_dir/"config.json").read_text())
        for key in ("updates","model_config","initial_density","volume_fraction","oc","uniform_threshold"):
            if previous_run[key] != config[key]:
                raise ValueError(f"参考运行参数不同: {key}")
        if previous_run["checkpoint"]["sha256"] != source["sha256"]:
            raise ValueError("参考运行权重不同")
        for route in config["routes"]:
            if route not in args.routes:
                finals[route] = np.load(args.reference_dir/route/"physical_density_final.npy")
                histories[route] = json.loads((args.reference_dir/route/"history.json").read_text())
    started = perf_counter()
    for route in args.routes:
        destination = args.output_dir / route
        destination.mkdir()
        design = np.full(grid,.12)
        history = []
        for iteration in range(args.updates+1):
            physical = apply_structured_density_filter(design,3.,(1.,1.,1.))
            values, fields, diagnostics = evaluator(physical,True)
            gradient = apply_structured_density_filter_adjoint(fields[route],3.,(1.,1.,1.))
            exact_gradient = apply_structured_density_filter_adjoint(fields["exact"],3.,(1.,1.,1.))
            analysis_route = "exact" if route == "exact" else "shape"
            record = {
                "oc_updates_completed":iteration,
                "compliance":values[analysis_route], "exact_reanalysis_compliance":values["exact"],
                "analysis_relative_error":abs(values[analysis_route]/values["exact"]-1),
                "physical_volume_fraction":float(physical.mean()),
                "gradient_relative_error_to_exact":float(np.linalg.norm(gradient-exact_gradient)/np.linalg.norm(exact_gradient)),
                "gradient_cosine_to_exact":float(np.vdot(gradient,exact_gradient)/np.linalg.norm(gradient)/np.linalg.norm(exact_gradient)),
                "positive_gradient_count":int(np.count_nonzero(gradient>0)),
                "uniform_substructure_count":diagnostics["shape"]["uniform_substructure_count"] if route != "exact" else 0,
                "equilibrium_residual":diagnostics[analysis_route]["equilibrium_residual"],
                "constraint_residual":diagnostics[analysis_route]["constraint_residual"],
                "symmetry_error_z":float(np.max(np.abs(physical-physical[:,:,::-1]))),
            }
            np.save(destination/f"physical_density_{iteration:02d}.npy",physical)
            history.append(record)
            print(f"[{route}] updates={iteration}: C={record['compliance']:.6f}, "
                  f"C_exact={values['exact']:.6f}, vol={physical.mean():.8f}, "
                  f"positive_grad={record['positive_gradient_count']}",flush=True)
            if iteration < args.updates:
                updated = np.asarray(OCOptimizer.update_design_variable(
                    design_variable=design,objective_gradient=gradient,constraint_gradient=weights,
                    constraint_function=lambda candidate: np.mean(weights*candidate)-.12,**options))
                if not np.isfinite(updated).all() or updated.min()<0 or updated.max()>1:
                    raise FloatingPointError("OC 返回无效设计密度")
                next_volume=float(apply_structured_density_filter(updated,3.,(1.,1.,1.)).mean())
                if next_volume > .120001:
                    raise RuntimeError("OC 更新违反物理体积约束")
                record["maximum_design_change"] = float(np.max(np.abs(updated-design)))
                record["updated_volume_fraction"] = next_volume
                design=updated.copy()
            (destination/"history.json").write_text(json.dumps(history,indent=2,allow_nan=False),encoding="utf-8")
        np.save(destination/"design_density_final.npy",design)
        np.save(destination/"physical_density_final.npy",physical)
        finals[route] = physical.copy()
        histories[route] = history
    reference = finals["exact"]
    summary = {"status":"COMPLETED","accuracy_acceptance":"not_evaluated",
               "updates":args.updates,"converged":False,"elapsed_seconds":perf_counter()-started,"routes":{}}
    baseline=histories["exact"][-1]["exact_reanalysis_compliance"]
    for route in config["routes"]:
        h=histories[route]
        summary["routes"][route]={
            "final":h[-1], "initial_compliance":h[0]["compliance"],
            "exact_performance_ratio_to_baseline":h[-1]["exact_reanalysis_compliance"]/baseline,
            "physical_density_relative_difference_to_baseline":float(np.linalg.norm(finals[route]-reference)/np.linalg.norm(reference)),
            "physical_density_mean_absolute_difference_to_baseline":float(np.mean(np.abs(finals[route]-reference))),
            "maximum_volume_fraction":max(a["physical_volume_fraction"] for a in h),
            "maximum_positive_gradient_count":max(a["positive_gradient_count"] for a in h),
            "maximum_equilibrium_residual":max(a["equilibrium_residual"] for a in h),
        }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    print(f"[完成] {args.output_dir}",flush=True)


if __name__ == "__main__":
    main()
