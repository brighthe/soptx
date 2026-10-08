"""验证 shape 路线的完整柔顺度导数与冻结形函数的近似灵敏度.

Notes
-----
复用已保存的光滑密度场, 分别视为物理密度和待过滤的设计密度.
共用 linear_corner 接口, 单位载荷, end_lines 支承和 SIMP 插值.
全程禁用均匀子结构替代, 不执行优化或更新网络权重.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from soptx.backend import backend_manager as bm
from soptx.fem.substructure import (
    GlobalAssembler, StructuredSubstructureLayout, build_interface_space,
    build_modulus_substructures, solve_constrained_system,
)
from soptx.ml.substructure.independent_checkpoints import (
    decoder_metadata_matches, load_analysis_provider, load_independent_network,
)
from soptx.fem.substructure.independent_targets import IndependentPredictionDecoder
from soptx.problems.elasticity import FullMBBBeam3d
from soptx.topology.filters.structured import (
    apply_structured_density_filter, apply_structured_density_filter_adjoint,
)

ROOT = Path.home() / "codespace/data/soptx/piml_substructure/independent_15_layer"
NAME = "mbb_linear_corner_m5_h1_nu0p3_emin1e-7_seed2026_trainseed2026"
CASE = "mbb_linear_corner_12x2x2_smooth_rho0p04_0p20_force_network"


def build_evaluator(previous, provider, network, uniform_threshold=None):
    """构建共用的精确与 shape 路线分析器.

    Parameters
    ----------
    previous : dict
        提供域, 网格, 材料及载荷配置的记录.
    provider : IndependentTargetProvider
        与权重相容的参考子结构与标签提供器.
    network : torch.nn.Module
        已加载的形函数网络.
    uniform_threshold : float or None
        None 禁用均匀替代; 指定时按局部最大与平均密度之差判别,
        使用实体形函数和平均密度对应模量缩放的实体刚度.

    Returns
    -------
    callable
        evaluate(physical, gradients=False) 返回柔顺度及可选梯度.

    Notes
    -----
    均匀判别是分段选择; 完整导数在当前分支内计算, 不跨阈值求导.
    冻结形函数采用各细单元应变能公式, 不对均匀分支执行平均化导数替代.
    """
    metadata = provider.metadata()
    layout = StructuredSubstructureLayout(
        domain_size=tuple(previous["domain"]), n_sub=tuple(previous["n_sub"]),
        n_fine=tuple(metadata["n_fine"]), E_base=1.,
        nu=metadata["poisson_ratio"], hypothesis="3D",
    )
    proto, meshes, _ = build_modulus_substructures(layout, integration_order=2)
    decoder = IndependentPredictionDecoder(proto, trace_kind="linear_corner")
    assert decoder_metadata_matches(metadata, decoder.metadata())
    space = build_interface_space("linear_corner", GlobalAssembler(layout), meshes, proto)
    psi = np.asarray(provider.trace.matrix)
    problem = FullMBBBeam3d(
        domain=tuple(v for d in previous["domain"] for v in (0, d)),
        P=-1., E=1., nu=metadata["poisson_ratio"], support="end_lines",
        load_subdivisions=(previous["fine_grid"][0], previous["fine_grid"][2]),
    )
    load, constraints = space.constrained_conditions(problem)
    indices = np.asarray(space.local_dofs)
    ke = torch.from_numpy(np.asarray(proto.KE_unit))
    cell_dofs = torch.from_numpy(np.asarray(proto.cell2dof))
    emin, penal = previous["min_modulus"], previous["penal"]
    solid = provider.exact_matrices(np.ones((1, proto.n_cells))) if uniform_threshold is not None else None

    def to_cells(field):
        """按参考单元排序提取局部场.

        Parameters
        ----------
        field : numpy.ndarray
            全局结构化单元场.

        Returns
        -------
        numpy.ndarray
            按子结构和局部有限元单元编号排列的场.
        """
        return np.stack([proto.grid_to_cell_field(b)
                         for b in layout.split_global_cell_field(field)])

    def to_grid(cells):
        """将局部单元场还原到全局结构化网格.

        Parameters
        ----------
        cells : numpy.ndarray
            按局部有限元单元编号排列的场.

        Returns
        -------
        numpy.ndarray
            全局结构化单元场.
        """
        return np.asarray(layout.merge_substructure_cell_field(
            np.stack([proto.cell_to_grid_field(b) for b in cells])))

    def evaluate(physical, gradients=False):
        """共用全局约束求解两条路线, 可选计算三种物理密度梯度.

        Parameters
        ----------
        physical : numpy.ndarray
            固定物理密度场.
        gradients : bool
            是否返回精确, 网络完整及冻结形函数的物理密度梯度.

        Returns
        -------
        dict or tuple
            两条路线的柔顺度; gradients 为 True 时同时返回梯度与求解诊断.
        """
        rho = to_cells(physical)
        modulus = emin + (1-emin) * rho ** penal
        exact = provider.exact_matrices(modulus)
        uniform = (rho.max(axis=1)-rho.mean(axis=1) < uniform_threshold
                   if uniform_threshold is not None else np.zeros(len(rho), dtype=bool))
        with torch.no_grad():
            tp = provider.shape_codec.decode(network(torch.from_numpy(modulus))).numpy()
        if uniform.any():
            tp[uniform] = solid["shape"][0]
        basis = np.empty((len(meshes), proto.n_total_dofs, psi.shape[1]))
        basis[:, proto.b_dofs] = psi
        basis[:, proto.i_dofs] = tp
        kp = basis.swapaxes(1, 2) @ exact["local_stiffness"] @ basis
        if uniform.any():
            mean_modulus = emin + (1-emin)*rho[uniform].mean(axis=1)**penal
            kp[uniform] = mean_modulus[:, None, None]*solid["stiffness"][0]
        values, solutions, diagnostics = {}, {}, {}
        for route, stiffness in (("exact", exact["stiffness"]), ("shape", kp)):
            system = space.assemble([SimpleNamespace(start=0, end=len(meshes), stiffness=stiffness)])
            solved = solve_constrained_system(system, load, constraints, solver="scipy")
            q = np.asarray(solved.displacement)
            values[route] = float(load @ q)
            solutions[route] = q[indices]
            diagnostics[route] = {
                "equilibrium_residual": solved.equilibrium_relative_residual,
                "constraint_residual": solved.constraint_relative_residual,
                "uniform_substructure_count": int(uniform.sum()) if route == "shape" else 0,
            }
        if not gradients:
            return values
        fields = {}
        for route in ("exact", "shape"):
            q = torch.from_numpy(solutions[route])
            rt = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
            effective_rt = (torch.where(torch.from_numpy(uniform)[:, None],
                            rt.mean(dim=1, keepdim=True).expand_as(rt), rt)
                            if route == "shape" else rt)
            et = emin + (1-emin) * effective_rt ** penal
            t = (provider.shape_codec.decode(network(et)) if route == "shape"
                 else torch.from_numpy(exact["shape"]))
            if route == "shape" and uniform.any():
                t = torch.where(torch.from_numpy(uniform)[:, None, None],
                                torch.from_numpy(solid["shape"][0])[None], t)
            u = torch.zeros((len(meshes), proto.n_total_dofs), dtype=torch.float64)
            u = u.index_copy(1, torch.from_numpy(np.asarray(proto.b_dofs)), q @ torch.from_numpy(psi.T))
            u = u.index_copy(1, torch.from_numpy(np.asarray(proto.i_dofs)),
                             torch.einsum("bij,bj->bi", t, q))
            ue = u[:, cell_dofs]
            energy = torch.einsum("bei,eij,bej->be", ue, ke, ue)
            total = (et * energy).sum()
            if not np.isclose(total.item(), values[route], rtol=1e-10, atol=1e-9):
                raise ValueError("单元能量与接口柔顺度不一致")
            full = -torch.autograd.grad(total, rt)[0].numpy()
            fields[route if route == "exact" else "shape_full"] = to_grid(full)
            if route == "shape":
                frozen = -penal * (1-emin) * rho ** (penal-1) * energy.detach().numpy()
                fields["shape_frozen"] = to_grid(frozen)
        return values, fields, diagnostics

    return evaluate


def main():
    """运行 CPU 方向导数验证并保存梯度和逐项有限差分记录.

    Returns
    -------
    None
        结果写入不存在的新目录, 不覆盖历史输出.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖: {args.output_dir}")
    bm.set_backend("numpy")
    source = ROOT / "structure_validation" / CASE
    previous = json.loads((source / "run_config.json").read_text())
    density = np.load(source / "physical_density.npy", allow_pickle=False)
    provider = load_analysis_provider(ROOT / "training/shape" / NAME)
    metadata = provider.metadata()
    assert metadata == previous["provider"]
    network, checkpoint = load_independent_network(
        ROOT / "training/shape" / NAME / "shape_best.pt", metadata, route="shape",
    )
    network.requires_grad_(False)
    evaluate = build_evaluator(previous, provider, network)

    results = {}
    artifacts = {}
    epsilons = (1e-4, 1e-5, 1e-6)
    for filtered in (False, True):
        mode = "density_filter_r3" if filtered else "physical_density"
        def forward(design):
            return apply_structured_density_filter(design, 3., (1., 1., 1.)) if filtered else design
        physical = forward(density)
        values, physical_gradients, diagnostics = evaluate(physical, True)
        gradients = {key: apply_structured_density_filter_adjoint(g, 3., (1., 1., 1.))
                     if filtered else g for key, g in physical_gradients.items()}
        ge, gf, gz = (gradients[k] for k in ("exact", "shape_full", "shape_frozen"))
        comparisons = {}
        for key, g in (("shape_full", gf), ("shape_frozen", gz)):
            comparisons[key] = {
                "relative_error_to_exact": float(np.linalg.norm(g-ge)/np.linalg.norm(ge)),
                "cosine_to_exact": float(np.vdot(g,ge)/np.linalg.norm(g)/np.linalg.norm(ge)),
                "positive_derivative_count": int(np.count_nonzero(g>0)),
                "count": g.size,
            }
        comparisons["full_vs_frozen_relative_difference"] = float(np.linalg.norm(gf-gz)/np.linalg.norm(gf))
        directions = {"uniform": np.ones_like(density),
                      "random_seed2029": np.random.default_rng(2029).uniform(-1.,1.,density.shape)}
        one = np.zeros_like(density)
        one.flat[np.argmax(np.abs(ge))] = 1.
        directions["largest_exact_derivative_cell"] = one
        records = []
        for label, direction in directions.items():
            analytic = {key: float(np.sum(g*direction)) for key,g in gradients.items()}
            for epsilon in epsilons:
                plus = evaluate(forward(density+epsilon*direction))
                minus = evaluate(forward(density-epsilon*direction))
                finite = {k: (plus[k]-minus[k])/(2*epsilon) for k in plus}
                errors = {}
                for key, route in (("exact","exact"),("shape_full","shape"),("shape_frozen","shape")):
                    errors[key] = abs(analytic[key]-finite[route])/max(abs(analytic[key]),abs(finite[route]),1e-12)
                records.append({"direction":label,"epsilon":epsilon,
                                "analytical":analytic,"finite_difference":finite,"relative_errors":errors})
        representative = [r for r in records if r["epsilon"]==1e-5]
        checks = {key: max(r["relative_errors"][key] for r in representative)
                  for key in ("exact","shape_full","shape_frozen")}
        results[mode] = {
            "compliance": values, "solve_diagnostics": diagnostics,
            "gradient_comparison": comparisons, "finite_difference_max_relative_error_at_1e-5":checks,
            "finite_difference_records":records,
            "physical_density_range":[float(physical.min()),float(physical.max())],
        }
        artifacts[mode] = (physical,gradients)
        print(json.dumps({"mode":mode,"compliance":values,"gradient_comparison":comparisons,
                          "finite_difference_max_relative_error_at_1e-5":checks},indent=2),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for mode,(physical,gradients) in artifacts.items():
        np.save(args.output_dir/f"{mode}_physical_density.npy",physical)
        for key,g in gradients.items():
            np.save(args.output_dir/f"{mode}_{key}_gradient.npy",g)
    config = {"checkpoint":checkpoint,"input_source":str(source),"fine_grid":density.shape,
              "density_role":"physical in unfiltered case, design in filtered case",
              "filter_radius":3.,"spacing":[1.,1.,1.],"min_modulus":emin,"penal":penal,
              "epsilons":epsilons,"device":"cpu","uniform_shortcut":"disabled",
              "formula":"dC=-Q^T(dK_Q)Q; full includes dT/d(rho); frozen excludes dT/d(rho)"}
    for name,data in (("config",config),("summary",results)):
        (args.output_dir/f"{name}.json").write_text(
            json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"[完成] {args.output_dir}",flush=True)


if __name__ == "__main__":
    main()
