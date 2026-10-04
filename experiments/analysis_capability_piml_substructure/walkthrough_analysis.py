"""PIML 子结构在线分析阶段: 恢复权重并构造局部缩聚刚度."""

from __future__ import annotations

import argparse
from math import isclose, isfinite, prod
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
# 两路合训的正式训练目录, 同时含 shape_best.pt 与 stiffness_best.pt; 来源见 data_provenance.json.
TRAINING_DIR = Path(
    "/home/brighthe/workspace/data/soptx/piml_substructure/"
    "independent_15_layer/training/20260922T065924289373Z"
)


def parse_args(argv=None):
    """解析在线分析参数; 局部问题配置从训练权重恢复."""
    parser = argparse.ArgumentParser(description="PIML 子结构在线分析走查")
    parser.add_argument(
        "--shape-dir", type=Path, default=TRAINING_DIR,
        help=(
            "含 shape_best.pt 的训练结果目录, 两条路线均需要; "
            "相对路径以脚本目录为基准; 默认 %(default)s"
        ),
    )
    parser.add_argument(
        "--stiffness-dir", type=Path,
        help=(
            "含 stiffness_best.pt 的训练结果目录, 仅用于 --route stiffness; "
            f"相对路径以脚本目录为基准; 默认 {TRAINING_DIR}"
        ),
    )
    parser.add_argument(
        "--domain", type=float, nargs="+",
        default=[0.0, 78.0, 0.0, 13.0, 0.0, 13.0],
        help="整体求解域区间端点 x_min x_max y_min y_max [z_min z_max]",
    )
    parser.add_argument(
        "--n-sub", type=int, nargs="+", default=[78, 13, 13],
        help="各方向子结构数; 维数须与训练权重一致",
    )
    parser.add_argument(
        "--route", choices=["shape", "stiffness"], default="shape",
        help="预测路线; stiffness 同时加载形函数网络用于内部位移恢复",
    )
    parser.add_argument(
        "--E-simp-penalty", dest="E_simp_penalty", type=float, default=3.0,
        help="Young's modulus E 的 SIMP 惩罚指数; 默认 %(default)s",
    )
    parser.add_argument(
        "--local-batch-size", type=int, default=32,
        help="局部网络推理及 shape 路线刚度装配的最大子结构批量; 默认 %(default)s",
    )
    parser.add_argument(
        "--solver", choices=["mumps", "scipy"], default="mumps",
        help="接口系统的直接求解器; 供后续求解使用",
    )
    parser.add_argument("--seed", type=int, default=0, help="整体密度场随机种子")
    parser.add_argument(
        "--mem-limit-gb", type=float, default=35.0,
        help="进程虚拟地址空间上限 (GiB); 默认 35 GiB",
    )
    args = parser.parse_args(argv)

    if args.route == "shape" and args.stiffness_dir is not None:
        parser.error("--stiffness-dir 仅用于 --route stiffness")
    if args.route == "stiffness" and args.stiffness_dir is None:
        args.stiffness_dir = TRAINING_DIR
    for name in ("shape_dir", "stiffness_dir"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser()
            setattr(args, name, path if path.is_absolute() else CURRENT_DIR / path)
    if len(args.n_sub) not in (2, 3) or min(args.n_sub) <= 0:
        parser.error("--n-sub 须为 2 或 3 个正整数")
    domain = args.domain
    if len(domain) not in (4, 6) or not all(isfinite(v) for v in domain):
        parser.error("--domain 须为 4 或 6 个有限数")
    if any(domain[2 * d] >= domain[2 * d + 1] for d in range(len(domain) // 2)):
        parser.error("--domain 各方向须满足下端小于上端")
    if any(domain[2 * d] != 0.0 for d in range(len(domain) // 2)):
        parser.error("--domain 下端须全为 0: 当前整体网格以原点为下角")
    if len(args.n_sub) != len(domain) // 2:
        parser.error("--n-sub 个数须等于 --domain 的空间维数")
    if not isfinite(args.E_simp_penalty) or args.E_simp_penalty <= 0.0:
        parser.error("--E-simp-penalty 须为有限正数")
    if args.local_batch_size <= 0:
        parser.error("--local-batch-size 须为正整数")
    if args.seed < 0:
        parser.error("--seed 不能为负数")
    if not isfinite(args.mem_limit_gb) or args.mem_limit_gb < 1 / 2**30:
        parser.error("--mem-limit-gb 须为有限正数且至少为 1 字节")
    return args


def main(argv=None):
    """恢复局部配置与网络, 分批构造局部缩聚刚度."""
    args = parse_args(argv)
    n_sub = tuple(args.n_sub)
    route = args.route
    required = ("shape",) if route == "shape" else ("shape", "stiffness")
    checkpoints = {
        name: getattr(args, f"{name}_dir") / f"{name}_best.pt" for name in required
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--{name}-dir 缺少必要权重: {checkpoint}")

    from soptx.backend import backend_manager as bm
    from soptx.ml.substructure.independent_checkpoints import (
        decoder_metadata_matches,
        load_independent_network, load_independent_provider_metadata,
    )
    from soptx.ml.substructure.independent_contract import provider_metadata_matches
    from soptx.ml.substructure.inference import predict_independent_outputs

    bm.set_backend("numpy")

    # ------------------------------------------------------------------
    # 已训练模型与参考子结构配置
    # ------------------------------------------------------------------
    # 从形函数权重读取子结构几何尺寸、细网格划分、材料参数、接口空间及独立分量编号;
    # 刚度路线的两份权重可来自不同训练目录, 须对应同一子结构配置.
    metadata = load_independent_provider_metadata(checkpoints["shape"], route="shape")
    if route == "stiffness" and not provider_metadata_matches(
        load_independent_provider_metadata(checkpoints["stiffness"], route="stiffness"),
        metadata,
    ):
        raise ValueError(
            "--shape-dir 与 --stiffness-dir 权重的子结构配置或独立分量编号不一致"
        )

    # 整体尺寸由用户的求解域确定, 权重只约束每个子结构的尺寸.
    domain = tuple(args.domain)
    domain_size = tuple(
        domain[2 * d + 1] - domain[2 * d] for d in range(len(n_sub))
    )
    dim = int(metadata["spatial_dimension"])
    if len(n_sub) != dim:
        raise ValueError(
            f"--domain 和 --n-sub 须与权重的 {dim} 维配置一致; "
            "二维权重需要显式给出二维求解域和两个方向的子结构数"
        )
    cell_size = tuple(metadata["cell_size"])
    if len(cell_size) != dim or any(
        not isclose(domain_size[d] / n_sub[d], cell_size[d],
                    rel_tol=1e-12, abs_tol=0.0)
        for d in range(dim)
    ):
        raise ValueError(
            f"--domain 与 --n-sub 确定的子结构尺寸 "
            f"{tuple(domain_size[d] / n_sub[d] for d in range(dim))} "
            f"与权重 cell_size={cell_size} 不一致"
        )

    # 加载所选路线所需的网络.
    networks = {}
    for name in required:
        networks[name], _ = load_independent_network(
            checkpoints[name],
            metadata,
            route=name,
        )
    print(
        f"[权重] 路线 {route}, trace {metadata['trace']}, "
        f"n_fine {tuple(metadata['n_fine'])}"
    )

    # 限制虚拟地址空间, 不代表物理内存用量; 权重加载完成后设置.
    import resource

    GIB = 2**30
    limit = int(args.mem_limit_gb * GIB)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    import numpy as np
    from soptx.fem.substructure import (
        GlobalAssembler,
        IndependentPredictionDecoder,
        ShapeFunctionCondensation,
        StructuredSubstructureLayout,
        build_modulus_substructures,
    )
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.topology.interpolation import MaterialInterpolationScheme

    # ------------------------------------------------------------------
    # 局部配置: 材料、离散与接口空间
    # ------------------------------------------------------------------
    n_fine = tuple(metadata["n_fine"])
    E_base = float(
        metadata["material_scaling"]["reference_young_modulus"]
    )
    nu = float(metadata["poisson_ratio"])
    hypothesis = "3D" if dim == 3 else metadata["material_hypothesis"]
    trace_kind = metadata["trace"]

    # ------------------------------------------------------------------
    # 子结构划分: 整体布局, 参考子结构, 子结构排列
    # ------------------------------------------------------------------
    # 创建整体布局, 统一管理有限元上下文、材料场切分和局部到全局映射.
    layout = StructuredSubstructureLayout(
        domain_size, n_sub, n_fine, E_base=E_base, nu=nu,
        hypothesis=hypothesis,
    )
    # 装配器组合布局, 只负责后续接口系统与 trace 系统装配.
    assembler = GlobalAssembler(layout)
    # 创建共享参考子结构及其位置实例
    prototype, sub_meshes, positions = build_modulus_substructures(layout)
    # 基于同一参考子结构构造接口空间及独立预测条目的约束补全器.
    decoder = IndependentPredictionDecoder(prototype, trace_kind=trace_kind)
    if not decoder_metadata_matches(metadata, decoder.metadata()):
        raise ValueError(
            "整体布局创建的 decoder 与权重中的局部离散或独立分量编号不一致"
        )

    # ------------------------------------------------------------------
    # 在线材料输入: 密度场, E 的 SIMP 插值, 归一化模量
    # ------------------------------------------------------------------
    density_range = (0.5, 0.9)
    seed = args.seed
    density = bm.asarray(
        np.random.default_rng(seed).uniform(
            *density_range, size=layout.total_fine,
        )
    )
    local_density = layout.split_global_cell_field(density)

    material = IsotropicLinearElasticMaterial(
        youngs_modulus=E_base, poisson_ratio=nu,
        hypothesis=hypothesis,
    )
    interpolation = MaterialInterpolationScheme(
        density_location="element", interpolation_method="simp",
        options={
            "penalty_factor": args.E_simp_penalty,
            "void_youngs_modulus": 0.0,
            "target_variables": ["E"],
        },
        enable_logging=False,
    )
    young_modulus = interpolation.interpolate_material(
        material=material, rho_val=bm.reshape(density, (-1,)),
    )
    modulus = young_modulus / E_base
    local_modulus = layout.split_global_cell_field(modulus)
    network_inputs = np.asarray(
        bm.to_numpy(prototype.to_cell_density(local_modulus)),
        dtype=np.float64,
    )
    expected_input_shape = (prod(n_sub), int(metadata["n_cells"]))
    if network_inputs.shape != expected_input_shape:
        raise ValueError(
            f"网络输入形状应为 {expected_input_shape}; "
            f"当前为 {network_inputs.shape}"
        )
    if not np.all(np.isfinite(network_inputs)):
        raise ValueError("归一化杨氏模量包含非有限值.")
    if not np.all((network_inputs > 0.0) & (network_inputs <= 1.0)):
        raise ValueError("归一化杨氏模量必须严格大于 0 且不超过 1.")

    # ------------------------------------------------------------------
    # 基于网络预测构造局部缩聚刚度矩阵
    # ------------------------------------------------------------------
    n_trace = int(metadata["n_trace"])
    local_condensed_stiffness = np.empty(
        (network_inputs.shape[0], n_trace, n_trace), dtype=np.float64,
    )
    codec = decoder.codecs[route]
    if route == "shape":
        rigid, deformation, rigid_interior = (
            prototype.trace_interface_bases(decoder.trace)
        )
        builder = ShapeFunctionCondensation(
            prototype.i_dofs,
            prototype.b_dofs,
            rigid_basis=rigid,
            deformation_basis=deformation,
            rigid_interior=rigid_interior,
            trace=decoder.trace,
        )

    for start in range(0, network_inputs.shape[0], args.local_batch_size):
        end = min(start + args.local_batch_size, network_inputs.shape[0])
        prediction = predict_independent_outputs(
            networks[route], network_inputs[start:end], route,
        )
        decoded = np.asarray(codec.decode(prediction), dtype=np.float64)

        if route == "shape":
            # 只保留当前批次的细网格刚度与内部延拓, 投影后立即释放.
            local_stiffness = prototype.assemble_local_stiffness_batch(
                network_inputs[start:end],
                chunk_size=args.local_batch_size,
            )
            condensed = builder.assemble_reduced_stiffness(
                local_stiffness,
                bm.asarray(decoded, dtype=bm.float64),
            )
            del local_stiffness
        else:
            # 直接刚度路线由独立条目直接补全 K_s^j, 不装配细网格 K^j.
            condensed = decoded

        expected_batch_shape = (end - start, n_trace, n_trace)
        if condensed.shape != expected_batch_shape:
            raise ValueError(
                f"{route} 路线缩聚刚度形状应为 {expected_batch_shape}; "
                f"当前为 {condensed.shape}"
            )
        if not np.all(np.isfinite(condensed)):
            raise ValueError(f"{route} 路线缩聚刚度包含非有限值.")
        local_condensed_stiffness[start:end] = condensed
        del prediction, decoded, condensed

    print(
        f"[局部预测] 路线 {route}, local_density {tuple(local_density.shape)}, "
        f"network_inputs {tuple(network_inputs.shape)}, "
        f"K_s {tuple(local_condensed_stiffness.shape)}"
    )

    # 当前可执行流程止于 local_condensed_stiffness. 启用全局求解前还须分批检查
    # 预测刚度在变形子空间上的正定性; 以下阶段仍为未启用草稿.

    # # ------------------------------------------------------------------
    # # 接口装配: 沿用精确走查的接口空间与局部到全局编号
    # # ------------------------------------------------------------------
    # if trace_kind == "full_trace":
    #     system = assembler.assemble_interface_system(
    #         sub_meshes, _StiffnessView(local_condensed_stiffness),
    #     )
    #     if not np.array_equal(
    #         bm.to_numpy(system.global_dofs), bm.to_numpy(interface_dofs),
    #     ):
    #         raise ValueError("full_trace 接口系统与完整接口自由度编号不一致.")
    # else:
    #     system = assembler.assemble_macro_system(
    #         sub_meshes, local_condensed_stiffness,
    #     )
    # print(f"[接口装配] 系统 {system.stiffness.shape}", flush=True)

    # # ------------------------------------------------------------------
    # # 边界处理与求解
    # # ------------------------------------------------------------------
    # # 当前仅支持齐次内部恢复, 不缩聚内部载荷.
    # # 载荷与支承必须落在接口自由度上; 与精确走查复用同一投影函数.
    # conditions = project_problem_conditions_to_interface_system(
    #     pde, assembler, interface_view,
    # )
    # macro_force = projection.T @ conditions.interface_force
    # constraints = projection[conditions.interface_fixed_dofs]
    # solved = solve_constrained_system(
    #     system, macro_force, constraints, solver=solver,
    # )
    # displacement = np.asarray(
    #     bm.to_numpy(solved.displacement), dtype=np.float64,
    # )
    # boundary = np.asarray(projection @ displacement, dtype=np.float64)
    # if trace_kind == "full_trace":
    #     local_indices = assembler.interface_indices(sub_meshes, interface_dofs)
    # else:
    #     local_indices = assembler.macro_corner_indices(sub_meshes)
    # residual = float(solved.equilibrium_relative_residual)
    # constraint_residual = float(solved.constraint_relative_residual)
    # if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(boundary)):
    #     raise ValueError("接口求解产生非有限位移.")
    # solution = {
    #     "trace": displacement, "boundary": boundary,
    #     "local_indices": np.asarray(
    #         bm.to_numpy(local_indices), dtype=np.int64,
    #     ),
    #     "equilibrium_relative_residual": residual,
    #     "constraint_relative_residual": constraint_residual,
    # }
    # print(
    #     f"[求解] 平衡相对残差 {residual:.3e}, "
    #     f"约束相对残差 {constraint_residual:.3e}"
    # )

    # # ------------------------------------------------------------------
    # # 完整位移恢复: 两条路线均使用预测的内部延拓矩阵
    # # ------------------------------------------------------------------
    # # stiffness 路线的刚度与恢复矩阵分别预测, 不保证变分一致性.
    # # 为避免保存 full_trace 下的大型 shape_recovery, 应按 local_batch_size
    # # 重新分批预测 shape、补全内部延拓并立即恢复当前批次位移.
    # internal, full = _recover_in_batches(
    #     networks["shape"], decoder, network_inputs, solution,
    #     batch_size=args.local_batch_size,
    # )
    # compliance = float(np.dot(
    #     np.asarray(bm.to_numpy(conditions.full_force), dtype=np.float64), full,
    # ))
    # print(f"整体位移形状: {full.shape}; 内部位移形状: {internal.shape}")
    # print(f"路线: {route}; 柔顺度: {compliance:.12g}")


if __name__ == "__main__":
    main()
