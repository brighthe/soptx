"""PIML 近似 Matrix-Free vs 精确 Matrix-Free: 直接对比 + 误差归因 + SPD 前提.

核心对比: 同一个无预条件 CG、同一个 ``InterfaceOperator`` 容器, 唯一变量是把
精确缩聚刚度 ``K_s`` 换成 PIML 预测的 ``K_s_hat``. 两次求解直接对比给出本脚本
的核心结果——PIML 近似 Matrix-Free 的解误差与迭代代价.

对比本身无法归因: 只看「PIML-MF vs 精确-MF 差百分之几」分不清误差来自代理预测
还是算子路径对 PIML 输入的潜在缺陷. 因此附两层证据:

- **归因**: 同一批 ``K_s_hat`` 走显式装配路径求解, 与算子路径解互差应在 1e-13
  量级 (单次作用恒等阈值相对 1e-13)——证明误差 100% 来自预测, Matrix-Free 化
  零贡献;
- **前提 (SPD 证书)**: 零回退 (任何回退都会把预测换成精确 Schur 补, 使对比失去
  意义)、自由子空间算子最小特征值 > 0 (稠密特征值仅验证规模可行)、CG 收敛且无
  breakdown——PIML 预测算子确实能进 CG.

本脚本**不验证**的东西 (边界): 预测精度本身归 ``piml_substructure_elasticity/
verify_stiffness_route.py``; 精确 ``K_s`` 上的算子正确性 (七项算子级判据 + 裸
CG 端到端) 归 ``matrix_free_substructure_elasticity/verify_matrix_free_ea.py``.

门禁判据 (任一失败退出码 1, 且对比结果不可引用): 零回退、自由子空间最小特征值
> 0、CG 全部收敛且无 breakdown、算子作用与显式装配恒等 (< 1e-13)、对称性
(< 1e-13)、两条路径真残差 (≤ 10·rtol)、迭代数之差 (≤ 1)、解互差 (≤ 1e-8).
核心对比量 (误差、迭代数增量) 不设阈值——其大小由训练质量决定, 由
``verify_stiffness_route.py`` 一侧负责评价.

使用方法:
    python examples/piml_matrix_free_substructure_elasticity/verify_piml_matrix_free.py
    python examples/piml_matrix_free_substructure_elasticity/verify_piml_matrix_free.py --backend pytorch
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np

from fealpy.backend import backend_manager as bm
from soptx.solvers import CGSolver

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    InterfaceOperator,
    PIMLStaticCondensation,
    build_substructures,
    make_density_fields,
    set_random_seed,
    train_surrogate,
)


# PyTorch 首次构造 CSR 张量时的 beta 状态警告, 无 Python API 可关闭, 按消息原文
# 精确屏蔽; 理由与 matrix_free_substructure_elasticity/verify_matrix_free_ea.py 相同.
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)


# 恒等判据阈值: 同一批 K_s_hat 上算子与显式装配代数恒等, 差异只来自浮点求和次序.
OPERATOR_TOLERANCE = 1e-13

# CG 停机参数, 与 matrix_free_substructure_elasticity 的精确 K_s 验证一致.
CG_RTOL = 1e-10
CG_MAXITER = 20000

# 真残差判据相对 rtol 的松弛倍数.
RESIDUAL_SLACK = 10.0

# 算子/显式两条路径 CG 迭代数允许的最大差值.
NITER_TOLERANCE = 1

# 算子/显式两条路径解相互一致的阈值.
SOLUTION_MUTUAL_TOL = 1e-8

# 算例与训练配置, 与 piml_substructure_elasticity/deployment_config.py 及
# verify_stiffness_route.py 的默认值保持一致, 使各目录证据可对照.
DOMAIN = (0.0, 12.0, 0.0, 2.0)
N_SUB = (12, 2)
N_FINE = (5, 5)
DENSITY_RANGE = (0.3, 1.0)
N_TRAIN = 2000
N_EPOCHS = 4000
LEARNING_RATE = 0.005


def relative_error(actual: Any, expected: Any) -> float:
    """两个向量的相对 2-范数误差; 参照量为零时退化为绝对误差."""
    diff = float(bm.sqrt(bm.sum((actual - expected) ** 2)))
    scale = float(bm.sqrt(bm.sum(expected ** 2)))
    return diff / scale if scale > 0.0 else diff


def random_vectors(rng: np.random.Generator, n_vector: int, length: int) -> List[Any]:
    """生成一组后端无关的随机向量 (NumPy 生成后转后端, 两后端输入逐位相同)."""
    samples = rng.standard_normal((n_vector, length))
    return [bm.asarray(samples[k], dtype=bm.float64) for k in range(n_vector)]


def check_against_matrix(
    operator: InterfaceOperator,
    matrix: Any,
    vectors: List[Any],
) -> float:
    """算子作用与显式矩阵作用的最大相对误差."""
    return max(relative_error(operator @ x, matrix @ x) for x in vectors)


def check_symmetry(operator: InterfaceOperator, vectors: List[Any]) -> float:
    """成对检验 ``x^T A y == y^T A x`` 的最大相对偏差."""
    worst = 0.0
    for x, y in zip(vectors[::2], vectors[1::2]):
        left = float(bm.sum(x * (operator @ y)))
        right = float(bm.sum(y * (operator @ x)))
        scale = max(abs(left), abs(right))
        deviation = abs(left - right) / scale if scale > 0.0 else abs(left - right)
        worst = max(worst, deviation)
    return worst


def run_verification(
    backend: Literal["numpy", "pytorch"],
    seed: int,
    n_random: int,
    n_train: int,
    n_epochs: int,
    learning_rate: float,
    tolerance: float,
    rtol: float,
    maxiter: int,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    """训练代理, 分别用精确 K_s 与预测 K_s_hat 走 Matrix-Free 求解并对比."""
    bm.set_backend(backend)
    set_random_seed(seed)
    rng = np.random.default_rng(seed)

    problem = FullMBBBeam2d(domain=DOMAIN, P=-1.0, E=1.0, nu=0.3)
    domain_size = (problem.domain[1], problem.domain[3])

    assembler = GlobalAssembler(
        domain_size, N_SUB, N_FINE, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    density = make_density_fields(sub_meshes, domain_size, DENSITY_RANGE)

    print("=" * 78)
    print("PIML 近似 Matrix-Free vs 精确 Matrix-Free (FullMBBBeam2d)")
    print("=" * 78)
    print(f"后端           : {backend}")
    print(f"子结构划分     : {N_SUB[0]} x {N_SUB[1]} (共 {len(sub_meshes)} 个)")
    print(f"训练           : {n_train} 组随机密度, {n_epochs} 轮, lr {learning_rate}")
    print(f"CG 停机        : rtol {rtol:.1e}, maxiter {maxiter}, 无预条件")
    print("-" * 78)

    # ------------------------------------------------------------------
    # 代理训练与逐子结构预测缩聚.
    # ------------------------------------------------------------------
    print("[训练] Cholesky 因子代理网络...")
    net, final_loss = train_surrogate(
        prototype, n_train, n_epochs, learning_rate, DENSITY_RANGE
    )
    print(f"       最终训练 MSE: {final_loss:.3e}")
    print("-" * 78)

    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    piml_condensors: List[PIMLStaticCondensation] = []
    for idx, sub_mesh in enumerate(sub_meshes):
        condensor = PIMLStaticCondensation(
            sub_mesh.i_dofs, sub_mesh.b_dofs, model=net, is_cholesky=True,
            range_basis=sub_mesh.deformation_basis,
        )
        condensor.condense(K_local_batch[idx], density[idx])
        piml_condensors.append(condensor)
    n_fallback = sum(1 for c in piml_condensors if c.used_fallback)

    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    exact_condensor.condense(K_local_batch)

    # ------------------------------------------------------------------
    # 接口系统, 外载与约束 (PIML 两条路径与精确参照共享).
    # ------------------------------------------------------------------
    system = assembler.assemble_interface_system(sub_meshes, piml_condensors)
    n_interface = int(len(system.global_dofs))

    # 分析器在此只作为外载与约束的来源, 不装配全局刚度.
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=assembler.full_mesh,
        pde=problem,
        material=assembler.material,
        space_degree=1,
        operator_level="fa",
        topopt_algorithm="density_based",
        solve_method="scipy",
        interpolation_scheme=MaterialInterpolationScheme(
            density_location="element",
            interpolation_method="simp",
            options={"penalty_factor": 3.0, "stress_penalty_factor": 1.0},
        ),
        enable_logging=False,
    )
    global_load = bm.asarray(analyzer.assemble_external_load(), dtype=bm.float64)
    _, fixed_mask = analyzer.tensor_space.boundary_interpolate(
        gd=problem.dirichlet_bc,
        threshold=cast(Any, problem.is_dirichlet_boundary()),
        method="interp",
    )
    fixed_global_dofs = bm.nonzero(fixed_mask)[0]
    fixed_interface_dofs = assembler.project_global_dofs(system, fixed_global_dofs)

    op_free = InterfaceOperator(
        assembler, sub_meshes, piml_condensors, fixed_dofs=fixed_interface_dofs
    )
    op_free_exact = InterfaceOperator(
        assembler, sub_meshes, exact_condensor, fixed_dofs=fixed_interface_dofs
    )
    free = op_free.free
    n_free = op_free.n_free
    K_free = system.stiffness[free, free]

    load_interface = assembler.project_global_vector(system, global_load)
    b_f = load_interface[free]

    # ------------------------------------------------------------------
    # 核心对比: 同一 CG, 唯一变量是 K_s -> K_s_hat.
    # ------------------------------------------------------------------
    def _cg():
        return CGSolver(atol=0.0, rtol=rtol, maxit=maxiter)

    u_exact, info_exact = _cg().setup(op_free_exact).solve(b_f)
    u_op, info_op = _cg().setup(op_free).solve(b_f)
    niter_exact = int(info_exact["niter"])
    niter_op = int(info_op["niter"])
    error_vs_exact = relative_error(u_op, u_exact)

    # ------------------------------------------------------------------
    # 归因证据: 同一批 K_s_hat 的显式装配路径, 与算子路径应恒等.
    # ------------------------------------------------------------------
    u_exp, info_exp = _cg().setup(K_free).solve(b_f)
    niter_exp = int(info_exp["niter"])
    mutual = relative_error(u_op, u_exp)
    error_explicit_vs_exact = relative_error(u_exp, u_exact)

    vectors_free = random_vectors(rng, n_random, n_free)
    matvec_error = check_against_matrix(op_free, K_free, vectors_free)
    symmetry_error = check_symmetry(op_free, vectors_free)
    residual_op = relative_error(op_free @ u_op, b_f)
    residual_exp = relative_error(K_free @ u_exp, b_f)

    # ------------------------------------------------------------------
    # 前提: SPD 证书 (稠密特征值仅验证规模可行).
    # ------------------------------------------------------------------
    dense_free = bm.to_numpy(op_free.to_dense())
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(dense_free)))
    breakdown = bool(info_op.get("breakdown", False)) or bool(
        info_exp.get("breakdown", False)
    )
    converged = (
        bool(info_op.get("converged", False))
        and bool(info_exp.get("converged", False))
        and bool(info_exact.get("converged", False))
    )

    # ------------------------------------------------------------------
    # 结果输出: 对比置顶, 归因与前提随后, 门禁逐项收尾.
    # ------------------------------------------------------------------
    print("[对比] 同一无预条件 CG, 唯一变量: 精确 K_s -> PIML 预测 K_s_hat")
    print(f"  精确 Matrix-Free       : {niter_exact} 步收敛")
    print(
        f"  PIML 近似 Matrix-Free  : {niter_op} 步收敛"
        f" (回退 {n_fallback}, breakdown {'有' if breakdown else '无'})"
    )
    print(f"  两解相对差             : {error_vs_exact:.3e}   <-- 核心结果")
    print("-" * 78)
    print("[归因] 上述误差全部来自代理预测, Matrix-Free 化零贡献")
    print(
        f"  同一批 K_s_hat 显式装配路径: {niter_exp} 步,"
        f" 与算子路径解互差 {mutual:.3e}"
    )
    print(
        f"  单次作用恒等 {matvec_error:.3e}, 对称性 {symmetry_error:.3e},"
        f" 真残差 {residual_op:.1e} / {residual_exp:.1e}"
    )
    print("-" * 78)
    print("[前提] SPD 证书: PIML 预测算子能进 CG")
    print(
        f"  自由子空间最小特征值 {min_eigenvalue:.3e} > 0"
        f" (n_free = {n_free}), 回退 {n_fallback} / {len(sub_meshes)}"
    )
    print("-" * 78)

    checks: Dict[str, float] = {
        "fallback_count": float(n_fallback),
        "min_eigenvalue_positive": 1.0 if min_eigenvalue > 0.0 else 0.0,
        "cg_converged_no_breakdown": 1.0 if (converged and not breakdown) else 0.0,
        "piml_free_matvec": matvec_error,
        "symmetry": symmetry_error,
        "operator_cg_true_residual": residual_op,
        "explicit_cg_true_residual": residual_exp,
        "niter_difference": float(abs(niter_op - niter_exp)),
        "solution_mutual_agreement": mutual,
    }
    verdicts: Dict[str, bool] = {
        "fallback_count": n_fallback == 0,
        "min_eigenvalue_positive": min_eigenvalue > 0.0,
        "cg_converged_no_breakdown": converged and not breakdown,
        "piml_free_matvec": matvec_error < tolerance,
        "symmetry": symmetry_error < tolerance,
        "operator_cg_true_residual": residual_op <= RESIDUAL_SLACK * rtol,
        "explicit_cg_true_residual": residual_exp <= RESIDUAL_SLACK * rtol,
        "niter_difference": abs(niter_op - niter_exp) <= NITER_TOLERANCE,
        "solution_mutual_agreement": mutual <= SOLUTION_MUTUAL_TOL,
    }
    labels = {
        "fallback_count": "门禁回退计数 (须为 0)",
        "min_eigenvalue_positive": "自由子空间最小特征值 > 0",
        "cg_converged_no_breakdown": "CG 全部收敛且无 breakdown",
        "piml_free_matvec": "K_s_hat 算子作用 vs 显式装配",
        "symmetry": "对称性 x^T A y vs y^T A x",
        "operator_cg_true_residual": "算子路径 CG 真残差",
        "explicit_cg_true_residual": "显式路径 CG 真残差",
        "niter_difference": "算子/显式路径迭代数之差",
        "solution_mutual_agreement": "算子/显式路径解互差",
    }
    print("[门禁] 逐项判据")
    for key, value in checks.items():
        verdict = "通过" if verdicts[key] else "失败"
        print(f"  {labels[key]:<34} {value:.3e}  {verdict}")
    print("=" * 78)

    passed = all(verdicts.values())
    if passed:
        print(
            f"结论: PIML 近似 Matrix-Free 相对精确 Matrix-Free 误差"
            f" {error_vs_exact:.2%} (全由预测主导), CG 多"
            f" {niter_op - niter_exact} 步 ({niter_exact} -> {niter_op})."
        )
    else:
        print("结论: 存在失败判据 (见 [门禁]), 上方对比结果不可引用.")

    result: Dict[str, Any] = {
        "backend": backend,
        "seed": seed,
        "n_random": n_random,
        "n_train": n_train,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "final_train_mse": final_loss,
        "operator_tolerance": tolerance,
        "n_sub": list(N_SUB),
        "n_fine": list(N_FINE),
        "n_substructures": len(sub_meshes),
        "n_interface": n_interface,
        "n_free": n_free,
        "n_fallback": n_fallback,
        "min_eigenvalue": min_eigenvalue,
        "cg_rtol": rtol,
        "cg_maxiter": maxiter,
        "niter_operator": niter_op,
        "niter_explicit": niter_exp,
        "niter_exact_reference": niter_exact,
        "checks": checks,
        "verdicts": verdicts,
        "error_operator_vs_exact_reference": error_vs_exact,
        "error_explicit_vs_exact_reference": error_explicit_vs_exact,
        "passed": passed,
    }

    if output_dir is not None:
        target = Path(output_dir)
        if not target.is_absolute():
            target = Path(__file__).parent / target
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"piml_matrix_free_{backend}.json"
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"证据已写入: {path}")

    if not passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 PIML 近似 Matrix-Free 与精确 Matrix-Free, 附归因与 SPD 前提."
    )
    parser.add_argument("--backend", choices=["numpy", "pytorch"], default="numpy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=N_TRAIN)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--tol", type=float, default=OPERATOR_TOLERANCE)
    parser.add_argument("--rtol", type=float, default=CG_RTOL)
    parser.add_argument("--maxiter", type=int, default=CG_MAXITER)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_verification(
        backend=args.backend,
        seed=args.seed,
        n_random=args.n_random,
        n_train=args.train_samples,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        tolerance=args.tol,
        rtol=args.rtol,
        maxiter=args.maxiter,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
