"""子结构缩聚 EA Matrix-Free 的统一验证: 算子级判据 + 求解级判据.

``InterfaceOperator`` 与 ``GlobalAssembler.assemble_interface_system`` 是同一装配
的两种形式: 前者保留批量缩聚刚度按需作用, 后者散加成显式 ``CSRTensor``. 验证分
两组判据, 共享同一算例设置 (FullMBBBeam2d, 精确 Schur 补缩聚), 全脚本只构造一次
接口系统.

算子级判据 (7 条, 阈值相对 1e-13): 算子的单次作用与显式装配代数恒等, 不是近似,
差异只来自浮点求和次序; 超出阈值的偏差指向 gather/scatter 索引错误而非精度损失.

1. 无约束算子作用与 ``system.stiffness @ x`` 一致;
2. 自由子空间上的算子作用与 ``system.stiffness[free, free] @ x_free`` 一致, 与
   ``solve_interface_system`` 施加 Dirichlet 的口径相同;
3. 对称性 ``x^T A y == y^T A x``. 这条不冗余: 缩聚刚度本身对称, 所以把 gather 与
   scatter 的索引写反时判据 1 仍可能通过, 只有对称性检验能暴露该错误;
4. 缩聚结果的两种输入形式 (单个批量缩聚器与逐子结构列表) 以及一个模拟预测扰动的
   批量都能走通, 且各自与自身的显式装配一致. 算子只读 ``K_s``, 不关心它来自精确
   Schur 补还是网络预测, 因此此处不涉及任何代理训练;
5. ``diagonal()`` 与算子逐列展开 (``to_dense``) 的对角一致, 供 Jacobi 类预条件
   使用;
6. ``apply_full`` 的反力路径: 对只在固定自由度上非零的 ``u_c``,
   ``apply_full(u_c)[free]`` 与 ``(system.stiffness @ u_c)[free]`` 一致. 这是
   非齐次 Dirichlet 右端修正 ``-(K u_c)_free`` 的唯一 Matrix-Free 表达, 自由子
   空间上的 ``__matmul__`` 覆盖不到它.

求解级判据 (4 条, 阈值随求解容差): 把算子放进完整迭代求解流程 (构造右端、迭代到
停机、对照参照解) 验证端到端行为, 即 "显式矩阵可以从求解流程中整个拿掉". 三条
求解路径共用同一自由子空间右端 ``b_f = load_interface[free]``: 显式子块交给
``soptx.solvers`` 的 ``DirectSolver`` (SuperLU) 求解作参照; ``CGSolver`` 分别吃
``InterfaceOperator`` (协议 ``SupportsMatmul`` 只要求 ``@``, 无需包装) 与显式
``CSRTensor`` 子块, 参数相同.

7.  算子路径 CG 收敛, 真残差 ``||b - A u|| / ||b||`` 不超过 ``rtol`` 的小倍数
    (取 10, 容纳递推残差与真残差之间的正常漂移);
8.  显式路径 CG 同上;
9.  两条 CG 迭代数之差不超过 1: MatVec 逐位近同, Krylov 子空间与残差轨迹应重合,
    迭代数明显不同说明算子在迭代环境中的行为与显式矩阵不同;
10. 两条 CG 的解相互一致, 阈值 1e-8: 轨迹重合的直接后果. 阈值远松于单次 MatVec
    的 1e-13, 因为每步 1e-16 量级的求和次序差异在数百步迭代中会被条件数放大;
    该判据只为捕捉 "轨迹分岔", 不复述判据 7/8 的残差精度.

两条 CG 的解相对参照直接解的误差作为信息输出, 不设阈值: 该误差由条件数与
``rtol`` 共同决定 (量级 ``kappa * rtol``), 不是算子正确性的判据. 无预条件是有意
的: 求解级本步只引入 "迭代" 一个新变量; Jacobi 预条件 (``diagonal()``) 是求解级
的下一步, PIML 预测 ``K_s`` 代入求解属 ``piml_substructure_elasticity/``.

使用方法:
    python examples/matrix_free_substructure_elasticity/verify_matrix_free_ea.py
    python examples/matrix_free_substructure_elasticity/verify_matrix_free_ea.py --backend pytorch
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.solvers import CGSolver, create

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    InterfaceOperator,
    build_substructures,
    make_density_fields,
)


# PyTorch 首次构造 CSR 张量时, 从 C++ 侧 (SparseCsrTensorImpl.cpp) 发一条
# TORCH_WARN_ONCE 声明该特性处于 beta. 它没有任何 Python API 可以关闭, 是版本状态
# 通知而非缺陷指示, 且本脚本的判据已覆盖所依赖的全部 CSR 行为, 故按消息原文精确屏蔽.
# 过滤器装在脚本里而不是 fealpy 里: 库不该替调用方决定警告策略. 屏蔽只针对这一条
# 消息, 其余 UserWarning 照常显示.
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)


# 算子与显式装配代数恒等, 差异只来自浮点求和次序, 故算子级判据紧到相对 1e-13.
OPERATOR_TOLERANCE = 1e-13

# CG 停机参数; 无预条件下迭代数本身就是被观测量, 上限须留足.
CG_RTOL = 1e-10
CG_MAXITER = 20000

# 求解级: 真残差判据相对 rtol 的松弛倍数, 见模块 docstring.
RESIDUAL_SLACK = 10.0

# 求解级: 两条 CG 迭代数允许的最大差值.
NITER_TOLERANCE = 1

# 求解级: 两条 CG 解相互一致的阈值, 见模块 docstring.
SOLUTION_MUTUAL_TOL = 1e-8

# 密度场取值区间, 与 piml_substructure_elasticity/_common.py 的
# DENSITY_RANGE 保持一致, 使本目录与该目录的证据落在同一密度场族上可对照.
DENSITY_RANGE = (0.3, 1.0)


class CondensorStub:
    """只携带 ``K_s`` 的最小缩聚器替身.

    ``GlobalAssembler.normalize_condensors`` 对缩聚器只读 ``K_s``, 因此验证输入
    形式时无需构造完整缩聚器. ``recover`` 在本脚本的路径上不会被调用.
    """

    def __init__(self, K_s: Any) -> None:
        self.K_s = K_s

    def recover(self, u_b: Any) -> Any:
        raise NotImplementedError("验证脚本不恢复内部位移.")


def relative_error(actual: Any, expected: Any) -> float:
    """两个向量的相对 2-范数误差; 参照量为零时退化为绝对误差."""
    diff = float(bm.sqrt(bm.sum((actual - expected) ** 2)))
    scale = float(bm.sqrt(bm.sum(expected ** 2)))
    return diff / scale if scale > 0.0 else diff


def random_vectors(rng: np.random.Generator, n_vector: int, length: int) -> List[Any]:
    """生成一组后端无关的随机向量.

    说明:
        用 NumPy 生成再转到当前后端, 使同一 ``--seed`` 下 numpy 与 pytorch 两个后端
        拿到逐位相同的输入, 两次运行的误差数字可直接对比.
    """
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
    tolerance: float,
    rtol: float,
    maxiter: int,
    output_dir: Optional[str],
) -> Dict[str, Any]:
    """构造 MBB 梁接口系统, 依次执行算子级与求解级两组判据."""
    bm.set_backend(backend)
    rng = np.random.default_rng(seed)

    problem = FullMBBBeam2d(domain=(0.0, 12.0, 0.0, 2.0), P=-1.0, E=1.0, nu=0.3)
    domain_size = (problem.domain[1], problem.domain[3])
    n_sub = (12, 2)
    n_fine = (5, 5)

    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
    density = make_density_fields(sub_meshes, domain_size, DENSITY_RANGE)

    print("=" * 78)
    print("子结构缩聚 EA Matrix-Free 验证 · 算子级 + 求解级 (FullMBBBeam2d)")
    print("=" * 78)
    print(f"后端           : {backend}")
    print(f"子结构划分     : {n_sub[0]} x {n_sub[1]} (共 {len(sub_meshes)} 个)")
    print(f"随机向量组数   : {n_random}")
    print(f"算子级判据阈值 : {tolerance:.1e} (相对)")
    print(f"CG 停机        : rtol {rtol:.1e}, maxiter {maxiter}")
    print("-" * 78)

    # 精确 Schur 补缩聚作为算子的输入; 算子对 K_s 的来源无感.
    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_exact, _ = exact_condensor.condense(K_local_batch)

    system = assembler.assemble_interface_system(sub_meshes, exact_condensor)
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

    op_full = InterfaceOperator(assembler, sub_meshes, exact_condensor)
    op_free = InterfaceOperator(
        assembler, sub_meshes, exact_condensor, fixed_dofs=fixed_interface_dofs
    )
    free = op_free.free
    n_free = op_free.n_free

    # ------------------------------------------------------------------
    # 算子级判据: 单次作用与显式装配的代数恒等.
    # ------------------------------------------------------------------
    operator_checks: Dict[str, float] = {}

    # 判据 1: 无约束算子作用.
    vectors_full = random_vectors(rng, n_random, n_interface)
    operator_checks["unconstrained_matvec"] = check_against_matrix(
        op_full, system.stiffness, vectors_full
    )

    # 判据 2: 自由子空间算子作用, 对照 solve_interface_system 的同一子块.
    K_free = system.stiffness[free, free]
    vectors_free = random_vectors(rng, n_random, n_free)
    operator_checks["free_subblock_matvec"] = check_against_matrix(
        op_free, K_free, vectors_free
    )

    # 判据 3: 对称性, 暴露 gather/scatter 索引写反.
    operator_checks["symmetry"] = check_symmetry(op_free, vectors_free)

    # 判据 4: 缩聚结果的不同输入形式.
    stub_list = [CondensorStub(K_s_exact[j]) for j in range(len(sub_meshes))]
    op_list = InterfaceOperator(assembler, sub_meshes, stub_list)
    operator_checks["condensor_list_form"] = check_against_matrix(
        op_list, system.stiffness, vectors_full
    )

    # 模拟预测扰动: 在精确 K_s 上叠加对称扰动, 幅值取最大元的 1%.
    raw = rng.standard_normal(tuple(int(s) for s in K_s_exact.shape))
    amplitude = 0.01 * float(bm.max(bm.abs(K_s_exact)))
    noise = amplitude * (raw + raw.transpose(0, 2, 1))
    K_s_perturbed = K_s_exact + bm.asarray(noise, dtype=bm.float64)
    perturbed = CondensorStub(K_s_perturbed)
    system_perturbed = assembler.assemble_interface_system(sub_meshes, perturbed)
    op_perturbed = InterfaceOperator(assembler, sub_meshes, perturbed)
    operator_checks["perturbed_batch_form"] = check_against_matrix(
        op_perturbed, system_perturbed.stiffness, vectors_full
    )

    # 判据 5: diagonal() 对照逐列展开的对角. to_dense 为 n_free 次算子作用,
    # 只在本验证规模下使用.
    dense_free = op_free.to_dense()
    idx = bm.arange(n_free, dtype=bm.int64)
    operator_checks["diagonal"] = relative_error(
        op_free.diagonal(), dense_free[idx, idx]
    )

    # 判据 6: apply_full 的反力路径, 对应非齐次 Dirichlet 右端修正 -(K u_c)_free.
    u_c = bm.zeros((n_interface,), dtype=bm.float64)
    fixed_values = bm.asarray(
        rng.standard_normal(int(len(op_free.fixed))), dtype=bm.float64
    )
    u_c = bm.set_at(u_c, op_free.fixed, fixed_values)
    operator_checks["apply_full_reaction"] = relative_error(
        op_free.apply_full(u_c)[free], (system.stiffness @ u_c)[free]
    )

    operator_labels = {
        "unconstrained_matvec": "无约束算子作用 vs 显式装配",
        "free_subblock_matvec": "自由子块算子作用 vs 显式子块",
        "symmetry": "对称性 x^T A y vs y^T A x",
        "condensor_list_form": "逐子结构列表输入形式",
        "perturbed_batch_form": "扰动批量 (模拟预测输入)",
        "diagonal": "diagonal() vs 逐列展开对角",
        "apply_full_reaction": "apply_full 反力路径 (固定自由度输入)",
    }
    print("[算子级判据] 单次作用与显式装配恒等")
    for key, value in operator_checks.items():
        verdict = "通过" if value < tolerance else "失败"
        print(f"  {operator_labels[key]:<34} {value:.3e}  {verdict}")

    # ------------------------------------------------------------------
    # 求解级判据: 完整迭代求解流程中显式矩阵可整个拿掉.
    # ------------------------------------------------------------------
    load_interface = assembler.project_global_vector(system, global_load)
    b_f = load_interface[free]

    # 路径 1: 显式子块直接法 (SuperLU), 作为参照解.
    direct = create("scipy")
    try:
        u_ref, _ = direct.setup(K_free).solve(b_f)
    finally:
        direct.close()

    # 路径 2: soptx CGSolver 吃 InterfaceOperator (matrix-free).
    u_op, info_op = CGSolver(atol=0.0, rtol=rtol, maxit=maxiter).setup(
        op_free
    ).solve(b_f)

    # 路径 3: 同一个 CGSolver 配置吃显式 CSRTensor 子块.
    u_exp, info_exp = CGSolver(atol=0.0, rtol=rtol, maxit=maxiter).setup(
        K_free
    ).solve(b_f)

    niter_op = int(info_op["niter"])
    niter_exp = int(info_exp["niter"])

    solve_checks: Dict[str, float] = {
        "operator_cg_true_residual": relative_error(op_free @ u_op, b_f),
        "explicit_cg_true_residual": relative_error(K_free @ u_exp, b_f),
        "niter_difference": float(abs(niter_op - niter_exp)),
        "solution_mutual_agreement": relative_error(u_op, u_exp),
    }
    solve_thresholds: Dict[str, float] = {
        "operator_cg_true_residual": RESIDUAL_SLACK * rtol,
        "explicit_cg_true_residual": RESIDUAL_SLACK * rtol,
        "niter_difference": float(NITER_TOLERANCE),
        "solution_mutual_agreement": SOLUTION_MUTUAL_TOL,
    }
    solve_labels = {
        "operator_cg_true_residual": "算子路径 CG 真残差",
        "explicit_cg_true_residual": "显式路径 CG 真残差",
        "niter_difference": "两条 CG 迭代数之差",
        "solution_mutual_agreement": "两条 CG 解相互一致",
    }
    print("-" * 78)
    print("[求解级判据] 无预条件 CG 端到端一致性")
    for key, value in solve_checks.items():
        verdict = "通过" if value <= solve_thresholds[key] else "失败"
        print(
            f"  {solve_labels[key]:<26} {value:.3e}"
            f"  (阈值 {solve_thresholds[key]:.1e})  {verdict}"
        )

    error_op_vs_ref = relative_error(u_op, u_ref)
    error_exp_vs_ref = relative_error(u_exp, u_ref)

    print("-" * 78)
    print("[信息] 不设阈值的观测量")
    print(f"  接口自由度总数 (含约束) : {n_interface}")
    print(f"  自由接口自由度数 n_free : {n_free}")
    print(f"  算子路径 CG 迭代数      : {niter_op}")
    print(f"  显式路径 CG 迭代数      : {niter_exp}")
    print(f"  算子解 vs 直接法参照解  : {error_op_vs_ref:.3e}")
    print(f"  显式解 vs 直接法参照解  : {error_exp_vs_ref:.3e}")
    print("=" * 78)

    operator_passed = all(value < tolerance for value in operator_checks.values())
    solve_passed = all(
        solve_checks[key] <= solve_thresholds[key] for key in solve_checks
    )
    passed = operator_passed and solve_passed
    print(
        "结论: "
        + (
            "全部判据通过."
            if passed
            else f"存在失败判据 (算子级 {'通过' if operator_passed else '失败'}, "
            f"求解级 {'通过' if solve_passed else '失败'}), 见上表."
        )
    )

    result: Dict[str, Any] = {
        "backend": backend,
        "seed": seed,
        "n_random": n_random,
        "operator_tolerance": tolerance,
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "n_substructures": len(sub_meshes),
        "n_b_per_substructure": int(prototype.n_b),
        "n_interface": n_interface,
        "n_free": n_free,
        "operator_checks": operator_checks,
        "operator_passed": operator_passed,
        "cg_rtol": rtol,
        "cg_maxiter": maxiter,
        "niter_operator": niter_op,
        "niter_explicit": niter_exp,
        "solve_checks": solve_checks,
        "solve_thresholds": solve_thresholds,
        "error_operator_vs_reference": error_op_vs_ref,
        "error_explicit_vs_reference": error_exp_vs_ref,
        "solve_passed": solve_passed,
        "passed": passed,
    }

    if output_dir is not None:
        target = Path(output_dir)
        if not target.is_absolute():
            target = Path(__file__).parent / target
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"matrix_free_ea_{backend}.json"
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"证据已写入: {path}")

    if not passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证子结构缩聚 EA Matrix-Free: 算子级恒等 + 求解级可替换."
    )
    parser.add_argument("--backend", choices=["numpy", "pytorch"], default="numpy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--tol", type=float, default=OPERATOR_TOLERANCE)
    parser.add_argument("--rtol", type=float, default=CG_RTOL)
    parser.add_argument("--maxiter", type=int, default=CG_MAXITER)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_verification(
        backend=args.backend,
        seed=args.seed,
        n_random=args.n_random,
        tolerance=args.tol,
        rtol=args.rtol,
        maxiter=args.maxiter,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
