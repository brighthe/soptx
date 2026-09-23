"""缩聚刚度预测路线的二维验证与精度评估入口.

接口基线: 基于完整周边接口 full_trace (单块 5x5 Q1 单元周边 20 个边界节点, 共 40 个接口自由度全保留, 无角点降阶投影).
本脚本在同一物理问题, 同一密度场与同一接口系统上比较两条缩聚路径:
精确 FEAStaticCondensation 批量 Schur 补, 与 ReducedStiffnessCondensation 逐子结构代理预测.
1. 算子层: K_s 的相对 Frobenius 误差;
2. 解层: 接口位移, 全场位移与结构柔度的相对误差;
3. 误差归因诊断: 参数化上限校验、训练同分布留出集评估与零空间模态伪刚度污染分析.

使用方法:
    # 默认训练配置 (2000 样本, 4000 轮)
    python examples/piml_substructure_elasticity/verify_stiffness_route.py

    # 严格模式: 要求代理全程生效, 出现回退即失败
    python examples/piml_substructure_elasticity/verify_stiffness_route.py --strict

    # 可选精度验收: 相对误差 0.01 表示 1%
    python examples/piml_substructure_elasticity/verify_stiffness_route.py \
        --max-ks-error 0.01 --max-displacement-error 0.01 \
        --max-compliance-error 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    ReducedStiffnessCondensation,
    SubstructureMesh,
    SubstructurePrototype,
    build_substructures,
    make_density_fields,
    sample_random_density,
    set_random_seed,
    solve_interface_system,
    train_reduced_stiffness_surrogate,
)
from soptx.ml.substructure import ReducedStiffnessSurrogateNet
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.interpolation import MaterialInterpolationScheme

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# --- 几何与物理模型 ---
DOMAIN = (0.0, 12.0, 0.0, 2.0)
N_SUB = (12, 2)
N_FINE = (5, 5)
P_LOAD = -1.0
E_BASE = 1.0
NU = 0.3
DENSITY_RANGE = (0.3, 1.0)

# --- 默认训练配置 ---
STIFFNESS_N_TRAIN = 2000
STIFFNESS_N_EVAL = 100
STIFFNESS_EPOCHS = 4000
STIFFNESS_LEARNING_RATE = 0.005
STIFFNESS_SEED = 2026

RIGID_BASIS_RESIDUAL_TOL = 1.0e-10
PARAMETERIZATION_ERROR_TOL = 1.0e-4

TABLE_WIDTHS = (38, 22, 22)
DIAG_LABEL_WIDTH = 46


def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计."""
    return sum(2 if unicodedata.east_asian_width(char) in ("F", "W") else 1 for char in s)


def format_table_row(col1: str, col2: str, col3: str) -> str:
    """按显示宽度对齐三列表格行."""
    return (
        f"{col1}{' ' * (TABLE_WIDTHS[0] - display_width(col1))} | "
        f"{col2}{' ' * (TABLE_WIDTHS[1] - display_width(col2))} | "
        f"{col3}{' ' * (TABLE_WIDTHS[2] - display_width(col3))}"
    )


def format_diag_row(label: str, value: str) -> str:
    """按显示宽度对齐诊断段落的标签与数值两列."""
    pad = max(1, DIAG_LABEL_WIDTH - display_width(label))
    return f"{label}{' ' * pad} : {value}"


def solve_with_condensors(
    assembler: Any,
    sub_meshes: List[Any],
    condensors: Any,
    global_load: Any,
    fixed_global_dofs: Any,
) -> Tuple[Any, Any, int]:
    """用给定的缩聚结果装配并求解全局接口系统, 再恢复全场位移."""
    system = assembler.assemble_interface_system(sub_meshes, condensors)
    interface_fixed = assembler.project_global_dofs(system, fixed_global_dofs)
    u_interface = solve_interface_system(
        system,
        assembler.project_global_vector(system, global_load),
        interface_fixed,
    )
    u_full = assembler.recover_full_displacement(
        sub_meshes, condensors, system, u_interface
    )
    n_free = int(len(system.global_dofs)) - int(len(interface_fixed))
    return u_interface, u_full, n_free


class _ExactCholeskyStub(nn.Module):
    """恒定输出给定 Cholesky 条目的桩网络, 用于模拟"完美代理"."""

    def __init__(self, entries: Any) -> None:
        super().__init__()
        self.register_buffer(
            "entries",
            torch.as_tensor(bm.to_numpy(entries), dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.entries)


def verify_parameterization_parity(
    prototype: SubstructurePrototype,
) -> Dict[str, Any]:
    """校验刚体模态基的正确性, 以及训练目标与推理重构是同一个参数化."""
    basis = prototype.deformation_basis
    n_reduced = int(basis.shape[1])
    rho = sample_random_density(prototype, 1, DENSITY_RANGE)
    K_local = prototype.assemble_local_stiffness_batch(rho)
    condensor_exact = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    condensor_exact.condense(K_local)
    K_s = condensor_exact.K_s[0]

    K_s_norm = float(bm.linalg.norm(K_s))
    rigid_residual = float(bm.linalg.norm(K_s @ prototype.rigid_basis)) / K_s_norm

    L = bm.linalg.cholesky(basis.T @ K_s @ basis)
    tril_mask = bm.tril(bm.ones((n_reduced, n_reduced), dtype=bm.bool))

    condensor = ReducedStiffnessCondensation(
        prototype.i_dofs, prototype.b_dofs,
        model=_ExactCholeskyStub(L[tril_mask]), is_cholesky=True,
        range_basis=basis,
    )
    condensor.condense(K_local[0], rho[0])
    K_s_pred = condensor.K_s
    ceiling = float(bm.linalg.norm(K_s_pred - K_s)) / K_s_norm
    return {
        "rigid_basis_residual": rigid_residual,
        "parameterization_error_ceiling": ceiling,
        "parameterization_used_fallback": bool(condensor.used_fallback),
    }


def evaluate_holdout(
    prototype: SubstructurePrototype,
    net: ReducedStiffnessSurrogateNet,
    n_val: int,
) -> Dict[str, Any]:
    """在与训练同分布的留出集上评估代理的算子层精度."""
    rho_val = sample_random_density(prototype, n_val, DENSITY_RANGE)
    K_val_batch = prototype.assemble_local_stiffness_batch(rho_val)

    ref_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    ref_condensor.condense(K_val_batch)
    K_s_ref = ref_condensor.K_s

    piml_condensor = ReducedStiffnessCondensation(
        prototype.i_dofs, prototype.b_dofs, model=net, is_cholesky=True,
        range_basis=prototype.deformation_basis,
    )
    K_s_pred_list: List[Any] = []
    n_fallback = 0
    for i in range(n_val):
        piml_condensor.condense(K_val_batch[i], rho_val[i])
        n_fallback += int(piml_condensor.used_fallback)
        K_s_pred_list.append(piml_condensor.K_s)
    K_s_pred = bm.stack(K_s_pred_list, axis=0)

    err = bm.linalg.norm(
        K_s_pred - K_s_ref, axis=(-2, -1)
    ) / bm.linalg.norm(K_s_ref, axis=(-2, -1))

    return {
        "holdout_samples": n_val,
        "holdout_ks_relative_error_max": float(bm.max(err)),
        "holdout_ks_relative_error_mean": float(bm.mean(err)),
        "holdout_n_fallback": n_fallback,
    }


def diagnose_rigid_mode_pollution(
    K_s_exact: Any,
    K_s_piml: Any,
    u_b_sub: Any,
    n_rigid: int,
) -> Dict[str, Any]:
    """量化代理在精确 K_s 零空间上注入的伪刚度及其能量后果."""
    evals, evecs = bm.linalg.eigh(K_s_exact)
    V = evecs[..., :n_rigid]
    Vt = bm.swapaxes(V, -1, -2)

    lam_rigid = bm.max(bm.abs(evals[:, :n_rigid]), axis=-1)
    lam_soft = evals[:, n_rigid]

    proj_exact = Vt @ K_s_exact @ V
    proj_piml = Vt @ K_s_piml @ V
    pol_exact = bm.max(bm.abs(bm.linalg.eigvalsh(proj_exact)), axis=-1)
    pol_piml = bm.max(bm.abs(bm.linalg.eigvalsh(proj_piml)), axis=-1)
    pol_ratio = pol_piml / lam_soft

    e_exact = bm.einsum("bi,bij,bj->b", u_b_sub, K_s_exact, u_b_sub)
    e_piml = bm.einsum("bi,bij,bj->b", u_b_sub, K_s_piml, u_b_sub)
    a = bm.einsum("bkr,bk->br", V, u_b_sub)
    e_rigid = bm.einsum("br,brs,bs->b", a, proj_piml, a)
    rigid_fraction = bm.linalg.norm(a, axis=-1) / bm.linalg.norm(u_b_sub, axis=-1)

    e_exact_total = float(bm.sum(e_exact))
    e_piml_total = float(bm.sum(e_piml))
    e_rigid_total = float(bm.sum(e_rigid))

    return {
        "n_rigid_modes": n_rigid,
        "exact_rigid_eigenvalue_max": float(bm.max(lam_rigid)),
        "exact_soft_eigenvalue_min": float(bm.min(lam_soft)),
        "exact_rigid_residual_max": float(bm.max(pol_exact)),
        "rigid_pollution_max": float(bm.max(pol_piml)),
        "rigid_pollution_ratio_max": float(bm.max(pol_ratio)),
        "rigid_pollution_ratio_mean": float(bm.mean(pol_ratio)),
        "rigid_displacement_fraction_mean": float(bm.mean(rigid_fraction)),
        "energy_exact": e_exact_total,
        "energy_piml": e_piml_total,
        "energy_stiffening_factor": e_piml_total / e_exact_total,
        "energy_rigid_pollution_share": e_rigid_total / e_exact_total,
    }


def _find_nonfinite_numbers(value: Any, path: str = "result") -> List[str]:
    """递归查找结果记录中的 NaN 和 Inf 数值.

    Parameters
    ----------
    value : Any
        待检查的嵌套结果.
    path : str, optional
        当前值在结果记录中的路径.

    Returns
    -------
    List[str]
        所有非有限数值的字段路径.
    """
    if isinstance(value, dict):
        paths: List[str] = []
        for key, item in value.items():
            paths.extend(_find_nonfinite_numbers(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_find_nonfinite_numbers(item, f"{path}[{index}]"))
        return paths
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    return [] if math.isfinite(float(value)) else [path]


def build_validation(
    result: Dict[str, Any],
    strict: bool,
    max_ks_error: float | None,
    max_displacement_error: float | None,
    max_compliance_error: float | None,
) -> Dict[str, Any]:
    """构造数学一致性、有限性和可选精度验收记录.

    Parameters
    ----------
    result : Dict[str, Any]
        待验收的算子层、解层与参数化结果.
    strict : bool
        是否无条件要求全局解层和留出集零回退.
    max_ks_error : float or None
        留出与在役最大刚度相对误差门槛.
    max_displacement_error : float or None
        接口与全场位移相对误差门槛.
    max_compliance_error : float or None
        柔度相对误差门槛.

    Returns
    -------
    Dict[str, Any]
        可写入 JSON 的验收配置、逐项结果与总结果.
    """
    thresholds = {
        "max_ks_error": max_ks_error,
        "max_displacement_error": max_displacement_error,
        "max_compliance_error": max_compliance_error,
    }
    accuracy_enabled = any(value is not None for value in thresholds.values())
    checks: List[Dict[str, Any]] = []

    def add_check(
        name: str,
        category: str,
        passed: bool,
        value: Any,
        limit: Any,
        message: str,
    ) -> None:
        checks.append({
            "name": name,
            "category": category,
            "passed": bool(passed),
            "value": value,
            "limit": limit,
            "message": message,
        })

    add_check(
        "rigid_basis_residual",
        "mathematical",
        result["rigid_basis_residual"] <= RIGID_BASIS_RESIDUAL_TOL,
        result["rigid_basis_residual"],
        RIGID_BASIS_RESIDUAL_TOL,
        "刚体模态基必须张成精确 K_s 的零空间.",
    )
    add_check(
        "parameterization_error_ceiling",
        "mathematical",
        result["parameterization_error_ceiling"] <= PARAMETERIZATION_ERROR_TOL,
        result["parameterization_error_ceiling"],
        PARAMETERIZATION_ERROR_TOL,
        "完美代理的训练目标排布必须与推理重构一致.",
    )
    add_check(
        "parameterization_zero_fallback",
        "mathematical",
        not result["parameterization_used_fallback"],
        int(result["parameterization_used_fallback"]),
        0,
        "完美 Cholesky 条目不得触发精确回退.",
    )

    nonfinite_paths = _find_nonfinite_numbers(result)
    add_check(
        "finite_numeric_results",
        "finite",
        not nonfinite_paths,
        nonfinite_paths,
        [],
        "所有数值结果必须为有限值.",
    )

    if strict or accuracy_enabled:
        add_check(
            "active_zero_fallback",
            "runtime",
            result["n_fallback"] == 0,
            result["n_fallback"],
            0,
            "strict 模式或精度门槛要求全局解层不得回退.",
        )
        add_check(
            "holdout_zero_fallback",
            "runtime",
            result["holdout_n_fallback"] == 0,
            result["holdout_n_fallback"],
            0,
            "strict 模式或精度门槛要求留出集不得回退.",
        )

    if max_ks_error is not None:
        add_check(
            "holdout_ks_error",
            "accuracy",
            result["holdout_ks_relative_error_max"] <= max_ks_error,
            result["holdout_ks_relative_error_max"],
            max_ks_error,
            "留出集最大 K_s 相对误差不得超过配置门槛.",
        )
        add_check(
            "active_ks_error",
            "accuracy",
            result["ks_relative_error_max"] <= max_ks_error,
            result["ks_relative_error_max"],
            max_ks_error,
            "在役密度场最大 K_s 相对误差不得超过配置门槛.",
        )
    if max_displacement_error is not None:
        add_check(
            "interface_displacement_error",
            "accuracy",
            result["interface_displacement_relative_error"] <= max_displacement_error,
            result["interface_displacement_relative_error"],
            max_displacement_error,
            "接口位移相对误差不得超过配置门槛.",
        )
        add_check(
            "full_displacement_error",
            "accuracy",
            result["displacement_relative_error"] <= max_displacement_error,
            result["displacement_relative_error"],
            max_displacement_error,
            "全场位移相对误差不得超过配置门槛.",
        )
    if max_compliance_error is not None:
        add_check(
            "compliance_error",
            "accuracy",
            result["compliance_relative_error"] <= max_compliance_error,
            result["compliance_relative_error"],
            max_compliance_error,
            "柔度相对误差不得超过配置门槛.",
        )

    return {
        "scope": "二维 full_trace 刚度代理能力验证, 不是 Huang2023 论文的完整复现.",
        "accuracy_thresholds_configured": accuracy_enabled,
        "accuracy_status": (
            "passed"
            if accuracy_enabled and all(check["passed"] for check in checks)
            else "failed"
            if accuracy_enabled
            else "not_requested"
        ),
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def validate_and_write_result(
    result: Dict[str, Any], output_dir: str | None
) -> None:
    """写入完整验收证据, 并对失败的硬检查抛出异常.

    Parameters
    ----------
    result : Dict[str, Any]
        含 validation 字段的完整结果.
    output_dir : str or None
        JSON 证据输出目录; None 表示不写文件.

    Raises
    ------
    AssertionError
        任一数学、有限性、回退或显式精度检查失败.
    """
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "piml_exact_comparison.json"
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[证据] 结果已写入: {target}")

    failures = [
        check for check in result["validation"]["checks"] if not check["passed"]
    ]
    if failures:
        details = "; ".join(
            f"{check['name']}: value={check['value']}, limit={check['limit']}"
            for check in failures
        )
        raise AssertionError(f"验收失败: {details}")


def plot_comparison(
    assembler: GlobalAssembler,
    u_full_exact: Any,
    u_full_piml: Any,
    K_s_exact: Any,
    K_s_piml: Any,
    domain_size: Sequence[float],
    fig_path: Path,
) -> None:
    Lx, Ly = domain_size[0], domain_size[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    for ax, (title, u_full) in zip(
        axes[0],
        (
            ("Exact Schur Condensation: U_y", u_full_exact),
            ("PIML Surrogate Condensation: U_y", u_full_piml),
        ),
    ):
        field = bm.to_numpy(assembler.to_node_grid(u_full[1::2])).T
        im = ax.imshow(
            field, origin="lower", cmap="viridis", extent=[0.0, Lx, 0.0, Ly]
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    for ax, (title, K_s) in zip(
        axes[1],
        (
            ("Exact K_s (substructure 0)", K_s_exact),
            ("PIML K_s (substructure 0)", K_s_piml),
        ),
    ):
        im = ax.imshow(bm.to_numpy(K_s[0]), cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def run_comparison(
    n_train: int = STIFFNESS_N_TRAIN,
    n_epochs: int = STIFFNESS_EPOCHS,
    learning_rate: float = STIFFNESS_LEARNING_RATE,
    n_val: int = STIFFNESS_N_EVAL,
    seed: int = STIFFNESS_SEED,
    output_dir: str | None = None,
    strict: bool = False,
    backend: Literal["numpy", "pytorch"] = "numpy",
    verbose: bool = False,
    max_ks_error: float | None = None,
    max_displacement_error: float | None = None,
    max_compliance_error: float | None = None,
) -> Dict[str, Any]:
    bm.set_backend(backend)
    set_random_seed(seed)

    problem = FullMBBBeam2d(domain=DOMAIN, P=P_LOAD, E=E_BASE, nu=NU)
    domain_size = (problem.domain[1], problem.domain[3])
    n_sub = N_SUB
    n_fine = N_FINE

    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, positions = build_substructures(assembler)
    n_substructures = len(sub_meshes)

    print("\n【配置摘要】")
    print("组别 / 路线 : 1 / stiffness")
    print(f"问题 / 材料 : FullMBBBeam2d, domain={domain_size}, "
          f"{prototype.material.hypothesis}, E0={E_BASE:g}, nu={NU:g}, SIMP penalty=3")
    print(f"载荷        : 顶边中点竖向集中力 P={P_LOAD:g}")
    print(f"网格 / 接口 : Q1, 子结构={n_sub}, 每块细单元={n_fine}, "
          f"full_trace; 每块内部={prototype.n_i}, 接口={prototype.n_b}")
    reduced_dim = int(prototype.deformation_basis.shape[1])
    print(f"全局细网格  : {N_SUB[0] * N_FINE[0]} x {N_SUB[1] * N_FINE[1]} 单元")
    print(f"网络        : {n_fine[0] * n_fine[1]} -> 128 -> 128 -> "
          f"{reduced_dim * (reduced_dim + 1) // 2}, SiLU")
    print(f"训练        : {n_train} 样本, {n_epochs} epochs, Adam, lr={learning_rate:g}, full-batch")
    print(f"目标 / 留出 : Cholesky 独立条目 MSE / {n_val} 样本")
    print(f"密度        : 训练和留出为随机场 {DENSITY_RANGE}; 在役为光滑场")
    print(f"种子 / 求解 : {seed} (采样与初始化), backend={backend}, scipy")
    print("比较基准    : 同网格、密度、载荷与支承的精确 full_trace 缩聚", flush=True)

    density = make_density_fields(sub_meshes, domain_size, DENSITY_RANGE)

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

    # 路径 A: 精确批量 Schur 补
    if verbose:
        print("[路径 A] 精确 Schur 补批量缩聚...")
    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    t0 = time.time()
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    exact_condensor.condense(K_local_batch)
    K_s_exact = exact_condensor.K_s
    t_exact = time.time() - t0

    u_b_exact, u_full_exact, n_free = solve_with_condensors(
        assembler, sub_meshes, exact_condensor, global_load, fixed_global_dofs
    )

    # 代理训练
    parity = verify_parameterization_parity(prototype)
    if verbose:
        print(
            f"[自检]   刚体基残差 {parity['rigid_basis_residual']:.4e}, "
            f"完美代理下 K_s 相对误差上限 {parity['parameterization_error_ceiling']:.4e}"
        )

    if verbose:
        print(f"[训练]   {n_train} 组随机密度, {n_epochs} 轮...")
    net, final_loss = train_reduced_stiffness_surrogate(
        prototype, n_train, n_epochs, learning_rate, DENSITY_RANGE
    )
    if verbose:
        print(f"         最终训练 MSE: {final_loss:.6e}")

    # 路径 B: PIML 代理缩聚
    if verbose:
        print("[路径 B] PIML 代理逐子结构缩聚...")
    t0 = time.time()
    piml_condensors: List[ReducedStiffnessCondensation] = []
    for idx, sub_mesh in enumerate(sub_meshes):
        condensor = ReducedStiffnessCondensation(
            sub_mesh.i_dofs, sub_mesh.b_dofs, model=net, is_cholesky=True,
            range_basis=sub_mesh.deformation_basis,
        )
        condensor.condense(K_local_batch[idx], density[idx])
        piml_condensors.append(condensor)
    t_piml = time.time() - t0

    n_fallback = sum(1 for c in piml_condensors if c.used_fallback)
    K_s_piml = bm.stack([c.K_s for c in piml_condensors], axis=0)

    u_b_piml, u_full_piml, _ = solve_with_condensors(
        assembler, sub_meshes, piml_condensors, global_load, fixed_global_dofs
    )

    err_ks_each = bm.linalg.norm(
        K_s_piml - K_s_exact, axis=(-2, -1)
    ) / bm.linalg.norm(K_s_exact, axis=(-2, -1))
    err_ks_max = float(bm.max(err_ks_each))
    err_ks_mean = float(bm.mean(err_ks_each))

    err_u_b = float(
        bm.linalg.norm(u_b_piml - u_b_exact) / bm.linalg.norm(u_b_exact)
    )
    err_u_full = float(
        bm.linalg.norm(u_full_piml - u_full_exact) / bm.linalg.norm(u_full_exact)
    )

    c_exact = float(bm.sum(global_load * u_full_exact))
    c_piml = float(bm.sum(global_load * u_full_piml))
    err_c = abs(c_piml - c_exact) / abs(c_exact)

    if verbose:
        print(f"[诊断]   {n_val} 组训练同分布留出样本...")
    holdout = evaluate_holdout(prototype, net, n_val)

    b_global = bm.stack(
        [
            assembler.get_substructure_global_dofs(*pos, sm)[sm.b_dofs]
            for pos, sm in zip(positions, sub_meshes)
        ],
        axis=0,
    )
    dim = assembler.dim
    rigid = diagnose_rigid_mode_pollution(
        K_s_exact, K_s_piml, u_full_exact[b_global], dim * (dim + 1) // 2
    )

    result: Dict[str, Any] = {
        "problem": type(problem).__name__,
        "domain": list(problem.domain),
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "n_substructures": n_substructures,
        "n_interface_dofs_local": int(prototype.n_b),
        "n_interior_dofs_local": int(prototype.n_i),
        "n_free_interface_dofs": n_free,
        "n_train_samples": n_train,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "final_train_mse": final_loss,
        "n_reduced_dofs": int(prototype.deformation_basis.shape[1]),
        "density_range": list(DENSITY_RANGE),
        "n_fallback": n_fallback,
        "ks_relative_error_max": err_ks_max,
        "ks_relative_error_mean": err_ks_mean,
        "interface_displacement_relative_error": err_u_b,
        "displacement_relative_error": err_u_full,
        "compliance_exact": c_exact,
        "compliance_piml": c_piml,
        "compliance_relative_error": err_c,
        "condensation_time_exact": t_exact,
        "condensation_time_piml": t_piml,
        "backend": backend,
        "strict": strict,
    }
    result.update(parity)
    result.update(holdout)
    result.update(rigid)
    result["validation"] = build_validation(
        result,
        strict=strict,
        max_ks_error=max_ks_error,
        max_displacement_error=max_displacement_error,
        max_compliance_error=max_compliance_error,
    )

    print("\n【公共误差表】")
    rows = [
        ("留出刚度误差 (mean / max)", f"{holdout['holdout_ks_relative_error_mean']:.2%} / {holdout['holdout_ks_relative_error_max']:.2%}"),
        ("在役刚度误差 (mean / max)", f"{err_ks_mean:.2%} / {err_ks_max:.2%}"),
        ("接口位移相对误差", f"{err_u_b:.2%}"),
        ("全场位移相对误差", f"{err_u_full:.2%}"),
        ("柔度 (精确 / 代理)", f"{c_exact:.8f} / {c_piml:.8f}"),
        ("柔度相对误差", f"{err_c:.2%}"),
    ]
    for label, value in rows:
        print(format_diag_row(label, value))
    print("\n【路线专项诊断】")
    print(format_diag_row("最终训练 MSE", f"{final_loss:.4e}"))
    print(format_diag_row("刚体基相对残差", f"{parity['rigid_basis_residual']:.4e}"))
    print(format_diag_row("完美代理参数化相对误差", f"{parity['parameterization_error_ceiling']:.4e}"))
    print(format_diag_row("刚体子空间伪刚度 (max)", f"{rigid['rigid_pollution_max']:.4e}"))
    print(format_diag_row("精确位移上二次型比值 (代理 / 精确)", f"{rigid['energy_stiffening_factor']:.4f}"))
    if verbose:
        print(format_diag_row("局部耗时 (精确批量 / 代理逐块, s)", f"{t_exact:.6f} / {t_piml:.6f}"))
        print("内部恢复使用精确 N; 上述局部耗时不表示纯网络推理加速比.")
        for label, value in rigid.items():
            print(format_diag_row(label, f"{value:.4e}"))
    print("\n【门禁与结果文件】")
    print(f"全局解层 : 已启用, 回退 {n_fallback}/{n_substructures}")
    print(f"留出集   : 已启用, 回退 {holdout['holdout_n_fallback']}/{n_val}")
    print("误差统计包含回退后的精确结果." if n_fallback or holdout["holdout_n_fallback"]
          else "误差统计未混入精确回退结果.")
    print(f"strict   : {strict} (仅要求零回退, 不设置精度门槛)")
    validation = result["validation"]
    if validation["accuracy_thresholds_configured"]:
        print(f"精度验收 : 已启用, thresholds={validation['thresholds']}")
    else:
        print("精度验收 : 未启用, 当前误差仅报告、不作为通过条件.")
    print("验证范围 : 二维 full_trace 刚度代理能力验证, 不是 Huang2023 论文的完整复现.")
    print(f"验收结果 : {'通过' if validation['passed'] else '失败'}")
    for check in validation["checks"]:
        state = "PASS" if check["passed"] else "FAIL"
        print(format_diag_row(f"[{state}] {check['name']}", check["message"]))

    validate_and_write_result(result, output_dir)

    if output_dir is not None:
        fig_path = Path(output_dir)
        fig_path.mkdir(parents=True, exist_ok=True)
        target = fig_path / "piml_exact_comparison.png"
        plot_comparison(
            assembler, u_full_exact, u_full_piml, K_s_exact, K_s_piml,
            domain_size, target,
        )
        print(f"[产物] 对比云图已保存: {target}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PIML 代理缩聚与精确 Schur 补缩聚对比", allow_abbrev=False
    )
    parser.add_argument(
        "--trace-basis", choices=("full_trace",), default="full_trace",
        help="子结构接口迹空间. 当前仅支持 full_trace (默认), 尚不支持 linear_corner.",
    )
    parser.add_argument(
        "--train-samples", type=int, default=STIFFNESS_N_TRAIN, help="随机密度训练样本数"
    )
    parser.add_argument("--epochs", type=int, default=STIFFNESS_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=STIFFNESS_LEARNING_RATE, help="Adam 学习率")
    parser.add_argument(
        "--val-samples",
        type=int,
        default=STIFFNESS_N_EVAL,
        help="训练同分布留出样本数, 用于区分欠拟合与分布错配",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=STIFFNESS_SEED,
        help="随机数种子, 覆盖训练采样, 留出采样与网络初始化",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="要求代理全程生效, 出现回退即以异常失败",
    )
    parser.add_argument(
        "--max-ks-error",
        type=float,
        default=None,
        help="K_s 最大相对误差门槛, 同时检查留出与在役最大值; 0.01 表示 1%%",
    )
    parser.add_argument(
        "--max-displacement-error",
        type=float,
        default=None,
        help="位移相对误差门槛, 同时检查接口与全场位移; 0.01 表示 1%%",
    )
    parser.add_argument(
        "--max-compliance-error",
        type=float,
        default=None,
        help="柔度相对误差门槛; 0.01 表示 1%%",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "outputs"),
        help="写入 JSON 证据与对比云图的目录",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "pytorch"),
        default="numpy",
        help="bm 后端, 默认 numpy",
    )
    parser.add_argument("--verbose", action="store_true", help="打印过程日志与详细专项诊断")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("train_samples", "val_samples", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " 必须 > 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须为有限数且 > 0.")
    for name in ("max_ks_error", "max_displacement_error", "max_compliance_error"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(name.replace("_", "-") + " 必须为有限数且 >= 0.")
    run_comparison(
        n_train=args.train_samples,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        n_val=args.val_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        strict=args.strict,
        max_ks_error=args.max_ks_error,
        max_displacement_error=args.max_displacement_error,
        max_compliance_error=args.max_compliance_error,
        backend=args.backend,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
