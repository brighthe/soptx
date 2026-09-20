"""形函数预测路线的二维力学验证与精度评估.

支持两种子结构接口迹空间模式:
- full_trace: 完整周边接口 (保留全部 20 个边界节点, 40 个接口自由度, 输出 1184 维, 对应 Huang 2023 组 1)
- linear_corner: 角点线性迹接口 (边界向 4 个角点线性插值, 8 个角点自由度, 输出 160 维, 对应 Huang 2023 式 (16) 及组 2)

本脚本验证 Huang 2023 式 (17) 的二阶误差性质, 不代表完整论文复现:
1. 刚体分量密度无关性与解析构造校验;
2. 式 (17) 误差闭式与误差矩阵半正定性验证;
3. 受控扰动扫描 (拟合 log-log 理论二阶斜率 2.0);
4. 形函数神经网络训练 (变形子空间投影分量 M) 与留出集两层误差评估;
5. 固定密度 MBB 梁 (12x2 子结构) 解层精度评估, 不启用运行时回退;
6. full_trace 独立门禁诊断与故障注入检查.

使用方法:
    # 默认 full_trace (组 1)
    python examples/piml_substructure_elasticity/verify_shape_function_route.py

    # 切换为 linear_corner (组 2, 8 个角点自由度, 160 维输出)
    python examples/piml_substructure_elasticity/verify_shape_function_route.py --trace-basis linear_corner

    # 解析恒等式与二阶受控扫描 (跳过网络训练与解层验证)
    python examples/piml_substructure_elasticity/verify_shape_function_route.py --trace-basis linear_corner --skip-train
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fealpy.backend import backend_manager as bm

from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    LinearCornerTraceBasis,
    LocalReductionBatchResult,
    ReductionDiagnostics,
    ShapeFunctionCondensation,
    SubstructurePrototype,
    SurrogateContractError,
    build_substructures,
    make_density_fields,
    solve_constrained_system,
    solve_interface_system,
)
from soptx.ml.substructure import ShapeFunctionSurrogateNet
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.interpolation import MaterialInterpolationScheme

_SCRIPT_DIR = Path(__file__).resolve().parent

# --- 几何与物理模型 ---
DOMAIN = (0.0, 12.0, 0.0, 2.0)
N_SUB = (12, 2)
N_FINE = (5, 5)
P_LOAD = -1.0
E_BASE = 1.0
NU = 0.3

# --- 训练与评估区间 ---
DENSITY_RANGE = (0.3, 1.0)
DOMAIN_SIZE = (DOMAIN[1] - DOMAIN[0], DOMAIN[3] - DOMAIN[2])
SUB_SIZE = (DOMAIN_SIZE[0] / N_SUB[0], DOMAIN_SIZE[1] / N_SUB[1])

# --- 形函数训练超参数 ---
SHAPE_N_TRAIN = 2000
SHAPE_N_EVAL = 200
SHAPE_EPOCHS = 4000
SHAPE_LEARNING_RATE = 0.005
SHAPE_SEED = 2026
SHAPE_HIDDEN_DIM = 256
SHAPE_SIMP_PENALTY = 3.0
SHAPE_RHO_MIN = 0.0


def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计."""
    return sum(2 if unicodedata.east_asian_width(char) in ("F", "W") else 1 for char in s)


def set_random_seed(seed: int) -> None:
    """统一固定 numpy, torch 与 bm 后端的随机数种子."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    bm.random.seed(seed)


def sample_random_density(n_sample: int, seed_offset: int = 0) -> Any:
    """按训练分布采样一批随机局部密度."""
    rng = np.random.default_rng(SHAPE_SEED + seed_offset)
    lo, hi = DENSITY_RANGE
    return bm.asarray(
        lo + (hi - lo) * rng.random((n_sample, *N_FINE)), dtype=bm.float64
    )


def solve_with_condensors(
    assembler: Any,
    sub_meshes: List[Any],
    condensors: Any,
    global_load: Any,
    fixed_global_dofs: Any,
) -> Tuple[Any, Any, int]:
    """用给定的缩聚结果装配并求解全局接口系统, 再恢复全场位移 (full_trace 路径)."""
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


class Eq17Evaluator:
    """在固定子结构原型上评估式 (17) 与两条缩聚路径的精度 (支持 full_trace 与 linear_corner)."""

    def __init__(self, trace_basis: str = "full_trace") -> None:
        self.trace_basis = trace_basis
        self.prototype = SubstructurePrototype(
            SUB_SIZE, N_FINE, E_base=E_BASE, nu=NU,
            penal=SHAPE_SIMP_PENALTY, rho_min=SHAPE_RHO_MIN,
        )
        self.condensor = FEAStaticCondensation(
            self.prototype.i_dofs, self.prototype.b_dofs
        )
        self.i_dofs = bm.to_numpy(self.prototype.i_dofs)
        self.b_dofs = bm.to_numpy(self.prototype.b_dofs)
        self.n_i = int(self.prototype.n_i)
        self.n_b = int(self.prototype.n_b)
        self.n_dof = self.n_i + self.n_b

        # full_trace 即 T = I, 核心库以 trace=None 表示, 从而完全绕开迹矩阵乘法.
        if trace_basis == "full_trace":
            self.trace: Optional[Any] = None
        elif trace_basis == "linear_corner":
            self.trace = LinearCornerTraceBasis.from_prototype(self.prototype)
        else:
            raise ValueError(f"未知的接口空间模式: {trace_basis!r}")
        self.L = None if self.trace is None else bm.to_numpy(self.trace.matrix)

        # 迹空间刚体基 R_q, 变形子空间基 R_perp 与刚体内部取值 Phi_i 全部由核心库
        # 按维数无关的方式构造, 脚本不再自行做 pinv/QR/特征分解.
        R_q, R_perp, Phi_i = self.prototype.trace_interface_bases(self.trace)
        self.R_rigid = bm.to_numpy(R_q)
        self.R_perp = bm.to_numpy(R_perp)
        self.Phi_i = bm.to_numpy(Phi_i)
        self.n_interface = int(self.R_rigid.shape[0])
        self.n_rigid = int(self.R_rigid.shape[1])
        self.n_reduced = int(self.R_perp.shape[1])

        # 无网络的核心缩聚器, 只借用其参数化与变分式实现; 脚本中所有 K_r 构造都
        # 经由它, 与生产路径 PIMLShapeReduction 共用同一份代码.
        self.core = ShapeFunctionCondensation(
            self.prototype.i_dofs,
            self.prototype.b_dofs,
            rigid_basis=R_q,
            deformation_basis=R_perp,
            rigid_interior=Phi_i,
            trace=self.trace,
        )
        # 完整接口上的同一构造, 供解层把迹空间延拓重新摊回全边界时复用.
        self.core_full = self.core if self.trace is None else (
            ShapeFunctionCondensation(
                self.prototype.i_dofs,
                self.prototype.b_dofs,
                rigid_basis=self.prototype.rigid_basis,
                deformation_basis=self.prototype.deformation_basis,
                rigid_interior=self.prototype.rigid_interior_modes,
            )
        )

    def exact_batch(self, rho: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量求解精确缩聚刚度与内部延拓, 并降到当前迹空间."""
        K_batch = self.prototype.assemble_local_stiffness_batch(rho)
        K_s, N = self.condensor.condense(K_batch)
        if self.trace is None:
            return bm.to_numpy(K_batch), bm.to_numpy(K_s), bm.to_numpy(N)
        return (
            bm.to_numpy(K_batch),
            bm.to_numpy(self.trace.project_stiffness(K_s)),
            bm.to_numpy(self.trace.reduce_recovery(N)),
        )

    def reduced_stiffness(
        self, K_local: np.ndarray, recovery: np.ndarray
    ) -> np.ndarray:
        """由内部延拓构造迹空间降阶刚度, 直接调用核心库的变分式实现."""
        return bm.to_numpy(self.core.assemble_reduced_stiffness(
            bm.asarray(K_local, dtype=bm.float64),
            bm.asarray(recovery, dtype=bm.float64),
        ))

    def assemble_recovery(self, M: np.ndarray) -> np.ndarray:
        """由变形子空间分量 M 合成内部延拓 B (满足刚体不变性约束)."""
        return bm.to_numpy(
            self.core.assemble_recovery(bm.asarray(M, dtype=bm.float64))
        )

    def project_deformation(self, B: np.ndarray) -> np.ndarray:
        """把内部延拓投影到变形子空间, 提取训练目标 M."""
        return bm.to_numpy(
            self.core.project_deformation(bm.asarray(B, dtype=bm.float64))
        )


def prepare_training_arrays(ev: Eq17Evaluator, n_train: int):
    """生成训练集输入与标签张量."""
    rho = sample_random_density(n_train, seed_offset=0)
    _, _, N = ev.exact_batch(rho)
    target = ev.project_deformation(N)
    return (
        torch.tensor(bm.to_numpy(rho).reshape(n_train, -1), dtype=torch.float32),
        torch.tensor(target.reshape(n_train, -1), dtype=torch.float32),
    )


def fit_full_batch(net, X, Y, n_epochs: int, learning_rate: float):
    """全批量 Adam 训练循环."""
    if n_epochs <= 0 or learning_rate <= 0:
        raise ValueError("训练轮数与学习率必须为正。")
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.train()
    losses = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(net(X), Y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    net.eval()
    return tuple(losses)


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(a - b, axis=(-2, -1))
    den = np.linalg.norm(b, axis=(-2, -1))
    return num / den


def step3_trained_network(
    ev: Eq17Evaluator, net: ShapeFunctionSurrogateNet, n_eval: int
) -> Dict[str, Any]:
    """在留出集上评估形函数路径的两层误差."""
    rho = sample_random_density(n_eval, seed_offset=555)
    K_local, K_s, N = ev.exact_batch(rho)

    with torch.no_grad():
        X = torch.tensor(
            bm.to_numpy(rho).reshape(n_eval, -1),
            dtype=torch.float32,
            device=next(net.parameters()).device,
        )
        M_pred = net(X).detach().cpu().numpy().reshape(n_eval, ev.n_i, ev.n_reduced)

    N_hat = ev.assemble_recovery(M_pred.astype(np.float64))
    K_tilde = ev.reduced_stiffness(K_local, N_hat)

    eps_N = relative_frobenius(N_hat, N)
    eps_K17 = relative_frobenius(K_tilde, K_s)

    rigid_dev = np.linalg.norm(
        N_hat @ ev.R_rigid - ev.Phi_i, axis=(-2, -1)
    ) / np.linalg.norm(ev.Phi_i)

    diff = K_tilde - K_s
    min_eig = np.array([
        np.linalg.eigvalsh(0.5 * (d + d.T))[0] for d in diff
    ])
    scale = np.linalg.norm(K_s, axis=(-2, -1))

    return {
        "n_eval": int(n_eval),
        "eps_N_mean": float(eps_N.mean()),
        "eps_N_max": float(eps_N.max()),
        "eps_N_p95": float(np.quantile(eps_N, 0.95)),
        "eps_K17_mean": float(eps_K17.mean()),
        "eps_K17_max": float(eps_K17.max()),
        "eps_K17_p95": float(np.quantile(eps_K17, 0.95)),
        "rigid_constraint_deviation_max": float(rigid_dev.max()),
        "eq17_error_min_eigenvalue_min": float(min_eig.min()),
        "eq17_error_relative_min_eigenvalue_min": float((min_eig / scale).min()),
    }


def make_condensor(
    ev: Eq17Evaluator,
    i_dofs: Any,
    b_dofs: Any,
    model: Optional[nn.Module],
) -> ShapeFunctionCondensation:
    """构造适配当前接口模式的形函数缩聚器.

    两种迹基走同一条构造路径, 区别只在传入的 ``trace``: ``full_trace`` 传
    ``None`` (即 ``T = I``), ``linear_corner`` 传 ``LinearCornerTraceBasis``.
    """
    return ShapeFunctionCondensation(
        i_dofs,
        b_dofs,
        model=model,
        rigid_basis=bm.asarray(ev.R_rigid),
        deformation_basis=bm.asarray(ev.R_perp),
        rigid_interior=bm.asarray(ev.Phi_i),
        trace=ev.trace,
    )


def step4_solution_layer(
    ev: Eq17Evaluator, net: ShapeFunctionSurrogateNet
) -> Dict[str, Any]:
    """在完整 MBB 梁上比较形函数路径与精确缩聚的解层误差 (适配 full_trace 与 linear_corner)."""
    problem = FullMBBBeam2d(domain=DOMAIN, P=P_LOAD, E=E_BASE, nu=NU)
    domain_size = (problem.domain[1], problem.domain[3])
    assembler = GlobalAssembler(
        domain_size, N_SUB, N_FINE, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, _ = build_substructures(assembler)
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

    K_local_batch = prototype.assemble_local_stiffness_batch(density)

    # 1. 精确缩聚基线
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    exact_condensor.condense(K_local_batch)
    K_s_exact_full = exact_condensor.K_s

    # 2. 预测形函数并由此构造缩聚刚度
    with torch.no_grad():
        X = torch.tensor(
            bm.to_numpy(density).reshape(len(sub_meshes), -1),
            dtype=torch.float32,
            device=next(net.parameters()).device,
        )
        M_pred = net(X).detach().cpu().numpy().reshape(len(sub_meshes), ev.n_i, ev.n_reduced)
    N_pred = ev.assemble_recovery(M_pred.astype(np.float64))
    K_s_route_np = ev.reduced_stiffness(bm.to_numpy(K_local_batch), N_pred)
    K_s_route = bm.asarray(K_s_route_np, dtype=bm.float64)

    # 将角点恢复关系延拓到完整边界, 仅在 u_b = L q 上使用该延拓.
    # N_boundary @ L = N_pred; 完整边界刚度按同一个恢复关系构造,
    # 因而满足公共批量结果契约, 不借用精确刚度或伪造流式接口.
    N_boundary = N_pred if ev.L is None else N_pred @ np.linalg.pinv(ev.L)
    K_boundary = bm.to_numpy(ev.core_full.assemble_reduced_stiffness(
        K_local_batch, bm.asarray(N_boundary, dtype=bm.float64)
    ))
    predicted_result = LocalReductionBatchResult(
        stiffness=bm.asarray(K_boundary, dtype=bm.float64),
        recovery=bm.asarray(N_boundary, dtype=bm.float64),
        diagnostics=tuple(
            ReductionDiagnostics(
                requested_method="shape_function",
                stiffness_source="variational_boundary_extension",
                recovery_source="predicted_shape_function",
            )
            for _ in sub_meshes
        ),
    )
    if ev.trace is None:
        projected_K, projected_N = K_boundary, N_boundary
    else:
        projected_K = bm.to_numpy(ev.trace.project_stiffness(K_boundary))
        projected_N = bm.to_numpy(ev.trace.reduce_recovery(N_boundary))
    recovery_consistency = float(np.max(relative_frobenius(projected_N, N_pred)))
    stiffness_consistency = float(np.max(relative_frobenius(projected_K, K_s_route_np)))
    solve_diagnostics: Dict[str, float] = {}

    # 解层使用原始预测, 与独立留出门禁诊断分开.
    n_fallback = 0

    if ev.trace_basis == "full_trace":
        # 全边界系统求解
        u_b_exact, u_full_exact, _ = solve_with_condensors(
            assembler, sub_meshes, exact_condensor, global_load, fixed_global_dofs
        )
        # 组装预测系统
        u_b_route, u_full_route, _ = solve_with_condensors(
            assembler, sub_meshes, predicted_result, global_load, fixed_global_dofs
        )
        K_s_ref = K_s_exact_full
    else:
        # linear_corner 系统求解
        trace_basis = LinearCornerTraceBasis.from_prototype(prototype)
        interface_dofs = assembler.build_interface_dofs(sub_meshes)
        interface_view = SimpleNamespace(global_dofs=interface_dofs)
        projection = assembler.build_linear_corner_projection(
            sub_meshes, interface_view, trace_basis
        )

        interface_force = np.asarray(bm.to_numpy(
            assembler.project_global_vector(interface_view, global_load)
        ))
        fixed_interface = np.asarray(bm.to_numpy(
            assembler.project_global_dofs(interface_view, fixed_global_dofs)
        ), dtype=np.int64)
        force = projection.T @ interface_force
        constraints = projection[fixed_interface]

        # 精确宏观角点系统
        K_s_exact_corner = bm.to_numpy(
            trace_basis.project_stiffness(K_s_exact_full)
        )
        exact_macro = assembler.assemble_macro_system(sub_meshes, bm.asarray(K_s_exact_corner))
        exact_solve = solve_constrained_system(exact_macro, force, constraints)
        q_exact = exact_solve.displacement
        interface_u_exact = bm.asarray(projection @ bm.to_numpy(q_exact), dtype=bm.float64)
        u_full_exact = assembler.recover_full_displacement(
            sub_meshes, exact_condensor, interface_view, interface_u_exact
        )
        u_b_exact = interface_u_exact

        # 代理宏观角点系统
        route_macro = assembler.assemble_macro_system(sub_meshes, K_s_route)
        route_solve = solve_constrained_system(route_macro, force, constraints)
        q_route = route_solve.displacement
        interface_u_route = bm.asarray(projection @ bm.to_numpy(q_route), dtype=bm.float64)

        u_full_route = assembler.recover_full_displacement(
            sub_meshes, predicted_result, interface_view, interface_u_route
        )
        solve_diagnostics = {
            "equilibrium_relative_residual_max": max(
                exact_solve.equilibrium_relative_residual,
                route_solve.equilibrium_relative_residual,
            ),
            "constraint_relative_residual_max": max(
                exact_solve.constraint_relative_residual,
                route_solve.constraint_relative_residual,
            ),
        }
        u_b_route = interface_u_route
        K_s_ref = bm.asarray(K_s_exact_corner)

    err_ks = bm.linalg.norm(
        K_s_route - K_s_ref, axis=(-2, -1)
    ) / bm.linalg.norm(K_s_ref, axis=(-2, -1))

    c_exact = float(bm.dot(global_load, u_full_exact))
    c_route = float(bm.dot(global_load, u_full_route))

    return {
        "in_service_ks_relative_error_max": float(bm.max(err_ks)),
        "in_service_ks_relative_error_mean": float(bm.mean(err_ks)),
        "interface_displacement_relative_error": float(
            bm.linalg.norm(u_b_route - u_b_exact) / bm.linalg.norm(u_b_exact)
        ),
        "displacement_relative_error": float(
            bm.linalg.norm(u_full_route - u_full_exact) / bm.linalg.norm(u_full_exact)
        ),
        "compliance_exact": c_exact,
        "compliance_route": c_route,
        "compliance_relative_error": abs(c_route - c_exact) / abs(c_exact),
        "n_fallback": n_fallback,
        "gate_enabled": False,
        "trace_recovery_relative_error": recovery_consistency,
        "trace_stiffness_relative_error": stiffness_consistency,
        **solve_diagnostics,
    }


def step0_rigid_part(ev: Eq17Evaluator) -> Dict[str, Any]:
    """步骤 0: 校验刚体模态解析不变性与密度无关性."""
    rho = sample_random_density(8, seed_offset=31)
    _, _, N = ev.exact_batch(rho)
    Phi = N @ ev.R_rigid
    dev = np.linalg.norm(Phi - Phi[0], axis=(-2, -1)) / np.linalg.norm(Phi[0])
    dev_analytic = (
        np.linalg.norm(Phi - ev.Phi_i, axis=(-2, -1)) / np.linalg.norm(ev.Phi_i)
    )
    return {
        "rigid_part_density_independence_max": float(dev.max()),
        "rigid_part_analytic_deviation_max": float(dev_analytic.max()),
    }


def step1_identity(ev: Eq17Evaluator) -> Dict[str, Any]:
    """步骤 1: 验证式 (17) 二阶误差闭式与半正定性."""
    rng = np.random.default_rng(11)
    rho = sample_random_density(1, seed_offset=17)
    K_local, K_s, N = ev.exact_batch(rho)
    K, Ks, Nstar = K_local[0], K_s[0], N[0]
    K_ii = K[np.ix_(ev.i_dofs, ev.i_dofs)]
    norm_N = np.linalg.norm(Nstar)

    rows: List[Dict[str, float]] = []
    for scale in (1.0e-1, 1.0e-2, 1.0e-3):
        E = rng.standard_normal(Nstar.shape)
        E *= scale * norm_N / np.linalg.norm(E)
        lhs = ev.reduced_stiffness(K, Nstar + E) - Ks
        rhs = E.T @ K_ii @ E
        rows.append({
            "eps_N": float(scale),
            "closed_form_relative_deviation": float(
                np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
            ),
            "min_eigenvalue_of_error": float(
                np.linalg.eigvalsh(0.5 * (lhs + lhs.T))[0]
            ),
            "relative_min_eigenvalue_of_error": float(
                np.linalg.eigvalsh(0.5 * (lhs + lhs.T))[0] / np.linalg.norm(Ks)
            ),
        })

    exact_err = float(
        np.linalg.norm(ev.reduced_stiffness(K, Nstar) - Ks) / np.linalg.norm(Ks)
    )
    return {"eq17_at_exact_N_relative_error": exact_err, "identity_checks": rows}


def step2_controlled_sweep(ev: Eq17Evaluator, n_dir: int = 8) -> Dict[str, Any]:
    """步骤 2: 受控扰动扫描, 拟合理论二阶斜率 (期望为 2.00)."""
    rng = np.random.default_rng(101)
    rho = sample_random_density(1, seed_offset=17)
    K_local, K_s, N = ev.exact_batch(rho)
    K, Ks, Nstar = K_local[0], K_s[0], N[0]
    norm_N = np.linalg.norm(Nstar)
    norm_Ks = np.linalg.norm(Ks)

    # 各扰动幅值复用相同方向, 避免方向变化污染二阶斜率.
    directions = rng.standard_normal((n_dir, *Nstar.shape))
    directions *= norm_N / np.linalg.norm(directions, axis=(-2, -1))[:, None, None]
    points: List[Dict[str, float]] = []
    for eps in np.logspace(-4.0, -0.5, 12):
        vals = []
        for direction in directions:
            E = eps * direction
            vals.append(
                np.linalg.norm(ev.reduced_stiffness(K, Nstar + E) - Ks) / norm_Ks
            )
        points.append({"eps_N": float(eps), "eps_K17": float(np.mean(vals))})

    x = np.log10([p["eps_N"] for p in points])
    y = np.log10([p["eps_K17"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "sweep_points": points,
        "loglog_slope": float(slope),
        "amplification_C": float(10.0 ** intercept),
    }


def train_shape_function_net(
    ev: Eq17Evaluator, n_train: int, n_epochs: int, learning_rate: float,
    hidden_dim: int = SHAPE_HIDDEN_DIM,
) -> Tuple[ShapeFunctionSurrogateNet, float]:
    """步骤 3: 训练形函数代理网络 (针对当前接口空间维度)."""
    X, Y = prepare_training_arrays(ev, n_train)
    net = ShapeFunctionSurrogateNet(
        input_dim=N_FINE[0] * N_FINE[1],
        output_dim=ev.n_i * ev.n_reduced,
        hidden_dims=(hidden_dim, hidden_dim),
    )
    losses = fit_full_batch(net, X, Y, n_epochs, learning_rate)
    return net, losses[-1]


class _CorruptedNet(nn.Module):
    def __init__(self, net: nn.Module, mode: str, scale: float = 1.0) -> None:
        super().__init__()
        self.net = net
        self.mode = mode
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.mode == "scale":
            return y * self.scale
        if self.mode == "nan":
            return y * float("nan")
        return y[..., :-1]


def step5_gate_calibration(
    ev: Eq17Evaluator, net: ShapeFunctionSurrogateNet, n_eval: int
) -> Dict[str, Any]:
    """步骤 5: 门禁阈值标定与故障注入测试."""
    rho = sample_random_density(n_eval, seed_offset=555)
    K_local, _, _ = ev.exact_batch(rho)

    c = make_condensor(ev, ev.prototype.i_dofs, ev.prototype.b_dofs, net)
    rows: List[Dict[str, float]] = []
    for k in range(n_eval):
        c.condense(bm.asarray(K_local[k]), rho[k])
        rows.append(dict(c.gate_report))
        if c.used_fallback:
            rows[-1]["fallback"] = 1.0

    excess = np.array([r.get("excess_ratio", np.nan) for r in rows])
    rigid = np.array([r.get("rigid_residual", np.nan) for r in rows])
    rcond = np.array([r.get("reduced_rcond", np.nan) for r in rows])

    faults: Dict[str, Any] = {}
    for tag, mode, scale in (
        ("scale_x3", "scale", 3.0),
        ("scale_x10", "scale", 10.0),
        ("nan", "nan", 1.0),
    ):
        bad = make_condensor(
            ev, ev.prototype.i_dofs, ev.prototype.b_dofs,
            _CorruptedNet(net, mode, scale),
        )
        bad.condense(bm.asarray(K_local[0]), rho[0])
        faults[tag] = {
            "used_fallback": bool(bad.used_fallback),
            "excess_ratio": bad.gate_report.get("excess_ratio"),
        }

    contract_raised = False
    try:
        bad = make_condensor(
            ev, ev.prototype.i_dofs, ev.prototype.b_dofs,
            _CorruptedNet(net, "truncate"),
        )
        bad.condense(bm.asarray(K_local[0]), rho[0])
    except SurrogateContractError:
        contract_raised = True
    faults["truncate_raises_contract_error"] = contract_raised

    boundary: Dict[str, Any] = {"flip_scale": None}
    for scale in np.arange(1.05, 3.01, 0.05):
        probe = make_condensor(
            ev, ev.prototype.i_dofs, ev.prototype.b_dofs,
            _CorruptedNet(net, "scale", float(scale)),
        )
        probe.condense(bm.asarray(K_local[0]), rho[0])
        if probe.used_fallback:
            boundary["flip_scale"] = float(scale)
            boundary["flip_excess_ratio"] = probe.gate_report.get("excess_ratio")
            break
        boundary["last_pass_scale"] = float(scale)
        boundary["last_pass_excess_ratio"] = probe.gate_report.get("excess_ratio")

    ex_finite = excess[np.isfinite(excess)]
    ex_max = float(ex_finite.max()) if len(ex_finite) else float("nan")
    ex_mean = float(ex_finite.mean()) if len(ex_finite) else float("nan")
    rg_finite = rigid[np.isfinite(rigid)]
    rg_max = float(rg_finite.max()) if len(rg_finite) else float("nan")
    rc_finite = rcond[np.isfinite(rcond)]
    rc_min = float(rc_finite.min()) if len(rc_finite) else float("nan")

    return {
        "n_eval": int(n_eval),
        "boundary_scan": boundary,
        "excess_ratio_max": ex_max,
        "excess_ratio_mean": ex_mean,
        "excess_rtol": c.excess_rtol,
        "excess_margin_factor": float(c.excess_rtol / max(ex_max, 1e-300)) if np.isfinite(ex_max) else float("nan"),
        "rigid_residual_max": rg_max,
        "rigid_tol": c.rigid_tol,
        "reduced_rcond_min": rc_min,
        "rcond_min": c.rcond_min,
        "n_fallback_holdout": int(sum(1 for r in rows if "fallback" in r)),
        "fault_injection": faults,
    }


def evaluate_validation(result: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """汇总数学验收与用户指定的网络精度验收.

    Parameters
    ----------
    result : dict
        已计算的解析、留出和解层指标.
    args : argparse.Namespace
        CLI 配置, 精度阈值使用相对误差小数.

    Returns
    -------
    dict
        各检查的数值、阈值、通过状态及总状态.
        未设置精度阈值时不宣称网络精度通过验收.
    """
    checks: Dict[str, Any] = {}

    def check(name: str, value: float, lower: float, upper: Optional[float]) -> None:
        """记录有限值区间检查."""
        checks[name] = {
            "value": float(value),
            "lower": lower,
            "upper": upper,
            "passed": bool(np.isfinite(value) and value >= lower and (upper is None or value <= upper)),
        }

    for key in (
        "rigid_part_density_independence_max",
        "rigid_part_analytic_deviation_max",
        "eq17_at_exact_N_relative_error",
    ):
        check(key, result[key], 0.0, 1.0e-10)
    for index, identity in enumerate(result["identity_checks"]):
        check(
            f"identity_{index}_relative_deviation",
            identity["closed_form_relative_deviation"], 0.0, 1.0e-7,
        )
        check(
            f"identity_{index}_relative_min_eigenvalue",
            identity["relative_min_eigenvalue_of_error"], -1.0e-10, None,
        )
    check("loglog_slope", result["loglog_slope"], 1.98, 2.02)

    if not args.skip_train:
        solution = result["solution_layer"]
        finite_values = [
            result[key] for key in (
                "final_train_mse", "eps_N_mean", "eps_N_max", "eps_N_p95",
                "eps_K17_mean", "eps_K17_max", "eps_K17_p95",
            )
        ] + [
            value for value in solution.values()
            if isinstance(value, (int, float))
        ]
        check("finite_training_and_solution_metrics", float(
            all(np.isfinite(value) for value in finite_values)
        ), 1.0, 1.0)
        check(
            "predicted_rigid_constraint",
            result["rigid_constraint_deviation_max"], 0.0, 1.0e-10,
        )
        check(
            "predicted_relative_min_eigenvalue",
            result["eq17_error_relative_min_eigenvalue_min"], -1.0e-10, None,
        )
        for key in ("trace_recovery_relative_error", "trace_stiffness_relative_error"):
            check(key, solution[key], 0.0, 1.0e-10)
        for key in ("equilibrium_relative_residual_max", "constraint_relative_residual_max"):
            if key in solution:
                check(key, solution[key], 0.0, 1.0e-8)
        if "gate_calibration" in result:
            faults = result["gate_calibration"]["fault_injection"]
            check("nan_prediction_falls_back", float(faults["nan"]["used_fallback"]), 1.0, 1.0)
            check("truncated_prediction_raises", float(
                faults["truncate_raises_contract_error"]
            ), 1.0, 1.0)

    precision: Dict[str, Any] = {}
    if not args.skip_train:
        candidates = (
            ("ks", args.max_ks_error, (
                result["eps_K17_max"], solution["in_service_ks_relative_error_max"],
            )),
            ("displacement", args.max_displacement_error, (
                solution["interface_displacement_relative_error"],
                solution["displacement_relative_error"],
            )),
            ("compliance", args.max_compliance_error, (
                solution["compliance_relative_error"],
            )),
        )
        for name, limit, values in candidates:
            if limit is not None:
                precision[name] = {
                    "values": list(values),
                    "limit": limit,
                    "passed": bool(all(
                        np.isfinite(value) and 0.0 <= value <= limit for value in values
                    )),
                }

    mathematics_passed = all(item["passed"] for item in checks.values())
    precision_passed = all(item["passed"] for item in precision.values())
    return {
        "mathematics": {"passed": mathematics_passed, "checks": checks},
        "precision": {
            "status": ("passed" if precision_passed else "failed") if precision else "not_requested",
            "checks": precision,
        },
        "passed": mathematics_passed and precision_passed,
    }


def run_verification(args) -> None:
    bm.set_backend("numpy")
    set_random_seed(args.seed)

    ev = Eq17Evaluator(trace_basis=args.trace_basis)
    mode_name = "full_trace (未降阶完整接口, 40 自由度)" if args.trace_basis == "full_trace" else \
                "linear_corner (角点线性迹降阶接口, 8 自由度, 式 16)"

    print("\n【配置摘要】")
    print(f"组别 / 路线 : {1 if args.trace_basis == 'full_trace' else 2} / shape_function")
    print(f"问题 / 材料 : FullMBBBeam2d, domain={DOMAIN_SIZE}, "
          f"{ev.prototype.material.hypothesis}, E0={E_BASE:g}, nu={NU:g}, SIMP penalty={SHAPE_SIMP_PENALTY:g}")
    print(f"载荷        : 顶边中点竖向集中力 P={P_LOAD:g}")
    print(f"网格 / 接口 : Q1, 子结构={N_SUB}, 每块细单元={N_FINE}, "
          f"{args.trace_basis}; 每块内部={ev.n_i}, 接口={ev.n_interface}")
    print(f"全局细网格  : {N_SUB[0] * N_FINE[0]} x {N_SUB[1] * N_FINE[1]} 单元")
    print(f"网络        : {N_FINE[0] * N_FINE[1]} -> {args.hidden_dim} -> "
          f"{args.hidden_dim} -> {ev.n_i * ev.n_reduced}, SiLU")
    if args.skip_train:
        print("训练 / 留出 : 未执行 (--skip-train); 全局解层未执行")
    else:
        print(f"训练        : {args.n_train} 样本, {args.epochs} epochs, Adam, lr={args.lr:g}, full-batch")
        print(f"目标 / 留出 : 形函数变形分量 M 的 MSE / {args.n_eval} 样本")
    print(f"密度        : 训练和留出为随机场 {DENSITY_RANGE}; 在役为光滑场")
    print(f"种子 / 求解 : 初始化={args.seed}, 训练采样={SHAPE_SEED}, "
          f"留出采样={SHAPE_SEED + 555}, backend=numpy, scipy")
    print(f"比较基准    : 同网格、密度、载荷与支承的精确 {args.trace_basis} 缩聚", flush=True)

    if args.verbose:
        print("=" * 78)
        print(f"Huang 2023 式 (17): 形函数路径实测 ({mode_name})")
        print("=" * 78)
        print(f"子结构        : {N_FINE[0]}x{N_FINE[1]} Q1, "
              f"n_i={ev.n_i}, 接口维度 n_b={ev.n_interface}, n_dof={ev.n_dof}")
        print(f"刚体/变形子空间: n_rigid={ev.n_rigid}, 变形维数 m={ev.n_reduced}")
        print(f"网络输出维    : {ev.n_i * ev.n_reduced}")
        print(f"随机数种子    : {args.seed}")
        print("-" * 78)

    result: Dict[str, Any] = {
        "trace_basis": args.trace_basis,
        "substructure": {
            "n_fine": list(N_FINE), "n_i": ev.n_i, "n_interface": ev.n_interface,
            "n_rigid": ev.n_rigid, "n_reduced": ev.n_reduced,
            "network_output_dim": ev.n_i * ev.n_reduced,
        },
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "density_range": list(DENSITY_RANGE),
        "skip_train": args.skip_train,
        "comparison_reference": f"exact_{args.trace_basis}",
    }

    r0 = step0_rigid_part(ev)
    result.update(r0)
    if args.verbose:
        print("\n[0] 刚体分量的密度无关性与解析构造")
        print(f"    max |Phi(rho_k) - Phi(rho_0)| / |Phi(rho_0)| = "
              f"{r0['rigid_part_density_independence_max']:.3e}")
        print(f"    max |Phi(rho_k) - Phi_analytic| / |Phi_analytic| = "
              f"{r0['rigid_part_analytic_deviation_max']:.3e}")

    r1 = step1_identity(ev)
    result.update(r1)
    if args.verbose:
        print("\n[1] 式 (17) 误差闭式  K_tilde(N*+E) - K_s == E^T K_ii E")
        print(f"    精确 N 代入式 (17) 的相对误差 : "
              f"{r1['eq17_at_exact_N_relative_error']:.3e}")
        print(f"    {'eps_N':>8} {'闭式相对偏差':>16} {'误差阵最小特征值':>18}")
        for row in r1["identity_checks"]:
            print(f"    {row['eps_N']:>8.0e} "
                  f"{row['closed_form_relative_deviation']:>16.3e} "
                  f"{row['min_eigenvalue_of_error']:>18.3e}")

    r2 = step2_controlled_sweep(ev)
    result.update(r2)
    if args.verbose:
        print("\n[2] 受控扰动扫描")
        print(f"    log-log 斜率  = {r2['loglog_slope']:.4f}  (理论值 2)")
        print(f"    放大系数 C    = {r2['amplification_C']:.4f}  "
              f"即 eps_K17 ~ C * eps_N^2")

    if args.skip_train:
        if args.verbose:
            print("\n[3][4] --skip-train: 跳过网络训练与解层验证")
    else:
        if args.verbose:
            print(f"\n[3] 训练形函数网络 (输出维={ev.n_i * ev.n_reduced}) "
                  f"(n_train={args.n_train}, epochs={args.epochs}, lr={args.lr})")
        net, final_loss = train_shape_function_net(
            ev, args.n_train, args.epochs, args.lr, args.hidden_dim
        )
        result["final_train_mse"] = final_loss
        result["n_train"] = args.n_train
        result["n_epochs"] = args.epochs
        result["learning_rate"] = args.lr
        if args.verbose:
            print(f"    最终训练 MSE  = {final_loss:.6e}")

        r3 = step3_trained_network(ev, net, args.n_eval)
        result.update(r3)
        C = r2["amplification_C"]
        predicted = C * r3["eps_N_mean"] ** 2
        result["eps_K17_predicted_from_sweep"] = float(predicted)

        if args.verbose:
            print(f"\n    留出集 ({r3['n_eval']} 样本):")
            print(f"      形函数误差   eps_N    : "
                  f"mean {r3['eps_N_mean'] * 100:.3f}%  max {r3['eps_N_max'] * 100:.3f}%")
            print(f"      式(17) 后    eps_K17  : "
                  f"mean {r3['eps_K17_mean'] * 100:.4f}%  "
                  f"max {r3['eps_K17_max'] * 100:.4f}%")
            print(f"      由 C*eps_N^2 预测      : {predicted * 100:.4f}%")
            print(f"      降幅                   : "
                  f"{r3['eps_N_mean'] / max(r3['eps_K17_mean'], 1e-300):.1f} 倍")
            print(f"      刚体约束偏差 (构造性)  : "
                  f"{r3['rigid_constraint_deviation_max']:.3e}")
            print(f"      式(17) 误差阵最小特征值: "
                  f"{r3['eq17_error_relative_min_eigenvalue_min']:.3e} (应 >= 0)")

        r4 = step4_solution_layer(ev, net)
        result["solution_layer"] = r4
        if args.verbose:
            print(f"\n[4] 解层: 完整 MBB 梁 (12x2 子结构), {args.trace_basis} 接口")
            print(f"      在役 K_s 误差 (max/mean) : "
                  f"{r4['in_service_ks_relative_error_max'] * 100:.4f}% / "
                  f"{r4['in_service_ks_relative_error_mean'] * 100:.4f}%")
            print(f"      接口位移相对误差         : "
                  f"{r4['interface_displacement_relative_error'] * 100:.4f}%")
            print(f"      全场位移相对误差         : "
                  f"{r4['displacement_relative_error'] * 100:.4f}%")
            print(f"      柔度 精确 / 本路径       : "
                  f"{r4['compliance_exact']:.6f} / {r4['compliance_route']:.6f}")
            print(f"      柔度相对误差             : "
                  f"{r4['compliance_relative_error'] * 100:.4f}%")
            print("      全局解层门禁             : 未启用")

        # 仅 full_trace 支持完整的式 17 门禁标定步骤
        if args.trace_basis == "full_trace":
            r5 = step5_gate_calibration(ev, net, args.n_eval)
            result["gate_calibration"] = r5
            if args.verbose:
                print("\n[5] 门禁标定与故障注入")
                print(f"      留出集刚化幅度 max/mean  : "
                      f"{r5['excess_ratio_max']:.3e} / {r5['excess_ratio_mean']:.3e}")
                print(f"      阈值 excess_rtol         : {r5['excess_rtol']:.3e}  "
                      f"(裕度 {r5['excess_margin_factor']:.1f} 倍)")
                print(f"      刚体残量 max / 阈值      : "
                      f"{r5['rigid_residual_max']:.3e} / {r5['rigid_tol']:.3e}")
                print(f"      条件数 min / 阈值        : "
                      f"{r5['reduced_rcond_min']:.3e} / {r5['rcond_min']:.3e}")
                print(f"      留出集回退数             : "
                      f"{r5['n_fallback_holdout']}/{r5['n_eval']}")
                bs = r5["boundary_scan"]
                print(f"      门禁翻转扫描 : {bs}")
                print("      故障注入 (NaN 必须回退, 缩放仅诊断):")
                for tag, row in r5["fault_injection"].items():
                    if tag == "truncate_raises_contract_error":
                        print(f"        {tag:24s}: {row}")
                    else:
                        print(f"        {tag:24s}: 回退={row['used_fallback']}  "
                              f"刚化幅度={row['excess_ratio']}")

    validation = evaluate_validation(result, args)
    result["validation"] = validation
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_analytic" if args.skip_train else ""
    out_path = out_dir / f"eq17_second_order_{args.trace_basis}{suffix}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n【公共误差表】")
    def row(label: str, value: str) -> None:
        """按终端显示宽度打印单项指标."""
        print(f"{label}{' ' * max(1, 46 - display_width(label))} : {value}")

    if args.skip_train:
        for label in (
            "留出刚度误差 (mean / max)", "在役刚度误差 (mean / max)",
            "接口位移相对误差", "全场位移相对误差",
            "柔度 (精确 / 代理)", "柔度相对误差",
        ):
            row(label, "未执行")
    else:
        row("留出刚度误差 (mean / max)", f"{r3['eps_K17_mean']:.2%} / {r3['eps_K17_max']:.2%}")
        row("在役刚度误差 (mean / max)", f"{r4['in_service_ks_relative_error_mean']:.2%} / {r4['in_service_ks_relative_error_max']:.2%}")
        row("接口位移相对误差", f"{r4['interface_displacement_relative_error']:.2%}")
        row("全场位移相对误差", f"{r4['displacement_relative_error']:.2%}")
        row("柔度 (精确 / 代理)", f"{r4['compliance_exact']:.8f} / {r4['compliance_route']:.8f}")
        row("柔度相对误差", f"{r4['compliance_relative_error']:.2%}")
    print("\n【路线专项诊断】")
    row("受控扰动 log-log 斜率 (理论 2)", f"{r2['loglog_slope']:.4f}")
    row("精确形函数代入变分式的相对误差", f"{r1['eq17_at_exact_N_relative_error']:.4e}")
    row("刚体解析分量相对偏差", f"{r0['rigid_part_analytic_deviation_max']:.4e}")
    if args.skip_train:
        row("形函数误差 (mean / max)", "未执行")
    else:
        row("最终训练 MSE", f"{final_loss:.4e}")
        row("形函数误差 (mean / max)", f"{r3['eps_N_mean']:.2%} / {r3['eps_N_max']:.2%}")
        row("形函数 / 刚度平均相对误差之比", f"{r3['eps_N_mean'] / max(r3['eps_K17_mean'], 1e-300):.2f}")
        row("预测形函数刚体约束相对偏差", f"{r3['rigid_constraint_deviation_max']:.4e}")
    print("\n【门禁与结果文件】")
    print("全局解层 : 未执行" if args.skip_train else "全局解层 : 已评估原始代理, 回退门禁未启用")
    if not args.skip_train and args.trace_basis == "full_trace":
        print(f"留出门禁 : 独立诊断已执行, 回退 {r5['n_fallback_holdout']}/{r5['n_eval']}")
        print("公共误差表使用原始代理预测; 独立门禁回退不参与该表.")
    else:
        print("留出门禁 : 未执行" + (" (--skip-train)" if args.skip_train else " (linear_corner 尚未接入标定)"))
    print("数学验收 : " + ("通过" if validation["mathematics"]["passed"] else "失败"))
    precision_status = validation["precision"]["status"]
    print("精度验收 : " + (
        "未执行 (未设置精度阈值, 不代表网络精度通过)"
        if precision_status == "not_requested" else
        ("指定指标通过" if precision_status == "passed" else "失败")
    ))
    for name, item in validation["precision"]["checks"].items():
        print(f"  {name}: max={max(item['values']):.4e}, limit={item['limit']:.4e}")
    print(f"[证据] 结果已写入: {out_path}")
    if not validation["passed"]:
        failed = [
            name for name, item in validation["mathematics"]["checks"].items()
            if not item["passed"]
        ] + [
            name for name, item in validation["precision"]["checks"].items()
            if not item["passed"]
        ]
        raise AssertionError("验收失败: " + ", ".join(failed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huang 2023 式 (17) 二维能力验证 (支持 full_trace 与 linear_corner)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--trace-basis", "--interface", choices=("full_trace", "linear_corner"),
        default="full_trace", dest="trace_basis",
        help="子结构接口迹空间模式. 默认 full_trace (组 1); 可选 linear_corner (组 2, 8 角点自由度).",
    )
    parser.add_argument("--n-train", type=int, default=SHAPE_N_TRAIN,
                        help="随机密度训练样本数, 缺省对齐 verify_stiffness_route.py")
    parser.add_argument("--epochs", type=int, default=SHAPE_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=SHAPE_LEARNING_RATE, help="Adam 学习率")
    parser.add_argument("--n-eval", type=int, default=SHAPE_N_EVAL, help="留出评估样本数")
    parser.add_argument("--hidden-dim", type=int, default=SHAPE_HIDDEN_DIM, help="网络隐藏层宽度")
    parser.add_argument("--seed", type=int, default=SHAPE_SEED, help="网络初始化种子; 密度采样保持配置中的固定种子")
    parser.add_argument("--verbose", action="store_true", help="打印详细开发期自检与故障注入日志")
    parser.add_argument("--skip-train", action="store_true",
                        help="只跑 [0][1][2] 三步解析验证, 不训练网络, 不进入解层")
    parser.add_argument("--max-ks-error", type=float, default=None,
                        help="可选留出/在役最大刚度相对误差上限, 0.01 表示 1%%")
    parser.add_argument("--max-displacement-error", type=float, default=None,
                        help="可选接口/全场位移相对误差上限")
    parser.add_argument("--max-compliance-error", type=float, default=None,
                        help="可选柔度相对误差上限")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="证据输出目录, 缺省为本脚本同级 outputs/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("n_train", "n_eval", "epochs", "lr", "hidden_dim"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " 必须为有限正数.")
    for name in ("max_ks_error", "max_displacement_error", "max_compliance_error"):
        value = getattr(args, name)
        if value is not None:
            if not np.isfinite(value) or value < 0.0:
                parser.error(name.replace("_", "-") + " 必须为有限非负数.")
            if args.skip_train:
                parser.error("--skip-train 不执行网络精度评估, 不能设置精度阈值.")
    run_verification(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
