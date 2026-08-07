# 胡张混合有限元 2D 算例数学规范 (Math Spec)

本文件把 `examples/huzhang_elasticity` 的数学构造与代码符号 1-to-1 固定下来。
**完整数学过程**（混合弱形式、鞍点系统、低阶稳定化的数学格式与缩放律、收敛阶结果）
见知识库概念页 `C:\workspace\dut-postdoc\concepts\huzhang-mixed-fem.md`，
本文档只保留符号—代码映射，不复述数学公式。

> **理论事实源**（本文档只做映射）：
>
> - 完整数学过程与公式定义（弱形式 (2)–(3)、鞍点系统 (4)、稳定化 (5)–(8)、
>   收敛阶结果 §5）：`C:\workspace\dut-postdoc\concepts\huzhang-mixed-fem.md`
> - 制造解完整数学定义：[制造解文档](../../docs/problems/manufactured-elasticity.md)
> - 低阶稳定化与 $H(\mathrm{div})$ 降阶归因：`brightPhD.pdf` §5.4.2–5.4.3

---

## 1. 符号—代码映射表 (Symbol-to-Code Mapping)

各符号的数学定义见概念页（§2 混合弱形式、§4 低阶稳定化），此处只给代码位置。

### 1.1 混合弱形式与鞍点系统

| 数学符号 | 代码位置 | 含义 |
|---|---|---|
| 应力次数 $k$ | `HuZhangMFEMAnalyzer(space_degree=k)` → `HuZhangFESpace(p=k)` | $k\le2$ 时装配稳定化 |
| $\Sigma_h$ | `HuZhangFESpace` | 应力空间（$H(\mathrm{div})$) |
| $V_h$ | `TensorFunctionSpace(scalar_space=..., shape=(-1, GD))` | 位移空间（不连续 Lagrange） |
| $A$ | `HuZhangStressIntegrator(lambda0, lambda1, method='fast')` | 柔度矩阵块 |
| $B$ | `HuZhangMixIntegrator` / `_cached_mix_matrix` | 应力—位移耦合块 |
| 边界语义 | `_essential_bc`（traction Γ_N）/ `_natural_bc`（位移 Γ_D） | traction 本质强加，位移弱加进右端项 |

### 1.2 低阶稳定化

| 数学符号 | 代码位置 | 含义 |
|---|---|---|
| $[[\boldsymbol w]]$（矩阵跳量） | [`jump_penalty_integrator.py:_fetch_matrix_jump`](../../src/soptx/fem/integrators/jump_penalty_integrator.py) 的 `M_R`/`M_L`/`M` | 对称梯度型跳量，定义见概念页 (6) |
| $\int_F[[\phi_i]]:[[\phi_j]]\,\mathrm ds$ | `integrand = einsum('q, f, fqikl, fqjkl -> fij', ws, fm, matrix_jump, matrix_jump)` | 面测度 $f_m$ 已乘入 |
| $h_F$ | `mesh.entity_measure('face')`（2D 即边长） | 面特征尺度 |
| $\alpha=\mu/L_0^2$ | `assembly`: `alpha = mu / L0 ** 2` | 物理量纲缩放系数 |
| $L_0$ | `L0 = max(bbox_max - bbox_min)`（`mesh.entity('node')`） | 计算域特征尺度（单位域上 $L_0=1$） |
| $c(\cdot,\cdot)$ | `JumpPenaltyIntegrator(q=..., threshold=valid_faces_idx, method='matrix_jump', material=..., penalty_scaling='physical_h')` | 面跳量惩罚双线性型 |
| $\mathcal F_h$ | `assemble_stiff_matrix`: `valid_faces_bool = is_internal \| is_dirichlet` | 内部面 + 位移边界面，不含 $\Gamma_N$ |
| $-J$ 进入 $(2,2)$ 块 | `bmat([[A, B], [B.T, -J]])`（scipy bmat 构造） | 稳定化取负号 |

缩放律选择（`penalty_scaling`，默认 `'physical_h'`）：

| 取值 | 系数 | 代码分支 | 性质 |
|---|---|---|---|
| `'physical_h'`（默认） | $\alpha\cdot h_F$ | `assembly`: `alpha * hF` | 论文式物理量纲缩放，$h_F$ 幂次 $+1$ |
| `'gamma_hinv'` | $\gamma/h_F$，$\gamma=0.01E$（$k=1$）/ $0.01\mu$（$k\ge2$） | `assembly`: `gamma * hF ** -1` | 旧缩放，仅作回归 |

---

## 2. 数值验收标准 (Acceptance Criteria)

判据只取无歧义的两项，阈值沿用 `experiments/huzhang_topopt_paper/cases.toml` 的 acceptance：
相对平衡残差与状态矩阵对称性缺陷。**收敛阶只打印不判定**（`theory-audit-required`），不作为通过条件。

| 判据 | 数学定义 | 阈值 | 代码位置 |
|---|---|---|---|
| 相对平衡残差 | $\|\mathbf K\mathbf x-\mathbf f\|_2\big/\max(\|\mathbf f\|_2,\epsilon)$ | $<10^{-8}$ | `analyzer.relative_state_residual()` |
| 状态矩阵对称性缺陷 | $\|\mathbf K-\mathbf K^{\mathsf T}\|_F\big/\max(\|\mathbf K\|_F,\epsilon)$（相对 Frobenius 缺陷） | $<10^{-12}$ | `analyzer.state_matrix_symmetry_error()` |

收敛阶结果（高阶 $k\ge3$ 应力 $h^{k+1}$；低阶表 5.2 的 $k=1$ 为 1/1.5/1、$k=2$ 为 2/2/1 含 $H(\mathrm{div})$ 降阶）与归因见概念页 §5。
实测在 `MixedBoundarySinusoidalElasticity2D` 上逐格复现论文表 5.2，见
[results_analysis.md 1.1](outputs/results_analysis.md)。

---

## 3. 相关文档

- 使用说明与 CLI：`README.md`
- 实测数据、判据与诊断：`outputs/results_analysis.md`
- 制造解数学定义：[`docs/problems/manufactured-elasticity.md`](../../docs/problems/manufactured-elasticity.md)
- 完整数学过程：`C:\workspace\dut-postdoc\concepts\huzhang-mixed-fem.md`
