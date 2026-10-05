# PIML 子结构力学验证结果分析

本页是 [`verify_shape_function_route.py`](verify_shape_function_route.py) 与 [`verify_stiffness_route.py`](verify_stiffness_route.py) 的数学—代码映射契约与验收契约。运行入口、参数说明与论文对应边界见 [`README.md`](README.md)。本目录不保存实测证据，证据见第 3 节。

## 1. 数学—代码映射契约

### 1.1 算例与维数

两个脚本共用同一算例：`FullMBBBeam2d`，`DOMAIN = (0.0, 12.0, 0.0, 2.0)`，`N_SUB = (12, 2)`，`N_FINE = (5, 5)`，`P_LOAD = -1.0`，`E_BASE = 1.0`，`NU = 0.3`，`DENSITY_RANGE = (0.3, 1.0)`。训练与留出密度在 `DENSITY_RANGE` 上逐单元独立均匀采样；在役密度由 `make_density_fields` 生成光滑场。形函数路线的采样用脚本内 `sample_random_density`（`np.random.default_rng(SHAPE_SEED + offset)`），`--seed` 只影响网络初始化；刚度路线用 `soptx.fem.substructure.sample_random_density`，`--seed` 同时固定采样与初始化。比较基准一律为同接口空间的 `ExactSchurCondensation`，$\mathbf N = -\mathbf K_{ii}^{-1}\mathbf K_{ib}$，$\mathbf K_s = \mathbf K_{bb} - \mathbf K_{ib}^{\mathsf T}\mathbf K_{ii}^{-1}\mathbf K_{ib}$。

记迹矩阵 $\mathbf T \in \mathbb R^{n_b \times n_q}$，$\boldsymbol u_b = \mathbf T \boldsymbol q$；$n_{\mathrm{rigid}} = 3$，变形子空间维数 $m = n_q - n_{\mathrm{rigid}}$。

| 接口空间 | $\mathbf T$ 的代码 | $n_i$ | $n_q$ | $m$ | 形函数输出 $n_i m$ | 刚度输出 $m(m+1)/2$ |
|---|---|---:|---:|---:|---:|---:|
| `full_trace` | `trace=None`，即 $\mathbf T = \mathbf I$ | 32 | 40 | 37 | 1184 | 703 |
| `linear_corner` | `LinearCornerTraceBasis.from_prototype(prototype)`，矩阵即 `prototype.linear_boundary_matrix`（式 (16) 的 $\mathbf L$） | 32 | 8 | 5 | 160 | 未实现 |

### 1.2 形函数 $\mathbf N$ 路线

`Eq17Evaluator` 持有原型 `SubstructurePrototype(SUB_SIZE, N_FINE, ..., penal=3.0, rho_min=0.0)` 与无网络的核心缩聚器 `ShapeFunctionCondensation`（`self.core`）；脚本中所有 $\mathbf K_r$ 都经核心库构造，不另写变分式。

| 数学量 | 代码 | 形状 |
|---|---|---|
| $\mathbf R_q,\ \mathbf R_\perp,\ \boldsymbol\Phi_i$ | `prototype.trace_interface_bases(trace)` 的三个返回值 | $(n_q,3)$、$(n_q,m)$、$(32,3)$ |
| 精确迹空间量 $\mathbf K_r = \mathbf T^{\mathsf T}\mathbf K_s\mathbf T$，$\mathbf B = \mathbf N\mathbf T$ | `trace.project_stiffness`、`trace.reduce_recovery`（`Eq17Evaluator.exact_batch`） | $(n_q,n_q)$、$(32,n_q)$ |
| 训练目标 $\mathbf M = \mathbf B\mathbf R_\perp$ | `ShapeFunctionCondensation.project_deformation` | $(32,m)$ |
| 延拓合成 $\widehat{\mathbf B} = \boldsymbol\Phi_i\mathbf R_q^{\mathsf T} + \widehat{\mathbf M}\mathbf R_\perp^{\mathsf T}$ | `assemble_recovery`，构造性保证 $\widehat{\mathbf B}\mathbf R_q = \boldsymbol\Phi_i$ | $(32,n_q)$ |
| 式 (17) $\widetilde{\mathbf K}_r = (\mathbf H^{\mathsf T})^{\mathsf T}\mathbf K\,\mathbf H^{\mathsf T}$，$\mathbf H^{\mathsf T} = [\mathbf B;\mathbf T]$ | `assemble_reduced_stiffness` → `_variational_stiffness`，展开为 $\mathbf T^{\mathsf T}\mathbf K_{bb}\mathbf T + \mathbf A + \mathbf A^{\mathsf T} + \mathbf B^{\mathsf T}\mathbf K_{ii}\mathbf B$，$\mathbf A = (\mathbf K_{ib}\mathbf T)^{\mathsf T}\mathbf B$ | $(n_q,n_q)$ |
| 网络 $\boldsymbol\rho \mapsto \operatorname{vec}\widehat{\mathbf M}$ | `ShapeFunctionSurrogateNet(input_dim=prototype.n_cells, output_dim=n_i * n_reduced, hidden_dims=(256, 256))`，默认 SiLU | $25 \to n_i m$ |
| 训练 | `fit_full_batch`：全批量 Adam + `nn.MSELoss`，`SHAPE_N_TRAIN = 2000`、`SHAPE_EPOCHS = 4000`、`SHAPE_LEARNING_RATE = 0.005` | — |

式 (17) 的误差结构：记 $\mathbf B = \mathbf B^\ast + \mathbf E$，一阶项被 $\mathbf K_{ii}\mathbf B^\ast = -\mathbf K_{ib}\mathbf T$ 抵消，

$$
\widetilde{\mathbf K}_r(\mathbf B^\ast + \mathbf E) - \mathbf K_r = \mathbf E^{\mathsf T}\mathbf K_{ii}\mathbf E \succeq 0 .
$$

脚本各步与该结构的对应：

| 步骤 | 函数 | 计算量 |
|---|---|---|
| 0 | `step0_rigid_part` | 8 个随机密度下 $\mathbf B\mathbf R_q$ 相对首样本及相对 $\boldsymbol\Phi_i$ 的偏差 |
| 1 | `step1_identity` | $\mathbf B^\ast$ 代入式 (17) 的相对误差；$\varepsilon_N \in \{10^{-1},10^{-2},10^{-3}\}$ 下上式两侧的相对偏差及误差阵最小特征值 / $\lVert\mathbf K_r\rVert_F$ |
| 2 | `step2_controlled_sweep` | 8 个固定方向、`np.logspace(-4.0, -0.5, 12)` 幅值，$\log\varepsilon_K$ 对 $\log\varepsilon_N$ 的拟合斜率与 $C = 10^{\text{截距}}$ |
| 3 | `step3_trained_network` | 留出集 $\varepsilon_N = \lVert\widehat{\mathbf B} - \mathbf B\rVert_F / \lVert\mathbf B\rVert_F$、$\varepsilon_K$（键名 `eps_K17_*`）、$\lVert\widehat{\mathbf B}\mathbf R_q - \boldsymbol\Phi_i\rVert / \lVert\boldsymbol\Phi_i\rVert$ |
| 4 | `step4_solution_layer` | MBB 梁上在役刚度、接口位移、全场位移与柔度误差 |
| 5 | `step5_gate_calibration`（仅 `full_trace`） | `gate_report` 的 `rigid_residual`、`excess_ratio`、`reduced_rcond`，对应缺省阈值 `rigid_tol = 1e-10`、`excess_rtol = 2e-2`、`rcond_min = 1e-8`；故障注入 |

第 4 步的接线：`full_trace` 用 `GlobalAssembler.assemble_interface_system` 求解；`linear_corner` 用 `build_linear_corner_projection`、`assemble_macro_system` 与 `solve_constrained_system`。预测结果以 `LocalReductionBatchResult` 交给 `recover_full_displacement`，其中完整边界恢复矩阵取 $\mathbf N_\partial = \widehat{\mathbf B}\mathbf L^{+}$（`np.linalg.pinv`），仅在 $\boldsymbol u_b = \mathbf L\boldsymbol q$ 上使用；脚本复核 $\mathbf N_\partial\mathbf L$ 与 $\mathbf L^{\mathsf T}\mathbf K_\partial\mathbf L$ 回到迹空间预测（`trace_recovery_relative_error`、`trace_stiffness_relative_error`）。解层使用原始预测，`gate_enabled` 恒为 `False`。

### 1.3 刚度 $\mathbf K_s$ 路线（仅 `full_trace`）

| 数学量 | 代码 | 形状 |
|---|---|---|
| $\mathbf R_{\mathrm{rigid}}$、$\mathbf R_\perp$ | `prototype.rigid_basis`、`prototype.deformation_basis`（与密度无关，QR 正交化） | $(40,3)$、$(40,37)$ |
| 训练目标 $\mathbf C = \operatorname{chol}(\mathbf R_\perp^{\mathsf T}\mathbf K_s\mathbf R_\perp)$ 的下三角条目 | `train_reduced_stiffness_surrogate`：`L_train[i][tril_mask]`，行优先，无正则 | $703$ |
| 网络 | `ReducedStiffnessSurrogateNet(input_dim=prototype.n_cells, output_dim=n_tril, hidden_dims=(128, 128))`，默认 SiLU | $25 \to 703$ |
| 训练 | `TrainingConfig(epochs, batch_size=n_train, optimizer_params={"lr": ...}, select_final_state=True)` + `train_surrogate`，Adam + MSE，全批量 | — |
| 重构 $\widehat{\mathbf K}_s = \mathbf R_\perp\mathbf C\mathbf C^{\mathsf T}\mathbf R_\perp^{\mathsf T}$ | `ReducedStiffnessCondensation(..., is_cholesky=True, range_basis=deformation_basis)`：`entry_mask` 写回，对角取绝对值 | $(40,40)$ |
| 门禁 | 预测有限且 $\min_k\lvert c_{kk}\rvert > $ `rcond_min` $\cdot\max_k\lvert c_{kk}\rvert$（`rcond_min = 1e-8`）；失败回退 `fallback_solver` 并置 `used_fallback`；输出维 $\neq$ `n_output` 抛 `SurrogateContractError`，不回退 | — |
| 内部恢复 $\mathbf N$ | 恒取精确值，代理只替换 $\mathbf K_s$ | $(32,40)$ |

由构造，$\widehat{\mathbf K}_s\mathbf R_{\mathrm{rigid}} = \mathbf 0$ 且 $\widehat{\mathbf K}_s$ 在变形子空间上正定。三项诊断：`verify_parameterization_parity` 用 `_ExactCholeskyStub` 送入精确 $\mathbf C$ 的 `float32` 条目，得到参数化误差上限；`evaluate_holdout` 在训练同分布留出集上走同一推理路径；`diagnose_rigid_mode_pollution` 取精确 $\mathbf K_s$ 最小 3 个特征向量 $\mathbf V$，报告 $\lambda_{\max}\lvert\mathbf V^{\mathsf T}\widehat{\mathbf K}_s\mathbf V\rvert$ 及其在精确接口位移上的能量份额。

## 2. 验收契约

### 2.1 形函数路线（`evaluate_validation`）

| 检查键 | 断言 | 生效条件 |
|---|---|---|
| `rigid_part_density_independence_max`、`rigid_part_analytic_deviation_max`、`eq17_at_exact_N_relative_error` | $\in [0, 10^{-10}]$ | 总是 |
| `identity_{k}_relative_deviation` | $\in [0, 10^{-7}]$ | 总是，$k = 0,1,2$ |
| `identity_{k}_relative_min_eigenvalue` | $\ge -10^{-10}$ | 总是 |
| `loglog_slope` | $\in [1.98, 2.02]$ | 总是 |
| `finite_training_and_solution_metrics` | 训练 MSE、$\varepsilon_N$、$\varepsilon_K$ 统计与解层数值全部有限 | 未 `--skip-train` |
| `predicted_rigid_constraint`、`trace_recovery_relative_error`、`trace_stiffness_relative_error` | $\in [0, 10^{-10}]$ | 未 `--skip-train` |
| `predicted_relative_min_eigenvalue` | $\ge -10^{-10}$ | 未 `--skip-train` |
| `equilibrium_relative_residual_max`、`constraint_relative_residual_max` | $\in [0, 10^{-8}]$ | `linear_corner` 且未 `--skip-train` |
| `nan_prediction_falls_back`、`truncated_prediction_raises` | NaN 输出回退；截断输出抛 `SurrogateContractError` | `full_trace` 且未 `--skip-train` |

`scale_x3`、`scale_x10` 与 `boundary_scan`（缩放 1.05 至 3.00，步长 0.05）只记录，不判定。

### 2.2 刚度路线（`build_validation`）

| 检查键 | 断言 | 生效条件 |
|---|---|---|
| `rigid_basis_residual` | $\lVert\mathbf K_s\mathbf R_{\mathrm{rigid}}\rVert_F / \lVert\mathbf K_s\rVert_F \le$ `RIGID_BASIS_RESIDUAL_TOL = 1.0e-10` | 总是 |
| `parameterization_error_ceiling` | $\le$ `PARAMETERIZATION_ERROR_TOL = 1.0e-4` | 总是 |
| `parameterization_zero_fallback` | 完美代理不回退 | 总是 |
| `finite_numeric_results` | 结果记录中无 NaN/Inf（`_find_nonfinite_numbers`） | 总是 |
| `active_zero_fallback`、`holdout_zero_fallback` | 在役与留出回退数均为 0 | `--strict` 或任一精度阈值 |

### 2.3 可选精度阈值

两个入口共用 `--max-ks-error`（留出最大 $\varepsilon_K$ 与在役最大刚度误差）、`--max-displacement-error`（接口与全场位移）、`--max-compliance-error`（柔度），取相对误差小数；未设置时 `precision.status` / `accuracy_status` 为 `not_requested`。形函数路线 `--skip-train` 时设置阈值即报参数错误。验收失败时先写 JSON（`eq17_second_order_{trace_basis}[_analytic].json`、`piml_exact_comparison.json`）再抛 `AssertionError`；刚度路线的 `piml_exact_comparison.png` 只在验收通过后生成。

## 3. 实验证据

本目录不保留实测证据。两条路线的已登记数值集中在 [`experiments/analysis_capability_piml_substructure/results_analysis.md` 第 7 节](../../experiments/analysis_capability_piml_substructure/results_analysis.md#7-2-隐藏层网络)，原示例报告归档于 [`legacy_examples_results.md`](../../experiments/analysis_capability_piml_substructure/legacy_examples_results.md)（该档声明“不代表当前代码已重新验证”）。

| 主题 | 位置 | 该处记录（原文摘录） |
|---|---|---|
| 构造正确性 | [§7.2](../../experiments/analysis_capability_piml_substructure/results_analysis.md#72-不依赖训练的构造正确性验证) | 扰动扫描斜率 `full_trace` `2.0030`、`linear_corner` `2.0026`；刚体解析分量相对偏差 `3.96e-16` / `1.193`（标注“待查”）；Cholesky 参数化误差上限 `3.84e-08`，刚体基残差 `8.77e-17` |
| 留出集精度 | [§7.3](../../experiments/analysis_capability_piml_substructure/results_analysis.md#73-网络训练结果与留出集精度) | 形函数路线 $\varepsilon_N$ “均值 $8.97\%$，最大 $14.60\%$”，缩聚刚度“均值 $0.44\%$，最大 $1.22\%$”；刚度路线“均值 $4.30\%$，最大 $8.09\%$” |
| 解层精度 | [§7.4](../../experiments/analysis_capability_piml_substructure/results_analysis.md#74-整体结构求解精度与路线对比) | `full_trace` 柔度相对误差：直接预测刚度 3.64%，形函数变分 0.20%；`linear_corner` 历史产物“全场位移误差为 $99.9\%$、柔度误差为 $96.4\%$”，未纳入对比 |
| 未解决问题 | [§7.5](../../experiments/analysis_capability_piml_substructure/results_analysis.md#75-当前结论与未解决问题) | `linear_corner` 需先排查解析刚体分解 |

上述数值的来源产物与 git revision 未在该报告中逐项登记（见其第 9 节），且早于 71a03d6 的目录重组。本目录两个入口在重组后的状态如下，首轮运行后把结果与 revision 补入 experiments 侧报告：

| 入口 | 结果文件 | 状态 |
|---|---|---|
| `verify_shape_function_route.py --skip-train` | `eq17_second_order_full_trace_analytic.json` | 待跑 |
| `verify_shape_function_route.py --trace-basis linear_corner --skip-train` | `eq17_second_order_linear_corner_analytic.json` | 待跑；若仍复现 §7.2 的 `1.193`，`rigid_part_analytic_deviation_max` 检查将不通过 |
| `verify_shape_function_route.py`（缺省训练配置） | `eq17_second_order_full_trace.json` | 待跑 |
| `verify_stiffness_route.py`（缺省训练配置） | `piml_exact_comparison.json` / `.png` | 待跑 |
