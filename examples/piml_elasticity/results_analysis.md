# PIML 子结构静力缩聚契约、验证与已知问题 (PIML Contract & Report)

本报告承载本算例的数学—代码映射、当前验证范围、已知问题与后续工作。本目录为**早期原型**：PIML 预测器尚未进入全局接口系统评估，README 描述与代码的实际执行路径对齐。

## 1. 数学—代码映射契约

### 1.1 精确缩聚（基线）

| 数学对象 | 代码 | 含义 |
|---|---|---|
| $K_{ii}^j$ | `K_local[np.ix_(i_dofs, i_dofs)]` | 子结构内部自由度刚度。 |
| $K_{ib}^j$ | `K_local[np.ix_(i_dofs, b_dofs)]` | 内部—接口耦合刚度。 |
| $N^j$ | `-np.linalg.solve(K_ii, K_ib)` | 内部位移恢复映射：$u_i^j=N^j u_b^j$。 |
| $K_s^j$ | `K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)` | Schur 补缩聚刚度。 |

`StaticCondensationBase` 为 FEA 与 PIML 提供了统一接口：`condense(K_local, rho_local=None)` → `(K_s, N)`；`recover(u_b)` → `u_i`。

### 1.2 PIML 预测器（当前实现）

| 组件 | 代码 | 含义 |
|---|---|---|
| 输入 | `rho.flatten()` (25,) | 子结构元素密度，固定尺寸 $n_{\text{fine}}^x \times n_{\text{fine}}^y$。 |
| 模型 | `PIMLSurrogateNet(25, 210)` | 三层 MLP，SiLU 激活。 |
| 输出 | `pred_triu` (210,) | 直接预测 $\mathbf K_s$ 上三角元素，手工对称化恢复完整矩阵。 |
| 结构检查 | `eigvalsh(K_s_pred)` | 检查特征值；见 §3 已知问题。 |
| 回退 | `fallback_solver.condense()` | 模型缺失、无密度输入或预测异常时回退精确缩聚。 |

## 2. 当前验证范围

`minimal_demo.py` 执行下列步骤：

1. 4×2 悬臂梁子结构划分，每个子结构 5×5 Q4 单元；
2. 精确缩聚求解全尺度 FE 参考解，验证位移恢复 $L_2$ 误差；
3. 在**单个子结构**（sub_0）上以 200 个随机密度样本训练 MLP；
4. 对该子结构的密度场，比较预测 $\widehat{\mathbf K}_s$ 与精确 $\mathbf K_s$ 的相对 Frobenius 误差。

**当前未验证的功能**（README 中的描述与代码不一致，以下功能尚未实现）：

- 将预测 $\widehat{\mathbf K}_s$ 装配到全局接口系统；
- 用预测算子求解接口位移、柔顺度、细尺度恢复；
- 对 PIML 预测路径的下游误差评估。

## 3. 已知问题

| 编号 | 严重程度 | 问题 | 影响 |
|---|---|---|---|
| #1 | 中 | PSD 检查用 `evals[-1] <= 0`（最大特征值），应检查 `evals[0]`（最小特征值） | 正定门禁形同虚设，病态矩阵可通过。 |
| #2 | 中 | 预测成功时取精确 N 但替换预测 K_s，构造上不满足 $K_s = N^T K N$ | 破坏能量一致性，全局装配与内部恢复使用不同源。 |
| #3 | 低 | 训练样本 `uniform(0.3, 1.0)`，测试为 sin·cos 密度场，分布不同 | 报告的 `rel_err_Ks` 不是有定义的测试指标。 |
| #4 | 低 | 无 OOD 检测（密度接近 0 为 TO 常见状态） | 训练分布外行为未评估。 |

## 4. 路线说明

- 本原型预测 $\widehat{\mathbf K}_s$（路线 B 雏形），非 Cholesky 参数化，非预测形函数 $\widehat{\mathbf N}$（路线 A）。
- `math_spec.md`（已并入）中「路线 A ∧ 路线 B」的文字描述属于论文计划的通用说明，不等于本原型的实现承诺。

## 5. 后续工作

- [ ] 将预测 $\widehat{\mathbf K}_s$ 装配到全局接口系统，计算接口位移、柔顺度与细尺度恢复；
- [ ] 修复 PSD 检查逻辑（#1）和回退能量一致性（#2）；
- [ ] 统一 train/val/test 划分与 OOD 评估；
- [ ] 按统一比较契约，并列评价路线 A（预测 N）与路线 B（Cholesky 参数化）。
