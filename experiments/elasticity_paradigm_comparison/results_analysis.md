# 四范式对比：数学—代码映射契约与实验证据报告

本文件是本实验的数学定义与代码实现之间的映射契约，以及实验证据的落盘位置。方法知识、文献证据与长期技术路线由 `dut-postdoc` 维护，不复制到这里。

---

## 1. 数学—代码映射契约

### 1.1 四条路径的求解对象

| 路径 | 学习/求解对象 | 核心库接口 | 产出 |
|---|---|---|---|
| `lagrange` | 全局位移 $\mathbf U$，解 $\mathbf{KU}=\mathbf F$ | `soptx.fem.analyzers.LagrangeFEMAnalyzer` | 参考真解 $\mathbf U_\text{full}$、全局刚度 $\mathbf K$ |
| `substructure` | 局部 $\mathbf K_s^j=\mathbf K_{bb}^j-\mathbf K_{bi}^j(\mathbf K_{ii}^j)^{-1}\mathbf K_{ib}^j$ | `soptx.fem.substructure.FEAStaticCondensation` | $\mathbf U_\text{full}$、批量 $\mathbf K_s$ |
| `pinn` | 连续解场 $\hat u_\theta(x)$ | 待下沉至 `soptx.ml` | 细网格节点采样位移 |
| `piml` | 代理映射 $\rho^j\mapsto\widehat{\mathbf K}_s^j$ 或 $\widehat{\mathbf N}^j$ | `soptx.fem.substructure.PIMLStaticCondensation` | $\mathbf U_\text{full}$、批量 $\widehat{\mathbf K}_s$、回退统计 |

### 1.2 统一度量

全部度量在 `metrics.py` 中唯一实现，各适配器只产出位移场与算子，不自带误差定义。

| 指标 | 数学定义 | 实现 | 适用路径 |
|---|---|---|---|
| 位移相对 $L_2$ | $\|u-u_\text{ref}\|_2/\|u_\text{ref}\|_2$ | `relative_l2` | 四者 |
| 能量范数相对误差 | $\sqrt{e^\mathsf{T}\mathbf Ke}/\sqrt{u_\text{ref}^\mathsf{T}\mathbf Ku_\text{ref}}$ | `energy_norm_relative_error` | 四者 |
| 柔顺度相对误差 | $\lvert C-C_\text{ref}\rvert/\lvert C_\text{ref}\rvert$，$C=\mathbf f^\mathsf{T}\mathbf u$ | `compliance`、`relative_error` | 四者 |
| 接口/内部分层误差 | 子集上的相对 $L_2$ | `subset_relative_l2` | `substructure`、`piml` |
| 最小特征值 | $\lambda_\min(\widehat{\mathbf K}_s)$ | `min_eigenvalue` | `piml` |
| 刚体模态残差 | $\|\widehat{\mathbf K}_s\mathbf R\|_F/\|\widehat{\mathbf K}_s\|_F$ | `rigid_mode_residual` | `piml` |
| 摊销盈亏点 | $N^\*=T_\text{off}/(t_\text{ref}-t_\text{on})$ | `breakeven_count` | `pinn`、`piml` |

能量范数取为主指标：Schur 补缩聚在数学上是能量二次型上的 Ritz 投影，该范数使「精确缩聚是变分投影、PIML 是带扰动的投影」在数值上闭合。

刚体模态基取自 `SubstructurePrototype.rigid_basis`，`metrics.py` 按参数接收，不重建。

### 1.3 各路径的既有正确性证据

本实验不重复验证各路径自身的正确性，只在冻结算例上取用其产出。正确性证据的所有者：

| 路径 | 证据来源 | 判据 |
|---|---|---|
| `lagrange` | `examples/lagrange_elasticity/manufactured_convergence_demo.py` | 制造解 L2 观测收敛阶（阈值 1.5）+ 真相对残差 |
| `lagrange` | `examples/lagrange_elasticity/concentrated_load_demo.py` | 真相对残差 + 载荷等效性 $\lvert\sum F_{\sigma_h}-P\rvert$ |
| `substructure` | `examples/substructure_elasticity/compare_lagrange.py` | 与全装配解的柔度、全节点位移相对误差落在机器精度内 |
| `piml` | `examples/piml_substructure_elasticity/verify_stiffness_route.py` | $\mathbf K_s$ 相对 Frobenius 误差、回退统计 |
| `pinn` | `examples/pinn_elasticity/minimal_demo.py` | 制造解相对 $L_2$ 误差 |

`lagrange` 两条判据的实测数值见 [`examples/lagrange_elasticity/results_analysis.md`](../../examples/lagrange_elasticity/results_analysis.md) §4：$p=1$ 的 $L_2$ 观测阶趋于 2.0、$p=2$ 趋于 3.0，真相对残差在 $10^{-15}\sim10^{-12}$ 量级，集中力载荷等效性偏差恒为零。该目录的两个 demo 在门禁通过后会把逐层结果写入 `examples/lagrange_elasticity/outputs/` 下的 JSON（文件名编码维数、网格、模型、次数与求解器），引用其数值时应取 JSON 而非人工转录，并连同其中的 `fealpy_path` 字段一起记录。

### 1.4 受控消融的四条边

| 边 | 变量 | 预期与职责 |
|---|---|---|
| `lagrange` ↔ `substructure` | 仅载体 | 代数等价，偏差应在 $10^{-13}$ 量级；实现正确性自检，非精度比较 |
| `substructure` ↔ `piml` | 仅是否代理 | 载体相同，给出纯代理误差 |
| `lagrange` ↔ `pinn` | 仅是否代理 | 载体相同 |
| `pinn` ↔ `piml` | 仅载体 | 均为代理，回答复用边界差异 |

四条路径的相对 $L_2$ 误差不构成同一张排名表：精确路径在 $10^{-13}$、代理路径在 $10^{-2}\sim10^{-4}$，排名不含信息。结论落在摊销盈亏点与结构保持退化上。

---

## 2. 实验证据报告

**尚无证据。** 适配器接线完成后，本节按 Tier A / Tier B 分小节填写，每条结论必须给出对应 `outputs/<case-id>/manifest.json` 中的字段路径。

计划的证据落盘结构：

```text
outputs/
`-- <case-id>/
    |-- manifest.json          # 机器可读结果: 四路径 x 全部度量
    |-- <method>/summary.json  # 单路径产出与计时
    `-- breakeven.json         # 摊销曲线数据点
```

---

## 3. 已知问题与阻塞

按解除顺序排列。

### 3.1 PINN 求解器未下沉（阻塞 `pinn` 适配器）

`PINNElasticityNet` 与残差构造目前只存在于 [`examples/pinn_elasticity/minimal_demo.py`](../../examples/pinn_elasticity/minimal_demo.py)（`:47`），并在 [`experiments/elasticity_paradigm_comparison/legacy/compare_piml_pinn.py`](legacy/compare_piml_pinn.py)（`:253`）被重复实现一次；后者还在 `:93` 重造了一个 `gradient`，而 `fealpy.ml.grad.gradient` 已有该能力。`src/soptx/ml/` 当前仅含 `networks.py`。

解除方式：按 `CLAUDE.md` 规则 3 将网络、残差与训练循环下沉至 `src/soptx/ml/`，消除两处重复，本目录只作薄适配。

### 3.2 算例不可比（阻塞 Tier A 全部四条路径）

`compare_piml_pinn.py` 的既有算例采用制造解 `ExponentialSineManufacturedElasticity2D`，其位移在四条边上恒为零、唯一驱动是体力；而缩聚路径按 $\mathbf f_i=\mathbf 0$ 建模，在该算例上只能解出零位移场，位移误差退化为 $0/0$。同时子结构库的平面假设固定为 `plane_stress`，而该制造解是 plane strain，两条路径的物理模型并不相同。

解除方式：本实验改用边界载荷驱动的 `FullMBBBeam2d`（`verify_stiffness_route.py` 已冻结的设置），四条路径均可产出非平凡位移场；PINN 在其上配硬约束边界。`cases.toml` 已按此配置。

### 3.3 PIML 训练循环未下沉（阻塞 `piml` 适配器）

预测器 `PIMLSurrogateNet` 与 `PIMLStaticCondensation` 已在 `soptx.fem.substructure` 中，但密度快照采样与训练循环只存在于 `verify_stiffness_route.py` 脚本内部，无法被本目录复用。

解除方式：将采样与训练循环下沉至核心库，脚本与本实验共同调用。

### 3.4 上下文组装未接线（阻塞全部四条路径）

`run.py:build_context` 当前恒抛 `NotImplementedError`。所需的 `GlobalAssembler` 构造与外载/约束提取范式已存在于 `verify_stiffness_route.py`，下沉后在此调用即可，无能力缺口。

### 3.5 参考解在 `FullMBBBeam2d` 上缺证据（阻塞 Tier A/B 的参考路径）

`cases.toml` 取 `FullMBBBeam2d` 以对齐 PIML 侧已冻结的 `verify_stiffness_route.py`（Huang 2023 §4.1）。但 Lagrange 侧的 2D 算例只覆盖 `HalfMBBBeamRight2d`（对称半域），完整域只有 3D 的 `FullMBBBeam3d`；`compare_lagrange.py` 同样是 2D 半域 + 3D 完整域。因此本实验所用的 2D 完整域组合，参考路径尚无正确性证据。该缺口已同步记录在 [`examples/lagrange_elasticity/results_analysis.md`](../../examples/lagrange_elasticity/results_analysis.md) §5 第 3 条。

解除方式（待定）：

- **方案 A**：保留 `FullMBBBeam2d`，在 `concentrated_load_demo.py` 增加 2D 完整域 `--problem` 选项并补跑一次。该文件已支持 `FullMBBBeam3d`，属同技术栈同文件加配置。
- **方案 B**：`cases.toml` 改用 `HalfMBBBeamRight2d`。代价是与 PIML 侧已冻结算例脱钩，既有 PIML 证据不可比。

### 3.6 度量口径尚未收编

现有三个成对比较脚本各报各的量：`substructure_elasticity/compare_lagrange.py` 报收敛阶，`verify_stiffness_route.py` 报 $\mathbf K_s$ 的 Frobenius 误差，`compare_piml_pinn.py` 报全场位移 $L_2$。三者无法横向对齐。`metrics.py` 已给出统一实现，但既有脚本尚未改为调用它。
