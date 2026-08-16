# PIML 子结构静力缩聚契约、验证与已知问题 (PIML Contract & Report)

本报告承载本算例的数学—代码映射、当前验证范围与后续工作。

> **理论事实源**：👉 [子结构有限元与静力缩聚](file:///mnt/c/workspace/dut-postdoc/concepts/substructural-condensation.md) & 👉 [子结构 PIML 算子与物理正定范式](file:///mnt/c/workspace/dut-postdoc/concepts/piml/piml-substructural.md)

## 1. 数学—代码映射契约

### 1.1 精确缩聚（基线）

精确缩聚由 [`src/soptx/fem/substructure/`](../../src/soptx/fem/substructure/) 的 `FEAStaticCondensation` 提供；
`StaticCondensationBase` 为 FEA 与 PIML 提供了基于 FEALPy `bm` 后端的统一接口：`condense(K_local, rho_local=None)` → `(K_s, N)`；`recover(u_b)` → `u_i`。

### 1.2 PIML 代理预测器 (变形子空间上的 Cholesky 参数化)

自由漂浮子结构的 $\mathbf K_s$ 以刚体模态为**精确**零空间：若 $(\boldsymbol u_b,\boldsymbol u_i)$ 是刚体运动，
则 $\mathbf K[\boldsymbol u_b;\boldsymbol u_i]=\mathbf 0$，于是 $\boldsymbol u_i=\mathbf N\boldsymbol u_b$ 且
$\mathbf K_s\boldsymbol u_b=\mathbf 0$（对 P1 拉格朗日单元精确成立）。参数化据此把预测限制到刚体子空间的
正交补 $\mathbf R_\perp\in\mathbb R^{n_b\times m}$（$m=n_b-n_{\text{rigid}}$）上：

$$\widehat{\mathbf K}_s \;=\; \mathbf R_\perp\,\mathbf L\mathbf L^{\mathsf T}\,\mathbf R_\perp^{\mathsf T}$$

由此同时得到三条**构造性**保证：$\widehat{\mathbf K}_s$ 在刚体子空间上恒为零（拟合误差无法泄漏进最软的方向，
见 2.2(c)）；在变形子空间上正定，与精确 $\mathbf K_s$ 的秩亏结构一致；训练目标
$\operatorname{cholesky}(\mathbf R_\perp^{\mathsf T}\mathbf K_s\mathbf R_\perp)$ 无需任何正则项，
因而不带系统性正偏置。$\mathbf R_\perp$ 由 `SubstructurePrototype.deformation_basis` 解析构造
（六个/三个刚体模态在接口自由度上的取值经完整 QR 正交化），与密度无关，全部子结构共享。

| 组件 | 代码 | 含义 |
|---|---|---|
| 后端管理 | `fealpy.backend.backend_manager as bm` | 统一矩阵、向量运算与特征值检测。 |
| 输入 | `bm.reshape(rho, (-1,))`，形状 `(25,)` | 子结构单元密度，固定 $5 \times 5$ 细网格。 |
| 刚体/变形基 | `prototype.rigid_basis` $(40,3)$、`prototype.deformation_basis` $(40,37)$ | 解析构造并 QR 正交化；实测 $\lVert\mathbf K_s\mathbf R_{\text{rigid}}\rVert/\lVert\mathbf K_s\rVert\sim10^{-17}$。 |
| 模型 | `PIMLSurrogateNet(25, 703)` | 三层 MLP，SiLU 激活；输出维数为 $m(m+1)/2$，$m=37$。 |
| 输出 | `pred_tril`，形状 `(703,)` | 变形子空间上 Cholesky 因子 $\mathbf L$ 的下三角条目，对角线取 `abs` 使其为正。 |
| 训练目标 | `cholesky(R^T K_s R)` 的下三角条目 | 限制后的算子严格正定（实测最小特征值 $1.0\times10^{-2}$，条件数 $\approx94$），无需正则。 |
| 门禁与回退 | $\mathbf L$ 的有限性 + $\min\lvert\operatorname{diag}\rvert/\max\lvert\operatorname{diag}\rvert>10^{-8}$ | 门禁作用在 $\mathbf L$ 而非 $\widehat{\mathbf K}_s$ 上：后者按构造秩亏，$\lambda_{\min}\equiv0$，用正定判据会恒定回退。异常时回退精确 FEA 并置 `used_fallback`。 |
| 契约错误 | `SurrogateContractError` | 网络输出维与 $m(m+1)/2$ 不符属配置错误，**不**回退，直接上抛。 |

推理路径不再对 $\widehat{\mathbf K}_s$ 求特征值，每次推理省去一次 $40\times40$ 的 `eigvalsh`。

## 2. 算例与验证范围

### 2.1 代理缩聚 vs 精确缩聚 (`compare_exact.py`)

采用 Huang 2023 第 4.1 节完整 MBB 梁物理问题 (`soptx.problems.elasticity.FullMBBBeam2d`)，
$12 \times 2$ 子结构划分（共 24 个子结构），每个子结构 $5 \times 5$ Q4 单元（全网格 600 单元）。
两条路径共用同一批 `SubstructureMesh`，因此接口自由度编号一致，接口位移可逐分量直接相减：

- **路径 A**：`FEAStaticCondensation` 批量 Schur 补，一次装配 + 一次缩聚；
- **路径 B**：`PIMLStaticCondensation` 逐子结构代理预测，复用路径 A 已装配的 `K_local_batch`。

误差**分两层**报告，两层之比即误差在求解链路上的放大倍率：

| 层 | 指标 | 语义 |
|---|---|---|
| 算子层 | $\lVert\widehat{\mathbf K}_s^j-\mathbf K_s^j\rVert_F / \lVert\mathbf K_s^j\rVert_F$ 的 max / mean | 代理本征精度，与外载、边界条件无关 |
| 解层 | 接口位移、全场位移相对 $L_2$，柔度相对误差 | 算子误差经接口系统放大后的后果 |

**精确缩聚在此仅作基线，不是被验对象**：它与 Lagrange 全装配的机器精度等价已由
[`examples/substructure_elasticity/compare_lagrange.py`](../substructure_elasticity/compare_lagrange.py)
建立，本脚本不重复验证，因此不构造全尺度参考解。`LagrangeFEMAnalyzer` 仍被实例化，但只用作
外载与约束的来源（`force_vector` 是施加 Dirichlet 条件*之前*的外载向量，
`tensor_space.boundary_interpolate()` 给出固定自由度掩码），从不调用 `solve_state()`。

评估密度场**逐细单元变化**而非每个子结构取常值，幅值落在训练区间 `DENSITY_RANGE = (0.3, 1.0)`
内部。均匀密度只是训练分布中测度为零的切片，用它评估无法反映代理在真实输入上的精度。

验收：代理精度不设阈值断言（拟合误差随训练配置连续变化，硬编阈值只是伪门禁）；`--strict` 下
若**在役子结构或留出样本**的回退计数大于零则以异常失败，使「代理是否真正生效」成为可回归状态。
两者都纳入是因为在役的 24 个子结构只覆盖一组特定密度，而留出集覆盖整个训练分布：仅前者通过
而后者失败，说明分布内存在成片的退化预测，只是恰好没落在本次评估的密度上（实测曾出现 4/100）。

训练采样、留出采样与网络初始化的随机性统一由 `--seed`（缺省 `2026`）固定，同一组参数逐位可复现；
比较不同训练配置时必须保持种子不变，否则观测差异混有采样噪声。

### 2.2 误差归因诊断

两层误差只能说明「解层误差比算子层大多少」，不能说明责任在谁。`compare_exact.py` 因此额外输出
三组归因指标，把「算子层误差本身从哪来」拆开。

**(a) 刚体基正确性与参数化误差下界（`verify_parameterization_parity`）**

该检查串联两件事，任一失败即以异常终止。

其一，解析构造的刚体模态基必须张成精确 $\mathbf K_s$ 的零空间，否则把预测限制到它的正交补上会
抹掉真实刚度。指标为 $\lVert\mathbf K_s\mathbf R_{\text{rigid}}\rVert/\lVert\mathbf K_s\rVert$，
阈值 $10^{-10}$，实测约 $8\times10^{-17}$。

其二，训练侧用布尔掩码按行优先取 $\mathbf L$ 的下三角条目，推理侧用同一掩码写回。两处排布若不一致，
网络会去拟合一个被置换过的目标——训练损失照常下降、门禁照常通过、预测的 $\widehat{\mathbf K}_s$
完全错误。该检查把一组**精确**的 Cholesky 条目送进推理路径，重构误差应只剩浮点精度。

后者同时是本次运行的**误差下界**：代理的 $\mathbf K_s$ 相对误差不可能低于它。桩网络按 `float32`
存储条目，与真实网络的输出精度一致，实测约 $3.8\times10^{-8}$（旧参数化为 $2.6\times10^{-4}$，
由 $10^{-6}$ 正则与 `abs(diag) + 1e-4` 下界共同贡献；两者均已移除）。百分量级的观测误差因此
全部是网络拟合的责任。

**(b) 训练分布内 vs 光滑评估场**

留出集由 `sample_random_density` 采样，与训练集共用同一函数因而严格同分布，且以概率 1 不重合。
两者之比给出判据：

| 观测 | 结论 |
|---|---|
| 光滑场误差 $\approx$ 留出集误差 | 瓶颈是**欠拟合**，网络在自己的分布上就没学会 |
| 光滑场误差 $\gg$ 留出集误差 | 瓶颈是**分布错配**，逐单元 i.i.d. 训练集撑不起空间相关的光滑输入 |

留出集推理走与主流程同一条 `PIMLStaticCondensation` 路径，因此对角线取绝对值与投影回全部接口
自由度等推理期处理都被计入，度量的是实际投入使用的算子而非网络原始输出。

**(c) 零空间污染与能量归因**

精确 $\mathbf K_s$ 恰有 $n_{\text{rigid}}$ 个零特征值（二维 3，三维 6）。刚体模态是最软的方向，
也正是装配后各子结构位移的主要成分（实测占比 $99.98\%$），因此该子空间上的任何伪刚度，其能量
后果都可以远大于它在 Frobenius 范数下的占比——这正是相对 Frobenius 误差不适合度量近奇异算子的原因。

**该诊断曾定位出主导误差源。** 历史上的 $\mathbf L\mathbf L^{\mathsf T}$ 参数化严格正定，
结构上无法表示秩亏，必然给刚体模态注入正伪刚度：实测伪刚度仅为最小非零特征值的 $1.6\%$，
却因位移几乎全在该子空间而被 $(1/0.02)^2\approx2500$ 倍放大，贡献了观测刚化的 $96.8\%$。
1.2 节的参数化改造关闭了这条通道——$\widehat{\mathbf K}_s$ 在刚体子空间上恒为零是构造性质，
本诊断因而转为该构造的**运行期校验**，`rigid_pollution_*` 应读到机器精度。

零空间由精确 $\mathbf K_s$ 的最小 $n_{\text{rigid}}$ 个特征向量 $\mathbf V$ 给出，
**刻意不复用** `prototype.rigid_basis`：由此它同时是对解析刚体基的独立校验，且不假设网格坐标，
二维三维通用（所有指标都在零空间的正交基选取下不变）：

| 指标 | 定义 | 语义 |
|---|---|---|
| `exact_rigid_residual_max` | $\lambda_{\max}\lvert\mathbf V^{\mathsf T}\mathbf K_s\mathbf V\rvert$ | $\mathbf V$ 的自检，应在机器精度量级 |
| `rigid_pollution_ratio` | $\lambda_{\max}\lvert\mathbf V^{\mathsf T}\widehat{\mathbf K}_s\mathbf V\rvert / \lambda_{\text{soft}}$ | 伪刚度相对最小非零特征值，**与载荷无关** |
| `energy_stiffening_factor` | $\sum_j \boldsymbol u_b^{j\mathsf T}\widehat{\mathbf K}_s^j\boldsymbol u_b^j \big/ \sum_j \boldsymbol u_b^{j\mathsf T}\mathbf K_s^j\boldsymbol u_b^j$ | 在**精确解**上求值的刚化倍率 |
| `energy_rigid_pollution_share` | 上式分子中仅由 $\mathbf V$ 子空间贡献的部分 / 精确应变能 | 刚化中有多少来自零空间污染 |

分母 $\sum_j \boldsymbol u_b^{j\mathsf T}\mathbf K_s^j\boldsymbol u_b^j$ 应等于精确柔度 $\mathbf f^{\mathsf T}\boldsymbol u$，
可作为接线自检。若 `energy_stiffening_factor` 接近实测柔度比 $c_{\text{exact}}/c_{\text{piml}}$，
说明柔度误差已被该刚化完全解释；再看 `energy_rigid_pollution_share` 占其中多少，即可判定
零空间污染是不是主因。改造后该 share 应趋近零，剩余的刚化即变形子空间上的纯拟合误差。

### 2.3 PIML vs PINN 跨范式对比 (`compare_piml_pinn.py`)

两条路径在同一块方域、同一个制造解问题 (`ExponentialSineManufacturedElasticity2D`) 上运行，
但**精度指标不是同一个量**，这一差别本身就是范式分界：

| 评估维度 | PIML 子结构静力缩聚 | PINN 强形式求解器 |
|---|---|---|
| 范式类型 | Problem-Independent（局部算子） | Problem-Dependent（坐标映射） |
| 边界条件/外载变更 | 免重训，直接复用 | 必须整轮重训 |
| 网络输入语义 | 局部细网格密度 $\boldsymbol\rho^j$ | 空间点坐标 $(x, y)$ |
| 网络输出语义 | Schur 补 $\mathbf K_s$ 的 Cholesky 因子 | 该点位移 $(u_x, u_y)$ |
| 精度指标 | $\mathbf K_s$ 相对 Frobenius 误差 | 位移相对 $L_2$ 误差 |

PINN 学的是绑定单个定解问题的位移场，PIML 学的是与外载、边界条件都无关的局部算子，
两者的自然误差量纲不同，不应强行并列为同一行数字。

**为什么本算例不报告 PIML 的位移误差**：该制造解的位移在四条边上恒为零
（$u_1 = e^{x-y}x(1-x)y(1-y)$、$u_2=\sin\pi x\sin\pi y$），唯一的驱动是体力；
而缩聚路径按内部自由度不受载的建模假设成立（见 3.2），在本算例上只能解出零位移场，
位移误差退化为无意义的 $0/0$。带真实外载的位移场验证由 `compare_exact.py` 承担。

具体数值以同一次运行的终端输出与 [outputs/piml_vs_pinn_comparison.png](outputs/piml_vs_pinn_comparison.png)
为准，本文档不保存脱离脚本、commit 与运行环境的历史实测值。

**已知口径差异**：子结构库的平面假设固定为 `plane_stress`（`mesh.py` 与 `assembler.py` 中硬编码），
而该制造解是 plane strain。由于 PIML 路径只在自身离散算子内部与精确 Schur 补对比，
该差异不影响所报误差，但它使 PIML 路径的离散算子与 PINN 路径的连续问题并非同一个物理模型。

## 3. 已知问题

### 3.1 秩亏与门禁：三次迭代的历史（已解决）

自由漂浮子结构的 $\mathbf K_s$ 秩亏 $n_{\text{rigid}}$，而 Cholesky 参数化天然产出正定阵——
这一矛盾以三种形态出现过，记录于此以免重蹈：

| 版本 | 做法 | 后果 |
|---|---|---|
| 一 | 训练目标 `cholesky(K_s + 1e-6 I)`，重构后扣回 $-10^{-6}\mathbf I$，门禁要求 $\lambda_{\min}>10^{-8}$ | 即使完美拟合也只得 $\lambda_{\min}\approx-10^{-6}$，**门禁构造上无法通过**，24/24 与 4/4 静默回退，报告的「PIML」数值实为精确解 |
| 二 | 移除扣除，保留 $10^{-6}$ 正则与 `abs(diag) + 1e-4` 下界，门禁降级为 `NaN` 与退化拦截 | 代理真正生效，但 $\widehat{\mathbf K}_s$ 结构上无法秩亏，在最软方向注入伪刚度，被位移的刚体成分平方放大，贡献 $96.8\%$ 的刚化（见 2.2(c)） |
| 三（当前） | 预测限制到 $\mathbf R_\perp$ 上，$\widehat{\mathbf K}_s=\mathbf R_\perp\mathbf L\mathbf L^{\mathsf T}\mathbf R_\perp^{\mathsf T}$，无正则无下界，门禁改作用于 $\mathbf L$ | 秩亏成为构造性质，正偏置与零空间污染同时消失；参数化误差下界由 $2.6\times10^{-4}$ 降至 $3.8\times10^{-8}$ |

版本三的门禁必须作用在 $\mathbf L$ 上：$\widehat{\mathbf K}_s$ 按构造 $\lambda_{\min}\equiv0$，
若沿用「要求 $\lambda_{\min}>10^{-8}$」就会退化成版本一的恒定回退——同一个 bug 换一层壳。
判据取 $\mathbf L$ 对角线的 $\min/\max$ 之比（量纲无关，只看对角线内部的相对尺度）。

两个脚本都打印「回退到精确缩聚的子结构数」；`compare_exact.py --strict` 下在役子结构或留出集的
回退计数大于零即以异常失败，使该状态不再可能悄悄退化。

### 3.2 内部自由度不受载假设（继承自论文，非实现缺陷）

`FEAStaticCondensation` 只缩聚刚度，恢复关系固定为 $u_i^j = \mathbf N^j u_b^j$。这与
Huang 2023 一致：论文式 (6) 的右端项直接写成 $(f_{jb}^h, \mathbf 0)^{\mathsf T}$，并明言
"不失一般性地假设与 $u_{ji}^h$ 相关的外部载荷为零"，其式 (7) 的"缩聚载荷"在
$f_{ji}^h = \mathbf 0$ 下退化为 $f_{jb}^h$ 本身，恢复式同样无 $(\mathbf K_{ii}^j)^{-1}f_i^j$ 项。

因此当前实现与论文同样只对**内部自由度不受载**的问题成立：集中载荷、面载荷必须落在
接口自由度上，体力问题无法表达。**这正是 2.3 节中 PIML 路径无法求解制造解算例的原因**
——该算例唯一的驱动恰是体力。若要覆盖，需真正实现
$f_s^j = f_b^j - \mathbf K_{bi}^j(\mathbf K_{ii}^j)^{-1}f_i^j$ 并在恢复式中补项，
这是超出论文覆盖范围的扩展。

## 4. 后续工作与路线规划

- [x] 基于 `bm` 重构代理与矩阵计算；
- [x] 对齐 Huang 2023 第 4.1 节完整 MBB 梁算例 (`FullMBBBeam2d`)；
- [x] 引入 Cholesky 下三角预测范式；
- [x] 开发 PIML vs PINN 跨范式对比程序 (`compare_piml_pinn.py`)；
- [x] 共享参考子结构 + 批量缩聚，消除同构子结构上的 Python 循环；
- [x] 修复正定性门禁，使代理网络真正投入使用（见 3.1）；
- [x] 开发代理 vs 精确缩聚对比程序 (`compare_exact.py`)，误差分算子层与解层两层报告；
- [x] 增加误差归因诊断（参数化自洽性与误差下界、训练分布内留出集、零空间污染与能量归因），
  并以 `--seed` 固定全部随机性，使不同训练配置之间可比（见 2.2）；
- [x] 训练目标改为在变形子空间上分解，$\widehat{\mathbf K}_s=\mathbf R_\perp\mathbf L\mathbf L^{\mathsf T}\mathbf R_\perp^{\mathsf T}$，
  同时消除 $10^{-6}$ 正偏置与零空间污染（见 1.2 与 3.1）；
- [ ] **提升代理拟合精度**：零空间通道关闭后，剩余误差是变形子空间上的纯拟合误差；按 2.2(b)
  的判据决定走输入/输出归一化，还是改造训练密度的空间相关性；
- [ ] **实现 3.2 的载荷缩聚**，解除内部自由度不受载的限制；
- [ ] 将 `used_fallback` 由标量改为 `(...,)` 掩码，支持 PIML 路径批量化；
- [ ] 开展路线 A（预测多尺度形函数 $\mathbf N$）与 Cholesky 路线 B 的跨算法消融对比。
