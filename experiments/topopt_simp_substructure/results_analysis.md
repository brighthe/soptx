# 精确子结构变密度拓扑优化

---

## 1. 研究范围与算例设计

本实验验证基于**局部 Exact Schur 缩聚**的变密度拓扑优化（SIMP/OC）全流程，**不引入任何神经网络代理（PIML）**。
核心目标是评估子结构接口迹空间（`trace`）对拓扑优化多步迭代演化、目标收敛性及最终构型的影响，并为下游 PIML 局部代理提供纯有限元基准。

### 1.1 研究内容与验证矩阵

本实验共设立 4 个基准工况，分别在二维与三维空间中开展全接口代数等价性检验与角点迹降阶误差评估：

| 研究内容 / 验证目标 | 算例模型 | 接口迹 | 核心考察指标 | 对应工况 ID |
| :--- | :---: | :---: | :--- | :--- |
| **二维代数等价性检验** | `CantileverCorner2d` | `full_trace` | 首步与逐迭代柔度相对偏差、拓扑分布与 FA 的舍入级一致性 | `cantilever_2d_ft` |
| **二维角点迹降阶评估** | `CantileverCorner2d` | `linear_corner` | 相对全接口/FA 的柔度偏差、收敛稳定性与构型演化 | `cantilever_2d_lc` |
| **三维代数等价性检验** | `FullMBBBeam3d` | `full_trace` | 三维六面体网格下与 FA 的逐步柔度及拓扑一致性 | `mbb_3d_ft` |
| **三维角点迹基准评估** | `FullMBBBeam3d` | `linear_corner` | 三维实体角点迹的柔度偏差，作为下游 PIML 的纯有限元参照 | `mbb_3d_lc` |

### 1.2 物理模型与力学控制方程

基准算例在线弹性小变形假设下建立，位移场满足线弹性平衡方程 $\nabla\cdot\boldsymbol{\sigma} = \mathbf{0}$ 与本构关系 $\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}$。空间离散解耦并选取两个经典基准算例：

| 算例模型 | 应力状态 / 本构模型 | 计算域尺寸 $\Omega$ | 实体材料参数 | 位移边界条件 (Dirichlet) | 载荷边界条件 (Neumann) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `CantileverCorner2d` | 平面应力 (`plane_stress`) | 矩形域 $[0, 160]\times[0, 100]$ | $E_0=1.0$, $\nu=0.3$ | 左端整边固支：$x=0$ 处 $u_x=u_y=0$ | 右下角点 $(160, 0)$ 竖向集中力 $\mathbf P=(0, -1.0)$ |
| `FullMBBBeam3d` | 三维各向同性线弹性 | 长方体域 $[0, 12]\times[0, 2]\times[0, 2]$ | $E_0=1.0$, $\nu=0.3$ | • 左底边 ($x=0, y=0$)：$u_x=u_y=0$<br>• 右底边 ($x=12, y=0$)：$u_y=0$<br>• 底面中线 ($y=0, z=1$)：$u_z=0$ | 顶面中心 $(6, 2, 1)$ 竖向集中力 $\mathbf P=(0, -1.0, 0)$ |

### 1.3 接口迹空间与误差传递

算法变化轴为**接口迹空间 `trace`**：

| 接口迹 `trace` | 局部降阶定义 | 相对 FA（全域装配）的误差属性 | 定位说明 |
| :--- | :--- | :--- | :--- |
| **`full_trace`** | 保留全部子结构边界自由度（$\mathbf T_j = \mathbf I$） | **无方法误差，严格代数等价**，仅含求解器舍入误差 | 正确性基准：检验子结构装配、消元、位移恢复与灵敏度分析的等价性 |
| **`linear_corner`** | 仅保留角点自由度，边界由角点线性插值确定 | **引入接口迹降阶误差**（结构变硬，柔度给出下界） | 前置基准：定量评估角点迹空间对拓扑优化收敛及构型的影响 |

误差传递链条严格遵循阶梯：
$$
\text{FA (全域有限元)} \xrightarrow[\text{代数等价}]{\text{数值一致性检验}} \text{full\_trace} \xrightarrow[\text{迹降阶}]{\text{物理近似}} \text{linear\_corner} \xrightarrow[\text{代理误差}]{\text{神经网络}} \text{PIML 局部代理}
$$

---

## 2. 数值一致性检验与误差评定标准

### 2.1 `full_trace` 与 FA 的代数等价性检验

`full_trace` 与 FA 求解的是同一个离散有限元系统。在相同设计变量下，正问题位移场、伴随灵敏度及 OC 更新轨迹理论上逐步严格重合。
通过 `compare.py` 逐层核验以下五项指标：

| 层级 | 评定指标 | 容差阈值 | 数理依据 |
| :--- | :--- | :---: | :--- |
| **L0 结构性** | 迹基维度与自由度映射 | 逐位精确相等 | $\mathbf T_j = \mathbf I$，$n_{\mathrm{trace}} \equiv n_{\mathrm{boundary}}$ |
| **L1 单步一致性** | 首步未优化柔度相对偏差 | $|C^{(0)}_{\mathrm{sub}} - C^{(0)}_{\mathrm{fa}}| / C^{(0)}_{\mathrm{fa}} \le 10^{-10}$ | 线性方程组求解器与数值舍入误差 |
| **L2 轨迹一致性** | 逐迭代柔度最大相对偏差、体积分数最大绝对偏差 | 柔度 $\le 10^{-8}$；体积 $\le 10^{-10}$ | 同一离散系统 + 同一 OC/SIMP 算子 |
| **L3 拓扑一致性** | 最终设计密度场相对范数 $\|\boldsymbol{\rho}_{\mathrm{sub}} - \boldsymbol{\rho}_{\mathrm{fa}}\|$ | $L_2 \le 10^{-8}$，$L_\infty \le 10^{-8}$ | 设计变量演化轨迹一致 |
| **L4 迭代数** | 优化收敛总步数 | 完全相同 | 相同收敛判据与步长控制 |

> **说明**：`full_trace` 与 FA 属于同一离散系统的不同求解路径，理论上不存在方法误差。若 L1--L4 出现大于舍入量级的偏差，表明子结构缩聚、位移恢复或灵敏度计算的实现存在缺陷，需先行排查修正。

### 2.2 `linear_corner` 降阶误差评估

`linear_corner` 将子结构界面位移限制在角点线性插值子空间内，不再与 FA 保持严格代数等价。该工况重点考察：
1. **柔度偏差**：量化角点迹对各迭代步柔度评估的影响（理论上模型偏硬，柔度低于真值）；
2. **优化收敛性**：检查是否仍能平稳、单调收敛，是否存在数值震荡；
3. **拓扑差异**：与 `full_trace` 及 FA 最终拓扑的材料分布、构型特征进行对比。

---

## 3. 离散参数与结果汇总 (Parameters & Results)

### 3.1 离散网格与优化参数

| 工况 ID | 物理问题 | trace 模式 | 子结构划分 $n_{\mathrm{sub}}$ | 细网格剖分 $n_{\mathrm{fine}}$ | 目标体积分数 $\gamma$ | 滤波半径 $r_{\min}$ | 对应 FA 参照 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `cantilever_2d_ft` | `CantileverCorner2d` | `full_trace` | $8 \times 5$ | $10 \times 10$ (80×50 单元) | 0.50 | 3.0 | `topopt_simp_fa:cantilever_corner_2d_concentrated` |
| `cantilever_2d_lc` | `CantileverCorner2d` | `linear_corner` | $8 \times 5$ | $10 \times 10$ (80×50 单元) | 0.50 | 3.0 | 对照自身 `full_trace` 工况 |
| `mbb_3d_ft` | `FullMBBBeam3d` | `full_trace` | $12 \times 2 \times 2$ | $4 \times 4 \times 4$ (48×8×8 单元) | 0.30 | 0.4 | `topopt_simp_fa:full_mbb_3d_concentrated` (待补) |
| `mbb_3d_lc` | `FullMBBBeam3d` | `linear_corner` | $12 \times 2 \times 2$ | $4 \times 4 \times 4$ (48×8×8 单元) | 0.30 | 0.4 | 对照自身 `full_trace` 工况 |

### 3.2 运行结果与对比（待正式执行后回填）

| 工况 ID | trace | 自由度数 (细观/全局迹) | 迭代步数 | 初始柔度 $C^{(0)}$ | 最终柔度 $C^*$ | 体积分数 | 相对 FA 柔度偏差 | 判定结论 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cantilever_2d_ft` | `full_trace` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 (要求 $\le 10^{-8}$) | 待运行 |
| `cantilever_2d_lc` | `linear_corner` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 (报告降阶误差) | 待运行 |
| `mbb_3d_ft` | `full_trace` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 (要求 $\le 10^{-8}$) | 待运行 |
| `mbb_3d_lc` | `linear_corner` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 (报告降阶误差) | 待运行 |

---

## 4. 上下游协作边界与说明

1. **FA 参照对齐**：
   - 二维悬臂梁工况可直接映射至 `topopt_simp_fa` 既有的 `cantilever_corner_2d_concentrated`（需保证网格、滤波与优化参数完全一致）；
   - 三维 MBB 梁工况需在 `experiments/topopt_simp_fa/cases.toml` 中补齐同网格同参数的 `full_mbb_3d_concentrated` 工况并执行落盘，以支持三维 `full_trace` 与 FA 的对照核验。
2. **`pipeline.py` 问题分发**：
   - 目前在 `_build_problem` 中需根据工况注册表动态实例化 `CantileverCorner2d` 或 `FullMBBBeam3d`。
3. **边界条件与角点相容性**：
   - `CantileverCorner2d` 的右下角集中力天然落在右下角子结构的最外侧角点；
   - `FullMBBBeam3d` 顶面中心集中力要求在 $x$ 方向划分子结构数量为偶数（如 12），以确保载荷精确落在子结构交界角点上，避免产生内部载荷截断。
4. **与 PIML 拓扑优化的下游承接**：
   - 本模块产出的 `mbb_3d_linear_corner` 将作为 `experiments/topopt_simp_piml_substructure/` 的精确有限元真值（取代其原先在 PIML 内部内嵌的基线计算），实现代码与基准的统一解耦。
