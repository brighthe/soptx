# 胡张混合有限元算例实验分析与诊断报告 (Results & Diagnostic Analysis)

本文档记录 `soptx/examples/huzhang_elasticity` 算例在制造解收敛验证
(`minimal_demo.py`) 与 traction 载荷路径验证 (`concentrated_load_demo.py`)
下的实测结果、误差分布与诊断结论。离散采用 Hu--Zhang 混合有限元：应力
$\boldsymbol{\sigma} \in \Sigma_h \subset H(\mathrm{div})$、位移
$\boldsymbol{u} \in V_h \subset [L_2]^d$，网格为 `triangle-checkerboard`
（角点松弛要求每个几何角点恰好连接两个三角形且共享一条从角点出发的内部边）。

---

## 0. 评估指标与数学表达式 (Evaluation Metrics Formulation)

### 0.1 混合形式鞍点系统与边界条件语义

离散方程是应力--位移鞍点系统：

$$
\begin{bmatrix} A & B \\ B^{T} & 0 \end{bmatrix}
\begin{bmatrix} \boldsymbol{\sigma}_h \\ \boldsymbol{u}_h \end{bmatrix}
=
\begin{bmatrix} \boldsymbol{F}_\sigma \\ \boldsymbol{F}_u \end{bmatrix}
$$

与拉格朗日位移元相比，边界条件语义**相反**：

* **traction 边界 $\boldsymbol{\sigma}\cdot\boldsymbol{n}=\boldsymbol{t}$ 是本质边界条件**，由
  `apply_traction_boundary_condition` 通过 `set_dirichlet_bc` 强加在应力自由度上；
* **位移边界 $\boldsymbol{u}=\bar{\boldsymbol{u}}$ 是自然边界条件**，通过
  `assemble_displacement_bc_vector` 弱加进应力方程的右端项。

因此胡张元没有"装配成节点力向量"这一步，traction 载荷路径的正确性改由**边界合力守恒**度量（见 0.2）。

### 0.2 数值误差指标 (Numerical Error Metrics)

* **位移 / 应力 $L_2$ 误差**（制造解有精确解时）：
  $$
  \|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2(\Omega)},\quad
  \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2(\Omega)},\quad
  \|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2(\Omega)},\quad
  \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}
  $$
  其中 $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}
  = \big( \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}^2
  + \|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}^2 \big)^{1/2}$。
* **真相对残差**（线性系统是否真正解开）：
  $$
  r_{\mathrm{res}} = \frac{\|\boldsymbol{R}(\boldsymbol{\sigma}_h, \boldsymbol{u}_h)\|}{\|\boldsymbol{F}\|}
  $$
  阈值沿用 `experiments/huzhang_topopt_paper/cases.toml` 的 acceptance（$1\times10^{-8}$）。
* **状态矩阵对称性缺陷**：$\|A - A^{T}\|$，鞍点系统的 $(1,1)$ 块必须对称。
* **traction 边界合力守恒**（制造解无精确解判定时的载荷等效性）：
  $$
  \boldsymbol{F} = \int_{\Gamma_N} \boldsymbol{\sigma}\cdot\boldsymbol{n}\,\mathrm{d}s,\qquad
  e_{\mathrm{load}} = \frac{\|\boldsymbol{F}_{\mathrm{num}} - \boldsymbol{F}_{\mathrm{ref}}\|}{\|\boldsymbol{F}_{\mathrm{ref}}\|}
  $$
  数值合力由解出的应力场 $\boldsymbol{\sigma}_h$ 在 traction 边界边高斯点上积分得到，
  解析合力由制造解精确应力积分得到。该量是应力场的**边界泛函**（由散度定理 +
  div 交换图可证其精确等于位移边界上的反作用力误差），收敛一般**快于**应力逼近的
  阶：$k=3$ 时实测 $\sim4$ 阶、$k=4$ 时实测 $\sim6$ 阶（应力场 $L_2$ 误差为
  $\sim k+1$ 阶，见 1.1 节与第 2 节）。低阶时单独看它不够：$k=1$ 合力未达
  阈值，但默认稳定化下场误差已收敛（见 2.1）。该量由 traction 路径（本质边界
  强加）主导，实测**不随稳定化缩放变化**（k=1,2 对比旧缩放逐位一致）。判据取
  最细一层的相对差阈值（$1\times10^{-5}$）并辅以逐层递减趋势，需与场误差阶
  结合判断。

---

## 1. 制造解收敛验证基线 (minimal_demo)

默认配置：`k=3`（应力空间次数，二维胡张元 $k\ge3$ 才无需低阶稳定化）、
`plane_strain`、`scipy` 直接求解、角点松弛开启。$k\le2$ 时自动装配矩阵型跳量
惩罚稳定化，默认采用论文式物理量纲缩放 $\alpha=\mu/L_0^2\cdot h_F$
（见 [已知限制#3](#5-已知限制与未来工作)），实测恢复细层收敛。

### 1.1 mixed-sinusoidal 基线实测结果汇总 ([`MixedBoundarySinusoidalElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundarysinusoidalelasticity2d))

本节给出 `MixedBoundarySinusoidalElasticity2D` 在应力空间次数 `k=1,2,3,4`
四组实测结果（`--degree k`）。测试命令：

* **k=1**：`python examples/huzhang_elasticity/minimal_demo.py --degree 1`
* **k=2**：`python examples/huzhang_elasticity/minimal_demo.py --degree 2`
* **k=3**：`python examples/huzhang_elasticity/minimal_demo.py`
* **k=4**：`python examples/huzhang_elasticity/minimal_demo.py --degree 4`

#### k=1（应力空间次数 1，位移空间为 0 次不连续 Lagrange）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | residual | symmetry |
|---|---|---|---|---|---|---|---|---|
| 4 | 143 | 0.2500 | $6.1513\times10^{-1}$ | $1.1286\times10^{0}$ | $7.6018\times10^{0}$ | $7.6851\times10^{0}$ | $2.1\times10^{-16}$ | $1.2\times10^{-18}$ |
| 8 | 503 | 0.1250 | $3.6247\times10^{-1}$ | $3.6748\times10^{-1}$ | $3.8532\times10^{0}$ | $3.8707\times10^{0}$ | $1.6\times10^{-16}$ | $5.0\times10^{-19}$ |
| 16 | 1 895 | 0.0625 | $1.7547\times10^{-1}$ | $1.2459\times10^{-1}$ | $1.9329\times10^{0}$ | $1.9369\times10^{0}$ | $2.1\times10^{-16}$ | $5.3\times10^{-20}$ |
| 32 | 7 367 | 0.0312 | $8.5071\times10^{-2}$ | $4.3179\times10^{-2}$ | $9.6793\times10^{-1}$ | $9.6889\times10^{-1}$ | $1.9\times10^{-16}$ | $7.2\times10^{-20}$ |
| 64 | 29 063 | 0.0156 | $4.1913\times10^{-2}$ | $1.5162\times10^{-2}$ | $4.8435\times10^{-1}$ | $4.8458\times10^{-1}$ | $2.9\times10^{-16}$ | $1.6\times10^{-21}$ |

**观测收敛阶**（网格每层减半，阶数 = 误差比值以 2 为底对数）：

| 误差量 | 观测阶（nx=4→8, 8→16, 16→32, 32→64） |
|---|---|
| $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | 0.76, 1.05, 1.04, 1.02 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | 1.62, 1.56, 1.53, 1.51 |
| $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | 0.98, 1.00, 1.00, 1.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | 0.99, 1.00, 1.00, 1.00 |

一阶胡张元需要低阶稳定化；在默认矩阵型跳量惩罚（论文式物理量纲缩放
$\alpha=\mu/L_0^2\cdot h_F$，见 [已知限制#3](#5-已知限制与未来工作)）下，
位移 $L_2$ 收敛到 1 阶、应力 $L_2$ 收敛到 $\sim1.5$ 阶（超收敛，Chen--Hu--Huang
2018）、$H(\mathrm{div})$ 收敛到 1 阶，与论文表 5.2 逐格一致。首段（nx=4→8）
位移阶 0.76 偏低是粗层过渡，随后稳定在 1 阶。

#### k=2（应力空间次数 2，位移空间为 1 次不连续 Lagrange）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | residual | symmetry |
|---|---|---|---|---|---|---|---|---|
| 4 | 479 | 0.2500 | $4.5518\times10^{-2}$ | $1.4691\times10^{-1}$ | $1.7288\times10^{0}$ | $1.7350\times10^{0}$ | $1.9\times10^{-16}$ | $2.2\times10^{-18}$ |
| 8 | 1 815 | 0.1250 | $1.1490\times10^{-2}$ | $3.3842\times10^{-2}$ | $7.5773\times10^{-1}$ | $7.5849\times10^{-1}$ | $2.4\times10^{-16}$ | $8.3\times10^{-19}$ |
| 16 | 7 079 | 0.0625 | $2.8823\times10^{-3}$ | $8.1458\times10^{-3}$ | $3.6375\times10^{-1}$ | $3.6384\times10^{-1}$ | $2.0\times10^{-16}$ | $2.9\times10^{-19}$ |
| 32 | 27 975 | 0.0312 | $7.2131\times10^{-4}$ | $2.0081\times10^{-3}$ | $1.7991\times10^{-1}$ | $1.7992\times10^{-1}$ | $9.9\times10^{-16}$ | $1.1\times10^{-19}$ |
| 64 | 111 239 | 0.0156 | $1.8038\times10^{-4}$ | $4.9946\times10^{-4}$ | $8.9704\times10^{-2}$ | $8.9706\times10^{-2}$ | $5.3\times10^{-16}$ | $3.8\times10^{-20}$ |

**观测收敛阶**（网格每层减半，阶数 = 误差比值以 2 为底对数）：

| 误差量 | 观测阶（nx=4→8, 8→16, 16→32, 32→64） |
|---|---|
| $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | 1.99, 2.00, 2.00, 2.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | 2.12, 2.05, 2.02, 2.01 |
| $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | 1.19, 1.06, 1.02, 1.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | 1.19, 1.06, 1.02, 1.00 |

二阶胡张元在默认稳定化下位移与应力 $L_2$ 误差均收敛到 2 阶（与论文表 5.2
一致）；$H(\mathrm{div})$ 误差收敛到 1 阶，出现向 1 阶的降阶 —— 根源是混合
边界条件下稳定化惩罚项不施加于 $\Gamma_N$（见 [已知限制#3](#5-已知限制与未来工作)），
非实现缺陷。

#### k=3（应力空间次数 3，位移空间为 2 次不连续 Lagrange）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | residual | symmetry |
|---|---|---|---|---|---|---|---|---|
| 4 | 975 | 0.2500 | $4.3337\times10^{-3}$ | $5.7302\times10^{-3}$ | $1.2452\times10^{-1}$ | $1.2466\times10^{-1}$ | $2.0\times10^{-16}$ | $3.4\times10^{-19}$ |
| 8 | 3 767 | 0.1250 | $5.4942\times10^{-4}$ | $3.3124\times10^{-4}$ | $1.5808\times10^{-2}$ | $1.5811\times10^{-2}$ | $2.1\times10^{-16}$ | $2.5\times10^{-19}$ |
| 16 | 14 823 | 0.0625 | $6.8937\times10^{-5}$ | $1.9966\times10^{-5}$ | $1.9836\times10^{-3}$ | $1.9837\times10^{-3}$ | $3.0\times10^{-16}$ | $6.7\times10^{-20}$ |
| 32 | 58 823 | 0.0312 | $8.6254\times10^{-6}$ | $1.2276\times10^{-6}$ | $2.4819\times10^{-4}$ | $2.4819\times10^{-4}$ | $4.1\times10^{-16}$ | $1.7\times10^{-20}$ |
| 64 | 234 375 | 0.0156 | $1.0784\times10^{-6}$ | $7.6137\times10^{-8}$ | $3.1031\times10^{-5}$ | $3.1031\times10^{-5}$ | $9.7\times10^{-15}$ | $8.6\times10^{-21}$ |

**观测收敛阶**（网格每层减半，阶数 = 误差比值以 2 为底对数）：

| 误差量 | 观测阶（nx=4→8, 8→16, 16→32, 32→64） |
|---|---|
| $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | 2.98, 2.99, 3.00, 3.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | 4.11, 4.05, 4.02, 4.01 |
| $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | 2.98, 2.99, 3.00, 3.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | 2.98, 2.99, 3.00, 3.00 |

#### k=4（应力空间次数 4，位移空间为 3 次不连续 Lagrange）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | residual | symmetry |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 631 | 0.2500 | $3.7870\times10^{-4}$ | $3.8336\times10^{-4}$ | $1.0893\times10^{-2}$ | $1.0900\times10^{-2}$ | $2.4\times10^{-16}$ | $1.7\times10^{-18}$ |
| 8 | 6 359 | 0.1250 | $2.3998\times10^{-5}$ | $1.2611\times10^{-5}$ | $6.9053\times10^{-4}$ | $6.9065\times10^{-4}$ | $3.7\times10^{-16}$ | $3.7\times10^{-19}$ |
| 16 | 25 127 | 0.0625 | $1.5052\times10^{-6}$ | $4.0460\times10^{-7}$ | $4.3311\times10^{-5}$ | $4.3313\times10^{-5}$ | $4.2\times10^{-16}$ | $1.4\times10^{-19}$ |
| 32 | 99 911 | 0.0312 | $9.4157\times10^{-8}$ | $1.2777\times10^{-8}$ | $2.7093\times10^{-6}$ | $2.7094\times10^{-6}$ | $2.5\times10^{-15}$ | $8.0\times10^{-20}$ |
| 64 | 398 471 | 0.0156 | $5.8861\times10^{-9}$ | $4.0088\times10^{-10}$ | $1.6937\times10^{-7}$ | $1.6937\times10^{-7}$ | $9.9\times10^{-15}$ | $1.8\times10^{-20}$ |

**观测收敛阶**（网格每层减半，阶数 = 误差比值以 2 为底对数）：

| 误差量 | 观测阶（nx=4→8, 8→16, 16→32, 32→64） |
|---|---|
| $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | 3.98, 3.99, 4.00, 4.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | 4.93, 4.96, 4.98, 4.99 |
| $\|\mathrm{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2}$ | 3.98, 3.99, 4.00, 4.00 |
| $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ | 3.98, 4.00, 4.00, 4.00 |

#### 跨次数小结

**k=3,4 收敛性成立**：位移 $L_2$ 误差 ~$k$ 阶（位移空间是不连续 Lagrange，次数
$k-1$，对应最优阶 $k$）；应力 $L_2$ 误差 ~$k+1$ 阶（胡张元超收敛，$k=3$ 实测
$\sim4$ 阶、$k=4$ 实测 $\sim5$ 阶，文献依据为 Chen--Hu--Huang 2018（*Math. Comp.*，
Corollary 3.7(3.18)）：应力次数 $k\ge n+1$ 时
$\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{0,h}\lesssim h^{k+1}$，该网格依赖
范数强于 $L_2$，故纯 $L_2$ 应力误差亦为 $h^{k+1}$）；
$H(\mathrm{div})$ 误差由散度项主导、~$k$ 阶。超收敛经三种网格交叉验证
（checkerboard、统一对角线、随机扰动 0.1h 的 checkerboard，$k=4$ 实测应力 $L_2$ 阶
分别为 4.93 / 4.94 / 4.91），不是特定网格结构导致。

**k=1,2 已恢复收敛**：默认矩阵型跳量稳定化（论文式物理量纲缩放
$\alpha=\mu/L_0^2\cdot h_F$，见 [已知限制#3](#5-已知限制与未来工作)）下，
k=1 位移/应力/$H(\mathrm{div})$ 分别收敛到 1 / 1.5（超收敛）/ 1 阶，k=2 分别
收敛到 2 / 2 / 1 阶，与论文表 5.2 逐格一致（包括 k=2 的 $H(\mathrm{div})$ 降阶）。
此前 $k=1,2$ 细层发散（k=1 阶塌陷、k=2 的 div 负阶）的根因是稳定化缩放律错误
—— 旧缩放 $\gamma/h_F$ 与面测度 $h_F$ 抵消后净效果是 O(γ) 常数、没有论文式
的 hF 幂次，细层惩罚强度随网格不匹配，已由 $\alpha\cdot h_F$ 取代。residual 与
symmetry 两项判据在低阶始终通过，说明**求解链正确、问题仅在稳定化缩放**；
收敛性验证在 $k\ge3$ 与 $k=1,2$ 均成立。

**观测阶目前只打印不判定** —— 该实验目录记录的预期阶仍为 `theory-audit-required`，
理论核查完成前不作为通过条件；通过条件由真相对残差（$<1\times10^{-8}$）与对称性缺陷
（$<1\times10^{-12}$）守住，两者实测均达 $O(10^{-15}\sim10^{-21})$，远超门禁。

---

## 2. traction 载荷路径验证 (concentrated_load_demo)

本算例没有"装配节点力向量"这一步（traction 是本质边界条件，强加在应力自由度上），
载荷等效性改用 **traction 边界合力守恒**（见 0.2）度量。判据取最细层合力相对差
$<1\times10^{-5}$ 并辅以逐层递减趋势 —— 若 traction 数据被强加错，数值合力不会随加密
逼近解析值，逐层递减正是路径正确性的证据。

### 2.1 mixed-sinusoidal 基线实测结果汇总 ([`MixedBoundarySinusoidalElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundarysinusoidalelasticity2d))

合力相对差 $e_{\mathrm{load}}$ 是应力场的**边界泛函**（由散度定理 + div 交换图
可证其精确等于位移边界 $\Gamma_D$ 上的反作用力误差），仅作为 traction 路径的
判据（阈值 $1\times10^{-5}$ + 逐层递减），不再单独报告其收敛阶。表中输出位移
与**应力场 $L_2$ 误差**及其收敛阶（`u_order`/`s_order`）。本节给出
`MixedBoundarySinusoidalElasticity2D` 在应力空间次数 `k=1,2,3,4` 四组实测
结果（`--degree k`），其中 `s_order` 在 $k\ge3$ 时与 1.1 节应力 $L_2$ 阶一致
（胡张元超收敛，$k=3$ 为 ~4 阶、$k=4$ 为 ~5 阶）。测试命令：

* **k=1**：`python examples/huzhang_elasticity/concentrated_load_demo.py --degree 1 --levels 5`
* **k=2**：`python examples/huzhang_elasticity/concentrated_load_demo.py --degree 2 --levels 5`
* **k=3**：`python examples/huzhang_elasticity/concentrated_load_demo.py --levels 5`
* **k=4**：`python examples/huzhang_elasticity/concentrated_load_demo.py --degree 4 --levels 5`

#### k=1（应力空间次数 1）—— 合力判据未通过，场误差已收敛

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | u_order | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | s_order | residual | $e_{\mathrm{load}}$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 47 | 0.5000 | $1.1431\times10^{0}$ | n/a | $3.7890\times10^{0}$ | n/a | $1.2\times10^{-16}$ | $2.15\times10^{-1}$ |
| 4 | 143 | 0.2500 | $6.1513\times10^{-1}$ | 0.89 | $1.1286\times10^{0}$ | 1.75 | $2.1\times10^{-16}$ | $5.19\times10^{-2}$ |
| 8 | 503 | 0.1250 | $3.6247\times10^{-1}$ | 0.76 | $3.6748\times10^{-1}$ | 1.62 | $1.6\times10^{-16}$ | $1.29\times10^{-2}$ |
| 16 | 1 895 | 0.0625 | $1.7547\times10^{-1}$ | 1.05 | $1.2459\times10^{-1}$ | 1.56 | $2.1\times10^{-16}$ | $3.21\times10^{-3}$ |
| 32 | 7 367 | 0.0312 | $8.5071\times10^{-2}$ | 1.04 | $4.3179\times10^{-2}$ | 1.53 | $1.9\times10^{-16}$ | $8.03\times10^{-4}$ |

合力相对差逐层递减但收敛缓慢，最细层 $8.03\times10^{-4}$ 未达 $1\times10^{-5}$
阈值，合力判据未通过。注意 $e_{\mathrm{load}}$ 由 traction 路径（本质边界强加）
主导，实测**不随稳定化缩放变化**（与旧缩放数值逐位一致）；场精度上，默认
稳定化下 `u_order`/`s_order` 稳定在 1.04 / 1.53（应力 $L_2$ 超收敛阶成立，
与 1.1 节 k=1 一致）。合力判据单独不能代表场精度，二者需结合判断。

#### k=2（应力空间次数 2）—— 判据通过，场误差 2 阶收敛

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | u_order | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | s_order | residual | $e_{\mathrm{load}}$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 135 | 0.5000 | $1.4696\times10^{-1}$ | n/a | $5.6832\times10^{-1}$ | n/a | $1.2\times10^{-16}$ | $2.28\times10^{-3}$ |
| 4 | 479 | 0.2500 | $4.5518\times10^{-2}$ | 1.69 | $1.4691\times10^{-1}$ | 1.95 | $1.9\times10^{-16}$ | $1.35\times10^{-4}$ |
| 8 | 1 815 | 0.1250 | $1.1490\times10^{-2}$ | 1.99 | $3.3842\times10^{-2}$ | 2.12 | $2.4\times10^{-16}$ | $8.30\times10^{-6}$ |
| 16 | 7 079 | 0.0625 | $2.8823\times10^{-3}$ | 2.00 | $8.1458\times10^{-3}$ | 2.05 | $2.0\times10^{-16}$ | $5.17\times10^{-7}$ |
| 32 | 27 975 | 0.0312 | $7.2131\times10^{-4}$ | 2.00 | $2.0081\times10^{-3}$ | 2.02 | $9.9\times10^{-16}$ | $3.23\times10^{-8}$ |

合力差收敛到最细层 $3.23\times10^{-8}$（~4 阶），判据通过；默认稳定化下
`u_order`/`s_order` 稳定在 2.00 / 2.02，场误差 2 阶收敛（与 1.1 节 k=2 一致）。
合力判据与场误差阶在此一致，traction 路径与稳定化均验证通过。

#### k=3（应力空间次数 3）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | u_order | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | s_order | residual | $e_{\mathrm{load}}$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 263 | 0.5000 | $4.5705\times10^{-2}$ | n/a | $1.2638\times10^{-1}$ | n/a | $1.6\times10^{-16}$ | $1.00\times10^{-3}$ |
| 4 | 975 | 0.2500 | $4.3337\times10^{-3}$ | 3.40 | $5.7302\times10^{-3}$ | 4.46 | $2.0\times10^{-16}$ | $5.97\times10^{-5}$ |
| 8 | 3 767 | 0.1250 | $5.4942\times10^{-4}$ | 2.98 | $3.3124\times10^{-4}$ | 4.11 | $2.1\times10^{-16}$ | $3.69\times10^{-6}$ |
| 16 | 14 823 | 0.0625 | $6.8937\times10^{-5}$ | 2.99 | $1.9966\times10^{-5}$ | 4.05 | $3.0\times10^{-16}$ | $2.30\times10^{-7}$ |
| 32 | 58 823 | 0.0312 | $8.6254\times10^{-6}$ | 3.00 | $1.2276\times10^{-6}$ | 4.02 | $4.1\times10^{-16}$ | $1.43\times10^{-8}$ |

合力相对差从最粗层 $1.0\times10^{-3}$ 严格逐层递减到最细层 $1.4\times10^{-8}$，
**应力场 $L_2$ 误差阶 `s_order` 稳定在 $\sim4.0$**（与 1.1 节 k=3 的应力 $L_2$
超收敛阶一致，胡张元超收敛 $h^{k+1}$）。位移 $L_2$ 误差同步输出，**位移阶收敛到
$3.0$**（位移空间是 $k-1=2$ 次不连续 Lagrange，对应理论阶 $k=3$，见 1.1 节）。

#### k=4（应力空间次数 4）

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | u_order | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | s_order | residual | $e_{\mathrm{load}}$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 431 | 0.5000 | $1.6863\times10^{-3}$ | n/a | $2.5984\times10^{-3}$ | n/a | $2.1\times10^{-16}$ | $8.43\times10^{-6}$ |
| 4 | 1 631 | 0.2500 | $3.7870\times10^{-4}$ | 2.15 | $3.8336\times10^{-4}$ | 2.76 | $2.4\times10^{-16}$ | $1.24\times10^{-7}$ |
| 8 | 6 359 | 0.1250 | $2.3998\times10^{-5}$ | 3.98 | $1.2611\times10^{-5}$ | 4.93 | $3.7\times10^{-16}$ | $1.90\times10^{-9}$ |
| 16 | 25 127 | 0.0625 | $1.5052\times10^{-6}$ | 3.99 | $4.0460\times10^{-7}$ | 4.96 | $4.2\times10^{-16}$ | $2.96\times10^{-11}$ |
| 32 | 99 911 | 0.0312 | $9.4157\times10^{-8}$ | 4.00 | $1.2777\times10^{-8}$ | 4.98 | $2.5\times10^{-15}$ | $4.63\times10^{-13}$ |

合力差收敛到最细层 $4.6\times10^{-13}$，仅作 traction 路径判据。**应力场 $L_2$
误差阶 `s_order` 稳定在 $\sim5.0$**（胡张元超收敛 $h^{k+1}$，与 1.1 节 k=4
一致）。位移阶收敛到 $4.0$（位移空间是 $k-1=3$ 次不连续 Lagrange，对应理论阶
$k=4$）。

（收敛阶记录为 `theory-audit-required`，只展示不判定。）

### 2.2 mixed-exp-sine 实测结果汇总 ([`MixedBoundaryExponentialSineElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundaryexponentialsineelasticity2d))

* **测试命令**：`python examples/huzhang_elasticity/concentrated_load_demo.py --problem mixed-exp-sine`

| nx | gdof | h | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ | u_order | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ | s_order | residual | $e_{\mathrm{load}}$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 263 | 0.5000 | $3.1691\times10^{-2}$ | n/a | $7.8657\times10^{-2}$ | n/a | $5.3\times10^{-16}$ | $9.45\times10^{-4}$ |
| 4 | 975 | 0.2500 | $3.0778\times10^{-3}$ | 3.36 | $3.9607\times10^{-3}$ | 4.31 | $4.1\times10^{-16}$ | $5.66\times10^{-5}$ |
| 8 | 3 767 | 0.1250 | $3.9026\times10^{-4}$ | 2.98 | $2.3548\times10^{-4}$ | 4.07 | $5.9\times10^{-16}$ | $3.51\times10^{-6}$ |
| 16 | 14 823 | 0.0625 | $4.8959\times10^{-5}$ | 2.99 | $1.4338\times10^{-5}$ | 4.04 | $6.1\times10^{-16}$ | $2.19\times10^{-7}$ |
| 32 | 58 823 | 0.0312 | $6.1253\times10^{-6}$ | 3.00 | $8.8583\times10^{-7}$ | 4.02 | $1.8\times10^{-15}$ | $1.37\times10^{-8}$ |

### 2.3 变体回归

| 配置 | 最细层 $e_{\mathrm{load}}$ | 递减 | 结论 |
|---|---|---|---|
| `--no-relaxation`（nx=8） | $3.49\times10^{-6}$ | 是 | 通过 |
| `--levels 5 --save-vtu`（nx=32） | $1.43\times10^{-8}$ | 是 | 通过 |

---

## 3. 与拉格朗日位移元对比 (vs. `examples/lagrange_elasticity`)

| 维度 | 胡张混合元 (`huzhang_elasticity`) | 拉格朗日位移元 (`lagrange_elasticity`) |
|---|---|---|
| 离散形式 | $\boldsymbol{\sigma}\in H(\mathrm{div})$，$\boldsymbol{u}\in[L_2]^d$ 鞍点系统 | $\boldsymbol{u}\in[H^1]^d$ 单场 |
| 边界条件语义 | **traction 本质、位移自然**（语义相反） | 位移本质、traction 自然 |
| 载荷路径 | 无节点力向量；traction 强加在应力自由度 | 装配进右端项，`sum(F)-P` 校验 |
| 求解器 | 鞍点系统不能用 CG，仅 `scipy`/`mumps` | 可 CG，也可直接法 |
| 位移空间 | 不连续 Lagrange（VTU 导出需节点插值） | 连续 Lagrange（直接映射节点） |
| 应力求解 | 直接解出 $\boldsymbol{\sigma}_h$（含 $\mathrm{div}$ 强约束） | 需后处理由位移梯度导出 |
| 网格约束 | `triangle-checkerboard` + 角点松弛 | 任意四边形/三角形网格 |

**实测收敛阶对比**（位移 $L_2$ 误差）：同域不同问题下，拉格朗日元 `p=1`
（[`SinusoidalPlaneStrainElasticity2D`](../../docs/problems/manufactured-elasticity.md#sinusoidalplanestrainelasticity2d)，全 Dirichlet）观测阶 ~1.94/1.98（理论 2 阶）；
胡张元 `k=3` 时位移空间为 $k-1=2$ 次不连续 Lagrange，观测阶 3.00。两者都达到
各自空间次数对应的理论最优阶。

---

## 4. 可视化格式标准：VTU 导出 + VTK 离屏渲染

可视化管线统一收口到 `soptx.visualization`（该命名空间为仓库预留的可选可视化模块，
`import soptx` 不触发 `pyevtk`/`vtk` 加载）：

* **VTU 导出**：`soptx.visualization.vtk_export.write_vtu`（通用，自动识别 2D 三角形/
  四边形与 3D 四面体）与 `export_vtu`（单位移场便捷封装，字段 `u_x`/`u_y`/`u_mag`）。
  胡张元的位移空间是不连续 Lagrange，导出前需经 `extract_nodal_displacement` 做
  单元顶点求值 + 节点平均。
* **离屏渲染**：`soptx.visualization.vtk_render`（`load_vtu`/`create_warped_actor`/
  `render_and_save`），直接导出 Warp 变形 PNG，无需 ParaView GUI。

生成命令：

```bash
python examples/huzhang_elasticity/concentrated_load_demo.py --levels 5 --save-vtu
python examples/huzhang_elasticity/render_warped.py
```

产物：`outputs/vtu/mixed-sinusoidal_p3_tri_32x32.vtu` 与
`outputs/figures/huzhang_warped.png`（制造解域 $(0,1)^2$，位移幅值范围
$[1.2\times10^{-6},\,1.41]$，单位缩放位移即为变形量）。

---

## 5. 已知限制与未来工作

1. **全 Dirichlet 制造解未开放**：`sinusoidal`/`exp-sine` 全位移边界问题理论上可走
   `AllDisplacementBoundaryMixin` 的混合形式边界接口，但该路径未经充分测试，暂不开放。
2. **收敛阶只打印不判定**：观测阶记录为 `theory-audit-required`，理论核查完成前不设
   通过门槛；当前通过条件由真相对残差 + 对称性缺陷 + 合力守恒守住。
3. **低阶稳定化（矩阵型跳量惩罚，已恢复 k=1,2 收敛）**：二维胡张元 $k<3$ 时需
   低阶稳定化（完整符号—代码映射见 [math_spec.md](math_spec.md)）。代码对
   $k\le2$ 装配 `JumpPenaltyIntegrator` 的 `matrix_jump` 变体：
   跳量 $[[\boldsymbol{u}]]=\tfrac12(\boldsymbol{u}\boldsymbol{\nu}^T
   +\boldsymbol{\nu}\boldsymbol{u}^T)$，惩罚项
   $c(\boldsymbol{u}_h,\boldsymbol{v}_h)=\sum_F \alpha\,h_F\int_F
   [[\boldsymbol{u}_h]]:[\boldsymbol{v}_h]\,\mathrm{d}s$，系数默认论文式物理量纲缩放
   $\alpha=\mu/L_0^2$（$L_0$ 为计算域特征尺度，单位域上 $\alpha=\mu$），加在内部面
   与位移边界面上、不施加于 $\Gamma_N$，进入鞍点系统 $(2,2)$ 块取负号。实测
   $k=1,2$ 细层收敛恢复：k=1 位移/应力/$H(\mathrm{div})$ 阶 1 / 1.5 / 1，k=2 阶
   2 / 2 / 1（与论文表 5.2 逐格一致）；$k=2$ 的 $H(\mathrm{div})$ 降阶是混合边界
   下的预期现象（惩罚不加 $\Gamma_N$ + 非多项式牵引在 $\Gamma_N$ 的投影误差），
   非实现缺陷。早期 $\gamma/h_F$ 缩放（$\gamma=0.01E$（k=1）/ $0.01\mu$（k≥2））
   净效果 O(γ)、细层阶坍塞/发散，已从默认路径弃用，`penalty_scaling='gamma_hinv'`
   保留作回归对比。
4. **网格限制**：角点松弛要求每个几何角点恰好连接两个三角形，因此固定使用
   `triangle-checkerboard`；这是当前软件实现的支持范围，不是对胡--张方法数学理论的永久限制。
5. **3D 未开放**：`soptx` 的 `HuZhangFESpace3D` 与 `huzhang_fe_space_3d.py` 已存在，
   但 `examples/huzhang_elasticity` 当前仅覆盖 2D，3D 混合元验证留作未来工作。
