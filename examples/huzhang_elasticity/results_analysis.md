# 胡张混合有限元算例实验分析与诊断报告 (Results & Diagnostic Analysis)

本文档记录 `soptx/examples/huzhang_elasticity` 算例在制造解收敛验证
(`manufactured_convergence_demo.py`) 与集中力工程基准 (`concentrated_load_demo.py`)
下的实测结果、误差分布与诊断结论. 离散采用 Hu--Zhang 混合有限元: 应力
$\boldsymbol{\sigma} \in \Sigma_h \subset H(\mathrm{div})$、位移
$\boldsymbol{u} \in V_h \subset [L_2]^d$, 网格为 `triangle-checkerboard`
(角点松弛要求每个几何角点恰好连接两个三角形且共享一条从角点出发的内部边).


## 0. 目录范围

本目录只保存 Hu--Zhang 单元、边界载荷和小规模拓扑优化 smoke 校验的验证记录. 投稿论文的完整参数矩阵、历史脚本快照、运行清单与论文证据均位于 `experiments/huzhang_topopt_paper/`, 不得将两类结果混用.


## 1. 评估指标与数学表达式 (Evaluation Metrics Formulation)

### 1.1 混合形式鞍点系统与边界条件语义

离散方程是应力--位移鞍点系统:

$$
\begin{bmatrix} A & B \\ B^{T} & 0 \end{bmatrix}
\begin{bmatrix} \boldsymbol{\sigma}_h \\ \boldsymbol{u}_h \end{bmatrix}
=
\begin{bmatrix} \boldsymbol{F}_\sigma \\ \boldsymbol{F}_u \end{bmatrix}
$$

与拉格朗日位移元相比, 边界条件语义**相反**:

* **traction 边界 $\boldsymbol{\sigma}\cdot\boldsymbol{n}=\boldsymbol{t}$ 是本质边界条件**, 由
  `apply_traction_boundary_condition` 通过 `set_dirichlet_bc` 强加在应力自由度上;
* **位移边界 $\boldsymbol{u}=\bar{\boldsymbol{u}}$ 是自然边界条件**, 通过
  `assemble_displacement_bc_vector` 弱加进应力方程的右端项.

因此胡张元没有 "装配成节点力向量" 这一步, traction 载荷路径的正确性改由**边界合力守恒**度量 (见 1.2 节).

### 1.2 数值误差指标 (Numerical Error Metrics)

* **位移 / 应力 $L_2$ 误差与组合 $H(\mathrm{div})$ 误差** (制造解有精确解时):

  $$
  \|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2(\Omega)},\quad
  \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2(\Omega)},\quad
  \|\operatorname{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2(\Omega)}
  $$

  其中组合 $H(\operatorname{div})$ 误差定义为:

  $$
  \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\operatorname{div})} = \left( \|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2(\Omega)}^2 + \|\operatorname{div}(\boldsymbol{\sigma}-\boldsymbol{\sigma}_h)\|_{L_2(\Omega)}^2 \right)^{1/2}
  $$
* **真相对残差** (线性系统是否真正解开):
  $$
  r_{\mathrm{res}} = \frac{\|\boldsymbol{R}(\boldsymbol{\sigma}_h, \boldsymbol{u}_h)\|}{\|\boldsymbol{F}\|}
  $$
  阈值沿用 `experiments/huzhang_topopt_paper/cases.toml` 的 acceptance ($1\times10^{-8}$).
* **状态矩阵对称性缺陷**: $\|A - A^{T}\|$, 鞍点系统的 $(1,1)$ 块必须对称.
* **traction 边界合力守恒** (制造解无精确解判定时的载荷等效性):
  $$
  \boldsymbol{F} = \int_{\Gamma_N} \boldsymbol{\sigma}\cdot\boldsymbol{n}\,\mathrm{d}s,\qquad
  e_{\mathrm{load}} = \frac{\|\boldsymbol{F}_{\mathrm{num}} - \boldsymbol{F}_{\mathrm{ref}}\|}{\|\boldsymbol{F}_{\mathrm{ref}}\|}
  $$
  数值合力由解出的应力场 $\boldsymbol{\sigma}_h$ 在 traction 边界边高斯点上积分得到,
  解析合力由制造解精确应力积分得到. 该量是应力场的**边界泛函** (由散度定理 +
  div 交换图可证其精确等于位移边界上的反作用力误差), 收敛一般**快于**应力逼近的
  阶: $k=3$ 时实测 $\sim4$ 阶、$k=4$ 时实测 $\sim6$ 阶 (应力场 $L_2$ 误差为
  $\sim k+1$ 阶, 见 2.1 节与第 4 节). 低阶时单独看它不够: $k=1$ 合力未达
  阈值, 但默认稳定化下场误差已收敛. 该量由 traction 路径 (本质边界
  强加) 主导, 实测**不随稳定化缩放变化** (k=1,2 对比旧缩放逐位一致). 判据取
  最细一层的相对差阈值 ($1\times10^{-5}$) 并辅以逐层递减趋势, 需与场误差阶结合判断.

### 1.3 符号—代码映射与离散实现 (Symbol-to-Code Mapping)

| 数学符号 | 代码位置 | 含义 |
|---|---|---|
| 应力空间次数 $k$ | `HuZhangMFEMAnalyzer(space_degree=k)` → `HuZhangFESpace(p=k)` | $k\le2$ 时装配稳定化 |
| $\boldsymbol{\sigma}_h\in\Sigma_h$ | `HuZhangFESpace` | 应力有限元空间 ($H(\mathrm{div})$) |
| $\boldsymbol{u}_h\in V_h$ | `TensorFunctionSpace(scalar_space=..., shape=(-1, GD))` | 位移有限元空间 (不连续 Lagrange) |
| 柔度矩阵 $A$ | `HuZhangStressIntegrator(lambda0, lambda1, method='fast')` | 柔度张量积分块 |
| 耦合矩阵 $B$ | `HuZhangMixIntegrator` / `_cached_mix_matrix` | 应力—位移散度耦合块 |
| 边界语义 | `_essential_bc` (traction $\Gamma_N$) / `_natural_bc` (位移 $\Gamma_D$) | traction 本质强施加, 位移弱加进右端项 |
| 矩阵跳量 $[[\boldsymbol w]]$ | [`jump_penalty_integrator.py:_fetch_matrix_jump`](../../src/soptx/fem/integrators/jump_penalty_integrator.py) | 对称梯度型跳量 |
| 跳量惩罚 $c(\cdot,\cdot)$ | `JumpPenaltyIntegrator(penalty_scaling='physical_h')` | 论文式物理量纲缩放 $\alpha\cdot h_F$ ($\alpha=\mu/L_0^2$) |
| 稳定化块 $-J$ | `bmat([[A, B], [B.T, -J]])` | 稳定化进入 $(2,2)$ 块 (取负号) |

---

## 2. 制造解收敛验证基线 (manufactured_convergence_demo)

制造解算例拥有解析解, 判据为 **$L_2$ 收敛阶 + 鞍点真相对残差 + 状态矩阵对称性** (详见
[README.md](README.md#1-制造解收敛验证-manufactured_convergence_demo)). 物理问题为
[`MixedBoundarySinusoidalElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundarysinusoidalelasticity2d):
$[0,1]\times[0,1]$ 单位正方形域, 混合边界条件 (顶底位移弱施加, 左右 traction 本质强加), 平面应变假设.

应力空间 $\Sigma_h$ 采用 $k$ 次胡张元, 位移空间 $V_h$ 采用 $k-1$ 次分片不连续 Lagrange 元. $k\ge3$ 时自然满足 inf-sup 条件; $k\le2$ 时自动装配矩阵型跳量惩罚项 (采用物理量纲缩放 $\alpha=\mu/L_0^2\cdot h_F$), 消除低阶压力奇异性并恢复最优收敛阶. 棋盘格角点松弛 (Corner Relaxation) 消除几何角点处的应力过约束.

**实测** (`python examples/huzhang_elasticity/manufactured_convergence_demo.py --degree <k>`,
`triangle-checkerboard` 多层加密 $4\times4 \to 64\times64$、角点松弛开启、`scipy` 直接法):

| 空间次数 $k$ | 位移空间 $V_h$ | $\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L_2}$ 渐近阶 | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{L_2}$ 渐近阶 | $\|\boldsymbol{\sigma}-\boldsymbol{\sigma}_h\|_{H(\mathrm{div})}$ 渐近阶 | 最细网格 ($64\times64$) 残差 | 对称性缺陷 |
|---|---|---|---|---|---|---|
| 1 | $P_0$ (稳定化) | 1.02 ($O(h)$) | 1.51 ($O(h^{1.5})$ 超收敛) | 1.00 ($O(h)$) | $2.9\times10^{-16}$ | $1.6\times10^{-21}$ |
| 2 | $P_1$ (稳定化) | 2.00 ($O(h^2)$) | 2.01 ($O(h^2)$) | 1.00 (边界降阶) | $5.3\times10^{-16}$ | $3.8\times10^{-20}$ |
| 3 | $P_2$ (无稳定化) | 3.00 ($O(h^3)$) | 4.01 ($O(h^4)$ 超收敛) | 3.00 ($O(h^3)$) | $9.7\times10^{-15}$ | $8.6\times10^{-21}$ |
| 4 | $P_3$ (无稳定化) | 4.00 ($O(h^4)$) | 4.99 ($O(h^5)$ 超收敛) | 4.00 ($O(h^4)$) | $9.9\times10^{-15}$ | $1.8\times10^{-20}$ |

其中 $k=3,4$ 高阶元位移达到最优 $O(h^k)$ 阶、应力呈现 $O(h^{k+1})$ 超收敛阶 (文献依据 Chen--Hu--Huang 2018 *Math. Comp.*, Corollary 3.7(3.18)), 组合 $H(\mathrm{div})$ 误差稳定达到最优 $O(h^k)$ 阶. $k=1,2$ 低阶元在物理量纲跳量稳定化下恢复理论收敛阶 (与博士论文表 5.2 完全一致). 鞍点相对残差 ($\le 10^{-14}$) 与状态矩阵对称性缺陷 ($\le 10^{-19}$) 全部大幅超越门禁要求 ($10^{-8}$ 与 $10^{-12}$).

---

## 3. 集中力工程基准 (concentrated_load_demo)

工程算例没有解析解, 判据由收敛阶改为**载荷路径**: 真相对残差 + 结构合力守恒 (详见
[README.md](README.md#2-集中力工程基准-concentrated_load_demo)). 物理问题为
[`FixedFixedBeamCenterLoad2d`](../../docs/problems/engineering-benchmarks.md#fixedfixedbeamcenterload2d):
$160\times20$ 全域两端固支梁, 底边中点长度 $l=1$ 的贴片上作用 $t=P/l$, $P=-3\,\mathrm{N}$.

两条离散链共用同一载荷函数——原始贴片牵引在底边连续 P1 迹空间上的 L2 投影
$t_h$ ([`project_patch_traction_to_p1_trace`](../../src/soptx/fem/boundary_loads.py)).
常数落在该空间内, 故 $\int_{\Gamma_N}t_h\,\mathrm{d}s=P$ 精确成立; 连续分片线性函数
又能被胡张元迹插值精确重现、被 $q=2k+2$ 的高斯积分精确积分, 因此两种方法
看到的是同一个载荷泛函, 柔顺度差异可以归因于离散格式本身.

**实测** (`python examples/huzhang_elasticity/concentrated_load_demo.py --model fixed-fixed --nx 160 --ny 20 --degrees 1 2 3`,
`triangle-checkerboard` $160\times20$、6400 单元、角点松弛开启、`scipy` 直接法):

| method | k | q | gdof | residual | $P_y$(结构) | 被吞载荷 |
|---|---|---|---|---|---|---|
| lfem | 1 | 4 | 6 762 | $1.16\times10^{-12}$ | -3 | $7.52\times10^{-60}$ |
| huzhang | 1 | 4 | 22 947 | $7.36\times10^{-14}$ | -3 | - |
| lfem | 2 | 6 | 26 322 | $3.99\times10^{-12}$ | -3 | $-1.52\times10^{-46}$ |
| huzhang | 2 | 6 | 87 307 | $8.16\times10^{-14}$ | -3 | - |
| lfem | 3 | 8 | 58 682 | $9.88\times10^{-12}$ | -3 | $-6.84\times10^{-47}$ |
| huzhang | 3 | 8 | 183 667 | $6.47\times10^{-14}$ | -3 | - |

其中 $P_y$(结构) 对胡张元取 $\int_{\Gamma_N}\boldsymbol\sigma_h\cdot\boldsymbol n\,\mathrm{d}s$,
对 LFEM 取支座反力 $-\sum(Ku)|_{\Gamma_D}$ —— 它抓的是 "载荷落在被强加自由度上被静默吞掉"
这一类错误, 此时残差依然为 0, 其余判据全都看不见. 载荷函数本身的守恒性
($\int t_h=P$、一阶矩保持) 由 `tests/unit/test_p1_trace_load_projection.py` 覆盖,
demo 只在表头打印一次 `common_load.resultant()` 作为对照, 不重复算边界积分.

能量诊断 (互补能、耦合项、牵引对偶功) 属于同一批诊断量, 在
`experiments/huzhang_topopt_paper/run.py --mode state-compare` 中报告, 本目录不重复.

**P1 投影的振荡尾**: 局部贴片的 P1 投影在贴片外按 P1 质量矩阵的 Green 函数几何衰减
(每单元约 $0.27$, 正负交替). 网格过粗时尾部触及固支端, 那部分载荷被 Dirichlet 自由度
真实吞掉, 量级约 $P\cdot0.27^{n_x/2}$. demo 把它作为结果表的最后一列 (该诊断只对 LFEM
可算, 胡张元行显示 `-`): 上表 $n_x=160$ 下衰减至 $10^{-46}\sim 10^{-60}$, 完全降为 0;
$n_x$ 取到 20 以下时该项会升到 $10^{-6}$ 量级并触发告警, 这是物理事实而非实现缺陷.

优化侧在 $\rho=0.4$ + msimp 插值下的柔顺度对比属于**密度相关**部分, 留在
`experiments/huzhang_topopt_paper/ --mode state-compare`, 本目录不重复.

---

## 4. 与拉格朗日位移元对比 (vs. `examples/lagrange_elasticity`)

| 维度 | 胡张混合元 (`huzhang_elasticity`) | 拉格朗日位移元 (`lagrange_elasticity`) |
|---|---|---|
| 离散形式 | $\boldsymbol{\sigma}\in H(\mathrm{div})$, $\boldsymbol{u}\in[L_2]^d$ 鞍点系统 | $\boldsymbol{u}\in[H^1]^d$ 单场 |
| 边界条件语义 | **traction 本质、位移自然** (语义相反) | 位移本质、traction 自然 |
| 载荷路径 | 无节点力向量; traction 强加在应力自由度 | 装配进右端项, `sum(F)-P` 校验 |
| 求解器 | 鞍点系统不能用 CG, 仅 `scipy`/`mumps` | 可 CG, 也可直接法 |
| 位移空间 | 不连续 Lagrange (VTU 导出需节点插值) | 连续 Lagrange (直接映射节点) |
| 应力求解 | 直接解出 $\boldsymbol{\sigma}_h$ (含 $\mathrm{div}$ 强约束) | 需后处理由位移梯度导出 |
| 网格约束 | `triangle-checkerboard` + 角点松弛 | 任意四边形/三角形网格 |

**实测收敛阶对比** (位移 $L_2$ 误差): 同域不同问题下, 拉格朗日元 `p=1`
([`SinusoidalPlaneStrainElasticity2D`](../../docs/problems/manufactured-elasticity.md#sinusoidalplanestrainelasticity2d), 全 Dirichlet) 观测阶 ~1.94/1.98 (理论 2 阶);
胡张元 `k=3` 时位移空间为 $k-1=2$ 次不连续 Lagrange, 观测阶 3.00. 两者都达到
各自空间次数对应的理论最优阶.