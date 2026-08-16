# 胡张混合有限元 2D 求解算例 (Hu--Zhang Mixed FEM)

本目录提供基于 FEALPy 的二维线弹性 Hu--Zhang 混合有限元求解器：应力
$\boldsymbol{\sigma}\in\Sigma_h\subset H(\mathrm{div})$ 与位移
$\boldsymbol{u}\in V_h\subset[L_2]^d$ 联合求解，离散成应力--位移鞍点系统。

算例采用自包含的单文件代码设计，方便快速阅读与二次扩展。

---

## 算例核心文件结构

```text
soptx/examples/huzhang_elasticity/
├── manufactured_convergence_demo.py  <-- [核心代码] 制造解收敛验证 (L2 观测阶 + 残差 + 对称性)
├── concentrated_load_demo.py         <-- [核心代码] 集中力工程基准 (残差 + 载荷等效性 + 结构合力)
├── results_analysis.md               <-- [实验分析] 符号—代码映射、实测数据表与诊断报告
├── README.md                         <-- [使用说明] 本文件
└── outputs/                          <-- [测试成果与报告]
    ├── figures/                      <-- 离屏渲染截图 (.png)
    └── vtu/                          <-- 自动导出的 ParaView 场文件 (.vtu)
```

---

## 数学模型与物理控制方程

强形式控制方程与 Hooke 本构如下：

$$
-\nabla\cdot\boldsymbol\sigma(\boldsymbol u)=\boldsymbol b,\qquad
\boldsymbol\varepsilon(\boldsymbol u)=\frac12(\nabla\boldsymbol u+\nabla\boldsymbol u^{\mathsf T}),
$$

$$
\boldsymbol\sigma=\lambda\operatorname{tr}(\boldsymbol\varepsilon)\mathbf I
+2\mu\boldsymbol\varepsilon,\qquad
\boldsymbol u=\bar{\boldsymbol u}\quad\text{on }\partial\Omega.
$$

**Hu--Zhang 混合弱形式**：找 $(\boldsymbol\sigma_h,\boldsymbol u_h)\in\Sigma_h\times V_h$，使对所有
$(\boldsymbol\tau_h,\boldsymbol v_h)\in\Sigma_h\times V_h$ 成立：

$$
\int_\Omega (A\boldsymbol\sigma_h):\boldsymbol\tau_h\,\mathrm{d}x
-\int_\Omega \boldsymbol u_h\cdot(\mathrm{div}\,\boldsymbol\tau_h)\,\mathrm{d}x
=-\int_{\Gamma_D}\bar{\boldsymbol u}\cdot(\boldsymbol\tau_h\boldsymbol n)\,\mathrm{d}s,
$$

$$
\int_\Omega (\mathrm{div}\,\boldsymbol\sigma_h)\cdot\boldsymbol v_h\,\mathrm{d}x
=-\int_\Omega \boldsymbol b\cdot\boldsymbol v_h\,\mathrm{d}x,
$$

其中 $A$ 为柔度张量。与拉格朗日位移元相比，**边界条件语义相反**：

* **traction 边界 $\boldsymbol\sigma\cdot\boldsymbol n=\boldsymbol t$ 是本质边界条件**，强加在应力自由度上；
* **位移边界 $\boldsymbol u=\bar{\boldsymbol u}$ 是自然边界条件**，经 $\int_{\Gamma_D}\bar{\boldsymbol u}\cdot(\boldsymbol\tau\boldsymbol n)\,\mathrm{d}s$ 弱加进应力方程右端项。

求解器限制：鞍点系统 $[[A,B],[B^{T},0]]$ 对称不定，不能用 CG，仅支持 `scipy`/`mumps`。

制造解问题（`p=3` 默认，位移空间为 $p-1=2$ 次不连续 Lagrange）：

* **mixed-sinusoidal**（默认，[`MixedBoundarySinusoidalElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundarysinusoidalelasticity2d)）：精确位移 $u_1=u_2=\sin(\pi x)\sin(\pi y)$；
* **mixed-exp-sine**（[`MixedBoundaryExponentialSineElasticity2D`](../../docs/problems/manufactured-elasticity.md#mixedboundaryexponentialsineelasticity2d)）：指数/正弦制造解。

各制造解的完整数学定义、体力表达式与边界条件详见：
👉 [制造解文档](../../docs/problems/manufactured-elasticity.md)

低阶稳定化 ($k<3$ 时 $\Sigma_h\times V_h$ 不满足离散 inf-sup) 的完整数学过程与
缩放律论证详见方法实现文档 [`docs/fem/huzhang-mixed-fem-implementation.md`](../../docs/fem/huzhang-mixed-fem-implementation.md),
符号—代码映射与基线实测数据详见:
👉 [results_analysis.md](results_analysis.md)

---

## 快速开始 (CLI 使用指南)

### 1. 制造解收敛验证 (manufactured_convergence_demo)

```bash
# 默认正弦制造解: k=3, 5 层网格加密
python examples/huzhang_elasticity/manufactured_convergence_demo.py

# 指数正弦制造解: k=2 (低阶跳量稳定化)
python examples/huzhang_elasticity/manufactured_convergence_demo.py --model mixed-exp-sine --degree 2

# 关闭角点松弛对比
python examples/huzhang_elasticity/manufactured_convergence_demo.py --no-relaxation
```

### 可选参数选项

* `--model`：制造解模型名称 (默认 `mixed-sinusoidal`, 可选 `mixed-exp-sine`);
* `--degree`：应力空间有限元次数 (默认 3; $k\le2$ 时自动装配跳量稳定化);
* `--levels`：网格加密层数 (默认 5, 对应剖分 $4\times4 \to 64\times64$);
* `--relaxation` / `--no-relaxation`：角点松弛开关 (默认开启);
* `--solver`：直接线性求解器 (`scipy` 默认, 或 `mumps`).

### 2. 集中力工程基准 (concentrated_load_demo)

无解析解的工程算例：[`FixedFixedBeamCenterLoad2d`](../../docs/problems/engineering-benchmarks.md#fixedfixedbeamcenterload2d)，
$160\times20$ 全域两端固支梁，底边中点长度 $l=1\,\mathrm{mm}$ 的贴片上作用等效均布牵引
$t=P/l$，合力 $P=-3\,\mathrm{N}$。判据不再是收敛阶，而是**载荷路径**：

* **真相对残差** —— 各自的线性系统确实解开了；
* **结构合力守恒** —— 从解出的场反算真正传进结构的力：胡张元取
  $\int_{\Gamma_N}\boldsymbol\sigma_h\cdot\boldsymbol n\,\mathrm{d}s$，LFEM 取支座反力
  $\sum(Ku)|_{\Gamma_D}$，两者都必须等于 $P$。

第二条是端到端判据。载荷函数本身是否守恒（$\int t_h=P$、一阶矩保持）由
[`tests/unit/test_p1_trace_load_projection.py`](../../tests/unit/test_p1_trace_load_projection.py)
保证，不在 demo 里重算；而载荷若在装配或边界处理中丢失，残差依然为 0，只有结构合力能抓到。
胡张元位移属 $L_2$ 空间、不能直接解释为边界外力功，相关的能量诊断（互补能、耦合项、
牵引对偶功）由 [`experiments/huzhang_topopt_paper/run.py --mode state-compare`](../../experiments/huzhang_topopt_paper/run.py)
报告，本 demo 不重复。

原始阶跃牵引在贴片边缘的跳变一般落在单元内部，跨边连续的胡张元迹空间在跳变点上
只有一个单值自由度，加密网格也装不下这个跳跃。因此两条离散链共同使用该牵引在底边
连续 P1 迹空间上的 L2 投影（[`project_patch_traction_to_p1_trace`](../../src/soptx/fem/boundary_loads.py)）：
常数落在该空间内，投影精确保持合力；连续分片线性函数又能被胡张元的迹插值精确重现、
被 $q=2k+2$ 的高斯积分精确积分。

```bash
# 默认快速验证: fixed-fixed 模型, 80x10 网格, k=2, 3 次数
python examples/huzhang_elasticity/concentrated_load_demo.py

# 论文精细基准: 160x20 网格, k=1, 2, 3, 4 次数
python examples/huzhang_elasticity/concentrated_load_demo.py --model fixed-fixed --nx 160 --ny 20 --degrees 1 2 3 4

# 使用 MUMPS 直接求解器
python examples/huzhang_elasticity/concentrated_load_demo.py --solver mumps
```

### 可选参数选项

* `--model`：工程物理模型名称 (默认 `fixed-fixed`, 对应两端固支中点受载梁);
* `--degrees`：参与成对比较的空间有限元次数列表 (默认 `2 3`, 允许 1/2/3/4; 同一 $k$ 下
  LFEM 用 $p=k$、胡张元用应力空间次数 $k$, 统一积分阶 $q=2k+2$);

> $k=1$（$P_1$ 应力 / $P_0$ 位移）在本算例开放：实体材料单次静力求解下它在跳量稳定化
> 后可解（1 阶最优），$P_1$ 迹载荷也能被精确强施加，载荷路径判据同样成立。投稿对比
> （`experiments/huzhang_topopt_paper/cases.toml` 的 `comparison_orders`）只取 2/3/4，
> 下限 $k=2$ 的理由是 $P_0$ 位移不完备包含刚体位移空间——这只在变密度演化中致命，与本
> 算例无关，见博士论文 §5.6.2 与
> [`docs/fem/huzhang-mixed-fem-implementation.md`](../../docs/fem/huzhang-mixed-fem-implementation.md)。
* `--nx` / `--ny`：棋盘格剖分数（默认 `80` / `10`，角点松弛要求均为正偶数）；
* `--relaxation` / `--no-relaxation`：角点松弛开关（默认开启）；
* `--solver`：`scipy`（默认）/ `mumps`。

> P1 投影在贴片外有几何衰减的振荡尾（每单元约 $0.27$），网格过粗时尾部会触及固支端，
> 那部分载荷被强加自由度真实吞掉。报告中的 `被吞载荷` 给出该量，量级约
> $P\cdot0.27^{n_x/2}$；默认 $n_x=80$ 下它远在容差之下，$n_x\le20$ 时会触发未通过。

优化侧在 $\rho=0.4$ + msimp 插值下的柔顺度对比留在
[`experiments/huzhang_topopt_paper/`](../../experiments/huzhang_topopt_paper/)，
本目录只做实体材料（$\rho=1$、无材料插值）的单次状态求解。

---

## 实验诊断与多 Case 分析

关于制造解收敛阶、traction 边界合力守恒判据、与拉格朗日位移元的对比实测数据，请参阅专属报告：
👉 [results_analysis.md](results_analysis.md)