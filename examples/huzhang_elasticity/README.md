# 胡张混合有限元 2D 求解算例 (Hu--Zhang Mixed FEM)

本目录提供基于 FEALPy 的二维线弹性 Hu--Zhang 混合有限元求解器：应力
$\boldsymbol{\sigma}\in\Sigma_h\subset H(\mathrm{div})$ 与位移
$\boldsymbol{u}\in V_h\subset[L_2]^d$ 联合求解，离散成应力--位移鞍点系统。

算例采用自包含的单文件代码设计，方便快速阅读与二次扩展。

---

## 算例核心文件结构

```text
soptx/examples/huzhang_elasticity/
├── minimal_demo.py              <-- [核心代码] 制造解收敛验证 (L2 观测阶 + 残差 + 对称性)
├── concentrated_load_demo.py    <-- [核心代码] traction 载荷路径验证 + VTU 导出
├── render_warped.py             <-- [可视化] VTK 离屏渲染 Warp 变形 PNG
├── math_spec.md                 <-- [数学规范] 符号—代码映射 (混合弱形式 + 低阶稳定化)
├── results_analysis.md          <-- [实验分析] 实测数据表与诊断报告
├── README.md                    <-- [使用说明] 本文件
└── outputs/                     <-- [测试成果与报告]
    ├── figures/                 <-- 离屏渲染截图 (.png)
    └── vtu/                     <-- 自动导出的 ParaView 场文件 (.vtu)
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

低阶稳定化（$p<3$ 时 $\Sigma_h\times V_h$ 不满足离散 inf-sup）的完整数学过程与
缩放律论证见知识库概念页 [胡张混合有限元](file:///C:/workspace/dut-postdoc/concepts/huzhang-mixed-fem.md)，
符号—代码映射与收敛性判据详见：
👉 [math_spec.md](math_spec.md)

---

## 快速开始 (CLI 使用指南)

### 1. 制造解收敛验证 (minimal_demo)

```bash
python examples/huzhang_elasticity/minimal_demo.py
python examples/huzhang_elasticity/minimal_demo.py --model mixed-exp-sine
python examples/huzhang_elasticity/minimal_demo.py --degree 2 --no-relaxation
```

### 2. traction 载荷路径验证 (concentrated_load_demo)

```bash
python examples/huzhang_elasticity/concentrated_load_demo.py
python examples/huzhang_elasticity/concentrated_load_demo.py --problem mixed-exp-sine
python examples/huzhang_elasticity/concentrated_load_demo.py --levels 5 --save-vtu
```

### 3. 可视化 (render_warped)

```bash
python examples/huzhang_elasticity/concentrated_load_demo.py --levels 5 --save-vtu
python examples/huzhang_elasticity/render_warped.py
```

### 可选参数选项

* `--degree`：应力空间次数（二维胡张元要求 $p\ge3$ 才无需低阶稳定化，默认 3）；
* `--relaxation` / `--no-relaxation`：角点松弛开关（默认开启）；
* `--solver`：`scipy`（默认）/ `mumps`（需 PyMUMPS 包）；
* `--save-vtu`：导出最密层位移场为 `.vtu` 至 `outputs/vtu/`。

---

## 实验诊断与多 Case 分析

关于制造解收敛阶、traction 边界合力守恒判据、与拉格朗日位移元的对比实测数据，请参阅专属报告：
👉 [results_analysis.md](results_analysis.md)
