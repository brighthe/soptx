# 线弹性 PINN 2D/3D 求解算例 (Linear Elasticity PINN)

本目录提供基于 PyTorch 与 FEALPy 的二维平面应变与三维各向同性线弹性强形式 Physics-Informed Neural Networks (PINN) 求解器。

算例采用自包含的单文件代码设计，方便快速阅读与二次扩展。

---

## 算例核心文件结构

```text
soptx/examples/pinn_elasticity/
├── minimal_demo.py         <-- [核心代码] 2D/3D PINN 线弹性单文件自包含求解器
├── math_spec.md            <-- [数学规范] 控制方程、Autograd 链与 Shape 契约
├── README.md               <-- [使用说明] 算法概述与运行 CLI 指南
└── outputs/                <-- [测试成果与报告]
    ├── results_analysis.md <-- [实验分析] 2D/3D 消融实验诊断与实测表格报告
    ├── figures/            <-- 自动生成的 2D 三场对比云图 (.png)
    └── vtu/                <-- 自动导出的 2D/3D ParaView 场文件 (.vtu)
```

---

## 数学模型与物理控制方程

强形式控制方程与 Hooke 本构公式如下：

$$
-\nabla\cdot\boldsymbol\sigma(\boldsymbol u)=\boldsymbol b,\qquad
\boldsymbol\varepsilon(\boldsymbol u)
=\frac12(\nabla\boldsymbol u+\nabla\boldsymbol u^{\mathsf T}),
$$

$$
\boldsymbol\sigma
=\lambda\operatorname{tr}(\boldsymbol\varepsilon)\mathbf I
+2\mu\boldsymbol\varepsilon,\qquad
\boldsymbol u=\bar{\boldsymbol u}\quad\text{on }\partial\Omega.
$$

* **2D 平面应变制造解 (`ExponentialSineManufacturedElasticity2D`)**：$\lambda=1.0, \mu=0.5$
* **3D 各向同性制造解 (`DivergenceFreePolynomialElasticity3D`)**：$\lambda=1.0, \mu=1.0$

各制造解的完整数学定义、体力表达式与边界条件详见：
👉 [制造解文档](../../docs/problems/manufactured-elasticity.md)

完整物理算子求导链与 Shape 契约规范请参阅：👉 [math_spec.md](math_spec.md)

---

## 快速开始 (CLI 使用指南)

通过运行单文件脚本 [`minimal_demo.py`](minimal_demo.py) 即可完成训练与误差评估：

### 1. 运行 2D 基线算例 (平面应变)
```bash
python examples/pinn_elasticity/minimal_demo.py --dim 2 --epochs 2000
```

### 2. 运行 3D 基线算例 (各向同性)
```bash
python examples/pinn_elasticity/minimal_demo.py --dim 3 --epochs 2000
```

### 3. 可选参数选项
* `--plot`：训练结束后自动弹出 2D 三场对比云图；
* `--save-vtu`：自动导出 ParaView `.vtu` 体场文件至 `outputs/vtu/` 目录；
* `--save-model`：自动保存神经网络权重至 `outputs/checkpoints/`。

---

## 实验诊断与多 Case 分析

关于 2D 与 3D 算例的收敛性物理机理诊断、配点加密与网络容量消融实验数据，请参阅专属报告：
👉 [outputs/results_analysis.md](outputs/results_analysis.md)
