# PINN 求解线弹性方程算法规范

本文档阐述 `soptx/examples/pinn_elasticity` 算例中 Physics-Informed Neural Network (PINN) 求解 2D/3D 各向同性线弹性方程的连续物理模型、自动微分残差算子与离散 Loss 契约。

> **通用范式说明**：关于 PINN 的通用 5 步求解范式、MLP 逐层数学表达与激活函数 $C^k$ 阶数约束等通用理论，请参阅知识库概念页 [PINN 通用概念与范式](file:///C:/workspace/dut-postdoc/concepts/pinn.md)。

## 1. 连续介质力学控制方程

在轴对齐有界区域 $\Omega \subset \mathbb{R}^d \ (d \in \{2, 3\})$ 上，考虑各向同性线弹性小变形静力平衡问题：

$$
-\nabla \cdot \boldsymbol{\sigma}(\boldsymbol{u}) = \boldsymbol{b} \quad \text{in } \Omega
$$

其中 $\boldsymbol{u}: \Omega \to \mathbb{R}^d$ 为位移场，$\boldsymbol{b}$ 为已知体力。

### 1.1 几何方程与本构关系
* **小变形 Cauchy 应变张量** $\boldsymbol{\varepsilon} \in \mathbb{R}^{d \times d}$：
  $$
  \boldsymbol{\varepsilon}(\boldsymbol{u}) = \frac{1}{2}\left(\nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^{\mathsf{T}}\right), \quad \varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
  $$
* **各向同性 Hooke 本构关系**（Lamé 参数 $\lambda, \mu$ 表示）：
  $$
  \boldsymbol{\sigma}(\boldsymbol{u}) = \lambda \operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{I} + 2\mu\boldsymbol{\varepsilon}, \quad \sigma_{ij} = \lambda \left(\sum_{k=1}^d \varepsilon_{kk}\right)\delta_{ij} + 2\mu \varepsilon_{ij}
  $$

### 1.2 边界条件
本算例采用全 Dirichlet 边界条件 $\partial\Omega = \Gamma_D$：
$$
\boldsymbol{u}(\boldsymbol{x}) = \bar{\boldsymbol{u}}(\boldsymbol{x}) \quad \text{on } \partial\Omega
$$

---

## 2. 神经网络参数化规范

* **模型选择**：多层感知机（MLP）逼近连续位移场 $\hat{\boldsymbol{u}}(\boldsymbol{x}; \boldsymbol{\theta})$。
* **网络隐层维度**：$d \to 32 \to 32 \to 16 \to d$。
  * 2D 算例：$2 \to 32 \to 32 \to 16 \to 2$
  * 3D 算例：$3 \to 32 \to 32 \to 16 \to 3$
* **激活函数**：隐藏层统一使用双曲正切函数 $\tanh(z)$（保障二阶偏导 $\nabla \cdot \boldsymbol{\sigma}$ 在自动微分中可导且平滑）。
* **张量 Shape 契约**：输入坐标为 $(N, d)$，输出位移预测为 $(N, d)$。

---

## 3. 基于 PyTorch Autograd 的物理残差求导链

利用自动微分（`torch.autograd.grad`）精准计算物理算子残差：

### 3.1 一阶求导与应力张量构造
由位移预测求一阶位移梯度矩阵 $\mathbf{J} \in \mathbb{R}^{d \times d}$：
$$
J_{ij} = \frac{\partial \hat{u}_i}{\partial x_j} = \text{autograd}\left(\hat{u}_i, x_j; \text{create\_graph=True}\right)
$$
对称化得到应变 $\hat{\boldsymbol{\varepsilon}}$，代入 Hooke 定律得到点值应力张量 $\hat{\boldsymbol{\sigma}}(\boldsymbol{x}; \boldsymbol{\theta})$。

### 3.2 二阶求导与平衡残差
对应力张量各分量求坐标散度：
$$
(\nabla \cdot \hat{\boldsymbol{\sigma}})_i = \sum_{j=1}^d \frac{\partial \hat{\sigma}_{ij}}{\partial x_j} = \sum_{j=1}^d \text{autograd}\left(\hat{\sigma}_{ij}, x_j; \text{create\_graph=True}\right)
$$
构造域内平衡残差 $\boldsymbol{R}_{\text{int}}$ 与 Dirichlet 边界残差 $\boldsymbol{R}_{\text{bnd}}$：
$$
\boldsymbol{R}_{\text{int}}(\boldsymbol{x}; \boldsymbol{\theta}) = -\nabla \cdot \hat{\boldsymbol{\sigma}}(\boldsymbol{x}; \boldsymbol{\theta}) - \boldsymbol{b}(\boldsymbol{x}), \quad \boldsymbol{R}_{\text{bnd}}(\boldsymbol{x}; \boldsymbol{\theta}) = \hat{\boldsymbol{u}}(\boldsymbol{x}; \boldsymbol{\theta}) - \bar{\boldsymbol{u}}(\boldsymbol{x})
$$

---

## 4. 损失函数与配点离散 (Loss Formulation)

损失函数为采样配点集上的 MSE 加权组合：
$$
\mathcal{L}(\boldsymbol{\theta}) = w_{\text{int}} \mathcal{L}_{\text{int}}(\boldsymbol{\theta}) + w_{\text{bnd}} \mathcal{L}_{\text{bnd}}(\boldsymbol{\theta})
$$

$$
\mathcal{L}_{\text{int}}(\boldsymbol{\theta}) = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left\| \boldsymbol{R}_{\text{int}}(\boldsymbol{x}_i^{(int)}; \boldsymbol{\theta}) \right\|_2^2, \quad \mathcal{L}_{\text{bnd}}(\boldsymbol{\theta}) = \frac{1}{N_{\text{bnd}}} \sum_{j=1}^{N_{\text{bnd}}} \left\| \boldsymbol{R}_{\text{bnd}}(\boldsymbol{x}_j^{(bnd)}; \boldsymbol{\theta}) \right\|_2^2
$$

* **默认权重配置**：$(w_{\text{int}}, w_{\text{bnd}}) = (1, 30)$。

---

## 5. 优化与精度评价

* **优化器**：Adam（初始学习率 $\eta=10^{-3}$），通过 `zero_grad()` $\to$ `loss.backward()` $\to$ `optimizer.step()` 更新权重。
* **评估指标**：固定诊断网格（Diagnostic Mesh）上的相对位移 $L_2$ 误差 $E_{L_2}$。
