# PIML 二维线弹性静力缩聚范式与代数规范 (Math Spec)

本文档规范 `soptx/examples/piml_elasticity` 中基于 **Problem-Independent Machine Learning (PIML)** 的二维线弹性子结构静力缩聚（Schur Complement）数学原理、降维表达与代码实现约定。

---

## 1. 静力缩聚 (Substructure Static Condensation) 数学推导

设宏观域 $\Omega$ 被剖分为 $M$ 个无重叠的局部子结构（粗单元）$\Omega^j$ ($j = 1, \dots, M$)。在子结构 $\Omega^j$ 内部，有限元离散自由度被划分为两类：
- **内部自由度 (Internal DOFs)**：完全位于子结构内部的节点自由度，记为下标 $i$；
- **接口自由度 (Interface/Boundary DOFs)**：位于子结构边界/接口上的节点自由度，记为下标 $b$。

子结构 $\Omega^j$ 的局部有限元平衡方程写为分块矩阵形式：

$$
\begin{bmatrix}
\mathbf{K}_{ii}^j & \mathbf{K}_{ib}^j \\
\mathbf{K}_{bi}^j & \mathbf{K}_{bb}^j
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{u}_i^j \\
\boldsymbol{u}_b^j
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{f}_i^j \\
\mathbf{f}_b^j
\end{bmatrix}
$$

假设内部自由度上无外载荷（即 $\mathbf{f}_i^j = \boldsymbol{0}$），由第一行方程求得内部位移与接口位移之间的线性映射关系：

$$
\boldsymbol{u}_i^j = - (\mathbf{K}_{ii}^j)^{-1} \mathbf{K}_{ib}^j \boldsymbol{u}_b^j = \mathbf{N}^j \boldsymbol{u}_b^j
$$

其中 $\mathbf{N}^j = - (\mathbf{K}_{ii}^j)^{-1} \mathbf{K}_{ib}^j \in \mathbb{R}^{n_i \times n_b}$ 即为**子结构多尺度形函数矩阵 (Substructure Shape Function Matrix)**。

将 $\boldsymbol{u}_i^j$ 代入第二行方程，得仅作用于接口自由度的缩聚平衡方程：

$$
\mathbf{K}_s^j \boldsymbol{u}_b^j = \mathbf{f}_b^j
$$

其中 $\mathbf{K}_s^j \in \mathbb{R}^{n_b \times n_b}$ 即为 **Schur 补缩聚刚度矩阵 (Schur Complement Condensed Stiffness Matrix)**：

$$
\mathbf{K}_s^j = \mathbf{K}_{bb}^j - \mathbf{K}_{bi}^j (\mathbf{K}_{ii}^j)^{-1} \mathbf{K}_{ib}^j = (\mathbf{N}^j)^{\mathsf T} \mathbf{K}^j \mathbf{N}^j
$$

---

## 2. 全局接口方程组装与细尺度恢复

### 2.1 全局接口方程
将各子结构的缩聚刚度矩阵 $\mathbf{K}_s^j$ 按照全局接口自由度编号进行 **Scatter-Add 装配**，得到全局粗/接口系统：

$$
\mathbf{K}_{\text{global}} \boldsymbol{U}_b = \mathbf{F}_b
$$

施加宏观外载荷与固定位移边界条件后，求解出全局接口位移向量 $\boldsymbol{U}_b$。

### 2.2 细尺度位移与应力恢复
由求解得到的接口位移 $\boldsymbol{U}_b$ 截取各子结构接口位移 $\boldsymbol{u}_b^j$，利用多尺度形函数矩阵前向乘法，**秒级恢复**子结构内部细尺度位移：

$$
\boldsymbol{u}_i^j = \mathbf{N}^j \boldsymbol{u}_b^j
$$

全尺度细观位移场 $\boldsymbol{U}_{\text{full}} = [\boldsymbol{u}_i^1, \dots, \boldsymbol{u}_i^M, \boldsymbol{U}_b]$ 进而用于计算单元柯西应力与全局结构柔顺度 $C = \mathbf{F}_b^{\mathsf T} \boldsymbol{U}_b$。

---

## 3. PIML 代理神经网络与代数保持

PIML 的目标是用深度神经网络替代高开销的子结构求逆消元 $(\mathbf{K}_{ii}^j)^{-1}$：

$$
\boldsymbol{\rho}^j \in [0, 1]^m \xrightarrow[\quad \text{代理网络} \quad]{\text{PIML}} \widehat{\mathbf{K}}_s^j \text{ 或 } \widehat{\mathbf{N}}^j
$$

- **路线 A (预测形函数 $\widehat{\mathbf{N}}^j$)**：
  预测 $\widehat{\mathbf{N}}^j$，再通过 $\widehat{\mathbf{K}}_s^j = (\widehat{\mathbf{N}}^j)^{\mathsf T} \mathbf{K}^j \widehat{\mathbf{N}}^j$ 构造刚度。**物理硬保持**对称正定性与变分能量关系。
- **路线 B (直接预测刚度 $\widehat{\mathbf{K}}_s^j$)**：
  预测 Cholesky 上三角因子 $\mathbf{L}^j$，令 $\widehat{\mathbf{K}}_s^j = (\mathbf{L}^j)^{\mathsf T} \mathbf{L}^j$，**代数硬保持**对称半正定性。

---

## 4. 验证与验收标准

1. **V1 缩聚精确性 (Exact Baseline Verification)**：
   在 `ExactPredictor` 模式下，$\mathbf{K}_s^j$ 与细网格全局 Schur 补的相对 Frobenius 范数误差必须满足：
   $$
   \frac{\|\mathbf{K}_{\text{global}} - \mathbf{S}_{\text{full}}\|_F}{\|\mathbf{S}_{\text{full}}\|_F} < 10^{-12} \quad (\text{机器精度})
   $$
2. **细尺度位移恢复精确性**：
   恢复出的内部位移 $\boldsymbol{u}_i^j$ 与全尺度细网格有限元直解 $\boldsymbol{U}_{\text{FEA}}$ 的相对 $L_2$ 误差必须满足 $< 10^{-12}$。
