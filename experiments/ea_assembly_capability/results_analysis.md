# EA (Element Assembly) 单元装配无矩阵算子内存机制与容量极限分析报告

本报告系统分析与评测 **EA (Element Assembly / Element-by-Element，单元装配无矩阵算子)** 在三维线弹性分析中的内存组成机理、单价模型 $b_{\text{EA}}$、单机与单卡容量极限 $N_{\max}$，并与 FA（全组装）的各历史及生产路线展开严密横向对比。

---

## 1. EA 算子机理与内存模型

在三维四面体 P1 有限元离散中（$N_C \approx 6 N_{\text{node}}$，$N_{\text{dof}} = 3 N_{\text{node}}$）：

### 1.1 静态常驻存储（Static Cache Footprint）

EA 算子预先计算并持久化保存每个单元的局部刚度矩阵 $\mathbf{K}_e$（$12 \times 12$ 浮点数），但不进行全局总刚稀疏矩阵的组装。其静态常驻内存由两部分构成：

1. **单元刚度张量** $\{\mathbf{K}_e\}_{e=1}^{N_C}$：$N_C \times 144 \times 8\text{ 字节} = 6 N_{\text{node}} \times 1152\text{ B} = \mathbf{2.304\text{ KB/dof}}$；
2. **单元自由度映射** `cell_to_dof`：$N_C \times 12 \times 8\text{ 字节} = 6 N_{\text{node}} \times 96\text{ B} = \mathbf{0.192\text{ KB/dof}}$；

$$\text{理论静态单价 } b_{\text{static}} = 2.304 + 0.192 = \mathbf{2.496\text{ KB/dof}} \approx \mathbf{2.50\text{ KB/dof}}.$$

### 1.2 动态算子乘积（MatVec Transient Footprint）

在执行矩阵向量乘 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 时，数据流沿因子链往返：
1. **Gather**：提取单元位移 $\mathbf{x}_e = \mathbf{x}[\text{cell\_to\_dof}]$（开销 $N_C \times 12 \times 8\text{ B} = 0.192\text{ KB/dof}$）；
2. **Local MatVec**：批量张量收缩 $\mathbf{y}_e = \mathbf{K}_e \mathbf{x}_e$（开销 $0.192\text{ KB/dof}$）；
3. **Scatter-Add**：原子累加回填至全局向量 $\mathbf{y}[\text{cell\_to\_dof}] += \mathbf{y}_e$；

算子作用期间无任何全局稀疏矩阵遍历，瞬态临时缓冲极低（$< 0.4\text{ KB/dof}$）。

### 1.3 求解器常驻与容量天花板

搭载无预条件共轭梯度法（CG）时，常驻存储为：
$$\text{总内存 } M_{\text{total}} = M_{\text{cache}} + 5 \times 8\text{ B} \times N_{\text{dof}} \approx 2.536\text{ KB/dof} \approx \mathbf{2.54\text{ KB/dof}}.$$

* **单机 47.04 GiB 内存容量天花板**：
  $$N_{\max, \text{47G}} = \frac{47.04 \times 1024^3\text{ 字节}}{2.54 \times 1000\text{ 字节/dof}} \approx \mathbf{1988\text{ 万自由度}}.$$
* **单卡 16.0 GiB GPU 显存容量天花板**：
  $$N_{\max, \text{GPU}} = \frac{16.0 \times 1024^3\text{ 字节}}{2.54 \times 1000\text{ 字节/dof}} \approx \mathbf{677\text{ 万自由度}}.$$

---

## 2. 实测数据与横向对比表

基于三维线弹性制造解算例（DivergenceFreePolynomialElasticity3D），实测数据如下：

### 2.1 阶段 1: 单元刚度张量缓存 (n = 32, 10.8 万 DOFs)

| 单刚算法 | 净峰值内存 | 缓存张量大小 | 单刚单价 $b_1$ | 计算耗时 | 归因说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fast`** | **$247.3\text{ MiB}$** | **$216.0\text{ MiB}$** | **$2.3\text{ KB/dof}$** | **$12.3\text{ ms}$** | 不变张量预计算, 消除高维缓冲 (生产推荐) |
| **`standard`** | $4.77\text{ GiB}$ | $216.0\text{ MiB}$ | $47.5\text{ KB/dof}$ | $5.68\text{ s}$ | 9 个分块梯度张量分别累加 |
| **`voigt`** | $9.00\text{ GiB}$ | $216.0\text{ MiB}$ | $89.7\text{ KB/dof}$ | $412.7\text{ ms}$ | `einsum` 隐式物化 6 维应变张量 |

### 2.2 FA vs EA 跨层级全景综合对比

| 装配层级与路线 | 存储范式 | 全长三元组 | 内存单价 $b$ | 47G 自由度天花板 | GPU 吞吐与特性 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FA (FEALPy coalesce)** | 历史 COO 排序 | 是 | $22.2\text{ KB/dof}$ | 约 230 万 | 受制于 COO 排序墙 |
| **FA (SciPy csr)** | 过渡 COO 压缩 | 是 | $4.0\text{ KB/dof}$ | 约 382 万 | 仍需物化三元组过渡态 |
| **FA (模式先行 CSRPattern)** | 原地原子累加 | **否** | **$2.0\text{ KB/dof}$** | **约 3580 万** | **单机容量最大**, 需符号骨架 |
| **EA (单元装配 Matrix-Free)** | 缓存单元矩阵 $\{K_e\}$ | **否** | **$2.54\text{ KB/dof}$** | **约 1980 万** | **零稀疏图构建**, 极高 GPU MatVec 并行吞吐 |

---

## 3. 核心学术结论与物理启示

1. **为什么低阶 Tet4 下 EA 静态单价略高于 FA CSR？**
   在低阶四面体网格中，单元节点共享度高（$N_C / N_N \approx 6$），EA 独立存储每个单元的 144 个浮点数带来了局部冗余（$2.30\text{ KB/dof}$），而 FA 的 CSR 压缩存储每行仅存约 43 个非零元（$1.14\text{ KB/dof}$）。
2. **EA 的核心战略价值**：
   * **彻底免除稀疏矩阵符号装配**：在大规模动态演化网格中，EA 无需耗时构建与遍历 CSR 拓扑图；
   * **GPU 极致并行化**：局部单元张量乘天然规避了稀疏矩阵 SpMV 的非规则内存访问，可直接映射至 GPU Tensor Core 或高效原子指令，在大规模 Krylov 迭代求解中具备极高的 FLOPs 吞吐。
