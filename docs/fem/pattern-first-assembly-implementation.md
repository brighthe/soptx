# 模式先行 (Pattern-First) 稀疏矩阵装配架构设计与实现

> **模块定位**：SOPTX 自主可控的有限元稀疏刚度矩阵高性能装配引擎与双线性型数学门面。  
> **核心组件**：
> - 底层执行内核：[`soptx.fem.matrix.CSRPattern`](../../src/soptx/fem/matrix/csr_pattern.py)、[`build_csr_pattern`](../../src/soptx/fem/matrix/csr_pattern.py)、[`assemble_csr`](../../src/soptx/fem/matrix/csr_pattern.py)
> - 高层数学门面：[`soptx.fem.BilinearForm`](../../src/soptx/fem/bilinear_form.py)  
> **实测证据**：[`experiments/fa_assembly_capability/results_analysis.md`](../../experiments/fa_assembly_capability/results_analysis.md)

---

## 一、背景与痛点：传统装配的“内存墙”

在有限元显式装配（Full Assembly, FA）中，全局刚度矩阵 $\mathbf K \in \mathbb{R}^{N_{\text{dof}} \times N_{\text{dof}}}$ 是由所有单元的局部刚度矩阵 $\mathbf K_e \in \mathbb{R}^{N_{\text{ldof}} \times N_{\text{ldof}}}$ 散加合并而成：

$$
\mathbf K = \sum_{e=1}^{N_C} \mathbf L_e^T \mathbf K_e \mathbf L_e
$$

### 1. 传统 FEALPy 装配的瓶颈机理
传统有限元框架（如 FEALPy `BilinearForm.assembly()`）在执行上述合并时，采用**全长坐标格式（COO）三元组后处理**机制：
1. **全长三元组物化**：在内存中先分配并物化所有单元的三元组数组 $(I, J, V)$。在三维一阶四面体网格（3D tet4, $p=1$）下，每个自由度平均产生 288 个三元组贡献，仅基础索引与浮点开销即达 $24\text{ B/triplet}$；
2. **全局排序与归并去重（`coalesce`）**：调用 PyTorch/CUB 或 NumPy 进行全局大张量基数排序与分桶去重，产生庞大的排序临时缓冲；
3. **最终转为 CSR**：去重完成后再构造 CSR 矩阵。

```
【传统 COO 装配路径（慢速、耗内存）】：
单刚 K_e ──► 物化全长 (I, J, V) ──► 全局排序去重 (coalesce) ──► FEALPy CSRTensor
           (占用 52~77 B/triplet)    (排序缓冲膨胀 15~20 倍)       (仅需 2.6 B/triplet)
```

### 2. 物理极限与 OOM 归因
在 47 GiB 物理内存单台工作站上：
* 传统 `coalesce` 路线的装配过渡态峰值内存单价高达 **$14.9\sim 22.2\text{ KB/dof}$**；
* 当网格规模达到 $n = 96$（274 万自由度）时，瞬态内存突破 $55\text{ GiB}$，直接被 Linux OOM Killer 强杀；
* 传统装配的单机物理极限被**刚性锁死在约 230 万自由度**。

---

## 二、核心架构思想：符号与数值两阶段解耦

模式先行（Pattern-First）装配的核心哲学是：**在拓扑优化与线性力学分析中，背景网格与自由度拓扑结构在整个迭代过程中是静态不变的，变化的仅仅是单元材料刚度数值。**

因此，SOPTX 将装配过程严格解耦为**符号阶段（一次性静态编译）**与**数值阶段（每轮极速原子累加）**：

```
┌────────────────────────────────────────────────────────────────────────┐
│               SOPTX 模式先行 (Pattern-First) 装配架构                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
【阶段 0：符号阶段 (Symbolic Phase)】                  【阶段 1~N：数值阶段 (Numeric Phase)】
build_csr_pattern(space)                               assemble_csr(K_e, pattern)
 • 节点邻接图分析 (轻量 COO->CSR)                        • 零三元组物化 (Zero Intermediate Triplets)
 • 展开为 DOF 级 CSR 骨架 (crow, col)                    • 槽位预映射原地原子累加:
 • 矢量化提取槽位映射 slot_map (NC * ldof^2,)              - NumPy: np.add.at / bincount
 • 一次性搬运并常驻计算设备 (CPU / CUDA GPU)                - PyTorch: scatter_add_ (CUDA atomicAdd)
                                                        • O(1) 零拷贝直接产出标准 FEALPy CSRTensor
```

### 1. 阶段 0：符号阶段 (`build_csr_pattern`)
在初始化或空间变更时执行一次（耗时几十毫秒），完成以下拓扑编译：
1. **提取全局自由度映射**：获取空间的 `cell_to_dof()` 映射；
2. **构建 CSR 稀疏骨架**：利用轻量笛卡尔积去重提取全局唯一且排序的行指针 `crow`（`indptr`）与列索引 `col`（`indices`）；
3. **计算单刚槽位映射表 (`slot_map`)**：
   * 构建每行非零元的全局有序唯一键 $\text{Key} = \text{row} \times N_{\text{dof}} + \text{col}$；
   * 通过 `searchsorted` 矢量化查找出每个单元单刚元素 $(e, i, j)$ 在全局 CSR `values` 数组中的绝对槽位索引 `slot_map`；
4. **常驻目标设备**：若为 GPU 环境，一次性将 `crow`、`col`、`slot_map` 和预分配的数值缓冲 `buffer` 转移至 GPU 显存常驻。

### 2. 阶段 1~N：数值阶段 (`assemble_csr`)
在拓扑优化的每一步正向求解循环中：
1. 单刚积分器产出当前轮次的单元刚度张量 $\mathbf K_e(\boldsymbol{\rho})$；
2. **原地置零与原子累加**：
   * **NumPy (CPU)**：`buffer.fill(0.0); np.add.at(buffer, pattern.slot_map, K_e.ravel())`；
   * **PyTorch (CUDA GPU)**：`buffer.zero_(); buffer.scatter_add_(0, pattern.slot_map, K_e.view(-1))`；
3. **$O(1)$ 零拷贝包装**：直接构造并返回 `fealpy.sparse.CSRTensor(crow, col, buffer, shape)`，整个过程耗时仅数毫秒，零动态内存申请。

---

## 三、高层双线性型门面：`soptx.fem.BilinearForm`

为了保持有限元变分形式的经典数学美感与 API 友好性，SOPTX 在底层执行内核之上提供了统一的高层门面类 [`soptx.fem.BilinearForm`](../../src/soptx/fem/bilinear_form.py)。

### 1. 继承与重写的面向对象设计
`soptx.fem.BilinearForm` 继承自 `fealpy.fem.BilinearForm`，100% 兼容其数学定义与积分子容器接口：
* **重写 `assembly()` 引擎**：
  ```python
  bform = BilinearForm(space)
  bform.add_integrator(integrator)
  K = bform.assembly(format='csr')  # 默认自动调用 CSRPattern 高性能路径
  ```
* **支持多积分子（Multi-Integrator）矢量化叠加**：
  若物理问题同时包含弹性积分子与附加刚度积分子，`BilinearForm` 在单刚阶段将所有积分子求和为统一的 $\mathbf K_e$，仅执行一次 `assemble_csr`，避免多次稀疏矩阵加法开销。
* **无缝支持 EA 算子（Matrix-Free）**：
  在 `operator_level='ea'` 下，`BilinearForm` 保持轻量算子属性，提供 `@` 矩阵向量乘（Cell-wise Matvec），与 FA 装配分支清晰解耦。

---

## 四、多后端与 GPU 显存内装配机制

模式先行架构天然支持多计算后端，并完美适配现代 GPU 的硬件并行原语：

| 计算后端 / 硬件平台 | 符号阶段数据载体 | 数值阶段原子累加原语 | 显存与数据流特性 |
| :--- | :--- | :--- | :--- |
| **NumPy (CPU)** | `np.ndarray` (int64) | `np.add.at` / `bincount` | 零全长 COO 数组，消除 GC 压力 |
| **PyTorch (CPU)** | `torch.Tensor` (cpu) | `scatter_add_` | 支持 PyTorch CPU 稠密/稀疏张量混算 |
| **PyTorch (CUDA GPU)** | `torch.Tensor` (cuda:0) | **CUDA 硬件级 `atomicAdd`** | **显存全流程驻留（Zero H2D/D2H Copy）**、零 GPU 排序、零动态显存申请 |

---

## 五、实测性能与容量跃升

在受控单机环境（47.04 GiB 内存，3D tet4 线弹性基准模型）下的实测与理论对照：

| 装配方案与组合 | 每自由度峰值内存 $b$ | 47G 内存承载天花板 $N_{\max}$ | 极限网格规模 $(n \times n \times n)$ | 工程物理刻画能力 |
| :--- | :---: | :---: | :---: | :--- |
| **传统生产组合**<br>(`fast + coalesce`) | $15.1\sim 22.2\text{ KB/dof}$ | **约 230 万** | $n \le 90$<br>(约 437 万四面体) | **粗/中等网格**：$n=96$ 即耗尽 47G 内存触发 Linux OOM 强杀。 |
| **中间过渡组合**<br>(`fast + scipy`) | $13.3\text{ KB/dof}$ | **约 382 万** | $n \le 105$<br>(约 695 万四面体) | **提升有限**：依然受阻于全长三元组物化所导致的刚性内存墙。 |
| **SOPTX 模式先行**<br>(`fast + pattern`) | **$2.0\text{ KB/dof}$** | **约 3580 万**<br>(**提升 $15.6\times$**) | **$n \approx 225$**<br>(**超 6800 万四面体**) | **工业级超精细网格**：单机直接支撑数千万自由度，高保真刻画微细点阵与高阶拓扑演化。 |

---

## 六、快速上手与 API 示例

### 1. 标准有限元分析与拓扑优化中的典型用法
```python
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.fem import BilinearForm
from soptx.fem.integrators import LinearElasticIntegrator
from soptx.materials.linear_elasticity import IsotropicLinearElasticMaterial

# 1. 建立网格与向量有限元空间
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 32, 32, 32)
space = LagrangeFESpace(mesh, p=1)
tspace = TensorFunctionSpace(scalar_space=space, shape=(-1, 3))

# 2. 建立材料与积分器
material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3, hypothesis="3D")
integrator = LinearElasticIntegrator(material=material, method="fast")

# 3. 构造 SOPTX 双线性型并装配
bform = BilinearForm(tspace)
bform.add_integrator(integrator)

# 高性能模式先行装配 (产出标准 FEALPy CSRTensor)
K = bform.assembly(format="csr")
print(f"装配完成: 形状 {K.shape}, 非零元总数 {K.nnz:,}")
```

### 2. 底层 CSRPattern 显式控制（进阶用法）
```python
from soptx.fem.matrix import build_csr_pattern, assemble_csr

# 显式构建并常驻 GPU
pattern = build_csr_pattern(tspace, device="cuda:0")

# 每轮迭代直接原子累加
K_e_gpu = integrator.assembly(tspace).to("cuda:0")
K_global_gpu = assemble_csr(K_e_gpu, pattern)
```

---

## 七、验证与质量门禁

SOPTX 模式先行装配体系已通过完备的自动化回归测试门禁：
1. **数学代数严格等价性**：[`tests/unit/test_csr_pattern.py`](../../tests/unit/test_csr_pattern.py) 验证在 2D/3D 网格下与 FEALPy 传统基准的稠密残差严格为 **$0.0$**；
2. **多后端与 GPU 显存驻留**：覆盖 NumPy、PyTorch CPU 与 PyTorch CUDA GPU 显存内装配；
3. **全流程物理收敛性**：制造解线弹性测试（`test_lagrange_fem_analyzer_standard.py`）严格验证位移场二阶收敛（$L_2$ 收敛阶 $1.982$），全套 344 项测试 100% 绿灯通过。