# Exact 子结构与 PIML 子结构拓扑优化结果对照

硬件：Intel Core i9-14900KF（24 核 32 线程，8P+16E），运行内存 45 GiB；NVIDIA GeForce RTX 5080，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（内置 scipy-openblas 0.3.34），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.13.0（CUDA 13.0）；PyMUMPS 0.4.0，链接系统 MUMPS 5.6.2（OpenBLAS 0.3.26、Open MPI 4.1.6、ScaLAPACK 2.2.1，单进程运行）；pyevtk 1.7.0。

## 1. 研究对象

对照精确子结构的  `full_trace`、`linear_corner` 与 PIML 子结构的  `full_trace`、`linear_corner` 在 CPU 上对 Huang2023 三维 MBB 梁的完整优化结果。

| 项目 | Exact `full_trace` | PIML `full_trace` | Exact `linear_corner` | PIML `linear_corner` |
|---|---|---|---|---|
| 接口空间 | 完整边界迹，$\boldsymbol\Psi=\boldsymbol I$ | 完整边界迹，$\boldsymbol\Psi=\boldsymbol I$ | 角点线性迹，$\boldsymbol\Psi=\boldsymbol L$ | 角点线性迹，$\boldsymbol\Psi=\boldsymbol L$ |
| 局部计算 | 精确内部消元，构造局部 Schur 补 | `shape` 路线：预测内部位移映射，再通过变分能量构造局部接口刚度 | 在角点线性迹下精确内部消元，构造局部角点刚度 | `shape` 路线：预测角点线性迹下的内部位移映射，再通过变分能量构造局部角点刚度 |
| 全局刚度算子 | 接口算子 $\boldsymbol K_\Gamma$ | 接口算子 $\widehat{\boldsymbol K}_\Gamma$ | 角点算子 $\boldsymbol K_C$ | 角点算子 $\widehat{\boldsymbol K}_C$ |
| 全局未知量 | 全部接口自由度 $\boldsymbol U_\Gamma$ | 全部接口自由度 $\boldsymbol U_\Gamma$ | 角点自由度 $\boldsymbol U_C$ | 角点自由度 $\boldsymbol U_C$ |
| 求解器 | 待确定 | 与 Exact `full_trace` 相同 | 待确定 | 与 Exact `linear_corner` 相同 |

材料、载荷与支承（`end_corners`，即 `FullMBBBeamPoint3d`）四者相同，Dirichlet 条件均为齐次，在各自接口未知量上施加与同一物理支承一致的约束。同一接口空间内的 Exact 与 PIML 使用相同的全局装配方式、求解器及求解容差；四者共用细网格、子结构划分及拓扑优化参数。

## 2. full_trace 下精确子结构与 PIML 子结构拓扑优化结果对照

两者拟采用同一细网格：$210\times35\times35$ 个 Q1 六面体单元，共 $257\,250$ 个单元、$273\,456$ 个节点、$820\,368$ 个自由度。

子结构划分：按 $m=5$ 划成 $42\times7\times7=2\,058$ 个子结构，每个含 $5^3=125$ 个细单元、$6^3=216$ 个节点，其中内部节点 $4^3=64$ 个、边界节点 152 个，对应 192 个内部自由度与 456 个边界自由度。两者均采用 `full_trace`，全局接口未知量为全部接口节点，共 $141\,744$ 个节点、$425\,232$ 个自由度，均为施加支承约束前的数量。

### 2.1 结果对照

### 2.2 计算成本

#### 2.2.1 耗时（s）

#### 2.2.2 内存

## 3. linear_corner 下精确子结构与 PIML 子结构拓扑优化结果对照

两者拟采用与第 2 章相同的细网格及子结构划分：$210\times35\times35$ 个 Q1 六面体单元，按 $m=5$ 划成 $42\times7\times7=2\,058$ 个子结构。

两者均采用 `linear_corner`，每个子结构的边界位移由 8 个角点的 24 个位移自由度线性插值得到，内部位移仍在 192 个内部自由度上恢复。全局接口未知量为子结构角点，共 $43\times8\times8=2\,752$ 个节点、$8\,256$ 个自由度，均为施加支承约束前的数量。

### 3.1 结果对照

### 3.2 计算成本

#### 3.2.1 耗时（s）

#### 3.2.2 内存
