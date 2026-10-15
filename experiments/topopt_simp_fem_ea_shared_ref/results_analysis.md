# 传统有限元标准 EA 与共享参考单元 EA 结果对照

硬件：AMD Ryzen 9 9950X（16 核 32 线程），运行内存 45 GiB（WSL2 上限 48 GB）；NVIDIA GeForce RTX 5070 Ti，显存 16 GB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（conda-forge，链接 BLIS 2.0 与 netlib LAPACK），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.11.0（CUDA 12.8）；PyMUMPS 0.4.0，链接 MUMPS 5.8.2（BLIS 2.0、netlib LAPACK、Open MPI 5.0.11、ScaLAPACK 2.2.0）；pyevtk 1.7.0。

## 1. 研究对象

对照传统有限元下两种 `EA` 刚度数据组织方式在 CPU 上对 Huang2023 三维 MBB 梁的完整优化结果。

|      | 标准 EA | 共享参考单元 EA |
| ---- | ------- | --------------- |
| 常驻刚度数据 | 每个单元完整的 $K_e$ | 一份 $K^0$ 与逐单元 $s_e$ |
| 单元作用 | $K_ex_e$ | $s_e(K^0x_e)$ |

标准 EA 走 `ElementAssembly`，保存已经乘入材料系数的 $K_e$。仅去掉 `reference_classes=1` 不构成标准 EA：当前分析器在单元密度模式下仍使用 `SharedReferenceElementAssembly`，只是改为保存逐单元 $K_e^0$ 与 $s_e$。

材料、载荷、支承（`end_corners`，即 `FullMBBBeamPoint3d`）、边界条件消元、MGCG 求解器与容差、过滤、OC 与停止条件均沿用 `../topopt_simp_fem/`，两侧保持一致。两侧采用相同的多重网格设置；共享参考是否只用于最细层待定，实际实现在运行后注明。

## 2. 固定密度算子对照

### 2.1 正确性

两侧在同一密度下的 `matvec` 一致性待验证。

### 2.2 耗时（s）

初始化、系数更新与 `matvec` 的耗时待填。

### 2.3 内存

待填。

## 3. 完整拓扑优化对照

### 3.1 结果一致性

两侧优化轨迹、停止轮数与最终设计的一致性待验证。

### 3.2 计算成本

#### 3.2.1 耗时（s）

待填。

#### 3.2.2 内存

待填。
