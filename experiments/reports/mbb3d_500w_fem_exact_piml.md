# 三维 MBB 梁 500 万自由度：全尺度 FEM、精确子结构与 PIML 子结构的对照

硬件：CPU：16 核 32 线程，可用内存 45 GiB；NVIDIA GeForce RTX 5070 Ti，可用显存 16 GB。

软件：Ubuntu 24.04，Python 3.12.13；NumPy 2.5.3，SciPy 1.18.1，PyTorch 2.11.0（CUDA 12.8）。

## 1. 研究对象

三种方法求解同一个拓扑优化问题，差别只在每轮状态分析所用的模型：

| 方法 | 分析模型 | 全局未知量 |
|---|---|---|
| 全尺度 FEM | 在细网格上直接求解全部自由度 | $5\,109\,588$ |
| 精确子结构 | 子结构边界位移由角点三线性插值，内部精确静力缩聚（Schur 补） | $46\,452$ |
| PIML 子结构 | 子结构边界位移由角点三线性插值，内部形函数由网络预测，按变分式缩聚 | $46\,452$ |

## 2. 实验设置

| 参数 | 取值 |
|---|---|
| 问题 | `FullMBBBeamPoint3d`，顶面中心集中力 $-1$，左端两底角点铰支，右端两底角点 $u_y=u_z=0$ |
| 网格 | 结构化六面体 $390\times65\times65$，$h=1$，$1\,647\,750$ 个单元 |
| 材料 | $E_0=1$，$E_{\min}=10^{-7}$，$\nu=0.3$，修正 SIMP，$q=3$ |
| 离散 | 线性元，二阶高斯积分，`fast` 单元刚度 |
| 优化 | 体积分数 $0.12$，均匀初值；密度过滤，半径 $3h$；OC，移动限 $0.2$，阻尼 $0.5$ |
| 收敛 | 柔顺度相对变化连续 5 轮小于 $2\times10^{-4}$，最多 300 轮 |
| 线性求解 | CG + Jacobi，相对右端项容差 $10^{-6}$，最多 20000 步，第 2 轮起以上一轮的解为初值 |
| 子结构划分 | $78\times13\times13$ 个子结构，每个 $5\times5\times5$ 个细单元 |

PIML 子结构的网络在离线阶段训练，在线优化只做推理：

| 离线阶段 | 取值 |
|---|---|
| 样本 | 单个子结构（$m=5$，$h=1$，$\nu=0.3$）上逐单元独立均匀采样归一化模量 $\widehat E_e\in[10^{-7},1]$；训练 $400\,000$、验证 $40\,000$；标签为精确 $N_i^j$ 的独立条目 |
| 网络 | 4 个子网络并联，输入 125 维、输出 3456 维；每个子网络 15 个隐藏层，宽度 60–200，tanh 与 elu 交替；约 $1.28\times10^6$ 个参数 |
| 训练 | 损失为补全后 $\widehat N_i^j$ 与 $N_i^j$ 的逐元素 MSE；Adam，初始学习率 $10^{-3}$，batch 256，500 轮 |

## 3. 精度对照

### 3.1 精确子结构与全尺度 FEM：角点三线性插值的模型误差

| 指标 | 全尺度 FEM | 精确子结构 |
|---|---|---|
| 终止状态 / 轮数 | 收敛 / 65 | 收敛 / 68 |
| 最终 $C_{\mathrm{opt}}$ | $9.422055$ | $6.974166$ |
| 相对全尺度 FEM 的 $C_{\mathrm{opt}}$ 差 | — | $-26.0\%$ |
| 最终物理体积分数 | $0.1199987$ | $0.1199990$ |
| 物理密度不对称量 $x$ / $z$ | $6.2\times10^{-4}$ / $2.3\times10^{-7}$ | $3.1\times10^{-4}$ / $2.9\times10^{-7}$ |
| CG 步数，首轮 / 均值 / 最大 / 总计 | 2445 / 5827 / 8176 / $378\,756$ | 471 / 1050 / 1633 / $71\,421$ |
| 真残差，末轮 / 最大 | $1.0\times10^{-6}$ / $1.0\times10^{-6}$ | $9.9\times10^{-7}$ / $1.0\times10^{-6}$ |

1. **精确子结构的 $C_{\mathrm{opt}}$ 比全尺度 FEM 低 $26.0\%$**。对同一密度场，角点三线性插值把子结构边界位移限制在更小的子空间中，按 Ritz 原理模型偏硬，恒有 $C_{\mathrm{Exact}}\le C_{\mathrm{FEM}}$。两者收敛到不同的密度场，$C_{\mathrm{opt}}$ 之差反映模型偏硬，不表示设计更好。
2. **两者的设计都保持对称**。载荷与支承关于 $z$ 中面严格对称，两者的不对称量都在 $10^{-7}$ 量级；$x$ 方向两端支承不同，不要求对称，两者都在 $10^{-4}$ 量级。

| 统一细网格重分析 | 全尺度 FEM 设计 | 精确子结构设计 |
|---|---|---|
| $C_{\mathrm{Fine}}$ | $9.422055$（即自身 $C_{\mathrm{opt}}$） | 待填 |
| 相对全尺度 FEM 设计 | — | 待填 |
| 模型偏差 $(C_{\mathrm{opt}}-C_{\mathrm{Fine}})/C_{\mathrm{Fine}}$ | 0 | 待填 |

精确子结构设计的 $C_{\mathrm{Fine}}$ 相对全尺度 FEM 设计的差，即角点三线性插值带来的设计损失，两种子结构方法共有。

| 全尺度 FEM | 精确子结构 |
|---|---|
| ![全尺度 FEM 390 最终拓扑](../topopt_full_scale_fem/assets/ea_hex_390x65x65_final_topology.png) | ![精确子结构 390 最终拓扑](../topopt_exact_substructure/assets/linear_corner_hex_390x65x65_final_topology.png) |

图中只显示物理密度不小于 0.5 的单元，低于 0.5 的单元（含中间密度）不显示。

### 3.2 PIML 子结构与精确子结构：网络预测的额外误差

| 指标 | 精确子结构 | PIML 子结构 |
|---|---|---|
| 终止状态 / 轮数 | 收敛 / 68 | 收敛 / 83 |
| 最终 $C_{\mathrm{opt}}$ | $6.974166$ | $7.007897$ |
| 相对精确子结构的 $C_{\mathrm{opt}}$ 差 | — | $+0.48\%$ |
| 最终物理体积分数 | $0.1199990$ | $0.1199983$ |
| 物理密度不对称量 $x$ / $z$ | $3.1\times10^{-4}$ / $2.9\times10^{-7}$ | $0.417$ / $0.449$ |
| CG 步数，首轮 / 均值 / 最大 / 总计 | 471 / 1050 / 1633 / $71\,421$ | 936 / 2136 / 3538 / $177\,325$ |
| 真残差，末轮 / 最大 | $9.9\times10^{-7}$ / $1.0\times10^{-6}$ | $1.0\times10^{-6}$ / $1.0\times10^{-6}$ |

1. **PIML 子结构的 $C_{\mathrm{opt}}$ 比精确子结构高 $0.48\%$**。对同一密度场，网络预测的误差只会使角点刚度变硬，恒有 $\widehat C\le C_{\mathrm{Exact}}$。现在反而更高，说明 PIML 子结构收敛到的设计在其自身模型下也更差。
2. **PIML 子结构破坏了关于 $z$ 中面的对称性**。物理密度不对称量第 1 轮为 $3\times10^{-16}$，第 2 轮 $4.8\times10^{-3}$，第 10 轮 $0.14$，第 20 轮起稳定在 $0.4$ 左右，设计变量的不对称量自第 30 轮起达到 $1$。网络预测的形函数不满足镜像等变，带来的偏差在 OC 迭代中逐轮放大。这不是舍入：CPU 与 GPU 两次完整运行轮数相同，最终柔顺度相对差 $1.3\times10^{-8}$，阈值 0.5 的二值拓扑完全一致，不对称量也相同（4.1）。
3. **PIML 子结构末轮的预测诊断有离群子结构**：相对内部固支刚度的刚化比中位数 $-0.0118$、最大 $+0.094$（精确 Schur 补恒不大于 0），变形子空间条件数倒数中位数 $0.048$、最小 $3.4\times10^{-5}$。

| 统一细网格重分析 | 精确子结构设计 | PIML 子结构设计 |
|---|---|---|
| $C_{\mathrm{Fine}}$ | 待填 | 待填 |
| 相对精确子结构设计 | — | 待填 |
| 模型偏差 $(C_{\mathrm{opt}}-C_{\mathrm{Fine}})/C_{\mathrm{Fine}}$ | 待填 | 待填 |

PIML 子结构设计的 $C_{\mathrm{Fine}}$ 相对精确子结构设计的差，即网络预测额外带来的设计损失。PIML 子结构的 CPU 与 GPU 设计二值拓扑一致，只重分析一份。重分析入口与命令待定。

| 精确子结构 | PIML 子结构 |
|---|---|
| ![精确子结构 390 最终拓扑](../topopt_exact_substructure/assets/linear_corner_hex_390x65x65_final_topology.png) | 待渲染 |

图中只显示物理密度不小于 0.5 的单元。

## 4. 计算成本：GPU 相对于 CPU

CPU 运行为 NumPy 后端、单线程（PIML 子结构的网络推理为 PyTorch CPU，同为单线程），GPU 运行为 PyTorch / CUDA，主机侧同为单线程。单线程 CPU 是保守的参照，多线程 CPU 下的加速比会更小。

### 4.1 PIML 子结构

```bash
# CPU
python experiments/topopt_piml_substructure/run_piml_substructure.py --grid 390 65 65 --n-sub 78 13 13 --n-fine 5 --trace linear_corner --route shape --weights experiments/topopt_piml_substructure/outputs/independent_15_layer/training/shape/mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026_trainseed2026/shape_best.pt --assemble-level fa --preconditioner jacobi --max-iter 300 --backend numpy --device cpu --cpu-memory-limit 40 --threads 1
# GPU
python experiments/topopt_piml_substructure/run_piml_substructure.py --grid 390 65 65 --n-sub 78 13 13 --n-fine 5 --trace linear_corner --route shape --weights experiments/topopt_piml_substructure/outputs/independent_15_layer/training/shape/mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026_trainseed2026/shape_best.pt --assemble-level fa --preconditioner jacobi --max-iter 300 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
```

**结果一致性**

| 指标 | CPU | GPU | 差异 |
|---|---|---|---|
| 终止状态 / 轮数 | 收敛 / 83 | 收敛 / 83 | 相同 |
| 最终 $C_{\mathrm{opt}}$ | $7.007896699$ | $7.007896790$ | 相对差 $1.3\times10^{-8}$ |
| 最终物理密度 | — | — | 相对 L2 差 $2.3\times10^{-6}$，最大差 $1.4\times10^{-4}$ |
| 阈值 0.5 的二值拓扑 | — | — | 不一致比例 0 |
| 物理密度不对称量 $x$ / $z$ | $0.4173$ / $0.4492$ | $0.4173$ / $0.4492$ | 相同 |
| CG 步数，均值 / 最大 / 总计 | 2136 / 3536 / $177\,274$ | 2136 / 3538 / $177\,325$ | 几乎相同 |

**耗时与内存**（单位 s 与 GiB；每轮为全部轮次的均值，装配含边界条件）

| 指标 | CPU | GPU | CPU / GPU |
|---|---|---|---|
| 每轮装配（局部刚度、推理、变分式、诊断、散加） | 61.95 | 1.80 | 34 |
| 每轮求解（角点 CG） | 2.85 | 2.16 | 1.3 |
| 每轮恢复 | 0.12 | 0.03 | 4 |
| 整轮 | 68.54 | 5.87 | 11.7 |
| 准备阶段 | 25.0 | 37.3 | 0.67 |
| 优化循环 | 5690 | 488 | 11.7 |
| 进程 RSS 峰值 | 9.08 | 9.40 | — |
| 显存峰值 | — | 5.64 | — |

1. **CPU 下装配占整轮的九成**，其中第 1 轮分项为局部刚度生成 13.7、推理与补全 1.6、基函数拼接与变分式 23.5、预测诊断 28.6。诊断只记录不参与计算，却是最大的一项。
2. **GPU 下装配快 34 倍，瓶颈转到求解与输出**：角点系统只有 $46\,452$ 个未知量，GPU 上每步 CG 的固定开销使求解只快 1.3 倍；每轮写 VTU 约 1.8 s，在主机上进行，GPU 不加速。GPU 下装配分项的拆分受异步执行影响不可信，只看总量。
3. **准备阶段 GPU 反而更慢**：参考子结构与过滤矩阵在主机上构造后移到设备，为一次性开销。

数据来源：`topopt_piml_substructure/outputs/piml_shape_linear_corner_hex_390x65x65_m5_cg_jacobi_t1/` 与 `topopt_piml_substructure/outputs/pytorch_cuda_piml_shape_linear_corner_hex_390x65x65_m5_cg_jacobi_t1/`，日志 `topopt_piml_substructure/outputs/piml_shape_390.log` 与 `pytorch_cuda_piml_shape_390.log`。

### 4.2 精确子结构与全尺度 FEM

待定。390 档两者只有 GPU 完整运行（第 3 章），CPU 参照的取法未定：

- 精确子结构：可补一次 CPU 完整运行，旧 3 轮探针每轮约 51 s，外推约 1 h，同时可把第 3 章对精确子结构的设备无关性由推断改为实测。
- 全尺度 FEM：Jacobi-CG 单线程下每轮 CG 约 5800 步，完整运行不可行，宜以数轮探针比较每轮耗时，口径参照 [`fem_cpu_vs_gpu.md`](../topopt_full_scale_fem/fem_cpu_vs_gpu.md)。
- 线程数：是否另加多线程 CPU 参照待定。

## 5. 待办

1. 统一细网格重分析，填 3.1 与 3.2 的 $C_{\mathrm{Fine}}$。
2. 渲染 PIML 子结构最终拓扑，填 3.2。
3. 确定 4.2 的 CPU 参照并运行。
4. PIML 子结构的两次运行固化到 `topopt_piml_substructure/results/`，数据来源改为 `results/` 路径。
