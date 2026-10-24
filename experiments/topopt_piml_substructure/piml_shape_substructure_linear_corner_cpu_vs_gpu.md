# PIML shape 子结构 linear_corner 的 CPU 与 GPU 对照

硬件：AMD Ryzen 9 9950X（16 核 32 线程），运行内存 45 GiB（WSL2 上限 48 GB）；NVIDIA GeForce RTX 5070 Ti（70 SM，8960 CUDA 核心），显存 16 GiB。

软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13；NumPy 2.5.3（conda-forge，链接 BLIS 2.0 与 netlib LAPACK），SciPy 1.18.1（内置 scipy-openblas 0.3.31），PyTorch 2.11.0（CUDA 12.8）；PyMUMPS 0.4.0，链接 MUMPS 5.8.2（BLIS 2.0、netlib LAPACK、Open MPI 5.0.11、ScaLAPACK 2.2.0）；pyevtk 1.7.0。

## 1. 研究对象

对照 `PIML_shape` 在 `linear_corner` 下对 Huang2023 三维 MBB 梁的整条计算链分别在 `CPU` 与 `GPU` 上的计算代价与结果一致性。整条链分两段：离线阶段生成样本并训练网络，只做一次；在线阶段为每轮拓扑优化。

适用范围：PIML 只实现 linear_corner，full_trace 不做，见 [`substructure_linear_corner_piml_shape_vs_exact.md`](substructure_linear_corner_piml_shape_vs_exact.md) §4 第 5 条。trace 改变负载形态，Exact 两种 trace 的 CPU/GPU 结论相反：210 档 full_trace 的接口求解 GPU 快约 11 倍，linear_corner 的接口求解 GPU 反而慢约 5 倍，见 [`substructure_cpu_vs_gpu.md`](../topopt_simp_substructure/substructure_cpu_vs_gpu.md) §3.2 与 §4.2。本文结论不外推到 full_trace。

|      | CPU | GPU |
| ---- | --- | --- |
| 设备 | AMD Ryzen 9 9950X，16 线程 | NVIDIA GeForce RTX 5070 Ti，70 SM |
| 内存 | 主机内存 45 GiB | 显存 16 GiB |

### 1.1 离线阶段：样本生成与网络训练

| 步骤 | 说明 | CPU | GPU |
|---|---|---|---|
| 样本生成 | 每个样本对随机模量场做一次子结构内部精确消元，得到形函数标签 $N_{i,F}^j$；参考子结构与独立条目编号在 CPU 上构建，局部装配、批量消元、标签编码与补全检查在所选设备执行，标签转回 CPU 写盘 | `--backend numpy --device cpu`（已有正式运行） | `--backend pytorch --device cuda` |
| 网络训练 | 4 个 15 层全连接网络，每轮遍历 400 000 个训练样本与 40 000 个验证样本；样本以 memmap 存放在主机，每个 batch 按随机下标取出后拷到所选设备 | `--device cpu`（已有正式训练） | `--device cuda` |

两步当前均固定为 float64（`--dtype float64`，当前唯一取值）。RTX 5070 Ti 的 FP64 峰值约 0.69 TFLOPS，只有 FP32 的 1/64，因此 float64 下 GPU 的优势预期远小于常见的 FP32 经验；训练放行 float32 后另作对照。

对照内容：

1. **样本生成**：GPU 相对 CPU 是否加速，比较现状流程与全 GPU 流程。CPU 一侧为正式数据集的生成方式 NumPy CPU（已有，不重跑），GPU 一侧为 PyTorch GPU；GPU 只能用 PyTorch 后端，故倍数同时包含后端与设备的差别。在正式规模下比较含写盘的总耗时；生成批量固定为 32（正式值），不扫描批量。一致性只作最低限度核对：输入逐元素相同，标签只在舍入量级相同。
2. **网络训练**：每侧的训练数据与训练在同一设备上：CPU 一侧为已有正式训练（NumPy CPU 样本、PyTorch CPU 训练，不重跑），GPU 一侧用 PyTorch GPU 生成的样本在 GPU 上训练；比较正式训练（500 轮）的总耗时。训练只有 PyTorch 后端，两侧训练只差设备；但两份样本的独立条目编号不同，训练轨迹不可逐轮对照，只比较终态。

预期（未实测）：

- 样本生成的主项是批量小矩阵消元（内部 192 个自由度），与 Exact 在线缩聚的批量内部求解同型；GPU 上批量 float64 小矩阵分解受 FP64 吞吐与逐批固定开销限制，加速比不确定。
- 网络训练每轮约 $6\times1.28\times10^6\times4\times10^5\approx3\times10^{12}$ 次浮点运算，按 GPU FP64 峰值折算下限约 5 s；网络窄（宽度不超过 200）、batch 小（256），实际效率更低。另有两项开销不随设备加速：每个 batch 在主机上按随机下标从 memmap 取数（单线程），以及每个 batch 一次 `isfinite` 检查引起的设备同步。粗估加速 2–5 倍。

### 1.2 在线阶段：单轮拓扑优化

| 步骤 | 说明 | 现状设备 |
|---|---|---|
| 子结构局部组装 $K^j$ | 按批的稠密运算，每批 256 个子结构 | 随 `--device` |
| 网络推理与刚体补全 | 输入为各细单元归一化模量 $\widehat E_e=E(\tilde\rho_e)/E_0$，4 个 15 层网络输出 3456 个独立条目，经刚体约束补全为 $\widehat N_i^j\in\mathbb R^{192\times24}$ | 固定 CPU |
| 变分式角点刚度 | $\widehat K_C^j=(\widehat N^j)^{\mathsf T}K^j\widehat N^j$，再对称化 | 固定 CPU（`numpy`） |
| 诊断量 | 刚体零空间残量、相对 $L^{\mathsf T}K_{bb}^jL$ 的刚化幅度、变形子空间条件数；每个子结构三次 $24\times24$ 对称特征值分解，只记录不触发替换 | 固定 CPU（`numpy`） |
| 均匀子结构替换 | 子结构内模量 $\max-\text{mean}$ 小于阈值时改用实体形函数；默认关闭 | 固定 CPU |
| 组装全局角点总刚 $\widehat K_C$ | 符号模式只在首轮于主机上构建一次，之后移到所选设备，每轮只重填数值 | 随 `--device` |
| 求解 $\widehat K_C\widehat U_C=F_C$ | 约束只在首轮于主机上解析一次，CG 迭代与真实残差验收都在所选设备上 | 随 `--device` |
| 内部位移回代 $\widehat u_i^j=\widehat N_i^j\widehat q^j$ | 直接用组装时缓存的 $\widehat N_i^j$ 回代，不重做组装 | 随 `--device` |

### 1.3 在线前置改造

现有实现把 PIML 缩聚固定在 CPU，在线阶段的 GPU 一侧在以下改造完成前无法运行；改造属于库层任务，不在本文实现，完成后把提交号记入 §2.2。离线阶段不受影响。

| 位置 | 现状 | 目标 |
|---|---|---|
| `PIMLShapeCornerCondensation.__init__`（`src/soptx/fem/substructure/piml_corner.py`） | 要求网络全部参数位于 CPU，否则抛错 | 接受与 `--device` 一致的设备，网络在构造时移到该设备 |
| `PIMLShapeCornerCondensation.__call__` | 每批局部刚度先 `bm.to_numpy(local)` 拷回主机；基函数拼接、四项式、诊断量全部用 `numpy` 计算；结果再 `bm.asarray` 交给装配 | 全程留在所选设备：推理输入由设备上的模量直接构造，四项式与诊断量改为 backend 无关实现或 `torch` 批量运算 |
| `_diagnostics` | `np.linalg.eigvalsh` | 设备上的批量对称特征值分解；GPU 上批量小矩阵分解的性能须实测，见 §1.5 |
| `_predict` | `predict_independent_outputs` 在 CPU 推理 | 在所选设备推理，输出保持 `float64` |
| 入口 `run_piml_substructure.py` | `--backend`、`--device`、`--gpu-memory-limit` 已开出，但只接受 `numpy`、`cpu`，推理设备写死 `'cpu'` | 放开 `--backend pytorch --device cuda` 并接入 `limit_cuda_memory`；输出目录名的 `<backend>_<device>_` 前缀已按 Exact 入口实现。Exact 侧入口 `run_substructure_full_trace_vs_linear_corner.py` 已有 `--backend`、`--device` |

### 1.4 在线一致性预期

与 Exact 两侧只有舍入差异不同，PIML 两侧在三处可能出现超出舍入量级的差别，须在验收中单独检查：

1. **推理不逐位相同**：GPU 与 CPU 的 GEMM 内核与归约顺序不同，$\widehat N_i^j$ 两侧只在舍入量级相同，不逐位相同；其后四项式、$\widehat K_C$、位移与柔顺度的差别是推理差别的传播，与 Exact 两侧的差别性质相同，但量级可能更大。
2. **诊断量**：三项诊断量只记录不触发替换，两侧的 $\min/\text{median}/\max$ 应在舍入量级一致。若日后加入按阈值回退的门禁，须补记两侧判定翻转的子结构数及其比值与阈值的距离。
3. **均匀替换**：判别用模量差，两侧模量由相同密度经相同插值得到，判定应完全一致；默认关闭时不适用。

### 1.5 在线预期收益

PIML 用推理、四项式与诊断量代替 Exact 的批量内部求解，其余步骤与 Exact linear_corner FA 相同，这部分的设备对比已由 Exact CPU/GPU 文档 §4 给出。在线阶段的新增信息只在被替代的三步。

已有的分项依据：$60\times10\times10$（48 个子结构）单线程、CPU 的两轮小跑（`outputs/piml_shape_hex_60x10x10_m5_cg_jacobi_t1/`，第 2 轮）中，缩聚三项合计 0.178 s，其中 `diagnostics` 0.091 s、`basis_assembly` 0.080 s、`inference` 0.006 s；`local_stiffness` 0.016 s。即诊断量（三次 $24\times24$ 批量特征值分解）与四项式各占约一半，推理几乎可忽略。

据此的预期（未实测）：

- 四项式为批量稠密 GEMM（$256\times648\times648\times24$ 量级），GPU 应有明显加速。
- 诊断量为批量小矩阵对称特征值分解，GPU 上的批量 `eigvalsh` 通常是逐矩阵的固定开销而非算力瓶颈，可能不比 16 线程 CPU 快；若实测如此，可选择把诊断量留在主机或改为按需计算，并在 §3.2 记录所选方案。
- 推理规模很小（210 档 2058 个子结构、4 个网络），两侧都在毫秒到数十毫秒量级。
- 接口求解与 Exact linear_corner 同为 $8256$ 个未知量，GPU 侧每步 CG 为固定开销，预期仍慢于 CPU；输出 VTU 在主机上进行，仍限制 GPU 侧每轮加速比。

## 2. 实验设置

### 2.1 离线阶段

样本、网络与训练参数同 [`substructure_linear_corner_piml_shape_vs_exact.md`](substructure_linear_corner_piml_shape_vs_exact.md) §2.4，下表只列本文的设置与两侧差别。

**样本生成**

| 参数 | 数值或设置 |
|---|---|
| 局部配置 | 同 PIML 对照文档 §2.4：`--n-sub 78 13 13 --n-fine 5 --fine-cell-size 1.0 --nu 0.3 --trace-kind linear_corner` |
| 采样 | 同 PIML 对照文档 §2.4：`--sampling independent_uniform --min-modulus 1e-7 --sample-seed 2026 --targets shape` |
| 规模 | 正式规模，训练 $400\,000$、验证 $40\,000$：`--n-train 400000 --n-validation 40000` |
| 规模与资源 | 按存储格式估算，与设备无关。每个样本含输入 125 个与标签 3456 个 float64，共 $28\,648$ 字节：磁盘约 $28.6\ \mathrm{KB}\times N$，正式规模 12.6 GB（11.7 GiB），本机剩余约 880 GB，可容约 3 000 万个样本。生成时程序的工作内存只随生成批量变化，与样本数 $N$ 无关（标签经 memmap 逐批写盘），显存同理。内存约束出现在训练：每轮随机读取全部样本，数据集宜整体留在页缓存中，本机 WSL 可用约 45 GiB，留出系统与进程开销后约 160 万个样本；超过时训练每轮从磁盘随机读取，预计明显变慢（未实测） |
| 数据精度 | float64，两侧相同：`--dtype float64` |
| 后端与设备 | CPU：NumPy `--backend numpy --device cpu`，即正式数据集的生成方式，取已有正式运行及其日志 `outputs/generate_samples_shape_seed2026.log`，不重跑；GPU：PyTorch `--backend pytorch --device cuda` |
| 生成批量 | 两侧均为 32，与正式运行相同：`--generation-batch-size 32`。不扫描批量；32 是为 CPU 定的值，GPU 在更大批量下可能更快，故本节结论不代表 GPU 的最优吞吐 |
| 数据集名 | GPU：`full_pytorch_cuda_b32_shape_train400000_val40000_seed2026`：`--dataset-name`；运行完成后移到 `outputs/probes/`，避免样本根目录中出现第二份正式规模数据集 |
| 总耗时 | GPU 取 `/usr/bin/time -v` 的 Elapsed；CPU 取正式运行日志的起止时间戳；均含参考结构构建、两个划分及写盘 |
| 运行条件 | 机器空闲，不与其他 CPU 或 GPU 任务并行；开跑前记录系统负载与 GPU 利用率 |

**网络训练**

| 参数 | 数值或设置 |
|---|---|
| 样本 | 与训练设备一致：CPU 为 PIML 对照文档 §2.4 的正式数据集（NumPy CPU 生成）；GPU 为上文 GPU 生成的 `outputs/probes/full_pytorch_cuda_b32_shape_train400000_val40000_seed2026/`：`--samples-dir` |
| 网络与优化器 | 同 PIML 对照文档 §2.4，两侧相同：`--epochs 500 --patience 40` |
| 训练种子 | 2026，两侧相同：`--train-seed 2026` |
| 训练精度 | float64，两侧相同：`--dtype float64` |
| 设备 | CPU：`--device cpu`，PyTorch 默认线程数，取已有正式训练 `outputs/independent_15_layer/training/shape/mbb_linear_corner_m5_h1_sub78x13x13_nu0p3_emin1e-7_shape_train400000_val40000_seed2026_trainseed2026/` 及日志 `outputs/train_shape_seed2026.log`，不重跑；GPU：`--device cuda` |
| 输出目录 | GPU：`outputs/probes/full_pytorch_cuda_train_shape_seed2026/`：`--output-dir` |
| 总耗时 | GPU 取 `/usr/bin/time -v` 的 Elapsed；CPU 取训练日志的起止时间戳；均含每轮验证与 checkpoint 保存 |
| 运行条件 | 机器空闲，不与其他 CPU 或 GPU 任务并行；开跑前记录系统负载与 GPU 利用率；须在 GPU 样本生成完成后运行 |

### 2.2 在线阶段

实验脚本为 [`run_piml_substructure.py`](run_piml_substructure.py)（`--device cuda` 待实现），在仓库根目录运行；除必填的 `--trace linear_corner --route shape --weights`（权重见 PIML 对照文档 §2.4）及 `--device`、`--max-iter` 与 `--threads 16` 外，其余参数取默认值：`--grid 210 35 35`、`--n-fine 5`。模型、材料、离散、线性求解、优化参数及 PIML 模型同 PIML 对照文档 §2，以入口脚本顶部配置为准。下表只列本文特有或与 Exact CPU/GPU 文档对齐的设置。

| 参数 | 数值或设置 |
|---|---|
| 子结构边界位移 | `trace='linear_corner'`，两侧相同 |
| 局部刚度层级 | FA：组装完整 $K^j$ 后切片；EA 待 [`piml_shape_substructure_linear_corner_fa_vs_ea.md`](piml_shape_substructure_linear_corner_fa_vs_ea.md) 完成后再定 |
| 网络权重 | 两侧同一权重文件，待填，取 `config.json` 的 `checkpoint`（路径、sha256） |
| 推理精度 | `float64`，两侧相同；代码固定，不开参数 |
| 推理批量 | 与缩聚分批相同，`chunk_size=256`，两侧相同 |
| 均匀子结构判别 | `UNIFORM_THRESHOLD = None`，关闭，两侧相同 |
| 计算后端 | 两侧均为 PyTorch：`BACKEND = 'pytorch'`（待改，现为 `numpy`） |
| 线程数 | 两侧均为 16，与 Exact CPU/GPU 文档一致：`--threads 16`；GPU 侧的线程数只作用于主机端运算 |
| 库层改造提交 | 待填 |
| 输出目录 | `outputs/cpu_piml_shape_hex_210x35x35_m5_cg_jacobi_t16/`、`outputs/cuda_piml_shape_hex_210x35x35_m5_cg_jacobi_t16/`（命名待入口改造后确认） |

## 3. 结果对照

### 3.1 离线阶段

**样本生成**

```bash
# GPU; CPU 取已有正式运行, 不重跑
OUT=experiments/topopt_piml_substructure/outputs
name=full_pytorch_cuda_b32_shape_train400000_val40000_seed2026
mkdir -p $OUT/probes
/usr/bin/time -v python experiments/topopt_piml_substructure/generate_samples.py \
  --n-sub 78 13 13 --n-fine 5 --fine-cell-size 1.0 --nu 0.3 --trace-kind linear_corner \
  --sampling independent_uniform --n-train 400000 --n-validation 40000 \
  --min-modulus 1e-7 --sample-seed 2026 --targets shape \
  --backend pytorch --device cuda --dtype float64 --generation-batch-size 32 \
  --dataset-name $name 2>&1 | tee $OUT/probes/$name.log
mv $OUT/independent_15_layer/samples/$name $OUT/probes/
```

| 指标 | NumPy CPU（已有正式运行） | PyTorch GPU |
| --- | --- | --- |
| 总耗时 / s | 1573（26 min 13 s） | 94.5 |
| 平均吞吐 / 样本每秒 | 280 | 4655 |

速度：GPU 快约 16.6 倍。该倍数同时包含后端与设备的差别；3.5 万样本的探针中只换设备（PyTorch CPU 233 样本/s 对 PyTorch GPU 4962 样本/s）约 21 倍，NumPy CPU 只用约 1 个核（CPU 利用率 101%）。两侧都远未达到 FP64 峰值（按每样本约 $5\sim10$ MFLOP 估计，CPU 约 2 GFLOP/s、GPU 约 30 GFLOP/s），瓶颈是逐批调用开销而非算力；GPU 的批量内核摊薄了这部分开销，而 NumPy 路径只用单核，故该倍数反映的是现有 CPU 实现与 GPU 的差距，不是硬件上限。一致性：两者输入逐元素相同（待填）；NumPy 与 PyTorch 的独立条目编号不同，各自补全为物理矩阵 $N_i^j$ 后逐样本相对差最大为待填，满足 README 验收 $\lVert A_{\mathrm{GPU}}-A_{\mathrm{CPU}}\rVert_F\leq10^{-8}\lVert A_{\mathrm{CPU}}\rVert_F+10^{-10}$（待填），样本只差浮点舍入，GPU 生成的样本可替代正式数据集。

数据来源：CPU 为正式数据集及日志 `outputs/generate_samples_shape_seed2026.log`；GPU 为 `outputs/probes/full_pytorch_cuda_b32_shape_train400000_val40000_seed2026/` 及同名日志，2026-10-23 运行，开跑前连续 3 次确认机器空闲；探针为 `outputs/probes/probe_*_b32_shape_train32000_val3200_seed2026/` 及同名日志。

**网络训练**

```bash
# GPU, 用 GPU 生成的样本; CPU 取已有正式训练, 不重跑
OUT=experiments/topopt_piml_substructure/outputs
name=full_pytorch_cuda_train_shape_seed2026
/usr/bin/time -v python experiments/topopt_piml_substructure/train_network.py \
  --samples-dir $OUT/probes/full_pytorch_cuda_b32_shape_train400000_val40000_seed2026 \
  --route shape --num-networks 4 \
  --hidden-dims 60 80 100 120 140 160 180 200 180 160 140 120 100 80 60 \
  --activations tanh elu tanh elu tanh elu tanh elu elu tanh elu tanh elu tanh elu \
  --input-normalization none --loss decoded_full_matrix_mse --stiffness-loss-weight 0 \
  --epochs 500 --batch-size 256 --optimizer adam --lr 1e-3 --weight-decay 0 --patience 40 \
  --train-seed 2026 --backend pytorch --device cuda --dtype float64 \
  --output-dir $OUT/probes/$name 2>&1 | tee $OUT/probes/$name.log
```

| 指标 | CPU（已有正式训练） | GPU |
| --- | --- | --- |
| 总耗时 / h | 8.57（30 866 s） | 5.44（19 579 s） |
| 平均每轮耗时 / s | 61.7 | 39.2 |

速度：GPU 快约 1.58 倍，低于 §1.1 的预期；两侧训练只差设备，训练计算量与样本来源无关。每轮约 $3\times10^{12}$ FLOP，CPU 与 GPU 实际分别约 50 与 80 GFLOP/s，GPU 只达 FP64 峰值的约 11%，按半峰值估计每轮应约 9 s，故时间不在计算上。训练期间 `time -v` 的 CPU 利用率为 99%，即一个主机线程满载：每个 batch 在主机上单线程从 memmap 按随机下标取数，网络窄（宽度不超过 200）而 kernel 启动次数多，单个 GEMM 的计算时间与启动开销同量级，每个 batch 的 `isfinite` 检查还引起设备同步，GPU 主要在等主机（推断，未剖析）。改用 float32、数据整块常驻内存或显存、减少同步与 kernel 启动可望改善，均需改动训练库。一致性：两侧样本编号不同，网络学习的输出排列不同，训练轨迹不可逐轮对照；损失在补全后的物理矩阵上计算，终态数值可比：CPU 第 1 轮验证损失 $1.616105\times10^{-4}$、最佳轮 491、最佳验证损失 $6.392565\times10^{-6}$，GPU 分别为 $1.598735\times10^{-4}$、499、$6.402026\times10^{-6}$。两侧最佳验证损失相差约 0.15%，均跑满 500 轮、未触发提前停止，终态质量相当。

数据来源：CPU 见 §2.1 所列训练目录与日志；GPU 为 `outputs/probes/full_pytorch_cuda_train_shape_seed2026/` 及同名日志，2026-10-23 23:56 至 10-24 05:22 运行，开跑前连续 3 次确认机器空闲。
