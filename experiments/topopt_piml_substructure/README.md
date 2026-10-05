# PIML 子结构拓扑优化

## MBB 梁算例

采用 Huang2023 [1] 的三维 MBB 梁算例，在体积分数约束下最小化结构柔顺度。所有参数均为无量纲量。

### 模型参数

![MBB 梁的几何、载荷与支承](assets/Huang2023_Fig5.png)

梁的长、高、宽为 $6\times1\times1$，使用整体模型，不利用对称性缩减计算域。顶面中心施加竖直向下的单位集中力 $F=1$。两端支承在侧视图中为一端铰支、另一端滚支。

| 材料参数 | 数值 |
|---|---|
| 实体材料杨氏模量 $E_0$ | $1$ |
| 弱材料杨氏模量 $E_{\min}$ | $10^{-7}$ |
| 泊松比 $\nu$ | $0.3$ |

### 有限元离散

| 参数 | 数值或设置 |
|---|---|
| 有限元 | 八节点六面体单元（Q1） |
| 位移插值次数 $p$ | $1$ |
| 数值积分 | 二阶高斯积分 |

### 子结构参数

| 项目 | 数值或设置 |
|---|---|
| 接口类型 | `full_trace`、`linear_corner` |
| 子结构细分参数 $m$ | $5$，即每个子结构包含 $5\times5\times5$ 个细单元 |
| 子结构网格规模 | $78\times13\times13$ |
| 全局细网格规模 | $390\times65\times65$ |
| 细单元边长 | $h=1/65$，由几何尺寸与网格划分得到 |

两种接口以迹矩阵 $\boldsymbol\Psi$ 表示边界位移：

| 接口类型 | 迹矩阵 $\boldsymbol\Psi$ | 局部接口未知量 $\boldsymbol q_j$ |
|---|---|---|
| `full_trace` | $\boldsymbol I$ | 完整边界位移 |
| `linear_corner` | $\boldsymbol L$ | 角点位移 |

其中，$\boldsymbol L$ 为由角点位移插值得到边界位移的矩阵，两种接口均满足 $\boldsymbol u_{jb}^h=\boldsymbol\Psi\boldsymbol q_j$。

### 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度 |
| 体积分数 | $0.12$ |
| 过滤半径 | $3$ 个细单元尺寸；对应实际长度 $r_{\min}=3h=3/65$ |
| 设计变量更新 | OC |
| 停止条件 | 连续最后 5 次迭代的目标函数相对变化小于 $0.0002$ |

### 补定参数（文献未给出）

| 参数 | 补定值 |
|---|---|
| 支承（待确认的建模选择） | 左端底边 $x=0,\ y=0$ 整条线：$u_x=u_y=u_z=0$；右端底边 $x=6,\ y=0$ 整条线：$u_y=0$ |
| 集中力分配 | 顶面 $x=3$ 处，$z$ 向距中心最近的 2 个节点各施加 $-1/2$ |
| 初始密度 | 均匀取 $0.12$ |
| 材料插值 | 修正 SIMP：$E_e=E_{\min}+\tilde\rho_e^{\,q}(E_0-E_{\min})$；设计密度 $\rho_e\in[0,1]$，物理密度为过滤后的 $\tilde\rho_e$ |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$） |
| 过滤 | 密度过滤：$\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离；物理密度取 $\tilde\rho$，灵敏度按链式法则过滤 |
| 体积约束 | $\sum_e v_e\tilde\rho_e/\sum_e v_e\leq0.12$，$v_e$ 为细单元体积；OC 体积判断使用物理密度，体积灵敏度对设计密度按链式法则计算 |
| OC 更新 | 移动限 $0.2$，阻尼指数 $0.5$；二分法求 Lagrange 乘子，初始区间 $[0,10^9]$，相对精度 $10^{-3}$ |
| 停止条件公式 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止 |
| 最大迭代数 | $300$ |

## PIML 子结构分析

### 方法选择

两条预测路线的局部计算方式为：

| 预测路线 | 预测对象 | 局部接口刚度 |
|---|---|---|
| `shape` | 内部位移映射 $\boldsymbol T_j$ 的独立条目，约束补全后得到 $\widehat{\boldsymbol T}_j$ | $\widehat{\boldsymbol K}_{r,j}=(\widehat{\boldsymbol N}_j^{(\Psi)})^{\mathrm T}\boldsymbol K_j^h\widehat{\boldsymbol N}_j^{(\Psi)}$ |
| `stiffness` | 接口刚度的独立条目 | 约束补全得到 $\widehat{\boldsymbol K}_{r,j}$ |

其中 $\widehat{\boldsymbol N}_j^{(\Psi)}=\begin{bmatrix}\boldsymbol\Psi\\\widehat{\boldsymbol T}_j\end{bmatrix}$。`stiffness` 路线仍需相同接口配置的形函数权重，用于细网格位移恢复及灵敏度计算。

### 网络输入

| 参数 | 数值或设置 |
|---|---|
| 网络输入 | 各细单元归一化杨氏模量 $\widehat E_e=E(\tilde\rho_e)/E_0$ |
| 输入维度 | $m^3$（三维） |
| 输入排序 | 与训练数据的细单元顺序一致 |

样本与权重须与上述模型、有限元离散及所选接口类型一致；不同接口分别生成样本并训练权重。材料采用三维各向同性线弹性。

### 均匀子结构判别

| 参数 | 数值或设置 |
|---|---|
| 均匀子结构判别阈值 $\rho_v$ | $10^{-4}$ |
| 判别对象 | 物理密度 $\tilde\rho_e$（实现选择） |
| 子结构平均密度 | $\rho_m^j=\frac{1}{125}\sum_{e\in\Omega_h^j}\tilde\rho_e$ |
| 判别条件 | $\lvert\max_{e\in\Omega_h^j}\tilde\rho_e-\rho_m^j\rvert<\rho_v$，$\rho_v=10^{-4}$ |
| 满足判据时的形函数 | 采用相同配置的实体材料子结构形函数 |
| 满足判据时的缩聚刚度 | $\boldsymbol K_{s,j}=\frac{E(\rho_m^j)}{E_0}\boldsymbol K_{s,j}^{\mathrm{solid}}$（修正 SIMP 下的实现选择，阈值内按均匀材料近似） |
| 未满足判据时 | 按所选路线预测形函数后构造刚度，或直接预测接口刚度 |

## 样本生成与网络训练

### 样本生成参数

| 参数 | 数值或设置 |
|---|---|
| 训练样本数 | $400,000$（论文形函数训练配置） |
| 采样量 | 各细单元归一化杨氏模量 $\widehat E_e$ |
| 随机生成范围 | $[0,1]$ |
| 精确标签 | 局部静力缩聚得到内部形函数与接口刚度 |
| 空间维数 | 三维 |
| 接口类型 | 按“子结构参数”选择 |
| 子结构细分参数 | $m$，按“子结构参数”配置 |
| 子结构尺寸 | 由 MBB 几何与子结构网格推得 |
| 泊松比与有限元配置 | 与“模型参数”和“有限元离散”一致 |

### 样本生成补定参数（文献未给出）

| 参数 | 补定值 |
|---|---|
| 验证样本数 | $40,000$ |
| 实际采样范围与分布 | 各细单元独立均匀采样，$\widehat E_e\in[10^{-7},1)$ |
| 样本随机种子 | $2026$；训练集与验证集使用独立随机子流 |
| 数据精度 | `float64` |

### 网络结构

以下输出维度由三维 $m=5$ 的离散配置与六个刚体运动约束推得；网络配置按组合指定。

| 接口空间 | 预测路线 | 独立输出数 | 补全矩阵大小 | 目标网络配置 | 配置来源 |
|---|---|---:|---|---|---|
| `linear_corner` | `shape` | $3456$ | $192\times24$ | $4$ 个网络，每个 $15$ 个隐藏层 | 论文 MBB 路线 |
| `linear_corner` | `stiffness` | $171$ | $24\times24$ | $1$ 个网络，$11$ 个隐藏层 | 论文第 3.3 节刚度网络；数量采用单网络方案 |
| `full_trace` | `shape` | $86400$ | $192\times456$ | 网络数量与结构待确定 | 三维实验扩展 |
| `full_trace` | `stiffness` | $101475$ | $456\times456$ | 网络数量与结构待确定 | 三维实验扩展 |

| 网络配置 | 各隐藏层宽度 | 各隐藏层激活函数 | 来源 |
|---|---|---|---|
| 15 层形函数网络 | [60, 80, 100, 120, 140, 160, 180, 200, 180, 160, 140, 120, 100, 80, 60] | [tanh, elu, tanh, elu, tanh, elu, tanh, elu, elu, tanh, elu, tanh, elu, tanh, elu] | 论文 |
| 11 层角点刚度网络 | [50, 60, 70, 80, 90, 100, 90, 80, 70, 60, 50] | [tanh, elu, tanh, elu, tanh, elu, elu, tanh, elu, tanh, elu] | 论文 |

当前走查入口两条路线均默认 $4$ 个网络、$15$ 个隐藏层；上表的单网络 11 层刚度配置需单独配置，尚未接入该命令行入口。三维 `full_trace` 配置不直接沿用角点网络权重。

### 训练参数

| 参数 | 数值或设置 | 来源 |
|---|---|---|
| 输出分组 | 独立条目连续均分；默认角点形函数路线每个网络 $864$ 个输出，角点刚度单网络 $171$ 个输出 | 实验补定，采用现有分组实现 |
| 基本损失 | 按路线计算约束补全后内部形函数或接口刚度矩阵的 MSE | 论文采用补全后 MSE；此处按现有实现计算 |
| 后期一致性损失 | 当前未启用；论文在形函数训练最后阶段加入形函数构造刚度与预测刚度之间的 MSE | 与论文的差异，尚待接入 |
| 优化器 | Adam，初始学习率 $10^{-3}$，weight decay $0$ | 实验补定，参考现有脚本 |
| 训练 batch size | $256$ | 实验补定，参考现有脚本 |
| 最大训练轮数 | $500$ | 实验补定，参考现有脚本 |
| 学习率调整 | 验证损失连续 $10$ 轮未改善时减半，下限 $10^{-6}$ | 实验补定，参考现有脚本 |
| 提前停止 | 验证损失连续 $40$ 轮未改善 | 实验补定，参考现有脚本 |
| 选模规则 | 保存验证 MSE 最低的权重 | 实验补定，参考现有脚本 |
| 训练随机种子 | $2026$ | 实验补定，参考现有脚本 |
| 训练设备与精度 | CPU、`float64` | 实验补定，参考现有脚本 |
| 权重保存目录 | `~/codespace/data/soptx/piml_substructure/independent_15_layer/training/<route>/<UTC 时间戳>/`，文件名为 `<route>_best.pt`；目录名称不替代权重内的结构记录 | 现有脚本约定；11 层配置接入后须核对保存入口 |

## 执行入口

`generate_samples.py` 为三维 MBB 局部样本生成入口，同时生成形函数与接口刚度的独立条目标签，不启动网络训练。默认配置与上面的样本参数一致；本目录尚无 PIML 拓扑优化执行入口。

网络训练与子结构分析走查使用 [PIML 子结构分析](../analysis_capability_piml_substructure/README.md) 中的入口；这些入口尚不能替代完整 MBB 拓扑优化流程。上文配置为复现目标，尚未完成本算例的端到端数值验证。

在仓库根目录的 WSL 环境执行:

```bash
~/miniconda3/envs/ihpcm/bin/python experiments/topopt_piml_substructure/generate_samples.py
```

使用 `--n-sub NX NY NZ` 修改子结构网格，子结构尺寸自动按梁尺寸 $6\times1\times1$ 推导；`--n-fine` 设置 $m$，`--trace-kind` 选择接口类型，`--nu` 设置泊松比。`--n-train`、`--n-validation`、`--min-modulus` 和 `--sample-seed` 设置数据规模与采样；`--generation-batch-size` 与 `--outputs-root` 设置生成批量和产物根目录。空间维数固定为三维，有限元固定为 Q1、二阶高斯积分，数据精度固定为 `float64`。

每次生成写入独立 UTC 时间戳目录，保存 `manifest.json`、六个 NPY 文件及 `generation_config.json`。后者记录 MBB 网格、派生尺寸、积分阶数与生成批量等运行配置。只有全部样本写完后，`manifest.json` 的 `complete` 才为 `true`；已有同配置完整数据集时拒绝重复生成，现有目录不会被覆盖。

验收时核对尺寸、采样下界、随机种子、训练与验证数量、两类标签维度及完成标记。当前入口仅经静态检查，尚未执行样本生成或数值验证。

### 样本生成运行设置

| 项目 | 设置 |
|---|---|
| 计算后端 | `numpy`（默认）、`pytorch` |
| 计算设备 | `cpu`（默认）、`cuda` 或 `cuda:N` |
| 样本生成批量 | $32$ |
| 样本保存目录 | `~/codespace/data/soptx/piml_substructure/independent_15_layer/samples/<UTC 时间戳>/` |

生成批量控制内存与计算效率，保存目录指定输出位置。`--backend` 可选 `numpy` 或 `pytorch`，默认 `numpy`；`--device` 可选 `cpu`、`cuda` 或 `cuda:N`，默认 `cpu`。NumPy 仅支持 CPU，GPU 计算需选择 PyTorch；CUDA 不可用或设备编号越界时直接报错。

GPU 生成命令:

```bash
~/miniconda3/envs/ihpcm/bin/python experiments/topopt_piml_substructure/generate_samples.py \
  --backend pytorch --device cuda:0 \
  --generation-batch-size 32
```

参考子结构及独立条目编号在 CPU 上构建，局部装配、批量内部消元、接口刚度计算、标签编码与补全检查在所选设备执行，最终标签转回 CPU 写盘。采样继续使用 NumPy 的独立随机子流，CPU/GPU 路线的同种子输入一致；标签可能存在浮点舍入差异。数据均为 `float64`，`generation_config.json` 记录实际后端与设备。后端和设备属于运行设置，不改变样本的物理配置，也不作为重复数据集判定条件。

尚未执行 CPU/GPU 数值对照或性能测量。建议先在不同输出根目录生成小规模同种子数据:

```bash
~/miniconda3/envs/ihpcm/bin/python experiments/topopt_piml_substructure/generate_samples.py \
  --backend numpy --device cpu --n-train 64 --n-validation 16 \
  --outputs-root /tmp/soptx_piml_samples_cpu

~/miniconda3/envs/ihpcm/bin/python experiments/topopt_piml_substructure/generate_samples.py \
  --backend pytorch --device cuda:0 --n-train 64 --n-validation 16 \
  --outputs-root /tmp/soptx_piml_samples_gpu
```

验收条件为两套输入逐元素一致、两类标签有限且一致到双精度容差、完整记录与完成标记正确。标签对照建议按每个样本的 Frobenius 范数检验 $\lVert A_{GPU}-A_{CPU}\rVert_F\leq10^{-8}\lVert A_{CPU}\rVert_F+10^{-10}$，并使用同一参考 codec 补全矩阵后核对。之后再比较包含传输与写盘的总耗时，确定生成批量及是否采用 GPU。

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7；网络输入、训练与均匀子结构处理见第 3.1、3.3、3.4 节。
